import time
from pathlib import Path

import pytest
from ebooklib import epub

from diareader import config
from diareader.text_extract import (
    Chapter, epub_to_chapters, load_book, load_book_cached, save_book_cache, txt_to_text,
)


def test_txt_to_text_normalizes_whitespace(tmp_path: Path):
    path = tmp_path / "book.txt"
    path.write_text("Xin   chào.\r\n\r\n\r\nTạm biệt.\n", encoding="utf-8")
    text = txt_to_text(path)
    assert "Xin chào." in text
    assert "\n\n\n" not in text


def _build_sample_epub(path: Path) -> None:
    book = epub.EpubBook()
    book.set_identifier("id123")
    book.set_title("Sách mẫu")
    book.set_language("vi")

    c1 = epub.EpubHtml(title="Chương 1", file_name="chap1.xhtml", lang="vi")
    c1.content = "<h1>Chương 1</h1><p>Đây là nội dung chương một.</p>"
    c2 = epub.EpubHtml(title="Chương 2", file_name="chap2.xhtml", lang="vi")
    c2.content = "<h1>Chương 2</h1><p>Đây là nội dung chương hai.</p>"

    book.add_item(c1)
    book.add_item(c2)
    book.toc = (c1, c2)
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())
    book.spine = ["nav", c1, c2]

    epub.write_epub(str(path), book)


def test_epub_to_chapters(tmp_path: Path):
    epub_path = tmp_path / "sample.epub"
    _build_sample_epub(epub_path)

    chapters = epub_to_chapters(epub_path)
    titles = [c.title for c in chapters]

    assert any("Chương 1" in t for t in titles)
    assert any("Chương 2" in t for t in titles)
    assert any("nội dung chương một" in c.text for c in chapters)


def _build_boilerplate_epub(path: Path, n_chapters: int = 5) -> None:
    """Mimics a real-world ebook: every chapter repeats book title / author / a site
    watermark before the actual chapter heading."""
    book = epub.EpubBook()
    book.set_identifier("id-bp")
    book.set_title("Tên Sách")
    book.set_language("vi")

    items = []
    for i in range(1, n_chapters + 1):
        c = epub.EpubHtml(title="Tên Sách", file_name=f"c{i}.xhtml", lang="vi")
        c.content = (
            "<h1>Tên Sách</h1>"          # same heading on every chapter
            "<p>Tên Sách</p>"
            "<p>Tác Giả</p>"
            "<p>SITE-WATERMARK.COM</p>"
            f"<p>Chương {i}: Tiêu đề riêng {i}</p>"
            f"<p>Nội dung của chương {i}.</p>"
        )
        book.add_item(c)
        items.append(c)

    book.toc = tuple(items)
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())
    book.spine = ["nav", *items]
    epub.write_epub(str(path), book)


def test_epub_strips_repeated_boilerplate(tmp_path: Path):
    epub_path = tmp_path / "bp.epub"
    _build_boilerplate_epub(epub_path)
    chapters = epub_to_chapters(epub_path)

    joined = "\n".join(c.text for c in chapters)
    # the watermark and author repeat on every chapter, so they must not be narrated
    assert "SITE-WATERMARK.COM" not in joined
    assert "Tác Giả" not in joined
    # real content survives
    assert "Nội dung của chương 1." in chapters[0].text


def test_epub_titles_prefer_chapter_heading(tmp_path: Path):
    epub_path = tmp_path / "bp.epub"
    _build_boilerplate_epub(epub_path)
    titles = [c.title for c in epub_to_chapters(epub_path)]

    # every chapter's <h1> is the book title, so titles must come from the text instead
    assert len(set(titles)) == len(titles), f"titles not unique: {titles}"
    assert titles[0].startswith("Chương 1:")


def test_load_book_dispatches_by_extension(tmp_path: Path):
    txt_path = tmp_path / "book.txt"
    txt_path.write_text("Nội dung.", encoding="utf-8")
    chapters = load_book(txt_path)
    assert len(chapters) == 1
    assert chapters[0].text == "Nội dung."


@pytest.fixture
def book_cache_dir(tmp_path, monkeypatch):
    d = tmp_path / "book_cache"
    monkeypatch.setattr(config, "BOOK_CACHE_DIR", d)
    return d


def test_load_book_cached_reuses_a_previous_parse(tmp_path: Path, book_cache_dir, monkeypatch):
    """A second open of the same, unchanged file must skip parsing entirely — that's
    the whole point of the cache, and the only way to prove it here is to make the
    real parser explode if it's called again."""
    txt_path = tmp_path / "book.txt"
    txt_path.write_text("Nội dung gốc.", encoding="utf-8")

    first = load_book_cached(txt_path)
    assert first[0].text == "Nội dung gốc."

    def _boom(*a, **k):
        raise AssertionError("load_book_cached must not re-parse an unchanged file")

    monkeypatch.setattr("diareader.text_extract.load_book", _boom)
    second = load_book_cached(txt_path)
    assert second[0].text == "Nội dung gốc."


def test_load_book_cached_reparses_when_the_source_file_changes(tmp_path: Path, book_cache_dir):
    """A stale cache must not be served once the underlying book has actually changed."""
    txt_path = tmp_path / "book.txt"
    txt_path.write_text("Bản đầu tiên.", encoding="utf-8")
    load_book_cached(txt_path)

    time.sleep(0.01)
    txt_path.write_text("Bản đã sửa, dài hơn bản đầu.", encoding="utf-8")
    chapters = load_book_cached(txt_path)
    assert chapters[0].text == "Bản đã sửa, dài hơn bản đầu."


def test_reparse_discards_the_cache_and_reads_from_disk(tmp_path: Path, book_cache_dir):
    """The escape hatch when a cached parse (or the edits stored beside it) is wrong."""
    from diareader.text_extract import clear_book_cache, reparse_book

    txt_path = tmp_path / "book.txt"
    txt_path.write_text("Bản gốc.", encoding="utf-8")
    load_book_cached(txt_path)

    save_book_cache(txt_path, [Chapter(title="x", text="Bản đã sửa tay.")])
    assert load_book_cached(txt_path)[0].text == "Bản đã sửa tay."

    assert reparse_book(txt_path)[0].text == "Bản gốc.", "must re-read the file itself"
    assert load_book_cached(txt_path)[0].text == "Bản gốc.", "and re-cache what it read"

    assert clear_book_cache(txt_path) is True
    assert clear_book_cache(txt_path) is False, "second clear has nothing to remove"


def test_save_book_cache_persists_script_editor_edits(tmp_path: Path, book_cache_dir):
    """The other half of the feature: tags placed in the script editor must survive
    to the next time this same book is opened, not just for the rest of this run."""
    txt_path = tmp_path / "book.txt"
    txt_path.write_text("Bản gốc chưa gắn tag.", encoding="utf-8")
    chapters = load_book_cached(txt_path)

    tagged = [Chapter(title=chapters[0].title, text="Bản gốc. [GiongNu2] Đã gắn tag.")]
    save_book_cache(txt_path, tagged)

    reloaded = load_book_cached(txt_path)
    assert reloaded[0].text == "Bản gốc. [GiongNu2] Đã gắn tag."
