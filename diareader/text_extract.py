"""Extract plain-text chapters from .txt and .epub files."""
from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from bs4 import BeautifulSoup
from ebooklib import epub, ITEM_DOCUMENT

from . import config

log = logging.getLogger(__name__)

_WHITESPACE_RUN = re.compile(r"[ \t\f\v]+")
_BLANK_LINE_RUN = re.compile(r"\n{3,}")


@dataclass
class Chapter:
    title: str
    text: str


def _clean_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _WHITESPACE_RUN.sub(" ", text)
    lines = [ln.strip() for ln in text.split("\n")]
    text = "\n".join(lines)
    text = _BLANK_LINE_RUN.sub("\n\n", text)
    return text.strip()


def txt_to_text(path: str | Path) -> str:
    path = Path(path)
    raw = path.read_text(encoding="utf-8", errors="replace")
    return _clean_text(raw)


def _html_to_text(html: bytes | str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style"]):
        tag.decompose()
    block_tags = ["p", "div", "br", "h1", "h2", "h3", "h4", "h5", "h6", "li"]
    for tag in soup.find_all(block_tags):
        tag.append("\n")
    text = soup.get_text()
    return _clean_text(text)


def _chapter_title(item, index: int, fallback_text: str) -> str:
    try:
        soup = BeautifulSoup(item.get_content(), "html.parser")
        heading = soup.find(["h1", "h2", "h3", "title"])
        if heading and heading.get_text(strip=True):
            return heading.get_text(strip=True)
    except Exception:
        pass
    first_line = fallback_text.strip().split("\n", 1)[0][:60]
    return first_line or f"Chapter {index + 1}"


_BOILERPLATE_SCAN_LINES = 6
_BOILERPLATE_MIN_SHARE = 0.5


def _strip_repeated_boilerplate(chapters: list[Chapter]) -> list[Chapter]:
    """Drop per-chapter header/footer lines that repeat across the whole book.

    Real ebooks routinely prefix every chapter with the book title, author and a
    source-site watermark. Left in, the narrator reads all of it aloud at the start of
    every chapter — one file here would have announced a website 2466 times — and it
    also hides the actual chapter name. A line is treated as boilerplate only when it
    appears near the edge of at least half the chapters, so ordinary prose is safe.
    """
    if len(chapters) < 3:
        return chapters

    counts: dict[str, int] = {}
    for chapter in chapters:
        lines = [ln for ln in chapter.text.split("\n") if ln.strip()]
        edges = set(lines[:_BOILERPLATE_SCAN_LINES]) | set(lines[-_BOILERPLATE_SCAN_LINES:])
        for line in edges:
            counts[line] = counts.get(line, 0) + 1

    threshold = max(3, int(len(chapters) * _BOILERPLATE_MIN_SHARE))
    boilerplate = {line for line, n in counts.items() if n >= threshold}
    if not boilerplate:
        return chapters

    cleaned: list[Chapter] = []
    for chapter in chapters:
        lines = [ln for ln in chapter.text.split("\n") if ln.strip()]
        start, end = 0, len(lines)
        while start < end and lines[start] in boilerplate:
            start += 1
        while end > start and lines[end - 1] in boilerplate:
            end -= 1
        kept = lines[start:end]
        cleaned.append(Chapter(title=chapter.title, text="\n".join(kept)))

    dropped = sum(1 for c in boilerplate)
    log.info("Removed %d repeated boilerplate line(s) from every chapter", dropped)
    return cleaned


# "Chương 12: ..." / "Chapter 12 - ...". Books often open a chapter with a volume or
# part line before the chapter heading, so the first line is not reliably the title.
_CHAPTER_HEADING = re.compile(r"^\s*(chương|chapter|hồi|phần)\s+\d+", re.IGNORECASE)


def _first_line(text: str, limit: int = 60) -> str:
    lines = [ln.strip() for ln in text.strip().split("\n") if ln.strip()]
    if not lines:
        return ""
    for line in lines[:5]:
        if _CHAPTER_HEADING.match(line):
            return line[:limit]
    return lines[0][:limit]


def _deduplicate_titles(chapters: list[Chapter]) -> list[Chapter]:
    """Replace repeated headings with each chapter's opening line.

    Many ebooks put the *book* title in every chapter's <h1>/<title>, which makes a
    chapter list useless for picking what to read — one real file had the same
    heading on all 2466 chapters. When a heading isn't unique, the first line of the
    chapter's own text (after boilerplate removal) identifies it far better.
    """
    counts: dict[str, int] = {}
    for chapter in chapters:
        counts[chapter.title] = counts.get(chapter.title, 0) + 1

    result = []
    for index, chapter in enumerate(chapters):
        title = chapter.title
        if counts.get(title, 0) > 1:
            title = _first_line(chapter.text) or f"Chapter {index + 1}"
        result.append(Chapter(title=title, text=chapter.text))
    return result


ProgressCallback = Callable[[int, int], None]  # (done, total)


def epub_to_chapters(path: str | Path, on_progress: ProgressCallback | None = None) -> list[Chapter]:
    """Parse an epub into chapters.

    on_progress(done, total) fires as documents are converted — a 2467-chapter book
    takes long enough here that a silent UI looks frozen.
    """
    path = Path(path)
    book = epub.read_epub(str(path))
    items = list(book.get_items_of_type(ITEM_DOCUMENT))
    total = len(items)

    chapters: list[Chapter] = []
    for index, item in enumerate(items):
        text = _html_to_text(item.get_content())
        if text and len(text) >= 5:
            chapters.append(Chapter(title=_chapter_title(item, index, text), text=text))
        if on_progress and (index % 25 == 0 or index == total - 1):
            on_progress(index + 1, total)

    return _deduplicate_titles(_strip_repeated_boilerplate(chapters))


def load_book(path: str | Path, on_progress: ProgressCallback | None = None) -> list[Chapter]:
    """Load any supported book file into a list of chapters (txt -> single chapter)."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".txt":
        if on_progress:
            on_progress(1, 1)
        return [Chapter(title=path.stem, text=txt_to_text(path))]
    if suffix == ".epub":
        return epub_to_chapters(path, on_progress=on_progress)
    raise ValueError(f"Unsupported book format: {suffix}")


def _cache_file(path: Path) -> Path:
    """One JSON file per source book, named by a hash of its resolved path so the
    cache survives the book being opened from a different working directory."""
    digest = hashlib.sha1(str(path.resolve()).encode("utf-8")).hexdigest()
    return config.BOOK_CACHE_DIR / f"{digest}.json"


def _read_cache(path: Path) -> list[Chapter] | None:
    """Cached chapters, or None if there is no usable cache.

    Keyed on the source file's size and mtime rather than just its path: if the
    epub/txt on disk changed since it was cached, the stale parse (and any voice
    tags saved against it) must not be served silently.
    """
    cache_file = _cache_file(path)
    if not cache_file.exists():
        return None
    try:
        data = json.loads(cache_file.read_text(encoding="utf-8"))
        stat = path.stat()
        if data.get("size") != stat.st_size or data.get("mtime") != stat.st_mtime:
            return None
        return [Chapter(title=c["title"], text=c["text"]) for c in data["chapters"]]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as e:
        log.warning("Ignoring unreadable book cache for %s: %s", path, e)
        return None


def save_book_cache(path: str | Path, chapters: list[Chapter]) -> None:
    """Persist chapters (parsed text, or text edited in the script editor) against
    their source file, so the next open reuses them instead of re-parsing — and any
    voice markers placed in the script editor survive to the next export."""
    path = Path(path)
    config.BOOK_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    stat = path.stat()
    payload = {
        "source_path": str(path.resolve()),
        "size": stat.st_size,
        "mtime": stat.st_mtime,
        "chapters": [{"title": c.title, "text": c.text} for c in chapters],
    }
    _cache_file(path).write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def clear_book_cache(path: str | Path) -> bool:
    """Forget the cached parse for a book. Returns whether anything was removed.

    Also discards script-editor edits, since they live in the same file — the caller
    is expected to have said so before asking.
    """
    cache_file = _cache_file(Path(path))
    if not cache_file.exists():
        return False
    cache_file.unlink()
    log.info("Cleared the cached parse for %s", Path(path).name)
    return True


def reparse_book(path: str | Path, on_progress: ProgressCallback | None = None) -> list[Chapter]:
    """Parse a book from disk, ignoring and replacing any cached copy."""
    clear_book_cache(path)
    chapters = load_book(path, on_progress=on_progress)
    save_book_cache(path, chapters)
    return chapters


def load_book_cached(path: str | Path, on_progress: ProgressCallback | None = None) -> list[Chapter]:
    """Like load_book, but reuses a previous parse (including any saved script-editor
    edits) when the source file hasn't changed since it was cached — skipping the
    ebooklib/BeautifulSoup pass that makes opening a large epub feel slow."""
    path = Path(path)
    cached = _read_cache(path)
    if cached is not None:
        if on_progress:
            on_progress(1, 1)
        return cached
    chapters = load_book(path, on_progress=on_progress)
    save_book_cache(path, chapters)
    return chapters
