from pathlib import Path

import pytest

from diareader import config


@pytest.fixture
def settings(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "APP_DIR", tmp_path / "app")
    monkeypatch.setattr(config, "SETTINGS_PATH", tmp_path / "app" / "settings.json")
    return config.Settings()


def _book(tmp_path, name):
    path = tmp_path / name
    path.write_text("nội dung", encoding="utf-8")
    return str(path)


def test_most_recent_book_comes_first(settings, tmp_path):
    first, second = _book(tmp_path, "a.txt"), _book(tmp_path, "b.txt")
    settings.remember_book(first)
    settings.remember_book(second)
    assert settings.recent_books == [second, first]


def test_reopening_a_book_moves_it_up_without_duplicating(settings, tmp_path):
    a, b = _book(tmp_path, "a.txt"), _book(tmp_path, "b.txt")
    settings.remember_book(a)
    settings.remember_book(b)
    settings.remember_book(a)
    assert settings.recent_books == [a, b]


def test_the_list_stops_at_the_limit(settings, tmp_path):
    for i in range(config.RECENT_BOOKS_LIMIT + 3):
        settings.remember_book(_book(tmp_path, f"book{i}.txt"))
    assert len(settings.recent_books) == config.RECENT_BOOKS_LIMIT


def test_opening_a_book_remembers_where_it_came_from(settings, tmp_path):
    folder = tmp_path / "Sách"
    folder.mkdir()
    book = folder / "truyen.txt"
    book.write_text("x", encoding="utf-8")
    settings.remember_book(str(book))
    assert Path(settings.last_open_dir) == folder


def test_a_deleted_book_is_hidden_but_not_forgotten(settings, tmp_path):
    """Hidden on read, not pruned on write: a book on a drive that is unplugged
    right now has not stopped being recent."""
    kept, removed = _book(tmp_path, "kept.txt"), _book(tmp_path, "gone.txt")
    settings.remember_book(kept)
    settings.remember_book(removed)
    Path(removed).unlink()

    assert settings.existing_recent_books() == [kept]
    assert removed in settings.recent_books


def test_recents_survive_a_save_and_reload(settings, tmp_path, monkeypatch):
    book = _book(tmp_path, "a.txt")
    settings.remember_book(book)
    settings.save()

    reloaded = config.Settings.load()
    assert reloaded.recent_books == [book]
    assert reloaded.last_open_dir == str(tmp_path)


def test_a_settings_file_written_before_recents_existed_still_loads(tmp_path, monkeypatch):
    """Adding a field must not lock anyone out of the app — a settings file always
    outlives the code that wrote it."""
    monkeypatch.setattr(config, "APP_DIR", tmp_path / "app")
    monkeypatch.setattr(config, "SETTINGS_PATH", tmp_path / "app" / "settings.json")
    (tmp_path / "app").mkdir()
    (tmp_path / "app" / "settings.json").write_text('{"speed": 0.9}', encoding="utf-8")

    loaded = config.Settings.load()
    assert loaded.recent_books == []
    assert loaded.last_open_dir == ""
