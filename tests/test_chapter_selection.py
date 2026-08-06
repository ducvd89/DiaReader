"""Chapter selection must render exactly the chapters asked for.

A real run reported "export chapters 2 to 2" starting on chapter 1 (a 2467-entry table
of contents), so this pins the whole path: spinbox values -> indices -> which chapters
the pipeline actually generates.
"""
import numpy as np
import pytest

from diareader import config, pipeline
from diareader.text_extract import Chapter


class RecordingEngine:
    """Records what it was asked to speak, and returns plausible-length audio."""

    def __init__(self):
        self.spoken: list[str] = []

    def generate(self, tagged_text, voice_ref=None, **overrides):
        self.spoken.append(tagged_text)
        seconds = max(len(tagged_text) / 20.3, 0.3)
        return np.ones(int(seconds * config.SAMPLE_RATE), dtype=np.float32)


def _book() -> list[Chapter]:
    return [
        Chapter(title="Mục lục", text="Mục một. Mục hai. Mục ba."),
        Chapter(title="Chương 1", text="Nội dung chương một."),
        Chapter(title="Chương 2", text="Nội dung chương hai."),
        Chapter(title="Chương 3", text="Nội dung chương ba."),
    ]


def _titles_generated(chapter_indices) -> list[str]:
    engine = RecordingEngine()
    results = list(
        pipeline.iter_book_audio(
            "unused.epub", engine, config.Settings(),
            chapter_indices=chapter_indices,
            chapters=_book(),
        )
    )
    # dict.fromkeys keeps first-seen order without duplicates
    return list(dict.fromkeys(r.chapter_title for r in results))


def test_single_chapter_selection_skips_everything_else():
    assert _titles_generated([1]) == ["Chương 1"]


def test_table_of_contents_is_not_rendered_when_not_selected():
    titles = _titles_generated([2])
    assert "Mục lục" not in titles, "chapter 0 leaked into a selection that excluded it"
    assert titles == ["Chương 2"]


def test_range_selection_renders_only_that_range():
    assert _titles_generated([1, 2]) == ["Chương 1", "Chương 2"]


def test_none_renders_whole_book():
    assert _titles_generated(None) == ["Mục lục", "Chương 1", "Chương 2", "Chương 3"]


def test_chapter_index_is_reported_against_the_whole_book():
    """Progress must report the book-wide index, or the UI highlights the wrong row."""
    engine = RecordingEngine()
    results = list(
        pipeline.iter_book_audio(
            "unused.epub", engine, config.Settings(),
            chapter_indices=[2], chapters=_book(),
        )
    )
    assert {r.chapter_index for r in results} == {2}


# --- the dialog layer ---------------------------------------------------------------

@pytest.fixture(scope="module")
def qt_app():
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    return app


def _dialog(qt_app, preselected, n=10):
    from diareader.gui.export_dialog import ExportDialog

    return ExportDialog(
        config.Settings(), [f"Ch {i}" for i in range(n)], [5] * n,
        preselected=preselected, default_path="out.mp3",
    )


def test_dialog_single_chapter_maps_to_one_index(qt_app):
    dlg = _dialog(qt_app, preselected=[1])
    assert (dlg.from_spin.value(), dlg.to_spin.value()) == (2, 2)
    assert dlg._selected_indices() == [1]


def test_dialog_second_chapter_never_includes_the_first(qt_app):
    dlg = _dialog(qt_app, preselected=None)
    dlg.from_spin.setValue(2)
    dlg.to_spin.setValue(2)
    assert dlg._selected_indices() == [1]
    assert 0 not in dlg._selected_indices()


@pytest.mark.parametrize("first, second", [(5, 2), (2, 5), (1, 1), (7, 7), (10, 3)])
def test_dialog_range_always_matches_what_is_displayed(qt_app, first, second):
    """The invariant that matters: whatever the two spinboxes show is exactly what gets
    rendered. Silently sorting an inverted pair broke this — the boxes read "2 to 2"
    while the export actually covered chapters 1-2, dragging in a table of contents.
    """
    dlg = _dialog(qt_app, preselected=None)
    dlg.from_spin.setValue(first)
    dlg.to_spin.setValue(second)

    shown_from, shown_to = dlg.from_spin.value(), dlg.to_spin.value()
    assert shown_from <= shown_to, "the range must never be inverted"
    assert dlg._selected_indices() == list(range(shown_from - 1, shown_to)), (
        f"boxes show {shown_from}-{shown_to} but indices are {dlg._selected_indices()}"
    )


def test_dialog_shows_the_title_each_end_resolves_to(qt_app):
    """Numbers alone are easy to misread; the dialog must name the chapters it will
    actually render so a wrong pick is visible before any GPU time is spent."""
    from diareader.gui.export_dialog import ExportDialog

    titles = ["Mục lục", "Chương 1: Sơn Biên", "Chương 2: Thanh Ngưu", "Chương 3: Thất Huyền"]
    dlg = ExportDialog(config.Settings(), titles, [5] * len(titles), preselected=None)

    dlg.from_spin.setValue(2)
    dlg.to_spin.setValue(3)
    assert dlg.from_title.text() == "Chương 1: Sơn Biên"
    assert dlg.to_title.text() == "Chương 2: Thanh Ngưu"

    dlg.from_spin.setValue(1)
    assert dlg.from_title.text() == "Mục lục", "must track the spinbox, not go stale"


def test_dialog_elides_a_very_long_title(qt_app):
    from diareader.gui.export_dialog import ExportDialog

    long_title = "Chương 1: " + "rất dài " * 20
    dlg = ExportDialog(config.Settings(), [long_title], [5], preselected=None)
    assert len(dlg.from_title.text()) <= 70


def test_dialog_single_chapter_two_is_exactly_index_one(qt_app):
    """The exact case reported: 'export chapter 2 to 2' must never touch chapter 1."""
    dlg = _dialog(qt_app, preselected=None)
    dlg.from_spin.setValue(2)
    dlg.to_spin.setValue(2)
    assert dlg._selected_indices() == [1]
    assert _titles_generated(dlg._selected_indices()) == ["Chương 1"]
