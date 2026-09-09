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
#
# The range spinboxes this section used to test are gone. They protected an invariant —
# "whatever the dialog shows is exactly what gets rendered" — that an inverted pair had
# broken once, quietly widening an export downwards into a 2467-entry table of contents.
# A list of checkboxes cannot express an inverted range at all, so the invariant now
# holds by construction; what is left to test is that the list shows the right chapters.

@pytest.fixture(scope="module")
def qt_app():
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    return app


def _dialog(qt_app, preselected=None, n=10, cached=None):
    from diareader.gui.export_dialog import ExportDialog

    return ExportDialog(
        config.Settings(), [f"Ch {i}" for i in range(n)], [5] * n,
        cached_per_chapter=cached if cached is not None else {i: 5 for i in range(n)},
        preselected=preselected, default_path="out.mp3",
    )


def test_only_rendered_chapters_are_offered(qt_app):
    """A chapter with no audio has nothing to save, and offering it would write
    silence into the book."""
    dlg = _dialog(qt_app, cached={1: 5, 4: 3})
    labels = [dlg.chapter_list.item(r).text() for r in range(dlg.chapter_list.count())]
    assert len(labels) == 2
    assert labels[0].startswith("Ch 1")
    assert "3 of 5 sentences rendered" in labels[1]


def test_the_selection_is_exactly_what_is_ticked(qt_app):
    from PySide6.QtCore import Qt

    dlg = _dialog(qt_app, cached={0: 5, 1: 5, 2: 5})
    for row in range(dlg.chapter_list.count()):
        dlg.chapter_list.item(row).setCheckState(Qt.CheckState.Unchecked)
    dlg.chapter_list.item(1).setCheckState(Qt.CheckState.Checked)

    assert dlg._selected_indices() == [1]
    assert dlg.request().chapter_indices == [1]


def test_chapter_two_alone_never_drags_in_chapter_one(qt_app):
    """The exact case reported against the old range boxes."""
    from PySide6.QtCore import Qt

    dlg = _dialog(qt_app, cached={0: 5, 1: 5})
    dlg.chapter_list.item(0).setCheckState(Qt.CheckState.Unchecked)
    dlg.chapter_list.item(1).setCheckState(Qt.CheckState.Checked)
    assert dlg._selected_indices() == [1]
    assert _titles_generated(dlg._selected_indices()) == ["Chương 1"]


def test_the_chapter_list_selection_is_carried_in(qt_app):
    from PySide6.QtCore import Qt

    dlg = _dialog(qt_app, preselected=[2], cached={0: 5, 2: 5})
    states = {dlg.chapter_list.item(r).data(Qt.ItemDataRole.UserRole):
              dlg.chapter_list.item(r).checkState() for r in range(dlg.chapter_list.count())}
    assert states[2] == Qt.CheckState.Checked
    assert states[0] == Qt.CheckState.Unchecked


def test_nothing_rendered_leaves_nothing_to_save(qt_app):
    from PySide6.QtWidgets import QDialogButtonBox

    dlg = _dialog(qt_app, cached={})
    assert dlg.chapter_list.count() == 0
    assert dlg.request() is None
    assert not dlg.buttons.button(QDialogButtonBox.StandardButton.Save).isEnabled()


def test_no_destination_is_not_a_request(qt_app):
    dlg = _dialog(qt_app)
    dlg.path_edit.setText("   ")
    assert dlg.request() is None


def test_a_partly_rendered_chapter_says_it_will_leave_gaps(qt_app):
    """Saving a chapter that is missing sentences is allowed — one bad sentence must
    not make a book unsaveable — but it must not be a surprise."""
    dlg = _dialog(qt_app, cached={0: 3})
    assert "2 sentence(s) have no audio" in dlg.summary.text()


def test_it_does_not_ask_for_a_voice_or_a_speed(qt_app):
    """The voice comes from the script's markers and the speed is baked into each
    take. Offering either here could only contradict what was already rendered."""
    dlg = _dialog(qt_app)
    assert not hasattr(dlg, "voice_combo")
    assert not hasattr(dlg, "speed_combo")


# --- not overwriting a finished export by accident -----------------------------------


def _accept(dlg, answer, monkeypatch):
    from PySide6.QtWidgets import QMessageBox

    seen = {}

    def question(_self, title, message, *args, **kwargs):
        seen["title"], seen["message"] = title, message
        return answer

    monkeypatch.setattr("diareader.gui.export_dialog.QMessageBox.question", question)
    dlg.accept()
    return seen


def test_a_new_filename_is_not_questioned(qt_app, tmp_path, monkeypatch):
    from PySide6.QtWidgets import QMessageBox

    dlg = _dialog(qt_app)
    dlg.path_edit.setText(str(tmp_path / "book.mp3"))
    seen = _accept(dlg, QMessageBox.StandardButton.No, monkeypatch)
    assert seen == {}
    assert dlg.result() == dlg.DialogCode.Accepted


def test_an_existing_file_has_to_be_confirmed(qt_app, tmp_path, monkeypatch):
    """QFileDialog asks on its own, but only about a path chosen through it. One typed
    into the box beside it used to reach the writer unchallenged — and an export can
    be hours of work landing on hours of work."""
    from PySide6.QtWidgets import QMessageBox

    existing = tmp_path / "book.mp3"
    existing.write_bytes(b"the previous export")

    dlg = _dialog(qt_app)
    dlg.path_edit.setText(str(existing))
    seen = _accept(dlg, QMessageBox.StandardButton.No, monkeypatch)

    assert "book.mp3" in seen["message"]
    assert dlg.result() != dlg.DialogCode.Accepted
    assert existing.read_bytes() == b"the previous export"


def test_saying_replace_goes_ahead(qt_app, tmp_path, monkeypatch):
    from PySide6.QtWidgets import QMessageBox

    existing = tmp_path / "book.mp3"
    existing.write_bytes(b"the previous export")

    dlg = _dialog(qt_app)
    dlg.path_edit.setText(str(existing))
    _accept(dlg, QMessageBox.StandardButton.Yes, monkeypatch)
    assert dlg.result() == dlg.DialogCode.Accepted
