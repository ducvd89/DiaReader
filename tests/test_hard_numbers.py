"""Numbers the model cannot say, and the two places the app says so.

Measured on this checkpoint: 1000000 comes back "một ngàn", 100000 comes back "một
ngàn", 1500000000 comes back "một một năm ngàn". It has no scale words, so a figure of
five digits or more is read wrong however the checker counts it — and no counting rule
can fix that, because the audio really is wrong. Written out in words the same model
reads them perfectly. So the app flags them where they can be fixed, in the script, and
asks before spending twenty minutes of GPU time on a chapter that still has them.
"""
import pytest

pytest.importorskip("PySide6")

from PySide6.QtGui import QTextDocument  # noqa: E402
from PySide6.QtWidgets import QApplication, QMessageBox  # noqa: E402

from diareader import config, vn_numbers  # noqa: E402
from diareader.gui.script_editor import MarkerHighlighter  # noqa: E402
from diareader.gui.main_window import MainWindow  # noqa: E402


@pytest.fixture(scope="module")
def qt_app():
    yield QApplication.instance() or QApplication([])


def _formats(text: str):
    """Run the highlighter over one paragraph and hand back its character formats."""
    document = QTextDocument(text)
    highlighter = MarkerHighlighter(document, known_names=["Vietsu"])
    highlighter.rehighlight()
    return document.firstBlock().layout().formats()


def _highlighted(text: str, needle: str) -> bool:
    start = text.index(needle)
    end = start + len(needle)
    return any(
        f.start < end and f.start + f.length > start and f.format.background().color().alpha()
        for f in _formats(text)
    )


def test_a_long_number_is_marked_in_the_script(qt_app):
    assert _highlighted("Số dư của anh là 1500000000 đồng.", "1500000000")


def test_a_short_number_is_left_alone(qt_app):
    assert not _highlighted("Anh chờ đến 2026 nhé.", "2026")


def test_a_phone_number_is_left_alone(qt_app):
    """Long, and read correctly anyway — one digit at a time is the only way anyone
    says it."""
    assert not _highlighted("Gọi 0987654321 giúp em.", "0987654321")


def test_marking_a_number_does_not_disturb_the_voice_markers(qt_app):
    """Two highlighters over the same text, and the one that matters most is the one
    that says who is speaking."""
    text = "[Vietsu] Số dư 1500000000 đồng."
    assert _highlighted(text, "[Vietsu]")
    assert _highlighted(text, "1500000000")


# --- the confirmation before rendering ------------------------------------------------


class _Window:
    """Enough of MainWindow to exercise the check without building a window."""


def _asked(text, answer, monkeypatch):
    seen = {}

    def question(_self, title, message, *args, **kwargs):
        seen["title"], seen["message"] = title, message
        return answer

    monkeypatch.setattr(
        "diareader.gui.main_window.QMessageBox.question", question
    )
    proceed = MainWindow._confirm_hard_numbers(_Window(), text)
    return proceed, seen


def test_an_ordinary_chapter_is_not_interrupted(monkeypatch):
    proceed, seen = _asked("Dạ, em chào anh Minh. Anh chờ hai mươi hai giờ nhé.",
                           QMessageBox.StandardButton.Yes, monkeypatch)
    assert proceed
    assert seen == {}, "nothing to warn about, so nothing should be asked"


def test_saying_no_stops_the_render(monkeypatch):
    """Asked before the work rather than reported after it: afterwards is twenty
    minutes of GPU time and a chapter to redo."""
    proceed, _ = _asked("Số dư 1500000000 đồng.",
                        QMessageBox.StandardButton.No, monkeypatch)
    assert not proceed


def test_saying_yes_renders_anyway(monkeypatch):
    proceed, _ = _asked("Số dư 1500000000 đồng.",
                        QMessageBox.StandardButton.Yes, monkeypatch)
    assert proceed


def test_the_warning_names_the_numbers_it_found(monkeypatch):
    """"Some numbers are wrong" sends you hunting through four hundred sentences."""
    _, seen = _asked("Gọi 19001234 hoặc 0987654321, dư 1500000000 đồng.",
                     QMessageBox.StandardButton.No, monkeypatch)
    assert "19001234" in seen["message"]
    assert "1500000000" in seen["message"]
    assert "0987654321" not in seen["message"], "a phone number is read correctly"


def test_a_repeated_number_is_named_once(monkeypatch):
    _, seen = _asked("Dư 1500000000 đồng, tức 1500000000 đồng chẵn.",
                     QMessageBox.StandardButton.No, monkeypatch)
    assert seen["message"].count("1500000000") == 1


def test_the_warning_says_what_to_do_about_it(monkeypatch):
    _, seen = _asked("Số dư 1000000 đồng.", QMessageBox.StandardButton.No, monkeypatch)
    assert "words" in seen["message"]
    assert "một triệu" in seen["message"]
