"""Colouring the log panel by outcome, not only by severity.

A chapter's log is now one line per sentence, each carrying its score. That only helps
if the eye can separate the sentences that passed from the retries around them without
reading either, so an accepted sentence is green even though it is logged at INFO.
"""
import logging

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication  # noqa: E402

from diareader.gui.widgets import LogPanel, QtLogHandler  # noqa: E402


@pytest.fixture(scope="module")
def qt_app():
    yield QApplication.instance() or QApplication([])


def _record(level=logging.INFO, message="hello", **extra):
    record = logging.LogRecord("diareader.pipeline", level, __file__, 1, message, (), None)
    for key, value in extra.items():
        setattr(record, key, value)
    return record


class _Bridge:
    def __init__(self):
        self.sent = []

    class _Signal:
        def __init__(self, sink):
            self.sink = sink

        def emit(self, *args):
            self.sink.append(args)

    @property
    def message(self):
        return self._Signal(self.sent)


def test_the_handler_forwards_the_outcome_tag():
    """Passed through logging's `extra=` rather than matched from the message text, so
    rewording a log line cannot silently drop its colour."""
    bridge = _Bridge()
    QtLogHandler(bridge).emit(_record(outcome="pass"))
    assert bridge.sent[0][1] == logging.INFO
    assert bridge.sent[0][2] == "pass"


def test_a_record_without_an_outcome_forwards_an_empty_tag():
    bridge = _Bridge()
    QtLogHandler(bridge).emit(_record())
    assert bridge.sent[0][2] == ""


def test_a_pass_is_green(qt_app):
    panel = LogPanel()
    panel.append_record("Sentence 1 accepted at 96%", logging.INFO, "pass")
    assert LogPanel._PASS_COLOUR.lstrip("#") in panel.document().toHtml().lower().replace("#", "")


def test_ordinary_info_is_white(qt_app):
    panel = LogPanel()
    panel.append_record("Loading weights", logging.INFO)
    assert "ffffff" in panel.document().toHtml().lower()


def test_a_pass_stays_green_even_though_it_is_logged_at_info(qt_app):
    """The outcome wins over the level — that is the whole point of the tag."""
    panel = LogPanel()
    panel.append_record("accepted", logging.INFO, "pass")
    html = panel.document().toHtml().lower()
    assert LogPanel._PASS_COLOUR.lstrip("#") in html.replace("#", "")
    assert "ffffff" not in html


def test_warnings_and_errors_keep_their_own_colours(qt_app):
    panel = LogPanel()
    panel.append_record("regenerating", logging.WARNING)
    panel.append_record("SKIPPED", logging.ERROR)
    html = panel.document().toHtml().lower()
    assert LogPanel._COLOURS[logging.WARNING].lstrip("#") in html.replace("#", "")
    assert LogPanel._COLOURS[logging.ERROR].lstrip("#") in html.replace("#", "")


def test_markup_in_a_log_line_is_escaped(qt_app):
    """Log lines quote the model's own text, and a sentence containing < or & must not
    be able to close the span the colour is applied with."""
    panel = LogPanel()
    panel.append_record("heard '<b>bold</b> & co'", logging.INFO)
    assert "<b>bold</b>" not in panel.document().toHtml()
    assert "bold" in panel.toPlainText()
