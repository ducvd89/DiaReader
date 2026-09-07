"""Reusable widgets: chapter list, player controls, and a live log panel."""
from __future__ import annotations

import logging

from PySide6.QtCore import QObject, Signal
from PySide6.QtGui import QFont, QTextCursor
from PySide6.QtWidgets import (
    QHBoxLayout, QLabel, QListWidget, QPlainTextEdit, QProgressBar, QPushButton,
    QVBoxLayout, QWidget
)


class ChapterListWidget(QListWidget):
    """Multi-select chapter list.

    Selection is not a nicety: a real ebook here had 2467 chapters / 255k sentences,
    which is thousands of hours of generation, so rendering a whole book is rarely
    what anyone wants.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)

    def set_chapters(self, titles: list[str], chunk_counts: list[int] | None = None) -> None:
        self.clear()
        if not titles:
            self.addItem("(no chapters found)")
            return
        for i, title in enumerate(titles):
            count = chunk_counts[i] if chunk_counts and i < len(chunk_counts) else None
            label = f"{i + 1}. {title}" + (f"  ({count} sentences)" if count is not None else "")
            self.addItem(label)
        self.setCurrentRow(0)

    def selected_indices(self) -> list[int]:
        return sorted(i.row() for i in self.selectedIndexes())

    def highlight_chapter(self, index: int) -> None:
        if 0 <= index < self.count():
            self.scrollToItem(self.item(index))


# Offered tempos. atempo covers 0.5-2.0 natively, so this is its full useful range.
SPEED_CHOICES = [0.5, 0.6, 0.7, 0.75, 0.8, 0.9, 1.0, 1.1, 1.25, 1.5, 1.75, 2.0]


def speed_label(speed: float) -> str:
    return f"{speed:g}x" + ("  (normal)" if abs(speed - 1.0) < 0.001 else "")


class LogBridge(QObject):
    """Carries log records from worker threads to the GUI thread.

    Qt signals marshal across threads for us, so the handler can be attached to
    loggers that fire on QThread workers without touching widgets directly.
    """

    # formatted text, levelno, outcome tag ("" when the record carries none)
    message = Signal(str, int, str)


class QtLogHandler(logging.Handler):
    def __init__(self, bridge: LogBridge):
        super().__init__()
        self.bridge = bridge

    def emit(self, record: logging.LogRecord) -> None:
        try:
            # `outcome` is set by pipeline.py through logging's `extra=`, on the lines
            # that say a sentence was accepted. Passing the tag rather than matching
            # the message text keeps the colouring from breaking the next time that
            # wording changes.
            self.bridge.message.emit(
                self.format(record), record.levelno, getattr(record, "outcome", "") or ""
            )
        except RuntimeError:
            pass  # window already torn down


class LogPanel(QPlainTextEdit):
    """Shows pipeline progress and, crucially, the coverage warnings — this model can
    silently drop text, and those warnings are the only sign it happened."""

    _COLOURS = {
        logging.WARNING: "#b8860b",
        logging.ERROR: "#c0392b",
        logging.CRITICAL: "#c0392b",
    }
    # A sentence the pipeline accepted. Green so a chapter's log reads as a column of
    # outcomes — every sentence now reports its score, and the eye needs to separate
    # "accepted at 96%" from the retries around it without reading either.
    _PASS_COLOUR = "#3fb950"
    # Everything else. Explicit rather than inherited so ordinary progress lines keep
    # one colour whatever the widget's palette happens to be.
    _INFO_COLOUR = "#ffffff"

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setReadOnly(True)
        self.setMaximumBlockCount(2000)
        font = QFont("Consolas")
        font.setStyleHint(QFont.StyleHint.Monospace)
        font.setPointSize(9)
        self.setFont(font)

    def append_record(self, text: str, levelno: int, outcome: str = "") -> None:
        # A pass wins over its level: an accepted sentence is logged at INFO, and the
        # point of the colour is to mark the outcome rather than the severity.
        if outcome == "pass":
            colour = self._PASS_COLOUR
        else:
            colour = self._COLOURS.get(levelno, self._INFO_COLOUR)
        escaped = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        self.appendHtml(f'<span style="color:{colour};">{escaped}</span>')
        self.moveCursor(QTextCursor.MoveOperation.End)


class PlayerControls(QWidget):
    load_model_clicked = Signal()
    unload_model_clicked = Signal()
    edit_script_clicked = Signal()
    review_sentences_clicked = Signal()
    stop_clicked = Signal()
    export_clicked = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)

        self.status_label = QLabel("No book loaded")
        self.status_label.setWordWrap(True)

        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFormat("%v / %m chunks")
        self.progress_bar.setVisible(False)

        # Loading takes minutes and ~6.5 GB of VRAM. Having it as its own button means
        # you can pay that cost up front and watch it, instead of discovering it when
        # you press Play.
        self.load_model_button = QPushButton("Load model")
        self.unload_model_button = QPushButton("Unload")
        self.unload_model_button.setToolTip(
            "Release the model and free VRAM without closing the app."
        )
        self.edit_script_button = QPushButton("Edit script…")
        self.edit_script_button.setToolTip(
            "Cast the selected chapter: drag voice markers into the text, then export it."
        )
        self.review_button = QPushButton("Sentences…")
        self.review_button.setToolTip(
            "Every sentence of the selection, with the check's verdict on each. Tick "
            "the bad ones to render them again — in another voice if that helps — "
            "listen to the result, and keep it or leave the old one."
        )
        self.stop_button = QPushButton("Stop")
        self.export_button = QPushButton("Save final file…")
        self.export_button.setToolTip(
            "Assemble the cached sentences into a WAV or MP3. Anything not rendered "
            "yet is generated first."
        )

        self.load_model_button.clicked.connect(self.load_model_clicked)
        self.unload_model_button.clicked.connect(self.unload_model_clicked)
        self.edit_script_button.clicked.connect(self.edit_script_clicked)
        self.review_button.clicked.connect(self.review_sentences_clicked)
        self.stop_button.clicked.connect(self.stop_clicked)
        self.export_button.clicked.connect(self.export_clicked)

        buttons = QHBoxLayout()
        for b in (
            self.load_model_button, self.unload_model_button,
            self.edit_script_button, self.review_button,
            self.export_button, self.stop_button,
        ):
            buttons.addWidget(b)

        self.log_panel = LogPanel()

        layout = QVBoxLayout(self)
        layout.addWidget(self.status_label)
        layout.addWidget(self.progress_bar)
        layout.addLayout(buttons)
        layout.addWidget(QLabel("Log"))
        layout.addWidget(self.log_panel, stretch=1)

        self.set_idle()

    # --- state ---------------------------------------------------------------

    def set_idle(self, *, model_loaded: bool = False) -> None:
        self.load_model_button.setEnabled(not model_loaded)
        self.load_model_button.setText("Model loaded" if model_loaded else "Load model")
        self.unload_model_button.setEnabled(model_loaded)
        self.edit_script_button.setEnabled(True)
        self.review_button.setEnabled(True)
        self.export_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.progress_bar.setVisible(False)

    def set_running(self, *, total: int) -> None:
        self.load_model_button.setEnabled(False)
        self.unload_model_button.setEnabled(False)
        # Deliberately left usable. An export runs for hours on its own thread and
        # touches nothing the editor touches — it holds its own copy of the chapters
        # it is rendering — so tagging the *next* chapter while this one generates is
        # both safe and the obvious thing to want to do with the wait.
        self.edit_script_button.setEnabled(True)
        self.review_button.setEnabled(False)
        self.export_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.progress_bar.setVisible(True)
        if total > 0:
            self.progress_bar.setRange(0, total)
            self.progress_bar.setValue(0)
        else:
            self.progress_bar.setRange(0, 0)  # unknown total: indeterminate

    def set_indeterminate(self, label: str) -> None:
        """For the model load, where there is no chunk count to count against."""
        self.load_model_button.setEnabled(False)
        self.unload_model_button.setEnabled(False)
        self.edit_script_button.setEnabled(True)  # see set_running
        self.review_button.setEnabled(False)
        self.export_button.setEnabled(False)
        self.stop_button.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setFormat(label)

    def set_progress(self, done: int, total: int) -> None:
        self.progress_bar.setFormat("%v / %m chunks")
        if total > 0:
            self.progress_bar.setRange(0, total)
            self.progress_bar.setValue(done)

    def set_status(self, text: str) -> None:
        self.status_label.setText(text)
