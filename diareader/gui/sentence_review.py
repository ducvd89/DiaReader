"""Every sentence of the selection, what it sounds like, and a way to redo the bad ones.

The export used to be all-or-nothing: a chapter with three bad sentences in it was
regenerated from the top and the four hundred good ones were thrown away with them.
Sentences are cached individually now (see sentence_cache), so this table is where the
three get picked out, re-rendered — in a different voice if that helps — listened to,
and either adopted into the final file or discarded.

Nothing here writes the book. Adopting a take marks it as the one in use; the MP3 is
built later, from whatever is in use at that moment.
"""
from __future__ import annotations

import logging
import re

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QFont, QFontMetrics, QPalette
from PySide6.QtWidgets import (
    QAbstractItemView, QCheckBox, QComboBox, QDialog, QDialogButtonBox, QFormLayout,
    QHBoxLayout, QHeaderView, QLabel, QMessageBox, QProgressBar, QPushButton,
    QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget
)

from .. import config, sentence_cache, voice
from .audio_preview import AudioPreview
from .script_editor import MAX_FONT_PT, MIN_FONT_PT, voice_colour, voice_ink

log = logging.getLogger(__name__)

# Columns, in the order they are shown.
(COL_PICK, COL_WHERE, COL_TEXT, COL_VOICE, COL_SCORE, COL_HEARD,
 COL_TAKE, COL_PLAY) = range(8)
_HEADINGS = ["", "Where", "Sentence", "Voice", "Check", "Heard", "Take", ""]

# The same red the script editor paints an unresolvable marker in — one error colour
# across the app rather than two that nearly match.
_FAIL_HUE = QColor("#d03b3b")
# Mixed into the table's own background rather than used neat: a row is read, not
# glanced at, and the sentence and the transcript both have to stay legible on top of
# it. Strong enough to pick the row out of four hundred, weak enough to read.
_FAIL_TINT_STRENGTH = 0.32

# Blank space above and below a row's text, in pixels, total. Rows are sized from the
# font rather than from their contents: resizeRowsToContents measures the checkbox and
# the play button too, and those are taller than a line of text, so every row came out
# padded by a widget nobody was reading. A four-hundred-sentence chapter is scrolled,
# and each wasted pixel is a row that is not on screen.
ROW_PADDING_PX = 6
# Never let a row get so short the embedded controls stop being clickable.
MIN_ROW_HEIGHT_PX = 18
# Width the item delegate keeps for itself on each side of a cell. Over-reserving is
# the safe direction: it makes the measured text a shade taller than it needs, where
# under-reserving would clip the last line of a wrapped sentence.
CELL_MARGIN_PX = 6

# Starting width of each column. They are all draggable from here; these only decide
# what the table looks like before anyone touches it, and they add up to a little under
# the dialog's own 1100px so nothing starts off-screen.
_COLUMN_WIDTHS = {
    COL_PICK: 30, COL_WHERE: 150, COL_TEXT: 300, COL_VOICE: 110,
    COL_SCORE: 60, COL_HEARD: 264, COL_TAKE: 130, COL_PLAY: 36,
}


def _blend(base: QColor, hue: QColor, k: float) -> QColor:
    return QColor(
        round(base.red() * (1 - k) + hue.red() * k),
        round(base.green() * (1 - k) + hue.green() * k),
        round(base.blue() * (1 - k) + hue.blue() * k),
    )


# Wrapped text has to start at the top of its cell; centred, a three-line sentence
# sits against a one-line neighbour and the column stops reading as a list.
_TOP_LEFT = Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop

_HEARD_PREFIX = "heard "


def heard_text(transcript: str) -> str:
    """The transcript as the Heard column shows it.

    The stored detail reads "heard 'dạ em chào anh minh' — 5 of 5 syllables",
    which is right for a log line, where it has to say what it is. In a column
    already headed "Heard" the word is repeated four hundred times, and the quotes
    marking it as speech cost width the speech itself wants. Both go; the full
    detail stays on the tooltip, so nothing is lost.

    Left alone when it is not a transcript at all — "checker unavailable
    (OSError)" and "nothing to say" are both things this field can hold.
    """
    text = (transcript or "").strip()
    if text.lower().startswith(_HEARD_PREFIX):
        text = text[len(_HEARD_PREFIX):].lstrip()
    for quote in ("'", '"'):
        if not text.startswith(quote):
            continue
        closing = text.rfind(quote)
        if closing > 0:
            text = (text[1:closing] + text[closing + 1:]).strip()
        break
    return text


def _score_text(take) -> str:
    if take is None:
        return "not rendered"
    return f"{take.score * 100:.0f}%" + ("" if take.accepted else "  ✗")


def _human_size(num_bytes: int) -> str:
    """A figure someone can act on. Rounding a fresh cache to "0 MB" reads as broken
    rather than as small, so the units go down as well as up."""
    if num_bytes >= 1024 ** 3:
        return f"{num_bytes / 1024 ** 3:.2f} GB"
    if num_bytes >= 1024 ** 2:
        return f"{num_bytes / 1024 ** 2:.0f} MB"
    return f"{num_bytes / 1024:.0f} KB"


class CacheSettingsDialog(QDialog):
    """How much disk the takes may use, and a way to throw them away.

    Worth a dialog of its own because the two settings pull opposite ways: the cache
    is what makes one bad sentence fixable without regenerating the chapter, and it is
    also the thing quietly filling the disk. Showing what is actually being used is
    most of the point — without the figure the limit is a guess.
    """

    cache_changed = Signal()

    def __init__(self, cache, settings: config.Settings, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Cache settings")
        self.cache = cache
        self.settings = settings

        self.usage = QLabel()
        self.usage.setStyleSheet("color: gray;")

        self.limit_combo = QComboBox()
        for gigabytes, label in sentence_cache.CACHE_LIMIT_CHOICES:
            self.limit_combo.addItem(label, gigabytes)
        current = self.limit_combo.findData(float(settings.cache_limit_gb))
        self.limit_combo.setCurrentIndex(current if current >= 0 else 1)
        self.limit_combo.currentIndexChanged.connect(self._limit_chosen)
        self.limit_combo.setToolTip(
            "When the cache passes this, other books' takes go first, oldest first, "
            "then the takes you rendered here but never adopted. The takes this book "
            "is actually made of are never deleted to make room."
        )

        clear_book = QPushButton("Clear this book's takes")
        clear_book.clicked.connect(self._clear_book)
        clear_all = QPushButton("Clear every book's takes")
        clear_all.clicked.connect(self._clear_all)

        form = QFormLayout()
        form.addRow("Cache limit:", self.limit_combo)

        buttons = QHBoxLayout()
        buttons.addWidget(clear_book)
        buttons.addWidget(clear_all)
        buttons.addStretch(1)

        close = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        close.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addWidget(self.usage)
        layout.addLayout(form)
        layout.addLayout(buttons)
        layout.addWidget(close)
        self._refresh_usage()

    def _refresh_usage(self) -> None:
        self.usage.setText(
            f"Cached takes use {_human_size(sentence_cache.total_size())} in total, "
            f"{_human_size(self.cache.size_bytes())} of it for this book."
        )

    def _limit_chosen(self) -> None:
        self.settings.cache_limit_gb = float(self.limit_combo.currentData())
        self.settings.save()
        freed = sentence_cache.enforce_limit(self.settings.cache_limit_gb, keep=self.cache)
        self._refresh_usage()
        if freed:
            self.cache_changed.emit()

    def _confirm(self, question: str) -> bool:
        return QMessageBox.question(
            self, "Clear cache", question,
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        ) == QMessageBox.StandardButton.Yes

    def _clear_book(self) -> None:
        if not self._confirm(
            "Delete every rendered sentence for this book?\n\nExporting the audio "
            "afterwards will generate the whole selection again."
        ):
            return
        self.cache.clear()
        self._refresh_usage()
        self.cache_changed.emit()

    def _clear_all(self) -> None:
        if not self._confirm(
            "Delete every rendered sentence, for every book?\n\nNothing else is "
            "affected — scripts, voices and settings all stay."
        ):
            return
        sentence_cache.clear_all()
        self.cache.clear()
        self._refresh_usage()
        self.cache_changed.emit()


class SentenceReviewDialog(QDialog):
    """A row per sentence: tick the bad ones, choose a voice, render again, listen."""

    #: (plan entries to render, voice name or None to keep each sentence's own)
    render_requested = Signal(object, object)

    def __init__(self, plan, cache, settings: config.Settings, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Sentences")
        # Qt.Window rather than the default Qt.Dialog: on Windows a dialog frame has no
        # minimise or maximise box at all, and the hints below are ignored on it. This
        # table is worked through beside the editor and the main window for as long as a
        # chapter takes, so it needs to behave like a window — resizable, minimisable,
        # and not glued on top of everything else.
        self.setWindowFlags(
            Qt.WindowType.Window
            | Qt.WindowType.WindowMinimizeButtonHint
            | Qt.WindowType.WindowMaximizeButtonHint
            | Qt.WindowType.WindowCloseButtonHint
        )
        self.setModal(False)
        self.setSizeGripEnabled(True)
        self.resize(1100, 700)

        self.plan = list(plan)
        self.cache = cache
        self.settings = settings
        self.preview = AudioPreview(self)
        # Fixed once: the colour a voice wears here has to be the colour it wears in the
        # script editor, and that comes from its position in the library.
        self.library = voice.voice_names_in_library_order(settings)
        self._base_colour = self.palette().color(QPalette.ColorRole.Base)
        self._holders: dict[tuple[int, int], QWidget] = {}
        # Rows the current filter session has already shown. See _apply_filter.
        self._pinned: set[int] = set()
        # Set while ticks are being changed in bulk, so the filter is applied once at
        # the end rather than once per checkbox. Both passes are O(rows), and a chapter
        # can be four hundred of them.
        self._bulk = False
        self._font_pt = int(getattr(settings, "sentences_font_pt", 10))
        self._row_height = MIN_ROW_HEIGHT_PX

        self.table = QTableWidget(len(self.plan), len(_HEADINGS))
        self.table.setHorizontalHeaderLabels(_HEADINGS)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        # Wrap rather than elide. A sentence cut off at "…vướng " is the one thing this
        # table exists to show you, and the transcript beside it is how you tell a
        # dropped word from a mispronounced one — neither survives an ellipsis.
        self.table.setWordWrap(True)

        header = self.table.horizontalHeader()
        # Every column draggable. Which of the two long columns matters is the user's
        # call and it changes with the job: casting reads the sentence, checking a
        # failure reads the transcript.
        header.setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        header.setStretchLastSection(False)
        header.setMinimumSectionSize(24)
        for column, width in _COLUMN_WIDTHS.items():
            self.table.setColumnWidth(column, width)
        # A narrower column wraps to more lines, so the rows have to be measured again.
        header.sectionResized.connect(lambda *_: self._resize_rows())

        vertical = self.table.verticalHeader()
        vertical.setVisible(False)
        # Rows are measured here rather than by Qt. ResizeToContents looked right and
        # was not: a hidden vertical header still contributes its own section hint, so
        # a row whose content wanted 19px came back at 24 and no amount of padding
        # tuning moved it. _row_height_for measures the wrapped text and nothing else.
        vertical.setSectionResizeMode(QHeaderView.ResizeMode.Fixed)
        vertical.setMinimumSectionSize(MIN_ROW_HEIGHT_PX)

        self.checks: list[QCheckBox] = []
        self._fill()

        self.only_failed = QPushButton("Tick the ones that failed")
        self.only_failed.setToolTip(
            "Select every sentence the quality check rejected, or that has no audio yet."
        )
        self.only_failed.clicked.connect(self._tick_failed)
        clear = QPushButton("Clear ticks")
        clear.clicked.connect(lambda: self._set_all(False))

        self.filter_button = QPushButton("Show only failed")
        self.filter_button.setCheckable(True)
        self.filter_button.setToolTip(
            "Hide the sentences that passed, leaving the ones that failed and the ones "
            "you have ticked. Press again to show the whole chapter."
        )
        self.filter_button.toggled.connect(self._filter_toggled)

        self.voice_combo = QComboBox()
        self.voice_combo.addItem("Keep each sentence's own voice", None)
        for name in voice.voice_names_in_library_order(settings):
            self.voice_combo.addItem(name, name)
        self.voice_combo.setToolTip(
            "Re-render the ticked sentences in this voice. A voice that reads one "
            "sentence badly often reads it fine on another take, and a different "
            "voice is the next thing to try."
        )

        self.render_button = QPushButton("Render ticked")
        self.render_button.clicked.connect(self._request_render)

        self.progress = QProgressBar()
        self.progress.setVisible(False)

        self.status = QLabel()
        self.status.setStyleSheet("color: gray;")
        self._update_status()

        self.zoom_out_button = QPushButton("A−")
        self.zoom_in_button = QPushButton("A+")
        for button, step, tip in (
            (self.zoom_out_button, -1, "Smaller text"),
            (self.zoom_in_button, 1, "Larger text"),
        ):
            button.setFixedWidth(38)
            button.setToolTip(f"{tip} in the table below")
            button.clicked.connect(lambda _c=False, s=step: self.zoom_by(s))

        cache_settings = QPushButton("Cache…")
        cache_settings.setToolTip(
            "How much disk the rendered sentences may use, and a way to clear them."
        )
        cache_settings.clicked.connect(self._open_cache_settings)

        close = QPushButton("Close")
        close.clicked.connect(self.accept)

        picks = QHBoxLayout()
        picks.addWidget(self.only_failed)
        picks.addWidget(clear)
        picks.addWidget(self.filter_button)
        picks.addStretch(1)
        picks.addWidget(QLabel("Voice:"))
        picks.addWidget(self.voice_combo)
        picks.addWidget(self.render_button)

        footer = QHBoxLayout()
        footer.addWidget(self.status)
        footer.addStretch(1)
        footer.addWidget(self.zoom_out_button)
        footer.addWidget(self.zoom_in_button)
        footer.addWidget(cache_settings)
        footer.addWidget(close)

        layout = QVBoxLayout(self)
        layout.addLayout(picks)
        layout.addWidget(self.table, stretch=1)
        layout.addWidget(self.progress)
        layout.addLayout(footer)

        # Last: it sizes the rows and needs the zoom buttons to exist.
        self._apply_font()

    # -- building the table ------------------------------------------------------------

    # -- text size, and the row height that follows from it ----------------------------

    def zoom_by(self, steps: int) -> None:
        self.set_font_size(self._font_pt + steps)

    def set_font_size(self, points: int) -> None:
        points = max(MIN_FONT_PT, min(MAX_FONT_PT, int(points)))
        if points == self._font_pt:
            return
        self._font_pt = points
        self.settings.sentences_font_pt = points
        self.settings.save()
        self._apply_font()

    def _apply_font(self) -> None:
        """Set the table's text size and re-fit every row to it.

        The embedded controls are shrunk to match rather than left at their natural
        size: a QPushButton and a QComboBox are both taller than a line of text, and
        while they sit at their default height they, not the text, decide how tall a
        row is — which is exactly how the rows ended up padded out with blank space.
        """
        font = QFont(self.table.font())
        font.setPointSize(self._font_pt)
        self.table.setFont(font)

        # One line's worth. Rows themselves are as tall as their wrapped text needs;
        # this is the height the embedded controls are held to, so that a one-line
        # sentence still gets a one-line row.
        self._row_height = max(QFontMetrics(font).height() + ROW_PADDING_PX,
                               MIN_ROW_HEIGHT_PX)
        self.table.verticalHeader().setMinimumSectionSize(
            min(self._row_height, MIN_ROW_HEIGHT_PX)
        )
        for holder in self._holders.values():
            self._fit_holder(holder)
        self._resize_rows()

        self.zoom_out_button.setEnabled(self._font_pt > MIN_FONT_PT)
        self.zoom_in_button.setEnabled(self._font_pt < MAX_FONT_PT)

    def _resize_rows(self) -> None:
        for row in range(self.table.rowCount()):
            self.table.setRowHeight(row, self._row_height_for(row))

    def _row_height_for(self, row: int) -> int:
        """How tall this row has to be for none of its text to be hidden.

        Measured from the wrapped text of the columns that hold prose, floored at one
        line so a short sentence still gets a short row. The embedded controls are not
        consulted: _fit_holder has already pinned them to that same one-line height,
        which is the whole reason a table of one-line sentences is not a table of
        26-pixel rows.
        """
        metrics = QFontMetrics(self.table.font())
        tallest = 0
        for column in (COL_WHERE, COL_TEXT, COL_HEARD):
            item = self.table.item(row, column)
            if item is None or not item.text():
                continue
            width = max(self.table.columnWidth(column) - CELL_MARGIN_PX * 2, 40)
            tallest = max(tallest, metrics.boundingRect(
                0, 0, width, 0, int(Qt.TextFlag.TextWordWrap), item.text()
            ).height())
        return max(tallest + ROW_PADDING_PX, self._row_height)

    def _fit_holder(self, holder: QWidget) -> None:
        holder.setFixedHeight(self._row_height)
        inner = max(self._row_height - 2, 12)
        for child in holder.findChildren(QWidget):
            child.setFixedHeight(inner)
            font = QFont(child.font())
            font.setPointSize(self._font_pt)
            child.setFont(font)

    def _hold(self, row: int, column: int, widget: QWidget) -> QWidget:
        """Put a widget in a cell through a holder that can be given a background.

        A cell widget paints over whatever the QTableWidgetItem behind it says, so the
        checkbox and play columns would stay untinted while the rest of a failed row
        turned red. The holder is what carries the colour for them.
        """
        holder = QWidget()
        holder.setObjectName("cellHolder")
        box = QHBoxLayout(holder)
        box.setContentsMargins(2, 0, 2, 0)
        box.addWidget(widget, alignment=Qt.AlignmentFlag.AlignCenter)
        self.table.setCellWidget(row, column, holder)
        self._holders[(row, column)] = holder
        # Rebuilt holders — a take picker appearing after a re-render — must arrive at
        # the current size, or one row would silently grow taller than its neighbours.
        self._fit_holder(holder)
        return holder

    def _fill(self) -> None:
        self.checks.clear()
        for row, planned in enumerate(self.plan):
            take = self.cache.get(planned.chapter_index, planned.position)

            check = QCheckBox()
            check.toggled.connect(lambda _on=False, r=row: self._tick_changed(r))
            self._hold(row, COL_PICK, check)
            self.checks.append(check)

            where = QTableWidgetItem(f"{planned.chapter_title}  ·  {planned.position + 1}")
            where.setToolTip(planned.chapter_title)
            where.setTextAlignment(_TOP_LEFT)
            self.table.setItem(row, COL_WHERE, where)

            text = QTableWidgetItem(planned.text)
            text.setToolTip(planned.text)
            text.setTextAlignment(_TOP_LEFT)
            self.table.setItem(row, COL_TEXT, text)

            self._set_voice_cell(row, take)
            self._set_score_cell(row, take)
            self._set_heard_cell(row, take)
            self._build_take_picker(row)

            play = QPushButton("▶")
            play.setFixedWidth(34)
            play.setEnabled(take is not None)
            play.clicked.connect(lambda _checked=False, r=row: self._play(r))
            self._hold(row, COL_PLAY, play)

            self._paint_row(row, take)

    def _build_take_picker(self, row: int) -> None:
        """Which rendering of this sentence goes into the file.

        Only shown once there is a choice to make. Picking one adopts it immediately —
        the file is not written until "Export audio file", so there is nothing to undo
        and a confirmation step would only be in the way.
        """
        planned = self.plan[row]
        current = self.cache.get(planned.chapter_index, planned.position)
        others = self.cache.alternatives_for(planned.chapter_index, planned.position)
        if current is None or not others:
            self.table.setCellWidget(row, COL_TAKE, None)
            self._holders.pop((row, COL_TAKE), None)
            self.table.setItem(row, COL_TAKE, QTableWidgetItem("" if current is None else "1"))
            return

        picker = QComboBox()
        for number, take in enumerate([current] + others, start=1):
            label = f"{number}. {take.voice or 'default'} · {take.score * 100:.0f}%"
            picker.addItem(label, take.wav)
        picker.setCurrentIndex(0)
        picker.setToolTip("Which rendering of this sentence goes into the final file.")
        picker.currentIndexChanged.connect(
            lambda index, r=row: self._adopt(r, index)
        )
        self._hold(row, COL_TAKE, picker)

    def take_picker(self, row: int) -> QComboBox | None:
        """The take chooser for a row, or None while there is only one take.

        Reached through the holder rather than `table.cellWidget`, which now hands back
        the wrapper that carries the row's background colour.
        """
        holder = self._holders.get((row, COL_TAKE))
        return holder.findChild(QComboBox) if holder is not None else None

    def _adopt(self, row: int, index: int) -> None:
        planned = self.plan[row]
        current = self.cache.get(planned.chapter_index, planned.position)
        others = self.cache.alternatives_for(planned.chapter_index, planned.position)
        takes = [current] + others
        if not (0 <= index < len(takes)) or takes[index] is None:
            return
        self.cache.adopt(planned.chapter_index, planned.position, takes[index])
        self.cache.save()
        self.refresh_row(row)

    def _set_score_cell(self, row: int, take) -> None:
        item = QTableWidgetItem(_score_text(take))
        item.setTextAlignment(_TOP_LEFT)
        self.table.setItem(row, COL_SCORE, item)

    def _set_heard_cell(self, row: int, take) -> None:
        heard = QTableWidgetItem(heard_text(take.transcript) if take else "")
        if take is not None:
            heard.setToolTip(take.transcript)
        heard.setTextAlignment(_TOP_LEFT)
        self.table.setItem(row, COL_HEARD, heard)

    def _set_voice_cell(self, row: int, take) -> None:
        """The voice, wearing the colour it wears in the script editor.

        Same swatch, same ink, from the same library order — the point of a stable
        colour per voice is that it survives being shown somewhere else. This cell
        keeps it even on a failed row, so recasting a sentence can be checked at a
        glance against the markers in the script.
        """
        name = (take.voice if take else self.plan[row].voice) or ""
        item = QTableWidgetItem(name or "(default)")
        item.setTextAlignment(_TOP_LEFT)
        if name:
            item.setBackground(voice_colour(name, self.library))
            item.setForeground(voice_ink(name, self.library))
        self.table.setItem(row, COL_VOICE, item)

    def _paint_row(self, row: int, take) -> None:
        """Red behind a sentence the check rejected, so a bad one is found by looking
        rather than by reading four hundred percentages.

        Only a take that was generated and rejected. A sentence with no audio yet is
        not a failure, and painting the whole table red before the first render would
        make the colour mean nothing.
        """
        failed = take is not None and not take.accepted
        background = (
            _blend(self._base_colour, _FAIL_HUE, _FAIL_TINT_STRENGTH) if failed else None
        )
        for column in range(self.table.columnCount()):
            if column == COL_VOICE:
                continue  # carries the voice's own colour
            item = self.table.item(row, column)
            if item is not None:
                item.setBackground(background or Qt.GlobalColor.transparent)
            holder = self._holders.get((row, column))
            if holder is not None:
                holder.setStyleSheet(
                    f"QWidget#cellHolder {{ background: {background.name()}; }}"
                    if background else ""
                )

    def refresh_row(self, row: int) -> None:
        planned = self.plan[row]
        take = self.cache.get(planned.chapter_index, planned.position)
        self._set_voice_cell(row, take)
        self._set_score_cell(row, take)
        self._set_heard_cell(row, take)
        play_holder = self._holders.get((row, COL_PLAY))
        if play_holder is not None:
            for button in play_holder.findChildren(QPushButton):
                button.setEnabled(take is not None)
        self._build_take_picker(row)
        self._paint_row(row, take)
        self._apply_row_filter(row, take)
        self.table.setRowHeight(row, self._row_height_for(row))
        self._update_status()

    def refresh_all(self) -> None:
        for row in range(len(self.plan)):
            self.refresh_row(row)

    # -- actions -----------------------------------------------------------------------

    def _tick_changed(self, row: int) -> None:
        if self._bulk:
            return
        self._apply_row_filter(row)
        self._update_status()

    def _set_all(self, ticked: bool) -> None:
        self._bulk = True
        try:
            for check in self.checks:
                check.setChecked(ticked)
        finally:
            self._bulk = False
        self._apply_filter()

    # -- showing only what still needs work --------------------------------------------

    def _filter_toggled(self, _on: bool) -> None:
        """Each press starts a fresh session, so the visible set is recomputed."""
        self._pinned.clear()
        self._apply_filter()

    def _apply_filter(self) -> None:
        """Hide the sentences that passed and were not ticked.

        The visible set grows but never shrinks while the filter stays on, which is
        what `_pinned` is for. Without it, unticking a row would make it vanish from
        under the pointer at the exact moment of being unticked — and the only way
        back to it would be to turn the whole filter off. A row shown once stays shown
        until the filter is pressed again.

        The invariant this preserves matters beyond tidiness: a hidden row is always
        an unticked one, so "Render ticked" can never quietly act on something the
        table is not showing.
        """
        for row in range(len(self.plan)):
            self._apply_row_filter(row)
        self._update_status()

    def _apply_row_filter(self, row: int, take=None) -> None:
        """One row's visibility under the current filter. `take` saves a lookup for
        callers that have already fetched it."""
        if not self.filter_button.isChecked():
            self.table.setRowHidden(row, False)
            return
        planned = self.plan[row]
        if take is None:
            take = self.cache.get(planned.chapter_index, planned.position)
        visible = (
            take is None
            or not take.accepted
            or self.checks[row].isChecked()
            or row in self._pinned
        )
        if visible:
            self._pinned.add(row)
        self.table.setRowHidden(row, not visible)

    def visible_rows(self) -> list[int]:
        return [row for row in range(len(self.plan)) if not self.table.isRowHidden(row)]

    def _tick_failed(self) -> None:
        self._bulk = True
        try:
            for row, planned in enumerate(self.plan):
                take = self.cache.get(planned.chapter_index, planned.position)
                self.checks[row].setChecked(take is None or not take.accepted)
        finally:
            self._bulk = False
        self._apply_filter()

    def ticked_rows(self) -> list[int]:
        return [row for row, check in enumerate(self.checks) if check.isChecked()]

    def _play(self, row: int) -> None:
        planned = self.plan[row]
        take = self.cache.get(planned.chapter_index, planned.position)
        if take is None:
            return
        self.preview.play(self.cache.wav_path(take))

    def _open_cache_settings(self) -> None:
        dialog = CacheSettingsDialog(self.cache, self.settings, self)
        dialog.cache_changed.connect(self.refresh_all)
        dialog.exec()

    def _request_render(self) -> None:
        rows = self.ticked_rows()
        if not rows:
            QMessageBox.information(
                self, "Nothing ticked",
                "Tick the sentences to render, or press “Tick the ones that failed”.",
            )
            return
        self.render_requested.emit([self.plan[row] for row in rows],
                                   self.voice_combo.currentData())

    # -- progress, driven from the window that owns the engine -------------------------

    def set_running(self, running: bool, total: int = 0) -> None:
        self.render_button.setEnabled(not running)
        self.only_failed.setEnabled(not running)
        self.voice_combo.setEnabled(not running)
        self.progress.setVisible(running)
        if running:
            self.progress.setRange(0, max(1, total))
            self.progress.setValue(0)

    def advance(self) -> None:
        self.progress.setValue(self.progress.value() + 1)

    def _update_status(self) -> None:
        rendered = sum(
            1 for p in self.plan
            if self.cache.get(p.chapter_index, p.position) is not None
        )
        failed = sum(
            1 for p in self.plan
            if (t := self.cache.get(p.chapter_index, p.position)) is not None
            and not t.accepted
        )
        hidden = sum(1 for row in range(len(self.plan)) if self.table.isRowHidden(row))
        self.status.setText(
            f"{rendered} of {len(self.plan)} sentences rendered"
            + (f" · {failed} still failing the check" if failed else "")
            # Said out loud, because a filtered table looks exactly like a short book.
            + (f" · {hidden} hidden by the filter" if hidden else "")
        )

    def closeEvent(self, event) -> None:
        self.preview.stop()
        super().closeEvent(event)
