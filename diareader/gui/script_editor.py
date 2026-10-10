"""Script editor: drag coloured voice chips into a chapter to cast it.

The pipeline already understands inline markers like [GiongNu2]; this makes placing them
a visual job instead of hand-typing names that have to match the library exactly. Each
voice keeps one stable colour so a long script can be scanned for who speaks where.
"""
from __future__ import annotations

import logging
import re

from PySide6.QtCore import QMimeData, Qt, Signal
from PySide6.QtGui import (
    QColor, QDrag, QFont, QGuiApplication, QPalette, QSyntaxHighlighter, QTextCharFormat,
    QTextCursor
)
from PySide6.QtWidgets import (
    QComboBox, QDialog, QDialogButtonBox, QGroupBox, QHBoxLayout, QLabel,
    QMessageBox, QPlainTextEdit, QPushButton, QVBoxLayout, QWidget
)

from .. import chunking, config, vn_numbers, voice
from .flow_layout import wrapping_row

log = logging.getLogger(__name__)

# Hues and their order are both load-bearing, and both were computed rather than
# picked by eye (OKLab ΔE×100 under Machado-Oliveira-Fernandes colour-blindness
# simulation). Two markers can sit anywhere relative to each other in a script, so
# EVERY pair has to be separable, not just neighbours — the first five slots clear
# that all-pairs bar with room to spare (worst pair: CVD ΔE 13.0, normal-vision
# 16.3, against gates of 8 and 15). Slots 6-8 exist so a large library still gets
# distinct colours; they only clear the weaker adjacent bar, which is fine because
# a marker always spells its voice name out — colour is a scanning aid here, never
# the sole carrier of identity.
#
# What this replaced: colours used to come from `md5(name) % len(palette)`, which
# collided — "Latradio" and "NuMN1" both landed on #f6bd60, i.e. two different
# voices rendered identically. Positional assignment cannot collide at all, and
# because voices are numbered in library order (append-only), adding a voice hands
# it the next free slot instead of recolouring the ones already in use.
#
# The first FOUR are additionally the best set for the dark theme this app usually
# renders in: requiring each swatch to clear 3:1 against a dark editor background
# leaves exactly blue/yellow/magenta/green as the largest set that still clears the
# all-pairs bar. No five-hue set satisfies both, so slot 5 onward trades one for the
# other — violet sits there deliberately, because it separates cleanly from the
# first four and merely looks dim, whereas the brighter alternatives (red beside
# magenta) are the confusable kind of wrong this palette exists to prevent. Its
# white ink reads at 8.56:1, the highest of any swatch, so the marker text stays
# legible even where the block itself is quiet.
#
# Red sits LAST, not sixth, and that is a correctness fix rather than taste. The
# marker for a voice that no longer exists is painted #d03b3b, and series red
# #e34948 measures ΔE 4.8 from it — two near-identical reds meaning "a real voice"
# and "this voice is gone". A sixth voice hit that in practice. Aqua takes slot 6
# instead: ΔE 31.9 from the error colour, and it is the only candidate that also
# clears the normal-vision floor there (15.6 against red's 13.2 and orange's 12.9).
_PALETTE = [
    "#2a78d6",  # blue
    "#eda100",  # yellow
    "#e87ba4",  # magenta
    "#008300",  # green
    "#4a3aa7",  # violet
    "#1baf7a",  # aqua
    "#eb6834",  # orange
    "#e34948",  # red — last, see above
]

# Chosen per hue by measured WCAG contrast against the swatch itself rather than a
# single global ink: #4a3aa7 with dark text is 2.45:1 (unreadable), with white 8.56:1.
# Black here is pure #000000 rather than a softened near-black — on the blue swatch
# #111111 measures 4.28:1 and #000000 measures 4.76:1, which is the difference
# between failing and clearing the 4.5:1 bar. Worst pairing below is 4.76:1.
_PALETTE_INK = [
    "#000000", "#000000", "#000000", "#ffffff",
    "#ffffff", "#000000", "#000000", "#000000",
]

# Deliberately outside the palette above: a marker naming a voice that no longer
# exists is an error state, not a ninth voice.
#
# Its red is also NOT what identifies it. The first six slots stay ΔE >= 15 from it,
# but slots 7 (orange, 10.8) and 8 (red, 4.8) cannot: eight categorical hues cover
# most of the wheel, so no error colour can sit 15 away from all of them. Rather than
# pretend otherwise, the error marker additionally wears a wave underline — the
# distinction lives in a channel the palette does not compete for, and survives being
# read by someone who cannot separate the two reds at all.
_UNKNOWN_COLOUR = "#d03b3b"
_UNKNOWN_INK = "#ffffff"


def voice_slot(name: str, library: "list[str] | tuple[str, ...]") -> int | None:
    """Palette slot for a voice: its position in the library, wrapping if the
    library outgrows the palette. None for a name the library doesn't contain."""
    try:
        return library.index(name) % len(_PALETTE)
    except ValueError:
        return None


def voice_colour(name: str, library: "list[str] | tuple[str, ...]" = ()) -> QColor:
    slot = voice_slot(name, library)
    return QColor(_UNKNOWN_COLOUR if slot is None else _PALETTE[slot])


def voice_ink(name: str, library: "list[str] | tuple[str, ...]" = ()) -> QColor:
    """Text colour to lay over `voice_colour(name)`."""
    slot = voice_slot(name, library)
    return QColor(_UNKNOWN_INK if slot is None else _PALETTE_INK[slot])


# How much of a voice's hue to mix into the background of the text it reads. Enough
# to group a passage at a glance, little enough that the prose stays the thing being
# read rather than the colour.
SPAN_TINT_STRENGTH = 0.22


def voice_span_tint(name: str, library, base: QColor) -> QColor:
    """Background wash marking a passage as belonging to one voice.

    Mixed into the editor's own background rather than applied as the hue itself:
    the palette is built to be read as a *fill with ink on top*, and several entries
    are far too dark to serve as text colour — violet measures 1.95:1 against a dark
    editor, which is unreadable. Tinting the background instead keeps the exact hue
    identity of the chip while the prose keeps the contrast it had.

    Blended to an opaque colour rather than left as an alpha brush so the result does
    not depend on how the widget happens to composite translucent backgrounds.
    """
    hue = voice_colour(name, library)
    k = SPAN_TINT_STRENGTH
    return QColor(
        round(base.red() * (1 - k) + hue.red() * k),
        round(base.green() * (1 - k) + hue.green() * k),
        round(base.blue() * (1 - k) + hue.blue() * k),
    )


# Dia's documented non-verbal tags. The model reads these as performance directions
# rather than speaking them. Only these exact spellings are recognised — upstream
# warns that unlisted ones "may cause weird artifacts", which is why this is a fixed
# list of buttons instead of a free-text field.
#
# Defined in chunking, not here: the pipeline has to subtract them when estimating how
# long a chunk should take, so the list belongs with the text handling rather than in
# the GUI. A second copy here would drift, and the drift would be invisible until the
# coverage check started firing on whichever tag the two lists disagreed about.
NONVERBAL_TAGS = chunking.NONVERBAL_TAGS
_NONVERBAL_RE = chunking.NONVERBAL_RE

# Flat white so a performance direction never reads as a speaker. Voice markers own
# the colours; these stay deliberately colourless.
_NONVERBAL_BG = "#ffffff"
_NONVERBAL_INK = "#000000"

# A number the model will read wrong. Amber, the same warning colour the log panel
# uses, and deliberately not the palette's red: red means "this voice does not exist",
# which is broken, and this is not broken — the script just needs a word instead of a
# figure. Measured on this model: 1000000 comes back "một ngàn", 1500000000 comes back
# "một một năm ngàn". It has no scale words at all.
_HARD_NUMBER_BG = "#4a3a12"
_HARD_NUMBER_INK = "#ffcc55"


class MarkerHighlighter(QSyntaxHighlighter):
    """Paints [Voice] markers in that voice's colour, flags unknown ones, and marks
    non-verbal tags.

    A marker naming a voice that no longer exists silently falls back to the default at
    generation time, so it is worth making visible while editing rather than discovering
    it in the audio.
    """

    def __init__(self, document, known_names: list[str], start_voice: str | None = None,
                 base_colour: QColor | None = None):
        super().__init__(document)
        # Order matters, not just membership: it decides which palette slot each
        # voice gets, so this keeps the list rather than a set.
        self.known = list(known_names)
        self.start_voice = start_voice
        self.base_colour = base_colour or QColor("#1e1e1e")

    def set_known(self, names: list[str]) -> None:
        self.known = list(names)
        self.rehighlight()

    def set_start_voice(self, name: str | None) -> None:
        """The starting voice's passages are left untinted, so which voice that is
        changes what gets painted."""
        if name == self.start_voice:
            return
        self.start_voice = name
        self.rehighlight()

    def _tint_state(self, name: str | None) -> int:
        """Block state encoding the voice in force at the end of a block.

        -1 means "no tint": either nothing has overridden the starting voice yet, or
        a marker named the starting voice and handed narration back to it. Qt's own
        default for an unset block state is also -1, which makes the first block fall
        out correctly without a special case.
        """
        if name is None or name == self.start_voice or name not in self.known:
            return -1
        return self.known.index(name)

    def _voice_for_state(self, state: int) -> str | None:
        if state < 0 or state >= len(self.known):
            return None
        return self.known[state]

    def highlightBlock(self, text: str) -> None:
        # Voice carries across paragraphs — a marker in the first line governs
        # everything until the next one, however many blocks later that is.
        state = self.previousBlockState()
        if state < -1:
            state = -1

        def tint(start: int, end: int, current: int) -> None:
            name = self._voice_for_state(current)
            if name is None or end <= start:
                return
            fmt = QTextCharFormat()
            fmt.setBackground(voice_span_tint(name, self.known, self.base_colour))
            self.setFormat(start, end - start, fmt)

        position = 0
        for match in chunking.VOICE_MARKER.finditer(text):
            tint(position, match.start(), state)
            name = match.group(1)
            fmt = QTextCharFormat()
            fmt.setBackground(voice_colour(name, self.known))
            fmt.setForeground(voice_ink(name, self.known))
            if name not in self.known:
                fmt.setToolTip(f"No voice named {name}")
                # Not colour alone: see the note on _UNKNOWN_COLOUR.
                fmt.setUnderlineStyle(QTextCharFormat.UnderlineStyle.WaveUnderline)
                fmt.setUnderlineColor(QColor("#ffffff"))
            fmt.setFontWeight(QFont.Weight.Bold)
            self.setFormat(match.start(), match.end() - match.start(), fmt)
            state = self._tint_state(name)
            position = match.end()
        tint(position, len(text), state)
        self.setCurrentBlockState(state)

        for match in _NONVERBAL_RE.finditer(text):
            fmt = QTextCharFormat()
            fmt.setBackground(QColor(_NONVERBAL_BG))
            fmt.setForeground(QColor(_NONVERBAL_INK))
            fmt.setFontItalic(True)
            fmt.setToolTip("Non-verbal direction — performed, not spoken.")
            self.setFormat(match.start(), match.end() - match.start(), fmt)

        for match in vn_numbers.NUMBER_RE.finditer(text):
            if not vn_numbers.is_hard_to_read(match.group()):
                continue
            fmt = QTextCharFormat()
            fmt.setBackground(QColor(_HARD_NUMBER_BG))
            fmt.setForeground(QColor(_HARD_NUMBER_INK))
            # Underlined as well as coloured, for the same reason an unknown marker
            # is: this has to survive being read by someone who cannot separate the
            # two warm colours in this palette.
            fmt.setUnderlineStyle(QTextCharFormat.UnderlineStyle.DotLine)
            fmt.setUnderlineColor(QColor(_HARD_NUMBER_INK))
            fmt.setToolTip(
                "The model reads numbers this long wrong — 1000000 comes out "
                "\"một ngàn\". Write it in words and it will say exactly that."
            )
            self.setFormat(match.start(), match.end() - match.start(), fmt)


class VoiceChip(QLabel):
    """A draggable swatch. Dragging carries the marker text, so the editor's ordinary
    text-drop handling inserts it wherever it lands — no custom drop logic needed."""

    def __init__(self, name: str, library=(), parent=None):
        super().__init__(voice.display_name(name), parent)
        self.name = name
        colour = voice_colour(name, library)
        ink = voice_ink(name, library)
        self.setStyleSheet(
            f"background: {colour.name()}; color: {ink.name()}; font-weight: 600;"
            "padding: 4px 10px; border-radius: 9px;"
        )
        self.setCursor(Qt.CursorShape.OpenHandCursor)
        self.setToolTip(f"Insert [{name}] into the script.\n"
                        f"Everything after it is read by {voice.display_name(name)}, "
                        "until the next marker.")

    def mousePressEvent(self, event) -> None:
        if event.button() != Qt.MouseButton.LeftButton:
            return
        mime = QMimeData()
        mime.setText(f"[{self.name}] ")
        drag = QDrag(self)
        drag.setMimeData(mime)
        drag.setPixmap(self.grab())
        drag.exec(Qt.DropAction.CopyAction)


MIN_FONT_PT = 7
MAX_FONT_PT = 36
DEFAULT_FONT_PT = 11


class ScriptEditor(QPlainTextEdit):
    font_size_changed = Signal(int)

    def __init__(self, parent=None, font_pt: int = DEFAULT_FONT_PT):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.set_font_size(font_pt)

    def set_font_size(self, points: int) -> None:
        points = max(MIN_FONT_PT, min(MAX_FONT_PT, int(points)))
        if points == self.font().pointSize():
            return
        font = QFont(self.font())
        font.setPointSize(points)
        self.setFont(font)
        self.font_size_changed.emit(points)

    def zoom_by(self, steps: int) -> None:
        self.set_font_size(self.font().pointSize() + steps)

    def wheelEvent(self, event) -> None:
        """Ctrl+wheel zooms, matching what every other text surface does.

        QPlainTextEdit's built-in zoomIn/zoomOut are deliberately not used: they
        adjust the widget's zoom independently of its font, so the size cannot be
        read back to persist it.
        """
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            self.zoom_by(1 if event.angleDelta().y() > 0 else -1)
            event.accept()
            return
        super().wheelEvent(event)

    def insert_marker(self, name: str) -> None:
        self.insertPlainText(f"[{name}] ")
        self.setFocus()

    def insert_tag(self, tag: str) -> None:
        self.insertPlainText(f"{tag} ")
        self.setFocus()


def sentence_at(text: str, cursor_pos: int, start_voice: str | None = None):
    """Which sentence the cursor sits in, 1-based, and the voice reading it.

    The number has to be the same one the export log prints ("Sentence 75 of
    'Chương 200'..."), because looking a reported failure back up in the script is
    the whole reason to show it. So rather than counting sentences a second way,
    this runs the pipeline's own chunker over the text up to the cursor.

    The appended "x" is what makes boundaries come out right: it guarantees an
    in-progress sentence exists at the cursor, so a cursor resting at the very start
    of sentence N counts N rather than the N-1 complete sentences behind it.
    """
    chunks = chunking.chunk_text_with_voices(text[:cursor_pos] + "x", start_voice=start_voice)
    if not chunks:
        return 1, start_voice
    return len(chunks), chunks[-1].voice


class NonverbalButton(QPushButton):
    """One of Dia's non-verbal directions. Flat white on purpose — these are stage
    directions, not speakers, and must not read as another voice in the cast."""

    def __init__(self, tag: str, parent=None):
        super().__init__(tag, parent)
        self.tag = tag
        self.setStyleSheet(
            f"QPushButton {{ background: {_NONVERBAL_BG}; color: {_NONVERBAL_INK};"
            "border: 1px solid #b8b8b8; border-radius: 9px; padding: 3px 8px; }"
            "QPushButton:hover { border-color: #6d6d6d; }"
        )
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setToolTip(f"Insert {tag} at the cursor — performed, not read aloud.")


class ScriptEditorDialog(QDialog):
    """Edit one chapter's text, then export that text rather than the book's."""

    export_requested = Signal()
    # Saving keeps the dialog open, so the edited text has to travel out while it is
    # still up rather than being read off a closed dialog.
    script_saved = Signal(str)

    def __init__(self, settings: config.Settings, title: str, text: str, parent=None,
                 on_reset=None):
        super().__init__(parent)
        self.setWindowTitle(f"Script — {title}")
        # A QDialog gets only a close button from Windows. This is not a transient
        # prompt — it is where a whole chapter is cast and edited, sat with for a long
        # time beside other windows — so it earns the minimise and maximise a normal
        # window has.
        self.setWindowFlags(
            self.windowFlags()
            | Qt.WindowType.WindowMinimizeButtonHint
            | Qt.WindowType.WindowMaximizeButtonHint
        )
        self.resize(900, 640)
        self.settings = settings
        self.chapter_title = title
        # Callable returning the chapter's text as the book itself has it, or None if
        # the caller cannot produce it (then Reset is not offered rather than lying).
        self._on_reset = on_reset
        self._saved_text = text

        # Library order, not alphabetical: it is what assigns each voice its palette
        # slot, so the chips here show exactly the colours the markers will take.
        names = voice.voice_names_in_library_order(settings)

        self.editor = ScriptEditor(
            font_pt=int(getattr(settings, "script_font_pt", DEFAULT_FONT_PT))
        )
        self.editor.setPlainText(text)
        self.editor.font_size_changed.connect(self._remember_font_size)
        self.highlighter = MarkerHighlighter(
            self.editor.document(), names,
            base_colour=self.editor.palette().color(QPalette.ColorRole.Base),
        )

        if settings.tts_engine == "vieneu":
            # The bundled roster has 25 voices. A row of 25 chips would consume most
            # of the editor, so keep the whole list in a compact picker.
            self.voice_palette = QWidget()
            picker_row = QHBoxLayout(self.voice_palette)
            picker_row.setContentsMargins(0, 0, 0, 0)
            picker_row.addWidget(QLabel("Insert voice:"))
            self.voice_picker = QComboBox()
            self.voice_picker.setMaxVisibleItems(12)
            for name in names:
                self.voice_picker.addItem(voice.display_name(name), name)
            default = voice.resolve_name(settings, None)
            if default in names:
                self.voice_picker.setCurrentIndex(names.index(default))
            self.voice_picker.setEnabled(bool(names))
            picker_row.addWidget(self.voice_picker, stretch=1)
            self.insert_voice_button = QPushButton("Insert marker")
            self.insert_voice_button.setEnabled(bool(names))
            self.insert_voice_button.clicked.connect(
                lambda: self.editor.insert_marker(self.voice_picker.currentData())
            )
            picker_row.addWidget(self.insert_voice_button)
        else:
            # Dia libraries are small enough for draggable chips. They wrap at the
            # available width so the dialog still fits narrow screens.
            palette_items = [QLabel("Voices:")]
            if names:
                for name in names:
                    chip = VoiceChip(name, library=names)
                    chip.mouseDoubleClickEvent = (
                        lambda _e, n=name: self.editor.insert_marker(n)
                    )
                    palette_items.append(chip)
            else:
                hint = QLabel("(no voices configured — add some under Settings → Voices)")
                hint.setStyleSheet("color: gray;")
                palette_items.append(hint)
            self.voice_palette = wrapping_row(palette_items)

        # Was a QGridLayout at a fixed seven columns, which wraps but only ever at that
        # one width: too wide on a narrow window, and leaving empty space on a wide
        # one. The flow layout wraps at whatever width the box actually has.
        self.nonverbal_buttons = []
        for tag in NONVERBAL_TAGS:
            button = NonverbalButton(tag)
            button.clicked.connect(lambda _checked=False, t=tag: self.editor.insert_tag(t))
            self.nonverbal_buttons.append(button)

        nonverbal_box = QGroupBox("Non-verbal")
        nonverbal_layout = QVBoxLayout(nonverbal_box)
        nonverbal_layout.setContentsMargins(6, 6, 6, 6)
        nonverbal_layout.addWidget(wrapping_row(self.nonverbal_buttons, spacing=4))

        self.start_voice_combo = QComboBox()
        if names:
            for name in names:
                self.start_voice_combo.addItem(voice.display_name(name), name)
            default = voice.resolve_name(settings, None)
            if default in names:
                self.start_voice_combo.setCurrentIndex(names.index(default))
        else:
            self.start_voice_combo.addItem("(model default)")
            self.start_voice_combo.setEnabled(False)

        # Word wrap on every label in this row, and it is not cosmetic. A QLabel with
        # wrapping off reports its whole text as its *minimum* width, and QHBoxLayout
        # adds those minimums up — this one row demanded 1548px, which became the
        # dialog's minimum width and put its edges off the side of the screen with no
        # way to drag them back. Wrapping lets each label collapse instead.
        start_label = QLabel("Voice before the first marker:")
        self.start_voice_combo.setToolTip(
            "Who reads the text ahead of the first marker. Rendering writes this into "
            "the script as a marker of its own when it is not the default voice, so "
            "the casting lives in the script rather than being remembered separately."
        )
        start_label.setWordWrap(True)

        start_row = QHBoxLayout()
        start_row.addWidget(start_label)
        start_row.addWidget(self.start_voice_combo)
        start_row.addStretch(1)
        self.dirty_placeholder = None
        # Numbered the same way the export log numbers its failures, so a reported
        # "Sentence 75" can be found by moving the cursor until this matches.
        #
        # Deliberately not greyed out: this is the live readout, the one thing in
        # this row that answers a question you are asking right now, so it carries
        # the row's emphasis while the static summary beside it stays plain.
        self.position_label = QLabel()
        self.position_label.setStyleSheet("font-weight: 600;")
        self.position_label.setWordWrap(True)
        start_row.addWidget(self.position_label)
        self.summary = QLabel()
        self.summary.setWordWrap(True)
        start_row.addWidget(self.summary)

        voice_hint = (
            "Choose a voice above and insert its marker at the cursor. "
            if settings.tts_engine == "vieneu" else
            "Drag a chip into the text, or double-click it to insert at the cursor. "
        )
        performance_hint = (
            "Dia's non-verbal directions are skipped by VieNeu."
            if settings.tts_engine == "vieneu" else
            "Non-verbal tags are performed rather than read out. They come from base "
            "Dia, whose own docs call them unpredictable and say to use them sparingly "
            "— and this Vietnamese fine-tune was not trained on them, so check a short "
            "export before relying on one."
        )
        hint = QLabel(
            voice_hint +
            "A marker applies from where it sits until the next one. Markers are never "
            "spoken. A marker shown in dark red names a voice that no longer exists.\n"
            + performance_hint
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color: gray;")

        self._action = "cancel"

        zoom_out = QPushButton("A−")
        zoom_in = QPushButton("A+")
        for b, step, tip in ((zoom_out, -1, "Smaller text"), (zoom_in, 1, "Larger text")):
            b.setFixedWidth(38)
            b.setToolTip(f"{tip} (Ctrl+mouse wheel over the text)")
            b.clicked.connect(lambda _c=False, s=step: self.editor.zoom_by(s))
        self.zoom_out_button, self.zoom_in_button = zoom_out, zoom_in

        self.reset_button = QPushButton("Reset to original")
        self.reset_button.setToolTip(
            "Re-read this chapter from the book file, discarding every edit and marker "
            "saved for it."
        )
        self.reset_button.clicked.connect(self._reset_to_original)
        self.reset_button.setEnabled(on_reset is not None)

        self.dirty_label = QLabel()
        self.dirty_label.setStyleSheet("color: gray;")

        self.save_button = QPushButton("Save tags")
        self.export_button = QPushButton("Render this script")
        self.close_button = QPushButton("Close")
        # Laid out by hand rather than with QDialogButtonBox: its role-based ordering
        # put Close between Export and Save, so the two things you actually do with a
        # finished script sat either side of the way out.
        self.save_button.setToolTip(
            "Save the text and voice markers as they stand and keep editing — the "
            "window stays open. Next time you export this book, these tags are already "
            "in place."
        )
        self.export_button.setToolTip(
            "Generate every sentence of this chapter into the cache. No file is "
            "written here — the audio is kept per sentence so a bad one can be "
            "redone on its own, and “Export audio file” builds the MP3 from "
            "whatever is in the cache when you ask for it."
        )
        self.save_button.clicked.connect(self._save_without_closing)
        self.export_button.clicked.connect(self._accept_as("export"))
        self.close_button.clicked.connect(self._close_requested)

        button_row = QHBoxLayout()
        button_row.addWidget(self.dirty_label)
        button_row.addWidget(zoom_out)
        button_row.addWidget(zoom_in)
        button_row.addWidget(self.reset_button)
        button_row.addStretch(1)
        button_row.addWidget(self.save_button)
        button_row.addWidget(self.export_button)
        button_row.addWidget(self.close_button)

        layout = QVBoxLayout(self)
        layout.addWidget(self.voice_palette)
        layout.addWidget(self.editor, stretch=1)
        if settings.tts_engine != "vieneu":
            layout.addWidget(nonverbal_box)
        layout.addLayout(start_row)
        layout.addWidget(hint)
        layout.addLayout(button_row)

        self._fit_to_screen()

        self._install_signal_handlers()

    def _fit_to_screen(self) -> None:
        """Never open larger than the screen it opens on.

        resize() asks for 900x640, but a dialog is only ever as small as its layout's
        minimum allows — and the two bugs behind this method (a non-wrapping chip row,
        then labels reporting their whole text as a minimum) both pushed that minimum
        past what a laptop can show, leaving the lower half of the window unreachable.
        Both are fixed, so this is the guard rather than the cure: it keeps a future
        one from being invisible, and it is what makes the maximise button meaningful
        on a screen smaller than the design size.
        """
        screen = self.screen() or QGuiApplication.primaryScreen()
        if screen is None:
            return
        available = screen.availableGeometry()
        # Room for the title bar and whatever the desktop reserves around a window.
        margin = 80
        width = min(self.width(), available.width() - margin)
        height = min(self.height(), available.height() - margin)
        self.resize(max(width, 1), max(height, 1))

    def _install_signal_handlers(self) -> None:
        self.editor.textChanged.connect(self._update_summary)
        self.editor.textChanged.connect(self._update_position)
        self.editor.textChanged.connect(self._update_dirty)
        self.editor.cursorPositionChanged.connect(self._update_position)
        self.start_voice_combo.currentTextChanged.connect(self._update_summary)
        self.start_voice_combo.currentTextChanged.connect(self._update_position)
        self.start_voice_combo.currentTextChanged.connect(
            lambda: self.highlighter.set_start_voice(self.start_voice())
        )
        self.highlighter.set_start_voice(self.start_voice())
        self._update_summary()
        self._update_position()
        self._update_dirty()

    # --- saving, resetting, zoom -----------------------------------------------

    def _remember_font_size(self, points: int) -> None:
        """Persist the zoom, so it does not reset every time the editor is opened."""
        if getattr(self.settings, "script_font_pt", None) == points:
            return
        self.settings.script_font_pt = points
        try:
            self.settings.save()
        except OSError:  # a read-only settings file must not break editing
            log.warning("Could not persist the script editor font size.", exc_info=True)

    def _update_dirty(self) -> None:
        dirty = self.editor.toPlainText() != self._saved_text
        self.dirty_label.setText("Unsaved changes" if dirty else "Saved")
        self.save_button.setEnabled(dirty)

    def _save_without_closing(self) -> None:
        """Save and stay open — tagging a chapter is iterative, and a save that also
        closed the window meant reopening it to place the next marker."""
        text = self.editor.toPlainText()
        self._saved_text = text
        self._action = "save"
        self.script_saved.emit(text)
        self._update_dirty()

    def _reset_to_original(self) -> None:
        if self._on_reset is None:
            return
        answer = QMessageBox.question(
            self, "Reset to original",
            f"Re-read “{self.chapter_title}” from the book file?\n\n"
            "Every edit and voice marker saved for this chapter is discarded. "
            "This cannot be undone.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        try:
            original = self._on_reset()
        except Exception as e:  # noqa: BLE001 — surface it instead of silently doing nothing
            QMessageBox.critical(self, "Could not reset", str(e))
            return
        if original is None:
            QMessageBox.information(
                self, "Nothing to reset to",
                "The original text for this chapter could not be read from the book.",
            )
            return
        self.editor.setPlainText(original)
        # A reset is itself a change to be saved; leaving it "Saved" would imply the
        # stored copy had been reverted too, which it has not.
        self._update_dirty()

    def _close_requested(self) -> None:
        if self.editor.toPlainText() == self._saved_text:
            self.reject()
            return
        answer = QMessageBox.question(
            self, "Unsaved changes",
            "This script has changes that have not been saved. Close anyway?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer == QMessageBox.StandardButton.Yes:
            self.reject()

    def _update_summary(self) -> None:
        text = self.editor.toPlainText()
        sentences = chunking.chunk_text_with_voices(text, start_voice=self.start_voice())
        used = {c.voice for c in sentences if c.voice}
        minutes = len(sentences) * 40 / 60
        self._total_sentences = len(sentences)
        self.summary.setText(
            f"{len(sentences)} sentences, {len(used) or 1} voice(s) — "
            f"roughly {minutes:.0f} min to generate"
        )

    def _update_position(self) -> None:
        text = self.editor.toPlainText()
        number, voice_name = sentence_at(
            text, self.editor.textCursor().position(), start_voice=self.start_voice()
        )
        total = getattr(self, "_total_sentences", None)
        if total is None:
            total = len(chunking.chunk_text_with_voices(text, start_voice=self.start_voice()))
        # An empty editor has no sentence 1 to be in.
        if total == 0:
            self.position_label.setText("")
            return
        where = f"Sentence {min(number, total)} of {total}"
        self.position_label.setText(f"{where} · {voice_name}  " if voice_name else f"{where}  ")

    def sentence_position(self) -> tuple[int, str | None]:
        """Cursor's sentence number and active voice — exposed for tests."""
        return sentence_at(
            self.editor.toPlainText(), self.editor.textCursor().position(),
            start_voice=self.start_voice(),
        )

    def _accept_as(self, action: str):
        def handler():
            self._action = action
            self.accept()
        return handler

    def action(self) -> str:
        """Which button closed the dialog: 'save', 'export', or 'cancel'."""
        return self._action

    def start_voice(self) -> str | None:
        return self.start_voice_combo.currentData() if self.start_voice_combo.isEnabled() else None

    def script_text(self) -> str:
        return self.editor.toPlainText()
