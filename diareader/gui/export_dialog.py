"""Export dialog: chapter range, starting voice, and destination file."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PySide6.QtWidgets import (
    QComboBox, QDialog, QDialogButtonBox, QFileDialog, QFormLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QSpinBox, QVBoxLayout, QWidget
)

from .. import config, voice
from .widgets import SPEED_CHOICES, speed_label


@dataclass
class ExportRequest:
    out_path: str
    chapter_indices: list[int]
    start_voice: str | None
    speed: float = 1.0


class ExportDialog(QDialog):
    """Picks what to export before any GPU work starts.

    The range defaults to whatever is selected in the chapter list, because rendering a
    whole novel is rarely the intent — one real book here was 255k sentences.
    """

    # Rough throughput measured on this project's GPU; only used to warn about scale.
    _SECONDS_PER_SENTENCE = 40

    def __init__(
        self,
        settings: config.Settings,
        chapter_titles: list[str],
        chunks_per_chapter: list[int],
        preselected: list[int] | None = None,
        default_path: str = "",
        parent=None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Export audiobook")
        self.setMinimumWidth(560)

        self.settings = settings
        self.chapter_titles = chapter_titles
        self.chunks_per_chapter = chunks_per_chapter

        n = max(len(chapter_titles), 1)
        first = (preselected[0] + 1) if preselected else 1
        last = (preselected[-1] + 1) if preselected else n

        self.from_spin = QSpinBox()
        self.from_spin.setRange(1, n)
        self.from_spin.setValue(min(first, n))
        self.to_spin = QSpinBox()
        self.to_spin.setRange(1, n)
        self.to_spin.setValue(min(last, n))
        # Keep the two ends from crossing. Silently sorting an inverted pair looks
        # harmless but quietly renders chapters that were never asked for: with "to"
        # still at 1 and "from" set to 2, a reversed range became "1 to 2" and pulled a
        # 2467-entry table of contents into the export.
        self.from_spin.valueChanged.connect(self._on_from_changed)
        self.to_spin.valueChanged.connect(self._on_to_changed)

        # Show the chapter each end resolves to. Picking by number alone is easy to get
        # wrong — chapter 1 of this kind of file is often a table of contents, not prose.
        self.from_title = QLabel()
        self.to_title = QLabel()
        for label in (self.from_title, self.to_title):
            label.setWordWrap(True)
            label.setStyleSheet("font-weight: 600;")

        range_row = QWidget()
        range_layout = QVBoxLayout(range_row)
        range_layout.setContentsMargins(0, 0, 0, 0)
        range_layout.setSpacing(4)

        from_row = QHBoxLayout()
        from_row.addWidget(QLabel("from"))
        from_row.addWidget(self.from_spin)
        from_row.addWidget(self.from_title, stretch=1)

        to_row = QHBoxLayout()
        to_row.addWidget(QLabel("to  "))
        to_row.addWidget(self.to_spin)
        to_row.addWidget(self.to_title, stretch=1)

        range_layout.addLayout(from_row)
        range_layout.addLayout(to_row)

        self.voice_combo = QComboBox()
        names = voice.voice_names(settings)
        if names:
            self.voice_combo.addItems(names)
            default = settings.default_voice
            if default in names:
                self.voice_combo.setCurrentText(default)
        else:
            self.voice_combo.addItem("(no voices configured — model default)")
            self.voice_combo.setEnabled(False)

        self.speed_combo = QComboBox()
        for value in SPEED_CHOICES:
            self.speed_combo.addItem(speed_label(value), value)
        current = getattr(settings, "speed", 1.0) or 1.0
        index = self.speed_combo.findData(current)
        self.speed_combo.setCurrentIndex(index if index >= 0 else SPEED_CHOICES.index(1.0))

        self.path_edit = QLineEdit(default_path)
        browse = QPushButton("Browse…")
        browse.clicked.connect(self._browse)
        path_row = QWidget()
        path_layout = QHBoxLayout(path_row)
        path_layout.setContentsMargins(0, 0, 0, 0)
        path_layout.addWidget(self.path_edit)
        path_layout.addWidget(browse)

        self.summary = QLabel()
        self.summary.setWordWrap(True)

        form = QFormLayout()
        form.addRow("Chapters", range_row)
        form.addRow("Starting voice", self.voice_combo)
        form.addRow("Reading speed", self.speed_combo)
        form.addRow("Save to", path_row)
        form.addRow("", self.summary)

        hint = QLabel(
            "The starting voice applies until the text switches it with a marker such as "
            "[GiongNu2]. Markers are read as instructions, never spoken."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color: gray;")

        self.buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(hint)
        layout.addWidget(self.buttons)

        self._on_range_changed()

    def _on_from_changed(self) -> None:
        if self.to_spin.value() < self.from_spin.value():
            self.to_spin.setValue(self.from_spin.value())
        self._on_range_changed()

    def _on_to_changed(self) -> None:
        if self.from_spin.value() > self.to_spin.value():
            self.from_spin.setValue(self.to_spin.value())
        self._on_range_changed()

    def _selected_indices(self) -> list[int]:
        # The spinboxes push each other rather than crossing, so no sorting is needed —
        # and none is done, deliberately: reordering an inverted pair is what silently
        # widened the range downwards before.
        start = self.from_spin.value()
        end = max(self.to_spin.value(), start)
        return list(range(start - 1, end))

    def _title_for(self, index: int) -> str:
        if 0 <= index < len(self.chapter_titles):
            title = self.chapter_titles[index]
            return title if len(title) <= 70 else title[:67] + "…"
        return ""

    def _on_range_changed(self) -> None:
        indices = self._selected_indices()
        self.from_title.setText(self._title_for(self.from_spin.value() - 1))
        self.to_title.setText(self._title_for(self.to_spin.value() - 1))
        sentences = sum(
            self.chunks_per_chapter[i] for i in indices if i < len(self.chunks_per_chapter)
        )
        hours = sentences * self._SECONDS_PER_SENTENCE / 3600
        estimate = f"{hours:.1f} hours" if hours >= 1 else f"{hours * 60:.0f} minutes"
        self.summary.setText(
            f"{len(indices)} chapter(s), {sentences} sentences — roughly {estimate} of "
            f"generation at ~{self._SECONDS_PER_SENTENCE}s per sentence."
        )

    def _browse(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Export audiobook", self.path_edit.text() or "", "MP3 (*.mp3);;WAV (*.wav)"
        )
        if path:
            self.path_edit.setText(path)

    def request(self) -> ExportRequest | None:
        out_path = self.path_edit.text().strip()
        if not out_path:
            return None
        return ExportRequest(
            out_path=out_path,
            chapter_indices=self._selected_indices(),
            start_voice=self.voice_combo.currentText() if self.voice_combo.isEnabled() else None,
            speed=float(self.speed_combo.currentData()),
        )
