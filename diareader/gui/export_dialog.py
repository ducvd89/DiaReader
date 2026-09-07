"""Save final file: where to write it, and which of the rendered chapters go in.

Nothing is generated here. By the time this opens, the audio already exists — it was
made sentence by sentence from the script editor and lives in the cache — so the only
questions left are the destination and the selection.

Voice and speed used to be asked for here too, and both were wrong to ask. The voice a
sentence is read in is decided in the script, by its markers, and offering a different
one at save time could only contradict what was already rendered. Speed is baked into
each take when it is generated, so changing it here would not re-pace the book: it
would invalidate every take and quietly re-render the lot.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView, QDialog, QDialogButtonBox, QFileDialog, QFormLayout, QHBoxLayout,
    QLabel, QLineEdit, QListWidget, QListWidgetItem, QPushButton, QVBoxLayout, QWidget
)

from .. import config


@dataclass
class ExportRequest:
    out_path: str
    chapter_indices: list[int] = field(default_factory=list)
    # Both resolved from the script and the settings now; kept so the render paths
    # that still supply them keep working.
    start_voice: str | None = None
    speed: float | None = None


class ExportDialog(QDialog):
    """Pick the destination and the chapters, from what has actually been rendered."""

    def __init__(
        self,
        settings: config.Settings,
        chapter_titles: list[str],
        chunks_per_chapter: list[int],
        cached_per_chapter: dict[int, int] | None = None,
        preselected: list[int] | None = None,
        default_path: str = "",
        parent=None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Save final file")
        self.setMinimumWidth(620)

        self.settings = settings
        self.chapter_titles = chapter_titles
        self.chunks_per_chapter = chunks_per_chapter
        self.cached_per_chapter = dict(cached_per_chapter or {})

        self.chapter_list = QListWidget()
        self.chapter_list.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.chapter_list.setMinimumHeight(180)
        # Only chapters with audio. A chapter that has never been rendered has nothing
        # to save, and listing it would offer to write a file of silence.
        preselected = set(preselected or [])
        for index, title in enumerate(chapter_titles):
            rendered = self.cached_per_chapter.get(index, 0)
            if not rendered:
                continue
            total = chunks_per_chapter[index] if index < len(chunks_per_chapter) else rendered
            label = f"{title}   —   {rendered} of {total} sentences rendered"
            item = QListWidgetItem(label)
            item.setData(Qt.ItemDataRole.UserRole, index)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(
                Qt.CheckState.Checked
                if (not preselected or index in preselected) else Qt.CheckState.Unchecked
            )
            if rendered < total:
                item.setToolTip(
                    f"{total - rendered} sentence(s) have no audio yet and would be "
                    "left as short gaps. Render them from the script editor, or in "
                    "the Sentences window."
                )
            self.chapter_list.addItem(item)
        self.chapter_list.itemChanged.connect(lambda _item: self._refresh_summary())

        self.path_edit = QLineEdit(default_path)
        self.path_edit.textChanged.connect(lambda _t: self._refresh_summary())
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
        form.addRow("Chapters", self.chapter_list)
        form.addRow("Save to", path_row)
        form.addRow("", self.summary)

        self.buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel
        )
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        if not self.chapter_list.count():
            empty = QLabel(
                "No chapter has been rendered yet, so there is nothing to save.\n\n"
                "Open a chapter in the script editor and press “Render this script”. "
                "The sentences are generated into the cache, and this is where they "
                "become a file."
            )
            empty.setWordWrap(True)
            layout.addWidget(empty)
        layout.addLayout(form)
        layout.addWidget(self.buttons)
        self._refresh_summary()

    # -- selection ---------------------------------------------------------------------

    def _items(self):
        return [self.chapter_list.item(row) for row in range(self.chapter_list.count())]

    def _selected_indices(self) -> list[int]:
        return [
            item.data(Qt.ItemDataRole.UserRole)
            for item in self._items()
            if item.checkState() == Qt.CheckState.Checked
        ]

    def _refresh_summary(self) -> None:
        indices = self._selected_indices()
        rendered = sum(self.cached_per_chapter.get(i, 0) for i in indices)
        total = sum(
            self.chunks_per_chapter[i] if i < len(self.chunks_per_chapter)
            else self.cached_per_chapter.get(i, 0)
            for i in indices
        )
        gaps = max(0, total - rendered)
        text = f"{len(indices)} chapter(s), {rendered} sentences ready to write."
        if gaps:
            text += (f"  {gaps} sentence(s) have no audio and will be left as short "
                     "gaps in the file.")
        self.summary.setText(text)
        self.summary.setStyleSheet("color: #b8860b;" if gaps else "color: gray;")
        ok = self.buttons.button(QDialogButtonBox.StandardButton.Save)
        if ok is not None:
            ok.setEnabled(bool(indices) and bool(self.path_edit.text().strip()))

    def _browse(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Save final file", self.path_edit.text() or "", "MP3 (*.mp3);;WAV (*.wav)"
        )
        if path:
            self.path_edit.setText(path)

    def request(self) -> ExportRequest | None:
        out_path = self.path_edit.text().strip()
        indices = self._selected_indices()
        if not out_path or not indices:
            return None
        return ExportRequest(out_path=out_path, chapter_indices=indices)
