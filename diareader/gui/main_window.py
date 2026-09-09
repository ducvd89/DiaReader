"""Main window: library, chapter list, model control, export."""
from __future__ import annotations

import logging
import time
from pathlib import Path
from dataclasses import replace
from types import SimpleNamespace

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication, QFileDialog, QMainWindow, QMessageBox, QProgressDialog, QSplitter,
    QWidget, QVBoxLayout, QLabel
)

from .. import (
    chunking, config, naming, pipeline, sentence_cache, text_extract, vn_numbers, voice
)
from ..model_loader import CudaNotAvailableError, free_vram, require_cuda
from .export_dialog import ExportDialog
from .script_editor import ScriptEditorDialog
from .sentence_review import SentenceReviewDialog
from .settings_dialog import SettingsDialog
from .widgets import ChapterListWidget, LogBridge, PlayerControls, QtLogHandler
from .workers import AssembleWorker, ModelLoadWorker, RenderWorker

log = logging.getLogger(__name__)


class _ParseCancelled(Exception):
    """Raised from the parse progress callback to abort opening a large book."""


def eta_suffix(done: int, total: int, elapsed: float) -> str:
    """Live ETA from measured throughput, as a status-line suffix.

    Pure so it can be tested against the case that produced a *negative* time on a
    real run: `total` used to be shared with the chapter-list selection, so clicking
    the list mid-export moved it below the number of sentences already done.
    """
    if done < 2 or total <= 0 or elapsed <= 0:
        return ""
    per_chunk = elapsed / done
    # Clamped rather than trusted. The total is counted up front with the same
    # chunker the pipeline uses, so the two agree — but an estimate that has run past
    # its own total should read as nearly done, never as negative minutes left.
    remaining = max(0.0, (total - done) * per_chunk)
    if remaining < 90:
        return f" — about {remaining:.0f}s left"
    if remaining < 5400:
        return f" — about {remaining / 60:.0f} min left"
    return f" — about {remaining / 3600:.1f} h left"


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("DiaReader")
        self.resize(1000, 640)

        self.settings = config.Settings.load()
        config.ensure_dirs()

        self.engine = None
        self.current_book_path: str | None = None
        # Parsed once on open and reused everywhere. Re-parsing a 2467-chapter epub
        # costs seconds, and this used to happen on every click in the chapter list.
        self.chapters: list[text_extract.Chapter] = []
        self.chunks_per_chapter: list[int] = []
        # Two separate totals, and they must stay separate. `total_chunks` tracks
        # whatever is selected right now, and is rewritten every time the selection
        # changes; `run_total_chunks` belongs to the export that is actually running.
        # They were one attribute, and because the chapter list stays usable during an
        # export — deliberately, so the next chapter can be tagged while this one
        # renders — clicking it mid-run retargeted the progress bar at the new
        # selection. The bar then read against the wrong maximum, and once
        # chunks_done passed it the ETA went negative.
        self.total_chunks = 0
        self.run_total_chunks = 0
        self.chunks_done = 0
        # Rendering and assembling are two stages of one export, so the second has to
        # be remembered while the first runs.
        self._cache = None
        self._pending_assembly = None
        self._after_render = None
        self.render_worker = None
        self.assemble_worker = None
        self.review_dialog = None
        # Which chapter the open review table is showing, so that asking for a
        # different one replaces it rather than raising the wrong window.
        self._review_chapter: int | None = None
        self.editor_dialog = None
        self.run_started_at: float | None = None

        self.load_worker: ModelLoadWorker | None = None

        self.chapter_list = ChapterListWidget()
        self.controls = PlayerControls()

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.addWidget(QLabel("Chapters"))
        left_layout.addWidget(self.chapter_list)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(left)
        splitter.addWidget(self.controls)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([260, 740])
        self.setCentralWidget(splitter)

        self._build_menu()
        self._connect_controls()
        self.chapter_list.itemSelectionChanged.connect(self._update_selection_status)
        self._install_log_bridge()
        self._check_cuda_on_startup()

    def _build_menu(self) -> None:
        # Held on the window rather than left as a local. Reaching the menu again
        # through QAction.menu() hands PySide ownership of the wrapper, and the C++
        # QMenu is then destroyed when that wrapper is collected.
        self.file_menu = file_menu = self.menuBar().addMenu("&File")
        file_menu.addAction("Open Book…", self.open_book)
        self.recent_menu = file_menu.addMenu("Open Recent")
        self.recent_menu.setToolTipsVisible(True)  # off by default in Qt menus
        self._refresh_recent_menu()
        self.reparse_action = file_menu.addAction(
            "Re-read Book from Disk…", self.reparse_book
        )
        self.reparse_action.setToolTip(
            "Discard the cached parse (and every saved tag) and read the book again."
        )
        self.reparse_action.setEnabled(False)
        file_menu.addAction("Settings…", self.open_settings)
        file_menu.addSeparator()
        file_menu.addAction("Quit", self.close)

    def _connect_controls(self) -> None:
        self.controls.load_model_clicked.connect(self.load_model)
        self.controls.unload_model_clicked.connect(self.unload_model)
        self.controls.edit_script_clicked.connect(self.edit_script)
        self.controls.review_sentences_clicked.connect(self.review_sentences)
        self.controls.stop_clicked.connect(self.stop)
        self.controls.export_clicked.connect(self.export)

    def _install_log_bridge(self) -> None:
        """Mirror the pipeline's logging into the UI. The coverage warnings this model
        produces are the only signal that a chunk lost text, so they must be visible."""
        self.log_bridge = LogBridge()
        self.log_bridge.message.connect(self.controls.log_panel.append_record)
        handler = QtLogHandler(self.log_bridge)
        handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
        handler.setLevel(logging.INFO)
        diareader_log = logging.getLogger("diareader")
        diareader_log.addHandler(handler)
        diareader_log.setLevel(logging.INFO)

    def _check_cuda_on_startup(self) -> None:
        try:
            device = require_cuda()
        except CudaNotAvailableError as e:
            QMessageBox.critical(self, "No CUDA GPU found", str(e))
            self.controls.export_button.setEnabled(False)
            self.controls.set_status("No CUDA GPU — export is disabled")
            return
        try:
            import torch
            name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            self.controls.set_status(f"Ready — {name} ({vram:.0f} GB). Open a book to start.")
        except Exception:
            self.controls.set_status(f"Ready on {device}. Open a book to start.")

    # --- book loading -----------------------------------------------------

    def _load_into_view(self, path: str, loader, *, title: str, verb: str) -> bool:
        """Parse a book with a progress dialog and show the result.

        `loader(path, on_progress=...)` is what actually reads it — either the cached
        path or a forced re-parse. Parsing a large epub takes long enough to look like
        a freeze, so this shows real progress; it is CPU-bound and quick per document,
        so a modal bar is fine here rather than another worker thread.
        """
        progress = QProgressDialog(f"{verb} the book…", "Cancel", 0, 100, self)
        progress.setWindowTitle(title)
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setMinimumDuration(0)
        cancelled = False

        def on_parse_progress(done: int, total: int) -> None:
            nonlocal cancelled
            progress.setMaximum(total)
            progress.setValue(done)
            progress.setLabelText(f"{verb} the book… {done}/{total} sections")
            QApplication.processEvents()
            if progress.wasCanceled():
                cancelled = True
                raise _ParseCancelled()

        try:
            chapters = loader(path, on_progress=on_parse_progress)
            progress.setLabelText("Counting sentences…")
            QApplication.processEvents()
            per_chapter = pipeline.count_chunks_per_chapter(path, chapters=chapters)
        except _ParseCancelled:
            self.controls.set_status(f"{title} cancelled")
            return False
        except Exception as e:  # noqa: BLE001
            QMessageBox.critical(self, f"Could not {verb.lower()} book", str(e))
            return False
        finally:
            progress.close()

        if cancelled:
            return False

        self.current_book_path = path
        self.chapters = chapters
        self.chunks_per_chapter = per_chapter
        self.chunks_done = 0
        self.chapter_list.set_chapters([c.title for c in chapters], per_chapter)
        self.chapter_list.setCurrentRow(0)
        self.reparse_action.setEnabled(True)
        self._update_selection_status()
        log.info(
            "%s %s: %d chapters, %d sentences total",
            "Opened" if verb == "Reading" else "Re-read",
            Path(path).name, len(chapters), sum(per_chapter),
        )
        return True

    def _refresh_recent_menu(self) -> None:
        """Rebuild the recent list. Called after every open, so it stays in order."""
        self.recent_menu.clear()
        recent = self.settings.existing_recent_books()
        self.recent_menu.setEnabled(bool(recent))
        for path in recent:
            # Full path in the tooltip: two books can share a filename, and the menu
            # entry alone would not say which one this is.
            action = self.recent_menu.addAction(Path(path).name)
            action.setToolTip(path)
            action.triggered.connect(lambda _checked=False, p=path: self._open_path(p))

    def _open_path(self, path: str) -> None:
        if not Path(path).is_file():
            QMessageBox.information(
                self, "Book not found",
                f"“{path}” is no longer there. It may have been moved or renamed.",
            )
            self._refresh_recent_menu()
            return
        if self._load_into_view(
            path, text_extract.load_book_cached, title="Opening book", verb="Reading"
        ):
            self.settings.remember_book(path)
            self.settings.save()
            self._refresh_recent_menu()

    def open_book(self) -> None:
        # Start where the last book came from. Qt otherwise opens the process's
        # working directory, which for an installed app is the program folder — a
        # place no book has ever been.
        start_dir = self.settings.last_open_dir
        if not Path(start_dir or "").is_dir():
            start_dir = ""
        path, _ = QFileDialog.getOpenFileName(
            self, "Open book", start_dir, "Books (*.txt *.epub)"
        )
        if not path:
            return
        self._open_path(path)

    def reparse_book(self) -> None:
        """Throw away the cached parse and read the book again.

        Worth confirming rather than just doing: the cache is also where script-editor
        edits live, so this is the one action that can destroy tagging work.
        """
        if not self._require_book():
            return
        name = Path(self.current_book_path).name
        answer = QMessageBox.question(
            self, "Re-read book from disk",
            f"Read “{name}” again from disk?\n\n"
            "The cached parse is deleted, and with it every edit and voice marker "
            "saved from the script editor for this book. This cannot be undone.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        self._load_into_view(
            self.current_book_path, text_extract.reparse_book,
            title="Re-reading book", verb="Re-reading",
        )

    def _count_selected(self, indices: list[int] | None) -> int:
        if not indices:
            return sum(self.chunks_per_chapter)
        return sum(
            self.chunks_per_chapter[i] for i in indices if 0 <= i < len(self.chunks_per_chapter)
        )

    def _update_selection_status(self) -> None:
        """Show what the Play/Export buttons would actually act on.

        No time estimate here on purpose: throughput isn't known until generation
        starts, and a made-up number is worse than none — an earlier build claimed
        "~1 min" for work that takes over ten. A live ETA appears once running.
        """
        if not self.current_book_path:
            return
        selected = self.chapter_list.selected_indices()
        # Count from the cached per-chapter figures. Calling pipeline.count_chunks here
        # re-parsed the whole book on every selection change, which froze the UI for
        # seconds at a time on a large epub.
        self.total_chunks = self._count_selected(selected)
        name = Path(self.current_book_path).name
        total_all = sum(self.chunks_per_chapter)
        if selected:
            self.controls.set_status(
                f"{name} — {len(selected)} of {len(self.chunks_per_chapter)} chapters selected, "
                f"{self.total_chunks} sentences (book total {total_all})"
            )
        else:
            self.controls.set_status(
                f"{name} — nothing selected; Play/Export would render the whole book "
                f"({total_all} sentences)"
            )

    def open_settings(self) -> None:
        dialog = SettingsDialog(self.settings, self, model_loaded=self.engine is not None)
        if dialog.exec():
            self.settings = dialog.apply()
            # TF32 is only editable while no model is loaded, so it is applied at load
            # time. Nothing to re-apply here.
            log.info("Settings saved")

    # --- model loading ------------------------------------------------------

    def load_model(self) -> None:
        """Load the model on demand, without needing a book open.

        Loading costs minutes and ~6.5 GB of VRAM. Doing it as its own step means the
        wait (and any failure) happens where you're watching, rather than surfacing
        later as an unexplained delay after pressing Play.
        """
        if self.engine is not None:
            QMessageBox.information(self, "Already loaded", "The model is already in VRAM.")
            return
        self._ensure_engine(lambda: None)

    def unload_model(self) -> None:
        """Drop the model and hand its VRAM back.

        Releasing the reference is not enough on its own: torch's caching allocator
        keeps the memory reserved until it is explicitly emptied, so the GPU still
        looks full afterwards.
        """
        if self.engine is None:
            return
        self.engine = None
        remaining = free_vram()
        self.controls.set_idle(model_loaded=False)
        self.controls.set_status(f"Model unloaded — {remaining:.1f} GB still allocated")
        log.info("Model unloaded; %.1f GB CUDA memory still allocated", remaining)

    def _ensure_engine(self, on_ready) -> None:
        if self.engine is not None:
            on_ready()
            return
        self.controls.set_indeterminate("Loading model…")
        self.controls.set_status(
            "Loading model — the first run downloads ~6.4 GB and can take a while."
        )
        self.load_worker = ModelLoadWorker(self.settings)
        self.load_worker.stage.connect(self._on_model_stage)
        self.load_worker.finished_ok.connect(lambda engine: self._on_engine_loaded(engine, on_ready))
        self.load_worker.failed.connect(self._on_engine_load_failed)
        self.load_worker.start()

    def _on_model_stage(self, message: str) -> None:
        """Name the current loading phase. Loading takes minutes and downloads
        gigabytes; without this a stall is indistinguishable from a hang."""
        self.controls.set_indeterminate(message)
        self.controls.set_status(message)

    def _on_engine_loaded(self, engine, on_ready) -> None:
        self.engine = engine
        self.controls.set_idle(model_loaded=True)
        active = voice.get_voice(self.settings, None)
        self.controls.set_status(
            f"Model loaded — voice: {active.name}" if active
            else "Model loaded — no reference voice (model default)"
        )
        on_ready()

    def _on_engine_load_failed(self, message: str) -> None:
        self.controls.set_idle(model_loaded=self.engine is not None)
        self.controls.set_status("Model load failed")
        QMessageBox.critical(self, "Model load failed", message)

    def _eta_suffix(self) -> str:
        """Live ETA from measured throughput. Empty until enough chunks have run for
        the average to mean anything."""
        if not self.run_started_at:
            return ""
        return eta_suffix(
            self.chunks_done, self.run_total_chunks,
            time.monotonic() - self.run_started_at,
        )

    def _require_book(self) -> bool:
        if not self.current_book_path:
            QMessageBox.information(self, "No book open", "Open a .txt or .epub book first.")
            return False
        return True

    # A whole novel can be hundreds of thousands of sentences; at tens of seconds each
    # that is months of GPU time. Confirm before starting something that large.
    _HUGE_JOB_CHUNKS = 500

    def _selection(self) -> list[int] | None:
        """Chapters to render, or None for the whole book."""
        return self.chapter_list.selected_indices() or None

    def _current_chapter(self) -> int | None:
        """The one chapter being worked on, or None if the list says nothing.

        Falls back to the row the list has focus on when nothing is explicitly
        selected: clicking a chapter to read it is how a user says which one they
        mean, and making them select it a second time is a step for nothing.
        """
        selected = self.chapter_list.selected_indices()
        if selected:
            return selected[0]
        row = self.chapter_list.currentRow()
        return row if 0 <= row < len(self.chapters) else None

    def _confirm_if_huge(self, action: str) -> bool:
        if self.total_chunks < self._HUGE_JOB_CHUNKS:
            return True
        hours = self.total_chunks * 40 / 3600  # ~40s per sentence, measured on this GPU
        answer = QMessageBox.question(
            self, f"Large {action}",
            f"This will generate {self.total_chunks} sentences, which at roughly 40 seconds "
            f"each is on the order of {hours:.0f} hours of GPU time.\n\n"
            "Select individual chapters in the list to render only part of the book.\n\n"
            f"Start this {action} anyway?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        return answer == QMessageBox.StandardButton.Yes

    # --- script editing --------------------------------------------------------------

    def edit_script(self) -> None:
        """Open the selected chapter for casting, then export the edited text itself.

        Export runs from what the editor holds, not from the book — otherwise the markers
        just placed would be read back out of the untouched epub and ignored.
        """
        if not self._require_book():
            return
        selected = self.chapter_list.selected_indices()
        if not selected:
            QMessageBox.information(
                self, "Pick a chapter", "Select one chapter in the list to edit."
            )
            return
        index = selected[0]
        chapter = self.chapters[index]

        if self.editor_dialog is not None:
            self.editor_dialog.raise_()
            self.editor_dialog.activateWindow()
            return

        # No parent, deliberately. A parented window on Windows gets no taskbar
        # button of its own, cannot be reached with alt-tab, and is pinned above the
        # window that owns it — so editing a script hid the log that says how the
        # rendering is going. Each of these is a window in its own right; whichever
        # one you click is the one on top.
        dialog = ScriptEditorDialog(
            self.settings, chapter.title, chapter.text, parent=None,
            on_reset=lambda: self._original_chapter_text(index),
        )
        # Saving now leaves the dialog open, so the text arrives by signal rather than
        # being read off a closed dialog.
        dialog.script_saved.connect(lambda t, i=index: self._save_chapter_edit(i, t))
        # Non-modal for the same reason the review table is: the two are used together,
        # and a modal editor would block the Sentences window sitting beside it.
        dialog.setModal(False)
        dialog.finished.connect(
            lambda result, d=dialog, i=index, c=chapter: self._script_editor_closed(
                result, d, i, c
            )
        )
        self.editor_dialog = dialog
        dialog.show()

    def _script_editor_closed(self, result, dialog, index, chapter) -> None:
        """What used to follow `dialog.exec()`: save the script, and render if asked."""
        self.editor_dialog = None
        if not result:
            return

        text = dialog.script_text().strip()
        if not text:
            QMessageBox.information(self, "Empty script", "There is nothing to render.")
            return

        if dialog.action() != "export":
            self._save_chapter_edit(index, text)
            return

        # The opening voice becomes a marker in the script rather than a setting that
        # travels alongside it. Casting belongs to the script — that is the whole
        # reason Export audio file no longer asks for a voice — and a choice held only
        # in this combo would be lost the moment the editor closed, leaving the
        # chapter to be re-rendered in whatever the default happened to be.
        text = self._with_opening_marker(text, dialog.start_voice())
        self._save_chapter_edit(index, text)

        if not self._confirm_hard_numbers(text):
            return
        if not self._confirm_cache_room(text):
            return

        # This run renders the edited text of one chapter, so its total comes from
        # that text — not from the chapter list, whose selection is unrelated here.
        run_total = len(chunking.chunk_text(text))
        self.controls.set_status(
            f"Rendering {run_total} sentences of {chapter.title} into the cache…"
        )
        self._ensure_engine(
            lambda: self._start_export(
                SimpleNamespace(
                    out_path="",          # cache only; Export audio file writes the file
                    chapter_indices=[index],
                    start_voice=None,     # the script says who reads what
                    speed=getattr(self.settings, "speed", 1.0),
                ),
                total_chunks=run_total,
            )
        )

    def _with_opening_marker(self, text: str, start_voice: str | None) -> str:
        """Put the editor's opening voice into the script, unless it is already cast.

        Only when it differs from the voice an untouched script would get anyway, so
        the common case leaves the text exactly as the user typed it.
        """
        if not start_voice or chunking.VOICE_MARKER.match(text.lstrip()):
            return text
        if start_voice == voice.resolve_name(self.settings, None):
            return text
        log.info("Writing the opening voice [%s] into the script.", start_voice)
        return f"[{start_voice}] {text}"

    def _confirm_hard_numbers(self, text: str) -> bool:
        """Stop before rendering a chapter whose numbers the model cannot say.

        Measured on this model: 1000000 comes back "một ngàn" and 1500000000 comes
        back "một một năm ngàn" — it has no scale words, so a figure of five digits or
        more is read wrong however the checker counts it. Written out in words it is
        read perfectly, and that is a change only the person with the script can make.

        Asked here rather than warned about afterwards, because afterwards is twenty
        minutes of GPU time and a chapter that has to be redone.
        """
        numbers = vn_numbers.hard_to_read(text)
        if not numbers:
            return True
        shown = ", ".join(dict.fromkeys(numbers))
        if len(shown) > 200:
            shown = shown[:197] + "…"
        answer = QMessageBox.question(
            self, "Numbers this model reads badly",
            f"This chapter still has {len(numbers)} number(s) the model will not "
            f"say correctly:\n\n{shown}\n\n"
            "They are highlighted in the editor. Written out in words \u2014 "
            "\u201cm\u1ed9t tri\u1ec7u\u201d rather than \u201c1000000\u201d \u2014 "
            "the model reads them exactly right.\n\n"
            "Render anyway?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        return answer == QMessageBox.StandardButton.Yes

    def _confirm_cache_room(self, text: str) -> bool:
        """Warn before a chapter that will not fit under the cache limit.

        Rendering is the only thing that grows the cache, so this is the moment the
        user can still decide. Going ahead is allowed: the limit prunes other books'
        takes and this book's rejected ones first, and never a take the book is made
        of — but that pruning is worth knowing about before it happens rather than
        after.
        """
        limit_gb = getattr(self.settings, "cache_limit_gb", 0.0) or 0.0
        if limit_gb <= 0:
            return True
        needed = sentence_cache.estimated_bytes(
            chunking.chunk_text(text), self.settings.speed
        )
        used = sentence_cache.total_size()
        limit = int(limit_gb * 1024 ** 3)
        if used + needed <= limit:
            return True

        def mb(value: int) -> str:
            if value >= 1024 ** 3:
                return f"{value / 1024 ** 3:.2f} GB"
            return f"{value / 1024 ** 2:.0f} MB"

        answer = QMessageBox.question(
            self, "Cache limit",
            f"This chapter is roughly {mb(needed)} of audio, and the cache already "
            f"holds {mb(used)} against a limit of {limit_gb:g} GB.\n\n"
            "Going ahead will drop other books' cached sentences, oldest first, and "
            "then the takes here that you never adopted. Sentences this book is "
            "actually made of are never deleted.\n\n"
            "Render anyway?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes,
        )
        return answer == QMessageBox.StandardButton.Yes

    def _save_chapter_edit(self, index: int, text: str) -> None:
        """Persist one chapter's edited text (and its voice markers).

        Stored regardless of whether an export follows, so re-exporting this book
        later — even in a future session — reuses the tags instead of needing them
        redone.
        """
        title = self.chapters[index].title
        self.chapters[index] = text_extract.Chapter(title=title, text=text)
        text_extract.save_book_cache(self.current_book_path, self.chapters)
        # The sentence count can move when markers are added, and the chapter list
        # shows it.
        self.chunks_per_chapter[index] = len(chunking.chunk_text(text))
        self.chapter_list.set_chapters(
            [c.title for c in self.chapters], self.chunks_per_chapter
        )
        self.chapter_list.setCurrentRow(index)
        log.info("Saved script edits for %r", title)
        self.controls.set_status(f"Saved tags for {title}")

    def _original_chapter_text(self, index: int) -> str | None:
        """This chapter as the book file itself has it, ignoring every saved edit.

        Re-parses from disk rather than trusting the cache, because the cache is
        where the edits live — reading the "original" out of it would hand back the
        very text the user asked to discard. The cost is a full parse, which is why
        this sits behind a confirmation.
        """
        progress = QProgressDialog("Re-reading the book…", None, 0, 0, self)
        progress.setWindowTitle("Reset to original")
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setMinimumDuration(0)
        QApplication.processEvents()
        try:
            fresh = text_extract.load_book(self.current_book_path)
        finally:
            progress.close()
        if index >= len(fresh):
            return None
        return fresh[index].text

    # --- export ------------------------------------------------------------------

    def stop(self) -> None:
        """Cancel whatever is generating. It stops after the current sentence rather
        than mid-generation, and every sentence already rendered stays in the cache."""
        for worker in (self.render_worker,):
            if worker is not None and worker.isRunning():
                worker.cancel()
                self.controls.set_status("Cancelling after the current sentence…")
                return
        self.controls.set_idle(model_loaded=self.engine is not None)

    def export(self) -> None:
        if not self._require_book():
            return
        default_name = naming.safe_filename(Path(self.current_book_path).stem) + ".mp3"
        Path(self.settings.output_dir).mkdir(parents=True, exist_ok=True)
        default_path = str(Path(self.settings.output_dir) / default_name)

        dialog = ExportDialog(
            self.settings,
            [c.title for c in self.chapters],
            self.chunks_per_chapter,
            cached_per_chapter=self._cached_per_chapter(),
            preselected=self.chapter_list.selected_indices() or None,
            default_path=default_path,
            parent=self,
        )
        if not dialog.exec():
            return
        request = dialog.request()
        if request is None:
            QMessageBox.information(
                self, "Nothing to save",
                "Choose where to save the file and tick at least one chapter.",
            )
            return

        # Counted from the range the dialog returned, not from the list selection,
        # and captured here rather than read later: loading the model can take
        # minutes, and the selection may well have moved on by the time it finishes.
        run_total = self._count_selected(request.chapter_indices)
        self._ensure_engine(lambda: self._start_export(request, total_chunks=run_total))

    def _cached_per_chapter(self) -> dict[int, int]:
        """Sentences with a usable take, per chapter, as the book currently reads.

        Counted against the current plan rather than against the manifest: a take for
        text that has since been edited is not audio for this book any more, and
        offering the chapter as "rendered" on the strength of it would write the old
        sentence into the file.
        """
        if not self.current_book_path:
            return {}
        plan = pipeline.sentence_plan(self.chapters, self.settings)
        cache = self._sentence_cache()
        speed = self.settings.speed
        keys = [(p.chapter_index, p.position, p.text, p.voice) for p in plan]
        cache.realign_to(keys, speed)
        gaps = set(cache.missing(keys, speed))
        counts: dict[int, int] = {}
        for planned in plan:
            if (planned.chapter_index, planned.position) not in gaps:
                counts[planned.chapter_index] = counts.get(planned.chapter_index, 0) + 1
        return counts

    def _start_export(self, request, chapters=None, *, total_chunks: int | None = None) -> None:
        """Render whatever the cache is missing, then assemble the file from it.

        Two stages rather than one stream to disk. What that buys is the ability to
        stop at the end of the first: the sentences are on disk individually, so a
        chapter with three bad ones in it can have those three redone instead of all
        four hundred. It also means a run interrupted at sentence 900 keeps the 900.

        The two stages are also separately reachable, and that is the shape of the app
        now: the script editor runs the first (`request.out_path` empty, audio into the
        cache) and Export audio file runs the second.
        """
        chapters = chapters if chapters is not None else self.chapters
        plan = pipeline.sentence_plan(
            chapters, self.settings,
            chapter_indices=request.chapter_indices,
            start_voice=request.start_voice,
        )
        cache = self._sentence_cache()
        speed = request.speed if request.speed is not None else self.settings.speed
        keys = [(p.chapter_index, p.position, p.text, p.voice) for p in plan]
        # Before asking what is missing: an edit slides every later sentence into a new
        # slot, and a take that only looks missing because it is one slot up costs
        # about forty seconds to make again.
        cache.realign_to(keys, speed)
        gaps = set(cache.missing(keys, speed))
        todo = [p for p in plan if (p.chapter_index, p.position) in gaps]
        destination = (request.out_path or "").strip()

        self._pending_assembly = (
            (cache, plan, destination, speed) if destination else None
        )
        if not todo:
            if destination:
                self._assemble_pending()
            else:
                self.controls.set_status(
                    f"All {len(plan)} sentences are already rendered."
                )
                self._finish_export()
            return

        self.chunks_done = 0
        self.run_started_at = time.monotonic()
        self.run_total_chunks = len(todo)
        self.controls.set_running(total=self.run_total_chunks)
        reused = len(plan) - len(todo)
        target = f" for {Path(destination).name}" if destination else ""
        self.controls.set_status(
            f"Rendering {len(todo)} sentences"
            + (f" ({reused} already cached)" if reused else "")
            + f"{target}…"
        )
        self._start_render(
            cache, todo, speed,
            then=self._assemble_pending if destination else None,
        )

    def _sentence_cache(self):
        """The take cache for the book that is open, made once and kept."""
        if (self._cache is None
                or str(self._cache.book_path) != str(Path(self.current_book_path))):
            self._cache = sentence_cache.SentenceCache(self.current_book_path)
        return self._cache

    def _start_render(self, cache, plan, speed, *, then=None, in_use: bool = True) -> None:
        self._after_render = then
        self.render_worker = RenderWorker(
            self.engine, self.settings, cache, plan, speed=speed, in_use=in_use
        )
        self.render_worker.progress.connect(self._on_export_progress)
        self.render_worker.stage.connect(self._on_export_stage)
        self.render_worker.finished_ok.connect(self._on_render_finished)
        self.render_worker.cancelled.connect(self._on_export_cancelled)
        self.render_worker.failed.connect(self._on_export_failed)
        self.render_worker.start()

    def _on_render_finished(self, written: int) -> None:
        if self.review_dialog is not None:
            self.review_dialog.set_running(False)
            self.review_dialog.refresh_all()
        after = self._after_render
        self._after_render = None
        if after is not None:
            after()
            return
        self._finish_export()
        self.controls.set_status(f"Rendered {written} sentences.")

    def _assemble_pending(self) -> None:
        if self._pending_assembly is None:
            self._finish_export()
            return
        cache, plan, out_path, speed = self._pending_assembly
        self._pending_assembly = None
        self.controls.set_status(f"Writing {Path(out_path).name}…")
        self.assemble_worker = AssembleWorker(cache, plan, out_path, speed=speed)
        self.assemble_worker.stage.connect(self._on_export_stage)
        self.assemble_worker.finished_ok.connect(self._on_export_finished)
        self.assemble_worker.failed.connect(self._on_export_failed)
        self.assemble_worker.start()

    def review_sentences(self) -> None:
        """The table of sentences for the chapter being worked on.

        One chapter, never the whole book. A book here runs to 255k sentences and the
        table builds eight widgets a row; more to the point, this is where a chapter is
        gone through failure by failure, and a list spanning every other chapter is a
        list you have to search before you can start.
        """
        if not self._require_book():
            return
        index = self._current_chapter()
        if index is None:
            QMessageBox.information(
                self, "Pick a chapter", "Select the chapter you want to go through."
            )
            return

        if self.review_dialog is not None:
            if self._review_chapter == index:
                self.review_dialog.raise_()
                self.review_dialog.activateWindow()
                return
            # Showing a different chapter now, so the open window is stale rather than
            # the one being asked for.
            self.review_dialog.close()

        plan = pipeline.sentence_plan(
            self.chapters, self.settings, chapter_indices=[index]
        )
        if not plan:
            QMessageBox.information(
                self, "Nothing to show",
                f"{self.chapters[index].title} has no sentences in it.",
            )
            return

        # Realign first, or the table shows each sentence beside the recording of the
        # one before it for everything after an edit.
        cache = self._sentence_cache()
        cache.realign_to(
            [(p.chapter_index, p.position, p.text, p.voice) for p in plan],
            self.settings.speed,
        )

        # Parentless for the same reason the editor is: this table is worked through
        # for as long as re-rendering takes, and the log has to stay reachable.
        dialog = SentenceReviewDialog(plan, cache, self.settings, None)
        dialog.setWindowTitle(f"Sentences — {self.chapters[index].title}")
        self._review_chapter = index
        dialog.render_requested.connect(self._render_from_review)
        dialog.finished.connect(self._review_closed)
        self.review_dialog = dialog
        # Shown, not exec()'d. A chapter is worked through here for as long as it takes
        # to re-render its bad sentences, and a modal loop would lock the script editor
        # and the main window for all of it — including the chapter list this table's
        # own selection came from.
        dialog.show()

    def _review_closed(self) -> None:
        self.review_dialog = None
        self._review_chapter = None
        cache = self._sentence_cache()
        cache.save()
        sentence_cache.enforce_limit(self.settings.cache_limit_gb, keep=cache)

    def _render_from_review(self, planned, voice_name) -> None:
        """Re-render the ticked sentences, optionally recasting them.

        A chosen voice rewrites the plan rather than the book: the take is stored
        against that voice, and the script itself keeps whatever markers it had.
        """
        if voice_name:
            planned = [replace(p, voice=voice_name) for p in planned]
        speed = self.settings.speed
        dialog = self.review_dialog
        if dialog is not None:
            dialog.set_running(True, len(planned))

        def start():
            self.chunks_done = 0
            self.run_started_at = time.monotonic()
            self.run_total_chunks = len(planned)
            self.controls.set_running(total=len(planned))
            self.controls.set_status(f"Rendering {len(planned)} sentences…")
            self._start_render(self._sentence_cache(), planned, speed)

        self._ensure_engine(start)

    def _on_export_stage(self, message: str) -> None:
        """Phases outside the per-sentence loop (parsing, MP3 transcode) — long enough
        on a big book that silence reads as a hang."""
        self.controls.set_status(message)

    def _on_export_progress(self, result) -> None:
        self.chunks_done += 1
        self.chapter_list.highlight_chapter(result.chapter_index)
        self.controls.set_progress(self.chunks_done, self.run_total_chunks)
        voice_suffix = f" [{result.voice_name}]" if result.voice_name else ""
        self.controls.set_status(
            f"Exporting — {result.chapter_title} "
            f"(sentence {result.chunk_index + 1}/{result.total_chunks_in_chapter})"
            f"{voice_suffix}{self._eta_suffix()}"
        )

    def _finish_export(self) -> None:
        self.render_worker = None
        self.assemble_worker = None
        self._pending_assembly = None
        self._after_render = None
        if self._cache is not None:
            self._cache.save()
            # A run is the only thing that grows the cache, so this is where it is
            # brought back under the user's limit — never mid-render, where deleting a
            # file the assembly step is about to read would be the bug this avoids.
            sentence_cache.enforce_limit(self.settings.cache_limit_gb, keep=self._cache)
        if self.review_dialog is not None:
            self.review_dialog.set_running(False)
            self.review_dialog.refresh_all()
        self.controls.set_idle(model_loaded=self.engine is not None)

    def _on_export_finished(self, out_path: str) -> None:
        self._finish_export()
        self.controls.set_status(f"Exported to {out_path}")
        QMessageBox.information(
            self, "Export complete",
            f"Saved to {out_path}\n\nCheck the log for any coverage warnings — this model can "
            "occasionally drop words, and those warnings are the only sign of it.",
        )

    def _on_export_cancelled(self, message: str) -> None:
        self._finish_export()
        self.controls.set_status("Export cancelled")
        QMessageBox.information(self, "Export cancelled", message)

    def _on_export_failed(self, message: str) -> None:
        self._finish_export()
        self.controls.set_status("Export failed")
        QMessageBox.critical(self, "Export failed", message)

    # --- shutdown ------------------------------------------------------------------

    def closeEvent(self, event) -> None:
        """Stop background work and release the GPU before the window goes away.

        Two separate concerns. QThreads must not be destroyed while running, and the
        model must be dropped explicitly — a worker still holding a reference keeps
        several GB of VRAM occupied even once the window is gone, which looks exactly
        like the app "not letting go" of the card.
        """
        if self.render_worker is not None and self.render_worker.isRunning():
            self.render_worker.cancel()
            if not self.render_worker.wait(15000):
                log.warning("Render thread did not stop in time; terminating it.")
                self.render_worker.terminate()
                self.render_worker.wait(2000)
        self.render_worker = None
        # Assembling only reads the cache and writes a file, so it finishes quickly
        # and has nothing to cancel — waiting is enough.
        if self.assemble_worker is not None and self.assemble_worker.isRunning():
            self.assemble_worker.wait(30000)
        self.assemble_worker = None
        if self._cache is not None:
            self._cache.save()
        if self.load_worker and self.load_worker.isRunning():
            # A load in flight can't be cancelled safely, so wait it out rather than
            # tearing down CUDA underneath it.
            self.controls.set_status("Waiting for the model load to finish before closing…")
            self.load_worker.wait(120000)
        self.load_worker = None

        if self.engine is not None:
            self.engine = None
            remaining = free_vram()
            log.info("Released model on close; %.1f GB CUDA memory still allocated", remaining)

        # These have no parent, so Qt will not take them down with this window — and a
        # surviving top-level widget keeps the application alive with no way back to it.
        for window in (self.editor_dialog, self.review_dialog):
            if window is not None:
                window.close()
        self.editor_dialog = None
        self.review_dialog = None

        event.accept()
