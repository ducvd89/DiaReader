"""Main window: library, chapter list, model control, export."""
from __future__ import annotations

import logging
import time
from pathlib import Path
from types import SimpleNamespace

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication, QFileDialog, QMainWindow, QMessageBox, QProgressDialog, QSplitter,
    QWidget, QVBoxLayout, QLabel
)

from .. import chunking, config, naming, pipeline, text_extract, voice
from ..model_loader import CudaNotAvailableError, free_vram, require_cuda
from .export_dialog import ExportDialog
from .script_editor import ScriptEditorDialog
from .settings_dialog import SettingsDialog
from .widgets import ChapterListWidget, LogBridge, PlayerControls, QtLogHandler
from .workers import ExportWorker, ModelLoadWorker

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
        self.run_started_at: float | None = None

        self.load_worker: ModelLoadWorker | None = None
        self.export_worker: ExportWorker | None = None

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

        dialog = ScriptEditorDialog(
            self.settings, chapter.title, chapter.text, parent=self,
            on_reset=lambda: self._original_chapter_text(index),
        )
        # Saving now leaves the dialog open, so the text arrives by signal rather than
        # being read off a closed dialog.
        dialog.script_saved.connect(lambda t, i=index: self._save_chapter_edit(i, t))
        if not dialog.exec():
            return

        text = dialog.script_text().strip()
        if not text:
            QMessageBox.information(self, "Empty script", "There is nothing to export.")
            return

        self._save_chapter_edit(index, text)

        if dialog.action() != "export":
            return

        default_path = str(
            Path(self.settings.output_dir) / f"{naming.safe_filename(chapter.title)}.mp3"
        )
        Path(self.settings.output_dir).mkdir(parents=True, exist_ok=True)
        out_path, _ = QFileDialog.getSaveFileName(
            self, "Export script", default_path, "MP3 (*.mp3);;WAV (*.wav)"
        )
        if not out_path:
            return

        edited = [text_extract.Chapter(title=chapter.title, text=text)]
        # This run renders the edited text of one chapter, so its total comes from
        # that text — not from the chapter list, whose selection is unrelated here.
        run_total = len(chunking.chunk_text(text))
        self._ensure_engine(
            lambda: self._start_export(
                SimpleNamespace(
                    out_path=out_path,
                    chapter_indices=[0],
                    start_voice=dialog.start_voice(),
                    speed=getattr(self.settings, "speed", 1.0),
                ),
                chapters=edited,
                total_chunks=run_total,
            )
        )

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
        """Cancel a running export. It stops after the current sentence rather than
        mid-generation, and the partial WAV is kept."""
        if self.export_worker and self.export_worker.isRunning():
            self.export_worker.cancel()
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
            [self.chapter_list.item(i).text() for i in range(self.chapter_list.count())],
            self.chunks_per_chapter,
            preselected=self.chapter_list.selected_indices() or None,
            default_path=default_path,
            parent=self,
        )
        if not dialog.exec():
            return
        request = dialog.request()
        if request is None:
            QMessageBox.information(self, "No destination", "Choose where to save the file.")
            return

        # Counted from the range the dialog returned, not from the list selection,
        # and captured here rather than read later: loading the model can take
        # minutes, and the selection may well have moved on by the time it finishes.
        run_total = self._count_selected(request.chapter_indices)
        self._ensure_engine(lambda: self._start_export(request, total_chunks=run_total))

    def _start_export(self, request, chapters=None, *, total_chunks: int | None = None) -> None:
        self.chunks_done = 0
        self.run_started_at = time.monotonic()
        self.run_total_chunks = (
            total_chunks if total_chunks is not None else self.total_chunks
        )
        self.controls.set_running(total=self.run_total_chunks)
        self.controls.set_status(f"Exporting to {Path(request.out_path).name}…")
        self.export_worker = ExportWorker(
            self.current_book_path, request.out_path, self.engine, self.settings,
            chapter_indices=request.chapter_indices,
            start_voice=request.start_voice,
            chapters=chapters if chapters is not None else self.chapters,
            speed=request.speed,
        )
        self.export_worker.progress.connect(self._on_export_progress)
        self.export_worker.stage.connect(self._on_export_stage)
        self.export_worker.finished_ok.connect(self._on_export_finished)
        self.export_worker.cancelled.connect(self._on_export_cancelled)
        self.export_worker.failed.connect(self._on_export_failed)
        self.export_worker.start()

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
        self.export_worker = None
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
        if self.export_worker and self.export_worker.isRunning():
            self.export_worker.cancel()
            if not self.export_worker.wait(15000):
                log.warning("Export thread did not stop in time; terminating it.")
                self.export_worker.terminate()
                self.export_worker.wait(2000)
        self.export_worker = None
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

        event.accept()
