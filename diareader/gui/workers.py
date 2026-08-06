"""QThread workers so model loading / generation / export never block the UI thread."""
from __future__ import annotations

import logging

from PySide6.QtCore import QThread, Signal

from .. import config
from ..model_loader import CudaNotAvailableError, TorchCompileUnavailableError, load_model
from ..pipeline import ExportCancelled, export_book
from ..tts_engine import TtsEngine

log = logging.getLogger(__name__)


class ModelLoadWorker(QThread):
    finished_ok = Signal(object)  # TtsEngine
    stage = Signal(str)
    failed = Signal(str)

    def __init__(self, settings: config.Settings, parent=None):
        super().__init__(parent)
        self.settings = settings
        # Headroom for torch.compile, whose trace is by far the deepest thing that
        # runs on this thread. Qt's default stack is small enough that a deep trace
        # dies as a native stack overflow — the process simply exits, with no
        # exception to catch and nothing in the log. That is how the packaged
        # build's compile failure first presented, and giving it room here is what
        # turned it into a readable RecursionError long enough to find the real
        # cause (diareader/compile_support.py). Kept as insurance, not as the fix.
        # 256 MiB is rejected outright by Windows' CreateThread; 128 MiB is not.
        self.setStackSize(128 * 1024 * 1024)

    def run(self) -> None:
        try:
            loaded = load_model(self.settings, on_stage=lambda m: self.stage.emit(m))
            engine = TtsEngine(loaded)
        except (CudaNotAvailableError, TorchCompileUnavailableError) as e:
            # Both already say what is wrong and what to do about it; wrapping them
            # in "Failed to load model:" would only bury the instructions.
            self.failed.emit(str(e))
            return
        except Exception as e:  # noqa: BLE001 - surface any loading error to the UI
            log.exception("Model load failed")
            self.failed.emit(f"Failed to load model: {e}")
            return
        self.finished_ok.emit(engine)


class ExportWorker(QThread):
    progress = Signal(object)  # ChunkResult
    stage = Signal(str)
    finished_ok = Signal(str)  # output path
    cancelled = Signal(str)
    failed = Signal(str)

    def __init__(
        self, book_path: str, out_path: str, engine: TtsEngine, settings: config.Settings,
        chapter_indices=None, start_voice: str | None = None, chapters=None,
        speed: float | None = None, parent=None,
    ):
        super().__init__(parent)
        self.book_path = book_path
        self.out_path = out_path
        self.chapters = chapters
        self.speed = speed
        self.engine = engine
        self.settings = settings
        self.chapter_indices = chapter_indices
        self.start_voice = start_voice
        self._keep_going = True
        # See ModelLoadWorker.__init__ — the same torch.compile-in-a-frozen-build
        # stack overflow risk applies here, since generation is where compilation
        # of the first sentence actually happens.
        self.setStackSize(128 * 1024 * 1024)

    def cancel(self) -> None:
        self._keep_going = False

    def run(self) -> None:
        try:
            result_path = export_book(
                self.book_path, self.out_path, self.engine, self.settings,
                on_progress=lambda r: self.progress.emit(r),
                should_continue=lambda: self._keep_going,
                chapter_indices=self.chapter_indices,
                start_voice=self.start_voice,
                on_stage=lambda m: self.stage.emit(m),
                chapters=self.chapters,
                speed=self.speed,
            )
        except ExportCancelled as e:
            self.cancelled.emit(str(e))
            return
        except Exception as e:  # noqa: BLE001
            log.exception("Export failed")
            self.failed.emit(f"Export failed: {e}")
            return
        self.finished_ok.emit(str(result_path))
