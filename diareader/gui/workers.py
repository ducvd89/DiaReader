"""QThread workers so model loading / generation / export never block the UI thread."""
from __future__ import annotations

import logging

from PySide6.QtCore import QThread, Signal

from .. import config
from ..model_loader import (
    CudaNotAvailableError, TorchCompileUnavailableError, load_model
)
from ..pipeline import ExportCancelled
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
            engine = TtsEngine(
                load_model(self.settings, on_stage=lambda m: self.stage.emit(m))
            )
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


class RenderWorker(QThread):
    """Generate a given list of sentences into the sentence cache.

    Its unit of work is a list of sentences rather than a book, which is what lets one
    worker serve both jobs: rendering what an export is missing, and re-rendering the
    handful somebody ticked in the review table — possibly in a voice none of them was
    written with. It holds the GPU, so the window runs one at a time.
    """

    progress = Signal(object)   # ChunkResult
    stage = Signal(str)
    finished_ok = Signal(int)   # sentences written
    cancelled = Signal(str)
    failed = Signal(str)

    def __init__(self, engine, settings: config.Settings, cache, plan,
                 speed: float | None = None, in_use: bool = True, parent=None):
        super().__init__(parent)
        self.engine = engine
        self.settings = settings
        self.cache = cache
        self.plan = plan
        self.speed = speed
        self.in_use = in_use
        self._keep_going = True
        # See ModelLoadWorker.__init__.
        self.setStackSize(128 * 1024 * 1024)

    def cancel(self) -> None:
        self._keep_going = False

    def run(self) -> None:
        from ..pipeline import render_sentences

        try:
            written = render_sentences(
                self.engine, self.settings, self.cache, self.plan,
                speed=self.speed,
                on_progress=lambda r: self.progress.emit(r),
                should_continue=lambda: self._keep_going,
                on_stage=lambda m: self.stage.emit(m),
                in_use=self.in_use,
            )
        except ExportCancelled as e:
            self.cancelled.emit(str(e))
            return
        except Exception as e:  # noqa: BLE001 — surface it rather than dying silently
            log.exception("Rendering failed")
            self.failed.emit(f"Rendering failed: {e}")
            return
        self.finished_ok.emit(written)


class AssembleWorker(QThread):
    """Write the final file from the cache. Generates nothing, so it needs no GPU."""

    stage = Signal(str)
    finished_ok = Signal(str)
    failed = Signal(str)

    def __init__(self, cache, plan, out_path: str, speed: float | None = None, parent=None):
        super().__init__(parent)
        self.cache = cache
        self.plan = plan
        self.out_path = out_path
        self.speed = speed

    def run(self) -> None:
        from ..pipeline import assemble

        try:
            result = assemble(
                self.cache, self.plan, self.out_path, speed=self.speed,
                on_stage=lambda m: self.stage.emit(m),
            )
        except Exception as e:  # noqa: BLE001
            log.exception("Assembling the final file failed")
            self.failed.emit(f"Could not write {self.out_path}: {e}")
            return
        self.finished_ok.emit(str(result))
