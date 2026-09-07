"""Report on the things that make this app fail to work on someone else's machine.

Written for the packaged build, where there is no console to print to and no venv to
poke at: every failure looks the same from outside — the window does not appear, or
narration comes out silent. Each check here is one that has actually gone wrong:

  CUDA          stable PyTorch has no Blackwell support; the wrong wheel installs
                cleanly and then refuses at load time
  FFmpeg        torchcodec needs the shared libraries, and without them every
                voice-conditioned sentence produces no audio and no error, so a
                multi-hour export finishes as a file of silence
  torch.compile JIT-compiles through Triton at runtime, which is exactly the kind of
                thing that stops working once an app is frozen
  C compiler    inductor shells out to one per kernel, on the user's machine; the
                build machine has Visual Studio and never notices it is required
"""
from __future__ import annotations

import platform
import sys
import threading
import time
from pathlib import Path


def _line(ok: bool | None, label: str, detail: str) -> str:
    mark = {True: "  OK  ", False: " FAIL ", None: " WARN "}[ok]
    return f"[{mark}] {label:<22} {detail}"


def run() -> tuple[bool, str]:
    """(everything_essential_passed, report)."""
    rows: list[str] = []
    essential_ok = True

    frozen = getattr(sys, "frozen", False)
    rows.append(_line(True, "Build", f"{'packaged' if frozen else 'from source'} · "
                                     f"Python {platform.python_version()} · {platform.machine()}"))

    # --- writing to stdout/stderr ---
    # A windowed build has no console, so PyInstaller sets both streams to None and
    # anything that writes to them without checking raises AttributeError. That is
    # not hypothetical: huggingface_hub draws a tqdm progress bar on sys.stderr while
    # downloading, which crashed model loading outright. rthook_stdio.py substitutes
    # os.devnull; this checks the substitution is actually in place, through the same
    # call chain tqdm uses (bar.refresh -> fp_write -> stream.write).
    try:
        for name in ("stdout", "stderr"):
            stream = getattr(sys, name)
            if stream is None:
                raise AttributeError(f"sys.{name} is None — rthook_stdio did not run")
            stream.write("")
            stream.flush()

        from tqdm import tqdm

        # disable=False, not None: None means "disable unless a TTY", which devnull
        # never is, and a disabled bar writes nothing and would pass vacuously.
        with tqdm(total=1, file=sys.stderr, disable=False) as bar:
            bar.update(1)
            bar.refresh()
        rows.append(_line(True, "Console output", "stdout/stderr writable, progress bars safe"))
    except Exception as e:  # noqa: BLE001
        essential_ok = False
        rows.append(_line(False, "Console output",
                          f"{type(e).__name__}: {e} — downloads will crash the model load"))

    # --- torch and the GPU ---
    try:
        import torch

        rows.append(_line(True, "PyTorch", f"{torch.__version__} (CUDA {torch.version.cuda})"))
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            cap = torch.cuda.get_device_capability(0)
            rows.append(_line(True, "GPU", f"{name} · {vram:.1f} GB · sm_{cap[0]}{cap[1]}"))
        else:
            essential_ok = False
            rows.append(_line(False, "GPU", "no CUDA device — DiaReader cannot generate audio"))
    except Exception as e:  # noqa: BLE001
        essential_ok = False
        rows.append(_line(False, "PyTorch", f"{type(e).__name__}: {e}"))
        return essential_ok, "\n".join(rows)

    # --- reading voice reference audio ---
    # Reported separately from torchcodec itself: the import below only succeeds
    # because the package __init__ already registered this directory, so a failure
    # with a directory shown here means a broken FFmpeg build, while a failure with
    # no directory means none was found at all. Different fixes, same symptom.
    from diareader import ffmpeg_dlls

    ffmpeg_dir = ffmpeg_dlls.registered_dir()
    if ffmpeg_dir is not None:
        rows.append(_line(True, "FFmpeg libraries", str(ffmpeg_dir)))
    else:
        rows.append(_line(None, "FFmpeg libraries",
                          "no shared build found — 'winget install Gyan.FFmpeg.Shared', "
                          f"or set {ffmpeg_dlls.ENV_OVERRIDE}"))

    try:
        import torchaudio  # noqa: F401
        import torchcodec  # noqa: F401

        rows.append(_line(True, "Audio decoding", f"torchcodec {torchcodec.__version__}"))
    except Exception as e:  # noqa: BLE001
        detail = str(e)
        hint = ("FFmpeg's shared libraries are missing — install them with "
                "'winget install Gyan.FFmpeg.Shared' and restart"
                if "libtorchcodec" in detail else detail[:120])
        rows.append(_line(None, "Audio decoding", hint))
        rows.append(_line(None, "", "voices will not work; unconditioned narration still will"))

    # --- torch.compile, which is where 1.56x of the speed lives ---
    # Run on a thread with a large native stack, matching the workers that do this
    # for real (gui/workers.py): a deep dynamo trace on a default-sized stack dies
    # as a native stack overflow, killing the process outright instead of raising
    # something this function could report.
    from diareader import compile_support

    before = compile_support.describe()
    patched = compile_support.apply()
    bundled_cc = compile_support.ensure_c_compiler()

    # Inductor shells out to this for every kernel it generates, on the machine the
    # app is running on. A build machine has Visual Studio and never notices; the
    # machine the installer was carried to raised "Failed to find C compiler" once
    # per sentence and wrote an export of pure silence.
    headers = compile_support.use_bundled_python_headers()
    found_cc, cc = compile_support.c_compiler()

    app_dir = Path(sys.executable).resolve().parent

    def _inside_app(path: str) -> bool:
        try:
            return Path(path).resolve().is_relative_to(app_dir)
        except (OSError, ValueError):
            return False

    # Every input to a run-time compile, checked for the one property that decides
    # whether this build works anywhere but here: does it live inside the installed
    # folder? Three separate bugs have now been "resolved from the build machine" —
    # the compiler, then CPython's headers, then its import library — and each was
    # invisible on the machine that built the app and fatal on the next one.
    inputs: list[tuple[str, str]] = [("C compiler", cc if found_cc else "")]
    pybuild = compile_support.bundled_python_build_dirs()
    if pybuild is not None:
        inputs += [("Python headers", str(pybuild[0])), ("Python import lib", str(pybuild[1]))]

    rows.append(_line(found_cc or None, "C compiler",
                      cc if found_cc else f"{cc} — torch.compile cannot run"))
    if bundled_cc:
        rows.append(_line(True, "  bundled compiler", f"using {bundled_cc}"))
    if headers:
        rows.append(_line(True, "  bundled headers", headers))
    elif frozen:
        essential_ok = False
        rows.append(_line(False, "Python headers",
                          "not in this build — compiling needs Python.h and "
                          "pythonNNN.lib, which only exist here because Python is "
                          "installed on this machine"))

    if frozen:
        for label, path in inputs:
            if path and not _inside_app(path):
                essential_ok = False
                rows.append(_line(False, "", f"{label} resolves outside {app_dir} "
                                             f"({path}) — it will not exist on any "
                                             f"other machine"))

    result: dict = {}

    def _check_compile() -> None:
        try:
            t0 = time.perf_counter()

            # A module, not a bare function, and the graph breaks are counted. The
            # difference matters: the bundled-path bug (see compile_support) is fixed
            # by teaching dynamo which files are torch's, and getting that half-right
            # — skipping all of torch rather than none of it — would compile without
            # error and without speedup. A bare `x * 2 + 1` cannot tell those apart,
            # because it never asks dynamo to trace into torch.nn at all.
            import torch._dynamo
            from torch._dynamo.utils import counters

            torch._dynamo.reset()
            counters.clear()

            class _Tiny(torch.nn.Module):
                def __init__(self) -> None:
                    super().__init__()
                    self.fc = torch.nn.Linear(8, 8)

                def forward(self, x):
                    return torch.nn.functional.relu(self.fc(x)) * 2 + 1

            compiled = torch.compile(_Tiny().cuda())
            out = compiled(torch.ones(8, device="cuda"))
            torch.cuda.synchronize()
            assert out.shape == (8,) and bool(torch.isfinite(out).all())

            breaks = sum(counters["graph_break"].values())
            warmup = f"{time.perf_counter() - t0:.1f}s warmup"
            if breaks:
                result["row"] = _line(None, "torch.compile",
                                      f"compiles but breaks the graph {breaks}x — "
                                      f"the 1.56x speedup will not materialise")
            else:
                result["row"] = _line(True, "torch.compile",
                                      f"working, whole graph ({warmup})")
        except Exception as e:  # noqa: BLE001
            result["row"] = _line(None, "torch.compile",
                                  f"unavailable, generation will be ~1.56x slower — "
                                  f"{type(e).__name__}: {str(e)[:80]}")
            # Name the frames it died in. For a RecursionError the tail is the cycle
            # that is repeating, which is the only thing that says whether the fix is
            # "raise the limit" or "stop something recursing forever" — guessing
            # between those two cost one rebuild already.
            import traceback

            frames = traceback.extract_tb(e.__traceback__)
            result["detail"] = [f"{f.filename}:{f.lineno} in {f.name}" for f in frames[-8:]]

    prev_stack_size = threading.stack_size()
    threading.stack_size(128 * 1024 * 1024)
    try:
        t = threading.Thread(target=_check_compile)
        t.start()
        t.join()
    finally:
        threading.stack_size(prev_stack_size)
    rows.append(result.get("row", _line(None, "torch.compile", "check did not complete")))
    if patched:
        rows.append(_line(True, "  bundle paths", f"corrected {patched}"))
    if "detail" in result:
        # Only on failure: the two paths whose disagreement causes it, then the
        # frames it died in. Diagnosing this from "RecursionError" alone is not
        # possible — it took a build spent purely on printing this.
        for note in before:
            rows.append(_line(None, "", note))
        for frame in result["detail"]:
            rows.append(_line(None, "", frame))

    # --- settings and the folders they name ---
    try:
        from diareader import config, voice

        settings = config.Settings.load()
        config.ensure_dirs()
        voices = voice.load_voices(settings)
        rows.append(_line(True, "Settings", f"{config.SETTINGS_PATH}"))
        rows.append(_line(True, "Voices", f"{len(voices)} configured"
                                          + (f": {', '.join(voices)}" if voices else "")))
        token = "set" if settings.hf_token else "MISSING — the model cannot be downloaded"
        rows.append(_line(bool(settings.hf_token), "Hugging Face token", token))
        if not settings.hf_token:
            essential_ok = False
        cached = config.MODEL_CACHE_DIR.exists() and any(config.MODEL_CACHE_DIR.iterdir())
        rows.append(_line(True if cached else None, "Model cache",
                          str(config.MODEL_CACHE_DIR) if cached
                          else "empty — the first run downloads ~6.3 GB"))
    except Exception as e:  # noqa: BLE001
        essential_ok = False
        rows.append(_line(False, "Settings", f"{type(e).__name__}: {e}"))

    return essential_ok, "\n".join(rows)


def main(show_window: bool | None = None) -> int:
    ok, report = run()
    print(report, flush=True)

    out = Path.home() / "DiaReader-selftest.txt"
    try:
        out.write_text(report, encoding="utf-8")
        print(f"\nSaved to {out}", flush=True)
    except OSError:
        pass

    # A modal box is the only way to show this in the packaged build, which has no
    # console — and exactly the wrong thing from a terminal, where it blocks on a
    # click nobody is there to give. Default to whichever the build actually is.
    if show_window is None:
        show_window = bool(getattr(sys, "frozen", False))
    if show_window:
        try:
            from PySide6.QtWidgets import QApplication, QMessageBox

            app = QApplication.instance() or QApplication(sys.argv)
            box = QMessageBox()
            box.setWindowTitle("DiaReader self-test")
            box.setText("Everything needed is present." if ok
                        else "Something essential is missing — see below.")
            box.setDetailedText(report)
            box.exec()
            del app
        except Exception:  # noqa: BLE001 — the printed report is the real output
            pass
    return 0 if ok else 1
