"""Everything torch.compile needs in order to work where this app is actually run.

torch.compile is worth 1.56x and is on by default, and every obstacle to it so far
has been invisible on the machine that built the app and fatal on the machine that
ran it. Two are handled here — dynamo's view of the bundle, and the C compiler
inductor shells out to — plus a probe that decides whether to use it at all.

1. Where torch's own code lives
-------------------------------
Dynamo decides whether to trace *into* a function by matching that function's
filename against the directory torch was imported from. In a frozen build the two
disagree: bundled code objects carry a filename relative to the bundle
("torch/_dynamo/utils.py"), while torch.__file__ is absolute
("...\\_internal\\torch\\__init__.pyc"). No prefix matches, so dynamo stops
recognising torch's internals as internals and tries to trace them. The first one it
reaches, torch._dynamo.utils.get_current_stream, calls back into dynamo, which
compiles it, which calls it again — recursion with no fixed point.

The symptom was "RecursionError: maximum recursion depth exceeded in
__instancecheck__" the instant torch.compile ran, packaged only, never from source.
It reads like a depth problem and is not one: raising the recursion limit from 10000
to 50000 changed nothing, because the recursion has no bottom.

Everything derived from that directory is either computed lazily behind
functools.cache or read through _module_dir at call time, so correcting the one
function and dropping those caches fixes the whole chain.

2. The C compiler
-----------------
Inductor compiles the launcher for every generated kernel with a *host* C compiler,
at run time, on the user's machine. Triton ships a TinyCC for machines without
Visual Studio, but looks for it under `sysconfig.get_paths()["platlib"]`, which in a
bundle points at a directory that does not exist. On a machine with no other
compiler that raises "Failed to find C compiler" once per sentence.

3. Whether it works at all
--------------------------
`probe()` compiles something trivial before any book is touched. It exists because
the alternative is finding out per sentence: compilation fails the same way on every
one of them, each is retried five times, each retry yields empty audio, and the
export is written out as hours of silence. The load is refused instead, with
`remedy()` saying what to install — deliberately not a quiet fall back to eager,
which would hide a broken install behind a machine that is merely slower.
"""
from __future__ import annotations

import os
import sys


def _posix(path: str) -> str:
    return path.replace("\\", "/")


def _bundled_torch_dir() -> str | None:
    """The prefix torch's *code objects* actually carry, or None if unbundled.

    Read off a real torch function rather than assumed, so that a PyInstaller
    version that keeps absolute filenames makes this return None and the patch
    below correctly does nothing.
    """
    from torch._dynamo import utils as dynamo_utils

    path = _posix(dynamo_utils.get_current_stream.__code__.co_filename)
    marker = "torch/_dynamo/"
    index = path.find(marker)
    if index == -1:
        return None
    return path[: index + len("torch/")]


def ensure_c_compiler() -> str | None:
    """Point triton at the TinyCC shipped in the bundle. Returns its path, or None.

    Unconditional when frozen, rather than a fallback for when triton finds nothing.
    Triton locates its bundled TinyCC through ``sysconfig.get_paths()["platlib"]``,
    which in a PyInstaller app still names the *build* machine's site-packages. On
    the machine that built the app that path exists, so everything works and the bug
    is invisible; anywhere else it does not, and every sentence of an export dies
    with "Failed to find C compiler". Waiting for that failure before correcting the
    path would mean the correction never runs where it can be tested.

    A CC set by hand always wins, and triton itself prefers this TinyCC over an
    installed MSVC, so nothing is downgraded by making it explicit.
    """
    if not getattr(sys, "frozen", False) or os.environ.get("CC"):
        return None

    from pathlib import Path

    import triton
    import triton.runtime.build as triton_build

    tcc = Path(triton.__file__).resolve().parent / "runtime" / "tcc" / "tcc.exe"
    if not tcc.is_file():
        return None

    os.environ["CC"] = str(tcc)
    getattr(triton_build.get_cc, "cache_clear", lambda: None)()
    return str(tcc)


def c_compiler() -> tuple[bool, str]:
    """(found, description) — what inductor will shell out to on this machine."""
    try:
        from triton.runtime.build import get_cc

        return True, get_cc()
    except Exception as e:  # noqa: BLE001
        return False, f"{type(e).__name__}: {e}"


def probe(torch_module) -> tuple[bool, str]:
    """Compile something trivial. (usable, detail).

    Cheap next to a multi-minute model load, and the only honest way to answer the
    question, since every failure so far has been specific to the machine.
    """
    import time

    t0 = time.perf_counter()
    try:
        compiled = torch_module.compile(lambda x: x * 2 + 1)
        out = compiled(torch_module.ones(8, device="cuda"))
        torch_module.cuda.synchronize()
        if float(out[0]) != 3.0:
            return False, "compiled function returned the wrong value"
    except Exception as e:  # noqa: BLE001
        return False, f"{type(e).__name__}: {str(e)[:160]}"
    return True, f"{time.perf_counter() - t0:.1f}s"


def remedy(detail: str) -> str:
    """What to install, for a machine where torch.compile will not run."""
    found, cc = c_compiler()
    compiler_line = (
        f"The C compiler it found is:\n    {cc}\n"
        if found else
        f"No C compiler could be found ({cc}).\n"
    )
    return (
        "torch.compile is enabled but does not work on this machine, so DiaReader "
        "stopped rather than generate a book that might come out silent.\n\n"
        f"{compiler_line}\n"
        f"What failed:\n    {detail}\n\n"
        "torch.compile builds a small C launcher for every GPU kernel it generates, "
        "which needs a C compiler present at run time. DiaReader ships one (TinyCC) "
        "and normally uses it without any setup, so if you are seeing this, install "
        "Microsoft's C++ build tools:\n\n"
        "    winget install Microsoft.VisualStudio.2022.BuildTools "
        "--override \"--quiet --add Microsoft.VisualStudio.Workload.VCTools "
        "--includeRecommended\"\n\n"
        "Then restart DiaReader so it picks up the new PATH.\n\n"
        "Alternatively, turn off 'use_torch_compile' in Settings. Generation then "
        "runs about 1.56x slower, but needs no compiler at all."
    )


def describe() -> list[str]:
    """Diagnostics for the selftest: the two paths whose disagreement is the bug."""
    try:
        import torch
        from torch._dynamo import trace_rules
        from torch._dynamo import utils as dynamo_utils

        code_path = _posix(dynamo_utils.get_current_stream.__code__.co_filename)
        module_dir = trace_rules._module_dir(torch)
        skip = trace_rules.check_file(code_path)
        return [
            f"torch code objects say: {code_path}",
            f"torch module dir says:  {module_dir}",
            f"dynamo would skip it:   {skip.skipped} ({skip.reason})",
        ]
    except Exception as e:  # noqa: BLE001 — diagnostics must never be the thing that fails
        return [f"could not inspect dynamo paths: {type(e).__name__}: {e}"]


def apply() -> str | None:
    """Point dynamo at the bundled paths. Returns what changed, or None if nothing did."""
    if not getattr(sys, "frozen", False):
        return None

    import torch
    from torch._dynamo import trace_rules

    bundled = _bundled_torch_dir()
    if bundled is None:
        return None

    current = trace_rules._module_dir(torch)
    if current is not None and _posix(current).startswith(bundled):
        return None  # paths already agree; nothing to correct

    original = trace_rules._module_dir

    def _module_dir(module):
        # torch.nn -> "torch/nn/", matching how the bundle names the code objects.
        # Anything outside the torch package keeps the stock behaviour.
        name = getattr(module, "__name__", "")
        if name == "torch" or name.startswith("torch."):
            return name.replace(".", "/") + "/"
        return original(module)

    trace_rules._module_dir = _module_dir

    # These three are functools.cache'd and may already hold absolute paths.
    for getter in (
        trace_rules.get_mod_inlinelist,
        trace_rules.get_mod_skiplist,
        trace_rules.get_legacy_mod_inlinelist,
    ):
        getter.cache_clear()
    # Computed once at module import, so re-derive it rather than clear it.
    trace_rules.FORCE_SKIP_FILES = {f"{bundled}optim/lr_scheduler.py"}

    return f"{current} -> {bundled}"
