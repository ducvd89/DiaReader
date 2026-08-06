# PyInstaller spec for DiaReader.
#
# Built as a one-FOLDER app, not a single file. A one-file build unpacks ~4.5 GB to a
# temp directory on every launch, which is both slow and a good way to fill a disk.
#
# The collect_all list is not defensive padding — each entry is a package that loads
# something at runtime that static analysis cannot see:
#   torch/torchaudio  thousands of DLLs and .py files reached dynamically
#   triton            torch.compile JIT-compiles through it, from its own Python source
#   torchcodec        C++ shared libraries, one per FFmpeg major version
#   dia               the checkpoint architecture, plus a static/ folder
#   dac, audiotools   the DAC codec that turns tokens back into a waveform
#   numba, llvmlite   generate code at runtime the same way
#   soundfile         libsndfile lives beside the module, not in it
#   pyloudnorm, scipy loudness metering and the peak finder for the sound-count check
#   tensorboard       audiotools.ml.decorators does `from torch.utils.tensorboard import
#                      SummaryWriter` unconditionally at module import time, so it is
#                      reached just by loading dia/dac even though the app never logs to
#                      it. Excluding it breaks model load; the package itself is ~11 MB,
#                      not the "well over a gigabyte" originally assumed here.
import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_all

PROJECT = Path(SPECPATH).parent

datas, binaries, hiddenimports = [], [], []
for package in (
    "torch", "torchaudio", "torchcodec", "triton",
    "dia", "dac", "audiotools", "tensorboard",
    "numba", "llvmlite", "soundfile", "pyloudnorm", "scipy",
):
    try:
        d, b, h = collect_all(package)
    except Exception as exc:  # noqa: BLE001 — report, do not silently ship without it
        print(f"WARNING: could not collect {package}: {exc}", file=sys.stderr)
        continue
    datas += d
    binaries += b
    hiddenimports += h


def build_voice_bundle():
    """Pack this machine's voice library so a fresh install starts with voices.

    Voices live in %APPDATA%, which the installer never writes, so without this a
    new machine opens with an empty voice list no matter how many the build machine
    had — which is exactly what happened to the first installer that shipped.

    Only the voices are taken. settings.json is *not* copied: it holds the Hugging
    Face token in plain text, and an installer is something people pass around.
    """
    import json
    import tempfile

    sys.path.insert(0, str(PROJECT))
    try:
        from diareader import config as app_config
        from diareader import voice as app_voice
        from diareader import voice_backup
    except Exception as exc:  # noqa: BLE001
        print(f"WARNING: cannot read the voice library ({exc}); shipping without voices",
              file=sys.stderr)
        return []

    settings = app_config.Settings.load()
    names = list(app_voice.load_voices(settings))
    if not names:
        print("NOTE: no voices configured on this machine; shipping without any.")
        return []

    from diareader import bundled_voices

    out = Path(tempfile.mkdtemp(prefix="diareader-voices-")) / bundled_voices.BUNDLE_NAME
    count = voice_backup.export_library(settings, out)
    print(f"NOTE: bundling {count} voices ({', '.join(names)}) — reference audio only, "
          f"no token, no settings.")
    # Guard the promise above rather than just making it: read the manifest back and
    # fail the build if anything resembling a secret got in.
    import zipfile

    with zipfile.ZipFile(out) as zf:
        members = zf.namelist()
        manifest = json.loads(zf.read(voice_backup.MANIFEST_NAME).decode("utf-8"))
    allowed = {voice_backup.MANIFEST_NAME, "README.txt"}
    strays = [m for m in members if m not in allowed and not m.startswith("voices/")]
    if strays:
        raise SystemExit(f"Refusing to ship: unexpected files in the voice bundle: {strays}")
    leaked = [k for entry in manifest["voices"] for k in entry
              if k not in {"name", "file", "transcript", "default"}]
    if leaked:
        raise SystemExit(f"Refusing to ship: unexpected keys in the voice manifest: {leaked}")
    return [(str(out), ".")]


voice_bundle = build_voice_bundle()

a = Analysis(
    [str(PROJECT / "main.py")],
    pathex=[str(PROJECT)],
    binaries=binaries,
    datas=datas + voice_bundle + [
        (str(PROJECT / "README.md"), "."),
        # Needed at run time too, not just for the .exe's own icon: Qt draws the
        # title-bar and taskbar icon from this file.
        (str(PROJECT / "packaging" / "assets" / "icon.ico"), "."),
    ],
    hiddenimports=hiddenimports + ["diareader", "diareader.gui"],
    hookspath=[],
    runtime_hooks=[
        str(Path(SPECPATH) / "rthook_recursion.py"),
        str(Path(SPECPATH) / "rthook_stdio.py"),
    ],
    # Gradio and its web stack come in through nari-tts but the app never serves a
    # page; matplotlib likewise arrives via descript-audiotools but is only reached by
    # optional plotting helpers, not by the import chain itself (unlike tensorboard,
    # see the collect_all list above). Dropping them saves well over a gigabyte of
    # things that would never be imported.
    excludes=[
        "gradio", "gradio_client", "fastapi", "starlette", "uvicorn",
        "matplotlib", "tkinter", "IPython", "jedi", "pytest",
    ],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz, a.scripts, [],
    exclude_binaries=True,
    name="DiaReader",
    debug=False,
    strip=False,
    upx=False,          # UPX on torch's DLLs is slow to pack and risks breaking them
    console=False,      # a GUI app: no console window behind the window
    icon=str(PROJECT / "packaging" / "assets" / "icon.ico"),
)
coll = COLLECT(
    exe, a.binaries, a.datas,
    strip=False,
    upx=False,
    name="DiaReader",
)
