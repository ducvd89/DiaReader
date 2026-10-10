"""Make FFmpeg's shared libraries findable by torchcodec, on Windows.

``torchcodec`` loads ``libtorchcodec_core*.dll``, which links against FFmpeg's shared
libraries (``avutil-*.dll`` and friends). Installing a shared FFmpeg build puts those
DLLs on PATH — and that is not enough. Since Python 3.8, Windows extension modules are
loaded with ``LOAD_LIBRARY_SEARCH_DEFAULT_DIRS``, which deliberately ignores PATH when
resolving a DLL's *dependencies*; only directories registered through
``os.add_dll_directory`` are searched. So torchcodec walks core9 down to core4, every
one fails with "or one of its dependencies", and the user is told to install the FFmpeg
they already have.

Registering the directory here fixes it for every entry point, as long as this runs
before anything imports torchcodec — hence the call in ``diareader/__init__.py``.
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

log = logging.getLogger(__name__)

#: Set this to a directory holding avutil-*.dll to override the search entirely.
ENV_OVERRIDE = "DIAREADER_FFMPEG_BIN"

# os.add_dll_directory returns a handle that un-registers the directory when it is
# closed — and it closes on garbage collection. Keeping it alive here is what makes
# the registration last for the life of the process, so this is not a stray global.
_handle = None
_registered_dir: Path | None = None


def _has_ffmpeg_dlls(directory: Path) -> bool:
    """A shared build is identified by avutil, the library every other one needs.

    Checked by glob rather than by exact name because the soname carries FFmpeg's
    major version: avutil-56 (FFmpeg 4) through avutil-61 (FFmpeg 9) are all valid,
    and torchcodec tries each of its core libraries until one matches.
    """
    try:
        return any(directory.glob("avutil-*.dll"))
    except OSError:
        return False


def _is_vendored(directory: Path) -> bool:
    """True for another Python package's private FFmpeg bundle.

    PySide6 ships a complete FFmpeg 7 for QtMultimedia, and importing it puts that
    directory on PATH — before ``import diareader`` runs, since main.py imports Qt
    first. torchcodec does load against it, so this is not a failure, but Qt's
    private bundle is not ours to depend on: it is trimmed (no avdevice, no
    avfilter) and its version moves with the Qt wheel. Kept as a last resort rather
    than dropped, because a machine with only that is better served than not.
    """
    return "site-packages" in {part.lower() for part in directory.parts}


def _candidate_dirs() -> list[Path]:
    """Every plausible location, best first, without touching the filesystem yet."""
    override: list[Path] = []
    winget: list[Path] = []
    on_path: list[Path] = []
    vendored: list[Path] = []

    configured = os.environ.get(ENV_OVERRIDE)
    if configured:
        override.append(Path(configured))

    # winget writes PATH into the registry, so a shell (and any app launched from it)
    # started before the install keeps the old value until it is restarted — the one
    # case where the package is installed, correct, and invisible to a PATH scan.
    # Ranked above PATH as well as searched separately: it is the build the README
    # tells people to install, so when both exist it is the one they meant.
    local_appdata = os.environ.get("LOCALAPPDATA")
    if local_appdata:
        packages = Path(local_appdata) / "Microsoft" / "WinGet" / "Packages"
        try:
            for package in packages.glob("Gyan.FFmpeg*"):
                winget.extend(package.glob("*/bin"))
        except OSError:
            pass

    # PATH is where a shared build normally lands, and it is what makes a manual or
    # scoop/choco install work without any special-casing. It is searched despite
    # being useless to the DLL *loader*: as a list of directories to hand to
    # add_dll_directory it is exactly right.
    for entry in os.environ.get("PATH", "").split(os.pathsep):
        if entry.strip():
            directory = Path(entry)
            (vendored if _is_vendored(directory) else on_path).append(directory)

    return override + winget + on_path + vendored


def find_ffmpeg_dir() -> Path | None:
    """First candidate directory that actually holds the shared libraries."""
    seen: set[str] = set()
    for candidate in _candidate_dirs():
        key = str(candidate).rstrip(r"\/").lower()
        if key in seen:
            continue
        seen.add(key)
        if _has_ffmpeg_dlls(candidate):
            return candidate
    return None


def register() -> Path | None:
    """Register FFmpeg's directory with the DLL loader. Returns it, or None.

    Safe to call repeatedly and safe to call when FFmpeg is missing: a voice-less
    export never touches torchcodec, so a failure here must not stop the app from
    starting. ``voice.check_reference_readable`` is what turns it into an error, and
    only for the users it actually affects.
    """
    global _handle, _registered_dir

    if sys.platform != "win32":
        return None
    if _registered_dir is not None:
        return _registered_dir

    directory = find_ffmpeg_dir()
    if directory is None:
        log.debug("No FFmpeg shared build found; voice conditioning will not work.")
        return None

    try:
        _handle = os.add_dll_directory(str(directory))
    except OSError as e:
        log.warning("Found FFmpeg at %s but could not register it: %s", directory, e)
        return None

    _registered_dir = directory
    log.debug("Registered FFmpeg shared libraries from %s", directory)
    return directory


def registered_dir() -> Path | None:
    """The directory actually handed to the DLL loader, or None if there was none.

    Not the same as calling ``find_ffmpeg_dir`` again: registration happens once at
    import, and PATH keeps changing afterwards — importing PySide6 adds its own
    bundled FFmpeg to it. A report that re-searched would name a directory the loader
    was never told about.
    """
    return _registered_dir


def system_library() -> str | None:
    """Outside Windows: the libavutil the dynamic loader will hand torchcodec, or None.

    None of the above is needed there — ld.so resolves an extension's dependencies
    through the ordinary library path, so a distribution's FFmpeg package is simply
    found. This only answers "is there one", for the selftest and error messages.
    """
    if sys.platform == "win32":
        return None
    import ctypes.util

    return ctypes.util.find_library("avutil")


def install_hint() -> str:
    """The command that installs FFmpeg's shared libraries on this platform."""
    if sys.platform == "win32":
        return "winget install Gyan.FFmpeg.Shared"
    if Path("/etc/arch-release").exists():
        return "sudo pacman -S ffmpeg"
    return "install FFmpeg's shared libraries from your distribution (libavutil and friends)"
