"""The regression these guard against: FFmpeg installed, on PATH, and still unusable.

Since Python 3.8 Windows resolves an extension module's DLL *dependencies* through
``os.add_dll_directory`` only, ignoring PATH — so torchcodec failed to load its core
libraries on a machine that had exactly the FFmpeg build the error message told the
user to install.
"""
import os
import sys

import pytest

from diareader import ffmpeg_dlls


def _make_shared_build(directory, soname="avutil-61.dll"):
    """A directory that looks like a shared FFmpeg build to the finder."""
    directory.mkdir(parents=True, exist_ok=True)
    (directory / soname).write_bytes(b"")
    return directory


def test_finds_a_build_on_path(tmp_path, monkeypatch):
    ffmpeg_bin = _make_shared_build(tmp_path / "ffmpeg" / "bin")
    monkeypatch.delenv(ffmpeg_dlls.ENV_OVERRIDE, raising=False)
    monkeypatch.setenv("PATH", str(tmp_path / "nothing-here") + os.pathsep + str(ffmpeg_bin))
    monkeypatch.delenv("LOCALAPPDATA", raising=False)

    assert ffmpeg_dlls.find_ffmpeg_dir() == ffmpeg_bin


def test_ignores_directories_without_the_shared_libraries(tmp_path, monkeypatch):
    """winget puts a shim ``ffmpeg.exe`` on PATH in a directory holding no DLLs at
    all. Accepting it would register the wrong directory and change nothing."""
    shim_dir = tmp_path / "WinGet" / "Links"
    shim_dir.mkdir(parents=True)
    (shim_dir / "ffmpeg.exe").write_bytes(b"")

    monkeypatch.delenv(ffmpeg_dlls.ENV_OVERRIDE, raising=False)
    monkeypatch.setenv("PATH", str(shim_dir))
    monkeypatch.delenv("LOCALAPPDATA", raising=False)

    assert ffmpeg_dlls.find_ffmpeg_dir() is None


def test_finds_the_winget_package_when_path_is_stale(tmp_path, monkeypatch):
    """winget writes PATH to the registry, so a shell — and any app launched from it
    — started before the install keeps the old value. The package is installed and
    correct, and invisible to a PATH scan, which is exactly how this was first hit."""
    package = (tmp_path / "Microsoft" / "WinGet" / "Packages"
               / "Gyan.FFmpeg.Shared_Microsoft.Winget.Source_8wekyb3d8bbwe"
               / "ffmpeg-9.0.1-full_build-shared" / "bin")
    _make_shared_build(package)

    monkeypatch.delenv(ffmpeg_dlls.ENV_OVERRIDE, raising=False)
    monkeypatch.setenv("PATH", "")
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))

    assert ffmpeg_dlls.find_ffmpeg_dir() == package


def test_prefers_a_real_install_over_a_bundle_inside_site_packages(tmp_path, monkeypatch):
    """Importing PySide6 puts its own FFmpeg 7 on PATH, and main.py imports Qt before
    diareader — so without ranking, Qt's private bundle wins on every GUI launch. It
    does work, which is why it stays as a fallback, but it is trimmed and its version
    moves with the Qt wheel."""
    qt_bundle = _make_shared_build(
        tmp_path / "venv" / "Lib" / "site-packages" / "PySide6", soname="avutil-59.dll"
    )
    real = _make_shared_build(tmp_path / "ffmpeg" / "bin")

    monkeypatch.delenv(ffmpeg_dlls.ENV_OVERRIDE, raising=False)
    monkeypatch.setenv("PATH", str(qt_bundle) + os.pathsep + str(real))
    monkeypatch.delenv("LOCALAPPDATA", raising=False)

    assert ffmpeg_dlls.find_ffmpeg_dir() == real


def test_uses_a_vendored_bundle_when_it_is_all_there_is(tmp_path, monkeypatch):
    qt_bundle = _make_shared_build(
        tmp_path / "venv" / "Lib" / "site-packages" / "PySide6", soname="avutil-59.dll"
    )

    monkeypatch.delenv(ffmpeg_dlls.ENV_OVERRIDE, raising=False)
    monkeypatch.setenv("PATH", str(qt_bundle))
    monkeypatch.delenv("LOCALAPPDATA", raising=False)

    assert ffmpeg_dlls.find_ffmpeg_dir() == qt_bundle


def test_winget_package_outranks_path(tmp_path, monkeypatch):
    """Both installed: the winget one is what the README told the user to install,
    so it is the one they meant."""
    package = (tmp_path / "Microsoft" / "WinGet" / "Packages" / "Gyan.FFmpeg.Shared_x"
               / "ffmpeg-9.0.1-full_build-shared" / "bin")
    _make_shared_build(package)
    other = _make_shared_build(tmp_path / "elsewhere" / "bin", soname="avutil-56.dll")

    monkeypatch.delenv(ffmpeg_dlls.ENV_OVERRIDE, raising=False)
    monkeypatch.setenv("PATH", str(other))
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))

    assert ffmpeg_dlls.find_ffmpeg_dir() == package


def test_override_wins_over_path(tmp_path, monkeypatch):
    on_path = _make_shared_build(tmp_path / "old" / "bin")
    chosen = _make_shared_build(tmp_path / "chosen" / "bin", soname="avutil-59.dll")

    monkeypatch.setenv(ffmpeg_dlls.ENV_OVERRIDE, str(chosen))
    monkeypatch.setenv("PATH", str(on_path))
    monkeypatch.delenv("LOCALAPPDATA", raising=False)

    assert ffmpeg_dlls.find_ffmpeg_dir() == chosen


def test_missing_ffmpeg_is_not_an_error(tmp_path, monkeypatch):
    """An export with no voices never touches torchcodec, so a missing FFmpeg must
    not stop the app from starting. voice.check_reference_readable is what turns it
    into an error, and only for the users it affects."""
    monkeypatch.delenv(ffmpeg_dlls.ENV_OVERRIDE, raising=False)
    monkeypatch.setenv("PATH", str(tmp_path / "nothing-here"))
    monkeypatch.delenv("LOCALAPPDATA", raising=False)
    monkeypatch.setattr(ffmpeg_dlls, "_registered_dir", None)

    assert ffmpeg_dlls.register() is None


@pytest.mark.skipif(sys.platform != "win32", reason="add_dll_directory is Windows-only")
def test_register_is_idempotent_and_keeps_the_handle_alive(tmp_path, monkeypatch):
    """The handle add_dll_directory returns un-registers the directory when it is
    closed, and it closes on garbage collection — so dropping it would undo the fix
    at an arbitrary later moment."""
    ffmpeg_bin = _make_shared_build(tmp_path / "ffmpeg" / "bin")
    monkeypatch.setenv(ffmpeg_dlls.ENV_OVERRIDE, str(ffmpeg_bin))
    monkeypatch.setattr(ffmpeg_dlls, "_registered_dir", None)
    monkeypatch.setattr(ffmpeg_dlls, "_handle", None)

    assert ffmpeg_dlls.register() == ffmpeg_bin
    assert ffmpeg_dlls._handle is not None

    handle = ffmpeg_dlls._handle
    assert ffmpeg_dlls.register() == ffmpeg_bin
    assert ffmpeg_dlls._handle is handle, "a second call must not re-register"
