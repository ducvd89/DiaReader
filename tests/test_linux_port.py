"""What changes when DiaReader runs on Linux rather than Windows."""
import sys

import pytest

from diareader import config, ffmpeg_dlls

posix_only = pytest.mark.skipif(sys.platform == "win32", reason="XDG/ld.so behaviour")


@posix_only
def test_app_dir_follows_xdg_data_home(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
    assert config._app_dir() == tmp_path / "DiaReader"


@posix_only
def test_app_dir_defaults_to_local_share_not_the_home_directory(tmp_path, monkeypatch):
    """The old fallback dropped a visible ~/DiaReader holding the model weights."""
    monkeypatch.delenv("XDG_DATA_HOME", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    assert config._app_dir() == tmp_path / ".local" / "share" / "DiaReader"


@posix_only
def test_relative_xdg_data_home_is_ignored(tmp_path, monkeypatch):
    """The XDG spec says relative values are invalid and must be ignored."""
    monkeypatch.setenv("XDG_DATA_HOME", "relative/dir")
    monkeypatch.setenv("HOME", str(tmp_path))
    assert config._app_dir() == tmp_path / ".local" / "share" / "DiaReader"


@posix_only
def test_install_hint_never_suggests_winget():
    assert "winget" not in ffmpeg_dlls.install_hint()


@posix_only
def test_dll_registration_is_a_no_op():
    """ld.so finds a distribution's FFmpeg on its own; nothing to register."""
    assert ffmpeg_dlls.register() is None
