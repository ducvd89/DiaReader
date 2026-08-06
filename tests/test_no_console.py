"""Child processes must not flash a console window over the app.

The call sites that caused it are in third-party code — triton runs a C compiler per
generated kernel with no creation flags at all, and ffmpeg is spawned per export — so
the flag is applied to subprocess.Popen itself and these check it lands there.
"""
import subprocess
import sys

import pytest

from diareader import no_console

CREATE_NO_WINDOW = 0x08000000
CREATE_NEW_CONSOLE = 0x00000010
DETACHED_PROCESS = 0x00000008

pytestmark = pytest.mark.skipif(
    sys.platform != "win32", reason="console windows are a Windows problem"
)


@pytest.fixture
def spawns(monkeypatch):
    """Capture the kwargs Popen would have been constructed with."""
    seen = {}

    def fake_init(self, *args, **kwargs):
        seen.update(kwargs)

    monkeypatch.setattr(subprocess.Popen, "__init__", fake_init)
    # The patch records itself on the class; clear it so each test re-applies to the
    # fake above rather than skipping as already-done.
    monkeypatch.delattr(subprocess.Popen, "_diareader_hides_consoles", raising=False)
    no_console.hide_child_consoles()
    return seen


def test_a_plain_spawn_gets_the_no_window_flag(spawns):
    subprocess.Popen(["cmd", "/c", "echo hi"])
    assert spawns["creationflags"] & CREATE_NO_WINDOW


def test_existing_flags_are_kept(spawns):
    subprocess.Popen(["cmd"], creationflags=subprocess.HIGH_PRIORITY_CLASS)
    assert spawns["creationflags"] & subprocess.HIGH_PRIORITY_CLASS
    assert spawns["creationflags"] & CREATE_NO_WINDOW


@pytest.mark.parametrize("flag", [CREATE_NEW_CONSOLE, DETACHED_PROCESS])
def test_a_caller_that_asked_for_its_own_console_is_left_alone(spawns, flag):
    """Adding CREATE_NO_WINDOW next to these is contradictory, and Windows rejects
    the combination rather than quietly picking one."""
    subprocess.Popen(["cmd"], creationflags=flag)
    assert not spawns["creationflags"] & CREATE_NO_WINDOW
    assert spawns["creationflags"] & flag


def test_applying_it_twice_does_not_stack_wrappers(spawns):
    assert no_console.hide_child_consoles() is True
    assert no_console.hide_child_consoles() is True
    subprocess.Popen(["cmd"])
    assert spawns["creationflags"] == CREATE_NO_WINDOW


def test_the_real_thing_still_runs_and_returns_output():
    """The flag must not break ordinary capture — ffmpeg's output is read this way."""
    no_console.hide_child_consoles()
    out = subprocess.run(
        [sys.executable, "-c", "print('xin chao')"],
        capture_output=True, text=True, check=True,
    )
    assert out.stdout.strip() == "xin chao"
