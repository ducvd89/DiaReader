"""Stop child processes flashing a console window over the app.

A windowed build has no console of its own, so every child process Windows starts
for it gets a brand new one — which appears, sits on top of whatever the user is
doing, and vanishes. During an export that is not once but constantly: torch.compile
shells out to a C compiler per generated kernel (`triton/runtime/build.py` calls
``subprocess.check_call`` with no creation flags at all), and ffmpeg is run once per
export on top of that. The window steals focus each time, which makes the machine
unusable for anything else while a book renders.

None of those call sites are ours to fix — triton's and imageio-ffmpeg's are inside
third-party packages — so the flag is applied centrally, to ``subprocess.Popen``
itself. Everything that spawns a process goes through it.
"""
from __future__ import annotations

import logging
import subprocess
import sys

log = logging.getLogger(__name__)

# Value of subprocess.CREATE_NO_WINDOW, which only exists on Windows.
_CREATE_NO_WINDOW = 0x08000000
# Flags that mean "this child is supposed to have its own console". Adding
# CREATE_NO_WINDOW alongside either is contradictory, so leave those calls alone.
_WANTS_A_CONSOLE = 0x00000010 | 0x00000008  # CREATE_NEW_CONSOLE | DETACHED_PROCESS

_PATCHED = "_diareader_hides_consoles"


def hide_child_consoles() -> bool:
    """Make every subprocess start without a console window. True if now in effect."""
    if sys.platform != "win32":
        return False
    if getattr(subprocess.Popen, _PATCHED, False):
        return True

    original_init = subprocess.Popen.__init__

    def __init__(self, *args, **kwargs):  # noqa: N807 — replacing a dunder
        flags = kwargs.get("creationflags", 0)
        if not flags & _WANTS_A_CONSOLE:
            kwargs["creationflags"] = flags | _CREATE_NO_WINDOW
        original_init(self, *args, **kwargs)

    subprocess.Popen.__init__ = __init__
    setattr(subprocess.Popen, _PATCHED, True)
    log.debug("Child processes will start without a console window")
    return True
