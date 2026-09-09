"""DiaReader — Vietnamese ebook-to-audiobook conversion.

Importing the package registers FFmpeg's DLL directory. It has to happen before
anything pulls in torchcodec, and every entry point — the GUI, the CLI, the selftest,
the tests — starts by importing something from here, so this is the one place that
runs early enough for all of them. See ffmpeg_dlls for why PATH alone does not work.
"""
from __future__ import annotations

from . import ffmpeg_dlls as _ffmpeg_dlls

#: The one place the version is written. packaging/build.ps1 reads it from here and
#: passes it to Inno Setup, so the installer filename and the app can never disagree
#: — which they would the first time someone bumped one and forgot the other.
__version__ = "1.2.2"

_ffmpeg_dlls.register()
