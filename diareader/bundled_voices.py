"""Voices shipped inside the installer, installed into the library on first run.

Voices live in %APPDATA%, which an installer does not write, so a fresh install on
another machine started with an empty voice list however many voices the build
machine had. The build packs them into `bundled_voices.vbkp` (see
packaging/DiaReader.spec) and this installs them the first time the app runs.

It is the same `.vbkp` the Backup button writes, deliberately: one format to keep
working, and a build's bundled voices can be inspected — or restored by hand — with
nothing but the app's own Restore.

Only ever additive. A voice the user already has under the same name is left alone
rather than replaced, and a name they have taken for something else means the
bundled one arrives beside it under a suffix rather than overwriting their work.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

from . import config, voice, voice_backup

log = logging.getLogger(__name__)

BUNDLE_NAME = "bundled_voices.vbkp"
# Written once the bundled voices have been offered, so removing one does not just
# bring it back on the next launch.
_MARKER = "bundled_voices_installed"


def bundle_path() -> Path | None:
    """The packaged voice bundle, or None when running from source."""
    base = getattr(sys, "_MEIPASS", None)
    if base is None:
        return None
    candidate = Path(base) / BUNDLE_NAME
    return candidate if candidate.is_file() else None


def install_if_new(settings: "config.Settings") -> voice_backup.RestoreResult | None:
    """Install the bundled voices once. Returns what happened, or None if nothing did.

    Never raises: a broken bundle must not stop the app from opening. The voices are
    a convenience, and the app works with none configured at all — that is how this
    checkpoint was fine-tuned.
    """
    src = bundle_path()
    if src is None:
        return None

    marker = config.APP_DIR / _MARKER
    if marker.exists():
        return None

    try:
        existing = set(voice.load_voices(settings).keys())
        wanted = {name for name, _ in voice_backup.describe(src)}
        if wanted and wanted <= existing:
            # Already present — from a previous install, or carried over in an
            # %APPDATA% folder. Record that and do nothing.
            marker.parent.mkdir(parents=True, exist_ok=True)
            marker.touch()
            return None

        result = voice_backup.import_library(settings, src)
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.touch()
        log.info("Installed %d voices shipped with this build", result.total)
        return result
    except Exception:  # noqa: BLE001 — see docstring
        log.exception("Could not install the voices shipped with this build")
        return None
