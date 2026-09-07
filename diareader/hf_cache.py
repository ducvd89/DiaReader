"""Repair what huggingface_hub hands back when its symlinks do not work here.

``hf_hub_download`` stores each file once as a blob and points at it from the snapshot
folder with a symlink, falling back to a copy when it cannot *create* one. On this
machine creation succeeds and resolution then fails — under
``%APPDATA%\\DiaReader\\model_cache`` a freshly made symlink, relative or absolute,
cannot be opened by anything: not Python, not PowerShell. The same code in
``~/.cache`` and in ``%TEMP%`` on the same drive works, and Windows reports
local-to-local symlink evaluation enabled, so this is something in that subtree rather
than a machine-wide setting.

The failure it produced was thoroughly misleading: the download reported success, the
returned path existed as a directory entry, and the load died several steps later with
"Could not parse the architecture config" — as though the file were corrupt rather
than unreadable.

It only appeared once huggingface_hub was pinned below 1.0 (a hard requirement of the
``transformers`` version Gwen-TTS needs); 1.x does not lay the cache out this way. So
the repair belongs here rather than in either loader: it is a property of the pinned
hub version meeting this filesystem, and it applies to every file either engine pulls.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

log = logging.getLogger(__name__)


def resolve_download(path: str | Path) -> Path:
    """The path hf_hub_download returned, or the blob it points at if it is broken.

    Returns the original path untouched in the normal case, so a working cache pays
    nothing for this. A broken link that cannot be resolved is also returned as-is:
    the caller's own error will be about the file it wanted, which is more use than
    one about symlinks.
    """
    resolved = Path(path)
    if resolved.exists():
        return resolved
    if not os.path.islink(resolved):
        return resolved

    target = Path(os.path.normpath(os.path.join(os.path.dirname(resolved),
                                                os.readlink(resolved))))
    if not target.exists():
        return resolved

    log.warning(
        "huggingface_hub returned an unreadable symlink at %s; reading the blob at %s "
        "directly. See diareader/hf_cache.py.", resolved, target,
    )
    return target
