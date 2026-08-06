"""Turn a chapter title into something Windows will actually accept as a filename.

Export names are suggested from the book or chapter title, and Vietnamese chapter
headings routinely contain a colon — "Chương 2: Thanh Ngưu Trấn". Windows forbids
`:` in a filename, so the save dialog rejected the very name the app had just
proposed, with an error that pointed at the user rather than at the suggestion.

Diacritics are left alone: NTFS stores Unicode, and stripping them would mangle
every Vietnamese title for no gain.
"""
from __future__ import annotations

import re

# Windows reserves these outright, in any Unicode filename.
_FORBIDDEN = r'<>:"/\\|?*'
_FORBIDDEN_RE = re.compile(f"[{re.escape(_FORBIDDEN)}\x00-\x1f]")
_WHITESPACE_RE = re.compile(r"\s+")

# Reserved device names, matched without their extension and case-insensitively.
# "CON.mp3" is not a file Windows will create.
_RESERVED = {
    "CON", "PRN", "AUX", "NUL",
    *(f"COM{i}" for i in range(1, 10)),
    *(f"LPT{i}" for i in range(1, 10)),
}

# Long enough for any real chapter heading, short enough to leave room for the
# directory and extension inside a 260-character path.
_MAX_STEM = 120


def safe_filename(name: str, fallback: str = "export") -> str:
    """A filename stem that Windows will accept, as close to `name` as possible.

    Only the stem — pass it without an extension and add one afterwards, or a `.`
    in the title would be mistaken for the separator.
    """
    cleaned = _FORBIDDEN_RE.sub(" ", name or "")
    cleaned = _WHITESPACE_RE.sub(" ", cleaned).strip()

    # Windows silently drops trailing dots and spaces, so a name ending in one
    # resolves to something other than what was asked for.
    cleaned = cleaned.rstrip(" .")

    if len(cleaned) > _MAX_STEM:
        cleaned = cleaned[:_MAX_STEM].rstrip(" .")

    if not cleaned:
        return fallback
    if cleaned.upper() in _RESERVED:
        return f"{cleaned}_"
    return cleaned
