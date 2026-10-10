"""The voices bundled with the pinned VieNeu SDK, including script-safe markers."""
from __future__ import annotations

import json
import logging
import re
import unicodedata
from dataclasses import dataclass
from functools import lru_cache
from importlib.resources import files

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Preset:
    name: str
    marker: str
    description: str
    aliases: tuple[str, ...] = ()


def _marker(name: str) -> str:
    ascii_name = unicodedata.normalize("NFKD", name.replace("đ", "d").replace("Đ", "D"))
    ascii_name = ascii_name.encode("ascii", "ignore").decode("ascii")
    slug = re.sub(r"[^A-Za-z0-9]+", "_", ascii_name).strip("_")
    return f"VieNeu_{slug}"


@lru_cache(maxsize=1)
def presets() -> tuple[Preset, ...]:
    """Read the SDK's shipped v3 Turbo voice assets; this is its voice source of truth."""
    try:
        path = files("vieneu").joinpath("assets", "voices_v3_turbo.json")
        data = json.loads(path.read_text(encoding="utf-8"))
    except (ImportError, OSError, ValueError) as exc:
        log.warning("Could not read VieNeu's preset voices: %s", exc)
        return ()
    result = tuple(
        Preset(name, _marker(name), entry.get("description", ""),
               tuple(entry.get("aliases") or ()))
        for name, entry in data.get("presets", {}).items()
    )
    if len({p.marker for p in result}) != len(result):
        raise ValueError("VieNeu presets have duplicate script markers")
    return result


def canonical_name(name: str) -> str | None:
    for preset in presets():
        if name == preset.name or name in preset.aliases:
            return preset.name
    return None


def marker_for_name(name: str) -> str | None:
    canonical = canonical_name(name)
    return next((p.marker for p in presets() if p.name == canonical), None)


def name_for_marker(marker: str) -> str | None:
    return next((p.name for p in presets() if p.marker == marker), None)
