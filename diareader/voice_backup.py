"""Save the voice library to one `.vbkp` file, and read it back.

A voice is only half a file: the WAV lives in the app folder while its transcript
lives in settings.json, and Dia needs both — the transcript conditions the encoder
while the WAV is prefilled into the decoder, so a clip without its exact transcript
makes the model drift rather than fail. Copying WAVs alone therefore loses the
voice. A `.vbkp` keeps the pair together.

It is an ordinary zip, so a backup stays openable by hand if this code ever stops
existing. Restoring adds to the library and never replaces: a name already in use is
given a suffix instead, because the alternative is silently overwriting a voice
someone recorded and cannot get back.
"""
from __future__ import annotations

import json
import logging
import tempfile
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

from . import config, voice

log = logging.getLogger(__name__)

SUFFIX = ".vbkp"
MANIFEST_NAME = "manifest.json"
FORMAT = "diareader-voices"
VERSION = 1

# Descriptive rather than functional — the app reads whatever is in the zip. Kept
# so a human opening a backup in five years can tell what they are looking at.
_README = (
    "DiaReader voice library backup.\n\n"
    "This is a zip file. manifest.json lists each voice with the transcript of its\n"
    "reference clip; the clips themselves are under voices/.\n\n"
    "Restore it from DiaReader under Settings -> Voices -> Restore.\n"
)


class BackupUnreadable(RuntimeError):
    """Raised when a file is not a voice backup this version can read."""


@dataclass
class RestoreResult:
    added: list[str] = field(default_factory=list)
    # (name in the backup, name it was given here) for voices whose name was taken.
    renamed: list[tuple[str, str]] = field(default_factory=list)
    skipped: list[tuple[str, str]] = field(default_factory=list)  # (name, why)

    @property
    def total(self) -> int:
        return len(self.added) + len(self.renamed)


def export_library(settings: "config.Settings", dest: str | Path) -> int:
    """Write every configured voice to `dest`. Returns how many were written."""
    dest = Path(dest)
    if dest.suffix.lower() != SUFFIX:
        dest = dest.with_suffix(SUFFIX)
    dest.parent.mkdir(parents=True, exist_ok=True)

    voices = voice.load_voices(settings)
    entries = []
    # ZIP_STORED, not DEFLATE: PCM in a zip compresses by a few percent and costs
    # seconds per clip. Not worth it for a file this size.
    with zipfile.ZipFile(dest, "w", zipfile.ZIP_STORED) as zf:
        for name, ref in voices.items():
            member = f"voices/{ref.wav_path.name}"
            zf.write(ref.wav_path, member)
            entries.append({
                "name": name,
                "file": member,
                "transcript": ref.transcript,
                "default": name == settings.default_voice,
            })
        zf.writestr(MANIFEST_NAME, json.dumps(
            {"format": FORMAT, "version": VERSION, "voices": entries},
            indent=2, ensure_ascii=False,
        ))
        zf.writestr("README.txt", _README)
    log.info("Wrote %d voices to %s", len(entries), dest)
    return len(entries)


def describe(src: str | Path) -> list[tuple[str, str]]:
    """(name, transcript) for each voice in a backup, without importing anything."""
    return [(e["name"], e.get("transcript", "")) for e in _read_manifest(src)[1]]


def _read_manifest(src: str | Path) -> tuple[Path, list[dict]]:
    src = Path(src)
    if not zipfile.is_zipfile(src):
        raise BackupUnreadable(
            f"{src.name} is not a DiaReader voice backup — it is not a zip file."
        )
    with zipfile.ZipFile(src) as zf:
        try:
            manifest = json.loads(zf.read(MANIFEST_NAME).decode("utf-8"))
        except KeyError as e:
            raise BackupUnreadable(
                f"{src.name} has no {MANIFEST_NAME}, so it is not a voice backup."
            ) from e
        except (UnicodeDecodeError, json.JSONDecodeError) as e:
            raise BackupUnreadable(f"{src.name} has a damaged {MANIFEST_NAME}: {e}") from e

    if manifest.get("format") != FORMAT:
        raise BackupUnreadable(
            f"{src.name} is a {manifest.get('format', 'unknown')!r} file, not a "
            f"DiaReader voice backup."
        )
    if manifest.get("version", 0) > VERSION:
        raise BackupUnreadable(
            f"{src.name} was written by a newer version of DiaReader "
            f"(format {manifest['version']}, this build reads {VERSION})."
        )
    entries = manifest.get("voices") or []
    if not isinstance(entries, list):
        raise BackupUnreadable(f"{src.name} lists its voices in a form this build cannot read.")
    return src, entries


def _free_name(taken: set[str], wanted: str) -> str:
    """`wanted`, or the first `wanted_2`, `wanted_3`… that is not already used."""
    if wanted not in taken:
        return wanted
    for n in range(2, 1000):
        candidate = f"{wanted}_{n}"
        if candidate not in taken:
            return candidate
    raise BackupUnreadable(f"Too many voices already named like {wanted!r}.")


def import_library(settings: "config.Settings", src: str | Path) -> RestoreResult:
    """Add every voice in `src` to the library, renaming rather than replacing."""
    src, entries = _read_manifest(src)
    result = RestoreResult()
    taken = set(voice.load_voices(settings).keys())

    with zipfile.ZipFile(src) as zf, tempfile.TemporaryDirectory() as tmp:
        for entry in entries:
            raw_name = str(entry.get("name", "")).strip()
            member = str(entry.get("file", ""))
            transcript = str(entry.get("transcript", ""))
            try:
                wanted = voice.validate_name(raw_name)
            except voice.InvalidVoiceName as e:
                result.skipped.append((raw_name or "(unnamed)", str(e)))
                continue

            try:
                data = zf.read(member)
            except KeyError:
                result.skipped.append((wanted, f"its audio ({member}) is missing from the backup"))
                continue

            # Written to a path this code chooses, never to the one in the archive:
            # a zip entry may name "../../anything", and extracting it verbatim
            # would write outside the folder entirely.
            staged = Path(tmp) / f"{wanted}{Path(member).suffix or '.wav'}"
            staged.write_bytes(data)

            name = _free_name(taken, wanted)
            try:
                voice.add_voice(settings, name, staged, transcript)
            except Exception as e:  # noqa: BLE001 — one bad clip must not lose the rest
                result.skipped.append((wanted, f"{type(e).__name__}: {e}"))
                continue

            taken.add(name)
            if name == wanted:
                result.added.append(name)
            else:
                result.renamed.append((wanted, name))

    log.info("Restored %d voices from %s (%d skipped)",
             result.total, src, len(result.skipped))
    return result
