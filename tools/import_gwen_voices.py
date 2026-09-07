"""Import Gwen-TTS's reference speakers into the voice library.

Gwen-TTS ships nine reference clips with their transcripts in ``data/ref_info.json``.
They are worth having whichever engine ends up doing the reading: this app's voice
contract is exactly Gwen's — a WAV plus the transcript of what is said in it — so a
voice imported here works for Dia too.

Names are prefixed ``Gwen`` for two reasons. Markers only match ``[A-Za-z0-9_-]``, so
``nsnd_ha_phuong`` has to be rewritten anyway; and a library that already contains a
hand-made ``HaPhuong`` must not have it silently replaced by someone else's clip of
the same artist. Import is additive — a name already taken is left alone.

    python tools/import_gwen_voices.py                    # fetch from GitHub
    python tools/import_gwen_voices.py --from ../gwen-tts # use a local checkout
    python tools/import_gwen_voices.py --list             # show what would be added
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import tempfile
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from diareader import config, voice  # noqa: E402

RAW_BASE = "https://raw.githubusercontent.com/ggroup-ai-lab/gwen-tts/main/"
REF_INFO = "data/ref_info.json"
NAME_PREFIX = "Gwen"

log = logging.getLogger("import_gwen_voices")


def _camel(key: str) -> str:
    """``nsnd_ha_phuong`` -> ``NsndHaPhuong``, the only shape a marker can reference."""
    return "".join(part.capitalize() for part in key.split("_") if part)


def _read(source: Path | None, relative: str) -> bytes:
    if source is not None:
        return (source / relative).read_bytes()
    with urllib.request.urlopen(RAW_BASE + relative) as response:
        return response.read()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--from", dest="source", type=Path, default=None,
        help="path to a gwen-tts checkout (default: download from GitHub)",
    )
    parser.add_argument(
        "--list", action="store_true", help="show what would be imported, change nothing"
    )
    args = parser.parse_args()

    speakers = json.loads(_read(args.source, REF_INFO).decode("utf-8"))
    settings = config.Settings.load()
    existing = set(voice.load_voices(settings))

    added, skipped = [], []
    with tempfile.TemporaryDirectory() as tmp:
        for key, entry in sorted(speakers.items()):
            name = NAME_PREFIX + _camel(key)
            transcript = entry["text"]
            if name in existing:
                skipped.append(name)
                continue
            if args.list:
                added.append(name)
                continue

            # Staged through a temp file because add_voice copies from a path, and the
            # source may be a URL. It normalises loudness and downmixes on the way in.
            staged = Path(tmp) / f"{name}.wav"
            staged.write_bytes(_read(args.source, entry["audio_path"]))
            voice.add_voice(settings, name, staged, transcript)
            added.append(name)

    verb = "would add" if args.list else "added"
    print(f"{verb}: {len(added)}")
    for name in added:
        print(f"  {name}")
    if skipped:
        print(f"already present, left alone: {', '.join(skipped)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
