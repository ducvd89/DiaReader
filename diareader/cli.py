"""Command-line entry point for smoke-testing the pipeline without the GUI.

Usage:
    python -m diareader.cli chapters mybook.epub
    python -m diareader.cli export mybook.epub out.mp3 --chapters 2-5
"""
from __future__ import annotations

import argparse
import logging
import sys

from . import config, pipeline
from .model_loader import CudaNotAvailableError, load_model
from .pipeline import export_book
from .tts_engine import TtsEngine


def _force_utf8_console() -> None:
    """A Windows console defaults to cp1252, which cannot encode Vietnamese — printing
    a chapter title would raise UnicodeEncodeError and kill the command. Since this app
    exists to read Vietnamese, UTF-8 output is not optional."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass


_force_utf8_console()
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("diareader.cli")


def _load_engine():
    settings = config.Settings.load()
    if settings.tts_engine == "dia" and not settings.hf_token:
        log.error(
            "No Hugging Face token found. Copy .env.example to .env and set HF_TOKEN, "
            "or set it in Settings."
        )
        sys.exit(1)
    try:
        if settings.tts_engine == "vieneu":
            from .vieneu_engine import load_vieneu
            engine = load_vieneu(settings)
        else:
            engine = TtsEngine(load_model(settings))
    except CudaNotAvailableError as e:
        log.error(str(e))
        sys.exit(1)
    return engine, settings


def _parse_chapters(spec: str | None) -> list[int] | None:
    """'3' or '2-5' (1-based, inclusive) -> 0-based indices. None means whole book."""
    if not spec:
        return None
    if "-" in spec:
        start, end = spec.split("-", 1)
        return list(range(int(start) - 1, int(end)))
    return [int(spec) - 1]


def cmd_chapters(args: argparse.Namespace) -> None:
    from .text_extract import load_book

    chapters = load_book(args.book)
    counts = pipeline.count_chunks_per_chapter(args.book, chapters=chapters)
    for i, (chapter, count) in enumerate(zip(chapters, counts), start=1):
        print(f"{i:>5}  {count:>6} sentences  {chapter.title}")
    print(f"\n{len(chapters)} chapters, {sum(counts)} sentences total")


def cmd_voices(args: argparse.Namespace) -> None:
    from . import voice

    settings = config.Settings.load()
    voices = voice.available_voices(settings)
    if not voices:
        print("No voices configured. The model's own default voice will be used.")
        print("Add one in the GUI under Settings -> Voices.")
        return
    for name, ref in sorted(voices.items()):
        marker = " (default)" if name == settings.default_voice else ""
        if isinstance(ref, voice.PresetVoice):
            print(f"{ref.preset} [{name}]{marker} — VieNeu preset")
        else:
            print(f"{name}{marker}")
            print(f"    wav:        {ref.wav_path}")
            print(f"    transcript: {ref.transcript}")
    print()
    print("Use a name with --voice, or switch mid-text with a marker like [GiongNu2].")


def cmd_export(args: argparse.Namespace) -> None:
    engine, settings = _load_engine()

    def on_progress(result):
        log.info(
            "Chapter %d (%s) chunk %d/%d generated",
            result.chapter_index + 1, result.chapter_title,
            result.chunk_index + 1, result.total_chunks_in_chapter,
        )

    out_path = export_book(
        args.book, args.out, engine, settings, on_progress=on_progress,
        chapter_indices=_parse_chapters(args.chapters),
        start_voice=args.voice,
    )
    log.info("Wrote %s", out_path)



def main() -> None:
    parser = argparse.ArgumentParser(prog="diareader")
    sub = parser.add_subparsers(dest="command", required=True)

    export_p = sub.add_parser("export", help="Export a book to WAV/MP3")
    export_p.add_argument("book")
    export_p.add_argument("out")
    export_p.add_argument(
        "--chapters", help="1-based chapter selection, e.g. 3 or 2-5 (default: whole book)"
    )
    export_p.add_argument("--voice", help="starting voice name (see the `voices` command)")
    export_p.set_defaults(func=cmd_export)

    list_p = sub.add_parser("chapters", help="List chapters and their sentence counts")
    list_p.add_argument("book")
    list_p.set_defaults(func=cmd_chapters)

    voices_p = sub.add_parser("voices", help="List configured reference voices")
    voices_p.set_defaults(func=cmd_voices)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
