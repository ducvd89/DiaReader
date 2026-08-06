"""Check an exported audiobook for sentences that lost words.

    python tools/check_export.py out.wav book.epub 5 [--speed 0.9]

This model can drop words mid-sentence while still producing audio of plausible length.
An earlier check that only flagged near-silent output called such a run clean; three
sentences in it were audibly missing words. So coverage is measured per sentence against
the source text, and near-silence is only the most extreme case of it.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import soundfile as sf

from diareader import chunking, pipeline
from diareader.text_extract import load_book


def split_on_gaps(audio: np.ndarray, sr: int, min_gap_s: float = 0.3) -> list[tuple[int, int]]:
    """Split at the exact-zero runs the pipeline writes between sentences."""
    zeros = audio == 0.0
    edges = np.flatnonzero(np.diff(np.concatenate(([0], zeros.view(np.int8), [0]))))
    gaps = [(edges[i], edges[i + 1]) for i in range(0, len(edges), 2)
            if (edges[i + 1] - edges[i]) >= min_gap_s * sr]
    segments, prev = [], 0
    for a, b in gaps:
        segments.append((prev, a))
        prev = b
    segments.append((prev, len(audio)))
    return [(a, b) for a, b in segments if b > a]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wav")
    parser.add_argument("book")
    parser.add_argument("chapter", type=int, help="1-based chapter number that was exported")
    parser.add_argument("--speed", type=float, default=1.0,
                        help="tempo the export used, so durations can be un-stretched")
    parser.add_argument("--threshold", type=float, default=0.8)
    args = parser.parse_args()

    audio, sr = sf.read(args.wav)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    segments = split_on_gaps(audio, sr)

    chapters = load_book(args.book)
    sentences = chunking.chunk_text(chapters[args.chapter - 1].text)

    print(f"chapter   : {chapters[args.chapter - 1].title}")
    print(f"sentences : {len(sentences)}   segments found: {len(segments)}")
    if len(segments) != len(sentences):
        print("  WARNING: counts differ; sentence alignment below may be off.")
    print()

    rows = []
    for i, (a, b) in enumerate(segments):
        if i >= len(sentences):
            break
        # Undo the tempo stretch so coverage compares against generated length.
        seconds = (b - a) / sr * args.speed
        chars = len(sentences[i])
        expected = chars / pipeline._NATURAL_CHARS_PER_SECOND
        rows.append((seconds / expected if expected else 1.0, i + 1, seconds, chars, sentences[i]))

    rows.sort()
    bad = [r for r in rows if r[0] < args.threshold]
    if bad:
        print(f"{'cov':>5}  {'#':>4}  {'audio':>6}  {'chars':>5}  text")
        for cov, i, seconds, chars, text in bad:
            print(f"{cov*100:4.0f}%  #{i:<4} {seconds:5.1f}s  {chars:5}  {text[:60]}")
        print()
    print(f"below {args.threshold*100:.0f}%: {len(bad)}/{len(rows)} sentences")
    if bad:
        print("\nThese are the ones to listen to — expect missing words, not silence.")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
