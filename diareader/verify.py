"""Check generated audio by what is in it, instead of by how long it is.

Vietnamese is monosyllabic, so one syllable is one vowel nucleus and one clear peak
in the energy envelope. Counting those peaks and comparing against the syllable count
of the text answers the only question worth asking of a chunk — did the model say all
of it — with no model, no VRAM, and a millisecond of numpy.

Calibrated on the six human voice reference clips, whose exact syllable counts are
known from their transcripts, it recovers the true count within 7.1% (mean ratio
0.997, spread 0.91-1.08). It is checked in BOTH directions: too few nuclei means
dropped words, too many means the fragment-looping this decoder falls into.

This replaced two things. The duration check it supersedes was measured against
Whisper on 15 real sentences and found not to work: sentences it flagged as short
averaged 90.2% of their words actually present, sentences it passed averaged 88.9%,
and it passed a heading rendered as pure noise at 269% "coverage" — the counter scores
that same clip 2.50 and rejects it. Whisper itself was tried as the checker and
removed: it is far heavier than the question needs, and loading it beside Dia took the
app down with a CUDA OOM that kills the process rather than raising.
"""
from __future__ import annotations

import logging
import re

import numpy as np

from . import chunking, config

log = logging.getLogger(__name__)

_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)

# Acceptance band for (nuclei heard / syllables expected), used on ordinary sentences.
#
# The trade is asymmetric on purpose: a false alarm costs GPU time and nothing else,
# since the retry keeps the best attempt and the audio ends up no worse, while a missed
# shortfall is permanent and ships in the book. That argues for a tight bound — but
# only as far as the instrument can actually see. Calibrated with the parameters below,
# human speech reads 0.96-1.08 (mean 1.017, CV 4.4%); generated audio is messier, and
# of eight clips confirmed complete by transcription, one reads 0.83.
#
# The upper bound catches the other failure: a two-syllable heading rendered as noise
# measured 2.50, and the duration check it replaced called that same clip 269%
# "coverage" and passed it.
SYLLABLE_RATIO_MIN = 0.85
SYLLABLE_RATIO_MAX = 1.60

# Short text is judged by absolute count instead, because a ratio means nothing here.
# At three syllables one missed nucleus is already 33%, so any percentage strict enough
# to be useful on a sentence would condemn most headings — and at one syllable the old
# rule rejected two detected sounds as a "200% flood". Counting says what is actually
# meant: within one sound of what the text asks for. This also replaces the upper bound
# for short chunks, since ±1 constrains both directions.
SHORT_TEXT_SYLLABLES = 6
SHORT_TEXT_TOLERANCE = 1


def syllables(text: str) -> list[str]:
    """Vietnamese is monosyllabic: one whitespace-separated token is one syllable.

    Non-verbal tags are stripped first — they are performed, not spoken, so they must
    not be counted among the sounds the audio is expected to contain.
    """
    return _PUNCT.sub(" ", chunking.spoken_text(text).lower()).split()


def count_syllable_nuclei(audio: np.ndarray, sample_rate: int, *, hop_ms: float = 10.0,
                          win_ms: float = 25.0, floor_db: float = 20.0,
                          prominence_db: float = 3.0, min_gap_ms: float = 70.0) -> int:
    """Count syllable nuclei as prominent peaks in the energy envelope.

    Requiring each peak to rise `prominence_db` above the dips around it is what
    separates two syllables run together from one syllable's own wobble; `min_gap_ms`
    caps the rate at ~14 syllables/sec, comfortably above the 5.29/sec fastest
    measured across the reference clips.

    The defaults were swept against the six human reference clips rather than chosen:
    the first set (30ms window, 25dB floor, 90ms gap) read real speech at a mean ratio
    of 0.997 but with 7.1% spread and a worst clip at 0.91, which is too loose a ruler
    to hang a 0.90 threshold on. These read 1.017 with 4.4% spread and a worst clip at
    0.96. Tightening the instrument had to come before tightening the tolerance.
    """
    from scipy.signal import find_peaks

    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if audio.size == 0:
        return 0
    hop = max(1, int(sample_rate * hop_ms / 1000))
    win = max(hop, int(sample_rate * win_ms / 1000))
    n_frames = 1 + max(0, (audio.size - win) // hop)
    if n_frames < 3:
        return 0
    frames = np.lib.stride_tricks.as_strided(
        audio, shape=(n_frames, win),
        strides=(audio.strides[0] * hop, audio.strides[0]),
    )
    rms = np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1) + 1e-12)
    db = 20 * np.log10(rms)
    # Anything far below the loudest frame is silence between words, not a nucleus.
    db = np.maximum(db, db.max() - floor_db)
    peaks, _ = find_peaks(db, prominence=prominence_db,
                          distance=max(1, int(min_gap_ms / hop_ms)))
    return int(len(peaks))


def ratio(source_text: str, audio: np.ndarray,
          sample_rate: int = config.SAMPLE_RATE) -> float:
    """Sounds heard divided by syllables the text calls for. 1.0 is perfect."""
    expected = len(syllables(source_text))
    if expected == 0:
        return 1.0
    return count_syllable_nuclei(audio, sample_rate) / expected


class SyllableCounter:
    """Does the audio contain about as many sounds as the text calls for?

    Cannot say *which* words are missing — only that the count is wrong. That is
    enough for the failures this decoder actually has (dropping a clause, looping a
    fragment, emitting noise for a bare heading) and it costs nothing to run.
    """

    def check(self, source_text: str, audio: np.ndarray,
              sample_rate: int = config.SAMPLE_RATE) -> tuple[bool, float, str]:
        expected = len(syllables(source_text))
        if expected == 0:
            return True, 1.0, "nothing to say"
        heard = count_syllable_nuclei(audio, sample_rate)
        score = heard / expected
        if expected <= SHORT_TEXT_SYLLABLES:
            detail = (f"{heard} sounds for {expected} syllables "
                      f"(within {SHORT_TEXT_TOLERANCE})")
            return abs(heard - expected) <= SHORT_TEXT_TOLERANCE, score, detail
        detail = f"{heard} sounds for ~{expected} syllables"
        return SYLLABLE_RATIO_MIN <= score <= SYLLABLE_RATIO_MAX, score, detail

    def release(self) -> None:
        """Nothing to release — that is rather the point."""
