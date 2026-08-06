"""Orchestrates chapters -> chunks -> audio generation, shared by export and playback."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Iterator

import numpy as np

from . import audio_io, chunking, config, text_extract, voice
from .tts_engine import TtsEngine

log = logging.getLogger(__name__)

INTER_CHUNK_SILENCE_S = 0.35
INTER_CHAPTER_SILENCE_S = 1.0

# This checkpoint sometimes samples an EOS token early and silently drops the rest of
# a chunk's text — the audio just comes out short, with no error. Measured over one
# 7-chunk run: six chunks covered 82-99% of their text and one covered 33%, at
# effectively the same chunk length, so it is a sampling accident rather than a length
# limit. Shrinking chunks lowers the odds but cannot remove them; the reliable fix is
# to measure coverage and regenerate the chunk when it comes up short.
_NATURAL_CHARS_PER_SECOND = 20.3  # measured on the fine-tune's own training clips
_COVERAGE_MIN_RATIO = 0.8
# Five rather than three. Now that a rejection means the sounds really are missing
# rather than merely that the audio came out shorter than a rough duration model
# predicted, spending another two generations on it is worth it — the failures are
# stochastic, and every one seen so far was recovered by regenerating.
_MAX_ATTEMPTS = 5
# Retries drop temperature: less randomness means less chance of another stray EOS.
_RETRY_TEMPERATURE = 1.0

# The chars/sec figure comes from 9-30s training clips, so it does not describe very
# short fragments — headings and table-of-contents lines carry fixed onset/offset
# overhead and legitimately vary a lot. Applying the ratio there produced constant
# false alarms and wasted retries on a real ebook's contents page. Below this length
# only a near-empty result counts as a failure.
_COVERAGE_MIN_CHARS = 40
_EMPTY_AUDIO_S = 0.15


def _spoken_len(chunk_text_: str) -> int:
    """Characters that should become audio — non-verbal tags are performed, not read."""
    return len(chunking.spoken_text(chunk_text_))


def _coverage(chunk_text_: str, audio: np.ndarray) -> float:
    expected = _spoken_len(chunk_text_) / _NATURAL_CHARS_PER_SECOND
    if expected <= 0:
        return 1.0
    return (len(audio) / config.SAMPLE_RATE) / expected


def _is_acceptable(chunk_text_: str, audio: np.ndarray) -> bool:
    seconds = len(audio) / config.SAMPLE_RATE
    # "Produced nothing at all" has to be judged against what the chunk was worth.
    # A flat 0.15s floor fails text too short to ever reach it: measured on a real
    # export, a one-character chunk returned 0.1s — 118% of the 0.049s its text is
    # worth — and was rejected three times in a row, spending three generations to
    # replace audio that was already over-length. Below that, the floor is the
    # chunk's own expected duration.
    expected = _spoken_len(chunk_text_) / _NATURAL_CHARS_PER_SECOND
    if seconds < min(_EMPTY_AUDIO_S, expected):
        return False
    if _spoken_len(chunk_text_) < _COVERAGE_MIN_CHARS:
        return True
    return _coverage(chunk_text_, audio) >= _COVERAGE_MIN_RATIO


def _generate_chunk(
    engine: TtsEngine,
    raw_chunk: str,
    voice_ref,
    chapter_title: str,
    chunk_index: int,
    verifier=None,
) -> np.ndarray:
    """Generate one chunk, retrying while the result looks wrong.

    With a `verifier`, "looks wrong" means a transcription of the audio is missing
    the words — the thing actually worth knowing. Without one it falls back to
    comparing the audio's duration against what the text should take, which was
    measured to be a poor proxy (see verify.py) but needs no extra model.
    """
    tagged = chunking.wrap_for_narration(raw_chunk)
    best_audio: np.ndarray | None = None
    best_score = -1.0

    for attempt in range(_MAX_ATTEMPTS):
        overrides = {} if attempt == 0 else {"temperature": _RETRY_TEMPERATURE}
        audio = engine.generate(tagged, voice_ref=voice_ref, **overrides)

        if verifier is not None and len(audio):
            ok, score, heard = verifier.check(raw_chunk, audio)
            detail = f"heard {heard[:60]!r}"
        else:
            score = _coverage(raw_chunk, audio)
            ok = _is_acceptable(raw_chunk, audio)
            detail = (f"{len(audio) / config.SAMPLE_RATE:.1f}s for "
                      f"{_spoken_len(raw_chunk)} spoken chars")

        if score > best_score:
            best_audio, best_score = audio, score
        if ok:
            return best_audio

        log.warning(
            "Sentence %d of %r scored %.0f%% (%s); regenerating (attempt %d/%d).",
            chunk_index + 1, chapter_title, score * 100, detail,
            attempt + 1, _MAX_ATTEMPTS,
        )

    if best_audio is None or len(best_audio) == 0:
        # Nothing usable at all. Emit a short silence and carry on rather than aborting
        # a long export over a single sentence, but say so loudly.
        log.error(
            "Sentence %d of %r produced no audio after %d attempts and was SKIPPED: %r",
            chunk_index + 1, chapter_title, _MAX_ATTEMPTS, raw_chunk[:60],
        )
        return audio_io.silence(0.2)

    if verifier is not None:
        log.error(
            "Sentence %d of %r: only %.0f%% of its words are audible after %d "
            "attempts — this one really is missing text.",
            chunk_index + 1, chapter_title, best_score * 100, _MAX_ATTEMPTS,
        )
    else:
        # Deliberately hedged. Measured against Whisper, sentences this check called
        # short averaged 90.2% of their words actually present — claiming words are
        # missing was, more often than not, wrong.
        log.warning(
            "Sentence %d of %r came out shorter than its text predicts (%.0f%%) after "
            "%d attempts. That is a weak signal — turn the sound-count check back on "
            "in Settings to judge it by what is actually in the audio.",
            chunk_index + 1, chapter_title, best_score * 100, _MAX_ATTEMPTS,
        )
    return best_audio


@dataclass
class ChunkResult:
    chapter_index: int
    chapter_title: str
    chunk_index: int
    total_chunks_in_chapter: int
    audio: np.ndarray
    voice_name: str | None = None


def iter_book_audio(
    book_path: str | Path,
    engine: TtsEngine,
    settings: config.Settings,
    max_chars: int = chunking.DEFAULT_MAX_CHARS,
    chapter_indices: Iterable[int] | None = None,
    start_voice: str | None = None,
    chapters: list[text_extract.Chapter] | None = None,
    speed: float | None = None,
    verifier=None,
) -> Iterator[ChunkResult]:
    """Yields a ChunkResult per sentence, generating audio as it goes.

    chapter_indices restricts the run to those chapters (in book order); None means
    the whole book. Real ebooks make selection essential rather than optional — a
    2467-chapter novel came to 255k sentences, which is thousands of hours of
    generation, so "just render the book" is not a usable default there.

    start_voice names the voice to begin with; inline markers like [GiongNu2] in the
    text switch it from that point on. With no voices configured at all, generation is
    unconditioned — which is how this checkpoint was fine-tuned (plain text -> audio,
    single-voice corpus, no audio prompts).
    """
    if chapters is None:
        chapters = text_extract.load_book(book_path)
    wanted = set(chapter_indices) if chapter_indices is not None else None

    available = voice.load_voices(settings)
    if available:
        log.info(
            "Voices available: %s (starting with %s)",
            ", ".join(sorted(available)), start_voice or settings.default_voice or "default",
        )
        # Before committing hours of GPU time: if the reference audio cannot be
        # decoded, every single sentence will fail the same way and the export
        # completes as a silent file. One probe is enough — they all go through the
        # same decoder.
        voice.check_reference_readable(next(iter(available.values())))

    if speed is None:
        speed = getattr(settings, "speed", 1.0) or 1.0

    current_voice_name = start_voice
    for chapter_index, chapter in enumerate(chapters):
        if wanted is not None and chapter_index not in wanted:
            continue

        voiced = chunking.chunk_text_with_voices(
            chapter.text, start_voice=current_voice_name, max_chars=max_chars
        )
        for chunk_index, chunk in enumerate(voiced):
            # Carry the active voice across chapters so a marker keeps applying.
            current_voice_name = chunk.voice
            voice_ref = voice.get_voice(settings, chunk.voice) if available else None
            audio = _generate_chunk(
                engine, chunk.text, voice_ref, chapter.title, chunk_index,
                verifier=verifier,
            )

            # Levelled per sentence, because normalising the reference clips was
            # necessary but not sufficient: with every reference at -20 LUFS, the
            # narration they produced still spanned 10.1 LU (NuMB2 came out at -13.9
            # against NuMN1's -19.1 from an identical prompt level). The model adds
            # its own per-voice offset, so the only place to fix the level for good is
            # on the way out.
            if getattr(settings, "normalize_output", True):
                audio = audio_io.normalize_loudness(
                    audio, config.SAMPLE_RATE, max_boost_db=audio_io.MAX_BOOST_DB
                )

            # Applied only after the coverage check, which compares audio length against
            # the text — stretching first would skew that measurement.
            audio = audio_io.change_speed(audio, speed)

            yield ChunkResult(
                chapter_index=chapter_index,
                chapter_title=chapter.title,
                chunk_index=chunk_index,
                total_chunks_in_chapter=len(voiced),
                audio=audio,
                voice_name=voice_ref.name if voice_ref else None,
            )


def count_chunks_per_chapter(
    book_path: str | Path,
    max_chars: int = chunking.DEFAULT_MAX_CHARS,
    chapters: list[text_extract.Chapter] | None = None,
) -> list[int]:
    """Sentence count for each chapter, for progress and selection. No model needed.

    Pass `chapters` if you already loaded them — parsing a 2467-chapter epub twice
    just to count is slow enough to notice.
    """
    if chapters is None:
        chapters = text_extract.load_book(book_path)
    return [len(chunking.chunk_text(c.text, max_chars=max_chars)) for c in chapters]


def count_chunks(
    book_path: str | Path,
    max_chars: int = chunking.DEFAULT_MAX_CHARS,
    chapter_indices: Iterable[int] | None = None,
    chapters: list[text_extract.Chapter] | None = None,
) -> int:
    per_chapter = count_chunks_per_chapter(book_path, max_chars=max_chars, chapters=chapters)
    if chapter_indices is None:
        return sum(per_chapter)
    return sum(per_chapter[i] for i in chapter_indices if 0 <= i < len(per_chapter))


def _make_verifier(settings: config.Settings, stage=None):
    """Build the checker named by settings.verify_mode, or None for the length check."""
    mode = getattr(settings, "verify_mode", "syllables")

    def say(message: str) -> None:
        log.info(message)
        if stage:
            stage(message)

    if mode == "syllables":
        from .verify import SyllableCounter

        say("Checking each sentence by counting the sounds in it.")
        return SyllableCounter()
    return None


class ExportCancelled(RuntimeError):
    """Raised when a caller's should_continue() returns False mid-export."""


def export_book(
    book_path: str | Path,
    out_path: str | Path,
    engine: TtsEngine,
    settings: config.Settings,
    on_progress: Callable[[ChunkResult], None] | None = None,
    should_continue: Callable[[], bool] | None = None,
    chapter_indices: Iterable[int] | None = None,
    start_voice: str | None = None,
    on_stage: Callable[[str], None] | None = None,
    chapters: list[text_extract.Chapter] | None = None,
    speed: float | None = None,
) -> Path:
    """Generate the selected chapters (or the whole book) into a WAV/MP3 file.

    should_continue is polled between chunks so a GUI can cancel a long export;
    the partial WAV is left on disk rather than deleted. on_stage reports the phases
    that happen outside the per-sentence loop (parsing, transcoding), which on a large
    ebook take long enough that silence looks like a hang.
    """
    out_path = Path(out_path)
    wav_path = out_path if out_path.suffix.lower() == ".wav" else out_path.with_suffix(".wav")

    def stage(message: str) -> None:
        log.info(message)
        if on_stage:
            on_stage(message)

    if chapters is None:
        stage("Reading the book…")
        chapters = text_extract.load_book(book_path)

    verifier = _make_verifier(settings, stage)

    stage(f"{len(chapters)} chapters ready; generating audio…")

    with audio_io.StreamingWavWriter(wav_path) as writer:
        prev_chapter = None
        for result in iter_book_audio(
            book_path, engine, settings,
            chapter_indices=chapter_indices,
            start_voice=start_voice,
            chapters=chapters,
            speed=speed,
            verifier=verifier,
        ):
            if should_continue is not None and not should_continue():
                raise ExportCancelled(f"Export cancelled; partial audio kept at {wav_path}")
            if prev_chapter is not None and result.chapter_index != prev_chapter:
                writer.write(audio_io.silence(INTER_CHAPTER_SILENCE_S))
            elif result.chunk_index > 0:
                writer.write(audio_io.silence(INTER_CHUNK_SILENCE_S))
            writer.write(result.audio)
            prev_chapter = result.chapter_index
            if on_progress:
                on_progress(result)

    if out_path.suffix.lower() == ".mp3":
        stage("Converting to MP3…")
        audio_io.transcode_to_mp3(wav_path, out_path)
        return out_path
    return wav_path
