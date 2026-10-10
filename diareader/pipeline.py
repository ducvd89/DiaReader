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
# Ten rather than five. A rejection means the sounds really are missing rather than
# merely that the audio came out shorter than a rough duration model predicted, and
# the failures are stochastic, so more draws genuinely do recover more sentences.
#
# Ten costs ten on a sentence that fails every time, and nothing at all on the ordinary
# one that passes first go. The band it is judged against is narrow enough (see
# asr_verify.RATIO_MIN) that a sentence reaching the tenth attempt is genuinely wrong
# rather than merely mis-measured.
_MAX_ATTEMPTS = 10
# Retries drop temperature: less randomness means less chance of another stray EOS.
_RETRY_TEMPERATURE = 1.0

# There was a rule here that stopped early once a score repeated, on the grounds that
# the checker was unreliable in a band just below its acceptance bar. It was retired
# with the checker it described: the recogniser that replaced peak counting reads the
# reference clips at mean 1.000 with 2.8% spread, so there is no band of "probably
# fine, probably mis-measured" left to excuse. The budget below is spent in full.

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


def _distance(score: float) -> float:
    """How wrong a score is. Lower is better; a perfect reading scores 1.0.

    Not "higher is better", which is what this used to be and which quietly preferred
    the worst failure mode the checker has. The acceptance band is two-sided: below it
    the model dropped words, above it the model looped a fragment or emitted noise —
    verify.py records a heading rendered as pure noise measuring 2.50. Ranking by size
    alone made that clip beat every honest attempt, so a sentence that failed once by
    looping and then read perfectly would ship the loop.
    """
    if score < 0:
        return float("inf")
    return abs(score - 1.0)


def _judge(raw_chunk: str, audio: np.ndarray, verifier) -> tuple[bool, float, str]:
    """(acceptable, score, why) for one generated chunk.

    Split out of _generate_chunk so that whatever decides a sentence is good stays one
    thing, wherever it is asked from.
    """
    if verifier is not None and len(audio):
        # Passed through untouched. This used to re-wrap and truncate it —
        # f"heard {detail[:60]!r}" — which printed "heard" twice and cut the counts off
        # the end, so a log line read `heard "heard 'anh vui lòng làm theo các b` and
        # lost the "16 of 20 syllables" that says what actually went wrong.
        return verifier.check(raw_chunk, audio)
    score = _coverage(raw_chunk, audio)
    detail = (f"{len(audio) / config.SAMPLE_RATE:.1f}s for "
              f"{_spoken_len(raw_chunk)} spoken chars")
    return _is_acceptable(raw_chunk, audio), score, detail


@dataclass
class Attempt:
    """What one sentence's generation produced, and what the checker made of it.

    Returned rather than just the audio because the sentence cache and the review
    dialog both want the judgement: a take that was accepted at 100% and one that was
    kept after ten failures look identical as arrays, and the difference is the whole
    reason someone opens the review table.
    """

    audio: np.ndarray
    score: float = 1.0
    transcript: str = ""
    accepted: bool = True

    def __len__(self) -> int:
        return len(self.audio)


def _generate_chunk(
    engine: TtsEngine,
    raw_chunk: str,
    voice_ref,
    chapter_title: str,
    chunk_index: int,
    verifier=None,
    initial: np.ndarray | None = None,
) -> Attempt:
    """Generate one chunk, retrying while the result looks wrong.

    With a `verifier`, "looks wrong" means a transcription of the audio is missing
    the words — the thing actually worth knowing. Without one it falls back to
    comparing the audio's duration against what the text should take, which was
    measured to be a poor proxy (see verify.py) but needs no extra model.

    `initial` is audio already generated for this chunk, counted as the first attempt.
    Nothing passes it today; it is what let a batched generation hand its result in
    when there was a second engine that generated several sentences at once.

    One way out short of the full budget: a result the verifier accepts. There was a
    second — a score that had stopped changing — removed once the band it relied on
    turned out to hold no known-good speech.
    """
    tagged = chunking.wrap_for_narration(raw_chunk)
    best_audio: np.ndarray | None = None
    best_score = -1.0
    best_transcript = ""
    attempts_left = _MAX_ATTEMPTS

    if initial is not None:
        best_audio, best_score = initial, _judge(raw_chunk, initial, verifier)[1]
        attempts_left -= 1

    for attempt in range(attempts_left):
        # A retry only differs from the first attempt in temperature, and a chunk
        # arriving with `initial` has already had its temperature-0.3 turn.
        overrides = ({} if attempt == 0 and initial is None
                     else {"temperature": _RETRY_TEMPERATURE})
        audio = engine.generate(tagged, voice_ref=voice_ref, **overrides)
        ok, score, detail = _judge(raw_chunk, audio, verifier)
        attempt_number = _MAX_ATTEMPTS - attempts_left + attempt + 1

        if _distance(score) < _distance(best_score):
            best_audio, best_score, best_transcript = audio, score, detail
        if ok:
            log.info(
                "Sentence %d of %r accepted at %.0f%% on attempt %d/%d — within "
                "tolerance (%s).",
                chunk_index + 1, chapter_title, score * 100,
                attempt_number, _MAX_ATTEMPTS, detail,
                extra={"outcome": "pass"},
            )
            best_transcript = detail
            # The accepted clip, not the best-scoring one. They can differ: a rejected
            # 250% — a heading rendered as pure noise, which verify.py measured
            # exactly once — beats an accepted 95% on any "higher is better" ranking.
            return Attempt(audio, score, detail, accepted=True)

        log.warning(
            "Sentence %d of %r scored %.0f%% (%s); regenerating (attempt %d/%d).",
            chunk_index + 1, chapter_title, score * 100, detail,
            attempt_number, _MAX_ATTEMPTS,
        )

    if best_audio is None or len(best_audio) == 0:
        # Nothing usable at all. Emit a short silence and carry on rather than aborting
        # a long export over a single sentence, but say so loudly.
        log.error(
            "Sentence %d of %r produced no audio after %d attempts and was SKIPPED: %r",
            chunk_index + 1, chapter_title, _MAX_ATTEMPTS, raw_chunk[:60],
        )
        return Attempt(audio_io.silence(0.2), 0.0, "", accepted=False)

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
    return Attempt(best_audio, best_score, best_transcript, accepted=False)


@dataclass
class ChunkResult:
    chapter_index: int
    chapter_title: str
    chunk_index: int
    total_chunks_in_chapter: int
    audio: np.ndarray
    voice_name: str | None = None
    #: The sentence this audio was generated from, and how the checker judged it.
    #: Carried through so the caller can cache the take without re-deriving any of it —
    #: the sentence review dialog shows all three.
    text: str = ""
    score: float = 1.0
    transcript: str = ""
    accepted: bool = True


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

    available = voice.available_voices(settings)
    if available:
        log.info(
            "Voices available: %s (starting with %s)",
            ", ".join(sorted(available)), start_voice or settings.default_voice or "default",
        )
        # Before committing hours of GPU time: if the reference audio cannot be
        # decoded, every single sentence will fail the same way and the export
        # completes as a silent file. One probe is enough — they all go through the
        # same decoder.
        if settings.tts_engine == "dia":
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
            attempt = _generate_chunk(
                engine, chunk.text, voice_ref, chapter.title, chunk_index,
                verifier=verifier,
            )
            audio = attempt.audio


            # Levelled per sentence, because normalising the reference clips was
            # necessary but not sufficient: with every reference at -20 LUFS, the
            # narration they produced still spanned 10.1 LU (NuMB2 came out at
            # -13.9 against NuMN1's -19.1 from an identical prompt level). The
            # model adds its own per-voice offset, so the only place to fix the
            # level for good is on the way out.
            if getattr(settings, "normalize_output", True):
                audio = audio_io.normalize_loudness(
                    audio, config.SAMPLE_RATE, max_boost_db=audio_io.MAX_BOOST_DB
                )

            # Applied only after the coverage check, which compares audio length
            # against the text — stretching first would skew that measurement.
            audio = audio_io.change_speed(audio, speed)

            yield ChunkResult(
                chapter_index=chapter_index,
                chapter_title=chapter.title,
                chunk_index=chunk_index,
                total_chunks_in_chapter=len(voiced),
                audio=audio,
                voice_name=voice_ref.name if voice_ref else None,
                text=chunk.text,
                score=attempt.score,
                transcript=attempt.transcript,
                accepted=attempt.accepted,
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
    """The ASR checker, or None when the user has turned checking off.

    "off" does not mean unchecked: _judge still falls back to comparing the audio's
    length against the text, which is poor (see the note on _COVERAGE_MIN_RATIO) but
    catches a sentence that came out empty.
    """
    mode = getattr(settings, "verify_mode", "asr")

    def say(message: str) -> None:
        log.info(message)
        if stage:
            stage(message)

    if mode == "off":
        return None
    from .asr_verify import AsrCounter

    say("Checking each sentence by listening to it.")
    checker = AsrCounter()
    # Loaded now rather than on the first sentence. check() deliberately accepts a
    # sentence it cannot judge — a checker that raised would end a multi-hour export
    # over audio that may be fine — so a checker that never loads turns into quality
    # checking silently switched off for the whole book, reported once per sentence in
    # a log nobody reads live. Failing here says it once, before any GPU time is spent.
    try:
        checker._load()
    except Exception as e:  # noqa: BLE001 — reported, not fatal: see below
        log.error(
            "The speech checker could not be loaded, so sentences will NOT be checked "
            "for dropped words: %s", e,
        )
    return checker


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
    """Generate the selected chapters into a WAV/MP3, in one call.

    Render, then assemble — the same two steps the window runs, so there is one way a
    file gets made rather than two that can drift apart. Sentences already cached from
    an earlier run are reused, which is why an export stopped halfway resumes rather
    than starting over.

    should_continue is polled between sentences so a caller can cancel a long run;
    everything rendered so far stays in the cache. on_stage reports the phases outside
    the per-sentence loop, which on a large ebook take long enough that silence looks
    like a hang.
    """
    from . import sentence_cache

    def stage(message: str) -> None:
        log.info(message)
        if on_stage:
            on_stage(message)

    if chapters is None:
        stage("Reading the book…")
        chapters = text_extract.load_book(book_path)

    if speed is None:
        speed = getattr(settings, "speed", 1.0) or 1.0

    plan = sentence_plan(
        chapters, settings, chapter_indices=chapter_indices, start_voice=start_voice
    )
    cache = sentence_cache.SentenceCache(book_path, engine=config.cache_engine_id(settings))
    gaps = set(cache.missing(
        [(p.chapter_index, p.position, p.text, p.voice) for p in plan], speed
    ))
    todo = [p for p in plan if (p.chapter_index, p.position) in gaps]

    reused = len(plan) - len(todo)
    stage(f"{len(plan)} sentences; generating {len(todo)}"
          + (f", reusing {reused} already rendered" if reused else "") + "…")
    if todo:
        render_sentences(
            engine, settings, cache, todo, speed=speed,
            on_progress=on_progress, should_continue=should_continue, on_stage=on_stage,
        )
    return assemble(cache, plan, out_path, speed=speed, on_stage=on_stage)


@dataclass
class PlannedSentence:
    """One sentence as the book currently reads, before any audio exists for it."""

    chapter_index: int
    chapter_title: str
    position: int
    text: str
    voice: str | None


def sentence_plan(
    chapters: list[text_extract.Chapter],
    settings: config.Settings,
    chapter_indices: Iterable[int] | None = None,
    start_voice: str | None = None,
    max_chars: int = chunking.DEFAULT_MAX_CHARS,
) -> list[PlannedSentence]:
    """Every sentence the selection will produce, in order, with its voice resolved.

    The same walk iter_book_audio does, without generating anything. Two things need
    it: deciding which sentences the cache is still missing, and filling the review
    table before a single one has been rendered.
    """
    wanted = set(chapter_indices) if chapter_indices is not None else None
    current_voice = start_voice
    # Loaded once: resolving per sentence would stat every reference WAV per sentence.
    available = voice.available_voices(settings)
    plan: list[PlannedSentence] = []
    for chapter_index, chapter in enumerate(chapters):
        if wanted is not None and chapter_index not in wanted:
            continue
        voiced = chunking.chunk_text_with_voices(
            chapter.text, start_voice=current_voice, max_chars=max_chars
        )
        for position, chunk in enumerate(voiced):
            current_voice = chunk.voice
            plan.append(PlannedSentence(
                chapter_index=chapter_index,
                chapter_title=chapter.title,
                position=position,
                text=chunk.text,
                # The voice that will actually read it, not the one the text names.
                # Before the first marker the text names nothing, and recording that
                # None here made the cache key differ from the key the take was
                # stored under — so an unmarked book missed the cache on every
                # sentence of every export. See voice.resolve_name.
                voice=voice.resolve_name(settings, chunk.voice, available),
            ))
    return plan


def render_sentences(
    engine: TtsEngine,
    settings: config.Settings,
    cache,
    plan: list[PlannedSentence],
    *,
    speed: float | None = None,
    on_progress: Callable[[ChunkResult], None] | None = None,
    should_continue: Callable[[], bool] | None = None,
    on_stage: Callable[[str], None] | None = None,
    in_use: bool = True,
) -> int:
    """Generate every sentence in `plan` into `cache`. Returns how many were written.

    Takes a plan rather than a book so that one function serves both jobs: rendering
    what an export is missing, and re-rendering the handful somebody ticked in the
    review table. `in_use=False` renders an alternative take without displacing the
    one currently in the book — which is what "listen before you adopt it" needs.
    """
    def stage(message: str) -> None:
        log.info(message)
        if on_stage:
            on_stage(message)

    if speed is None:
        speed = getattr(settings, "speed", 1.0) or 1.0

    verifier = _make_verifier(settings, stage)
    available = voice.available_voices(settings)
    if available and settings.tts_engine == "dia":
        voice.check_reference_readable(next(iter(available.values())))

    written = 0
    for index, planned in enumerate(plan):
        if should_continue is not None and not should_continue():
            cache.save()
            raise ExportCancelled(f"Stopped after {written} sentences; the rest are unchanged.")

        voice_ref = voice.get_voice(settings, planned.voice) if available else None
        attempt = _generate_chunk(
            engine, planned.text, voice_ref, planned.chapter_title, planned.position,
            verifier=verifier,
        )
        audio = attempt.audio
        if getattr(settings, "normalize_output", True):
            audio = audio_io.normalize_loudness(
                audio, config.SAMPLE_RATE, max_boost_db=audio_io.MAX_BOOST_DB
            )
        audio = audio_io.change_speed(audio, speed)

        cache.put(
            planned.chapter_index, planned.position, audio,
            title=planned.chapter_title, text=planned.text,
            voice=voice_ref.name if voice_ref else None, speed=speed,
            score=attempt.score, transcript=attempt.transcript,
            accepted=attempt.accepted, in_use=in_use,
        )
        written += 1

        if on_progress:
            on_progress(ChunkResult(
                chapter_index=planned.chapter_index,
                chapter_title=planned.chapter_title,
                chunk_index=planned.position,
                total_chunks_in_chapter=len(plan),
                audio=audio,
                voice_name=voice_ref.name if voice_ref else None,
                text=planned.text,
                score=attempt.score,
                transcript=attempt.transcript,
                accepted=attempt.accepted,
            ))
        # Saved as it goes, not at the end. An export interrupted at sentence 900 of a
        # thousand must not lose the 900 — that is the whole reason the cache exists.
        if written % 25 == 0:
            cache.save()

    cache.save()
    return written


def assemble(
    cache,
    plan: list[PlannedSentence],
    out_path: str | Path,
    speed: float | None = None,
    on_stage: Callable[[str], None] | None = None,
) -> Path:
    """Write the final file from what is in the cache. Nothing is generated here.

    A sentence with no usable take contributes its silence and a warning rather than
    aborting: the alternative is a book that cannot be exported at all because one
    sentence out of four thousand never came out right.
    """
    def stage(message: str) -> None:
        log.info(message)
        if on_stage:
            on_stage(message)

    out_path = Path(out_path)
    wav_path = out_path if out_path.suffix.lower() == ".wav" else out_path.with_suffix(".wav")
    gap_scale = 1.0 / max(audio_io.MIN_SPEED, min(audio_io.MAX_SPEED, speed or 1.0))

    stage(f"Assembling {len(plan)} sentences…")
    missing = 0
    with audio_io.StreamingWavWriter(wav_path) as writer:
        previous_chapter = None
        for planned in plan:
            if previous_chapter is not None:
                if planned.chapter_index != previous_chapter:
                    writer.write(audio_io.silence(INTER_CHAPTER_SILENCE_S * gap_scale))
                else:
                    writer.write(audio_io.silence(INTER_CHUNK_SILENCE_S * gap_scale))
            take = cache.get(planned.chapter_index, planned.position)
            # The key has to be checked, not just the presence of a take. Editing a
            # chapter shifts every sentence after the edit into a new slot, and the
            # take sitting in that slot then belongs to the sentence before it — so
            # this wrote the wrong sentence into the book, silently, in the reader's
            # own voice. A gap is recoverable; a plausible-sounding wrong sentence in
            # a four-hour file is not. (SentenceCache.realign_to moves takes to follow
            # their sentences, which is what stops this being reached in practice.)
            usable = (
                take is not None
                and cache.matches(take, planned.text, planned.voice, speed or 1.0)
                and cache.wav_path(take).exists()
            )
            if not usable:
                missing += 1
                log.warning(
                    "No audio cached for sentence %d of %r; leaving a gap.",
                    planned.position + 1, planned.chapter_title,
                )
                writer.write(audio_io.silence(0.2))
            else:
                writer.write(cache.audio(take))
            previous_chapter = planned.chapter_index

    if missing:
        log.error("%d of %d sentences had no audio and were left silent.", missing, len(plan))

    if out_path.suffix.lower() == ".mp3":
        stage("Converting to MP3…")
        audio_io.transcode_to_mp3(wav_path, out_path)
        return out_path
    return wav_path
