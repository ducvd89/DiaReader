"""Named reference voices, and the markers that switch between them mid-text.

A book is narrated by one fixed voice so the audio stays consistent, but the text can
switch speaker with an inline marker like ``[GiongNu2]`` — useful for dialogue. The
marker names a voice from the user's registry; everything after it uses that voice
until the next marker.

Mechanically a voice is a reference WAV plus its transcript. Dia prefills the WAV into
the decoder while the text conditions the encoder, so the transcript must be prepended
to the text being spoken (handled in tts_engine.generate).
"""
from __future__ import annotations

import logging
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import soundfile as sf

from . import audio_io, config

if TYPE_CHECKING:
    from .tts_engine import TtsEngine

log = logging.getLogger(__name__)

# Must stay in step with chunking.VOICE_MARKER — a name outside this set can never be
# referenced by a marker.
_VALID_NAME = re.compile(r"[A-Za-z0-9_\-]{1,40}")


@dataclass(frozen=True)
class VoiceReference:
    name: str
    wav_path: Path
    transcript: str


def resolve_wav(stored: str) -> Path:
    """Locate a voice's WAV from whatever the settings file recorded.

    Voices are stored by bare filename so a settings file can be carried to another
    machine — an absolute path baked in the original user's home directory would
    silently drop the voice there. Absolute paths from older settings still work, and
    fall back to the same filename in the local voice folder if the original is gone.
    """
    path = Path(stored)
    if not path.is_absolute():
        return config.VOICE_REF_DIR / path
    if path.exists():
        return path
    return config.VOICE_REF_DIR / path.name


def load_voices(settings: "config.Settings") -> dict[str, VoiceReference]:
    """Every configured voice, keyed by name. Entries with a missing WAV are dropped."""
    voices: dict[str, VoiceReference] = {}
    for name, entry in (settings.voices or {}).items():
        wav = resolve_wav(entry.get("wav", ""))
        transcript = entry.get("transcript", "")
        if not wav.exists():
            log.warning("Voice %r points at a missing file: %s", name, wav)
            continue
        voices[name] = VoiceReference(name=name, wav_path=wav, transcript=transcript)
    return voices


def resolve_name(
    settings: "config.Settings", name: str | None, available: dict | None = None
) -> str | None:
    """Which voice will actually read `name`, by name. None if none is configured.

    Split out of get_voice because two places have to agree on the answer and they
    used to disagree. The export plan recorded the voice a sentence *asked* for —
    None for anything before the first marker — while the renderer stored the voice
    that actually read it. The sentence cache is keyed on that value, so for a book
    with no markers every key differed from the one it was stored under and the whole
    cache missed: every export regenerated every sentence, silently, at ~40s each.

    `available` lets a caller resolving hundreds of sentences load the library once;
    load_voices stats every WAV.
    """
    voices = load_voices(settings) if available is None else available
    if not voices:
        return None
    if name and name in voices:
        return name
    default = settings.default_voice
    if default and default in voices:
        return default
    return next(iter(voices))


def get_voice(settings: "config.Settings", name: str | None) -> VoiceReference | None:
    """Resolve a voice name. None (or an unknown name) falls back to the default voice,
    and if none is configured, to unconditioned generation."""
    voices = load_voices(settings)
    if name and voices and name not in voices:
        log.warning("Unknown voice %r; falling back to the default voice.", name)
    resolved = resolve_name(settings, name, voices)
    return voices[resolved] if resolved else None


def voice_names(settings: "config.Settings") -> list[str]:
    return sorted(load_voices(settings).keys())


def voice_names_in_library_order(settings: "config.Settings") -> list[str]:
    """Names in the order they were added, which is how the settings dict stores them.

    The script editor colours voices by position, so this order is what keeps a
    colour attached to a voice: adding one appends, taking the next free swatch and
    leaving every existing voice's colour alone. Sorting instead would reshuffle the
    whole cast the first time someone added a voice early in the alphabet.
    """
    return list(load_voices(settings).keys())


class ReferenceAudioUnreadable(RuntimeError):
    """Raised when a configured voice's WAV cannot be decoded at all."""


def check_reference_readable(ref: VoiceReference) -> None:
    """Fail loudly, once, if voice conditioning cannot possibly work.

    Dia loads the audio prompt through ``torchaudio.load``, which on Windows needs
    FFmpeg's *shared* libraries (the DLLs) present — ``imageio-ffmpeg`` ships only an
    ``ffmpeg.exe`` and does not satisfy it. Without them every generate() call raises
    inside dia, ``tts_engine`` turns that into empty audio, and the pipeline retries
    and skips each sentence in turn: a multi-hour export completes "successfully" as
    a file of silence, with the real cause buried in per-sentence warnings. That is
    exactly how one 54-sentence export was lost, so this checks once, up front, and
    says what to install.
    """
    import torchaudio

    from . import ffmpeg_dlls

    try:
        audio, _ = torchaudio.load(str(ref.wav_path), channels_first=True)
    except Exception as e:  # noqa: BLE001 — any decode failure is fatal the same way
        raise ReferenceAudioUnreadable(
            f"Cannot read the reference audio for voice {ref.name!r} ({ref.wav_path}).\n\n"
            f"{type(e).__name__}: {str(e)[:300]}\n\n"
            "If this mentions libtorchcodec, FFmpeg's shared libraries are missing. "
            "Install a shared build, then reopen the app so it can find them:\n"
            "    winget install Gyan.FFmpeg.Shared\n\n"
            "DiaReader searches PATH and winget's package directory at startup. If "
            f"your build lives elsewhere, point {ffmpeg_dlls.ENV_OVERRIDE} at the "
            "directory holding avutil-*.dll.\n\n"
            "Alternatively remove every voice under Settings -> Voices; this "
            "checkpoint was fine-tuned without audio prompts, so unconditioned "
            "generation is a perfectly good default."
        ) from e

    # Loading fine is not enough: the DAC encoder takes mono only, and a stereo
    # prompt fails per sentence rather than here. Voices registered before
    # add_voice started downmixing can still be stereo on disk.
    if audio.shape[0] != 1:
        raise ReferenceAudioUnreadable(
            f"The reference audio for voice {ref.name!r} has {audio.shape[0]} channels; "
            f"the model accepts mono only ({ref.wav_path}).\n\n"
            "Every sentence in this voice would produce no audio at all. Re-add the "
            "voice under Settings -> Voices — it is converted to mono on the way in now."
        )


class InvalidVoiceName(ValueError):
    """Raised for a name that could never work inside a [marker]."""


def validate_name(name: str) -> str:
    """Reject names a marker can't reference.

    Markers match [A-Za-z0-9_-] only, so a name like "Nữ (MN1)" would register happily
    and then never match anything — the marker would be read as ordinary text and the
    voice silently ignored. Better to refuse it at the point of naming.
    """
    name = name.strip()
    if not name:
        raise InvalidVoiceName("A voice needs a name.")
    if not _VALID_NAME.fullmatch(name):
        raise InvalidVoiceName(
            f"{name!r} cannot be used in a marker. Use letters, digits, underscore or "
            "hyphen only — no spaces, accents or punctuation. For example 'NuMN1'."
        )
    return name


def _install_reference_wav(src: Path, dest: Path) -> None:
    """Copy a reference WAV into the app folder, as mono at a consistent loudness.

    Two conversions, both learned the hard way.

    Mono: Dia feeds the prompt through the DAC encoder, whose first layer is a Conv1d
    with ``in_channels=1``. A stereo file fails at generation time with "expected
    input[1, 2, N] to have 1 channels, but got 2 channels instead" — and because
    tts_engine turns that into empty audio, the result is not an error but every
    sentence in that voice being retried and skipped. One stereo file cost 4 sentences
    of a real chapter before it was traced back here.

    Loudness: the model takes its level from the prompt, so a quiet reference makes a
    quiet narrator. Measured across six clips that had been added by hand, integrated
    loudness spanned 17.1 LU — the quietest was 7.17x smaller in amplitude than the
    loudest, which in a book that switches voices is a listener reaching for the volume
    every few paragraphs.
    """
    data, rate = sf.read(str(src), always_2d=True, dtype="float32")
    audio = data.mean(axis=1) if data.shape[1] > 1 else data[:, 0]
    if data.shape[1] > 1:
        log.info("Reference audio for %s is %d-channel; downmixing to mono.",
                 dest.stem, data.shape[1])

    before = audio_io.measure_loudness(audio, rate)
    audio = audio_io.normalize_loudness(audio, rate)
    after = audio_io.measure_loudness(audio, rate)
    if abs(after - before) >= 0.1:
        log.info("Reference audio for %s normalised %.1f -> %.1f LUFS.",
                 dest.stem, before, after)
    sf.write(str(dest), audio, rate, subtype="PCM_16")


def normalize_existing_references(settings: "config.Settings") -> list[tuple[str, float, float]]:
    """Bring already-registered voices up to the same loudness, in place.

    Voices added before normalisation existed — or carried over in a settings folder
    from another machine — are still at whatever level they were recorded at. Returns
    (name, before, after) for each clip that moved.
    """
    changed = []
    for name, ref in load_voices(settings).items():
        audio, rate = sf.read(str(ref.wav_path), always_2d=True, dtype="float32")
        mono = audio.mean(axis=1) if audio.shape[1] > 1 else audio[:, 0]
        before = audio_io.measure_loudness(mono, rate)
        normalized = audio_io.normalize_loudness(mono, rate)
        after = audio_io.measure_loudness(normalized, rate)
        if abs(after - before) < 0.1:
            continue
        sf.write(str(ref.wav_path), normalized, rate, subtype="PCM_16")
        log.info("Normalised %s: %.1f -> %.1f LUFS", name, before, after)
        changed.append((name, before, after))
    return changed


# Measured on the fine-tune's own training clips, same figure the pipeline uses to
# judge chunk coverage.
_NATURAL_CHARS_PER_SECOND = 20.3
# Leading/trailing silence and ordinary pacing put healthy references around
# 1.15-1.47x here, so this only fires on a real mismatch rather than on tempo.
_TRANSCRIPT_COVERAGE_LIMIT = 1.8


def expected_duration(transcript: str) -> float:
    """Roughly how long `transcript` takes to say, in seconds."""
    return len(transcript.strip()) / _NATURAL_CHARS_PER_SECOND


def transcript_coverage(duration_s: float, transcript: str) -> float | None:
    """Audio length over the length its transcript accounts for, or None if unknown.

    Dia prefills the prompt audio into the decoder while the transcript conditions
    the encoder, so speech the transcript omits is content the decoder has already
    "spoken" that the encoder never saw — and the model answers that by looping and
    repeating rather than moving on. A long prompt also eats the token budget: 21.8s
    of audio is ~1875 of the 3072 tokens a sentence gets.

    Healthy references measure about 1.0-1.5x here. NamMN1 arrived at 2.36x — 21.8s
    of speech behind a transcript worth 9.2s — and had to be re-cut.
    """
    expected = expected_duration(transcript)
    if expected <= 0 or duration_s <= 0:
        return None
    return duration_s / expected


def _warn_if_transcript_does_not_cover_audio(name: str, wav: Path, transcript: str) -> None:
    """Log the mismatch above. Heuristic, so it warns rather than refuses — only the
    person who recorded the clip knows what is actually in it."""
    if not transcript.strip():
        return
    try:
        duration = sf.info(str(wav)).duration
    except Exception:  # noqa: BLE001 — a warning is never worth failing registration
        return
    ratio = transcript_coverage(duration, transcript)
    if ratio is not None and ratio > _TRANSCRIPT_COVERAGE_LIMIT:
        log.warning(
            "Voice %r: the reference audio runs %.1fs but its transcript only accounts "
            "for about %.1fs. Dia prefills the whole clip into the decoder, so speech "
            "the transcript omits makes the model repeat and drift. Trim the WAV to just "
            "the transcribed sentence, or extend the transcript to cover everything "
            "spoken in it.",
            name, duration, expected_duration(transcript),
        )


def add_voice(
    settings: "config.Settings", name: str, wav_path: str | Path, transcript: str
) -> VoiceReference:
    """Register a voice, copying the WAV into the app's own folder so the reference
    survives the user moving or deleting the original."""
    name = validate_name(name)
    config.ensure_dirs()
    src = Path(wav_path)
    dest = config.VOICE_REF_DIR / f"{name}{src.suffix or '.wav'}"
    if src.resolve() != dest.resolve():
        _install_reference_wav(src, dest)

    _warn_if_transcript_does_not_cover_audio(name, dest, transcript)

    # Stored by filename, not full path, so the settings file stays portable between
    # machines (see resolve_wav).
    settings.voices = dict(settings.voices or {})
    settings.voices[name] = {"wav": dest.name, "transcript": transcript}
    if not settings.default_voice:
        settings.default_voice = name
    settings.save()
    return VoiceReference(name=name, wav_path=dest, transcript=transcript)


def remove_voice(settings: "config.Settings", name: str) -> None:
    voices = dict(settings.voices or {})
    voices.pop(name, None)
    settings.voices = voices
    if settings.default_voice == name:
        settings.default_voice = next(iter(voices), "")
    settings.save()


def save_generated_reference(
    audio: np.ndarray, transcript: str, name: str = "auto", sample_rate: int = config.SAMPLE_RATE
) -> VoiceReference:
    config.ensure_dirs()
    dest = config.VOICE_REF_DIR / f"{name}.wav"
    sf.write(str(dest), audio, sample_rate)
    return VoiceReference(name=name, wav_path=dest, transcript=transcript)


def capture_voice_from_text(
    engine: "TtsEngine", name: str, prompt_text: str, settings: "config.Settings"
) -> VoiceReference:
    """Generate a sample line unconditioned and register it as a named voice, so a user
    can create a voice without supplying their own recording."""
    from . import chunking

    audio = engine.generate(chunking.wrap_for_narration(prompt_text), voice_ref=None)
    if len(audio) == 0:
        raise RuntimeError("The model produced no audio for that sample text.")
    ref = save_generated_reference(audio, prompt_text, name=name)
    return add_voice(settings, name, ref.wav_path, prompt_text)


# --- legacy shim -------------------------------------------------------------------

def load_configured_reference(settings: "config.Settings") -> VoiceReference | None:
    """Back-compat for callers that just want "the" voice."""
    return get_voice(settings, None)
