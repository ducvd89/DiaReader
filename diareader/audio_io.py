"""Streaming WAV writing, speed control, and MP3 transcoding."""
from __future__ import annotations

import logging
import subprocess
from pathlib import Path

import imageio_ffmpeg
import numpy as np
import soundfile as sf

from . import config

log = logging.getLogger(__name__)


class StreamingWavWriter:
    """Writes audio chunks incrementally to a WAV file without holding the whole book in memory."""

    def __init__(self, path: str | Path, sample_rate: int = config.SAMPLE_RATE):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = sf.SoundFile(
            str(self.path), mode="w", samplerate=sample_rate, channels=1, subtype="PCM_16"
        )

    def write(self, audio: np.ndarray) -> None:
        if audio.size:
            self._file.write(audio)

    def close(self) -> None:
        self._file.close()

    def __enter__(self) -> "StreamingWavWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def transcode_to_mp3(wav_path: str | Path, mp3_path: str | Path, bitrate: str = "192k") -> Path:
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    mp3_path = Path(mp3_path)
    mp3_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg_exe, "-y", "-i", str(wav_path),
        "-codec:a", "libmp3lame", "-b:a", bitrate,
        str(mp3_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed converting {wav_path} -> {mp3_path}:\n{result.stderr}")
    return mp3_path


def silence(duration_s: float, sample_rate: int = config.SAMPLE_RATE) -> np.ndarray:
    return np.zeros(int(duration_s * sample_rate), dtype=np.float32)


MIN_SPEED = 0.5
MAX_SPEED = 2.0


# What counts as a pause, and how far one may be pushed.
#
# The threshold is a fraction of the chunk's own peak rather than an absolute level,
# so it follows a quiet voice down instead of declaring the whole sentence silent.
# Measured on real output at 5%: a generated sentence runs 16.6% pause by duration,
# which is ample for the 11% the default 0.9x asks for.
PAUSE_THRESHOLD_FRACTION = 0.05
# 40ms catches the gaps *between sounds* — the spacing that makes speech sound
# unhurried — not just the commas. Below that the "gaps" are inside phonemes.
PAUSE_MIN_S = 0.04
# A pause may grow to this multiple of itself, and shrink to that one. Uncapped, 0.5x
# would need every pause seven times longer, which reads as a fault rather than as
# slow speech.
PAUSE_MAX_GROWTH = 2.5
PAUSE_MIN_SHRINK = 0.4
# Envelope smoothing. Long enough to ignore the zero crossings inside a vowel, short
# enough to still see a gap between two words.
PAUSE_ENVELOPE_S = 0.02


def find_pauses(
    audio: np.ndarray,
    sample_rate: int = config.SAMPLE_RATE,
    threshold_fraction: float = PAUSE_THRESHOLD_FRACTION,
    min_pause_s: float = PAUSE_MIN_S,
) -> list[tuple[int, int]]:
    """(start, end) sample ranges that are quiet for at least `min_pause_s`."""
    if audio.size == 0:
        return []

    envelope = np.abs(audio)
    window = max(1, int(PAUSE_ENVELOPE_S * sample_rate))
    envelope = np.convolve(envelope, np.ones(window, dtype=np.float32) / window, mode="same")

    peak = float(envelope.max())
    if peak <= 0.0:
        return []

    quiet = envelope < peak * threshold_fraction
    # Pad both ends so a pause running to the edge of the chunk is still bounded.
    edges = np.diff(np.concatenate(([0], quiet.view(np.int8), [0])))
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)

    minimum = int(min_pause_s * sample_rate)
    return [(int(s), int(e)) for s, e in zip(starts, ends) if e - s >= minimum]


def _resize(segment: np.ndarray, length: int) -> np.ndarray:
    """Stretch or squeeze one near-silent segment to `length` samples.

    Interpolated rather than padded with zeros: a pause carries the recording's own
    noise floor, and splicing digital silence into it puts an audible edge at each
    join. Interpolating shifts that noise's spectrum slightly, which at 5% of peak is
    not something anyone can hear.
    """
    if length <= 0:
        return np.zeros(0, dtype=np.float32)
    if segment.size == length:
        return segment
    if segment.size < 2:
        return np.full(length, segment[0] if segment.size else 0.0, dtype=np.float32)
    source = np.linspace(0.0, 1.0, segment.size, dtype=np.float64)
    target = np.linspace(0.0, 1.0, length, dtype=np.float64)
    return np.interp(target, source, segment).astype(np.float32)


def stretch_pauses(
    audio: np.ndarray, factor: float, sample_rate: int = config.SAMPLE_RATE
) -> np.ndarray:
    """Change total duration by `factor`, taking it out of the pauses only.

    factor > 1 lengthens. Speech samples are copied through untouched, so syllables
    keep the rate and shape the model produced and only the spacing between them
    moves — which is what "read this more slowly" means to a listener, as opposed to
    drawing every vowel out.

    Returns the best it can manage: pauses have a finite budget, and a chunk with none
    at all comes back unchanged. change_speed makes up any shortfall.
    """
    if audio.size == 0 or abs(factor - 1.0) < 0.001:
        return audio

    pauses = find_pauses(audio, sample_rate)
    if not pauses:
        return audio

    wanted = int(round(audio.size * factor)) - audio.size
    lengths = np.array([end - start for start, end in pauses], dtype=np.float64)
    capacity = lengths * (
        (PAUSE_MAX_GROWTH - 1.0) if wanted > 0 else (1.0 - PAUSE_MIN_SHRINK)
    )
    total_capacity = float(capacity.sum())
    if total_capacity <= 0:
        return audio

    # Proportional to length, so a comma-sized gap and a full stop keep their relative
    # weight instead of every pause becoming the same size.
    share = min(1.0, abs(wanted) / total_capacity)
    deltas = np.round(capacity * share).astype(int)
    if wanted < 0:
        deltas = -deltas

    pieces: list[np.ndarray] = []
    cursor = 0
    for (start, end), delta in zip(pauses, deltas):
        pieces.append(audio[cursor:start])
        pieces.append(_resize(audio[start:end], (end - start) + int(delta)))
        cursor = end
    pieces.append(audio[cursor:])
    return np.concatenate(pieces).astype(np.float32)


def change_speed(
    audio: np.ndarray, speed: float, sample_rate: int = config.SAMPLE_RATE
) -> np.ndarray:
    """Change reading pace: spacing first, then the speech itself if that is not enough.

    speed < 1 is slower, > 1 is faster.

    What this used to do, and why it changed: everything went through ffmpeg's atempo,
    which stretches the whole waveform uniformly. At 0.9x every vowel was 11% longer as
    well as every gap, and drawn-out syllables are not what slow reading sounds like —
    an unhurried reader leaves more room between sounds and says each one at much the
    same rate. So the pauses absorb the change first, and atempo now only covers what
    they cannot: at 0.9x that is usually nothing at all, while 0.5x still needs it,
    because no amount of silence between words makes a sentence take twice as long
    without sounding broken.
    """
    if audio.size == 0 or abs(speed - 1.0) < 0.01:
        return audio
    speed = max(MIN_SPEED, min(MAX_SPEED, speed))

    factor = 1.0 / speed
    original_size = audio.size
    audio = stretch_pauses(audio, factor, sample_rate)

    # Whatever the pauses could not absorb, expressed again as a speed. Below 1 when
    # the chunk is still too short, above 1 when it is still too long.
    achieved = audio.size / original_size
    residual = achieved / factor
    if abs(residual - 1.0) < 0.01:
        return audio
    return _time_stretch(audio, residual, sample_rate)


def _time_stretch(
    audio: np.ndarray, speed: float, sample_rate: int = config.SAMPLE_RATE
) -> np.ndarray:
    """Uniform tempo change without pitch shift, via ffmpeg's atempo filter.

    Naive resampling would be simpler but shifts pitch, which would stop the narration
    sounding like the reference voice. atempo natively covers 0.5-2.0, which is exactly
    the range offered.

    Cost is a few tens of milliseconds per sentence against ~40s of generation, so this
    runs per chunk and applies to live playback as well as export.
    """
    speed = max(MIN_SPEED, min(MAX_SPEED, speed))

    cmd = [
        imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-loglevel", "error",
        "-f", "f32le", "-ar", str(sample_rate), "-ac", "1", "-i", "pipe:0",
        "-filter:a", f"atempo={speed:.4f}",
        "-f", "f32le", "-ar", str(sample_rate), "-ac", "1", "pipe:1",
    ]
    result = subprocess.run(cmd, input=np.ascontiguousarray(audio, dtype=np.float32).tobytes(),
                            capture_output=True)
    if result.returncode != 0 or not result.stdout:
        # Better to narrate at the wrong speed than to lose the sentence entirely.
        log.warning("Speed change to %.2fx failed; using the original audio. %s",
                    speed, result.stderr.decode(errors="replace")[:200])
        return audio
    return np.frombuffer(result.stdout, dtype=np.float32).copy()


# Spoken-word target. Sits near the middle of what the reference clips already were
# (Latradio -18.9, NuMB1 -20.8) so normalising moves everything a little rather than
# moving most things a lot, and low enough that the quiet clips can be lifted without
# running out of headroom.
TARGET_LUFS = -20.0
# True peak is left this far below full scale. Lifting a quiet clip to the target can
# otherwise push its peaks past 0 dBFS and clip them — NamMN1 needs +13.3 dB to reach
# the target and peaks at -11.6 dBFS, which would land at +1.6 dBFS.
PEAK_CEILING_DBFS = -1.0


def measure_loudness(audio: np.ndarray, sample_rate: int) -> float:
    """ITU-R BS.1770 integrated loudness in LUFS, or -inf if it cannot be measured.

    RMS is the tempting shortcut and the wrong one: it weights a low rumble the same
    as a vowel, so two clips can match on RMS and still sound unequal. LUFS is the
    measure that means "sounds equally loud".

    The meter needs at least one 400ms block, and refuses shorter input. That is a
    reasonable thing for a meter to do and an unreasonable thing to crash `add_voice`
    over, so an unmeasurable clip reports -inf and callers leave it alone.
    """
    import pyloudnorm as pyln

    audio = np.asarray(audio, dtype=np.float64).reshape(-1)
    try:
        return float(pyln.Meter(sample_rate).integrated_loudness(audio))
    except ValueError:
        return float("-inf")


# Cap on how far a single chunk may be lifted. A sentence that came out nearly silent
# is a generation failure, not a quiet delivery; amplifying it to the target would
# just make its noise floor audible at full volume.
MAX_BOOST_DB = 12.0


def normalize_loudness(audio: np.ndarray, sample_rate: int,
                       target_lufs: float = TARGET_LUFS,
                       peak_ceiling_dbfs: float = PEAK_CEILING_DBFS,
                       max_boost_db: float | None = None) -> np.ndarray:
    """Bring a clip to `target_lufs`, backing off where that would clip.

    Gain only — no limiting or compression. Squashing peaks would change the voice's
    character, which is the one thing a reference clip exists to preserve; a clip that
    cannot reach the target without clipping is left quieter instead, and the caller
    can see the shortfall by measuring again.

    `max_boost_db` caps upward gain only. Attenuation is never capped: something too
    loud is always safe to bring down.
    """
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if audio.size == 0:
        return audio
    peak = float(np.max(np.abs(audio)))
    if peak <= 0:
        return audio
    current = measure_loudness(audio, sample_rate)
    if not np.isfinite(current):
        return audio

    wanted_db = target_lufs - current
    if max_boost_db is not None:
        wanted_db = min(wanted_db, max_boost_db)
    headroom_db = peak_ceiling_dbfs - 20 * np.log10(peak)
    gain_db = min(wanted_db, headroom_db)
    return (audio * (10 ** (gain_db / 20))).astype(np.float32)


def trim_trailing_silence(
    audio: np.ndarray,
    sample_rate: int = config.SAMPLE_RATE,
    threshold: float = 0.02,
    keep_tail_s: float = 0.15,
) -> np.ndarray:
    """Cut dead air off the end of a generated chunk.

    Dia generates a fixed-length audio-token budget per call and doesn't reliably emit
    EOS on short/unconditioned text for this checkpoint, so chunks otherwise carry many
    extra seconds of near-silence padded out to max_tokens.
    """
    if audio.size == 0:
        return audio
    above_threshold = np.flatnonzero(np.abs(audio) > threshold)
    if above_threshold.size == 0:
        return audio
    end = min(audio.size, above_threshold[-1] + int(keep_tail_s * sample_rate))
    return audio[:end]
