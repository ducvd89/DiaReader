"""Slowing down by leaving more room, not by drawing out every syllable.

What this replaced: every speed change went through ffmpeg's atempo, which stretches
the whole waveform uniformly. At 0.9x each vowel ran 11% long as well as each gap, and
lengthened vowels do not sound like unhurried reading — they sound like a tape playing
slow. A reader who slows down mostly leaves more space between sounds.
"""
import numpy as np

from diareader import audio_io, config

RATE = config.SAMPLE_RATE


def _speech(seconds: float, hz: int = 220) -> np.ndarray:
    t = np.linspace(0, seconds, int(seconds * RATE), endpoint=False, dtype=np.float32)
    return (0.3 * np.sin(2 * np.pi * hz * t)).astype(np.float32)


def _with_pauses(word_s=0.30, pause_s=0.20, words=6) -> np.ndarray:
    """Alternating tone and silence — a stand-in for words separated by gaps."""
    parts = []
    for i in range(words):
        parts.append(_speech(word_s))
        if i < words - 1:
            parts.append(np.zeros(int(pause_s * RATE), dtype=np.float32))
    return np.concatenate(parts)


def test_pauses_are_found_between_words():
    audio = _with_pauses(words=4)
    pauses = audio_io.find_pauses(audio, RATE)
    assert len(pauses) == 3, f"three gaps between four words, found {len(pauses)}"


def test_continuous_speech_has_no_pauses():
    assert audio_io.find_pauses(_speech(2.0), RATE) == []


def test_gaps_shorter_than_the_floor_are_not_pauses():
    """Below ~40ms a quiet stretch is inside a phoneme, not between two of them."""
    audio = _with_pauses(word_s=0.3, pause_s=0.01, words=5)
    assert audio_io.find_pauses(audio, RATE) == []


def test_slowing_leaves_the_speech_samples_alone():
    """The whole point: syllables keep the rate and shape the model produced."""
    audio = _with_pauses()
    out = audio_io.stretch_pauses(audio, 1.0 / 0.9, RATE)

    # The first word runs from 0 to 0.30s in both, sample for sample.
    word = int(0.30 * RATE)
    assert np.array_equal(out[:word], audio[:word])


def test_slowing_lengthens_the_gaps():
    audio = _with_pauses()
    out = audio_io.stretch_pauses(audio, 1.0 / 0.8, RATE)

    before = sum(e - s for s, e in audio_io.find_pauses(audio, RATE))
    after = sum(e - s for s, e in audio_io.find_pauses(out, RATE))
    assert after > before * 1.2


def test_the_default_speed_needs_no_time_stretching_at_all(monkeypatch):
    """At 0.9x on speech with ordinary gaps, atempo must never run — measured on real
    output, the pauses cover 100% of the change."""
    called = []
    monkeypatch.setattr(
        audio_io, "_time_stretch",
        lambda *a, **k: called.append(a) or a[0],
    )
    audio_io.change_speed(_with_pauses(), 0.9, RATE)
    assert called == [], "the speech itself should not have been touched"


def test_a_large_slowdown_falls_back_to_stretching_for_the_remainder():
    """No amount of silence between words makes a sentence take twice as long without
    sounding broken, so past the pause budget the speech is stretched too."""
    audio = _with_pauses()
    out = audio_io.change_speed(audio, 0.5, RATE)
    assert abs(len(out) / len(audio) - 2.0) < 0.1


def test_duration_still_tracks_the_requested_speed():
    audio = _with_pauses()
    for speed in (0.8, 0.7, 1.25, 1.5):
        out = audio_io.change_speed(audio, speed, RATE)
        assert abs(len(out) / len(audio) - 1 / speed) < 0.1, f"at {speed}x"


def test_speeding_up_shortens_the_gaps_first():
    audio = _with_pauses()
    out = audio_io.stretch_pauses(audio, 1.0 / 1.25, RATE)
    assert len(out) < len(audio)

    word = int(0.30 * RATE)
    assert np.array_equal(out[:word], audio[:word])


def test_audio_with_no_pauses_is_left_to_the_stretcher():
    """A chunk of continuous sound has nothing to give, and must still change speed."""
    audio = _speech(2.0)
    assert audio_io.stretch_pauses(audio, 1.25, RATE) is audio
    out = audio_io.change_speed(audio, 0.8, RATE)
    assert abs(len(out) / len(audio) - 1.25) < 0.05


def test_silence_is_not_spliced_in():
    """A pause carries the recording's own noise floor. Padding with digital silence
    would put an audible edge at every join, so pauses are interpolated instead."""
    noisy_gap = np.concatenate([
        _speech(0.3),
        (np.random.default_rng(0).normal(0, 0.001, int(0.2 * RATE))).astype(np.float32),
        _speech(0.3),
    ])
    out = audio_io.stretch_pauses(noisy_gap, 1.5, RATE)

    middle = out[int(0.35 * RATE):int(0.45 * RATE)]
    assert np.count_nonzero(middle) > len(middle) * 0.9, "the gap went digitally silent"


def test_empty_audio_survives():
    assert audio_io.stretch_pauses(np.zeros(0, dtype=np.float32), 1.5, RATE).size == 0


def test_pitch_is_preserved_when_stretching_is_needed():
    audio = _with_pauses(word_s=0.4, pause_s=0.05, words=5)
    out = audio_io.change_speed(audio, 0.5, RATE)

    spectrum = np.abs(np.fft.rfft(out))
    freqs = np.fft.rfftfreq(len(out), 1 / RATE)
    assert abs(freqs[np.argmax(spectrum)] - 220) < 15
