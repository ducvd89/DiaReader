"""Tempo control: stretch without pitch shift, and never lose a sentence to it."""
import numpy as np
import pytest

from diareader import audio_io, config


def _tone(seconds: float, hz: int = 220) -> np.ndarray:
    t = np.linspace(0, seconds, int(seconds * config.SAMPLE_RATE), dtype=np.float32)
    return (0.3 * np.sin(2 * np.pi * hz * t)).astype(np.float32)


@pytest.mark.parametrize("speed, expected", [(0.5, 6.0), (0.8, 3.75), (1.5, 2.0), (2.0, 1.5)])
def test_duration_scales_inversely_with_speed(speed, expected):
    out = audio_io.change_speed(_tone(3.0), speed)
    assert abs(len(out) / config.SAMPLE_RATE - expected) < 0.05


def test_speed_one_is_a_no_op():
    audio = _tone(1.0)
    assert audio_io.change_speed(audio, 1.0) is audio, "should not re-encode for no change"


def test_out_of_range_speed_is_clamped():
    """atempo only accepts 0.5-2.0; a wilder value must clamp rather than fail and
    lose the audio."""
    out = audio_io.change_speed(_tone(2.0), 5.0)
    assert len(out) > 0
    assert abs(len(out) / config.SAMPLE_RATE - 1.0) < 0.05  # clamped to 2.0x


def test_empty_audio_survives():
    assert audio_io.change_speed(np.zeros(0, dtype=np.float32), 0.5).size == 0


def test_pitch_is_preserved():
    """Naive resampling would shift pitch and stop the narration sounding like the
    reference voice, so the dominant frequency must survive a tempo change."""
    slowed = audio_io.change_speed(_tone(2.0, hz=440), 0.5)

    spectrum = np.abs(np.fft.rfft(slowed))
    freqs = np.fft.rfftfreq(len(slowed), 1 / config.SAMPLE_RATE)
    peak = freqs[np.argmax(spectrum)]
    assert abs(peak - 440) < 15, f"pitch moved to {peak:.0f} Hz; tempo change resampled"


def test_settings_round_trip_speed(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "SETTINGS_PATH", tmp_path / "settings.json")
    monkeypatch.setattr(config, "APP_DIR", tmp_path)

    s = config.Settings()
    s.speed = 0.8
    s.save()
    assert config.Settings.load().speed == 0.8


def test_default_speed_matches_the_reference_voice_pace():
    """Not 1.0 on purpose. Measured over a 60-sentence chapter the model reads at
    19.8 chars/s against the reference clip's 17.6 — a 12% rush that is audible, so the
    shipped default slows it back to the pace of the voice being imitated."""
    assert config.Settings().speed == 0.9


def test_default_speed_is_an_offered_choice():
    """A default the UI cannot display would silently reset when Settings is opened."""
    from diareader.gui.widgets import SPEED_CHOICES

    assert config.Settings().speed in SPEED_CHOICES
