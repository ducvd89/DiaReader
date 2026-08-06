"""Settings must survive being carried to another machine.

Voice WAV paths used to be stored absolute, including the original user's home
directory, so a copied settings file silently lost every voice on a machine with a
different username.
"""
import numpy as np
import pytest
import soundfile as sf

from diareader import config, voice


@pytest.fixture
def voice_dir(tmp_path, monkeypatch):
    d = tmp_path / "voice_refs"
    d.mkdir()
    monkeypatch.setattr(config, "VOICE_REF_DIR", d)
    sf.write(str(d / "Latradio.wav"), np.zeros(config.SAMPLE_RATE, dtype=np.float32),
             config.SAMPLE_RATE)
    return d


def test_bare_filename_resolves_into_the_local_voice_folder(voice_dir):
    assert voice.resolve_wav("Latradio.wav") == voice_dir / "Latradio.wav"


def test_absolute_path_from_another_machine_falls_back_to_local_folder(voice_dir):
    """The exact migration case: settings written under one user account, opened on a
    machine where that path does not exist."""
    stale = r"C:\Users\someone-else\AppData\Roaming\DiaReader\voice_refs\Latradio.wav"
    assert voice.resolve_wav(stale) == voice_dir / "Latradio.wav"


def test_existing_absolute_path_is_still_honoured(voice_dir):
    real = voice_dir / "Latradio.wav"
    assert voice.resolve_wav(str(real)) == real


def test_voices_load_from_a_migrated_settings_file(voice_dir):
    settings = config.Settings(
        voices={"Latradio": {"wav": "Latradio.wav", "transcript": "xin chào"}},
        default_voice="Latradio",
    )
    loaded = voice.load_voices(settings)
    assert "Latradio" in loaded
    assert loaded["Latradio"].wav_path.exists()


def test_voices_survive_a_stale_absolute_path(voice_dir):
    settings = config.Settings(
        voices={"Latradio": {
            "wav": r"D:\SomeOtherUser\voice_refs\Latradio.wav", "transcript": "xin chào",
        }},
        default_voice="Latradio",
    )
    assert "Latradio" in voice.load_voices(settings), "voice lost after moving machine"


def test_add_voice_stores_a_bare_filename(voice_dir, tmp_path, monkeypatch):
    monkeypatch.setattr(config.Settings, "save", lambda self: None)
    src = tmp_path / "incoming.wav"
    sf.write(str(src), np.zeros(config.SAMPLE_RATE, dtype=np.float32), config.SAMPLE_RATE)

    settings = config.Settings()
    voice.add_voice(settings, "NewVoice", src, "lời thoại")

    stored = settings.voices["NewVoice"]["wav"]
    assert stored == "NewVoice.wav", f"stored {stored!r}; absolute paths are not portable"
