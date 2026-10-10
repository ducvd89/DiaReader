"""VieNeu's adapter must preserve DiaReader's audio and voice contracts."""
from pathlib import Path

import numpy as np

from diareader import config
from diareader.sentence_cache import SentenceCache
from diareader.vieneu_engine import VieNeuEngine, VIENEU_SAMPLE_RATE
from diareader.voice import VoiceReference
from diareader.voice import PresetVoice, available_voices, resolve_name
from diareader.vieneu_presets import marker_for_name, presets


class FakeVieNeu:
    def __init__(self):
        self.calls = []

    def list_preset_voices(self):
        return [("Hải Đăng", "Hải Đăng"), ("Mai Anh", "Mai Anh")]

    def infer(self, text, **kwargs):
        self.calls.append((text, kwargs))
        return np.full(VIENEU_SAMPLE_RATE, 0.25, dtype=np.float32)


def test_preset_voice_resamples_to_pipeline_rate():
    model = FakeVieNeu()
    audio = VieNeuEngine(model).generate("Xin chào.", temperature=0.7)
    assert model.calls == [("Xin chào.", {"temperature": 0.7, "voice": "Hải Đăng"})]
    assert len(audio) == config.SAMPLE_RATE
    assert audio.dtype == np.float32


def test_reference_voice_uses_clip_without_dia_transcript(tmp_path):
    model = FakeVieNeu()
    ref = VoiceReference("Sample", tmp_path / "sample.wav", "This is the Dia prompt.")
    VieNeuEngine(model).generate("The actual sentence.", voice_ref=ref)
    assert model.calls[0] == (
        "The actual sentence.",
        {"temperature": 0.8, "ref_audio": str(ref.wav_path)},
    )


def test_engine_cache_is_separate_and_dia_cache_path_stays_stable(tmp_path, monkeypatch):
    import diareader.sentence_cache as sentence_cache

    monkeypatch.setattr(sentence_cache, "CACHE_ROOT", tmp_path)
    book = tmp_path / "book.txt"
    dia = SentenceCache(book)
    vieneu = SentenceCache(book, config.cache_engine_id(config.Settings(tts_engine="vieneu")))
    assert dia.dir == tmp_path / sentence_cache.book_key(book)
    assert vieneu.dir != dia.dir


def test_preset_default_and_named_reference_markers():
    settings = config.Settings(tts_engine="vieneu", default_voice="OldDia")
    available = {
        "OldDia": VoiceReference("OldDia", Path("old.wav"), "old"),
        "Narrator2": VoiceReference("Narrator2", Path("two.wav"), "two"),
        marker_for_name("Hải Đăng"): PresetVoice(marker_for_name("Hải Đăng"), "Hải Đăng"),
    }
    assert resolve_name(settings, None, available) == marker_for_name("Hải Đăng")
    assert resolve_name(settings, "Narrator2", available) == "Narrator2"
    settings.vieneu_use_preset = False
    assert resolve_name(settings, None, available) == "OldDia"


def test_all_github_presets_have_unique_usable_markers():
    from diareader.chunking import VOICE_MARKER

    all_presets = presets()
    assert len(all_presets) == 25
    assert len({p.marker for p in all_presets}) == 25
    assert all(VOICE_MARKER.fullmatch(f"[{p.marker}]") for p in all_presets)
    assert len(available_voices(config.Settings(tts_engine="vieneu"))) == 25


def test_preset_marker_selects_that_voice():
    model = FakeVieNeu()
    preset = PresetVoice(marker_for_name("Mai Anh"), "Mai Anh")
    VieNeuEngine(model).generate("Xin chào.", voice_ref=preset)
    assert model.calls[0][1]["voice"] == "Mai Anh"


def test_dia_performance_tags_are_not_spoken_by_vieneu():
    model = FakeVieNeu()
    VieNeuEngine(model).generate("[S1] (laughs) Xin chào.")
    assert model.calls[0][0] == "Xin chào."
