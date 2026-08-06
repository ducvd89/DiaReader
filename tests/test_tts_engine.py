"""What text actually reaches the model, especially on the voice-clone path."""
import numpy as np
import pytest

from diareader import config
from diareader.model_loader import LoadedModel
from diareader.tts_engine import TtsEngine
from diareader.voice import VoiceReference


class FakeDia:
    """Captures the prompt text and kwargs handed to dia.generate()."""

    def __init__(self):
        self.calls = []

    def generate(
        self, text, max_tokens=None, cfg_scale=3.0, temperature=1.3, top_p=0.95,
        use_torch_compile=False, cfg_filter_top_k=35, audio_prompt=None,
        audio_prompt_path=None, use_cfg_filter=None, verbose=False,
    ):
        self.calls.append({"text": text, "max_tokens": max_tokens, "audio_prompt": audio_prompt,
                           "temperature": temperature})
        return np.ones(int(1.0 * config.SAMPLE_RATE), dtype=np.float32)


@pytest.fixture
def engine():
    dia = FakeDia()
    return TtsEngine(LoadedModel(dia=dia, device="cuda", generation_defaults={})), dia


@pytest.fixture
def voice_ref(tmp_path):
    import soundfile as sf

    wav = tmp_path / "ref.wav"
    sf.write(str(wav), np.zeros(config.SAMPLE_RATE * 2, dtype=np.float32), config.SAMPLE_RATE)
    return VoiceReference(name="Test", wav_path=wav, transcript="Đây là giọng mẫu.")


def test_no_voice_sends_the_sentence_unchanged(engine):
    eng, dia = engine
    eng.generate("Xin chào.")
    assert dia.calls[0]["text"] == "Xin chào."
    assert dia.calls[0]["audio_prompt"] is None


def test_voice_prompt_transcript_loses_its_terminal_mark(engine, voice_ref):
    """The checkpoint stops at the first terminal period. Leaving the transcript's
    period in place makes it fire EOS exactly where the prompt audio ends, so the real
    sentence never gets spoken."""
    eng, dia = engine
    eng.generate("Nội dung thật.", voice_ref=voice_ref)

    sent = dia.calls[0]["text"]
    assert sent == "Đây là giọng mẫu Nội dung thật."
    assert "mẫu." not in sent, "transcript kept its period; generation will stop there"
    assert sent.count(".") == 1, f"only the real sentence may end with a period: {sent!r}"


def test_voice_prompt_audio_is_passed(engine, voice_ref):
    eng, dia = engine
    eng.generate("Nội dung.", voice_ref=voice_ref)
    assert dia.calls[0]["audio_prompt"] == str(voice_ref.wav_path)


def test_max_tokens_budget_includes_the_prompt(engine, voice_ref):
    """generate() counts max_tokens from a step counter that starts at the prompt's
    length, so the prompt has to be added on top or the loop exits immediately."""
    eng, dia = engine
    eng.generate("Một câu dài vừa phải để kiểm tra ngân sách token.", voice_ref=voice_ref)
    with_voice = dia.calls[0]["max_tokens"]

    eng.generate("Một câu dài vừa phải để kiểm tra ngân sách token.")
    without_voice = dia.calls[1]["max_tokens"]

    assert with_voice > without_voice
    assert with_voice - without_voice >= 150, "2s of prompt audio is ~172 tokens"


def test_overrides_reach_the_model(engine):
    eng, dia = engine
    eng.generate("Xin chào.", temperature=0.7)
    assert dia.calls[0]["temperature"] == 0.7


def test_empty_generation_is_reported_as_empty_audio(engine):
    eng, dia = engine

    def boom(*a, **k):
        raise RuntimeError("Kernel size can't be greater than actual input size")

    dia.generate = boom
    assert len(eng.generate("Xin chào.")) == 0
