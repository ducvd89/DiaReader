from pathlib import Path

import pytest

from diareader import chunking, config, voice


def _ref(tmp_path):
    return voice.VoiceReference(
        name="Latradio", wav_path=Path(tmp_path) / "Latradio.wav", transcript="xin chào"
    )


def test_unreadable_reference_audio_fails_before_the_export_starts(tmp_path, monkeypatch):
    """The failure this guards against completed "successfully" as hours of silence:
    every sentence raised inside dia, became empty audio, and was retried and
    skipped, with the real cause only in per-sentence warnings."""
    def boom(*a, **k):
        raise RuntimeError("Could not load libtorchcodec. FFmpeg is not properly installed")

    monkeypatch.setattr("torchaudio.load", boom)
    with pytest.raises(voice.ReferenceAudioUnreadable) as excinfo:
        voice.check_reference_readable(_ref(tmp_path))

    message = str(excinfo.value)
    assert "Latradio" in message
    assert "Gyan.FFmpeg.Shared" in message, "the error must say how to fix it"


def test_readable_reference_audio_passes(tmp_path, monkeypatch):
    import torch

    monkeypatch.setattr("torchaudio.load", lambda *a, **k: (torch.zeros(1, 44100), 44100))
    voice.check_reference_readable(_ref(tmp_path))  # must not raise


def test_stereo_reference_audio_is_rejected_up_front(tmp_path, monkeypatch):
    """A stereo prompt loads fine and then fails inside the DAC encoder once per
    sentence — the real NamMN1.wav cost four sentences of a chapter that way,
    reported only as "produced no audio", before being traced back to its channel
    count."""
    import torch

    monkeypatch.setattr("torchaudio.load", lambda *a, **k: (torch.zeros(2, 44100), 44100))
    with pytest.raises(voice.ReferenceAudioUnreadable) as excinfo:
        voice.check_reference_readable(_ref(tmp_path))
    assert "2 channels" in str(excinfo.value)
    assert "mono" in str(excinfo.value)


def test_add_voice_downmixes_a_stereo_reference(tmp_path, monkeypatch):
    """The fix has to happen at registration; the model never sees a second channel."""
    import numpy as np
    import soundfile as sf

    refs = tmp_path / "voice_refs"
    refs.mkdir()
    monkeypatch.setattr(config, "VOICE_REF_DIR", refs)
    monkeypatch.setattr(config, "ensure_dirs", lambda: None)
    monkeypatch.setattr(config.Settings, "save", lambda self: None)

    src = tmp_path / "stereo.wav"
    left, right = _tone(2.0, 0.4), _tone(2.0, 0.2)
    sf.write(str(src), np.stack([left, right], axis=1), 44100)
    assert sf.info(str(src)).channels == 2

    settings = config.Settings()
    ref = voice.add_voice(settings, "Stereo1", src, "lời thoại")

    assert sf.info(str(ref.wav_path)).channels == 1, "stereo must not reach the model"
    written, _ = sf.read(str(ref.wav_path), dtype="float32")
    # the downmix is the mean of the channels, then scaled to the loudness target
    expected_shape = (left + right) / 2
    scale = np.max(np.abs(written)) / np.max(np.abs(expected_shape))
    assert np.allclose(written, expected_shape * scale, atol=2e-4)


def test_a_clip_too_short_to_measure_is_left_alone(tmp_path, monkeypatch):
    """The loudness meter needs a 400ms block and refuses less. That is fine for a
    meter and not a reason to fail registering a voice."""
    import numpy as np
    import soundfile as sf

    from diareader import audio_io

    refs = tmp_path / "voice_refs"
    refs.mkdir()
    monkeypatch.setattr(config, "VOICE_REF_DIR", refs)
    monkeypatch.setattr(config, "ensure_dirs", lambda: None)
    monkeypatch.setattr(config.Settings, "save", lambda self: None)

    tiny = _tone(0.1, 0.3)
    assert audio_io.measure_loudness(tiny, 44100) == float("-inf")
    assert np.array_equal(audio_io.normalize_loudness(tiny, 44100), tiny)

    src = tmp_path / "tiny.wav"
    sf.write(str(src), tiny, 44100)
    ref = voice.add_voice(config.Settings(), "Tiny", src, "a")   # must not raise
    assert ref.wav_path.exists()


def test_overlong_reference_audio_is_flagged(tmp_path, monkeypatch, caplog):
    """The real NamMN1.wav: 21.8s of speech behind a transcript worth ~9s. It loads,
    it is mono, it generates — and the model drifts, because the decoder was prefilled
    with 12 seconds the encoder never saw."""
    import logging

    import numpy as np
    import soundfile as sf

    refs = tmp_path / "voice_refs"
    refs.mkdir()
    monkeypatch.setattr(config, "VOICE_REF_DIR", refs)
    monkeypatch.setattr(config, "ensure_dirs", lambda: None)
    monkeypatch.setattr(config.Settings, "save", lambda self: None)

    src = tmp_path / "long.wav"
    sf.write(str(src), np.zeros(22 * 44100, dtype=np.float32), 44100)  # 22s

    with caplog.at_level(logging.WARNING, logger="diareader.voice"):
        voice.add_voice(config.Settings(), "TooLong", src, "Một câu ngắn.")

    assert any("transcript" in r.message for r in caplog.records), caplog.text


def test_a_matching_reference_is_not_flagged(tmp_path, monkeypatch, caplog):
    import logging

    import numpy as np
    import soundfile as sf

    refs = tmp_path / "voice_refs"
    refs.mkdir()
    monkeypatch.setattr(config, "VOICE_REF_DIR", refs)
    monkeypatch.setattr(config, "ensure_dirs", lambda: None)
    monkeypatch.setattr(config.Settings, "save", lambda self: None)

    transcript = "Những giây phút cuối cùng trước thềm năm mới luôn thật nhiều cảm xúc."
    seconds = len(transcript) / 20.3 * 1.4  # inside the healthy 1.15-1.47x band
    src = tmp_path / "ok.wav"
    sf.write(str(src), np.zeros(int(seconds * 44100), dtype=np.float32), 44100)

    with caplog.at_level(logging.WARNING, logger="diareader.voice"):
        voice.add_voice(config.Settings(), "Fine", src, transcript)

    assert not [r for r in caplog.records if "transcript" in r.message], caplog.text


def _tone(seconds, amplitude, rate=44100):
    import numpy as np

    t = np.linspace(0, seconds, int(rate * seconds), endpoint=False)
    return (np.sin(2 * np.pi * 220 * t) * amplitude).astype(np.float32)


def test_a_quiet_reference_is_brought_up_to_the_others(tmp_path, monkeypatch):
    """The model takes its level from the prompt, so a quiet reference makes a quiet
    narrator. Six clips added by hand spanned 17.1 LU — the quietest 7.17x smaller in
    amplitude than the loudest."""
    import soundfile as sf

    from diareader import audio_io

    refs = tmp_path / "voice_refs"
    refs.mkdir()
    monkeypatch.setattr(config, "VOICE_REF_DIR", refs)
    monkeypatch.setattr(config, "ensure_dirs", lambda: None)
    monkeypatch.setattr(config.Settings, "save", lambda self: None)

    src = tmp_path / "quiet.wav"
    sf.write(str(src), _tone(3.0, 0.02), 44100)          # very quiet
    quiet_before = audio_io.measure_loudness(_tone(3.0, 0.02), 44100)

    ref = voice.add_voice(config.Settings(), "Quiet", src, "lời thoại")
    installed, rate = sf.read(str(ref.wav_path), dtype="float32")

    after = audio_io.measure_loudness(installed, rate)
    assert after > quiet_before + 6, "a whisper-level clip must be lifted"
    assert after == pytest.approx(audio_io.TARGET_LUFS, abs=1.0)


def test_normalising_never_clips(tmp_path, monkeypatch):
    """Gain, not limiting: a clip that cannot reach the target without clipping is
    left quieter rather than having its peaks squashed, because squashing changes the
    voice's character — the one thing a reference exists to preserve."""
    from diareader import audio_io

    # loud peaks, quiet body: reaching the target would push the peaks past 0 dBFS
    import numpy as np

    audio = _tone(3.0, 0.02)
    audio[1000:1100] = 0.9
    out = audio_io.normalize_loudness(audio, 44100)

    peak_dbfs = 20 * np.log10(float(np.max(np.abs(out))))
    assert peak_dbfs <= audio_io.PEAK_CEILING_DBFS + 0.01, f"clipped at {peak_dbfs:.2f} dBFS"


def test_a_loud_reference_is_brought_down(tmp_path, monkeypatch):
    from diareader import audio_io

    loud = _tone(3.0, 0.9)
    out = audio_io.normalize_loudness(loud, 44100)
    assert audio_io.measure_loudness(out, 44100) == pytest.approx(
        audio_io.TARGET_LUFS, abs=1.0
    )


def test_normalize_existing_references_reports_what_moved(tmp_path, monkeypatch):
    import soundfile as sf

    refs = tmp_path / "voice_refs"
    refs.mkdir()
    monkeypatch.setattr(config, "VOICE_REF_DIR", refs)
    sf.write(str(refs / "Quiet.wav"), _tone(3.0, 0.02), 44100)
    sf.write(str(refs / "Loud.wav"), _tone(3.0, 0.9), 44100)

    settings = config.Settings(voices={
        "Quiet": {"wav": "Quiet.wav", "transcript": "a"},
        "Loud": {"wav": "Loud.wav", "transcript": "b"},
    })
    changed = dict((name, (b, a)) for name, b, a in
                   voice.normalize_existing_references(settings))

    assert set(changed) == {"Quiet", "Loud"}
    from diareader import audio_io
    for name in ("Quiet", "Loud"):
        audio, rate = sf.read(str(refs / f"{name}.wav"), dtype="float32")
        assert audio_io.measure_loudness(audio, rate) == pytest.approx(
            audio_io.TARGET_LUFS, abs=1.0
        )

    # already normalised: a second pass must be a no-op, not a slow drift
    assert voice.normalize_existing_references(settings) == []


def test_add_voice_preserves_the_recording_while_levelling_it(tmp_path, monkeypatch):
    """Normalisation is a gain change and nothing else: same samples, same rate, same
    duration, only scaled. Anything more would alter the voice the clip exists to
    capture."""
    import numpy as np
    import soundfile as sf

    refs = tmp_path / "voice_refs"
    refs.mkdir()
    monkeypatch.setattr(config, "VOICE_REF_DIR", refs)
    monkeypatch.setattr(config, "ensure_dirs", lambda: None)
    monkeypatch.setattr(config.Settings, "save", lambda self: None)

    src = tmp_path / "mono.wav"
    original = _tone(2.0, 0.05)
    sf.write(str(src), original, 44100, subtype="PCM_16")

    ref = voice.add_voice(config.Settings(), "Mono1", src, "lời thoại")
    installed, rate = sf.read(str(ref.wav_path), dtype="float32")

    assert rate == 44100
    assert installed.ndim == 1
    assert len(installed) == len(original), "duration must not change"
    # a pure gain: every sample scaled by the same factor
    scale = np.max(np.abs(installed)) / np.max(np.abs(original))
    assert np.allclose(installed, original * scale, atol=2e-4)


def test_text_without_markers_keeps_start_voice():
    segments = chunking.split_by_voice("Câu một. Câu hai.", start_voice="GiongNam")
    assert segments == [("Câu một. Câu hai.", "GiongNam")]


def test_marker_switches_voice_and_is_not_spoken():
    text = "Người dẫn chuyện nói. [GiongNu2] Cô ấy trả lời."
    segments = chunking.split_by_voice(text, start_voice="GiongNam")

    assert [v for _, v in segments] == ["GiongNam", "GiongNu2"]
    joined = " ".join(t for t, _ in segments)
    assert "[GiongNu2]" not in joined, "the marker must never reach the model"
    assert "Cô ấy trả lời." in joined


def test_marker_at_start_overrides_start_voice():
    segments = chunking.split_by_voice("[GiongNu2] Xin chào.", start_voice="GiongNam")
    assert len(segments) == 1
    assert segments[0][1] == "GiongNu2"


def test_voice_persists_across_following_sentences():
    text = "Một. [GiongNu2] Hai. Ba. Bốn."
    chunks = chunking.chunk_text_with_voices(text, start_voice="GiongNam")
    voices = [c.voice for c in chunks]

    assert voices[0] == "GiongNam"
    assert voices[1:] == ["GiongNu2"] * (len(voices) - 1), (
        "a marker applies until the next one, not just to one sentence"
    )


def test_multiple_switches():
    text = "A. [GiongNu2] B. [GiongNam3] C."
    chunks = chunking.chunk_text_with_voices(text, start_voice="Start")
    assert [c.voice for c in chunks] == ["Start", "GiongNu2", "GiongNam3"]


def test_no_start_voice_yields_none():
    chunks = chunking.chunk_text_with_voices("Xin chào.")
    assert chunks[0].voice is None


def test_markers_do_not_survive_into_chunk_text():
    chunks = chunking.chunk_text_with_voices("[GiongNu2] Xin chào. Tạm biệt.")
    assert not any("[" in c.text for c in chunks)


# --- voice names must be usable inside a marker ------------------------------------

import pytest

from diareader import voice as voice_mod


@pytest.mark.parametrize("name", ["NuMN1", "Latradio", "Giong_Nam-2", "A"])
def test_marker_safe_names_are_accepted(name):
    assert voice_mod.validate_name(name) == name


@pytest.mark.parametrize("name", ["Nữ (MN1)", "Giong Nam", "voice!", "", "   ", "x" * 41])
def test_names_a_marker_could_never_match_are_rejected(name):
    """Such a name registers happily and then never matches, so the voice is silently
    ignored at generation time — catch it at naming instead."""
    with pytest.raises(voice_mod.InvalidVoiceName):
        voice_mod.validate_name(name)


def test_accepted_names_actually_match_the_marker_pattern():
    """The validator and the marker regex must not drift apart."""
    for name in ["NuMN1", "Latradio", "Giong_Nam-2"]:
        assert chunking.VOICE_MARKER.fullmatch(f"[{name}]"), name
