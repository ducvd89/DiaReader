import numpy as np

from diareader import audio_io, chunking, config, pipeline


def _tone(seconds, amplitude, rate=config.SAMPLE_RATE):
    t = np.linspace(0, seconds, int(rate * seconds), endpoint=False)
    return (np.sin(2 * np.pi * 220 * t) * amplitude).astype(np.float32)


class FakeEngine:
    """Stands in for TtsEngine so these run without a GPU or the 6.4GB checkpoint."""

    def __init__(self, durations_s):
        self.durations = list(durations_s)
        self.calls = []

    def generate(self, tagged_text, voice_ref=None, **overrides):
        self.calls.append((tagged_text, overrides))
        seconds = self.durations.pop(0) if self.durations else 0.0
        return np.ones(int(seconds * config.SAMPLE_RATE), dtype=np.float32)


def test_empty_audio_is_skipped_not_fatal():
    # dia's DAC decode raises when the model emits zero tokens; TtsEngine turns that
    # into empty audio, and one bad sentence must not abort a whole export.
    engine = FakeEngine([0.0, 0.0, 0.0])
    audio = pipeline._generate_chunk(engine, "Quyển 2", None, "Ch. 1", 0).audio

    assert len(audio) > 0, "should substitute silence rather than return nothing"
    assert len(engine.calls) == pipeline._MAX_ATTEMPTS


def test_short_text_is_not_retried_for_coverage():
    # The ~20 chars/sec figure comes from 9-30s clips and does not describe a heading;
    # applying it there caused pointless retries on a real book's contents page.
    engine = FakeEngine([0.5])
    pipeline._generate_chunk(engine, "Mục lục", None, "Ch. 1", 0)
    assert len(engine.calls) == 1


def test_tiny_chunk_is_not_retried_when_it_already_over_delivers():
    """Straight from a real export: a 1-char chunk returned 0.1s of audio — 118% of
    the 0.049s its text is worth — and the flat 0.15s "produced nothing" floor
    rejected it three times, spending three generations to replace audio that was
    already over-length."""
    engine = FakeEngine([0.1])
    pipeline._generate_chunk(engine, "A", None, "Ch. 1", 0)
    assert len(engine.calls) == 1, "over-length audio must not be treated as empty"


def test_genuinely_empty_audio_still_retries():
    """The floor still has to catch real failures on text long enough to expect
    more than a fraction of a second."""
    engine = FakeEngine([0.0, 0.0, 0.0])
    pipeline._generate_chunk(engine, "Một câu văn hoàn chỉnh và đủ dài.", None, "Ch. 1", 0)
    assert len(engine.calls) == pipeline._MAX_ATTEMPTS


def test_a_nonverbal_tag_does_not_count_as_speech_to_be_covered():
    """Straight from the ASR run: "(clears throat) Chúng ta cũng nên xuất phát…" was
    judged 55.6% covered and regenerated three times, while Whisper found 88.2% of
    its syllables present. The 0.74s the check was hunting for was the tag itself."""
    text = "(clears throat) Chúng ta cũng nên xuất phát, hiện tại trong sảnh không còn."
    spoken = chunking.spoken_text(text)
    assert "(clears throat)" not in spoken

    # audio long enough for the spoken part, but short of the raw string's length
    seconds = len(spoken) / pipeline._NATURAL_CHARS_PER_SECOND
    engine = FakeEngine([seconds])
    pipeline._generate_chunk(engine, text, None, "Ch. 1", 0)
    assert len(engine.calls) == 1, "the tag's characters must not create a shortfall"


def test_a_chunk_that_is_only_a_nonverbal_tag_is_not_judged_on_length():
    engine = FakeEngine([0.3])
    pipeline._generate_chunk(engine, "(laughs)", None, "Ch. 1", 0)
    assert len(engine.calls) == 1


def test_long_text_retries_when_audio_is_too_short():
    text = "Đây là một câu văn dài " * 4  # ~92 chars -> ~4.5s expected
    engine = FakeEngine([0.5, 0.5, 4.6])
    audio = pipeline._generate_chunk(engine, text, None, "Ch. 1", 0).audio

    assert len(engine.calls) == 3
    assert engine.calls[1][1].get("temperature") == pipeline._RETRY_TEMPERATURE
    assert len(audio) / config.SAMPLE_RATE > 4.0, "should keep the best attempt"


def test_output_is_levelled_per_sentence():
    """Normalising the reference clips was necessary but not sufficient: with every
    reference at -20 LUFS the narration still spanned 10.1 LU, because the model adds
    its own per-voice offset. The only place to fix it for good is on the way out."""
    from diareader import text_extract

    # Spread comparable to what the model really produces across voices: measured
    # output ran -24.1 to -13.9 LUFS from references that were all at -20.
    quiet = _tone(2.0, 0.1)
    loud = _tone(2.0, 0.6)
    assert abs(audio_io.measure_loudness(quiet, config.SAMPLE_RATE)
               - audio_io.measure_loudness(loud, config.SAMPLE_RATE)) > 10

    class TwoLevels:
        def __init__(self):
            self.clips = [quiet, loud]

        def generate(self, tagged_text, voice_ref=None, **overrides):
            return self.clips.pop(0)

    chapters = [text_extract.Chapter(title="Ch", text="Câu một dài hơn. Câu hai dài hơn.")]
    settings = config.Settings(voices={}, speed=1.0)
    results = list(pipeline.iter_book_audio(
        "book.txt", TwoLevels(), settings, chapters=chapters))

    levels = [audio_io.measure_loudness(r.audio, config.SAMPLE_RATE) for r in results]
    assert abs(levels[0] - levels[1]) < 1.5, (
        f"a quiet and a loud sentence must end up within a hair of each other, "
        f"got {levels[0]:.1f} and {levels[1]:.1f} LUFS"
    )


def test_a_near_silent_failure_is_not_amplified_to_full_volume():
    """An almost-empty chunk is a generation failure, not a quiet delivery — lifting
    it to the target would just make its noise floor audible."""
    from diareader import audio_io

    nearly_silent = _tone(2.0, 0.0004)
    out = audio_io.normalize_loudness(nearly_silent, config.SAMPLE_RATE,
                                      max_boost_db=audio_io.MAX_BOOST_DB)
    gain = np.max(np.abs(out)) / np.max(np.abs(nearly_silent))
    assert 20 * np.log10(gain) <= audio_io.MAX_BOOST_DB + 0.01


def test_wrap_for_narration_adds_missing_terminal_mark():
    # 99/100 training transcripts end in terminal punctuation; bare fragments are what
    # made the model emit zero tokens.
    assert chunking.wrap_for_narration("Quyển 2") == "Quyển 2."
    assert chunking.wrap_for_narration("Xin chào.") == "Xin chào."
    assert chunking.wrap_for_narration("Thật sao?") == "Thật sao?"
