"""Model-free output verification: counting the sounds in the audio.

The counter runs on synthesised waveforms here, so none of this needs a GPU or the
checkpoint. The pipeline takes the checker by injection, so exports are testable too.
"""
import numpy as np
import pytest

from diareader import config, pipeline, verify

SR = config.SAMPLE_RATE


def speech_like(n_syllables: int, syllable_s: float = 0.22, gap_s: float = 0.08,
                sample_rate: int = SR) -> np.ndarray:
    """A waveform with `n_syllables` clear energy peaks separated by quiet gaps."""
    parts = []
    for _ in range(n_syllables):
        t = np.linspace(0, syllable_s, int(sample_rate * syllable_s), endpoint=False)
        # a vowel-ish tone under a raised-cosine envelope: one nucleus, one peak
        envelope = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(t.size) / t.size)
        parts.append((np.sin(2 * np.pi * 220 * t) * envelope).astype(np.float32))
        parts.append(np.zeros(int(sample_rate * gap_s), dtype=np.float32))
    return np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)


def test_counts_the_syllables_it_is_given():
    for n in (3, 8, 15):
        assert verify.count_syllable_nuclei(speech_like(n), SR) == n


def test_silence_contains_no_syllables():
    assert verify.count_syllable_nuclei(np.zeros(SR, dtype=np.float32), SR) == 0
    assert verify.count_syllable_nuclei(np.zeros(0, dtype=np.float32), SR) == 0


def test_text_syllables_ignore_punctuation_and_nonverbal_tags():
    assert verify.syllables("Xin chào, các bạn!") == ["xin", "chào", "các", "bạn"]
    # the tag is performed, not spoken — counting it would demand sounds that should
    # never be produced
    assert verify.syllables("(clears throat) Xin chào") == ["xin", "chào"]


def test_matching_audio_is_accepted():
    text = "Hai lão già đứng mỗi người một nơi trên mặt không lộ ra biểu tình"
    audio = speech_like(len(verify.syllables(text)))
    ok, score, _ = verify.SyllableCounter().check(text, audio)
    assert ok and score == pytest.approx(1.0)


def test_a_dropped_clause_is_rejected():
    """Half the sentence missing is exactly what the check exists to catch."""
    text = "Đương nhiên nếu không đi khi cái truyền tống trận kia biến mất thì hỏng hết"
    expected = len(verify.syllables(text))
    ok, score, _ = verify.SyllableCounter().check(text, speech_like(expected // 2))
    assert not ok
    assert score == pytest.approx(0.5, abs=0.05)


def test_looping_is_rejected_too():
    """The failure the old length check could not see: a two-syllable heading that
    came out as noise scored 269% 'coverage' and passed. Too many sounds is as wrong
    as too few."""
    text = "Một hai ba bốn năm sáu"
    expected = len(verify.syllables(text))
    ok, score, _ = verify.SyllableCounter().check(text, speech_like(expected * 3))
    assert not ok, "a flood of sounds must not pass"
    assert score > verify.SYLLABLE_RATIO_MAX


def test_the_normal_spread_of_healthy_audio_is_accepted():
    """Human speech reads 0.96-1.08 with the tuned parameters, and healthy generated
    audio mostly lands at or above 0.87."""
    counter = verify.SyllableCounter()
    text = " ".join(["âm"] * 20)
    for heard in (17, 18, 20, 21, 22):        # 0.85x to 1.10x
        ok, _, _ = counter.check(text, speech_like(heard))
        assert ok, f"{heard}/20 syllables should be within normal variation"


def test_a_real_shortfall_is_rejected():
    counter = verify.SyllableCounter()
    text = " ".join(["âm"] * 20)
    ok, score, _ = counter.check(text, speech_like(16))       # four syllables short
    assert not ok
    assert score == pytest.approx(0.80, abs=0.01)


def test_short_text_is_judged_by_count_not_percentage():
    """A ratio means nothing at this length: one missed nucleus in three syllables is
    already 33%. Within one sound of what the text asks for is what is actually meant."""
    counter = verify.SyllableCounter()
    text = "một hai ba bốn năm sáu"          # exactly the short-text limit
    assert len(verify.syllables(text)) == verify.SHORT_TEXT_SYLLABLES

    for heard in (5, 6, 7):                   # within +/-1
        ok, _, _ = counter.check(text, speech_like(heard))
        assert ok, f"{heard} of 6 sounds should be within tolerance"
    for heard in (4, 8):                      # outside it
        ok, _, _ = counter.check(text, speech_like(heard))
        assert not ok, f"{heard} of 6 sounds should be rejected"


def test_one_syllable_of_text_tolerates_one_extra_sound():
    """The rough edge the percentage rule left behind: a real export rejected '2
    sounds for ~1 syllables' as a 200% flood and regenerated it twice."""
    counter = verify.SyllableCounter()
    ok, _, _ = counter.check("Chương", speech_like(2))
    assert ok
    ok, _, _ = counter.check("Chương", speech_like(3))
    assert not ok


def test_short_text_still_fails_on_silence_and_on_noise():
    counter = verify.SyllableCounter()
    ok, _, _ = counter.check("Quyển bốn", np.zeros(SR, dtype=np.float32))
    assert not ok, "hearing nothing is always a failure"
    ok, _, _ = counter.check("Quyển bốn", speech_like(6))
    assert not ok, "a two-syllable heading rendered as six sounds is the gibberish case"


def test_the_boundary_between_the_two_rules_is_where_it_says():
    """Seven syllables uses the percentage; six uses the count."""
    counter = verify.SyllableCounter()
    six = " ".join(["âm"] * 6)
    seven = " ".join(["âm"] * 7)

    # 5 of 6 is within +/-1 -> accepted, though 0.83 would fail the ratio rule
    ok, score, _ = counter.check(six, speech_like(5))
    assert ok and score < verify.SYLLABLE_RATIO_MIN

    # 6 of 7 is 0.857 -> just clears the ratio rule
    ok, score, _ = counter.check(seven, speech_like(6))
    assert ok and score >= verify.SYLLABLE_RATIO_MIN


def test_a_chunk_with_no_syllables_is_never_failed():
    ok, _, _ = verify.SyllableCounter().check("(laughs)", speech_like(2))
    assert ok


class FakeVerifier:
    """Scripted verdicts, one per attempt."""

    def __init__(self, verdicts):
        self.verdicts = list(verdicts)
        self.calls = 0

    def check(self, source_text, audio, sample_rate=SR):
        self.calls += 1
        ok, score = self.verdicts.pop(0) if self.verdicts else (True, 1.0)
        return ok, score, "detail"


class FakeEngine:
    def __init__(self, durations_s):
        self.durations = list(durations_s)
        self.calls = []

    def generate(self, tagged_text, voice_ref=None, **overrides):
        self.calls.append((tagged_text, overrides))
        seconds = self.durations.pop(0) if self.durations else 0.0
        return np.ones(int(seconds * SR), dtype=np.float32)


def test_a_checker_accepts_audio_the_length_check_would_have_retried():
    """The whole point: audio short for its text but carrying all the sounds is fine
    and must not be regenerated."""
    text = "Vốn Hàn Lập không định truyền tống ở cùng một chỗ với thiếu nữ điều này lạ"
    engine = FakeEngine([1.0])                 # far shorter than the text predicts
    verifier = FakeVerifier([(True, 0.94)])

    pipeline._generate_chunk(engine, text, None, "Ch. 1", 0, verifier=verifier)

    assert len(engine.calls) == 1, "the sounds were all there; no retry was warranted"
    assert verifier.calls == 1


def test_a_checker_retries_audio_the_length_check_would_have_passed():
    text = "Hai lão già đứng mỗi người một nơi trên mặt không lộ ra biểu tình gì cả"
    n = pipeline._MAX_ATTEMPTS
    engine = FakeEngine([4.0] * n)
    verifier = FakeVerifier([(False, 0.1)] * n)

    pipeline._generate_chunk(engine, text, None, "Ch. 1", 0, verifier=verifier)

    assert len(engine.calls) == n
    assert verifier.calls == n


def test_a_late_attempt_can_still_rescue_the_sentence():
    """The reason for five attempts rather than three: these failures are stochastic,
    and every one seen in a real export was recovered by regenerating."""
    text = "Hai lão già đứng mỗi người một nơi trên mặt không lộ ra biểu tình gì cả"
    engine = FakeEngine([4.0] * pipeline._MAX_ATTEMPTS)
    verdicts = [(False, 0.3)] * (pipeline._MAX_ATTEMPTS - 1) + [(True, 0.98)]
    verifier = FakeVerifier(verdicts)

    pipeline._generate_chunk(engine, text, None, "Ch. 1", 0, verifier=verifier)

    assert len(engine.calls) == pipeline._MAX_ATTEMPTS
    assert verifier.calls == pipeline._MAX_ATTEMPTS


def test_the_best_attempt_is_kept_when_none_pass():
    text = "Một câu văn hoàn chỉnh và đủ dài để được kiểm tra cẩn thận ở đây"
    durations = [1.0, 5.0] + [2.0] * (pipeline._MAX_ATTEMPTS - 2)
    scores = [0.2, 0.55] + [0.3] * (pipeline._MAX_ATTEMPTS - 2)
    engine = FakeEngine(durations)
    verifier = FakeVerifier([(False, s) for s in scores])

    audio = pipeline._generate_chunk(engine, text, None, "Ch. 1", 0, verifier=verifier)

    assert len(audio) / SR == pytest.approx(5.0), "should keep the best attempt"


def test_syllable_counting_is_the_default_check():
    verifier = pipeline._make_verifier(config.Settings())
    assert isinstance(verifier, verify.SyllableCounter)


def test_duration_mode_falls_back_to_the_length_check():
    assert pipeline._make_verifier(config.Settings(verify_mode="duration")) is None
