"""The retry budget, and what a sentence is ranked by when none of the attempts pass.

Ten attempts because the failures are stochastic and more draws recover more sentences.
There was once an early exit for scores that stopped changing; it rested on the counter
being unreliable between 0.72 and 0.85, which turned out to be one clip with a bad
transcript and another that was a poor recording. Every reference clip now reads 0.88
or better, so that band holds no known-good speech and the budget is spent in full.
"""
import numpy as np

from diareader import asr_verify, config, pipeline, verify

SENTENCE = "Một câu dài vừa đủ để không bị coi là một tiêu đề ngắn ngủn."


_SAMPLES_PER_SYLLABLE = 4410  # 0.1s, so a clip's length carries its syllable count


def _pulses(count: int) -> np.ndarray:
    """Audio standing for `count` recognised syllables."""
    return np.ones(count * _SAMPLES_PER_SYLLABLE, dtype=np.float32)


def _encoded_count(audio: np.ndarray) -> int:
    return int(round(len(audio) / _SAMPLES_PER_SYLLABLE))


def _syllables_scoring(low: float, high: float) -> int:
    """A syllable count landing inside [low, high) of what the sentence asks for."""
    expected = len(verify.syllables(SENTENCE))
    for count in range(0, expected * 3):
        if low <= count / expected < high:
            return count
    raise AssertionError(f"no count scores within [{low}, {high})")


class ScriptedEngine:
    def __init__(self, nuclei_per_attempt):
        self.script = list(nuclei_per_attempt)
        self.calls = 0

    def generate(self, tagged_text, voice_ref=None, **overrides):
        self.calls += 1
        return _pulses(self.script[min(self.calls - 1, len(self.script) - 1)])


class _Verifier:
    """The real acceptance rule, fed a syllable count encoded in the audio length.

    Standing in for the recogniser rather than running it: this is testing the retry
    loop, and a 95M-parameter model in the middle of it would be measuring the wrong
    thing at a hundred times the cost.
    """

    def check(self, text, audio, sample_rate=config.SAMPLE_RATE):
        expected = len(verify.syllables(text))
        heard = _encoded_count(audio)
        score = heard / expected
        ok = expected <= heard <= expected * asr_verify.RATIO_MAX
        return ok, score, f"{heard} of {expected} syllables"


def test_the_budget_is_ten():
    assert pipeline._MAX_ATTEMPTS == 10


def test_there_is_no_early_exit_left():
    """A stable low score used to stop after three. It runs the full budget now."""
    stuck = _syllables_scoring(0.25, 0.4)
    engine = ScriptedEngine([stuck] * 12)
    pipeline._generate_chunk(engine, SENTENCE, None, "Ch. 1", 0, verifier=_Verifier())
    assert engine.calls == pipeline._MAX_ATTEMPTS


def test_a_stable_score_just_under_the_bar_also_runs_the_full_budget():
    """The band the old rule fired in. Nothing special happens there any more."""
    near = _syllables_scoring(0.55, 0.90)
    engine = ScriptedEngine([near] * 12)
    pipeline._generate_chunk(engine, SENTENCE, None, "Ch. 1", 0, verifier=_Verifier())
    assert engine.calls == pipeline._MAX_ATTEMPTS


def test_an_acceptable_result_returns_at_once():
    engine = ScriptedEngine([len(verify.syllables(SENTENCE))] * 10)
    pipeline._generate_chunk(engine, SENTENCE, None, "Ch. 1", 0, verifier=_Verifier())
    assert engine.calls == 1


def test_batched_audio_counts_as_one_of_the_ten():
    """A batch failure must cost the same generations as a solo one, not one more."""
    stuck = _syllables_scoring(0.25, 0.4)
    engine = ScriptedEngine([stuck] * 12)
    pipeline._generate_chunk(
        engine, SENTENCE, None, "Ch. 1", 0,
        verifier=_Verifier(), initial=_pulses(stuck),
    )
    assert engine.calls == pipeline._MAX_ATTEMPTS - 1


def test_the_accepted_clip_is_returned_not_the_highest_scoring_one():
    """The acceptance band is two-sided, so "best" cannot mean "largest". A rejected
    loop scoring far over the ceiling would otherwise beat the attempt that passed."""
    expected = len(verify.syllables(SENTENCE))
    loop_count = _syllables_scoring(asr_verify.RATIO_MAX + 0.2, 3.0)
    loop = _pulses(loop_count)
    engine = ScriptedEngine([loop_count, expected])

    audio = pipeline._generate_chunk(
        engine, SENTENCE, None, "Ch. 1", 0, verifier=_Verifier()
    ).audio

    assert engine.calls == 2
    assert len(audio) < len(loop), "returned the looping attempt instead of the good one"


def test_the_fallback_prefers_the_attempt_closest_to_a_full_reading():
    """When nothing passes, the sentence still has to ship something. Ranking by size
    alone would pick the worst failure the checker has — a clip of noise."""
    assert pipeline._distance(0.9) < pipeline._distance(2.5)
    assert pipeline._distance(1.1) < pipeline._distance(0.5)
    assert pipeline._distance(-1.0) == float("inf"), "the seed must lose to any real score"
