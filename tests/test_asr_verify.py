"""The acceptance rule, and the parts of the checker that need no model.

The recogniser itself is not loaded here — 95M parameters and a 378 MB download do not
belong in a unit test — so `transcribe` is stubbed and what is checked is the rule
built on top of it. The numbers that rule encodes came from clips judged by ear; they
are recorded in asr_verify's own comments.
"""
import numpy as np
import pytest

from diareader import asr_verify, config


@pytest.fixture
def counter(monkeypatch):
    """An AsrCounter whose recogniser returns whatever the test puts in `heard`."""
    checker = asr_verify.AsrCounter()
    checker.heard = ""
    monkeypatch.setattr(
        asr_verify.AsrCounter, "transcribe",
        lambda self, audio, rate: self.heard,
    )
    return checker


def _audio(seconds=1.0):
    return np.ones(int(seconds * config.SAMPLE_RATE), dtype=np.float32)


def test_the_band_is_the_measured_one():
    """0.95-1.05 was tried and rejects two of the seventeen human reference clips;
    0.90-1.10 admits all of them and still sits 0.30 clear of the worst truncation."""
    assert (asr_verify.RATIO_MIN, asr_verify.RATIO_MAX) == (0.90, 1.10)


def test_a_sentence_read_in_full_is_accepted(counter):
    counter.heard = "dạ em chào anh minh"
    ok, score, _ = counter.check("Dạ, em chào anh Minh", _audio())
    assert ok and score == 1.0


def test_word_errors_do_not_matter(counter):
    """"pháp luật" for "pháp lực" is still two syllables. The recogniser mishears this
    app's vocabulary constantly — cultivation jargon is not in its training data — and
    the check has to survive that."""
    counter.heard = "xem bộ giáng pháp luật của mấy người này"
    ok, score, _ = counter.check("Xem bộ dáng pháp lực của mấy người này", _audio())
    assert ok and score == 1.0


def test_a_dropped_clause_is_rejected(counter):
    """The case peak counting scored a perfect 5 of 5: two words wrong, and it heard
    the right number of sounds anyway."""
    counter.heard = "dạ em chào"
    ok, score, _ = counter.check("Dạ, em chào anh Minh", _audio())
    assert not ok
    assert score == pytest.approx(0.6)


def test_an_extra_word_is_rejected(counter):
    """A real take rendered "Nam dạ em chào anh Minh" — six syllables for five."""
    counter.heard = "nam dạ em chào anh minh"
    ok, _, _ = counter.check("Dạ, em chào anh Minh", _audio())
    assert not ok


def test_silence_is_rejected(counter):
    counter.heard = ""
    ok, score, _ = counter.check("Dạ, em chào anh Minh", _audio())
    assert not ok and score == 0.0


def test_short_text_gets_no_extra_tolerance(counter):
    """Peak counting needed ±1 because it could not resolve a short line. This counted
    all fifteen short clips exactly right, so ±1 on five syllables would just be
    admitting a fifth of the sentence missing."""
    counter.heard = "dạ em chào anh"          # one short of five
    ok, _, _ = counter.check("Dạ, em chào anh Minh", _audio())
    assert not ok


def test_a_foreign_word_raises_only_the_ceiling(counter):
    """"SmartBanking" is one token and four spoken syllables. Without headroom the
    sentence reads 119% and is regenerated ten times — which is what happened."""
    text = "Anh vui lòng làm theo các bước sau để mở ngay trên ứng dụng SmartBanking nhé."
    counter.heard = ("anh vui lòng làm theo các bước sau để mở ngay trên ứng dụng "
                     "sờ mát ben kin nhé")
    ok, score, detail = counter.check(text, _audio())
    assert ok, f"scored {score:.2f} and was rejected"
    assert score > asr_verify.RATIO_MAX, "the point is that the plain ratio is over"
    assert "SmartBanking" in detail


def test_the_floor_is_not_raised_with_it(counter):
    """The error only runs one way — the text under-counts what gets spoken — so a
    sentence that lost half its words is still caught, foreign word or not."""
    text = "Anh vui lòng làm theo các bước sau để mở ngay trên ứng dụng SmartBanking nhé."
    counter.heard = "anh vui lòng làm theo"
    ok, _, _ = counter.check(text, _audio())
    assert not ok


def test_headroom_does_not_excuse_a_loop(counter):
    """It buys a few syllables, not a repeated clause."""
    text = "Anh vui lòng làm theo các bước sau để mở ngay trên ứng dụng SmartBanking nhé."
    counter.heard = ("anh anh vui lòng làm theo các bước sau để mở ngay trên ứng dụng "
                     "sờ mát ben kin nhé mở ngay trên ứng dụng sờ mát")
    ok, _, _ = counter.check(text, _audio())
    assert not ok


def test_the_transcript_is_in_the_detail(counter):
    """The point of the whole change for anyone reading a log: "36% of the sounds"
    says a sentence is wrong, the transcript says how."""
    counter.heard = "dạ em chào"
    _, _, detail = counter.check("Dạ, em chào anh Minh", _audio())
    assert "dạ em chào" in detail
    assert "3 of 5" in detail


def test_text_with_nothing_to_say_is_accepted(counter):
    ok, score, _ = counter.check("(laughs)", _audio())
    assert ok and score == 1.0


def test_a_broken_recogniser_does_not_abort_the_export(monkeypatch):
    """A checker that raises would end a multi-hour run over a sentence that may well
    be fine. It reports and steps aside instead."""
    checker = asr_verify.AsrCounter()
    monkeypatch.setattr(
        asr_verify.AsrCounter, "transcribe",
        lambda self, audio, rate: (_ for _ in ()).throw(RuntimeError("no model")),
    )
    ok, _, detail = checker.check("Dạ, em chào anh Minh", _audio())
    assert ok
    assert "unavailable" in detail


def test_audio_too_short_to_transcribe_reads_as_nothing():
    """The recogniser's convolutional front end needs more samples than its kernel,
    and a failed take measured 0.01s. That must read as silence, not raise."""
    checker = asr_verify.AsrCounter()
    assert checker.transcribe(np.ones(100, dtype=np.float32), config.SAMPLE_RATE) == ""


def test_release_drops_the_model():
    checker = asr_verify.AsrCounter()
    checker._model = object()
    checker._processor = object()
    checker.release()
    assert checker._model is None and checker._processor is None
