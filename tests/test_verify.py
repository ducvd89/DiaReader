"""The text side of the quality check: how many syllables a sentence asks for.

The audio side lives in test_asr_verify. It used to live here too — verify.py counted
peaks in an energy envelope — and that counter was retired rather than kept as an
option, because it could not resolve five syllables in a one-second greeting and could
not see a substitution at all. See asr_verify for the measurements.
"""
import numpy as np
import pytest

from diareader import config, pipeline, verify


def test_a_sentence_counts_its_words():
    assert verify.syllables("Xin chào các bạn") == ["xin", "chào", "các", "bạn"]


def test_punctuation_is_not_a_syllable():
    assert verify.syllables("Thật vậy sao?!") == ["thật", "vậy", "sao"]


def test_non_verbal_tags_are_not_counted():
    """They are performed, not spoken, so the audio should not contain sounds for
    them — counting them would make every tagged sentence look short."""
    assert "laughs" not in " ".join(verify.syllables("Buồn cười thật (laughs)"))


def test_numbers_are_counted_as_they_are_read():
    """One token, five syllables. See vn_numbers for which way this error ran before
    it was fixed: under-counting what the text asks for inflates the score."""
    assert len(verify.syllables("123")) == 5
    assert verify.syllables("có 8.000 người") == ["có", "tám", "nghìn", "người"]


def test_empty_text_asks_for_nothing():
    assert verify.syllables("") == []
    assert verify.syllables("   ") == []


@pytest.fixture
def no_download(monkeypatch):
    """_make_verifier loads the recogniser eagerly so a broken one is reported before
    an export starts. These tests are about which checker is chosen, not about it
    working, and a 378 MB download does not belong in them."""
    from diareader import asr_verify

    monkeypatch.setattr(asr_verify.AsrCounter, "_load", lambda self: None)


def test_the_default_check_listens(no_download):
    verifier = pipeline._make_verifier(config.Settings())
    from diareader.asr_verify import AsrCounter

    assert isinstance(verifier, AsrCounter)


def test_checking_can_be_turned_off():
    settings = config.Settings()
    settings.verify_mode = "off"
    assert pipeline._make_verifier(settings) is None


def test_an_unknown_mode_still_checks(no_download):
    """A settings file outlives the code that wrote it, and every retired mode name is
    in someone's file. Falling back to no checking at all would be the silent kind of
    wrong this app keeps running into."""
    settings = config.Settings()
    settings.verify_mode = "syllables"       # retired
    assert pipeline._make_verifier(settings) is not None
    settings.verify_mode = "whisper"         # retired earlier still
    assert pipeline._make_verifier(settings) is not None


def test_the_length_fallback_still_rejects_silence():
    """What "off" leaves behind. Poor at judging a sentence, but a chunk that produced
    nothing at all must not sail through."""
    text = "Một câu dài vừa đủ để không bị coi là một tiêu đề ngắn ngủn."
    assert not pipeline._is_acceptable(text, np.zeros(0, dtype=np.float32))
