"""Words a Vietnamese syllable count does not describe, and the headroom they earn.

The case this exists for is in a real export log: "SmartBanking" is one token and was
read as four syllables, which put a sixteen-syllable sentence at 119% — over the
ceiling — so it was regenerated ten times and then recorded as missing text. It appears
eleven times in that book.
"""
import pytest

from diareader import foreign_words


@pytest.mark.parametrize("word", [
    # Undiacriticked Vietnamese, which is why plain ASCII cannot be the test.
    "anh", "em", "thanh", "chi", "cho", "hai", "ra", "khi", "trong", "xin",
    "ngay", "sau", "cao", "sinh", "vui", "sang", "nhu", "giao", "Nam", "Minh",
    # And with their diacritics on.
    "tiếng", "nghiêng", "đường", "khoản", "quyết", "chuyện",
])
def test_vietnamese_words_are_one_syllable(word):
    assert foreign_words.is_vietnamese_syllable(word)
    assert foreign_words.extra_syllables_allowed(word) == 0


@pytest.mark.parametrize("word", [
    "SmartBanking", "smart", "banking", "internet", "Facebook", "app",
    "OTP", "SMS", "wifi", "email", "Zalo",
])
def test_foreign_words_are_recognised(word):
    assert not foreign_words.is_vietnamese_syllable(word)


def test_letters_absent_from_the_alphabet_settle_it():
    """f, j, w and z are not Vietnamese letters, so a token containing one cannot be
    a Vietnamese syllable however it is otherwise shaped."""
    for word in ("wifi", "jazz", "zebra", "fan"):
        assert not foreign_words.is_vietnamese_syllable(word)


def test_an_ordinary_vietnamese_sentence_earns_no_headroom():
    """Almost every sentence, so almost every sentence keeps the ceiling exactly where
    the measurements put it."""
    text = "Anh vui lòng cho em xin số tài khoản của mình nhé."
    assert foreign_words.foreign_words(text) == []
    assert foreign_words.extra_syllables_allowed(text) == 0


def test_the_sentence_that_prompted_this():
    text = "Anh vui lòng làm theo các bước sau để mở ngay trên ứng dụng SmartBanking nhé."
    assert foreign_words.foreign_words(text) == ["SmartBanking"]
    # Counted as one syllable, spoken as four. The allowance has to cover the gap.
    assert foreign_words.extra_syllables_allowed(text) >= 3


def test_an_acronym_is_spelled_out():
    """OTP is "ô tê pê" — three syllables for one token."""
    assert foreign_words.extra_syllables_allowed("mã OTP") >= 2


def test_camel_case_is_split_before_counting():
    """SmartBanking is written as one word and said as two."""
    assert (foreign_words.extra_syllables_allowed("SmartBanking")
            > foreign_words.extra_syllables_allowed("banking"))


def test_numbers_and_punctuation_are_not_words():
    assert foreign_words.foreign_words("mã 8.000, số 47,5!") == []


# --- a tone mark is part of the spelling, not decoration on it -----------------------


def test_gi_with_a_tone_is_vietnamese_and_without_one_is_not():
    """The case that made this rule necessary: "gì" was being counted as a foreign
    word seven times across one book, handing those sentences a syllable of ceiling
    they had not earned."""
    assert foreign_words.is_vietnamese_syllable("gì")
    assert not foreign_words.is_vietnamese_syllable("GI")
    assert not foreign_words.is_vietnamese_syllable("gi")


@pytest.mark.parametrize("word", [
    "gì", "gí", "gỉ", "gĩ", "gị",          # the whole tone set on the same spelling
    "già", "giá", "giữ", "gia",            # gi- as an onset, which always did work
    "tiếng", "đường", "Đông", "cũng", "ạ", "ở",
])
def test_a_word_wearing_vietnamese_marks_is_vietnamese(word):
    assert foreign_words.is_vietnamese_syllable(word)


@pytest.mark.parametrize("word", ["señor", "naïve", "Müller", "Ångström"])
def test_marks_from_other_languages_do_not_count(word):
    """The rule is safe only because these marks on these bases are Vietnamese and
    nothing else. A tilde over an n is not one of them."""
    assert not foreign_words.is_vietnamese_syllable(word)


@pytest.mark.parametrize("word", ["SmartBanking", "Smart", "Banking", "online", "OTP"])
def test_the_real_foreign_words_are_still_caught(word):
    assert not foreign_words.is_vietnamese_syllable(word)


def test_a_marked_word_containing_a_foreign_letter_is_still_foreign():
    """f, j, w and z are not in the alphabet, and a tone mark cannot put them there."""
    assert not foreign_words.is_vietnamese_syllable("wờ")


def test_the_sentence_that_was_being_given_free_headroom():
    """Real text from one of the books. Its only "foreign" word was "gì"."""
    sentence = ("Trong quá trình sử dụng sản phẩm dịch vụ của Bi Đi Vi, anh Minh có "
                "gặp khó khăn hoặc vướng mắc gì cần em hỗ trợ không ạ.")
    assert foreign_words.foreign_words(sentence) == []
    assert foreign_words.extra_syllables_allowed(sentence) == 0


def test_a_sentence_with_a_real_english_word_still_gets_headroom():
    sentence = "Anh Minh có thể dùng SmartBanking để chuyển tiền gì cũng được ạ."
    assert foreign_words.foreign_words(sentence) == ["SmartBanking"]
    assert foreign_words.extra_syllables_allowed(sentence) > 0
