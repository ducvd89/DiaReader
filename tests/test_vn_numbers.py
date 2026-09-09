"""Counting a written number as the syllables it is actually read as.

The syllable check compares sounds heard against syllables the text calls for, and
treated one whitespace-separated token as one syllable. True of Vietnamese words,
false of every number. The error ran one way: under-counting what a sentence calls for
inflates its ratio, so a sentence full of numbers looked better than it was.
"""
import pytest

from diareader import verify, vn_numbers


@pytest.mark.parametrize("written, spoken", [
    ("0", "không"),
    ("5", "năm"),
    ("10", "mười"),
    ("11", "mười một"),
    ("15", "mười lăm"),
    ("21", "hai mươi mốt"),
    ("24", "hai mươi tư"),
    ("25", "hai mươi lăm"),
    ("30", "ba mươi"),
    ("47", "bốn mươi bảy"),
    ("100", "một trăm"),
    ("105", "một trăm linh năm"),
    ("110", "một trăm mười"),
    ("123", "một trăm hai mươi ba"),
    ("1000", "một nghìn"),
    ("8000", "tám nghìn"),
    ("1975", "một nghìn chín trăm bảy mươi lăm"),
    ("2025", "hai nghìn không trăm hai mươi lăm"),
])
def test_integers_are_read_the_way_they_are_said(written, spoken):
    assert " ".join(vn_numbers.read_integer(written)) == spoken


@pytest.mark.parametrize("written, spoken", [
    ("8.000", "tám nghìn"),
    ("1.234", "một nghìn hai trăm ba mươi tư"),
    ("47,5", "bốn mươi bảy phẩy năm"),
    ("3,14", "ba phẩy một bốn"),
])
def test_the_separators_follow_vietnamese_convention(written, spoken):
    """"." groups thousands and "," marks the decimal — the opposite of English, and
    the reason 8.000 is eight thousand rather than eight-point-nought."""
    assert " ".join(vn_numbers.read_number(written)) == spoken


@pytest.mark.parametrize("written, count", [
    ("8000", 2), ("8.000", 2), ("123", 5), ("47,5", 5),
])
def test_the_counts_this_was_reported_with(written, count):
    """The examples that prompted the fix, asserted end to end through verify."""
    assert len(verify.syllables(written)) == count


def test_a_number_inside_a_sentence_is_counted_with_the_words():
    text = "có 8.000 người nghèo"
    assert verify.syllables(text) == ["có", "tám", "nghìn", "người", "nghèo"]


def test_a_sentence_ending_in_a_number_keeps_its_full_stop_out_of_it():
    """The thousands separator and the end of a sentence are the same character."""
    assert verify.syllables("còn 100.") == ["còn", "một", "trăm"]


def test_a_leading_zero_marks_a_code_not_a_quantity():
    """Phone numbers, dates and ids are written with one and read digit by digit, the
    zero spoken. "007" is "không không bảy", not "bảy"."""
    assert vn_numbers.read_integer("007") == ["không", "không", "bảy"]
    assert vn_numbers.read_integer("0987654321") == [
        "không", "chín", "tám", "bảy", "sáu", "năm", "bốn", "ba", "hai", "một"
    ]


def test_a_very_long_run_of_digits_is_read_one_at_a_time():
    """Past a thousand billion the naming stops being something anyone says out loud."""
    assert vn_numbers.read_integer("9" * 13) == ["chín"] * 13


def test_empty_groups_do_not_speak_their_scale():
    """8000 is "tám nghìn", never "tám nghìn không trăm"."""
    assert vn_numbers.read_integer("8000") == ["tám", "nghìn"]


def test_expand_leaves_text_without_numbers_alone():
    text = "không có con số nào ở đây"
    assert vn_numbers.expand(text) == text


def test_the_old_rule_under_counted_which_inflated_the_score():
    """Direction matters: fewer expected syllables makes heard/expected larger, so the
    bug made sentences with numbers look more complete than they were."""
    assert len(verify.syllables("123")) > len("123".split())


# --- where the model stops reading a number as a quantity ----------------------------
#
# Measured, not assumed: one sentence per digit length, three renders each,
# transcribed back. Four digits is the switch.


@pytest.mark.parametrize("written, spoken", [
    ("2026", "hai nghìn không trăm hai mươi sáu"),   # came back exactly this
    ("1900", "một nghìn chín trăm"),
    ("8000", "tám nghìn"),
])
def test_four_digits_or_fewer_are_named(written, spoken):
    assert " ".join(vn_numbers.read_number(written)) == spoken


@pytest.mark.parametrize("written", ["12345", "1000000", "1500000000", "1.000.000"])
def test_a_longer_number_counts_one_syllable_per_digit(written):
    """Not because that is what the model says — nothing reliably is. It has no scale
    words at all: 1000000 came back "một ngàn", 1500000000 came back "một một năm
    ngàn". Counting digits is the fallback for a reader who renders anyway."""
    digits = [c for c in written if c.isdigit()]
    assert len(vn_numbers.read_number(written)) == len(digits)


def test_a_leading_zero_still_wins_over_length():
    """A number written with one is a phone number, a date or a code at any length,
    and everyone reads those a digit at a time — this model included."""
    assert vn_numbers.read_number("0987") == ["không", "chín", "tám", "bảy"]
    assert len(vn_numbers.read_number("0987654321")) == 10


# --- which numbers the script has to fix before rendering ----------------------------


@pytest.mark.parametrize("token", ["7", "47", "2026", "8000", "1900", "47,5"])
def test_a_short_number_needs_no_fixing(token):
    assert not vn_numbers.is_hard_to_read(token)


@pytest.mark.parametrize("token", ["12345", "123456", "1000000", "1500000000",
                                   "1.000.000"])
def test_a_long_number_is_flagged_for_the_reader(token):
    """The model reads these wrong however they are counted, and the fix is a word
    instead of a figure — which only the person with the script can make."""
    assert vn_numbers.is_hard_to_read(token)


@pytest.mark.parametrize("token", ["0987654321", "0912345678", "0123"])
def test_a_phone_number_is_not_flagged(token):
    """It is long, and it is read correctly anyway: one digit at a time is the only
    way such a token is ever said, by anyone."""
    assert not vn_numbers.is_hard_to_read(token)


def test_the_flag_finds_every_offender_in_a_sentence():
    text = "Gọi 19001234 hoặc 0987654321, số dư 1500000000 đồng, năm 2026."
    assert vn_numbers.hard_to_read(text) == ["19001234", "1500000000"]


def test_an_ordinary_sentence_has_nothing_to_fix():
    """Almost every sentence, and the reason the checker needs no number allowance
    at all now."""
    assert vn_numbers.hard_to_read("Dạ, em chào anh Minh.") == []
    assert vn_numbers.hard_to_read("Anh chờ hai mươi hai giờ nhé.") == []
