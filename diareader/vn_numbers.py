"""Reading digits aloud in Vietnamese, so they can be counted as the syllables they are.

verify.py judges a generated sentence by comparing the sounds it can hear against the
syllables the text calls for, and it counted a whitespace-separated token as one
syllable. That is true of Vietnamese words and false of every number: "123" is one
token and five syllables, "8000" is one token and two. Punctuation made it worse
rather than better — the old splitter turned "." into a space, so "8.000" came out as
two tokens and scored the right answer for entirely the wrong reason, while "8000"
scored one.

The direction of the error matters. Under-counting the syllables a sentence calls for
inflates its ratio, so a sentence full of numbers looked *better* than it was, and one
that really had dropped a clause could still pass.

Only the count is used. The words are produced anyway because a list of them can be
read in a test and in a log, which is worth more than a bare integer when the question
is "why did this sentence score 60%". Where Vietnamese offers a choice — tư or bốn,
nghìn or ngàn, linh or lẻ — one is picked and the other is equally many syllables, so
nothing downstream depends on which.
"""
from __future__ import annotations

import re

# Numbers as they appear in Vietnamese text: "." groups thousands, "," marks the
# decimal. The opposite of English convention, and the reason "8.000" is eight
# thousand here rather than eight-point-nought.
NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)*")

_DIGITS = ["không", "một", "hai", "ba", "bốn", "năm", "sáu", "bảy", "tám", "chín"]
# Scale words for each group of three, from the right. All one syllable.
_SCALES = ["", "nghìn", "triệu", "tỷ"]
# Where the model stops reading a number as a quantity, measured rather than assumed.
# One sentence per digit length, twice each, transcribed back:
#
#   4 digits  2026       "hai nghìn không trăm hai mươi sáu"    a quantity
#   5 digits  12345      "một hai ba bốn mơi lăm"               digit by digit
#   7 digits  1234567    "một hai ba bốn năm sáu bảy"           digit by digit
#   9 digits  123456789  "một hai ba bốn năm sáu bảy tám chín"  digit by digit
#
# It was twelve, on the reasoning that anything past a thousand billion is an id
# rather than a quantity. True of people, not of this model: it switches at five, and
# counting a seven-digit hotline as thirteen syllables when the model says eight is a
# five-syllable error in the direction the floor is now exact about.
#
# The 5-8 digit region also comes back as hybrids — "196085" was read "một ngàn chín
# trăm sáu không tám mơi lăm", a quantity that gives up halfway — which is what
# reading_slack keeps a margin for.
_MAX_GROUPED_DIGITS = 4


def _group(value: int, *, leading: bool) -> list[str]:
    """One group of three digits. `leading` suppresses the "không trăm" filler."""
    hundreds, tens, units = value // 100, (value // 10) % 10, value % 10
    words: list[str] = []

    if hundreds:
        words += [_DIGITS[hundreds], "trăm"]
    elif not leading:
        # "hai nghìn không trăm hai mươi lăm" — the empty hundreds is spoken when
        # something larger came before it, and silent when the number starts here.
        words += ["không", "trăm"]

    if tens == 0:
        if units:
            if words:
                words.append("linh")  # 105 -> một trăm linh năm
            words.append(_DIGITS[units])
    elif tens == 1:
        words.append("mười")
        if units == 5:
            words.append("lăm")  # 15 -> mười lăm
        elif units:
            words.append(_DIGITS[units])
    else:
        words += [_DIGITS[tens], "mươi"]
        if units == 1:
            words.append("mốt")  # 21 -> hai mươi mốt
        elif units == 4:
            words.append("tư")  # 24 -> hai mươi tư
        elif units == 5:
            words.append("lăm")  # 25 -> hai mươi lăm
        elif units:
            words.append(_DIGITS[units])

    return words


def read_integer(digits: str) -> list[str]:
    """Spoken words for a run of digits, e.g. "123" -> một trăm hai mươi ba."""
    # A leading zero means this is not a quantity: phone numbers, dates and codes are
    # written that way and read one digit at a time, the zero included — "0987..." is
    # "không chín tám bảy", never "chín trăm tám mươi bảy triệu". Stripping it would
    # both lose a syllable and read the rest as a number nobody said.
    if len(digits) > 1 and digits[0] == "0":
        return [_DIGITS[int(d)] for d in digits]

    digits = digits.lstrip("0") or "0"
    if len(digits) <= _MAX_GROUPED_DIGITS:
        return read_as_quantity(digits)

    # Past the switch, one syllable per digit — not because that is what the model
    # says, but because nothing reliably is. Measured, it does not have the scale
    # words at all: 1000000 came back "một ngàn", 1500000000 came back "một một năm
    # ngàn". No count can describe that, so these are flagged in the script instead
    # (see hard_to_read) and the reader is asked to write them out in words, which
    # the model reads perfectly well. Counting digits is what the check falls back on
    # when they choose to render anyway.
    return [_DIGITS[int(d)] for d in digits]


def read_as_quantity(digits: str) -> list[str]:
    """The quantity reading, even at lengths the model would not use it for.

    Split out so reading_slack can weigh the two readings against each other without
    read_integer's threshold deciding the answer first.
    """
    digits = digits.lstrip("0") or "0"
    if len(digits) > len(_SCALES) * 3:
        return [_DIGITS[int(d)] for d in digits]
    if digits == "0":
        return ["không"]

    # Split into groups of three from the right, keeping their scale.
    groups: list[tuple[int, int]] = []  # (value, scale index)
    remaining = digits
    scale = 0
    while remaining:
        groups.append((int(remaining[-3:]), scale))
        remaining = remaining[:-3]
        scale += 1
    groups.reverse()

    words: list[str] = []
    for value, scale_index in groups:
        if value == 0:
            continue  # its scale word is not spoken either: 8000 is "tám nghìn"
        words += _group(value, leading=not words)
        if _SCALES[scale_index]:
            words.append(_SCALES[scale_index])
    return words


def read_number(token: str) -> list[str]:
    """Spoken words for a written number, decimal separator and all.

    "8.000" -> tám nghìn. "47,5" -> bốn mươi bảy phẩy năm. Digits after the comma are
    read one at a time, which is what a reader does with "3,14" — ba phẩy một bốn.
    """
    whole, comma, fraction = token.partition(",")
    words = read_integer(whole.replace(".", ""))
    if comma:
        words.append("phẩy")
        words += [_DIGITS[int(d)] for d in fraction if d.isdigit()]
    return words


#: Numbers with at least this many digits are flagged in the script rather than
#: counted with any confidence. Four is where the model stops reading a number as a
#: quantity, measured one sentence per digit length: "2026" came back "hai nghìn
#: không trăm hai mươi sáu", "12345" came back "một hai trăm ba bốn mươi lăm".
HARD_TO_READ_DIGITS = 5


def read_as_digits(token: str) -> list[str]:
    """The same written number said one digit at a time."""
    return [_DIGITS[int(d)] for d in token if d.isdigit()]


def _only_digits(token: str) -> str:
    return "".join(c for c in token if c.isdigit())


def is_hard_to_read(token: str) -> bool:
    """Whether the model cannot be trusted with this number as written.

    A leading zero is not hard: it marks a phone number, a date or a code, and those
    are read one digit at a time by everyone including this model. Length is what
    makes a number hard — past four digits it stops naming and starts improvising,
    and what it improvises is wrong in a way no syllable count can describe.
    """
    digits = _only_digits(token)
    if len(digits) > 1 and digits[0] == "0":
        return False
    return len(digits) >= HARD_TO_READ_DIGITS


def hard_to_read(text: str) -> list[str]:
    """Numbers in `text` that should be written out in words before rendering."""
    return [m.group() for m in NUMBER_RE.finditer(text) if is_hard_to_read(m.group())]


def expand(text: str) -> str:
    """The same text with every written number replaced by how it is said."""
    return NUMBER_RE.sub(lambda m: " ".join(read_number(m.group())), text)
