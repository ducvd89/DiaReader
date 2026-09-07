"""Spotting words a Vietnamese syllable count does not describe.

verify.syllables counts whitespace-separated tokens, which is exact for Vietnamese and
wrong for everything else: "SmartBanking" is one token and reads as four syllables —
the recogniser heard "sờ mát ben kin". In a sixteen-syllable sentence that is +19% on
its own, over the acceptance ceiling, so a perfectly good sentence was regenerated ten
times and then logged as "this one really is missing text". It appears eleven times in
one of the books being read.

Rewriting it phonetically in the script would fix the count and ruin the audio — Dia
pronounces English correctly, and "Sờ-mát Ben-king" is what it would then say. So the
count stays wrong and the *ceiling* moves instead, for the sentences that contain such
a word and only upward: the error only ever runs one way, because the text under-counts
what gets spoken.

Detection is phonotactic rather than a word list. A Vietnamese syllable is
(onset)(nucleus)(coda) drawn from small closed sets, so "is this pronounceable as one
Vietnamese syllable" is decidable — which matters because plain ASCII is no signal at
all here: anh, em, thanh, chi, cho, hai, sinh and vui are all Vietnamese and all
undiacriticked.
"""
from __future__ import annotations

import re
import unicodedata

# Longest first: "ngh" must be tried before "ng" before "n".
_ONSETS = (
    "ngh", "ng", "nh", "ch", "gh", "gi", "kh", "ph", "qu", "th", "tr",
    "b", "c", "d", "đ", "g", "h", "k", "l", "m", "n", "p", "r", "s", "t", "v", "x",
)
# Vietnamese closes a syllable with one of these or with nothing at all.
_CODAS = ("ng", "nh", "ch", "c", "m", "n", "p", "t", "u", "o", "i", "y")
# Every vowel letter, accents stripped — tone and vowel diacritics are removed before
# matching so that "tiếng" and "tieng" travel the same path.
_VOWELS = set("aeiouy")
# Not in the Vietnamese alphabet at all. A token containing one cannot be Vietnamese.
_ABSENT_LETTERS = set("fjwz")

_WORD = re.compile(r"[^\W\d_]+", re.UNICODE)

# The five tone marks and the three vowel modifiers, as combining characters.
_TONE_MARKS = frozenset("\u0300\u0301\u0303\u0309\u0323")   # huyền sắc ngã hỏi nặng
_VOWEL_MODIFIERS = frozenset("\u0306\u0302\u031b")            # ă / âêô / ơư
_VIETNAMESE_MARKS = _TONE_MARKS | _VOWEL_MODIFIERS


def _strip_diacritics(word: str) -> str:
    """"tiếng" -> "tieng", keeping đ as its own letter."""
    word = word.replace("đ", "d").replace("Đ", "D")
    return "".join(
        c for c in unicodedata.normalize("NFD", word)
        if not unicodedata.combining(c)
    )


def has_vietnamese_diacritics(word: str) -> bool:
    """Whether `word` wears marks that only Vietnamese puts on those letters.

    "gì" and "GI" are different words and only the first is Vietnamese: the tone is
    part of the spelling, not decoration on it. Nothing written in the Latin alphabet
    outside Vietnamese combines these marks with vowels, so a token wearing one is
    Vietnamese and needs no further argument.

    Checked before the phonotactic analysis because the analysis structurally cannot
    see it. "gì" strips to "gi", the "gi" onset swallows both letters, and the
    syllable is left with no nucleus — so an ordinary Vietnamese word was being handed
    a syllable of ceiling headroom, seven times in one of the books being read.

    The marks have to sit on vowels. Vietnamese never marks a consonant — đ is a
    letter in its own right, not a d with something added — so requiring a vowel base
    is what keeps ñ and the like out.
    """
    if "đ" in word.lower():
        return True
    decomposed = unicodedata.normalize("NFD", word.lower())
    found = False
    for index, char in enumerate(decomposed):
        if not unicodedata.combining(char):
            continue
        if char not in _VIETNAMESE_MARKS:
            return False
        base = next(
            (c for c in reversed(decomposed[:index]) if not unicodedata.combining(c)),
            "",
        )
        if base not in _VOWELS:
            return False
        found = True
    return found


def is_vietnamese_syllable(word: str) -> bool:
    """Could this token be said as one Vietnamese syllable?

    Deliberately generous: it answers "does this fit the shape", not "is this a word".
    A false yes costs nothing — the token then counts as one syllable, which is what
    it would have done anyway — while a false no hands a sentence needless headroom.
    """
    stripped = _strip_diacritics(word.lower())
    if not stripped or _ABSENT_LETTERS & set(stripped):
        return False
    # Ahead of the shape analysis, and never the other way round: a tone mark settles
    # the question on its own, and the analysis gets some tone-bearing spellings wrong.
    if has_vietnamese_diacritics(word):
        return True

    word = stripped
    for onset in _ONSETS:
        if word.startswith(onset):
            rest = word[len(onset):]
            break
    else:
        rest = word

    if not rest or rest[0] not in _VOWELS:
        return False

    # Strip the longest matching coda, then everything left must be vowels.
    for coda in _CODAS:
        if len(rest) > len(coda) and rest.endswith(coda):
            rest = rest[: -len(coda)]
            break

    return bool(rest) and all(c in _VOWELS for c in rest)


def _spoken_syllable_estimate(word: str) -> int:
    """How many Vietnamese syllables a foreign word might take, at the high end.

    Not a prediction — an upper bound, because that is all the ceiling needs. An
    all-caps run is spelled out a letter at a time (OTP is "ô tê pê"), and anything
    else is counted by vowel groups and then doubled, since Vietnamese breaks the
    consonant clusters English allows: "smart" is one English syllable and two spoken
    ones, "banking" is two and two, so the pair measured four against the three vowel
    groups.
    """
    if word.isupper() and len(word) > 1:
        return len(word)

    # SmartBanking -> Smart, Banking. Written as one word, said as two.
    parts = re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?![a-z])", word) or [word]
    groups = sum(max(1, len(re.findall(r"[aeiouy]+", part.lower()))) for part in parts)
    return groups * 2


def foreign_words(text: str) -> list[str]:
    """Tokens in `text` that a Vietnamese syllable count does not describe."""
    return [w for w in _WORD.findall(text) if not is_vietnamese_syllable(w)]


def extra_syllables_allowed(text: str) -> int:
    """How many syllables beyond the token count this text might legitimately take.

    Zero for ordinary Vietnamese, which is almost every sentence, so almost every
    sentence keeps the ceiling exactly where the measurements put it.
    """
    return sum(max(0, _spoken_syllable_estimate(w) - 1) for w in foreign_words(text))
