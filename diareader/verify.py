"""What the text asks for, in syllables — the other half of the quality check.

Vietnamese is monosyllabic, so a syllable is a whitespace-separated token once numbers
are spelled out and non-verbal tags removed. asr_verify compares this count against
what a speech recogniser actually heard, and that comparison is the whole check.

This file used to also *measure* the audio, counting syllable nuclei as prominent peaks
in an energy envelope. That worked well enough on long clips and failed on short ones:
across fifteen generations of a five-syllable line judged correct by ear, it read 3, 4
or 5, and it scored a sentence that said "dạ em chào ạ" — a word dropped and another
added — a perfect 5 of 5, because counting peaks cannot see a substitution. Recognising
the speech reads all fifteen exactly right and catches that one. The peak counter is
gone rather than kept as an option; see asr_verify for what replaced it and what the
comparison measured.

Three corrections went into getting the *text* side right, and they still apply:
numbers are spelled out before counting (see vn_numbers, "123" is five syllables), a
transcript that over-describes its audio looks exactly like an instrument that
under-counts, and a poor recording of otherwise fine speech looks the same again. Two
reference clips were mistaken for both before their transcript and their audio were
corrected.
"""
from __future__ import annotations

import logging
import re

from . import chunking, vn_numbers

log = logging.getLogger(__name__)

_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)


def syllables(text: str) -> list[str]:
    """Vietnamese is monosyllabic: one whitespace-separated token is one syllable.

    Non-verbal tags are stripped first — they are performed, not spoken, so they must
    not be counted among the sounds the audio is expected to contain.

    Numbers are spelled out before that rule is applied, because it is not true of
    them: "123" is one token and five syllables. See vn_numbers for what this used to
    get wrong and which way the error ran.
    """
    spoken = vn_numbers.expand(chunking.spoken_text(text).lower())
    return _PUNCT.sub(" ", spoken).split()
