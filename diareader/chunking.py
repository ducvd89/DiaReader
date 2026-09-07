"""Split chapter text into TTS-sized, sentence-aware chunks for narration.

Dia generates a bounded number of audio tokens per call (~86 tokens/sec of audio),
so very long inputs either get truncated or degrade in quality. We keep chunks
comfortably small (character budget, not exact token count) and always split on
sentence boundaries so a chunk never cuts a sentence in half.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

# Vietnamese text uses the same terminal punctuation as English (. ! ? …); this is a
# simple heuristic splitter, not a full sentence tokenizer.
_SENTENCE_END = re.compile(r"(?<=[.!?…])\s+")

# Measured coverage on this checkpoint: it emits EOS early on long input and silently
# drops the tail rather than truncating cleanly, so audio comes out too short for the
# text. Fraction of text actually spoken, by chunk length (natural rate is ~20 chars/s):
#   216 chars -> 20.4 chars/s (99% read)   325 chars -> 37.1 chars/s (55% read)
#   289 chars -> 28.1 chars/s (72% read)   387 chars -> 38.7 chars/s (52% read)
# Degradation is gradual, not a hard cutoff (it is not the 512 text_length limit — 380
# and 439 byte chunks fit under it and still lost content). 220 sits at the measured
# safe point. Raising this silently loses words, so re-measure before changing it.
DEFAULT_MAX_CHARS = 220

# Base Dia is a *dialogue* model driven by [S1]/[S2] tags, but this Vietnamese
# fine-tune is not: 0 of 100 sampled training rows contain any speaker tag. Feeding
# it "[S1] ..." is out-of-distribution input and drives the decoder into repeating
# fragments, so narration is generated as plain untagged text, exactly as trained.
NARRATOR_TAG = ""


def is_speakable(text: str) -> bool:
    """Whether a fragment has anything for the model to say.

    Ebooks leave stray punctuation stranded on its own line — a lone opening quote
    survives sentence splitting as a one-character "sentence". There is no audio for
    the model to produce, so it returns nothing, the coverage check calls that a
    failure, and the chunk burns the full retry budget before being skipped anyway.
    Measured on one 80-sentence chapter: two such fragments cost 6 of the run's 108
    generations, none of which could ever have succeeded.
    """
    return any(ch.isalnum() for ch in text)


def split_sentences(text: str) -> list[str]:
    sentences: list[str] = []
    for paragraph in text.split("\n"):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        for piece in _SENTENCE_END.split(paragraph):
            piece = piece.strip()
            if piece and is_speakable(piece):
                sentences.append(piece)
    return sentences


def split_long_sentence(sentence: str, max_chars: int = DEFAULT_MAX_CHARS) -> list[str]:
    """Break a single over-long sentence at commas, never mid-word."""
    if len(sentence) <= max_chars:
        return [sentence]

    parts, current = [], ""
    for piece in re.split(r"(?<=,)\s+", sentence):
        if not current:
            current = piece
        elif len(current) + 1 + len(piece) <= max_chars:
            current = f"{current} {piece}"
        else:
            parts.append(current)
            current = piece
    if current:
        parts.append(current)
    return parts


# An inline marker like [GiongNu2] switches the narrator for everything that follows,
# until the next marker. The marker itself is never spoken.
VOICE_MARKER = re.compile(r"\[([A-Za-z0-9_\-]{1,40})\]")


@dataclass
class VoicedChunk:
    text: str
    voice: str | None  # None -> the run's starting voice


def split_by_voice(text: str, start_voice: str | None = None) -> list[tuple[str, str | None]]:
    """Split text into (segment, voice) pairs at voice markers.

    Returns segments in order; the first carries `start_voice` unless the text opens
    with a marker. Markers are stripped from the returned text.
    """
    segments: list[tuple[str, str | None]] = []
    current_voice = start_voice
    position = 0

    for match in VOICE_MARKER.finditer(text):
        before = text[position:match.start()]
        if before.strip():
            segments.append((before, current_voice))
        current_voice = match.group(1)
        position = match.end()

    remainder = text[position:]
    if remainder.strip():
        segments.append((remainder, current_voice))
    return segments


def chunk_text_with_voices(
    text: str, start_voice: str | None = None, max_chars: int = DEFAULT_MAX_CHARS
) -> list[VoicedChunk]:
    """Chapter text -> one VoicedChunk per sentence, carrying its active voice."""
    chunks: list[VoicedChunk] = []
    for segment, voice in split_by_voice(text, start_voice):
        for sentence in chunk_text(segment, max_chars=max_chars):
            chunks.append(VoicedChunk(text=sentence, voice=voice))
    return chunks


def chunk_text(text: str, max_chars: int = DEFAULT_MAX_CHARS) -> list[str]:
    """Chapter text -> one chunk per sentence.

    Sentences are deliberately NOT grouped. This checkpoint treats the first
    terminal period as end-of-utterance and stops there, silently dropping
    everything after it — its training transcripts carry no sentence-internal
    punctuation, just one terminal mark at the end. Measured: two-sentence chunks
    came out at 55-57% coverage, cutting off exactly at the first period, and
    repeated regeneration reproduced the same cut every time.
    """
    chunks: list[str] = []
    for sentence in split_sentences(text):
        chunks.extend(split_long_sentence(sentence, max_chars=max_chars))
    return chunks


# Dia's documented non-verbal directions. They are fed to the model as text but are
# *performed*, not read out, so they occupy characters without occupying time.
NONVERBAL_TAGS = (
    "(laughs)", "(chuckle)", "(sighs)", "(gasps)", "(coughs)",
    "(clears throat)", "(groans)", "(screams)", "(mumbles)", "(whistles)",
    "(sniffs)", "(sneezes)", "(burps)", "(inhales)", "(exhales)",
    "(humming)", "(singing)", "(sings)", "(claps)", "(applause)", "(beep)",
)
NONVERBAL_RE = re.compile("|".join(re.escape(t) for t in NONVERBAL_TAGS))


def spoken_text(text: str) -> str:
    """The part of a chunk that actually becomes speech.

    Anything estimating *duration* from text has to use this rather than the raw
    string. Measured against Whisper on a real chapter: "(clears throat) Chúng ta
    cũng nên xuất phát…" was judged 55.6% covered and regenerated three times, when
    the audio in fact carried 88.2% of its syllables — the missing 0.74s the check
    was hunting for was the tag itself, which the model correctly never spoke. Two
    of the four sentences flagged in that run failed this way.
    """
    return NONVERBAL_RE.sub("", text).strip()


_TERMINAL_PUNCTUATION = ".!?…"


def wrap_for_narration(chunk: str, tag: str = NARRATOR_TAG) -> str:
    """Prepare a chunk for the model. Every call into an engine goes through here.

    Including one-off experiments, which is not a style note. Hand-assembling
    ``f"{NARRATOR_TAG} {text}"`` skips the terminal mark below, and a benchmark run
    that way reported voice cloning as broken and generation as wildly unstable —
    both artefacts of feeding the model exactly the malformed input this function
    exists to prevent.

    A terminal mark is appended when missing: 99 of 100 sampled training transcripts
    end in one, and bare fragments (a stray heading line like "Quyển 2") are exactly
    the inputs that made the model emit zero audio tokens.
    """
    chunk = chunk.strip()
    if chunk and chunk[-1] not in _TERMINAL_PUNCTUATION:
        chunk = f"{chunk}."
    return f"{tag} {chunk}" if tag else chunk


def chunk_for_narration(text: str, max_chars: int = DEFAULT_MAX_CHARS, tag: str = NARRATOR_TAG) -> list[str]:
    """Chapter text -> list of strings ready to feed to the TTS engine."""
    return [wrap_for_narration(c, tag) for c in chunk_text(text, max_chars=max_chars)]
