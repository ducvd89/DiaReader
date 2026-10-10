"""One WAV per generated sentence, kept until the book is exported for real.

The export used to stream straight into a single WAV and transcode it, which made a
finished file the only artefact: a chapter with three bad sentences in it had to be
generated again from the top, and the good four hundred were thrown away with the bad
three. Sentences are written here instead, and the MP3 is assembled from them when the
user asks for it — so a sentence can be re-rendered on its own, listened to, and either
adopted or left alone.

What a cache entry is keyed on is the whole design. A take is valid for exactly one
(sentence text, voice, speed) triple: edit the text in the script editor and the old
audio no longer says it, switch the voice and it is the wrong person, change the
reading speed and it no longer matches its neighbours. All three go into the key, so a
stale take can never be picked up silently — the failure this app keeps meeting is the
silent one.

Position does not go into the key. Sentences shift when a chapter is edited, and a take
that is still correct for its text should survive that.
"""
from __future__ import annotations

import hashlib
import json
import logging
import shutil
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import soundfile as sf

from . import config

log = logging.getLogger(__name__)

CACHE_ROOT = config.APP_DIR / "sentence_cache"
_MANIFEST = "index.json"


def book_key(book_path: str | Path) -> str:
    """A folder name for one book, by resolved path — matching text_extract's cache."""
    return hashlib.sha1(str(Path(book_path).resolve()).encode("utf-8")).hexdigest()


def take_key(text: str, voice: str | None, speed: float) -> str:
    """Identity of one rendering. Anything that changes the audio belongs in here."""
    payload = f"{text}\x00{voice or ''}\x00{speed:.3f}"
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


@dataclass
class Take:
    """One generated rendering of one sentence."""

    key: str
    wav: str                     # filename inside the book's cache folder
    text: str
    voice: str | None
    speed: float
    seconds: float
    score: float = 1.0
    transcript: str = ""
    accepted: bool = True
    #: False for a take the user rendered but has not adopted. It stays on disk so it
    #: can be compared against the one in use, and is skipped when assembling.
    in_use: bool = True


@dataclass
class ChapterTakes:
    title: str = ""
    #: sentence index (as a string, because JSON keys are strings) -> the Take in use
    sentences: dict = field(default_factory=dict)
    #: sentence index -> other takes rendered for it but not adopted. Kept in a
    #: separate slot rather than replacing the one in use, which is what makes
    #: "listen before you keep it" mean anything.
    alternatives: dict = field(default_factory=dict)


class SentenceCache:
    """The takes for one book, and the file layout under them."""

    def __init__(self, book_path: str | Path, engine: str = "dia"):
        self.book_path = Path(book_path)
        # Keep the original Dia directory exactly as it was. Other engines get their
        # own takes, so switching engines never reuses another model's narration.
        suffix = "" if engine == "dia" else f"-{engine}"
        self.dir = CACHE_ROOT / f"{book_key(book_path)}{suffix}"
        self._chapters: dict[str, ChapterTakes] = {}
        self._load()

    # -- persistence -----------------------------------------------------------------

    @property
    def manifest_path(self) -> Path:
        return self.dir / _MANIFEST

    def _load(self) -> None:
        if not self.manifest_path.exists():
            return
        try:
            data = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            log.warning("Ignoring unreadable sentence cache for %s: %s", self.book_path, e)
            return
        for index, chapter in (data.get("chapters") or {}).items():
            takes = {
                position: Take(**take)
                for position, take in (chapter.get("sentences") or {}).items()
            }
            others = {
                position: [Take(**t) for t in takes_list]
                for position, takes_list in (chapter.get("alternatives") or {}).items()
            }
            self._chapters[index] = ChapterTakes(
                title=chapter.get("title", ""), sentences=takes, alternatives=others
            )

    def save(self) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "source_path": str(self.book_path.resolve()),
            "chapters": {
                index: {
                    "title": chapter.title,
                    "sentences": {p: asdict(t) for p, t in chapter.sentences.items()},
                    "alternatives": {
                        p: [asdict(t) for t in takes]
                        for p, takes in chapter.alternatives.items()
                    },
                }
                for index, chapter in self._chapters.items()
            },
        }
        self.manifest_path.write_text(
            json.dumps(payload, ensure_ascii=False), encoding="utf-8"
        )

    # -- reading ---------------------------------------------------------------------

    def chapter_indices(self) -> list[int]:
        return sorted(int(i) for i in self._chapters)

    def takes_for(self, chapter_index: int) -> dict[int, Take]:
        chapter = self._chapters.get(str(chapter_index))
        if chapter is None:
            return {}
        return {int(p): t for p, t in chapter.sentences.items()}

    def get(self, chapter_index: int, position: int) -> Take | None:
        return self.takes_for(chapter_index).get(position)

    def wav_path(self, take: Take) -> Path:
        return self.dir / take.wav

    def matches(self, take: Take | None, text: str, voice: str | None, speed: float) -> bool:
        """Whether a stored take still renders this exact sentence."""
        return take is not None and take.key == take_key(text, voice, speed)

    def audio(self, take: Take) -> np.ndarray:
        data, _ = sf.read(str(self.wav_path(take)), dtype="float32")
        return data.reshape(-1)

    # -- writing ---------------------------------------------------------------------

    def put(
        self,
        chapter_index: int,
        position: int,
        audio: np.ndarray,
        *,
        title: str,
        text: str,
        voice: str | None,
        speed: float,
        score: float = 1.0,
        transcript: str = "",
        accepted: bool = True,
        in_use: bool = True,
    ) -> Take:
        """Store one rendering. Replaces whatever was in use at that position."""
        self.dir.mkdir(parents=True, exist_ok=True)
        key = take_key(text, voice, speed)
        # The key identifies what was asked for; the filename has to identify this
        # particular draw. Re-rendering a sentence in the same voice is the commonest
        # thing anyone does here, and naming by key alone made that overwrite the take
        # it was meant to be compared against.
        name = f"{chapter_index:04d}-{position:05d}-{key}-{uuid.uuid4().hex[:8]}.wav"
        sf.write(str(self.dir / name), audio, config.SAMPLE_RATE, subtype="PCM_16")

        take = Take(
            key=key, wav=name, text=text, voice=voice, speed=speed,
            seconds=len(audio) / config.SAMPLE_RATE, score=score,
            transcript=transcript, accepted=accepted, in_use=in_use,
        )
        chapter = self._chapters.setdefault(str(chapter_index), ChapterTakes(title=title))
        if title:
            chapter.title = title
        slot = str(position)
        if in_use:
            # Only one take is in use per position. The one it displaces keeps its file
            # and moves aside, so it can still be compared against and adopted back.
            previous = chapter.sentences.get(slot)
            if previous is not None and previous.wav != name:
                previous.in_use = False
                chapter.alternatives.setdefault(slot, []).append(previous)
            chapter.sentences[slot] = take
        else:
            chapter.alternatives.setdefault(slot, []).append(take)
        return take

    def alternatives_for(self, chapter_index: int, position: int) -> list[Take]:
        """Takes rendered at this position but not in use, newest last."""
        chapter = self._chapters.get(str(chapter_index))
        if chapter is None:
            return []
        return list(chapter.alternatives.get(str(position), []))

    def adopt(self, chapter_index: int, position: int, take: Take) -> None:
        """Put a take into use at its position, moving the current one aside."""
        chapter = self._chapters.setdefault(str(chapter_index), ChapterTakes())
        slot = str(position)
        current = chapter.sentences.get(slot)
        if current is not None and current.wav != take.wav:
            current.in_use = False
            chapter.alternatives.setdefault(slot, []).append(current)
        chapter.alternatives[slot] = [
            t for t in chapter.alternatives.get(slot, []) if t.wav != take.wav
        ]
        take.in_use = True
        chapter.sentences[slot] = take

    def drop_alternatives(self, needed_bytes: int) -> int:
        """Delete unadopted takes, oldest first, until `needed_bytes` are freed.

        Only takes nobody chose. The one in use at each position is what the book is
        made of and is never a candidate here.
        """
        candidates = [
            (position, take, chapter)
            for chapter in self._chapters.values()
            for position, takes in chapter.alternatives.items()
            for take in takes
        ]

        def age(entry):
            try:
                return self.wav_path(entry[1]).stat().st_mtime
            except OSError:
                return 0.0

        freed = 0
        for position, take, chapter in sorted(candidates, key=age):
            if freed >= needed_bytes:
                break
            path = self.wav_path(take)
            try:
                size = path.stat().st_size
                path.unlink()
            except OSError:
                size = 0
            freed += size
            chapter.alternatives[position] = [
                t for t in chapter.alternatives.get(position, []) if t.wav != take.wav
            ]
        if freed:
            self.save()
            log.info("Cache over its limit: dropped unadopted takes worth %.0f MB.",
                     freed / 1024 / 1024)
        return freed

    def size_bytes(self) -> int:
        return dir_size(self.dir)

    def realign_to(self, plan, speed: float) -> int:
        """Move every take to the position its sentence now occupies.

        Takes are stored under a position but identified by their content, and an edit
        breaks the two apart: split one sentence in two and everything after it slides
        down a slot. The text moved and the audio did not, so the review table showed
        sentence N beside sentence N-1’s recording, the last sentence looked
        unrendered while its audio sat one slot up, and every sentence after the edit
        was regenerated at about forty seconds each.

        Matching is by key — (text, voice, speed) — which is what the module docstring
        has always claimed position is deliberately kept out of. This is where that
        claim is actually honoured.

        `plan` is (chapter index, position, text, voice) for the sentences the book now
        has. Chapters absent from it are left alone, so realigning one chapter never
        disturbs another. Returns how many takes moved.
        """
        by_chapter: dict[str, list[tuple[int, str, str | None]]] = {}
        for chapter_index, position, text, voice in plan:
            by_chapter.setdefault(str(chapter_index), []).append((position, text, voice))
        moved = 0
        for index, wanted in by_chapter.items():
            chapter = self._chapters.get(index)
            if chapter is not None:
                moved += self._realign_chapter(chapter, sorted(wanted), speed)
        if moved:
            log.info("Sentences shifted in the script; moved %d cached take(s) to follow.",
                     moved)
            self.save()
        return moved

    def _realign_chapter(self, chapter: ChapterTakes, wanted, speed: float) -> int:
        pool: dict[str, list[Take]] = {}
        for take in chapter.sentences.values():
            pool.setdefault(take.key, []).append(take)
        for takes in chapter.alternatives.values():
            for take in takes:
                pool.setdefault(take.key, []).append(take)
        # The take that was in use stays the one in use: an edit elsewhere in the
        # chapter must not undo a take the user listened to and chose.
        for takes in pool.values():
            takes.sort(key=lambda t: not t.in_use)

        sentences: dict = {}
        first_slot: dict[str, str] = {}
        moved = 0
        for position, text, voice in wanted:
            key = take_key(text, voice, speed)
            slot = str(position)
            first_slot.setdefault(key, slot)
            candidates = pool.get(key)
            if not candidates:
                continue
            take = candidates.pop(0)
            if chapter.sentences.get(slot) is not take:
                moved += 1
            take.in_use = True
            sentences[slot] = take

        # Whatever is left over is a take nothing in the book is currently asking for:
        # other renderings of a sentence that is still there, or the audio of a
        # sentence that was just edited. Both are parked as alternatives rather than
        # dropped, so undoing the edit finds them again instead of paying to remake
        # them — and so no WAV is left on disk that the manifest cannot reach.
        last_slot = str(wanted[-1][0]) if wanted else "0"
        alternatives: dict = {}
        for key, leftover in pool.items():
            slot = first_slot.get(key, last_slot)
            for take in leftover:
                take.in_use = False
                alternatives.setdefault(slot, []).append(take)

        chapter.sentences, chapter.alternatives = sentences, alternatives
        return moved

    def clear(self) -> None:
        """Forget everything for this book, files included."""
        self._chapters.clear()
        shutil.rmtree(self.dir, ignore_errors=True)

    # -- assembling ------------------------------------------------------------------

    def missing(self, plan: list[tuple[int, int, str, str | None]], speed: float) -> list[tuple[int, int]]:
        """(chapter, position) pairs in `plan` with no usable take.

        `plan` is what the book currently says: (chapter index, position, text, voice).
        A take whose key no longer matches counts as missing, which is what makes an
        edited sentence regenerate instead of shipping its old audio.
        """
        gaps = []
        for chapter_index, position, text, voice in plan:
            take = self.get(chapter_index, position)
            if not self.matches(take, text, voice, speed) or not self.wav_path(take).exists():
                gaps.append((chapter_index, position))
        return gaps

    def total_seconds(self) -> float:
        return sum(
            take.seconds
            for chapter in self._chapters.values()
            for take in chapter.sentences.values()
            if take.in_use
        )


# --- keeping the cache to a size -----------------------------------------------------
#
# (GB, label) in the order they are offered. 0 means no limit.
CACHE_LIMIT_CHOICES: list[tuple[float, str]] = [
    (1.0, "1 GB"), (2.0, "2 GB"), (5.0, "5 GB"), (0.0, "No limit"),
]
_GB = 1024 ** 3


# Bytes one second of cached audio costs: mono PCM_16 at the app's sample rate.
BYTES_PER_SECOND = config.SAMPLE_RATE * 2
# Speaking rate, measured off the fine-tune's own training clips — the same figure
# voice.py and tts_engine budget with. Checked against a real 40-sentence chapter
# here: 4343 characters produced 208.2s of speech, i.e. 20.9 char/s.
CHARS_PER_SECOND = 20.3
# What a chapter costs on disk over and above the audio that ends up in the book.
# A sentence the check rejects is kept — it is what the review table compares against
# — so the folder runs larger than the finished file. Measured on that same chapter:
# 23.8 MB on disk against 208.2s of in-use audio, a ratio of 1.36.
_REJECTED_TAKE_ALLOWANCE = 1.35


def estimated_bytes(texts, speed: float = 1.0) -> int:
    """Roughly what rendering these sentences will add to the cache.

    Deliberately rough, and only ever used to warn before starting: the real figure
    depends on how many takes each sentence needs, which is not knowable in advance.
    """
    characters = sum(len(t) for t in texts)
    seconds = characters / CHARS_PER_SECOND / max(speed, 0.1)
    return int(seconds * BYTES_PER_SECOND * _REJECTED_TAKE_ALLOWANCE)


def dir_size(path: Path) -> int:
    """Bytes used under `path`, ignoring anything that vanishes while counting."""
    total = 0
    if not path.exists():
        return 0
    for entry in path.rglob("*"):
        try:
            if entry.is_file():
                total += entry.stat().st_size
        except OSError:
            continue
    return total


def total_size() -> int:
    """Bytes used by every book's takes."""
    return dir_size(CACHE_ROOT)


def book_dirs() -> list[Path]:
    if not CACHE_ROOT.exists():
        return []
    return [p for p in CACHE_ROOT.iterdir() if p.is_dir()]


def _last_used(book_dir: Path) -> float:
    """When this book's cache was last written to. Newest file wins, because the
    manifest is rewritten on save but the WAVs are what the folder is full of."""
    times = []
    for entry in book_dir.rglob("*"):
        try:
            times.append(entry.stat().st_mtime)
        except OSError:
            continue
    return max(times, default=0.0)


def clear_all() -> None:
    """Forget every book's takes."""
    shutil.rmtree(CACHE_ROOT, ignore_errors=True)


def enforce_limit(limit_gb: float, keep: "SentenceCache | None" = None) -> int:
    """Bring the whole cache under `limit_gb`, returning the bytes freed.

    What goes first is the whole point. A take that is *in use* for the book being
    worked on is the one thing that must survive: deleting it does not lose data, but
    it does silently cost forty seconds of GPU time to make again, and the user would
    meet that as an export they thought was cached suddenly re-rendering. So the order
    is other books least-recently-used first, then this book's unadopted alternatives
    oldest first — the takes that were already rejected once.

    If that is not enough, the cache stays over the limit rather than eating the book
    in front of the user. A limit is a housekeeping preference, not a quota to enforce
    at the cost of the work.
    """
    if limit_gb <= 0:
        return 0
    limit = int(limit_gb * _GB)
    used = total_size()
    if used <= limit:
        return 0

    freed = 0
    protected = str(keep.dir.resolve()) if keep is not None else None
    others = [d for d in book_dirs()
              if protected is None or str(d.resolve()) != protected]
    for book_dir in sorted(others, key=_last_used):
        if used - freed <= limit:
            break
        size = dir_size(book_dir)
        shutil.rmtree(book_dir, ignore_errors=True)
        freed += size
        log.info("Cache over %.1f GB: dropped takes for %s (%.0f MB).",
                 limit_gb, book_dir.name, size / 1024 / 1024)

    if keep is not None and used - freed > limit:
        freed += keep.drop_alternatives((used - freed) - limit)
    return freed
