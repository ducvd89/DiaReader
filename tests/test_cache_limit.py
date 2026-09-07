"""Keeping the take cache to a size, and what it is allowed to throw away.

Deleting a take costs no data — it can always be generated again — but it costs about
forty seconds of GPU time each, and the user meets that as an export they believed was
cached re-rendering from the top. So the order things go in is the whole subject here,
and the one rule that matters is that the book in front of the user survives.
"""
import os
import time

import numpy as np
import pytest

from diareader import config, sentence_cache


@pytest.fixture
def root(tmp_path, monkeypatch):
    monkeypatch.setattr(sentence_cache, "CACHE_ROOT", tmp_path / "cache")
    return tmp_path


def _cache(root, name="book.txt"):
    book = root / name
    book.write_text(name, encoding="utf-8")
    return sentence_cache.SentenceCache(book)


def _put(cache, position=0, seconds=1.0, **kw):
    audio = np.ones(int(seconds * config.SAMPLE_RATE), dtype=np.float32)
    return cache.put(0, position, audio, title="Ch1", text=f"Câu {position}",
                     voice="Vietsu", speed=1.0, **kw)


def _gb(num_bytes: int) -> float:
    return num_bytes / sentence_cache._GB


def test_the_size_it_reports_is_the_size_on_disk(root):
    cache = _cache(root)
    _put(cache)
    cache.save()
    on_disk = sum(p.stat().st_size for p in cache.dir.rglob("*") if p.is_file())
    assert cache.size_bytes() == on_disk
    assert sentence_cache.total_size() == on_disk


def test_no_limit_deletes_nothing(root):
    cache = _cache(root)
    _put(cache)
    cache.save()
    before = sentence_cache.total_size()
    assert sentence_cache.enforce_limit(0.0, keep=cache) == 0
    assert sentence_cache.total_size() == before


def test_under_the_limit_deletes_nothing(root):
    cache = _cache(root)
    _put(cache)
    cache.save()
    assert sentence_cache.enforce_limit(1.0, keep=cache) == 0
    assert cache.get(0, 0) is not None


def test_other_books_go_before_this_one(root):
    """The book being worked on is the one the user would notice losing."""
    old = _cache(root, "old.txt")
    _put(old)
    old.save()
    os.utime(old.dir, (time.time() - 9999, time.time() - 9999))
    for entry in old.dir.rglob("*"):
        os.utime(entry, (time.time() - 9999, time.time() - 9999))

    current = _cache(root, "current.txt")
    _put(current)
    current.save()

    freed = sentence_cache.enforce_limit(_gb(current.size_bytes() + 1000), keep=current)

    assert freed > 0
    assert not old.dir.exists()
    assert current.get(0, 0) is not None
    assert current.wav_path(current.get(0, 0)).is_file()


def test_the_least_recently_used_book_goes_first(root):
    stale = _cache(root, "stale.txt")
    _put(stale)
    stale.save()
    fresh = _cache(root, "fresh.txt")
    _put(fresh)
    fresh.save()
    for entry in list(stale.dir.rglob("*")) + [stale.dir]:
        os.utime(entry, (time.time() - 9999, time.time() - 9999))

    keep = _cache(root, "current.txt")
    _put(keep)
    keep.save()
    # Room for the current book and one of the other two.
    sentence_cache.enforce_limit(_gb(keep.size_bytes() * 2 + 1000), keep=keep)

    assert not stale.dir.exists()
    assert fresh.dir.exists()


def test_unadopted_takes_go_before_the_ones_in_use(root):
    """A rejected take is the cheapest thing in the cache to lose: nobody chose it,
    and the sentence still has audio without it."""
    cache = _cache(root)
    in_use = _put(cache, position=0)
    rejected = _put(cache, position=0, in_use=False)
    cache.save()

    sentence_cache.enforce_limit(_gb(cache.wav_path(in_use).stat().st_size + 5000),
                                 keep=cache)

    assert cache.wav_path(in_use).is_file()
    assert not cache.wav_path(rejected).exists()
    assert cache.alternatives_for(0, 0) == []


def test_a_take_in_use_is_never_deleted_to_make_room(root):
    """Rather than eat the book to satisfy a housekeeping preference, the cache is
    allowed to stay over its limit."""
    cache = _cache(root)
    for position in range(3):
        _put(cache, position=position)
    cache.save()

    sentence_cache.enforce_limit(_gb(1000), keep=cache)

    assert [cache.get(0, p) is not None for p in range(3)] == [True] * 3
    assert sentence_cache.total_size() > 1000


def test_the_oldest_unadopted_take_goes_first(root):
    cache = _cache(root)
    _put(cache, position=0)
    older = _put(cache, position=0, in_use=False)
    newer = _put(cache, position=0, in_use=False)
    cache.save()
    os.utime(cache.wav_path(older), (time.time() - 9999, time.time() - 9999))

    cache.drop_alternatives(cache.wav_path(older).stat().st_size)

    assert not cache.wav_path(older).exists()
    assert cache.wav_path(newer).is_file()


def test_clearing_everything_leaves_no_book_behind(root):
    first = _cache(root, "one.txt")
    _put(first)
    first.save()
    second = _cache(root, "two.txt")
    _put(second)
    second.save()

    sentence_cache.clear_all()
    assert sentence_cache.total_size() == 0
    assert sentence_cache.book_dirs() == []


def test_the_limit_choices_include_an_unlimited_one(root):
    """"No limit" has to be reachable: the cache is what makes one bad sentence
    fixable, and someone rendering a whole book may well want all of it kept."""
    assert 0.0 in [gigabytes for gigabytes, _ in sentence_cache.CACHE_LIMIT_CHOICES]
    assert config.Settings().cache_limit_gb in [
        gigabytes for gigabytes, _ in sentence_cache.CACHE_LIMIT_CHOICES
    ]
