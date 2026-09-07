"""Per-sentence takes, and the rules that decide when one is still usable.

The export used to stream into a single WAV, which made a chapter all-or-nothing: three
bad sentences meant regenerating four hundred good ones. Takes are stored individually
now, so what matters most here is when a stored one stops being valid — a stale take
served silently is the failure mode this app keeps meeting.
"""
import numpy as np
import pytest

from diareader import config, sentence_cache


@pytest.fixture
def cache(tmp_path, monkeypatch):
    monkeypatch.setattr(sentence_cache, "CACHE_ROOT", tmp_path / "sentence_cache")
    book = tmp_path / "book.txt"
    book.write_text("nothing here", encoding="utf-8")
    return sentence_cache.SentenceCache(book)


def _audio(seconds=1.0):
    return np.ones(int(seconds * config.SAMPLE_RATE), dtype=np.float32)


def _put(cache, position=0, text="Xin chào", voice="Vietsu", speed=1.0, **kw):
    return cache.put(
        0, position, _audio(), title="Ch1", text=text, voice=voice, speed=speed, **kw
    )


def test_a_take_round_trips_through_disk(cache, tmp_path, monkeypatch):
    _put(cache)
    cache.save()

    monkeypatch.setattr(sentence_cache, "CACHE_ROOT", cache.dir.parent)
    reopened = sentence_cache.SentenceCache(cache.book_path)
    take = reopened.get(0, 0)
    assert take is not None
    assert take.text == "Xin chào"
    assert reopened.wav_path(take).is_file()
    assert len(reopened.audio(take)) == config.SAMPLE_RATE


def test_editing_the_sentence_invalidates_its_take(cache):
    """The old audio no longer says what the book says. Serving it would ship a
    sentence the reader never wrote."""
    _put(cache, text="Xin chào")
    assert cache.missing([(0, 0, "Xin chào", "Vietsu")], 1.0) == []
    assert cache.missing([(0, 0, "Xin chào các bạn", "Vietsu")], 1.0) == [(0, 0)]


def test_changing_the_voice_invalidates_its_take(cache):
    _put(cache, voice="Vietsu")
    assert cache.missing([(0, 0, "Xin chào", "Latradio")], 1.0) == [(0, 0)]


def test_changing_the_speed_invalidates_its_take(cache):
    """A take rendered at 1.0 does not match its neighbours at 0.9."""
    _put(cache, speed=1.0)
    assert cache.missing([(0, 0, "Xin chào", "Vietsu")], 0.9) == [(0, 0)]


def test_a_take_survives_the_sentence_moving(cache):
    """Position is deliberately not part of the key: editing a chapter shifts every
    sentence after the edit, and a take that still says the right words should not be
    thrown away for that."""
    first = _put(cache, position=3, text="Xin chào")
    second = _put(cache, position=7, text="Xin chào")
    assert first.key == second.key


def test_a_missing_file_counts_as_missing(cache):
    """The manifest and the folder can disagree — someone empties the cache directory
    without deleting index.json — and the export must regenerate rather than fail."""
    take = _put(cache)
    cache.wav_path(take).unlink()
    assert cache.missing([(0, 0, "Xin chào", "Vietsu")], 1.0) == [(0, 0)]


def test_a_new_take_displaces_the_one_in_use(cache):
    old = _put(cache, text="Xin chào", voice="Vietsu")
    new = _put(cache, text="Xin chào", voice="Latradio")
    assert cache.get(0, 0).voice == "Latradio"
    assert new.in_use
    # The displaced file stays on disk so it can still be compared and adopted back.
    assert cache.wav_path(old).is_file()


def test_a_take_can_be_rendered_without_being_adopted(cache):
    """"Listen before you keep it" — the alternative take must not change the book
    until somebody says so."""
    _put(cache, text="Xin chào", voice="Vietsu")
    _put(cache, text="Xin chào", voice="Latradio", in_use=False)
    assert cache.get(0, 0).voice == "Vietsu"


def test_adopting_puts_a_take_back_into_use(cache):
    _put(cache, text="Xin chào", voice="Vietsu")
    alternative = _put(cache, text="Xin chào", voice="Latradio", in_use=False)
    cache.adopt(0, 0, alternative)
    assert cache.get(0, 0).voice == "Latradio"
    assert cache.get(0, 0).in_use


def test_the_verdict_travels_with_the_take(cache):
    """A take accepted at 100% and one kept after ten failures are identical as
    arrays. The review table exists to tell them apart."""
    _put(cache, score=0.36, transcript="heard 'bước hai'", accepted=False)
    take = cache.get(0, 0)
    assert take.score == pytest.approx(0.36)
    assert not take.accepted
    assert "bước hai" in take.transcript


def test_an_unreadable_manifest_is_ignored_not_fatal(cache, monkeypatch):
    """A corrupt cache must cost a regeneration, not the ability to open the book."""
    cache.dir.mkdir(parents=True, exist_ok=True)
    cache.manifest_path.write_text("{ not json", encoding="utf-8")
    monkeypatch.setattr(sentence_cache, "CACHE_ROOT", cache.dir.parent)
    reopened = sentence_cache.SentenceCache(cache.book_path)
    assert reopened.chapter_indices() == []


def test_clearing_removes_the_files_too(cache):
    _put(cache)
    cache.save()
    cache.clear()
    assert not cache.dir.exists()
    assert cache.get(0, 0) is None


def test_re_rendering_the_same_sentence_keeps_both_takes(cache):
    """The commonest action in the review table: same sentence, same voice, another
    draw. Naming the file by the take key alone overwrote the very take the new one
    was meant to be compared against."""
    first = _put(cache, text="Xin chào", voice="Vietsu")
    second = _put(cache, text="Xin chào", voice="Vietsu")

    assert first.wav != second.wav
    assert cache.wav_path(first).is_file()
    assert cache.wav_path(second).is_file()
    assert cache.get(0, 0).wav == second.wav
    assert [t.wav for t in cache.alternatives_for(0, 0)] == [first.wav]


def test_an_alternative_can_be_adopted_back(cache):
    first = _put(cache, text="Xin chào", voice="Vietsu")
    _put(cache, text="Xin chào", voice="Vietsu")
    cache.adopt(0, 0, first)

    assert cache.get(0, 0).wav == first.wav
    assert first.wav not in [t.wav for t in cache.alternatives_for(0, 0)]
