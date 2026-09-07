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


# --- takes follow their sentence when an edit shifts it -------------------------------
#
# Position is deliberately not part of a take's key, which the module docstring has
# always said. Storage is by position all the same, so an edit pulls the two apart:
# split one sentence in two and everything after it slides down a slot. realign_to is
# where that claim is actually honoured.


def _chapter(cache, texts):
    for position, text in enumerate(texts):
        _put(cache, position=position, text=text)


def _keys(texts, voice="Vietsu"):
    return [(0, position, text, voice) for position, text in enumerate(texts)]


ORIGINAL = ["Câu một.", "Câu hai.", "Câu ba."]
SPLIT = ["Câu một.", "Câu hai a.", "Câu hai b.", "Câu ba."]


def test_a_take_follows_its_sentence_down_a_slot(cache):
    """The bug as reported: split a sentence and the text below moves while the audio
    stays put, so the table shows sentence N beside sentence N-1's recording."""
    _chapter(cache, ORIGINAL)
    cache.realign_to(_keys(SPLIT), 1.0)

    assert cache.get(0, 3).text == "Câu ba."
    assert cache.get(0, 0).text == "Câu một."


def test_only_the_genuinely_new_sentences_are_missing(cache):
    """Three of four were being regenerated when two were actually new, at about
    forty seconds each."""
    _chapter(cache, ORIGINAL)
    cache.realign_to(_keys(SPLIT), 1.0)
    assert cache.missing(_keys(SPLIT), 1.0) == [(0, 1), (0, 2)]


def test_a_slot_whose_sentence_changed_is_left_empty_not_wrong(cache):
    """Empty is recoverable. Holding the previous sentence's audio is what put the
    wrong words into the finished book."""
    _chapter(cache, ORIGINAL)
    cache.realign_to(_keys(SPLIT), 1.0)
    assert cache.get(0, 1) is None
    assert cache.get(0, 2) is None


def test_deleting_a_sentence_pulls_the_rest_back_up(cache):
    _chapter(cache, ORIGINAL)
    shorter = ["Câu một.", "Câu ba."]
    cache.realign_to(_keys(shorter), 1.0)

    assert [cache.get(0, p).text for p in range(2)] == shorter
    assert cache.missing(_keys(shorter), 1.0) == []


def test_an_untouched_chapter_does_not_move(cache):
    _chapter(cache, ORIGINAL)
    assert cache.realign_to(_keys(ORIGINAL), 1.0) == 0
    assert [cache.get(0, p).text for p in range(3)] == ORIGINAL


def test_the_adopted_take_stays_the_adopted_one(cache):
    """An edit elsewhere in the chapter must not undo a take the user listened to and
    chose."""
    _chapter(cache, ORIGINAL)
    chosen = _put(cache, position=2, text="Câu ba.", voice="Latradio")
    cache.adopt(0, 2, chosen)

    cache.realign_to(_keys(SPLIT, voice="Vietsu")
                     [:3] + [(0, 3, "Câu ba.", "Latradio")], 1.0)
    assert cache.get(0, 3).wav == chosen.wav
    assert cache.get(0, 3).in_use


def test_the_audio_of_an_edited_sentence_is_kept_not_dropped(cache):
    """Undoing the edit has to find it again rather than pay to make it a second time
    — and a WAV the manifest cannot reach would never be cleaned up either."""
    _chapter(cache, ORIGINAL)
    orphaned = cache.get(0, 1)
    cache.realign_to(_keys(SPLIT), 1.0)

    assert cache.wav_path(orphaned).is_file()
    parked = [t.wav for takes in cache._chapters["0"].alternatives.values() for t in takes]
    assert orphaned.wav in parked

    # And undoing the edit puts it back to work.
    cache.realign_to(_keys(ORIGINAL), 1.0)
    assert cache.get(0, 1).wav == orphaned.wav
    assert cache.missing(_keys(ORIGINAL), 1.0) == []


def test_realigning_one_chapter_leaves_the_others_alone(cache):
    _chapter(cache, ORIGINAL)
    cache.put(1, 0, _audio(), title="Ch2", text="Chương sau.", voice="Vietsu", speed=1.0)

    cache.realign_to(_keys(SPLIT), 1.0)
    assert cache.get(1, 0).text == "Chương sau."


def test_two_identical_sentences_each_keep_a_take(cache):
    """Same text twice in a chapter means two takes sharing one key; matching them
    off one at a time is what keeps both slots filled."""
    twice = ["Vâng ạ.", "Câu giữa.", "Vâng ạ."]
    _chapter(cache, twice)
    moved_to = ["Mở đầu.", "Vâng ạ.", "Câu giữa.", "Vâng ạ."]
    cache.realign_to(_keys(moved_to), 1.0)

    assert [cache.get(0, p) is not None for p in range(4)] == [False, True, True, True]
    assert cache.missing(_keys(moved_to), 1.0) == [(0, 0)]
