"""The two halves of making an audiobook, and the seam the user sees.

Rendering and saving used to be one button with one file dialog in front of it. They
are separate jobs now — the script editor fills the cache, Save final file writes the
file — and what has to hold is that neither half quietly does the other's work, and
that a choice made in one half is still true in the other.
"""
import numpy as np
import pytest
import soundfile as sf

from diareader import chunking, config, pipeline, sentence_cache


@pytest.fixture
def settings(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "SETTINGS_PATH", tmp_path / "settings.json")
    monkeypatch.setattr(config, "APP_DIR", tmp_path)
    refs = tmp_path / "refs"
    refs.mkdir()
    monkeypatch.setattr(config, "VOICE_REF_DIR", refs)
    s = config.Settings()
    for name in ("Latradio", "NuMN1"):
        sf.write(str(refs / f"{name}.wav"), np.zeros(4410, dtype="float32"),
                 config.SAMPLE_RATE)
        s.voices[name] = {"wav": f"{name}.wav", "transcript": "xin chào"}
    s.default_voice = "Latradio"
    return s


# --- estimating what a chapter will cost the cache -----------------------------------


def test_the_estimate_is_in_the_right_order_of_magnitude(settings):
    """Calibrated against a real chapter: 4343 characters of Vietnamese produced
    208.2s of speech and 23.8 MB on disk, alternatives included."""
    texts = ["x" * 4343]
    estimate = sentence_cache.estimated_bytes(texts, speed=1.0)
    actual = 23.8 * 1024 ** 2
    assert 0.7 * actual < estimate < 1.4 * actual, f"{estimate / 1024 ** 2:.0f} MB"


def test_a_longer_chapter_costs_more(settings):
    short = sentence_cache.estimated_bytes(["x" * 1000])
    long = sentence_cache.estimated_bytes(["x" * 4000])
    assert long > short * 3


def test_reading_slower_costs_more(settings):
    """Speed is applied before the take is written, so a slower book is a bigger one."""
    assert (sentence_cache.estimated_bytes(["x" * 1000], speed=0.8)
            > sentence_cache.estimated_bytes(["x" * 1000], speed=1.0))


def test_the_estimate_allows_for_the_takes_that_get_rejected(settings):
    """A sentence the check rejects stays on disk — the review table compares against
    it — so the folder is always bigger than the finished file."""
    bare = 1000 / sentence_cache.CHARS_PER_SECOND * sentence_cache.BYTES_PER_SECOND
    assert sentence_cache.estimated_bytes(["x" * 1000]) > bare


# --- the editor's opening voice ends up in the script --------------------------------


def _with_marker(window_self, text, start_voice):
    from diareader.gui.main_window import MainWindow
    return MainWindow._with_opening_marker(window_self, text, start_voice)


class _FakeWindow:
    def __init__(self, settings):
        self.settings = settings


def test_a_non_default_opening_voice_is_written_into_the_script(settings):
    """Held only in the combo, it would be lost the moment the editor closed — and
    Save final file no longer asks for a voice, so nothing downstream could recover it."""
    text = _with_marker(_FakeWindow(settings), "Câu một dài vừa đủ.", "NuMN1")
    assert text.startswith("[NuMN1] ")


def test_the_default_opening_voice_leaves_the_text_alone(settings):
    """It is what an untouched script gets anyway; writing it in would be noise."""
    original = "Câu một dài vừa đủ."
    assert _with_marker(_FakeWindow(settings), original, "Latradio") == original


def test_a_script_that_already_opens_with_a_marker_is_not_touched(settings):
    original = "[NuMN1] Câu một dài vừa đủ."
    assert _with_marker(_FakeWindow(settings), original, "Latradio") == original


def test_the_written_marker_actually_casts_the_opening(settings):
    """The point of writing it: the plan must come out the same as if the combo had
    been carried along as `start_voice`."""
    from diareader import text_extract

    text = _with_marker(_FakeWindow(settings), "Câu một dài vừa đủ. Câu hai cũng vậy.",
                        "NuMN1")
    chapters = [text_extract.Chapter(title="Ch1", text=text)]
    assert [p.voice for p in pipeline.sentence_plan(chapters, settings)] \
        == ["NuMN1", "NuMN1"]


def test_the_written_marker_is_never_spoken(settings):
    """Markers are instructions. One reaching the model as text would be a voice name
    read out loud in the middle of the book."""
    text = _with_marker(_FakeWindow(settings), "Câu một dài vừa đủ.", "NuMN1")
    spoken = [chunk.text for chunk in chunking.chunk_text_with_voices(text)]
    assert spoken == ["Câu một dài vừa đủ."]


def test_writing_the_marker_does_not_add_a_sentence(settings):
    """The chapter list and the progress bar both count sentences; a marker that
    counted as one would make every total off by one."""
    original = "Câu một dài vừa đủ. Câu hai cũng vậy."
    text = _with_marker(_FakeWindow(settings), original, "NuMN1")
    assert len(chunking.chunk_text(text)) == len(chunking.chunk_text(original))
