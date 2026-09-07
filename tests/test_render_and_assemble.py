"""The two halves of an export: generate sentences into the cache, then build a file.

Splitting them is what makes a single bad sentence fixable. Everything worth asserting
here is about the seam — that assembling reads only the cache, that a stopped render
keeps what it already made, and that the gaps and pacing survive the split.
"""
import numpy as np
import pytest
import soundfile as sf

from diareader import config, pipeline, sentence_cache, text_extract


@pytest.fixture
def cache(tmp_path, monkeypatch):
    monkeypatch.setattr(sentence_cache, "CACHE_ROOT", tmp_path / "cache")
    book = tmp_path / "book.txt"
    book.write_text("x", encoding="utf-8")
    return sentence_cache.SentenceCache(book)


@pytest.fixture
def settings(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "SETTINGS_PATH", tmp_path / "settings.json")
    monkeypatch.setattr(config, "APP_DIR", tmp_path)
    s = config.Settings()
    s.verify_mode = "off"
    s.normalize_output = False
    s.speed = 1.0
    return s


CHAPTERS = [
    text_extract.Chapter(title="Ch1", text="Câu một dài vừa đủ. Câu hai cũng vậy."),
    text_extract.Chapter(title="Ch2", text="Câu ba ở chương sau."),
]


class FakeEngine:
    """One second of tone per sentence, and a record of what it was asked for."""

    def __init__(self):
        self.asked = []

    def generate(self, tagged_text, voice_ref=None, **overrides):
        self.asked.append(tagged_text)
        return np.ones(config.SAMPLE_RATE, dtype=np.float32) * 0.5


def test_the_plan_lists_every_sentence_in_order(settings):
    plan = pipeline.sentence_plan(CHAPTERS, settings)
    assert [p.chapter_index for p in plan] == [0, 0, 1]
    assert [p.position for p in plan] == [0, 1, 0]
    assert plan[0].chapter_title == "Ch1"


def test_the_plan_honours_the_chapter_selection(settings):
    plan = pipeline.sentence_plan(CHAPTERS, settings, chapter_indices=[1])
    assert [p.chapter_index for p in plan] == [1]


def test_rendering_fills_the_cache(cache, settings):
    plan = pipeline.sentence_plan(CHAPTERS, settings)
    written = pipeline.render_sentences(FakeEngine(), settings, cache, plan, speed=1.0)

    assert written == 3
    assert cache.missing([(p.chapter_index, p.position, p.text, p.voice) for p in plan],
                         1.0) == []


def test_assembling_generates_nothing(cache, settings, tmp_path):
    """The whole point of the split: the file is built from what is on disk, so it can
    be rebuilt after changing one sentence without touching the model."""
    plan = pipeline.sentence_plan(CHAPTERS, settings)
    pipeline.render_sentences(FakeEngine(), settings, cache, plan, speed=1.0)

    out = tmp_path / "book.wav"
    result = pipeline.assemble(cache, plan, out, speed=1.0)

    audio, rate = sf.read(str(result), dtype="float32")
    assert rate == config.SAMPLE_RATE
    # Three seconds of speech, one inter-sentence gap and one inter-chapter gap.
    expected = 3 + pipeline.INTER_CHUNK_SILENCE_S + pipeline.INTER_CHAPTER_SILENCE_S
    assert abs(len(audio) / rate - expected) < 0.05


def test_a_second_export_regenerates_nothing(cache, settings):
    plan = pipeline.sentence_plan(CHAPTERS, settings)
    pipeline.render_sentences(FakeEngine(), settings, cache, plan, speed=1.0)

    gaps = cache.missing(
        [(p.chapter_index, p.position, p.text, p.voice) for p in plan], 1.0
    )
    assert gaps == []


def test_editing_one_sentence_regenerates_only_that_one(cache, settings):
    plan = pipeline.sentence_plan(CHAPTERS, settings)
    pipeline.render_sentences(FakeEngine(), settings, cache, plan, speed=1.0)

    edited = [text_extract.Chapter(title="Ch1", text="Câu một đã sửa. Câu hai cũng vậy."),
              CHAPTERS[1]]
    new_plan = pipeline.sentence_plan(edited, settings)
    gaps = cache.missing(
        [(p.chapter_index, p.position, p.text, p.voice) for p in new_plan], 1.0
    )
    assert gaps == [(0, 0)]


def test_a_stopped_render_keeps_what_it_made(cache, settings):
    """An export interrupted at sentence 900 of a thousand must not lose the 900 —
    that is the reason the cache exists at all."""
    plan = pipeline.sentence_plan(CHAPTERS, settings)
    calls = {"n": 0}

    def keep_going():
        calls["n"] += 1
        return calls["n"] <= 2       # stop before the third sentence

    with pytest.raises(pipeline.ExportCancelled):
        pipeline.render_sentences(
            FakeEngine(), settings, cache, plan, speed=1.0,
            should_continue=keep_going,
        )

    assert cache.get(0, 0) is not None
    assert cache.get(0, 1) is not None
    assert cache.get(1, 0) is None


def test_a_missing_take_becomes_a_gap_not_a_crash(cache, settings, tmp_path):
    """One sentence out of four thousand that never came out right must not make the
    book unexportable."""
    plan = pipeline.sentence_plan(CHAPTERS, settings)
    pipeline.render_sentences(FakeEngine(), settings, cache, plan[:2], speed=1.0)

    result = pipeline.assemble(cache, plan, tmp_path / "book.wav", speed=1.0)
    audio, rate = sf.read(str(result), dtype="float32")
    assert len(audio) > 0


def test_an_unadopted_take_stays_out_of_the_file(cache, settings, tmp_path):
    plan = pipeline.sentence_plan(CHAPTERS, settings)[:1]
    pipeline.render_sentences(FakeEngine(), settings, cache, plan, speed=1.0)
    in_use = cache.get(0, 0)

    class LouderEngine(FakeEngine):
        def generate(self, tagged_text, voice_ref=None, **overrides):
            return np.ones(config.SAMPLE_RATE * 2, dtype=np.float32) * 0.5

    pipeline.render_sentences(
        LouderEngine(), settings, cache, plan, speed=1.0, in_use=False
    )
    assert cache.get(0, 0).wav == in_use.wav

    result = pipeline.assemble(cache, plan, tmp_path / "book.wav", speed=1.0)
    audio, rate = sf.read(str(result), dtype="float32")
    assert abs(len(audio) / rate - 1.0) < 0.05, "the two-second alternative got in"


# --- the plan's voice must be the voice that reads it --------------------------------


@pytest.fixture
def voiced_settings(settings, tmp_path, monkeypatch):
    refs = tmp_path / "refs"
    refs.mkdir()
    monkeypatch.setattr(config, "VOICE_REF_DIR", refs)
    for name in ("Latradio", "NuMN1"):
        sf.write(str(refs / f"{name}.wav"), np.zeros(4410, dtype="float32"),
                 config.SAMPLE_RATE)
        settings.voices[name] = {"wav": f"{name}.wav", "transcript": "xin chào"}
    settings.default_voice = "Latradio"
    return settings


def test_the_plan_names_the_voice_that_will_read_it(voiced_settings):
    """Not the one the text asks for. Before the first marker the text asks for
    nothing, and "nothing" is not what ends up on the take."""
    plan = pipeline.sentence_plan(CHAPTERS, voiced_settings)
    assert [p.voice for p in plan] == ["Latradio"] * 3


def test_a_marker_still_wins(voiced_settings):
    chapters = [text_extract.Chapter(
        title="Ch1", text="Câu một dài vừa đủ. [NuMN1] Câu hai cũng vậy."
    )]
    plan = pipeline.sentence_plan(chapters, voiced_settings)
    assert [p.voice for p in plan] == ["Latradio", "NuMN1"]


def test_an_unmarked_book_hits_the_cache_on_the_second_export(cache, voiced_settings,
                                                              monkeypatch):
    """The failure this guards against was silent and total: the plan recorded None,
    the renderer stored "Latradio", and every key missed — so a book with no markers
    regenerated every sentence, every export, at about forty seconds each."""
    monkeypatch.setattr("diareader.voice.check_reference_readable", lambda ref: None)
    plan = pipeline.sentence_plan(CHAPTERS, voiced_settings)
    pipeline.render_sentences(FakeEngine(), voiced_settings, cache, plan, speed=1.0)

    assert [cache.get(p.chapter_index, p.position).voice for p in plan] \
        == [p.voice for p in plan]
    assert cache.missing(
        [(p.chapter_index, p.position, p.text, p.voice) for p in plan], 1.0
    ) == []


def test_no_voices_configured_still_plans(settings):
    """Unconditioned generation is a supported mode — the checkpoint was fine-tuned
    without audio prompts."""
    plan = pipeline.sentence_plan(CHAPTERS, settings)
    assert [p.voice for p in plan] == [None] * 3


# --- assembling refuses audio that is not this sentence's ----------------------------


def test_assembling_leaves_a_gap_rather_than_the_wrong_sentence(cache, settings, tmp_path):
    """A take was written into the book on the strength of sitting in the right slot,
    with nothing checking that it said the right words. After an edit shifts the
    sentences that is the *previous* sentence, read out in the reader's own voice in
    the middle of their book. A gap is recoverable; a plausible wrong sentence in a
    four-hour file is not."""
    original = [text_extract.Chapter(title="Ch1", text="Câu một dài. Câu hai dài.")]
    pipeline.render_sentences(
        FakeEngine(), settings, cache,
        pipeline.sentence_plan(original, settings), speed=1.0
    )

    edited = [text_extract.Chapter(title="Ch1", text="Câu một dài. Câu chen vào. Câu hai dài.")]
    plan = pipeline.sentence_plan(edited, settings)
    # Deliberately not realigned: this is the last line of defence, so it is tested
    # with the cache in exactly the state realign_to exists to prevent.
    result = pipeline.assemble(cache, plan, tmp_path / "book.wav", speed=1.0)

    audio, rate = sf.read(str(result), dtype="float32")
    # One second of real audio, two short gaps, and the two inter-sentence silences.
    expected = 1 + 2 * 0.2 + 2 * pipeline.INTER_CHUNK_SILENCE_S
    assert abs(len(audio) / rate - expected) < 0.05


def test_realigning_first_puts_the_shifted_sentence_back_in_the_file(cache, settings,
                                                                     tmp_path):
    original = [text_extract.Chapter(title="Ch1", text="Câu một dài. Câu hai dài.")]
    pipeline.render_sentences(
        FakeEngine(), settings, cache,
        pipeline.sentence_plan(original, settings), speed=1.0
    )

    edited = [text_extract.Chapter(title="Ch1", text="Câu một dài. Câu chen vào. Câu hai dài.")]
    plan = pipeline.sentence_plan(edited, settings)
    cache.realign_to(
        [(p.chapter_index, p.position, p.text, p.voice) for p in plan], 1.0
    )
    assert cache.get(0, 2).text == "Câu hai dài."

    result = pipeline.assemble(cache, plan, tmp_path / "book.wav", speed=1.0)
    audio, rate = sf.read(str(result), dtype="float32")
    # Both original sentences survive; only the inserted one is a gap.
    expected = 2 + 0.2 + 2 * pipeline.INTER_CHUNK_SILENCE_S
    assert abs(len(audio) / rate - expected) < 0.05
