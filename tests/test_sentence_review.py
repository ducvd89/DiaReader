"""The review table: picking sentences, choosing between takes, and what it does not do.

What it must never do is change the book. Ticking, listening and adopting all act on
the cache; the file is written later, by Export audio file, from whatever is in use.
"""
import numpy as np
import pytest
import soundfile as sf

pytest.importorskip("PySide6")

from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtGui import QFontMetrics  # noqa: E402
from PySide6.QtWidgets import QApplication, QComboBox, QMessageBox  # noqa: E402

from diareader import config, pipeline, sentence_cache, text_extract, voice  # noqa: E402
from diareader.gui import script_editor  # noqa: E402
from diareader.gui import sentence_review  # noqa: E402
from diareader.gui.sentence_review import (  # noqa: E402
    COL_HEARD, COL_SCORE, COL_TAKE, COL_TEXT, COL_VOICE, COL_WHERE, CacheSettingsDialog,
    SentenceReviewDialog
)


@pytest.fixture(scope="module")
def qt_app():
    yield QApplication.instance() or QApplication([])


@pytest.fixture
def settings(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "SETTINGS_PATH", tmp_path / "settings.json")
    monkeypatch.setattr(config, "APP_DIR", tmp_path)
    return config.Settings()


@pytest.fixture
def cache(tmp_path, monkeypatch):
    monkeypatch.setattr(sentence_cache, "CACHE_ROOT", tmp_path / "cache")
    book = tmp_path / "book.txt"
    book.write_text("x", encoding="utf-8")
    return sentence_cache.SentenceCache(book)


CHAPTERS = [text_extract.Chapter(
    title="Ch1", text="Câu một dài vừa đủ. Câu hai cũng vậy. Câu ba nữa."
)]


def _plan(settings):
    return pipeline.sentence_plan(CHAPTERS, settings)


def _store(cache, plan, index, *, score=1.0, accepted=True, voice="Vietsu", in_use=True):
    planned = plan[index]
    return cache.put(
        planned.chapter_index, planned.position,
        np.ones(config.SAMPLE_RATE, dtype=np.float32),
        title=planned.chapter_title, text=planned.text, voice=voice, speed=1.0,
        score=score, transcript=f"heard something — {score:.0%}",
        accepted=accepted, in_use=in_use,
    )


def test_every_sentence_gets_a_row(qt_app, cache, settings):
    plan = _plan(settings)
    dialog = SentenceReviewDialog(plan, cache, settings)
    assert dialog.table.rowCount() == len(plan)
    assert dialog.table.item(0, COL_TEXT).text() == plan[0].text


def test_an_unrendered_sentence_says_so(qt_app, cache, settings):
    dialog = SentenceReviewDialog(_plan(settings), cache, settings)
    assert dialog.table.item(0, COL_SCORE).text() == "not rendered"


def test_a_rendered_sentence_shows_its_verdict(qt_app, cache, settings):
    plan = _plan(settings)
    _store(cache, plan, 0, score=0.36, accepted=False)
    dialog = SentenceReviewDialog(plan, cache, settings)

    assert "36%" in dialog.table.item(0, COL_SCORE).text()
    assert "something" in dialog.table.item(0, COL_HEARD).text()


def test_ticking_the_failures_picks_the_right_rows(qt_app, cache, settings):
    """The one action anyone takes on a four-hundred-row table."""
    plan = _plan(settings)
    _store(cache, plan, 0, score=1.0, accepted=True)
    _store(cache, plan, 1, score=0.36, accepted=False)
    # plan[2] is left unrendered, which also counts as needing work.

    dialog = SentenceReviewDialog(plan, cache, settings)
    dialog._tick_failed()
    assert dialog.ticked_rows() == [1, 2]


def test_rendering_asks_for_the_ticked_sentences(qt_app, cache, settings):
    plan = _plan(settings)
    dialog = SentenceReviewDialog(plan, cache, settings)
    dialog.checks[2].setChecked(True)

    asked = []
    dialog.render_requested.connect(lambda planned, voice: asked.append((planned, voice)))
    dialog._request_render()

    (requested, voice_name), = asked
    assert [p.position for p in requested] == [2]
    assert voice_name is None


def test_choosing_a_voice_travels_with_the_request(qt_app, cache, settings):
    plan = _plan(settings)
    dialog = SentenceReviewDialog(plan, cache, settings)
    dialog.checks[0].setChecked(True)
    dialog.voice_combo.addItem("Latradio", "Latradio")
    dialog.voice_combo.setCurrentIndex(dialog.voice_combo.count() - 1)

    asked = []
    dialog.render_requested.connect(lambda planned, voice: asked.append(voice))
    dialog._request_render()
    assert asked == ["Latradio"]


def test_no_take_picker_until_there_is_a_choice(qt_app, cache, settings):
    plan = _plan(settings)
    _store(cache, plan, 0)
    dialog = SentenceReviewDialog(plan, cache, settings)
    assert dialog.take_picker(0) is None


def test_a_second_take_offers_a_choice(qt_app, cache, settings):
    plan = _plan(settings)
    _store(cache, plan, 0, voice="Vietsu")
    _store(cache, plan, 0, voice="Latradio")

    dialog = SentenceReviewDialog(plan, cache, settings)
    picker = dialog.take_picker(0)
    assert isinstance(picker, QComboBox)
    assert picker.count() == 2
    assert "Latradio" in picker.itemText(0), "the newest take is the one in use"


def test_picking_an_earlier_take_adopts_it(qt_app, cache, settings):
    plan = _plan(settings)
    first = _store(cache, plan, 0, voice="Vietsu")
    _store(cache, plan, 0, voice="Latradio")

    dialog = SentenceReviewDialog(plan, cache, settings)
    dialog.take_picker(0).setCurrentIndex(1)

    assert cache.get(0, 0).wav == first.wav
    assert cache.get(0, 0).voice == "Vietsu"


def test_the_dialog_never_touches_the_book(qt_app, cache, settings):
    """Adopting changes which take is in use and nothing else — no file is written
    until Export audio file."""
    plan = _plan(settings)
    before = [p.text for p in plan]
    _store(cache, plan, 0, voice="Vietsu")
    _store(cache, plan, 0, voice="Latradio")

    dialog = SentenceReviewDialog(plan, cache, settings)
    dialog.take_picker(0).setCurrentIndex(1)

    assert [p.text for p in dialog.plan] == before
    assert CHAPTERS[0].text.startswith("Câu một")


def test_the_status_line_counts_what_is_left(qt_app, cache, settings):
    plan = _plan(settings)
    _store(cache, plan, 0, score=0.36, accepted=False)
    dialog = SentenceReviewDialog(plan, cache, settings)
    assert "1 of 3" in dialog.status.text()
    assert "1 still failing" in dialog.status.text()


# --- colour, and what it is allowed to mean ------------------------------------------


@pytest.fixture
def voiced_settings(tmp_path, monkeypatch):
    """Settings with a real voice library, so colours can be resolved.

    Voices whose WAV is missing are dropped on load, and a dropped voice has no
    position in the library and therefore no colour — so the files have to exist.
    """
    monkeypatch.setattr(config, "SETTINGS_PATH", tmp_path / "settings.json")
    monkeypatch.setattr(config, "APP_DIR", tmp_path)
    refs = tmp_path / "voice_refs"
    refs.mkdir()
    monkeypatch.setattr(config, "VOICE_REF_DIR", refs)
    settings = config.Settings()
    settings.voices = {}
    for name in ("Vietsu", "Latradio"):
        wav = refs / f"{name}.wav"
        sf.write(str(wav), np.zeros(100, dtype="float32"), config.SAMPLE_RATE)
        settings.voices[name] = {"wav": wav.name, "transcript": "xin chào"}
    return settings


def _row_background(dialog, row, column=COL_TEXT):
    return dialog.table.item(row, column).background().color()


def test_a_rejected_sentence_turns_its_row_red(qt_app, cache, voiced_settings):
    """Finding the three bad sentences in four hundred is the whole job of this table,
    and reading four hundred percentages is not how anyone does it."""
    plan = _plan(voiced_settings)
    _store(cache, plan, 0, score=0.36, accepted=False)
    dialog = SentenceReviewDialog(plan, cache, voiced_settings)

    for column in (COL_WHERE, COL_TEXT, COL_SCORE, COL_HEARD):
        colour = _row_background(dialog, 0, column)
        assert colour.alpha() > 0, f"column {column} was left untinted"
        assert colour.red() > colour.blue(), f"column {column} is not red"


def test_an_accepted_sentence_keeps_a_plain_row(qt_app, cache, voiced_settings):
    plan = _plan(voiced_settings)
    _store(cache, plan, 0, score=1.0, accepted=True)
    dialog = SentenceReviewDialog(plan, cache, voiced_settings)
    assert _row_background(dialog, 0).alpha() == 0


def test_a_sentence_with_no_audio_yet_is_not_an_error(qt_app, cache, voiced_settings):
    """Red before the first render would make the colour mean nothing."""
    dialog = SentenceReviewDialog(_plan(voiced_settings), cache, voiced_settings)
    assert _row_background(dialog, 0).alpha() == 0


def test_the_row_stops_being_red_once_it_passes(qt_app, cache, voiced_settings):
    plan = _plan(voiced_settings)
    _store(cache, plan, 0, score=0.36, accepted=False)
    dialog = SentenceReviewDialog(plan, cache, voiced_settings)
    assert _row_background(dialog, 0).alpha() > 0

    _store(cache, plan, 0, score=1.0, accepted=True)
    dialog.refresh_row(0)
    assert _row_background(dialog, 0).alpha() == 0


def test_the_voice_wears_the_colour_it_wears_in_the_editor(qt_app, cache, voiced_settings):
    """One voice, one colour, wherever it is shown — that is the only reason a stable
    palette was worth computing in the first place."""
    plan = _plan(voiced_settings)
    _store(cache, plan, 0, voice="Latradio")
    dialog = SentenceReviewDialog(plan, cache, voiced_settings)

    library = voice.voice_names_in_library_order(voiced_settings)
    cell = dialog.table.item(0, COL_VOICE)
    assert cell.background().color() == script_editor.voice_colour("Latradio", library)
    assert cell.foreground().color() == script_editor.voice_ink("Latradio", library)


def test_the_voice_keeps_its_colour_on_a_failed_row(qt_app, cache, voiced_settings):
    """The red says "this take is wrong"; the swatch says "in this voice". Losing the
    second under the first is exactly when it is most wanted — recasting a sentence is
    the usual answer to a failure."""
    plan = _plan(voiced_settings)
    _store(cache, plan, 0, voice="Vietsu", score=0.36, accepted=False)
    dialog = SentenceReviewDialog(plan, cache, voiced_settings)

    library = voice.voice_names_in_library_order(voiced_settings)
    assert (dialog.table.item(0, COL_VOICE).background().color()
            == script_editor.voice_colour("Vietsu", library))


# --- the window itself ---------------------------------------------------------------


def test_it_does_not_block_the_windows_beside_it(qt_app, cache, settings):
    """The editor and the chapter list stay in use while a chapter is worked through
    here, which takes as long as re-rendering its bad sentences takes."""
    dialog = SentenceReviewDialog(_plan(settings), cache, settings)
    assert not dialog.isModal()
    assert dialog.windowModality() == Qt.WindowModality.NonModal


def test_it_has_a_window_frame_with_minimise_and_maximise(qt_app, cache, settings):
    """A dialog frame on Windows has neither box, and ignores the hints for them —
    the type itself has to be a window."""
    flags = SentenceReviewDialog(_plan(settings), cache, settings).windowFlags()
    assert flags & Qt.WindowType.WindowMinimizeButtonHint
    assert flags & Qt.WindowType.WindowMaximizeButtonHint
    assert flags & Qt.WindowType.Window


# --- the cache settings panel --------------------------------------------------------


def test_choosing_a_limit_records_and_applies_it(qt_app, cache, settings, monkeypatch):
    _store(cache, _plan(settings), 0)
    cache.save()
    dialog = CacheSettingsDialog(cache, settings)

    applied = []
    monkeypatch.setattr(sentence_cache, "enforce_limit",
                        lambda limit, keep=None: applied.append(limit) or 0)
    dialog.limit_combo.setCurrentIndex(dialog.limit_combo.findData(5.0))

    assert settings.cache_limit_gb == 5.0
    assert applied == [5.0]
    assert config.Settings.load().cache_limit_gb == 5.0, "the choice did not survive"


def test_the_panel_says_what_is_actually_being_used(qt_app, cache, settings):
    """Without the figure the limit is a guess — that is most of why this has a panel
    rather than a line in Settings."""
    _store(cache, _plan(settings), 0)
    cache.save()
    dialog = CacheSettingsDialog(cache, settings)
    # A small cache reported as "0 MB" reads as broken rather than as small.
    assert any(unit in dialog.usage.text() for unit in ("KB", "MB", "GB"))
    assert "0 MB" not in dialog.usage.text()


def test_clearing_the_book_empties_it_and_tells_the_table(qt_app, cache, settings, monkeypatch):
    plan = _plan(settings)
    _store(cache, plan, 0)
    cache.save()
    monkeypatch.setattr(
        "diareader.gui.sentence_review.QMessageBox.question",
        lambda *a, **k: QMessageBox.StandardButton.Yes,
    )
    dialog = CacheSettingsDialog(cache, settings)
    told = []
    dialog.cache_changed.connect(lambda: told.append(True))
    dialog._clear_book()

    assert cache.get(0, 0) is None
    assert told == [True]


def test_a_cleared_cache_puts_the_rows_back_to_unrendered(qt_app, cache, settings, monkeypatch):
    """The table is looking at takes that no longer exist; leaving stale scores on
    screen is the silent-wrong-state this app keeps meeting."""
    plan = _plan(settings)
    _store(cache, plan, 0, score=0.36, accepted=False)
    review = SentenceReviewDialog(plan, cache, settings)
    assert "36%" in review.table.item(0, COL_SCORE).text()

    monkeypatch.setattr(
        "diareader.gui.sentence_review.QMessageBox.question",
        lambda *a, **k: QMessageBox.StandardButton.Yes,
    )
    panel = CacheSettingsDialog(cache, settings, review)
    panel.cache_changed.connect(review.refresh_all)
    panel._clear_book()

    assert review.table.item(0, COL_SCORE).text() == "not rendered"
    assert _row_background(review, 0).alpha() == 0


# --- showing only what still needs work ----------------------------------------------


def test_the_filter_hides_the_sentences_that_passed(qt_app, cache, settings):
    plan = _plan(settings)
    _store(cache, plan, 0, score=1.0, accepted=True)
    _store(cache, plan, 1, score=0.36, accepted=False)
    _store(cache, plan, 2, score=1.0, accepted=True)

    dialog = SentenceReviewDialog(plan, cache, settings)
    dialog.filter_button.setChecked(True)
    assert dialog.visible_rows() == [1]


def test_a_sentence_with_no_audio_yet_survives_the_filter(qt_app, cache, settings):
    """Nothing rendered is work outstanding just as much as a rejection is — it is
    what "Tick the ones that failed" already means."""
    plan = _plan(settings)
    _store(cache, plan, 0, score=1.0, accepted=True)
    dialog = SentenceReviewDialog(plan, cache, settings)
    dialog.filter_button.setChecked(True)
    assert dialog.visible_rows() == [1, 2]


def test_a_ticked_sentence_stays_visible_even_though_it_passed(qt_app, cache, settings):
    plan = _plan(settings)
    for index in range(3):
        _store(cache, plan, index, score=1.0, accepted=True)
    dialog = SentenceReviewDialog(plan, cache, settings)
    dialog.checks[2].setChecked(True)

    dialog.filter_button.setChecked(True)
    assert dialog.visible_rows() == [2]


def test_turning_the_filter_off_brings_everything_back(qt_app, cache, settings):
    plan = _plan(settings)
    for index in range(3):
        _store(cache, plan, index, score=1.0, accepted=True)
    dialog = SentenceReviewDialog(plan, cache, settings)

    dialog.filter_button.setChecked(True)
    assert dialog.visible_rows() == []
    dialog.filter_button.setChecked(False)
    assert dialog.visible_rows() == [0, 1, 2]


def test_unticking_under_the_filter_does_not_make_the_row_vanish(qt_app, cache, settings):
    """The row would disappear from under the pointer at the moment of being unticked,
    and the only way back to it would be turning the whole filter off."""
    plan = _plan(settings)
    for index in range(3):
        _store(cache, plan, index, score=1.0, accepted=True)
    dialog = SentenceReviewDialog(plan, cache, settings)
    dialog.checks[1].setChecked(True)
    dialog.filter_button.setChecked(True)

    dialog.checks[1].setChecked(False)
    assert dialog.visible_rows() == [1]


def test_pressing_the_filter_again_recomputes_it(qt_app, cache, settings):
    plan = _plan(settings)
    for index in range(3):
        _store(cache, plan, index, score=1.0, accepted=True)
    dialog = SentenceReviewDialog(plan, cache, settings)
    dialog.checks[1].setChecked(True)
    dialog.filter_button.setChecked(True)
    dialog.checks[1].setChecked(False)

    dialog.filter_button.setChecked(False)
    dialog.filter_button.setChecked(True)
    assert dialog.visible_rows() == []


def test_a_hidden_row_is_never_a_ticked_one(qt_app, cache, settings):
    """"Render ticked" must not act on a sentence the table is not showing."""
    plan = _plan(settings)
    for index in range(3):
        _store(cache, plan, index, score=1.0, accepted=True)
    dialog = SentenceReviewDialog(plan, cache, settings)
    dialog.filter_button.setChecked(True)
    dialog._tick_failed()

    hidden = set(range(len(plan))) - set(dialog.visible_rows())
    assert hidden.isdisjoint(dialog.ticked_rows())


def test_a_sentence_that_passes_on_the_retake_stays_on_screen(qt_app, cache, settings):
    """It was just re-rendered because it failed; hiding the result the instant it
    succeeds takes away the one thing the user was waiting to see."""
    plan = _plan(settings)
    _store(cache, plan, 0, score=0.36, accepted=False)
    _store(cache, plan, 1, score=1.0, accepted=True)
    _store(cache, plan, 2, score=1.0, accepted=True)
    dialog = SentenceReviewDialog(plan, cache, settings)
    dialog.filter_button.setChecked(True)
    assert dialog.visible_rows() == [0]

    _store(cache, plan, 0, score=1.0, accepted=True)
    dialog.refresh_all()
    assert dialog.visible_rows() == [0]


def test_the_status_line_admits_it_is_hiding_things(qt_app, cache, settings):
    """A filtered table looks exactly like a short book."""
    plan = _plan(settings)
    for index in range(3):
        _store(cache, plan, index, score=1.0, accepted=True)
    dialog = SentenceReviewDialog(plan, cache, settings)
    assert "hidden" not in dialog.status.text()

    dialog.filter_button.setChecked(True)
    assert "3 hidden by the filter" in dialog.status.text()


# --- row height and text size --------------------------------------------------------


def test_rows_are_sized_from_the_text_not_from_the_widgets(qt_app, cache, settings):
    """The play button and the take picker both ask for 26px against a 17px line, and
    while they were the tallest thing in a row they, not the text, set its height."""
    from PySide6.QtWidgets import QPushButton as _Button

    plan = _plan(settings)
    _store(cache, plan, 0)
    dialog = SentenceReviewDialog(plan, cache, settings)

    line = QFontMetrics(dialog.table.font()).height()
    height = dialog.table.rowHeight(0)
    assert height < _Button("x").sizeHint().height()
    assert line < height <= line + 2 * sentence_review.ROW_PADDING_PX


def test_rows_of_equal_length_come_out_equal(qt_app, cache, settings):
    plan = _plan(settings)
    _store(cache, plan, 0)
    dialog = SentenceReviewDialog(plan, cache, settings)
    dialog.table.setColumnWidth(COL_TEXT, 600)   # wide enough that none of them wrap
    dialog._resize_rows()
    assert len({dialog.table.rowHeight(r) for r in range(len(plan))}) == 1


def test_zooming_in_makes_the_rows_taller(qt_app, cache, settings):
    dialog = SentenceReviewDialog(_plan(settings), cache, settings)
    before = dialog.table.rowHeight(0)
    dialog.zoom_by(4)
    assert dialog.table.rowHeight(0) > before
    dialog.zoom_by(-4)
    assert dialog.table.rowHeight(0) == before


def test_the_text_size_is_clamped_at_both_ends(qt_app, cache, settings):
    dialog = SentenceReviewDialog(_plan(settings), cache, settings)
    dialog.set_font_size(9999)
    assert dialog._font_pt == sentence_review.MAX_FONT_PT
    assert not dialog.zoom_in_button.isEnabled()
    dialog.set_font_size(-5)
    assert dialog._font_pt == sentence_review.MIN_FONT_PT
    assert not dialog.zoom_out_button.isEnabled()


def test_the_chosen_size_outlives_the_window(qt_app, cache, settings):
    dialog = SentenceReviewDialog(_plan(settings), cache, settings)
    dialog.zoom_by(3)
    assert config.Settings.load().sentences_font_pt == dialog._font_pt

    reopened = SentenceReviewDialog(_plan(settings), cache, config.Settings.load())
    assert reopened._font_pt == dialog._font_pt


def test_a_take_picker_built_later_still_fits_its_row(qt_app, cache, settings):
    """It appears only once a sentence has a second take, which is long after the
    table was sized — at its natural height it would push that one row taller."""
    plan = _plan(settings)
    _store(cache, plan, 0, voice="Vietsu")
    dialog = SentenceReviewDialog(plan, cache, settings)
    dialog.zoom_by(-2)

    _store(cache, plan, 0, voice="Latradio")
    dialog.refresh_row(0)

    picker = dialog.take_picker(0)
    assert picker is not None
    assert picker.height() <= dialog.table.rowHeight(0)


# --- wrapping, column widths, and the Heard column -----------------------------------


LONG = ("Trong quá trình sử dụng sản phẩm dịch vụ của Bi Đi Vi, anh Minh có gặp khó "
        "khăn hoặc vướng mắc gì cần em hỗ trợ không ạ.")


@pytest.fixture
def long_plan(settings):
    chapters = [text_extract.Chapter(title="Ch1", text="Câu một. " + LONG + " Câu ba.")]
    return pipeline.sentence_plan(chapters, settings)


def test_a_long_sentence_is_wrapped_not_hidden(qt_app, cache, settings, long_plan):
    """An ellipsis lands on the one thing this table exists to show you: which words
    the sentence actually contains."""
    dialog = SentenceReviewDialog(long_plan, cache, settings)
    assert dialog.table.wordWrap()
    assert dialog.table.item(1, COL_TEXT).text() == long_plan[1].text
    assert dialog.table.rowHeight(1) > dialog.table.rowHeight(0)


def test_narrowing_a_column_makes_its_rows_taller(qt_app, cache, settings, long_plan):
    """The text has to stay whole at any column width, which means the row has to
    grow when the column shrinks."""
    dialog = SentenceReviewDialog(long_plan, cache, settings)
    wide = dialog.table.rowHeight(1)
    dialog.table.setColumnWidth(COL_TEXT, 150)
    dialog._resize_rows()
    assert dialog.table.rowHeight(1) > wide


def test_every_column_can_be_dragged(qt_app, cache, settings):
    """Which of the two long columns matters changes with the job: casting reads the
    sentence, chasing a failure reads the transcript."""
    from PySide6.QtWidgets import QHeaderView

    dialog = SentenceReviewDialog(_plan(settings), cache, settings)
    header = dialog.table.horizontalHeader()
    for column in range(dialog.table.columnCount()):
        assert header.sectionResizeMode(column) == QHeaderView.ResizeMode.Interactive


def test_the_heard_column_does_not_repeat_its_own_heading(qt_app, cache, settings):
    plan = _plan(settings)
    _store(cache, plan, 0)
    dialog = SentenceReviewDialog(plan, cache, settings)
    cell = dialog.table.item(0, COL_HEARD)

    assert not cell.text().lower().startswith("heard")
    assert cell.toolTip().startswith("heard"), "the full detail stays on the tooltip"


@pytest.mark.parametrize("stored, shown", [
    ("heard 'dạ em chào anh minh' — 5 of 5 syllables",
     "dạ em chào anh minh — 5 of 5 syllables"),
    ("heard 'trong quá trình' — 29 of 29 syllables (+1 allowed for gì)",
     "trong quá trình — 29 of 29 syllables (+1 allowed for gì)"),
    # Not every value of this field is a transcript.
    ("checker unavailable (OSError)", "checker unavailable (OSError)"),
    ("nothing to say", "nothing to say"),
    ("", ""),
])
def test_what_the_heard_column_shows(stored, shown):
    assert sentence_review.heard_text(stored) == shown
