"""The script editor: stable colours, marker insertion, and export from edited text."""
import pytest

from diareader import chunking, config

QColorClass = pytest.importorskip("PySide6.QtGui").QColor


@pytest.fixture(scope="module")
def qt_app():
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def test_a_voice_keeps_its_colour_when_another_is_added(qt_app):
    """Adding a voice appends, so it takes the next free swatch instead of
    recolouring every script the user has already learned to read."""
    from diareader.gui.script_editor import voice_colour

    library = ["GiongNam", "GiongNu2"]
    before = voice_colour("GiongNam", library).name()

    grown = library + ["SomeoneElse"]
    assert voice_colour("GiongNam", grown).name() == before
    assert voice_colour("GiongNu2", grown).name() == voice_colour("GiongNu2", library).name()


def test_every_voice_gets_a_different_colour(qt_app):
    """The bug this replaced: md5(name) % len(palette) collided, so "Latradio" and
    "NuMN1" rendered in exactly the same colour and could not be told apart."""
    from diareader.gui.script_editor import _PALETTE, voice_colour

    names = ["Latradio", "NamMN1", "NuMN1", "GiongNu2", "GiongNam3"]
    colours = {voice_colour(n, names).name() for n in names}
    assert len(colours) == len(names), f"colour collision among {names}: {colours}"

    # and the whole palette stays collision-free up to its own length
    full = [f"V{i}" for i in range(len(_PALETTE))]
    assert len({voice_colour(n, full).name() for n in full}) == len(_PALETTE)


def test_unknown_voice_is_not_painted_as_a_real_one(qt_app):
    from diareader.gui.script_editor import _PALETTE, voice_colour

    library = ["Latradio", "NamMN1"]
    unknown = voice_colour("DeletedVoice", library).name()
    assert unknown not in [c.lower() for c in _PALETTE]


def test_no_realistic_voice_count_lands_on_the_error_red(qt_app):
    """A sixth voice used to be painted series red, ΔE 4.8 from the red that means
    "this voice no longer exists" — two near-identical colours for opposite facts."""
    import math

    from diareader.gui.script_editor import _PALETTE, _UNKNOWN_COLOUR

    def srgb_to_oklab(hex_colour):
        h = hex_colour.lstrip("#")
        rgb = []
        for i in (0, 2, 4):
            c = int(h[i:i + 2], 16) / 255
            rgb.append(c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4)
        r, g, b = rgb
        l = (0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b) ** (1 / 3)
        m = (0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b) ** (1 / 3)
        s = (0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b) ** (1 / 3)
        return (0.2104542553 * l + 0.7936177850 * m - 0.0040720468 * s,
                1.9779984951 * l - 2.4285922050 * m + 0.4505937099 * s,
                0.0259040371 * l + 0.7827717662 * m - 0.8086757660 * s)

    def delta_e(a, b):
        return 100 * math.dist(srgb_to_oklab(a), srgb_to_oklab(b))

    # Six voices is where this bit in practice, and six slots are hue-safe.
    for slot, colour in enumerate(_PALETTE[:6], start=1):
        d = delta_e(colour, _UNKNOWN_COLOUR)
        assert d >= 15, f"slot {slot} ({colour}) is only ΔE {d:.1f} from the error red"

    # Beyond that, honesty about the limit: eight categorical hues cover most of the
    # wheel, so no error colour can stay 15 from all of them. Slots 7-8 are close to
    # the error red on purpose-by-necessity, which is exactly why the error marker
    # also carries a wave underline (see the next test). Asserting otherwise here
    # would be asserting something no palette can deliver.
    warm_tail = [delta_e(c, _UNKNOWN_COLOUR) for c in _PALETTE[6:]]
    assert all(d < 15 for d in warm_tail), (
        "slots 7-8 became hue-separable from the error red; if that is real, tighten "
        "this test — but check it is not the error colour that drifted"
    )


def _span_backgrounds(editor):
    """Map every formatted fragment in the document to its background colour name."""
    out = []
    text = editor.toPlainText()
    block = editor.document().firstBlock()
    offset = 0
    while block.isValid():
        for r in block.layout().formats():
            frag = block.text()[r.start:r.start + r.length]
            out.append((frag, r.format.background().color().name()))
        offset += len(block.text()) + 1
        block = block.next()
    del text, offset
    return out


def test_text_after_a_marker_is_tinted_with_that_voice(qt_app):
    from diareader.gui.script_editor import (
        MarkerHighlighter, ScriptEditor, voice_span_tint,
    )

    names = ["Latradio", "GiongNu2"]
    editor = ScriptEditor()
    editor.setPlainText("Người dẫn chuyện nói. [GiongNu2] Cô ấy trả lời.")
    hl = MarkerHighlighter(editor.document(), names, start_voice="Latradio")
    hl.rehighlight()

    spans = dict(_span_backgrounds(editor))
    expected = voice_span_tint("GiongNu2", names, hl.base_colour).name()
    assert spans.get(" Cô ấy trả lời.") == expected


def test_the_starting_voice_is_left_untinted(qt_app):
    """The exception the tinting exists to make readable: only a voice that was
    *dropped in* is worth marking, so the default narration stays plain."""
    from diareader.gui.script_editor import MarkerHighlighter, ScriptEditor

    names = ["Latradio", "GiongNu2"]
    editor = ScriptEditor()
    editor.setPlainText("Người dẫn chuyện nói. [GiongNu2] Cô ấy trả lời.")
    hl = MarkerHighlighter(editor.document(), names, start_voice="Latradio")
    hl.rehighlight()

    fragments = [f for f, _ in _span_backgrounds(editor)]
    assert "Người dẫn chuyện nói. " not in fragments, "start-voice text must not be tinted"


def test_a_marker_naming_the_starting_voice_removes_the_tint(qt_app):
    """Handing narration back to the starting voice returns it to plain text — the
    tint tracks the voice, not merely the presence of a marker."""
    from diareader.gui.script_editor import MarkerHighlighter, ScriptEditor

    names = ["Latradio", "GiongNu2"]
    editor = ScriptEditor()
    editor.setPlainText("[GiongNu2] Cô nói. [Latradio] Người dẫn nói tiếp.")
    hl = MarkerHighlighter(editor.document(), names, start_voice="Latradio")
    hl.rehighlight()

    fragments = [f for f, _ in _span_backgrounds(editor)]
    assert " Người dẫn nói tiếp." not in fragments


def test_a_voice_keeps_tinting_across_paragraphs(qt_app):
    """A marker governs until the next one, however many blocks later that is —
    tinting only the paragraph it sits in would misrepresent who reads the rest."""
    from diareader.gui.script_editor import (
        MarkerHighlighter, ScriptEditor, voice_span_tint,
    )

    names = ["Latradio", "GiongNu2"]
    editor = ScriptEditor()
    editor.setPlainText("[GiongNu2] Dòng một.\nDòng hai.\nDòng ba.")
    hl = MarkerHighlighter(editor.document(), names, start_voice="Latradio")
    hl.rehighlight()

    spans = dict(_span_backgrounds(editor))
    expected = voice_span_tint("GiongNu2", names, hl.base_colour).name()
    assert spans.get("Dòng hai.") == expected
    assert spans.get("Dòng ba.") == expected


def test_changing_the_starting_voice_repaints(qt_app):
    from diareader.gui.script_editor import MarkerHighlighter, ScriptEditor

    names = ["Latradio", "GiongNu2"]
    editor = ScriptEditor()
    editor.setPlainText("[GiongNu2] Cô nói.")
    hl = MarkerHighlighter(editor.document(), names, start_voice="Latradio")
    hl.rehighlight()
    assert " Cô nói." in [f for f, _ in _span_backgrounds(editor)]

    hl.set_start_voice("GiongNu2")   # now it IS the starting voice
    assert " Cô nói." not in [f for f, _ in _span_backgrounds(editor)]


def test_span_tint_keeps_the_hue_but_not_the_intensity(qt_app):
    """Using the palette colour directly as text would be unreadable — violet
    measures 1.95:1 against a dark editor. The tint has to stay close to the
    background while still leaning towards the voice's hue."""
    from diareader.gui.script_editor import voice_colour, voice_span_tint

    names = ["Latradio"]
    base = QColorClass("#1e1e1e")
    tint = voice_span_tint("Latradio", names, base)
    hue = voice_colour("Latradio", names)

    for channel in ("red", "green", "blue"):
        b, t, h = (getattr(c, channel)() for c in (base, tint, hue))
        assert min(b, h) <= t <= max(b, h), f"{channel} tint is outside base..hue"
    assert tint.name() != hue.name(), "the wash must not be the full-strength colour"
    assert tint.name() != base.name(), "and it must actually be visible"


def test_an_unknown_marker_is_flagged_without_relying_on_colour(qt_app):
    """Slot 8 is series red and no hue is far enough from the error red to carry the
    distinction alone, so the error must also be marked in another channel."""
    from PySide6.QtGui import QTextCharFormat

    from diareader.gui.script_editor import MarkerHighlighter, ScriptEditor

    editor = ScriptEditor()
    editor.setPlainText("[Latradio] Xin chào. [DaBiXoa] Câu này hỏng.")
    highlighter = MarkerHighlighter(editor.document(), ["Latradio"])
    highlighter.rehighlight()

    text = editor.toPlainText()
    block = editor.document().firstBlock()
    styles = {}
    for fmt_range in block.layout().formats():
        fragment = text[fmt_range.start:fmt_range.start + fmt_range.length]
        styles[fragment] = fmt_range.format.underlineStyle()

    assert styles.get("[DaBiXoa]") == QTextCharFormat.UnderlineStyle.WaveUnderline
    assert styles.get("[Latradio]") != QTextCharFormat.UnderlineStyle.WaveUnderline


def test_text_stays_readable_on_every_swatch(qt_app):
    """Violet with dark ink is 2.45:1 — unreadable. Ink is picked per swatch, so
    every pairing has to clear 4.5:1."""
    from diareader.gui.script_editor import _PALETTE, voice_colour, voice_ink

    def luminance(qcolor):
        parts = []
        for c in (qcolor.redF(), qcolor.greenF(), qcolor.blueF()):
            parts.append(c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4)
        return 0.2126 * parts[0] + 0.7152 * parts[1] + 0.0722 * parts[2]

    library = [f"V{i}" for i in range(len(_PALETTE))]
    for name in library:
        bg, ink = voice_colour(name, library), voice_ink(name, library)
        hi, lo = sorted([luminance(bg), luminance(ink)], reverse=True)
        ratio = (hi + 0.05) / (lo + 0.05)
        assert ratio >= 4.5, f"{bg.name()} + {ink.name()} is only {ratio:.2f}:1"


def test_chip_drag_carries_the_marker_text(qt_app):
    """The drag payload is the marker itself, so the editor's ordinary text-drop
    handling places it — no custom drop logic to get wrong."""
    from diareader.gui.script_editor import VoiceChip

    chip = VoiceChip("GiongNu2")
    assert chip.name == "GiongNu2"


def test_insert_marker_puts_a_usable_tag_in_the_text(qt_app):
    from diareader.gui.script_editor import ScriptEditor

    editor = ScriptEditor()
    editor.setPlainText("Xin chào. ")
    editor.moveCursor(editor.textCursor().MoveOperation.End)
    editor.insert_marker("GiongNu2")

    text = editor.toPlainText()
    assert "[GiongNu2]" in text
    # and the pipeline must actually recognise what was inserted
    assert chunking.VOICE_MARKER.search(text)


def test_edited_text_drives_the_voices(qt_app):
    """The whole point: markers placed in the editor decide who reads what."""
    edited = "Người dẫn chuyện nói. [GiongNu2] Cô ấy trả lời. Rồi nói thêm."
    chunks = chunking.chunk_text_with_voices(edited, start_voice="Latradio")

    assert [c.voice for c in chunks] == ["Latradio", "GiongNu2", "GiongNu2"]
    assert not any("[" in c.text for c in chunks), "markers must not be spoken"


def test_dialog_reports_sentences_and_start_voice(qt_app):
    from diareader.gui.script_editor import ScriptEditorDialog

    settings = config.Settings()
    dlg = ScriptEditorDialog(settings, "Chương 1", "Câu một. Câu hai. Câu ba.")

    assert "3 sentences" in dlg.summary.text()
    assert dlg.script_text().startswith("Câu một.")


def test_dialog_with_no_voices_still_opens(qt_app):
    """A user may edit before configuring any voice; that must not crash."""
    from diareader.gui.script_editor import ScriptEditorDialog

    dlg = ScriptEditorDialog(config.Settings(voices={}), "Ch", "Xin chào.")
    assert dlg.start_voice() is None


def test_saving_emits_the_text_and_leaves_the_dialog_open(qt_app):
    """Tagging a chapter is iterative — a save that also closed the window meant
    reopening it to place the next marker."""
    from diareader.gui.script_editor import ScriptEditorDialog

    dlg = ScriptEditorDialog(config.Settings(), "Ch", "Xin chào.")
    saved, closed = [], []
    dlg.script_saved.connect(saved.append)
    dlg.finished.connect(closed.append)   # fires on accept() or reject()

    dlg.editor.setPlainText("Xin chào. [GiongNu2] Chào bạn.")
    dlg.save_button.click()

    assert saved == ["Xin chào. [GiongNu2] Chào bạn."]
    assert not closed, "saving must not close the editor"


def test_save_is_offered_only_when_there_is_something_to_save(qt_app):
    from diareader.gui.script_editor import ScriptEditorDialog

    dlg = ScriptEditorDialog(config.Settings(), "Ch", "Xin chào.")
    assert not dlg.save_button.isEnabled()
    assert dlg.dirty_label.text() == "Saved"

    dlg.editor.setPlainText("Xin chào. Thêm câu.")
    assert dlg.save_button.isEnabled()
    assert dlg.dirty_label.text() == "Unsaved changes"

    dlg.save_button.click()
    assert not dlg.save_button.isEnabled()
    assert dlg.dirty_label.text() == "Saved"


def test_export_button_reports_export_action(qt_app):
    from diareader.gui.script_editor import ScriptEditorDialog

    dlg = ScriptEditorDialog(config.Settings(), "Ch", "Xin chào.")
    dlg.export_button.click()
    assert dlg.action() == "export"


def test_cursor_sentence_number_matches_what_the_export_log_prints(qt_app):
    """The whole point of the readout: the export log reports failures as
    "Sentence 75 of '<chapter>'", so this number has to be that same number or it
    cannot be used to find them."""
    from diareader.gui.script_editor import sentence_at

    text = "Câu một. Câu hai. Câu ba. Câu bốn."
    pipeline_chunks = chunking.chunk_text_with_voices(text)

    # walk the cursor through the middle of every sentence and check the readout
    # against the index the pipeline would have given that same chunk
    for expected_index, chunk in enumerate(pipeline_chunks, start=1):
        middle = text.index(chunk.text) + len(chunk.text) // 2
        number, _ = sentence_at(text, middle)
        assert number == expected_index, f"cursor inside {chunk.text!r} reported {number}"


def test_cursor_at_the_very_start_of_a_sentence_reports_that_sentence(qt_app):
    """The off-by-one that makes a readout useless: resting at the start of
    sentence 3 must say 3, not the 2 completed sentences behind the cursor."""
    from diareader.gui.script_editor import sentence_at

    text = "Câu một. Câu hai. Câu ba."
    assert sentence_at(text, text.index("Câu ba."))[0] == 3
    assert sentence_at(text, 0)[0] == 1


def test_cursor_readout_reports_the_active_voice(qt_app):
    from diareader.gui.script_editor import sentence_at

    text = "Người dẫn chuyện nói. [GiongNu2] Cô ấy trả lời."
    assert sentence_at(text, 5, start_voice="Latradio")[1] == "Latradio"
    assert sentence_at(text, text.index("Cô ấy"), start_voice="Latradio")[1] == "GiongNu2"


def test_dialog_shows_the_cursor_position(qt_app):
    from diareader.gui.script_editor import ScriptEditorDialog

    dlg = ScriptEditorDialog(config.Settings(), "Ch", "Câu một. Câu hai. Câu ba.")
    cursor = dlg.editor.textCursor()
    cursor.setPosition(dlg.editor.toPlainText().index("Câu hai."))
    dlg.editor.setTextCursor(cursor)

    assert dlg.sentence_position()[0] == 2
    assert "Sentence 2 of 3" in dlg.position_label.text()


def test_position_readout_is_blank_for_an_empty_script(qt_app):
    from diareader.gui.script_editor import ScriptEditorDialog

    dlg = ScriptEditorDialog(config.Settings(), "Ch", "")
    assert dlg.position_label.text() == ""


def test_reset_is_not_offered_without_a_source_to_reset_from(qt_app):
    """Better a disabled button than one that silently does nothing."""
    from diareader.gui.script_editor import ScriptEditorDialog

    dlg = ScriptEditorDialog(config.Settings(), "Ch", "Xin chào.")
    assert not dlg.reset_button.isEnabled()

    dlg2 = ScriptEditorDialog(config.Settings(), "Ch", "Xin chào.", on_reset=lambda: "gốc")
    assert dlg2.reset_button.isEnabled()


def test_reset_replaces_the_text_and_marks_it_unsaved(qt_app, monkeypatch):
    """A reset is itself a change to be saved — leaving it "Saved" would imply the
    stored copy had been reverted too, which it has not."""
    from PySide6.QtWidgets import QMessageBox

    from diareader.gui.script_editor import ScriptEditorDialog

    monkeypatch.setattr(QMessageBox, "question",
                        staticmethod(lambda *a, **k: QMessageBox.StandardButton.Yes))
    dlg = ScriptEditorDialog(
        config.Settings(), "Ch", "Đã sửa. [GiongNu2] Có tag.",
        on_reset=lambda: "Văn bản gốc từ epub.",
    )
    dlg.reset_button.click()

    assert dlg.script_text() == "Văn bản gốc từ epub."
    assert dlg.dirty_label.text() == "Unsaved changes"


def test_declining_the_reset_confirmation_changes_nothing(qt_app, monkeypatch):
    from PySide6.QtWidgets import QMessageBox

    from diareader.gui.script_editor import ScriptEditorDialog

    monkeypatch.setattr(QMessageBox, "question",
                        staticmethod(lambda *a, **k: QMessageBox.StandardButton.No))
    dlg = ScriptEditorDialog(
        config.Settings(), "Ch", "Giữ nguyên.", on_reset=lambda: "KHÔNG ĐƯỢC DÙNG"
    )
    dlg.reset_button.click()
    assert dlg.script_text() == "Giữ nguyên."


def test_zoom_changes_the_font_and_is_remembered(qt_app):
    from diareader.gui.script_editor import MAX_FONT_PT, MIN_FONT_PT, ScriptEditorDialog

    settings = config.Settings()
    settings.save = lambda: None
    dlg = ScriptEditorDialog(settings, "Ch", "Xin chào.")
    start = dlg.editor.font().pointSize()

    dlg.zoom_in_button.click()
    assert dlg.editor.font().pointSize() == start + 1
    assert settings.script_font_pt == start + 1, "zoom must survive reopening the editor"

    dlg.zoom_out_button.click()
    assert dlg.editor.font().pointSize() == start

    dlg.editor.set_font_size(999)
    assert dlg.editor.font().pointSize() == MAX_FONT_PT
    dlg.editor.set_font_size(1)
    assert dlg.editor.font().pointSize() == MIN_FONT_PT


def test_editor_opens_at_the_remembered_size(qt_app):
    from diareader.gui.script_editor import ScriptEditorDialog

    settings = config.Settings(script_font_pt=18)
    settings.save = lambda: None
    dlg = ScriptEditorDialog(settings, "Ch", "Xin chào.")
    assert dlg.editor.font().pointSize() == 18


def test_nonverbal_button_inserts_the_tag_at_the_cursor(qt_app):
    from diareader.gui.script_editor import ScriptEditorDialog

    dlg = ScriptEditorDialog(config.Settings(), "Ch", "Hắn cười. ")
    dlg.editor.moveCursor(dlg.editor.textCursor().MoveOperation.End)
    button = next(b for b in dlg.nonverbal_buttons if b.tag == "(chuckle)")
    button.click()

    assert "(chuckle)" in dlg.editor.toPlainText()


def test_nonverbal_tags_are_only_the_documented_ones(qt_app):
    """Dia warns that unlisted non-verbals cause artifacts, so the buttons must not
    offer anything outside its published set."""
    from diareader.gui.script_editor import NONVERBAL_TAGS, _NONVERBAL_RE

    assert "(chuckle)" in NONVERBAL_TAGS and "(laughs)" in NONVERBAL_TAGS
    assert all(t.startswith("(") and t.endswith(")") for t in NONVERBAL_TAGS)
    # "(sigh)" is a real near-miss — it appears in book text, but Dia's tag is
    # "(sighs)", so it must not be highlighted as a recognised direction.
    assert _NONVERBAL_RE.fullmatch("(sighs)")
    assert not _NONVERBAL_RE.fullmatch("(sigh)")


def test_a_nonverbal_tag_does_not_become_a_voice_marker(qt_app):
    """Voice markers and non-verbal tags use different brackets on purpose; a tag
    must survive chunking as ordinary text for the model to perform."""
    chunks = chunking.chunk_text_with_voices(
        "Hắn cười (chuckle) rồi bỏ đi.", start_voice="Latradio"
    )
    assert [c.voice for c in chunks] == ["Latradio"]
    assert "(chuckle)" in chunks[0].text
