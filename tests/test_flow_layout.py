"""The wrapping voice palette.

What this replaced: a QHBoxLayout holding one chip per voice. At seventeen voices the
chips ran past the right edge of the script editor and widened the dialog off the
screen with them, which is not something the user can drag back.
"""
import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication, QLabel  # noqa: E402

from diareader.gui.flow_layout import FlowLayout, wrapping_row  # noqa: E402


@pytest.fixture(scope="module")
def qt_app():
    yield QApplication.instance() or QApplication([])


def _row(qt_app, count=17, width=500):
    chips = [QLabel(f"Voice{i:02d}") for i in range(count)]
    for chip in chips:
        chip.setStyleSheet("padding: 4px 10px;")
    holder = wrapping_row(chips)
    holder.show()
    holder.setFixedWidth(width)
    holder.adjustSize()
    qt_app.processEvents()
    return holder, chips


def test_nothing_is_placed_past_the_right_edge(qt_app):
    """The bug itself: content wider than the window, with no way to reach it."""
    for width in (1200, 800, 500, 320):
        holder, chips = _row(qt_app, width=width)
        assert max(c.x() + c.width() for c in chips) <= width


def test_narrower_windows_use_more_rows(qt_app):
    heights = []
    for width in (1200, 800, 500, 320):
        holder, _ = _row(qt_app, width=width)
        heights.append(holder.layout().heightForWidth(width))
    assert heights == sorted(heights), "height must grow as width shrinks"
    assert heights[0] < heights[-1]


def test_height_for_width_is_offered_at_all(qt_app):
    """Without this a wrapping layout gets one row of vertical space and clips the
    rest — the rows exist but cannot be seen."""
    layout = FlowLayout()
    assert layout.hasHeightForWidth()


def test_every_chip_is_placed(qt_app):
    holder, chips = _row(qt_app, count=17, width=400)
    assert holder.layout().count() == 17
    assert all(c.width() > 0 and c.height() > 0 for c in chips)


def test_an_item_wider_than_the_row_is_still_placed(qt_app):
    """A single very long voice name must be laid out, not wrapped forever."""
    long_chip = QLabel("A" * 200)
    holder = wrapping_row([long_chip])
    holder.show()
    holder.setFixedWidth(120)
    holder.adjustSize()
    qt_app.processEvents()
    assert long_chip.width() > 0


def test_empty_layout_has_no_height(qt_app):
    holder = wrapping_row([])
    assert holder.layout().heightForWidth(500) <= 0


def _settings_with_voices(tmp_path, monkeypatch, count=20):
    """Settings holding `count` voices whose WAVs exist.

    The files have to be real: load_voices drops any voice pointing at a missing one,
    so fabricating the dict alone yields a dialog with no chips at all — which passes
    a "is it narrow?" test for entirely the wrong reason.
    """
    from diareader import config

    refs = tmp_path / "voice_refs"
    refs.mkdir()
    monkeypatch.setattr(config, "SETTINGS_PATH", tmp_path / "settings.json")
    monkeypatch.setattr(config, "APP_DIR", tmp_path)
    monkeypatch.setattr(config, "VOICE_REF_DIR", refs)

    settings = config.Settings()
    settings.voices = {}
    for i in range(count):
        (refs / f"v{i}.wav").write_bytes(b"")
        settings.voices[f"VoiceNumber{i:02d}"] = {
            "wav": f"v{i}.wav", "transcript": "xin chào"
        }
    return settings


def test_the_script_editor_can_be_made_narrow(qt_app, tmp_path, monkeypatch):
    """The dialog demanded 1570px minimum, so on a laptop its edges sat off the screen
    with no way to drag them back. Two separate causes: chips on one non-wrapping row,
    and QLabels with word wrap off — a QLabel reports its whole text as its *minimum*
    width, and QHBoxLayout adds those minimums up.
    """
    from diareader.gui.script_editor import ScriptEditorDialog

    settings = _settings_with_voices(tmp_path, monkeypatch)
    dialog = ScriptEditorDialog(settings, "Chương 1", "Xin chào. Câu thứ hai.")
    dialog.show()
    qt_app.processEvents()

    palette = dialog.voice_palette.layout()
    assert palette.count() > 20, "the fixture must actually produce chips"
    assert dialog.minimumWidth() < 1000, (
        f"twenty voices must not force a {dialog.minimumWidth()}px dialog"
    )


def test_the_voice_palette_reflows_with_the_window(qt_app, tmp_path, monkeypatch):
    from diareader.gui.script_editor import ScriptEditorDialog

    settings = _settings_with_voices(tmp_path, monkeypatch)
    dialog = ScriptEditorDialog(settings, "Chương 1", "Xin chào.")
    dialog.show()
    qt_app.processEvents()

    palette = dialog.voice_palette.layout()
    chips = [palette.itemAt(i).widget() for i in range(palette.count())]

    rows = []
    for width in (1800, 1200, dialog.minimumWidth()):
        dialog.resize(width, 700)
        qt_app.processEvents()
        qt_app.processEvents()
        rows.append(len({c.y() for c in chips}))

    assert rows == sorted(rows), f"narrower windows must use more rows, got {rows}"
    assert rows[0] < rows[-1]


def test_the_script_editor_can_be_made_short(qt_app, tmp_path, monkeypatch):
    """The height version of the same bug, and the worse one: a 1107px minimum height
    put the non-verbal buttons and the Export row below the bottom of a laptop screen.

    Its cause is subtle. Qt derives a widget's minimum height from heightForWidth at
    the layout's *minimum* width — one chip wide for a flow layout — so it answers
    with every chip stacked in a single column.
    """
    from diareader.gui.script_editor import ScriptEditorDialog

    settings = _settings_with_voices(tmp_path, monkeypatch)
    dialog = ScriptEditorDialog(settings, "Chương 1", "Xin chào.")
    dialog.show()
    qt_app.processEvents()

    assert dialog.minimumHeight() < 700, (
        f"a {dialog.minimumHeight()}px minimum height does not fit a laptop screen"
    )


def test_the_editor_offers_minimise_and_maximise(qt_app, tmp_path, monkeypatch):
    """A QDialog gets only a close button from Windows, and this one is sat with for
    as long as casting a chapter takes."""
    from PySide6.QtCore import Qt

    from diareader.gui.script_editor import ScriptEditorDialog

    settings = _settings_with_voices(tmp_path, monkeypatch)
    dialog = ScriptEditorDialog(settings, "Chương 1", "Xin chào.")

    flags = dialog.windowFlags()
    assert flags & Qt.WindowType.WindowMinimizeButtonHint
    assert flags & Qt.WindowType.WindowMaximizeButtonHint


def test_the_editor_never_opens_larger_than_the_screen(qt_app, tmp_path, monkeypatch):
    from PySide6.QtGui import QGuiApplication

    from diareader.gui.script_editor import ScriptEditorDialog

    settings = _settings_with_voices(tmp_path, monkeypatch)
    dialog = ScriptEditorDialog(settings, "Chương 1", "Xin chào.")
    dialog.show()
    qt_app.processEvents()

    available = QGuiApplication.primaryScreen().availableGeometry()
    # A window can never be smaller than its own minimum, so that is the honest bound:
    # the guard shrinks the *requested* size to the screen, it cannot overrule the
    # layout. What it must never do is leave the dialog at its 900x640 design size on a
    # screen with less room than that.
    assert dialog.width() <= max(available.width(), dialog.minimumWidth())
    assert dialog.height() <= max(available.height(), dialog.minimumHeight())
    assert dialog.height() <= available.height(), (
        "height has no excuse — the minimum is well under any real screen"
    )


def test_the_nonverbal_buttons_reflow_too(qt_app, tmp_path, monkeypatch):
    """They were a QGridLayout at a fixed seven columns: it wraps, but only ever at
    the one width someone guessed when writing it."""
    from diareader.gui.script_editor import ScriptEditorDialog

    settings = _settings_with_voices(tmp_path, monkeypatch)
    dialog = ScriptEditorDialog(settings, "Chương 1", "Xin chào.")
    dialog.show()
    qt_app.processEvents()

    rows = []
    for width in (1800, dialog.minimumWidth()):
        dialog.resize(width, 700)
        qt_app.processEvents()
        qt_app.processEvents()
        rows.append(len({b.y() for b in dialog.nonverbal_buttons}))

    assert rows[0] < rows[-1], f"non-verbal buttons did not reflow, got {rows}"


def test_takeat_removes_items(qt_app):
    """QLayout's contract — Qt calls this when a widget is reparented or deleted."""
    layout = FlowLayout()
    layout.addWidget(QLabel("one"))
    layout.addWidget(QLabel("two"))
    assert layout.count() == 2
    assert layout.takeAt(0) is not None
    assert layout.count() == 1
    assert layout.takeAt(5) is None, "an out-of-range index must not raise"
