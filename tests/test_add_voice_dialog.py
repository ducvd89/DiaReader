"""The add-voice form.

Everything here used to be three modal prompts in a row with a single-line box for
the transcript, and every check was done after the fact — from logs, hours into an
export. These tests pin the checks to the moment of typing.
"""
import numpy as np
import pytest
import soundfile as sf

from diareader import config


@pytest.fixture(scope="module")
def qt_app():
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _wav(path, seconds=5.0, channels=1, rate=44100):
    t = np.linspace(0, seconds, int(rate * seconds), endpoint=False)
    tone = (np.sin(2 * np.pi * 220 * t) * 0.3).astype(np.float32)
    data = tone if channels == 1 else np.stack([tone] * channels, axis=1)
    sf.write(str(path), data, rate)
    return str(path)


# A transcript whose expected duration matches a 5s clip.
GOOD_TRANSCRIPT = "Những giây phút cuối cùng trước thềm năm mới luôn thật nhiều cảm xúc"


def _dialog(qt_app):
    from diareader.gui.settings_dialog import AddVoiceDialog

    return AddVoiceDialog()


def _ok_enabled(dlg):
    from PySide6.QtWidgets import QDialogButtonBox

    return dlg.buttons.button(QDialogButtonBox.StandardButton.Ok).isEnabled()


def test_the_form_starts_disabled(qt_app):
    assert not _ok_enabled(_dialog(qt_app))


def test_a_complete_form_can_be_submitted(qt_app, tmp_path):
    dlg = _dialog(qt_app)
    dlg.wav_edit.setText(_wav(tmp_path / "v.wav"))
    dlg.name_edit.setText("NuMN1")
    dlg.transcript_edit.setPlainText(GOOD_TRANSCRIPT)

    assert _ok_enabled(dlg)
    assert dlg.values() == (_wav(tmp_path / "v.wav"), "NuMN1", GOOD_TRANSCRIPT)


def test_the_transcript_field_takes_more_than_one_line(qt_app):
    """A real transcript is a whole sentence — the ones in use run 66 to 187
    characters — and the old single-line box made that miserable to enter."""
    from PySide6.QtWidgets import QPlainTextEdit

    dlg = _dialog(qt_app)
    assert isinstance(dlg.transcript_edit, QPlainTextEdit)
    dlg.transcript_edit.setPlainText("Dòng một.\nDòng hai.")
    assert "\n" in dlg.values()[2]


def test_a_missing_transcript_blocks_submission(qt_app, tmp_path):
    dlg = _dialog(qt_app)
    dlg.wav_edit.setText(_wav(tmp_path / "v.wav"))
    dlg.name_edit.setText("NuMN1")
    assert not _ok_enabled(dlg)
    assert "Required" in dlg.transcript_status.text()


def test_a_name_that_could_never_match_a_marker_is_refused(qt_app, tmp_path):
    """It used to register happily and then be silently ignored, because markers
    only match letters, digits, underscore and hyphen."""
    dlg = _dialog(qt_app)
    dlg.wav_edit.setText(_wav(tmp_path / "v.wav"))
    dlg.transcript_edit.setPlainText(GOOD_TRANSCRIPT)

    dlg.name_edit.setText("Nữ (MN1)")
    assert not _ok_enabled(dlg)

    dlg.name_edit.setText("NuMN1")
    assert _ok_enabled(dlg)
    assert "[NuMN1]" in dlg.name_status.text()


def test_a_missing_file_is_reported_before_submitting(qt_app):
    dlg = _dialog(qt_app)
    dlg.wav_edit.setText(r"C:\nope\missing.wav")
    assert not _ok_enabled(dlg)
    assert "No file" in dlg.wav_status.text()


def test_a_stereo_file_is_accepted_and_the_conversion_is_stated(qt_app, tmp_path):
    """Stereo is handled, not rejected — but silently converting it is how a
    surprise gets built, so the form says so."""
    dlg = _dialog(qt_app)
    dlg.wav_edit.setText(_wav(tmp_path / "s.wav", channels=2))
    dlg.name_edit.setText("Stereo1")
    dlg.transcript_edit.setPlainText(GOOD_TRANSCRIPT)

    assert _ok_enabled(dlg)
    assert "mono" in dlg.wav_status.text()


def test_audio_longer_than_its_transcript_is_flagged_while_typing(qt_app, tmp_path):
    """The NamMN1 case: 21.8s of speech behind a transcript worth 9.2s. It was found
    from a log line after the voice had already been used."""
    dlg = _dialog(qt_app)
    dlg.wav_edit.setText(_wav(tmp_path / "long.wav", seconds=22.0))
    dlg.name_edit.setText("TooLong")
    dlg.transcript_edit.setPlainText(GOOD_TRANSCRIPT)

    assert "loop" in dlg.transcript_status.text()
    # a warning, not a refusal: only the person who recorded it knows what is in it
    assert _ok_enabled(dlg)


def test_a_matching_pair_is_not_flagged(qt_app, tmp_path):
    dlg = _dialog(qt_app)
    dlg.wav_edit.setText(_wav(tmp_path / "v.wav", seconds=4.0))
    dlg.name_edit.setText("Fine")
    dlg.transcript_edit.setPlainText(GOOD_TRANSCRIPT)

    assert "loop" not in dlg.transcript_status.text()
    assert _ok_enabled(dlg)
