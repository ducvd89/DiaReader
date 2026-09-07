"""HF token, named reference voices, and device info."""
from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QDialogButtonBox, QFileDialog, QFormLayout, QHBoxLayout,
    QLabel, QLineEdit, QListWidget, QMessageBox, QPlainTextEdit, QPushButton, QTabWidget,
    QVBoxLayout, QWidget
)

from .. import config, voice, voice_backup
from .widgets import SPEED_CHOICES, speed_label


_VERIFY_HINT = (
        "Transcribes each sentence with a small Vietnamese speech model and counts "
        "what it recognised. The words themselves are never compared, so vocabulary "
        "the recogniser has never seen — a cultivation novel's jargon — still counts "
        "right. On seventeen human reference clips it reads the true syllable count "
        "with <b>2.8%</b> spread, against 6.9% for peak counting, and it is the only "
        "check here that caught a sentence which said <i>“dạ em chào ạ”</i> — peak "
        "counting scored that one a perfect 5 of 5.<br>"
    "Runs on the <b>CPU</b>, about 0.2s a sentence, and never touches the GPU.<br>"
    "Unchecked, only the crude length comparison remains, which was measured not to "
    "work: sentences it flagged averaged <b>90.2%</b> of their words present while "
    "ones it passed averaged <b>88.9%</b>."
)

# Keys match config.PRECISION_CHOICES; order is the order they are offered.
_PRECISION_LABELS = {
    "float32": "float32 — what the checkpoint is (recommended)",
    "tf32": "tf32 — float32 on tensor cores, ~1.22x",
    "bfloat16": "bfloat16 — half precision, wide range",
    "float16": "float16 — half precision, narrow range",
}

# Each option gets the evidence for it, because "faster" is the easy half of the
# story and the expensive half is what it costs in dropped words.
_PRECISION_HINTS = {
    "float32": (
        "What this checkpoint actually is: all 343 of its tensors are F32 and its own "
        "config records <code>training.dtype: float32</code>. The slowest option and the "
        "only one measured clean over a full chapter."
    ),
    "tf32": (
        "About <b>1.22x faster</b>, and it costs accuracy here: a 69-sentence chapter "
        "exported with it dropped words mid-sentence three times — audible, confirmed "
        "against the text at 68%, 75% and 78% coverage — against none in an fp32 run, "
        "and retries could not recover them. Only pick this if you will check a full "
        "chapter against the text afterwards."
    ),
    "bfloat16": (
        "Half the memory traffic, and bfloat16 keeps float32's exponent range so it will "
        "not underflow the way float16 can. <b>Untested on this checkpoint</b> — the "
        "weights are stored F32, so this casts them. Export a chapter and check it "
        "against the text before trusting it."
    ),
    "float16": (
        "<b>Known to misbehave here.</b> The checkpoint is F32 throughout, so running "
        "float16 leaves the model mixed-precision — torch logs <code>Mismatch dtype "
        "between input and weight: input dtype = Half, weight dtype = float</code> from "
        "rms_norm — and numerical instability in an autoregressive TTS decoder surfaces "
        "as repeated and looping phrases, not as slightly worse audio."
    ),
}


def _cuda_info_text() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            return f"{name} — {vram_gb:.1f} GB VRAM — CUDA available"
        return "No CUDA-capable GPU detected"
    except ImportError:
        return "PyTorch is not installed"


class AddVoiceDialog(QDialog):
    """One form for the whole voice, rather than three modals in a row.

    The transcript especially needed the change: it was a single-line QLineEdit, and
    a real transcript is a full sentence — the ones in use run 66 to 187 characters.

    Everything is checked while it is typed rather than after it is committed. The
    checks are the ones that have actually gone wrong here: a stereo file makes every
    sentence in that voice fail, a name with a space or an accent registers fine and
    then never matches a marker, and a transcript that does not cover the whole clip
    makes the model loop. All three were found after the fact, from logs, hours into
    an export.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Add a voice")
        self.setMinimumWidth(560)

        self.wav_edit = QLineEdit()
        self.wav_edit.setPlaceholderText("Pick a short WAV of the voice speaking…")
        browse = QPushButton("Browse…")
        browse.clicked.connect(self._browse)
        wav_row = QHBoxLayout()
        wav_row.setContentsMargins(0, 0, 0, 0)
        wav_row.addWidget(self.wav_edit)
        wav_row.addWidget(browse)
        wav_widget = QWidget()
        wav_widget.setLayout(wav_row)
        self.wav_status = self._status_label()

        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("NuMN1")
        self.name_status = self._status_label()

        self.transcript_edit = QPlainTextEdit()
        self.transcript_edit.setPlaceholderText("Exactly the words spoken in that file…")
        self.transcript_edit.setMinimumHeight(90)
        self.transcript_status = self._status_label()

        form = QFormLayout()
        form.addRow("Reference WAV", wav_widget)
        form.addRow("", self.wav_status)
        form.addRow("Name", self.name_edit)
        form.addRow("", self.name_status)
        form.addRow("Transcript", self.transcript_edit)
        form.addRow("", self.transcript_status)

        self.buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        self.buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Add voice")
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(self.buttons)

        self.wav_edit.textChanged.connect(self._revalidate)
        self.name_edit.textChanged.connect(self._revalidate)
        self.transcript_edit.textChanged.connect(self._revalidate)
        self._revalidate()

    @staticmethod
    def _status_label() -> QLabel:
        label = QLabel()
        label.setWordWrap(True)
        label.setStyleSheet("color: gray;")
        return label

    def _browse(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Choose a reference WAV", "", "Audio (*.wav *.flac *.mp3 *.ogg)"
        )
        if path:
            self.wav_edit.setText(path)
            if not self.name_edit.text().strip():
                self.name_edit.setText(Path(path).stem)

    # --- validation ---------------------------------------------------------

    def _check_wav(self) -> tuple[bool, str]:
        path = self.wav_edit.text().strip()
        if not path:
            return False, ""
        if not Path(path).exists():
            return False, "No file at that path."
        try:
            import soundfile as sf

            info = sf.info(path)
        except Exception as e:  # noqa: BLE001
            return False, f"Cannot read this file: {e}"
        self._duration = info.duration
        note = f"{info.samplerate} Hz · {info.duration:.2f}s"
        if info.channels > 1:
            note += f" · {info.channels} channels, converted to mono on the way in"
        if info.duration < 2:
            note += " · very short for a reference; 4-10s gives the model more to copy"
        return True, note

    def _check_name(self) -> tuple[bool, str]:
        raw = self.name_edit.text().strip()
        if not raw:
            return False, ""
        try:
            name = voice.validate_name(raw)
        except voice.InvalidVoiceName as e:
            return False, str(e)
        return True, f"Used in the text as [{name}]"

    def _check_transcript(self) -> tuple[bool, str]:
        text = self.transcript_edit.toPlainText().strip()
        if not text:
            return False, ("Required — the model is primed with the audio and "
                           "conditioned on this text, so the two must line up.")
        expected = voice.expected_duration(text)
        duration = getattr(self, "_duration", 0.0)
        ratio = voice.transcript_coverage(duration, text)
        if ratio is None:
            return True, f"about {expected:.1f}s of speech"
        note = f"about {expected:.1f}s of speech against {duration:.2f}s of audio ({ratio:.2f}x)"
        if ratio > voice._TRANSCRIPT_COVERAGE_LIMIT:
            return True, (f"{note} — the file says more than this text accounts for. "
                          "Trim the audio, or transcribe the rest, or the model will "
                          "loop.")
        if ratio < 0.6:
            return True, f"{note} — more text here than the file seems to contain."
        return True, note

    def _revalidate(self) -> None:
        ok = True
        for check, label in (
            (self._check_wav, self.wav_status),
            (self._check_name, self.name_status),
            (self._check_transcript, self.transcript_status),
        ):
            valid, message = check()
            ok &= valid
            label.setText(message)
            label.setStyleSheet("color: gray;" if valid else "color: #d03b3b;")
        self.buttons.button(QDialogButtonBox.StandardButton.Ok).setEnabled(ok)

    def values(self) -> tuple[str, str, str]:
        return (
            self.wav_edit.text().strip(),
            self.name_edit.text().strip(),
            self.transcript_edit.toPlainText().strip(),
        )


class VoicesTab(QWidget):
    """Manage named voices.

    Each voice is a short reference WAV plus its exact transcript. The transcript is not
    optional: Dia prefills the WAV into the decoder while the text conditions the
    encoder, so the two must line up or the model drifts.
    """

    def __init__(self, settings: config.Settings, parent=None):
        super().__init__(parent)
        self.settings = settings

        self.voice_list = QListWidget()
        self.transcript_label = QLabel()
        self.transcript_label.setWordWrap(True)
        self.transcript_label.setStyleSheet("color: gray;")
        self.voice_list.currentTextChanged.connect(self._show_transcript)

        add_button = QPushButton("Add from WAV…")
        add_button.clicked.connect(self._add_voice)
        default_button = QPushButton("Set as default")
        default_button.clicked.connect(self._set_default)
        remove_button = QPushButton("Remove")
        remove_button.clicked.connect(self._remove_voice)
        backup_button = QPushButton("Backup…")
        backup_button.setToolTip(
            "Save every voice and its transcript to one .vbkp file."
        )
        backup_button.clicked.connect(self._backup_voices)
        restore_button = QPushButton("Restore…")
        restore_button.setToolTip(
            "Add the voices from a .vbkp file. Existing voices are never replaced."
        )
        restore_button.clicked.connect(self._restore_voices)

        buttons = QHBoxLayout()
        for b in (add_button, default_button, remove_button):
            buttons.addWidget(b)
        library_buttons = QHBoxLayout()
        for b in (backup_button, restore_button):
            library_buttons.addWidget(b)
        library_buttons.addStretch(1)

        hint = QLabel(
            "The first voice added becomes the default. In a book's text, a marker like "
            "[GiongNu2] switches to that voice from that point on; the marker itself is "
            "never spoken. With no voices configured, the model uses its own default "
            "voice, which is how this checkpoint was trained."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color: gray;")

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Voices"))
        layout.addWidget(self.voice_list)
        layout.addWidget(self.transcript_label)
        layout.addLayout(buttons)
        layout.addLayout(library_buttons)
        layout.addWidget(hint)

        self._refresh()

    def _refresh(self) -> None:
        self.voice_list.clear()
        for name in voice.voice_names(self.settings):
            label = f"{name}  (default)" if name == self.settings.default_voice else name
            self.voice_list.addItem(label)

    def _current_name(self) -> str | None:
        item = self.voice_list.currentItem()
        if not item:
            return None
        return item.text().replace("  (default)", "")

    def _show_transcript(self) -> None:
        name = self._current_name()
        voices = voice.load_voices(self.settings)
        ref = voices.get(name) if name else None
        self.transcript_label.setText(f"Transcript: {ref.transcript}" if ref else "")

    def _add_voice(self) -> None:
        dialog = AddVoiceDialog(self)
        if not dialog.exec():
            return
        wav, name, transcript = dialog.values()
        try:
            voice.add_voice(self.settings, name, wav, transcript)
        except Exception as e:  # noqa: BLE001 — a bad file must not close Settings
            QMessageBox.critical(self, "Could not add the voice", str(e))
            return
        self._refresh()

    def _set_default(self) -> None:
        name = self._current_name()
        if name:
            self.settings.default_voice = name
            self.settings.save()
            self._refresh()

    def _remove_voice(self) -> None:
        name = self._current_name()
        if name:
            voice.remove_voice(self.settings, name)
            self._refresh()

    def _backup_voices(self) -> None:
        if not voice.voice_names(self.settings):
            QMessageBox.information(
                self, "Nothing to back up", "There are no voices configured yet."
            )
            return
        default = str(Path.home() / f"DiaReader voices{voice_backup.SUFFIX}")
        path, _ = QFileDialog.getSaveFileName(
            self, "Back up voices", default,
            f"DiaReader voice library (*{voice_backup.SUFFIX})",
        )
        if not path:
            return
        try:
            count = voice_backup.export_library(self.settings, path)
        except OSError as e:
            QMessageBox.critical(self, "Could not write the backup", str(e))
            return
        QMessageBox.information(
            self, "Voices backed up",
            f"Saved {count} voice{'s' if count != 1 else ''} to\n{path}",
        )

    def _restore_voices(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Restore voices", "",
            f"DiaReader voice library (*{voice_backup.SUFFIX})",
        )
        if not path:
            return
        try:
            result = voice_backup.import_library(self.settings, path)
        except (voice_backup.BackupUnreadable, OSError) as e:
            QMessageBox.critical(self, "Could not restore", str(e))
            return

        self._refresh()
        lines = [f"Added {result.total} voice{'s' if result.total != 1 else ''}."]
        if result.renamed:
            # Said plainly rather than buried: a marker like [NuMN1] in an already
            # tagged book will not find a voice that arrived as NuMN1_2.
            lines.append("\nThese names were already taken, so they were added as:")
            lines += [f"    {was} → {now}" for was, now in result.renamed]
        if result.skipped:
            lines.append("\nSkipped:")
            lines += [f"    {name}: {why}" for name, why in result.skipped]
        QMessageBox.information(self, "Voices restored", "\n".join(lines))


class SettingsDialog(QDialog):
    def __init__(self, settings: config.Settings, parent=None, model_loaded: bool = False):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.setMinimumWidth(560)
        self.settings = settings
        self.model_loaded = model_loaded

        self.token_edit = QLineEdit(settings.hf_token)
        self.token_edit.setEchoMode(QLineEdit.EchoMode.Password)

        self.speed_combo = QComboBox()
        for value in SPEED_CHOICES:
            self.speed_combo.addItem(speed_label(value), value)
        current = getattr(settings, "speed", 1.0) or 1.0
        index = self.speed_combo.findData(current)
        self.speed_combo.setCurrentIndex(index if index >= 0 else SPEED_CHOICES.index(1.0))

        self.output_dir_edit = QLineEdit(settings.output_dir)
        output_browse = QPushButton("Browse…")
        output_browse.clicked.connect(self._browse_output_dir)
        output_row = QWidget()
        output_layout = QHBoxLayout(output_row)
        output_layout.setContentsMargins(0, 0, 0, 0)
        output_layout.addWidget(self.output_dir_edit)
        output_layout.addWidget(output_browse)

        warning = (
            "<br><b>Unload the model to change this.</b>"
            if model_loaded else ""
        )

        self.precision_combo = QComboBox()
        for key, label in _PRECISION_LABELS.items():
            self.precision_combo.addItem(label, key)
        current = getattr(settings, "precision", config.DEFAULT_PRECISION)
        index = self.precision_combo.findData(current)
        self.precision_combo.setCurrentIndex(index if index >= 0 else 0)
        # Locked while the model is resident: it decides both the dtype the weights were
        # built in and how they are multiplied, so changing it mid-session would leave one
        # export generated partly each way — impossible to reason about when judging quality.
        self.precision_combo.setEnabled(not model_loaded)
        self.precision_combo.currentIndexChanged.connect(self._update_precision_hint)

        self.precision_hint = QLabel()
        self.precision_hint.setWordWrap(True)
        self.precision_hint.setStyleSheet("color: gray;")
        self._precision_warning = warning
        self._update_precision_hint()

        self.compile_check = QCheckBox("Compile the decoder with torch.compile (1.56x faster)")
        self.compile_check.setChecked(bool(getattr(settings, "use_torch_compile", True)))
        # Same reasoning as TF32: it changes how the resident model computes, so it is
        # fixed for the lifetime of a load.
        self.compile_check.setEnabled(not model_loaded)
        compile_hint = QLabel(
            "Measured here: 55.2s to 35.3s over three sentences, and unlike TF32 it costs "
            "nothing in accuracy — seeded, compiled and uncompiled produced audio of "
            "identical length sentence for sentence. The first sentence pays about 30 "
            "seconds of compilation, once per session. Turn this off if the app dies "
            "outright at the first sentence: on older PyTorch builds compiling aborts the "
            "process on this GPU rather than raising an error." + warning
        )
        compile_hint.setWordWrap(True)
        compile_hint.setStyleSheet("color: gray;")

        self.verify_check = QCheckBox("Check each sentence by listening to it")
        self.verify_check.setChecked(
            getattr(settings, "verify_mode", config.Settings().verify_mode) != "off"
        )
        verify_hint = QLabel(_VERIFY_HINT)
        verify_hint.setWordWrap(True)
        verify_hint.setStyleSheet("color: gray;")

        speed_hint = QLabel(
            "Slowing down lengthens the gaps between sounds and between sentences "
            "rather than drawing each syllable out, so the narrator still speaks at a "
            "natural rate — just with more room to breathe. Below about 0.75x there "
            "is not enough silence to take it all, and the speech itself is stretched "
            "for the remainder. Pitch never shifts, so the voice keeps sounding like "
            "its reference."
        )
        speed_hint.setWordWrap(True)
        speed_hint.setStyleSheet("color: gray;")

        general = QWidget()
        form = QFormLayout(general)
        form.addRow("Hugging Face token", self.token_edit)
        form.addRow("Reading speed", self.speed_combo)
        form.addRow("", speed_hint)
        form.addRow("Precision", self.precision_combo)
        form.addRow("", self.precision_hint)
        form.addRow("Performance", self.compile_check)
        form.addRow("", compile_hint)
        form.addRow("Quality check", self.verify_check)
        form.addRow("", verify_hint)
        form.addRow("Default export folder", output_row)
        form.addRow("GPU", QLabel(_cuda_info_text()))

        self.voices_tab = VoicesTab(settings)

        tabs = QTabWidget()
        tabs.addTab(general, "General")
        tabs.addTab(self.voices_tab, "Voices")

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addWidget(tabs)
        layout.addWidget(buttons)

    def _browse_output_dir(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Choose export folder", self.output_dir_edit.text())
        if path:
            self.output_dir_edit.setText(path)

    def _update_precision_hint(self) -> None:
        key = self.precision_combo.currentData() or config.DEFAULT_PRECISION
        self.precision_hint.setText(_PRECISION_HINTS[key] + self._precision_warning)

    def apply(self) -> config.Settings:
        self.settings.hf_token = self.token_edit.text().strip()
        self.settings.output_dir = self.output_dir_edit.text().strip()
        self.settings.speed = float(self.speed_combo.currentData())
        # Only writable while unlocked; otherwise keep whatever the loaded model uses.
        if self.precision_combo.isEnabled():
            self.settings.precision = self.precision_combo.currentData()
            # Keep the legacy flag consistent so anything still reading it, and any
            # older build opening this settings file, sees the same thing.
            self.settings.use_tf32 = config.resolve_precision(self.settings.precision)[1]
        if self.compile_check.isEnabled():
            self.settings.use_torch_compile = self.compile_check.isChecked()
        # Not locked to the model's lifetime. The ASR checker does hold a model, but
        # on the CPU — switching it for the next export never touches Dia's VRAM.
        self.settings.verify_mode = "asr" if self.verify_check.isChecked() else "off"
        self.settings.save()
        return self.settings
