"""Model-dependent GUI state: TF32 gating, and that playback really is gone."""
import pytest

from diareader import config


@pytest.fixture(scope="module")
def qt_app():
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _select(dlg, key):
    dlg.precision_combo.setCurrentIndex(dlg.precision_combo.findData(key))


def test_precision_is_editable_when_no_model_is_loaded(qt_app):
    from diareader.gui.settings_dialog import SettingsDialog

    dlg = SettingsDialog(config.Settings(), model_loaded=False)
    assert dlg.precision_combo.isEnabled()


def test_precision_is_locked_while_the_model_is_loaded(qt_app):
    """It decides both the dtype the weights were built in and how they are multiplied.
    Changing it mid-session would leave one export generated partly each way."""
    from diareader.gui.settings_dialog import SettingsDialog

    dlg = SettingsDialog(config.Settings(), model_loaded=True)
    assert not dlg.precision_combo.isEnabled()


def test_locked_precision_keeps_its_stored_value_on_save(qt_app):
    from diareader.gui.settings_dialog import SettingsDialog

    settings = config.Settings(precision="tf32")
    settings.save = lambda: None  # don't touch the real settings file
    dlg = SettingsDialog(settings, model_loaded=True)
    _select(dlg, "float32")  # a stray change must not take effect

    assert dlg.apply().precision == "tf32"


def test_unlocked_precision_is_written_on_save(qt_app):
    from diareader.gui.settings_dialog import SettingsDialog

    settings = config.Settings(precision="float32")
    settings.save = lambda: None
    dlg = SettingsDialog(settings, model_loaded=False)
    _select(dlg, "bfloat16")

    assert dlg.apply().precision == "bfloat16"


def test_every_precision_choice_is_offered_and_explained(qt_app):
    from diareader.gui.settings_dialog import _PRECISION_HINTS, _PRECISION_LABELS, SettingsDialog

    dlg = SettingsDialog(config.Settings(), model_loaded=False)
    offered = {dlg.precision_combo.itemData(i) for i in range(dlg.precision_combo.count())}
    assert offered == set(config.PRECISION_CHOICES)
    assert set(_PRECISION_LABELS) == set(config.PRECISION_CHOICES)
    assert set(_PRECISION_HINTS) == set(config.PRECISION_CHOICES)


def test_selecting_a_precision_shows_its_own_evidence(qt_app):
    """Each option's cost is the half people skip, so the hint has to follow the
    selection rather than describe one fixed option."""
    from diareader.gui.settings_dialog import SettingsDialog

    dlg = SettingsDialog(config.Settings(), model_loaded=False)
    _select(dlg, "tf32")
    assert "dropped words" in dlg.precision_hint.text()
    _select(dlg, "float16")
    assert "Mismatch dtype" in dlg.precision_hint.text()


def test_precision_defaults_to_float32():
    """Everything else trades accuracy for speed, and in an autoregressive decoder
    that surfaces as dropped words rather than as slightly worse audio."""
    assert config.Settings().precision == "float32"
    assert config.resolve_precision("float32") == ("float32", False)


def test_precision_choices_map_to_dtype_and_tf32():
    assert config.resolve_precision("tf32") == ("float32", True)
    assert config.resolve_precision("bfloat16") == ("bfloat16", False)
    assert config.resolve_precision("float16") == ("float16", False)


def test_unknown_precision_falls_back_to_the_safe_one():
    assert config.resolve_precision("int4") == ("float32", False)


def _load_from(tmp_path, monkeypatch, payload):
    import json

    path = tmp_path / "settings.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(config, "SETTINGS_PATH", path)
    return config.Settings.load()


def test_a_settings_file_predating_precision_still_loads(tmp_path, monkeypatch):
    """The real hazard: Settings.load splats the JSON into the constructor, so a key
    that no longer exists as a field is a TypeError, not a warning. use_tf32 is kept
    as a field for exactly that reason."""
    settings = _load_from(tmp_path, monkeypatch, {"use_tf32": False, "speed": 0.9})
    assert settings.precision == "float32"


def test_legacy_tf32_true_becomes_the_tf32_precision(tmp_path, monkeypatch):
    settings = _load_from(tmp_path, monkeypatch, {"use_tf32": True})
    assert settings.precision == "tf32"


def test_an_explicit_precision_wins_over_a_stale_tf32_flag(tmp_path, monkeypatch):
    """Once precision is written it is authoritative — otherwise choosing plain
    float32 would silently keep tensor cores on because use_tf32 is still true on
    disk from before the migration."""
    settings = _load_from(tmp_path, monkeypatch, {"precision": "float32", "use_tf32": True})
    assert settings.precision == "float32"
    assert config.resolve_precision(settings.precision) == ("float32", False)


def test_a_garbage_precision_on_disk_falls_back(tmp_path, monkeypatch):
    settings = _load_from(tmp_path, monkeypatch, {"precision": "float8_e4m3"})
    assert settings.precision == "float32"


def test_a_retired_setting_does_not_stop_the_app_from_starting(tmp_path, monkeypatch):
    """A settings file always outlives the code that wrote it. Handing an unknown key
    to the constructor is a TypeError at startup and the app simply refuses to open —
    which is how removing `verify_with_asr` broke it for real, after a test had
    already been written about this exact hazard for a different field."""
    settings = _load_from(tmp_path, monkeypatch, {
        "verify_with_asr": True,          # retired
        "asr_model": "large-v3",          # retired
        "something_from_the_future": 42,  # never existed
        "speed": 0.9,                     # still real, must survive
    })
    assert settings.speed == 0.9
    assert settings.verify_mode == config.Settings().verify_mode
    assert not hasattr(settings, "verify_with_asr")


def test_a_retired_setting_is_not_written_back(tmp_path, monkeypatch):
    """Dropping it on load has to mean dropping it, or it lies in wait for the next
    version that happens to reuse the name."""
    import json

    path = tmp_path / "settings.json"
    monkeypatch.setattr(config, "SETTINGS_PATH", path)
    path.write_text(json.dumps({"verify_with_asr": True, "speed": 0.9}), encoding="utf-8")

    config.Settings.load().save()
    assert "verify_with_asr" not in json.loads(path.read_text(encoding="utf-8"))


def test_torch_compile_defaults_on():
    """The opposite trade to TF32: measured 1.56x faster (55.2s -> 35.3s over three
    sentences) with audio of identical length sentence for sentence, so it buys speed
    without spending accuracy."""
    assert config.Settings().use_torch_compile is True


def test_torch_compile_is_locked_while_the_model_is_loaded(qt_app):
    from diareader.gui.settings_dialog import SettingsDialog

    dlg = SettingsDialog(config.Settings(), model_loaded=True)
    assert not dlg.compile_check.isEnabled()


def test_unlocked_torch_compile_is_written_on_save(qt_app):
    from diareader.gui.settings_dialog import SettingsDialog

    settings = config.Settings(use_torch_compile=True)
    settings.save = lambda: None
    dlg = SettingsDialog(settings, model_loaded=False)
    dlg.compile_check.setChecked(False)

    assert dlg.apply().use_torch_compile is False


def test_compile_flag_actually_reaches_dia_generate():
    """It has to arrive as a generate() kwarg, not just sit in settings — the whole
    1.56x depends on dia receiving it."""
    import numpy as np

    from diareader.model_loader import LoadedModel
    from diareader.tts_engine import TtsEngine

    seen = {}

    class FakeDia:
        def generate(self, text, max_tokens=None, use_torch_compile=False,
                     audio_prompt=None, temperature=1.3, **kw):
            seen.update(text=text, use_torch_compile=use_torch_compile)
            return np.ones(4410, dtype=np.float32)

    engine = TtsEngine(LoadedModel(
        dia=FakeDia(), device="cpu",
        generation_defaults={"max_tokens": 1536, "use_torch_compile": True},
    ))
    engine.generate("Xin chào.")

    assert seen["use_torch_compile"] is True


def test_playback_module_is_gone():
    """Removed deliberately: generation runs ~2.5x slower than realtime, so live
    playback could never keep up and was only extra surface to break."""
    with pytest.raises(ImportError):
        import diareader.playback  # noqa: F401


def test_controls_expose_no_playback_buttons(qt_app):
    from diareader.gui.widgets import PlayerControls

    controls = PlayerControls()
    for gone in ("play_button", "pause_button"):
        assert not hasattr(controls, gone), f"{gone} survived the playback removal"
    assert hasattr(controls, "export_button")
    assert hasattr(controls, "stop_button"), "Stop still cancels an export"


# --- the review table is scoped to one chapter ---------------------------------------


def test_the_current_chapter_falls_back_to_the_focused_row():
    """Clicking a chapter to read it is how a user says which one they mean; making
    them select it a second time is a step for nothing."""
    from diareader.gui.main_window import MainWindow

    class _List:
        def __init__(self, selected, current):
            self._selected, self._current = selected, current

        def selected_indices(self):
            return self._selected

        def currentRow(self):
            return self._current

    class _Window:
        def __init__(self, selected, current, chapters=3):
            self.chapter_list = _List(selected, current)
            self.chapters = [object()] * chapters

    def current(selected, row, chapters=3):
        return MainWindow._current_chapter(_Window(selected, row, chapters))

    assert current([2], 0) == 2, "an explicit selection wins"
    assert current([], 1) == 1, "otherwise the focused row"
    assert current([], -1) is None, "nothing at all"
    assert current([], 5, chapters=3) is None, "a row past the end is not a chapter"
    assert current([1, 2], 0) == 1, "one chapter, the first of them"
