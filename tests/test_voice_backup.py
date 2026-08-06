"""Round-tripping the voice library through a .vbkp file.

Everything here redirects config.APP_DIR/VOICE_REF_DIR/SETTINGS_PATH into tmp_path,
so no test can touch the real library.
"""
import json
import zipfile

import numpy as np
import pytest
import soundfile as sf

from diareader import config, voice, voice_backup


@pytest.fixture
def library(tmp_path, monkeypatch):
    """An isolated voice library, plus a helper that adds voices to it."""
    app_dir = tmp_path / "app"
    monkeypatch.setattr(config, "APP_DIR", app_dir)
    monkeypatch.setattr(config, "VOICE_REF_DIR", app_dir / "voice_refs")
    monkeypatch.setattr(config, "SETTINGS_PATH", app_dir / "settings.json")
    monkeypatch.setattr(config, "MODEL_CACHE_DIR", app_dir / "model_cache")
    monkeypatch.setattr(config, "BOOK_CACHE_DIR", app_dir / "book_cache")
    config.ensure_dirs()

    settings = config.Settings()

    def add(name: str, transcript: str, seconds: float = 1.0):
        src = tmp_path / f"src_{name}.wav"
        # Not silence: add_voice normalises loudness on the way in, and a silent
        # clip has none to measure.
        t = np.linspace(0, seconds, int(44100 * seconds), endpoint=False)
        sf.write(str(src), (0.2 * np.sin(2 * np.pi * 220 * t)).astype("float32"), 44100)
        return voice.add_voice(settings, name, src, transcript)

    return settings, add


def test_backup_then_restore_into_an_empty_library(library, tmp_path):
    settings, add = library
    add("NuMN1", "xin chào các bạn")
    add("NamMN1", "hôm nay trời đẹp")

    dest = tmp_path / "voices.vbkp"
    assert voice_backup.export_library(settings, dest) == 2

    empty = config.Settings()
    result = voice_backup.import_library(empty, dest)

    assert sorted(result.added) == ["NamMN1", "NuMN1"]
    assert not result.renamed and not result.skipped
    restored = voice.load_voices(empty)
    # The transcript is half the voice: Dia conditions the encoder with it while the
    # WAV is prefilled into the decoder, so a clip restored without it is unusable.
    assert restored["NuMN1"].transcript == "xin chào các bạn"
    assert restored["NuMN1"].wav_path.is_file()


def test_restoring_over_an_existing_name_adds_rather_than_replaces(library, tmp_path):
    settings, add = library
    original = add("NuMN1", "bản gốc")
    dest = tmp_path / "voices.vbkp"
    voice_backup.export_library(settings, dest)

    # Same name, different content, restored back into the same library.
    add("NuMN1", "bản gốc")  # unchanged on disk
    result = voice_backup.import_library(settings, dest)

    assert result.renamed == [("NuMN1", "NuMN1_2")]
    voices = voice.load_voices(settings)
    assert voices["NuMN1"].transcript == "bản gốc", "the existing voice must survive"
    assert "NuMN1_2" in voices
    assert original.wav_path.is_file()


def test_the_extension_is_applied_even_if_the_dialog_did_not(library, tmp_path):
    settings, add = library
    add("NuMN1", "xin chào")
    written = tmp_path / "mybackup"
    voice_backup.export_library(settings, written)
    assert (tmp_path / "mybackup.vbkp").is_file()


def test_a_backup_is_a_plain_zip_anyone_can_open(library, tmp_path):
    settings, add = library
    add("NuMN1", "xin chào")
    dest = tmp_path / "voices.vbkp"
    voice_backup.export_library(settings, dest)

    with zipfile.ZipFile(dest) as zf:
        manifest = json.loads(zf.read("manifest.json").decode("utf-8"))
        assert manifest["format"] == "diareader-voices"
        assert manifest["voices"][0]["transcript"] == "xin chào"
        assert any(m.startswith("voices/") for m in zf.namelist())


def test_a_backup_never_carries_the_hugging_face_token(library, tmp_path):
    """An installer ships one of these, and settings.json holds the token in clear."""
    settings, add = library
    settings.hf_token = "hf_thismustnotescape"
    add("NuMN1", "xin chào")
    dest = tmp_path / "voices.vbkp"
    voice_backup.export_library(settings, dest)
    assert b"hf_thismustnotescape" not in dest.read_bytes()


def test_a_file_that_is_not_a_backup_is_refused_by_name(tmp_path):
    junk = tmp_path / "notes.vbkp"
    junk.write_text("just some text")
    with pytest.raises(voice_backup.BackupUnreadable) as e:
        voice_backup.import_library(config.Settings(), junk)
    assert "notes.vbkp" in str(e.value)


def test_a_zip_without_a_manifest_is_refused(tmp_path):
    bogus = tmp_path / "photos.vbkp"
    with zipfile.ZipFile(bogus, "w") as zf:
        zf.writestr("hello.txt", "hi")
    with pytest.raises(voice_backup.BackupUnreadable):
        voice_backup.import_library(config.Settings(), bogus)


def test_a_newer_format_is_refused_rather_than_half_read(tmp_path):
    future = tmp_path / "future.vbkp"
    with zipfile.ZipFile(future, "w") as zf:
        zf.writestr("manifest.json", json.dumps(
            {"format": "diareader-voices", "version": 99, "voices": []}
        ))
    with pytest.raises(voice_backup.BackupUnreadable) as e:
        voice_backup.import_library(config.Settings(), future)
    assert "newer version" in str(e.value)


def test_a_traversal_path_in_the_archive_cannot_write_outside(library, tmp_path):
    """Zip members name their own path, and a hostile one names somebody else's."""
    settings, add = library
    add("NuMN1", "xin chào")
    good = tmp_path / "good.vbkp"
    voice_backup.export_library(settings, good)

    evil = tmp_path / "evil.vbkp"
    with zipfile.ZipFile(good) as src, zipfile.ZipFile(evil, "w") as dst:
        wav = next(m for m in src.namelist() if m.startswith("voices/"))
        dst.writestr("../../escaped.wav", src.read(wav))
        dst.writestr("manifest.json", json.dumps({
            "format": "diareader-voices", "version": 1,
            "voices": [{"name": "Evil", "file": "../../escaped.wav", "transcript": "x"}],
        }))

    voice_backup.import_library(config.Settings(), evil)
    assert not (tmp_path.parent / "escaped.wav").exists()
    assert not (tmp_path / "escaped.wav").exists()


def test_an_unusable_name_is_skipped_with_the_reason(library, tmp_path):
    """Marker names are [A-Za-z0-9_-]; anything else could never be referenced."""
    bad = tmp_path / "bad.vbkp"
    with zipfile.ZipFile(bad, "w") as zf:
        zf.writestr("voices/x.wav", b"RIFF")
        zf.writestr("manifest.json", json.dumps({
            "format": "diareader-voices", "version": 1,
            "voices": [{"name": "Nữ (MN1)", "file": "voices/x.wav", "transcript": "x"}],
        }))
    result = voice_backup.import_library(config.Settings(), bad)
    assert result.total == 0
    assert result.skipped and "marker" in result.skipped[0][1]


def test_one_broken_voice_does_not_lose_the_others(library, tmp_path):
    """A restore is often someone's only copy; it must not stop at the first fault."""
    settings, add = library
    add("NuMN1", "xin chào")
    good = tmp_path / "voices.vbkp"
    voice_backup.export_library(settings, good)

    # Same archive, plus one entry whose audio was never written into it.
    patched = tmp_path / "patched.vbkp"
    with zipfile.ZipFile(good) as src, zipfile.ZipFile(patched, "w") as dst:
        manifest = json.loads(src.read("manifest.json").decode("utf-8"))
        manifest["voices"].append(
            {"name": "Missing", "file": "voices/nope.wav", "transcript": "x"}
        )
        for member in src.namelist():
            if member != "manifest.json":
                dst.writestr(member, src.read(member))
        dst.writestr("manifest.json", json.dumps(manifest))

    result = voice_backup.import_library(config.Settings(), patched)
    assert result.added == ["NuMN1"], "the healthy voice must still arrive"
    assert [n for n, _ in result.skipped] == ["Missing"]
