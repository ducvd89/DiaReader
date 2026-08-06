"""App-wide paths, model repo ids, and persisted user settings."""
from __future__ import annotations

import json
import logging
import os
import sys
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

log = logging.getLogger(__name__)

APP_DIR = Path(os.environ.get("APPDATA", Path.home())) / "DiaReader"
SETTINGS_PATH = APP_DIR / "settings.json"
MODEL_CACHE_DIR = APP_DIR / "model_cache"
VOICE_REF_DIR = APP_DIR / "voice_refs"
# Parsed chapters (title + text) per book, keyed by source file path. Reused so
# reopening a book skips re-parsing it, and doubles as storage for script-editor
# edits (voice tags) so re-exporting later doesn't require re-tagging.
BOOK_CACHE_DIR = APP_DIR / "book_cache"

# The pre-refactor base the checkpoint was actually fine-tuned from (its model card
# says base_model: nari-labs/Dia-1.6B). Must stay in step with the pinned `nari-tts`
# commit in requirements.txt — see the note there before changing either.
BASE_MODEL_REPO = "nari-labs/Dia-1.6B"
FINETUNE_REPO = "cosrigel/dia-finetuning-vnese"
FINETUNE_WEIGHTS_FILE = "model.safetensors"
FINETUNE_CONFIG_FILE = "config_inference.json"
BASE_CONFIG_FILE = "config.json"

SAMPLE_RATE = 44100


DEFAULT_VOICE_NAME = "GiongMacDinh"

# How many books the File menu offers to reopen.
RECENT_BOOKS_LIMIT = 3


# Precision options, in the order they are offered. Each entry is
# (compute_dtype passed to Dia, whether float32 matmuls use tensor cores).
#
# Only float32 is known-good on this checkpoint; the rest are offered because the
# speed is real, but every one of them has a documented or expected failure mode.
PRECISION_CHOICES: dict[str, tuple[str, bool]] = {
    "float32": ("float32", False),
    "tf32": ("float32", True),
    "bfloat16": ("bfloat16", False),
    "float16": ("float16", False),
}
DEFAULT_PRECISION = "float32"


def resolve_precision(name: str) -> tuple[str, bool]:
    """(compute_dtype, tf32) for a precision name, falling back to the default."""
    return PRECISION_CHOICES.get(name, PRECISION_CHOICES[DEFAULT_PRECISION])


@dataclass
class Settings:
    hf_token: str = ""
    # Legacy single-voice fields, kept so old settings files still load; migrated into
    # `voices` on first load.
    voice_ref_wav: str = ""
    voice_ref_transcript: str = ""
    # name -> {"wav": path, "transcript": text}. Named voices let a book switch speaker
    # mid-text via markers like [GiongNu2] while staying fixed everywhere else.
    voices: dict = field(default_factory=dict)
    default_voice: str = ""
    # Playback/export tempo, applied without shifting pitch.
    #
    # 0.9 rather than 1.0 by design: measured over a 60-sentence chapter, the model
    # reads at 19.8 chars/s while the Latradio reference clip runs at 17.6 chars/s — a
    # 12% rush that is audible. 0.9 brings generated narration back to the pace of the
    # voice it is imitating. Re-measure before changing this if the voice changes.
    speed: float = 0.9
    # How the model computes. One of PRECISION_CHOICES.
    #
    # "float32" is the default because it is what the checkpoint *is*: all 343 tensors
    # in cosrigel's model.safetensors are F32 and its config_inference.json records
    # training.dtype: float32. The other three all trade accuracy for speed, and in an
    # autoregressive TTS decoder inaccuracy does not show up as slightly worse audio —
    # it shows up as dropped words and looping phrases. See PRECISION_CHOICES for what
    # each one measured.
    precision: str = "float32"
    # Legacy boolean, folded into `precision` on load — "tf32" is now one of the
    # choices rather than a separate switch, since it only ever meant "float32, but
    # with the matmuls run on tensor cores".
    use_tf32: bool = False
    # Compile dia's per-token decoder step. Measured on this machine (RTX 5080,
    # torch 2.14.0.dev20260803+cu130): 55.21s -> 35.29s over three sentences, a
    # 1.56x speedup, for ~28s of one-off compilation on the first sentence.
    #
    # Unlike use_tf32 this costs nothing in accuracy: seeded against the same three
    # sentences, compiled and eager produced audio of *identical* length, sentence
    # for sentence — the numerics did not shift enough to change what was sampled.
    #
    # It used to abort the process outright on this GPU ("LLVM ERROR: Cannot select:
    # intrinsic %llvm.nvvm.shfl.sync.bfly.i32" — Triton had no Blackwell/sm_120
    # backend). That is fixed in the current nightly. Note the failure mode if you
    # run this on an older stack: a hard abort, not an exception, so nothing can
    # catch it — turn this off in Settings if the app dies at the first sentence.
    use_torch_compile: bool = True
    # How a generated sentence is judged before the pipeline accepts it.
    #
    #   "syllables" (default) — count vowel nuclei in the waveform and compare with
    #       the syllable count of the text. No model, no VRAM, a millisecond of
    #       numpy. Recovers the true count to within 7.1% on human speech, and is
    #       checked both ways: too few means dropped words, too many means looping.
    #   "duration" — the original: compare audio length against what the text should
    #       take. Kept because it is what shipped, but measured not to work — it
    #       flagged sentences averaging 90.2% of their words present while passing
    #       ones averaging 88.9%, and passed a heading rendered as pure noise at 269%.
    #
    # A Whisper-based mode was built and removed: transcription answers far more than
    # "did it say everything", and loading it beside Dia crashed the app outright.
    verify_mode: str = "syllables"
    # Level every generated sentence to the same loudness. On by default because the
    # model's output level varies by voice well beyond what the reference clips
    # explain: with all six references normalised to -20 LUFS, the narration they
    # produced still spanned 10.1 LU. Without this, a book that switches voices makes
    # the listener reach for the volume.
    normalize_output: bool = True
    # Script editor text size, remembered so a zoom set for comfortable reading is
    # not undone every time the editor is reopened.
    script_font_pt: int = 11
    output_dir: str = str(Path.home() / "Music" / "DiaReader")
    # Books opened before, newest first, capped at RECENT_BOOKS_LIMIT.
    recent_books: list = field(default_factory=list)
    # Where the Open dialog starts. Books live wherever the user keeps them, which is
    # never the application folder Qt would otherwise default to.
    last_open_dir: str = ""

    def remember_book(self, path: str) -> None:
        """Move `path` to the front of the recent list and record its folder."""
        resolved = str(Path(path))
        recent = [p for p in (self.recent_books or []) if str(p) != resolved]
        recent.insert(0, resolved)
        self.recent_books = recent[:RECENT_BOOKS_LIMIT]
        self.last_open_dir = str(Path(resolved).parent)

    def existing_recent_books(self) -> list[str]:
        """Recent books that are still on disk.

        Filtered on read rather than pruned on write: a book on a drive that is not
        plugged in right now has not stopped being recent.
        """
        return [p for p in (self.recent_books or []) if Path(p).is_file()]

    @classmethod
    def load(cls) -> "Settings":
        data: dict = {}
        if SETTINGS_PATH.exists():
            data = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
            # Drop keys this version no longer has, rather than handing them to the
            # constructor. A settings file always outlives the code that wrote it:
            # every renamed or retired option would otherwise be a TypeError on
            # startup with the app refusing to open at all — which is exactly how
            # `verify_with_asr` broke it once the field behind it was removed.
            known = {f.name for f in fields(cls)}
            unknown = sorted(set(data) - known)
            if unknown:
                log.info("Ignoring settings this version no longer uses: %s",
                         ", ".join(unknown))
                data = {k: v for k, v in data.items() if k in known}
            settings = cls(**{**asdict(cls()), **data})
        else:
            settings = cls()
        if not settings.hf_token:
            settings.hf_token = os.environ.get("HF_TOKEN", "")
        settings._migrate_legacy_voice()
        settings._migrate_legacy_tf32(wrote_precision="precision" in data)
        return settings

    def _migrate_legacy_tf32(self, wrote_precision: bool) -> None:
        """Carry a pre-`precision` settings file over.

        Only applies when the file predates the field: once `precision` is written,
        it is authoritative, or picking plain float32 with a stale `use_tf32: true`
        still on disk would silently keep tensor cores on.
        """
        if not wrote_precision and self.use_tf32:
            self.precision = "tf32"
        if self.precision not in PRECISION_CHOICES:
            self.precision = DEFAULT_PRECISION

    def _migrate_legacy_voice(self) -> None:
        if self.voice_ref_wav and self.voice_ref_transcript and not self.voices:
            self.voices = {
                DEFAULT_VOICE_NAME: {
                    "wav": self.voice_ref_wav,
                    "transcript": self.voice_ref_transcript,
                }
            }
            self.default_voice = DEFAULT_VOICE_NAME

    def save(self) -> None:
        APP_DIR.mkdir(parents=True, exist_ok=True)
        SETTINGS_PATH.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")


def asset_path(name: str) -> Path:
    """Locate a file shipped with the app, packaged or not.

    PyInstaller unpacks data files under sys._MEIPASS; from source they sit in
    packaging/assets. The icon is needed at *run* time as well as build time — the
    `icon=` given to PyInstaller only decorates the .exe, and does nothing for the
    icon Qt draws in the window's title bar and the taskbar button.
    """
    base = getattr(sys, "_MEIPASS", None)
    if base is not None:
        return Path(base) / name
    return Path(__file__).resolve().parent.parent / "packaging" / "assets" / name


def ensure_dirs() -> None:
    for d in (APP_DIR, MODEL_CACHE_DIR, VOICE_REF_DIR, BOOK_CACHE_DIR):
        d.mkdir(parents=True, exist_ok=True)
