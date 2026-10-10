"""VieNeu v3 Turbo adapter for DiaReader's 44.1 kHz sentence pipeline."""
from __future__ import annotations

import logging
import os
import shutil
import uuid
from pathlib import Path
from unittest.mock import patch

import numpy as np
from scipy.signal import resample_poly
from huggingface_hub import hf_hub_download

from . import chunking, config
from .audio_io import trim_trailing_silence
from .voice import PresetVoice, VoiceReference
from .vieneu_presets import canonical_name

log = logging.getLogger(__name__)
VIENEU_SAMPLE_RATE = 48_000
_CODEC_REPO = "OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano-ONNX"
_GRAPH_FILES = (
    "vieneu_prefill.onnx", "vieneu_decode_step.onnx", "vieneu_acoustic_cached.onnx",
    "vieneu_backbone_shared.data", "vieneu_v3_heads.npz", "config.json", "tokenizer.json",
)
_CODEC_FILES = (
    "moss_audio_tokenizer_decode_full.onnx", "moss_audio_tokenizer_decode_shared.data",
    "moss_audio_tokenizer_decode_step.onnx", "codec_browser_onnx_meta.json",
    "moss_audio_tokenizer_encode.onnx", "moss_audio_tokenizer_encode.data",
)


def _stage_onnx_files(
    repo: str, files: tuple[str, ...], folder: Path, token: str, subfolder: str | None = None
) -> Path:
    """Hardlink cached blobs beside their ONNX graphs for external-data loading.

    Hugging Face snapshots use symlinks to blobs in another directory. Recent ONNX
    Runtime versions reject external tensor data that resolves outside the graph's
    directory, so the SDK's direct snapshot path cannot load these models.
    """
    folder.mkdir(parents=True, exist_ok=True)
    for name in files:
        target = folder / name
        if target.is_file():
            continue
        source = Path(hf_hub_download(
            repo, name, subfolder=subfolder, token=token or None
        )).resolve()
        pending = folder / f".{name}.{uuid.uuid4().hex}.tmp"
        try:
            try:
                os.link(source, pending)
            except OSError:
                shutil.copyfile(source, pending)
            os.replace(pending, target)
        finally:
            pending.unlink(missing_ok=True)
    return folder


class VieNeuEngine:
    def __init__(self, model, preset_voice: str = config.VIENEU_DEFAULT_VOICE):
        self.model = model
        self.preset_voice = canonical_name(preset_voice) or preset_voice
        # Check this at load time, before a long render discovers a bad setting.
        names = {voice_id for _, voice_id in model.list_preset_voices()}
        if self.preset_voice not in names:
            raise ValueError(f"VieNeu preset voice {preset_voice!r} is unavailable")

    def generate(
        self, tagged_text: str,
        voice_ref: VoiceReference | PresetVoice | None = None, **overrides
    ) -> np.ndarray:
        # DiaReader's markers have already been removed by chunking. VieNeu clones
        # directly from the reference clip; it must not hear Dia's prompt transcript.
        # Parenthesized performance directions belong to Dia. Keep them out of
        # VieNeu's spoken text, including in scripts saved before switching engines.
        text = chunking.spoken_text(tagged_text.removeprefix("[S1] "))
        kwargs = {"temperature": 0.8}
        if isinstance(voice_ref, PresetVoice):
            kwargs["voice"] = voice_ref.preset
        elif voice_ref is None:
            kwargs["voice"] = self.preset_voice
        else:
            kwargs["ref_audio"] = str(voice_ref.wav_path)
        kwargs.update({k: v for k, v in overrides.items() if k == "temperature"})
        audio = self.model.infer(text, **kwargs)
        if audio is None:
            return np.zeros(0, dtype=np.float32)
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        if not len(audio):
            return audio
        # Cache, verifier, speed control, player, and exporter all use 44.1 kHz.
        audio = resample_poly(audio, 147, 160).astype(np.float32)
        return trim_trailing_silence(audio, sample_rate=config.SAMPLE_RATE)


def load_vieneu(settings: config.Settings, on_stage=None) -> VieNeuEngine:
    """Load the model through the SDK's CPU/ONNX backend on any machine."""
    try:
        from vieneu import Vieneu
    except ImportError as exc:
        raise RuntimeError("VieNeu is not installed. Install the app requirements to use it.") from exc

    message = f"Loading {config.VIENEU_MODEL_REPO} (ONNX/CPU; first run downloads model files)…"
    log.info(message)
    if on_stage:
        on_stage(message)
    # Explicit ONNX avoids the SDK auto-selecting a CUDA/PyTorch stack that may be
    # incompatible with DiaReader's newer transformers/PyTorch versions.
    local_dir = config.MODEL_CACHE_DIR / "vieneu-v3-turbo"
    graphs = _stage_onnx_files(
        config.VIENEU_MODEL_REPO, _GRAPH_FILES, local_dir / "onnx_update",
        settings.hf_token, subfolder="onnx_update",
    )
    codec = _stage_onnx_files(_CODEC_REPO, _CODEC_FILES, local_dir / "moss-codec", settings.hf_token)

    # The SDK exposes onnx_dir but not codec_dir in its public factory. Supply the
    # staged codec path during construction; no SDK source or global HF cache is edited.
    from vieneu._v3_turbo_engine.onnx_runtime_lite import OnnxV3LiteEngine

    with patch.object(OnnxV3LiteEngine, "_fetch", staticmethod(lambda *_args: codec)):
        model = Vieneu(
            backbone_repo=config.VIENEU_MODEL_REPO, backend="onnx",
            precision="fp32", onnx_dir=str(graphs),
        )
    return VieNeuEngine(model, settings.vieneu_voice)
