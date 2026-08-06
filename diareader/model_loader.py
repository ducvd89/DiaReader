"""CUDA-only loader for the cosrigel Vietnamese Dia fine-tune.

The checkpoint at ``cosrigel/dia-finetuning-vnese`` ships only ``model.safetensors``
+ ``config_inference.json`` (no ``config.json``), so ``Dia.from_pretrained`` on it
fails. We fall back to building the model from the base Dia-1.6B architecture config
and loading the fine-tuned weights on top — verified against the real checkpoint:
all 343 tensors match by name and shape, F32 throughout.

IMPORTANT: "keys match" is *not* sufficient here. nari-labs' 2025-06-27 refactor kept
every parameter name while changing the computation, so these weights also load
without a single missing/unexpected key onto the newer Dia-1.6B-0626 code — and then
degenerate into looped fragments (1331 decode steps for a sentence that needs ~300).
Both the pinned ``nari-tts`` commit and ``config.BASE_MODEL_REPO`` must stay on the
pre-refactor version together.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import torch
from huggingface_hub import hf_hub_download

from . import config

log = logging.getLogger(__name__)


class CudaNotAvailableError(RuntimeError):
    """Raised when no CUDA-capable GPU is available. This app has no CPU fallback."""


class TorchCompileUnavailableError(RuntimeError):
    """Raised when torch.compile is switched on but cannot run on this machine.

    Deliberately fatal rather than a quiet fall back to eager. Falling back hides a
    broken install behind output that is merely slower, and the failure it is really
    guarding against is not slowness: compilation fails identically on every
    sentence, so an unattended export finishes as hours of silence.
    """


def require_cuda() -> torch.device:
    if not torch.cuda.is_available():
        raise CudaNotAvailableError(
            "No CUDA-capable NVIDIA GPU was detected. DiaReader requires CUDA and does "
            "not support CPU inference. Check your NVIDIA driver and that your installed "
            "PyTorch build matches your GPU/CUDA version (see README.md)."
        )
    return torch.device("cuda")


@dataclass
class LoadedModel:
    dia: Any  # the dia.model.Dia wrapper instance
    device: torch.device
    generation_defaults: dict = field(default_factory=dict)


# Generation kwargs that config_inference.json commonly names; anything else in that
# file is logged but not forwarded, since dia.Dia.generate()'s accepted kwargs may
# differ from what a training-time inference config records.
_KNOWN_GEN_KEYS = {
    "temperature", "top_p", "top_k", "cfg_scale", "cfg_filter_top_k",
    "max_tokens", "max_new_tokens", "seed", "speed_factor",
}


def _load_generation_defaults(path: Path) -> dict:
    """Read config_inference.json.

    Confirmed (against the real gated checkpoint) to be an architecture/training-config
    dump shaped like ``{"data": {"audio_length": 1536, ...}, "model": {...}, "training": {...}}``
    rather than a flat set of generate() kwargs. The one field worth carrying over is
    ``data.audio_length`` — the max audio-token length cosrigel actually trained/fine-tuned
    with — mapped to Dia.generate()'s ``max_tokens``. Its own default (3072, the base
    Dia-1.6B-0626 architecture's cap) produces ~35s of trailing silence/artifacts on short
    inputs because generation doesn't reliably stop at EOS for this checkpoint.
    """
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        log.warning("Could not parse %s as JSON; ignoring generation defaults.", path)
        return {}
    if not isinstance(raw, dict):
        return {}

    known = {k: v for k, v in raw.items() if k in _KNOWN_GEN_KEYS}

    audio_length = raw.get("data", {}).get("audio_length") if isinstance(raw.get("data"), dict) else None
    if isinstance(audio_length, int) and audio_length > 0:
        known["max_tokens"] = audio_length

    log.info("Generation defaults derived from config_inference.json: %s", known)
    return known


def _apply_state_dict(dia_wrapper: Any, state_dict: dict) -> None:
    """Find the underlying nn.Module on the Dia wrapper and load weights onto it."""
    candidate_attrs = ["model", "dia_model", "net", "module"]
    target = None
    if isinstance(dia_wrapper, torch.nn.Module):
        target = dia_wrapper
    else:
        for attr in candidate_attrs:
            obj = getattr(dia_wrapper, attr, None)
            if isinstance(obj, torch.nn.Module):
                target = obj
                break
    if target is None:
        raise RuntimeError(
            "Could not locate the underlying torch.nn.Module on the Dia wrapper "
            f"(tried attributes {candidate_attrs}). Inspect `dir(dia_wrapper)` and "
            "update _apply_state_dict() in model_loader.py."
        )

    result = target.load_state_dict(state_dict, strict=False)
    missing = list(getattr(result, "missing_keys", []))
    unexpected = list(getattr(result, "unexpected_keys", []))
    if missing or unexpected:
        log.warning(
            "Loaded cosrigel weights with mismatches — missing=%d unexpected=%d.\n"
            "First few missing: %s\nFirst few unexpected: %s",
            len(missing), len(unexpected), missing[:10], unexpected[:10],
        )
    else:
        log.info("cosrigel weights loaded cleanly onto the base Dia-1.6B architecture.")


def apply_tf32(enabled: bool) -> None:
    """Turn tensor-core matmuls on or off.

    These are global torch flags read at matmul time, not baked into the model, so this
    can be called after loading — which it must be: toggling the setting while the model
    already sits in VRAM would otherwise do nothing at all, and look like TF32 having no
    effect.

    Note torch.compile is NOT an alternative here: on sm_120 it aborts the process with
    "LLVM ERROR: Cannot select: intrinsic %llvm.nvvm.shfl.sync.bfly.i32".
    """
    torch.backends.cuda.matmul.allow_tf32 = enabled
    torch.backends.cudnn.allow_tf32 = enabled
    torch.set_float32_matmul_precision("high" if enabled else "highest")
    log.info("TF32 matmuls %s", "enabled" if enabled else "disabled")


def free_vram() -> float:
    """Drop cached CUDA allocations and report GB still held.

    Python frees the model when the last reference goes, but torch keeps a caching
    allocator that holds the memory until it is explicitly emptied — so without this,
    VRAM stays occupied even after the engine is discarded.
    """
    import gc

    gc.collect()
    if not torch.cuda.is_available():
        return 0.0
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    return torch.cuda.memory_allocated() / (1024 ** 3)


def load_model(
    settings: "config.Settings",
    *,
    compute_dtype: str | None = None,
    on_stage: "Callable[[str], None] | None" = None,
) -> LoadedModel:
    """Load the Vietnamese Dia fine-tune. Raises CudaNotAvailableError if no GPU.

    compute_dtype comes from ``settings.precision`` unless overridden here (tests and
    benchmarks pass it directly). It defaults to float32 to match what this checkpoint
    actually is: all 343 tensors in cosrigel's model.safetensors are F32, and its
    config_inference.json records ``training.dtype: float32``. Running it in
    float16 leaves the model in a mixed-precision state (torch logs
    "Mismatch dtype between input and weight: input dtype = Half, weight dtype =
    float" from rms_norm) and numerical instability in an autoregressive TTS
    decoder shows up exactly as repeated/looping phrases.
    """
    settings_dtype, tf32 = config.resolve_precision(
        getattr(settings, "precision", config.DEFAULT_PRECISION)
    )
    if compute_dtype is None:
        compute_dtype = settings_dtype
    def stage(message: str) -> None:
        # Loading takes minutes and downloads gigabytes; naming each phase is what makes
        # a stall diagnosable instead of just "it hangs".
        log.info(message)
        if on_stage:
            on_stage(message)

    stage("Checking CUDA…")
    device = require_cuda()
    config.ensure_dirs()

    # apply_tf32 logs the state itself; don't also announce it through stage().
    apply_tf32(tf32)
    stage(f"Precision: {getattr(settings, 'precision', config.DEFAULT_PRECISION)} "
          f"(compute dtype {compute_dtype})")

    from dia.config import DiaConfig
    from dia.model import Dia  # imported lazily so CUDA check runs first

    token = settings.hf_token or None

    # Only the base repo's config.json is needed — a few KB describing the architecture.
    # Dia.from_pretrained would also pull ~6.1 GB of base weights, and every one of those
    # 343 tensors is then overwritten by the fine-tune, so downloading them is pure waste.
    stage(f"Fetching architecture config from {config.BASE_MODEL_REPO}…")
    base_config_path = Path(
        hf_hub_download(
            config.BASE_MODEL_REPO,
            config.BASE_CONFIG_FILE,
            token=token,
            cache_dir=str(config.MODEL_CACHE_DIR),
        )
    )

    stage(f"Downloading fine-tuned weights {config.FINETUNE_REPO} (~6.1 GB on first run)…")
    weights_path = Path(
        hf_hub_download(
            config.FINETUNE_REPO,
            config.FINETUNE_WEIGHTS_FILE,
            token=token,
            cache_dir=str(config.MODEL_CACHE_DIR),
        )
    )
    config_path = Path(
        hf_hub_download(
            config.FINETUNE_REPO,
            config.FINETUNE_CONFIG_FILE,
            token=token,
            cache_dir=str(config.MODEL_CACHE_DIR),
        )
    )

    import safetensors.torch as st

    stage("Building the model…")
    dia_config = DiaConfig.load(str(base_config_path))
    if dia_config is None:
        raise RuntimeError(f"Could not parse the architecture config at {base_config_path}")
    dia = Dia(dia_config, compute_dtype, device)

    # Read the checkpoint into system RAM, not VRAM. DiaModel starts out on the CPU, so
    # loading the state dict straight to the GPU put a second full copy of the weights
    # there — 6.4 GB of tensors alongside the 6.4 GB model, and the dict stayed alive
    # afterwards. That is where "12.3 GB in use" for a 6.4 GB checkpoint came from.
    stage("Loading weights…")
    state_dict = st.load_file(str(weights_path), device="cpu")
    _apply_state_dict(dia, state_dict)
    del state_dict

    stage("Moving the model to VRAM…")
    dia.model.to(device)
    dia.model.eval()

    # The audio codec that turns generated tokens back into a waveform. from_pretrained
    # would have done this for us; building by hand means doing it explicitly.
    stage("Loading the DAC audio codec…")
    dia._load_dac_model()

    gen_defaults = _load_generation_defaults(config_path)

    # Fold the compile setting into the generation defaults so it reaches dia's
    # generate() through the same filtered-kwargs path as everything else.
    if getattr(settings, "use_torch_compile", False):
        # Repair what can be repaired about this machine, then check rather than
        # assume. Both repairs are no-ops from source; the check is not. Skipping it
        # is how an export becomes hours of silence: compilation fails identically on
        # every sentence, each one is retried five times, and each retry yields empty
        # audio that the pipeline dutifully writes out. See compile_support.
        from diareader import compile_support

        if compile_support.apply():
            log.info("Corrected dynamo's view of the bundled torch directory")
        tcc = compile_support.ensure_c_compiler()
        if tcc:
            log.info("Using the bundled C compiler at %s", tcc)
        headers = compile_support.use_bundled_python_headers()
        if headers:
            log.info("Compiling against the bundled CPython headers at %s", headers)

        stage("Checking torch.compile works on this machine…")
        usable, detail = compile_support.probe(torch)
        if not usable:
            log.error("torch.compile is unusable here: %s", detail)
            raise TorchCompileUnavailableError(compile_support.remedy(detail))
        gen_defaults["use_torch_compile"] = True
        stage(f"torch.compile enabled ({detail} warmup; the first sentence pays ~30s more).")

    try:
        used = torch.cuda.memory_allocated() / (1024 ** 3)
        stage(f"Model ready ({used:.1f} GB VRAM in use)")
    except Exception:
        stage("Model ready")

    return LoadedModel(dia=dia, device=device, generation_defaults=gen_defaults)
