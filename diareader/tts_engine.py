"""Thin wrapper around the loaded Dia model's generate() call.

Voice cloning contract (read off dia/model.py, not guessed): ``audio_prompt`` is
**prefilled into the decoder** as DAC codes, while ``text`` conditions the encoder.
So the transcript of the audio prompt MUST be prepended to the text you actually
want spoken — otherwise the decoder has already "spoken" content the encoder never
sees, cross-attention has to re-align mid-utterance, and the model loops/repeats
fragments instead of moving on. generate() then returns only the newly generated
span (it slices from ``prefill_steps``), so the prompt audio is not in the output.
"""
from __future__ import annotations

import inspect
import logging

import numpy as np
import soundfile as sf
import torch

from . import config
from .audio_io import trim_trailing_silence
from .model_loader import LoadedModel
from .voice import VoiceReference

log = logging.getLogger(__name__)

_AUDIO_PROMPT_PARAMS = ["audio_prompt", "audio_prompt_path"]
_TERMINAL_PUNCTUATION = ".!?…"

# Budget max_tokens per chunk from its text length so a chunk can't ramble past its
# content. Rate measured directly off the fine-tune's training set rather than guessed.
_CHARS_PER_SECOND = 20.0  # median 20.3 over sampled cosrigel/vn_tts_medium_clean clips
_TOKENS_PER_SECOND = 86.0  # Dia's documented output rate (86 tokens ~= 1s of audio)
_DURATION_MARGIN = 1.5  # headroom so natural pauses/prosody aren't cut short
_MIN_TOKENS = 150

# config_inference.json reports data.audio_length = 1536, but training clips run to
# ~30s (~2550 tokens), so 1536 is not a real generation ceiling — using it as one
# truncates longer chunks. Bound by the decoder's actual position limit instead.
_MAX_TOKENS_CEILING = 3072


def _strip_tag(tagged_text: str) -> str:
    return tagged_text[len("[S1] "):] if tagged_text.startswith("[S1] ") else tagged_text


def _prompt_tokens(voice_ref: VoiceReference | None) -> int:
    """Audio-prompt prefill length in decoder steps.

    generate() counts max_tokens against a decoder step counter that *starts* at the
    prefill length, so the prompt's tokens have to be added on top of the budget for
    the new speech or the loop can exit almost immediately.
    """
    if voice_ref is None:
        return 0
    try:
        info = sf.info(str(voice_ref.wav_path))
        return int(info.frames / info.samplerate * _TOKENS_PER_SECOND)
    except Exception:
        log.warning("Could not measure voice reference length: %s", voice_ref.wav_path)
        return 0


def _estimate_max_tokens(tagged_text: str, prompt_tokens: int = 0) -> int:
    # Non-verbal tags cost characters but no speaking time, so budgeting tokens from
    # the raw length hands extra headroom to exactly the chunks least likely to need
    # it — see chunking.spoken_text.
    from . import chunking

    spoken = chunking.spoken_text(_strip_tag(tagged_text))
    seconds = max(len(spoken), 1) / _CHARS_PER_SECOND
    tokens = max(int(seconds * _TOKENS_PER_SECOND * _DURATION_MARGIN), _MIN_TOKENS)
    return min(tokens + prompt_tokens, _MAX_TOKENS_CEILING)


class TtsEngine:
    def __init__(self, loaded: LoadedModel):
        self.loaded = loaded
        try:
            self._generate_params = set(inspect.signature(loaded.dia.generate).parameters)
        except (TypeError, ValueError):
            self._generate_params = set()

    def _audio_prompt_param(self) -> str | None:
        for name in _AUDIO_PROMPT_PARAMS:
            if name in self._generate_params:
                return name
        log.warning(
            "dia.Dia.generate() exposes no audio-prompt parameter (looked for %s among %s); "
            "voice conditioning will fall back to a text-only prefix.",
            _AUDIO_PROMPT_PARAMS, sorted(self._generate_params),
        )
        return None

    def _default_kwargs(self) -> dict:
        return {
            k: v for k, v in self.loaded.generation_defaults.items()
            if k in self._generate_params
        }

    def generate(
        self, tagged_text: str, voice_ref: VoiceReference | None = None, **overrides
    ) -> np.ndarray:
        """Generate audio for one chunk. `overrides` go straight to dia's generate()
        (used by the pipeline to retry a chunk with different sampling settings)."""
        kwargs = self._default_kwargs()
        text = tagged_text
        prompt_tokens = 0

        if voice_ref is not None:
            # The prompt transcript is prepended whether or not the audio itself is
            # passed: with audio it keeps encoder text and decoder prefill aligned,
            # without it it is the only conditioning available.
            #
            # Its terminal mark MUST be stripped. This checkpoint treats the first
            # terminal period as end-of-utterance, and that period lands exactly where
            # the prompt audio stops — so the model fires EOS the instant the prompt
            # ends and the real sentence comes out as ~0.1s of nothing. Measured on a
            # 60-sentence chapter: most sentences returned 0-9% of their text with the
            # period left in.
            prompt_text = voice_ref.transcript.strip().rstrip(_TERMINAL_PUNCTUATION).rstrip()
            text = f"{prompt_text} {tagged_text}"
            param = self._audio_prompt_param()
            if param:
                kwargs[param] = str(voice_ref.wav_path)
                prompt_tokens = _prompt_tokens(voice_ref)

        if "max_tokens" in self._generate_params:
            kwargs["max_tokens"] = _estimate_max_tokens(tagged_text, prompt_tokens)

        kwargs.update({k: v for k, v in overrides.items() if k in self._generate_params})

        try:
            with torch.inference_mode():
                audio = self.loaded.dia.generate(text, **kwargs)
        except RuntimeError as e:
            # When the model emits zero audio tokens, dia's DAC decode calls conv1d on
            # an empty tensor and raises "Kernel size can't be greater than actual input
            # size". One unlucky short sentence must not kill a multi-hour export, so
            # report empty audio and let the caller retry or skip.
            log.warning("Generation produced no audio for %r: %s", tagged_text[:60], e)
            return np.zeros(0, dtype=np.float32)

        # Hand back the blocks the KV cache and activations just freed. Torch's caching
        # allocator holds them otherwise, and the driver-visible footprint stays at the
        # generation peak for the rest of the session: measured 9.65 GB without this
        # versus 8.37 GB with it, at no measurable cost in generation time (13.3s vs
        # 13.9s per sentence, inside the run-to-run noise).
        torch.cuda.empty_cache()

        if audio is None:
            return np.zeros(0, dtype=np.float32)
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        return trim_trailing_silence(audio, sample_rate=config.SAMPLE_RATE)
