"""Check a generated sentence by listening to it, not by looking at its energy.

Same question the app has always asked — did the model say all of it — and the same
arithmetic, sounds heard over syllables the text calls for. What changed is the
instrument: an acoustic model counts what it recognised instead of counting peaks in an
energy envelope. verify.py now only supplies the text side of that comparison.

Why it replaced peak counting, measured on clips whose truth was known by ear:

  short line, 5 syllables   15 clips, every one counted correctly: 5 for the fourteen
                            that said five syllables, and 6 for the one that really
                            did render an extra word ("Nam dạ em chào anh Minh").
                            Peak counting read 3, 4 or 5 across the same clips and
                            overlapped the failures outright — one that said "dạ em
                            chào ạ", two words wrong, scored a perfect 5 by peaks.
  long sentence, 30         complete takes count 30 or 31 here (1.00-1.03) against
                            0.80-1.07 by peaks, which failed one of them outright.
  17 human reference clips  CV 2.8% here, 6.9% by peaks.

Word errors do not matter and are not looked at. "pháp luật" for "pháp lực" is still
two syllables, and Vietnamese is monosyllabic, so counting whitespace-separated tokens
in the transcript counts syllables. That also makes the check robust on vocabulary the
ASR has never seen — a cultivation novel's jargon transcribes wrongly and still counts
right.

It runs on the CPU, and that is not incidental. Whisper was tried as this app's checker
once and removed because loading it beside Dia took the process down with a CUDA OOM
that kills rather than raises. This model is 95M parameters, transcribes a five-second
sentence in 0.18s on the CPU, and never touches the GPU.
"""
from __future__ import annotations

import logging
import os

import numpy as np

from . import config, foreign_words, verify

log = logging.getLogger(__name__)

#: A Vietnamese wav2vec2 CTC model. Chosen over PhoWhisper for this job because CTC
#: decodes in one forward pass — no autoregressive loop — which is what makes it cheap
#: enough on the CPU to run on every sentence.
MODEL_REPO = "nguyenvulebinh/wav2vec2-base-vietnamese-250h"
ASR_SAMPLE_RATE = 16000

# Tighter than the 0.85-1.20 the peak counter needed, and it is the instrument that
# earns it:
# on the seventeen human reference clips this counter reads mean 1.000 with 2.8%
# spread, so 0.90 and 1.10 sit 3.6 standard deviations out and admit all seventeen.
# 0.95-1.05 was measured too: it rejects two of them (0.926 and 1.074, both ASR slips
# on real speech), which is the ~12% false-alarm rate this whole change existed to
# remove. Meanwhile the worst truncated generation measured 0.60, so the margin below
# is 0.30 wide — with peak counting there was no margin at all.
RATIO_MIN = 0.90
RATIO_MAX = 1.10

# Below this a clip is a failed generation rather than something to transcribe: the
# model's convolutional front end needs more samples than its kernel, and kimcuc's
# worst take was 0.01s.
MIN_AUDIO_S = 0.2


class AsrCounter:
    """The pipeline's checker: check(text, audio) -> (accepted, score, why).

    Short text gets no special tolerance. Peak counting needed it
    because it could not resolve five syllables in a one-second greeting; this counted
    all fifteen short clips exactly right, including the one that rendered a spurious
    extra word and which the band correctly rejects. On a five-syllable line ±1 would
    be admitting a fifth of the sentence missing, and there is no longer any
    measurement error to excuse it.
    """

    name = "asr"

    def __init__(self) -> None:
        self._model = None
        self._processor = None

    def _load(self):
        if self._model is not None:
            return

        # Hidden from CUDA rather than merely moved to the CPU. transformers and torch
        # both initialise CUDA context on import paths that are hard to predict, and
        # the failure this guards against killed the process outright.
        previous = os.environ.get("CUDA_VISIBLE_DEVICES")
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        try:
            from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

            log.info("Loading the speech checker %s (CPU)…", MODEL_REPO)
            # Deliberately the default Hugging Face cache rather than
            # config.MODEL_CACHE_DIR, which is where every other model in this app
            # lives. Symlinks created under %APPDATA%\DiaReader cannot be opened by
            # anything on this machine (see hf_cache), and hf_hub_download can be
            # repaired for that afterwards while from_pretrained resolves its files
            # internally and cannot. Pointing it there fails with "Can't load feature
            # extractor", which the caller then treats as "checker unavailable" —
            # quality checking silently off for the whole book.
            self._processor = Wav2Vec2Processor.from_pretrained(MODEL_REPO)
            self._model = Wav2Vec2ForCTC.from_pretrained(MODEL_REPO).eval()
        finally:
            if previous is None:
                os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            else:
                os.environ["CUDA_VISIBLE_DEVICES"] = previous

    def transcribe(self, audio: np.ndarray, sample_rate: int) -> str:
        """What the recogniser heard, lowercased. Empty when there is nothing to hear."""
        import torch

        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        if audio.size / sample_rate < MIN_AUDIO_S:
            return ""

        self._load()
        if sample_rate != ASR_SAMPLE_RATE:
            import librosa

            audio = librosa.resample(
                audio, orig_sr=sample_rate, target_sr=ASR_SAMPLE_RATE
            )
        inputs = self._processor(
            audio, sampling_rate=ASR_SAMPLE_RATE, return_tensors="pt", padding=True
        )
        with torch.inference_mode():
            logits = self._model(inputs.input_values).logits
        predicted = torch.argmax(logits, dim=-1)
        return self._processor.batch_decode(predicted)[0].strip().lower()

    def check(self, source_text: str, audio: np.ndarray,
              sample_rate: int = config.SAMPLE_RATE) -> tuple[bool, float, str]:
        expected = len(verify.syllables(source_text))
        if expected == 0:
            return True, 1.0, "nothing to say"

        try:
            transcript = self.transcribe(audio, sample_rate)
        except Exception as e:  # noqa: BLE001
            # A checker that raises would abort an export over a sentence that may
            # well be fine. Report it and let the length check downstream decide.
            log.warning("Speech check failed; accepting the sentence: %s", e)
            return True, 1.0, f"checker unavailable ({type(e).__name__})"

        heard = len(transcript.split())
        score = heard / expected

        # A foreign word is one token and several spoken syllables — "SmartBanking"
        # counts as one and is read as four — so the ceiling, and only the ceiling,
        # makes room for it. The error runs one way: the text under-counts what will
        # be said, never over-counts, so the floor stays exactly where it was.
        extra = foreign_words.extra_syllables_allowed(source_text)
        ceiling = expected * RATIO_MAX + extra
        ok = expected * RATIO_MIN <= heard <= ceiling

        # The transcript is the point of the detail line: "36% of the sounds" says a
        # sentence is wrong, "heard 'dạ em chào'" says how.
        detail = f"heard {transcript!r} — {heard} of {expected} syllables"
        if extra:
            detail += (f" (+{extra} allowed for "
                       f"{', '.join(foreign_words.foreign_words(source_text))})")
        return ok, score, detail

    def release(self) -> None:
        """Drop the model. It is CPU-only, so this frees RAM rather than VRAM."""
        self._model = None
        self._processor = None
