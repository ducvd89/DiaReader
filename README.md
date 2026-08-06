# DiaReader

Turn `.txt` / `.epub` ebooks into Vietnamese audiobook files (WAV/MP3) using the
[cosrigel/dia-finetuning-vnese](https://huggingface.co/cosrigel/dia-finetuning-vnese) checkpoint (a
Vietnamese fine-tune of Nari Labs' Dia-1.6B TTS model).

**Requires an NVIDIA GPU with CUDA. There is no CPU mode.**

## 1. Get access to the model

1. Log into Hugging Face and accept the license on the
   [model page](https://huggingface.co/cosrigel/dia-finetuning-vnese) (CC-BY-NC-SA-4.0, non-commercial).
2. Create an access token at https://huggingface.co/settings/tokens (read access is enough).
3. Copy `.env.example` to `.env` and paste your token into `HF_TOKEN=`.

## 2. Install the app, then (re)install PyTorch for your GPU

The `nari-tts` (Dia) package pulls in its own `torch` as a dependency, which will silently overwrite any
GPU build you installed first with a generic CPU-only one. So install in this order:

```bash
pip install -r requirements.txt
```

Then install/overwrite with the PyTorch build that matches your GPU. Check
https://pytorch.org/get-started/locally/ (or the nightly index for the newest builds) for the current
command.

- RTX 50-series (Blackwell, e.g. RTX 5080/5090): stable PyTorch does not yet support this GPU
  (`sm_120`). Install a **nightly** build instead, forcing it over whatever `nari-tts` installed.
  Take `torchcodec` from the same index and the same night's build as `torch` — it loads C++
  shared libraries linked against torch, so a mismatched pair fails at import:

  ```bash
  pip install --pre --force-reinstall --no-deps torch torchaudio torchcodec --index-url https://download.pytorch.org/whl/nightly/cu130
  ```

  The exact wheel tag (`cu128`, `cu130`, ...) changes over time — use whatever the PyTorch site
  currently recommends for your driver. Check what your driver supports with `nvidia-smi` (the
  "CUDA UMD Version" line) and pick the highest index that does not exceed it.

- RTX 40-series and older: the stable CUDA 12.x build is fine:

  ```bash
  pip install --force-reinstall --no-deps torch torchaudio --index-url https://download.pytorch.org/whl/cu126
  ```

Verify it worked:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

`torch.cuda.is_available()` must print `True` — the app refuses to run otherwise. If it prints `False`
after `pip install -r requirements.txt`, it's almost always this dependency-downgrade issue — rerun the
`--force-reinstall --no-deps` command above.

The first model load downloads the ~6.1 GB fine-tuned checkpoint and caches it locally — this only
happens once. The base repo contributes only its 941-byte `config.json`: every one of the 343 tensors it
would otherwise supply is overwritten by the fine-tune, so its weights are never fetched.

### FFmpeg (required if you use reference voices)

`dia` loads voice-prompt audio through `torchaudio.load`, which on Windows delegates to `torchcodec`,
which needs FFmpeg's **shared libraries** — the DLLs. The bundled `imageio-ffmpeg` ships only an
`ffmpeg.exe` and does *not* satisfy this:

```bash
winget install Gyan.FFmpeg.Shared
```

Then **restart the app** so it inherits the updated PATH. Without the DLLs, every sentence raises
inside `dia`, is turned into empty audio, retried and skipped — a multi-hour export finishes
"successfully" as a file of pure silence, with the only clue in per-sentence warnings. That is
exactly how one 54-sentence export was lost, so `pipeline.iter_book_audio` now probes the reference
audio once before any GPU work and fails with the command above. Configure no voices at all and this
is moot: the checkpoint was fine-tuned without audio prompts.

## 4. Run it

GUI:

```bash
python main.py
```

CLI:

```bash
python -m diareader.cli chapters mybook.epub
```

```bash
python -m diareader.cli export mybook.epub out.mp3 --chapters 2-5 --voice GiongNam1
```

```bash
python -m diareader.cli voices
```

## Voices

A book is narrated by one fixed voice so the audio stays consistent. Add voices under
**Settings → Voices** (a short reference WAV plus *the exact words spoken in it* — the transcript is
required, since Dia prefills the WAV into the decoder while the text conditions the encoder, and the two
must line up).

To change speaker partway through, put a marker in the text:

```
Người dẫn chuyện nói. [GiongNu2] Cô ấy trả lời.
```

The marker names a voice and applies from that point until the next marker. Markers are never spoken.
With no voices configured, generation is unconditioned — which is exactly how this checkpoint was
fine-tuned, and is a perfectly good default.

**Export** (button or dialog) asks for a chapter range and a starting voice before any GPU work begins,
and shows how many sentences that is and roughly how long it will take.

In the script editor each voice gets its own swatch, assigned by position in your voice library rather
than by hashing the name — the old hash collided, and two different voices could come out the same
colour. The hues and their order were both chosen by measurement (`scripts/validate_palette.js` in the
dataviz skill): every pair among the first five is separable under simulated colour blindness, and the
first four are additionally the largest such set that stays visible against a dark editor background.

Text a marker hands to another voice is washed in that voice's own hue, so a chapter's casting is
visible as blocks rather than as markers you have to read. The wash tracks the *voice*, not the
marker: passages read by the starting voice stay plain, including after a marker that hands narration
back to it. It is a background tint rather than coloured text because the palette is built to be read
as a fill with ink on top — violet as text measures 1.95:1 against a dark editor, which is unreadable.

**Save tags** stores the script and leaves the editor open, so casting a chapter is one continuous
pass rather than reopen-per-marker; the label beside the buttons says whether anything is unsaved.
**Reset to original** re-reads that one chapter from the book file and throws every saved edit away
(confirmed first). It deliberately re-parses from disk instead of reading the cache, because the
cache is where the edits live — the "original" in there would be the very text you asked to discard.
**A− / A+** (or Ctrl+wheel) size the text, and the size is remembered. The editor also stays
available while an export runs, so the wait can be spent tagging the next chapter.

**File → Re-read Book from Disk** is the escape hatch for a bad cached parse: it deletes the cache
and reads the book again. That also destroys every saved tag for the book, so it asks first.

A readout beside the summary tracks which sentence the cursor is in, and which voice is reading it.
It is numbered by running the pipeline's own chunker, so the number is the same one the export log
prints — when the log says a sentence came out short, move the cursor until the readout matches to
find it. (Verified against a real run: the log's failing sentences 75-78 of one chapter are exactly
the editor's 75-78.)

The editor also offers Dia's non-verbal tags — `(laughs)`, `(chuckle)`, `(sighs)` and so on — as
buttons, rendered white so a stage direction never reads as another speaker. Treat these as
experimental: they are a **base Dia** feature whose own docs call them unpredictable and advise using
them sparingly, and nothing suggests this Vietnamese fine-tune was trained on them. Test one on a short
export before putting them through a whole book.

## Moving to another machine

Three things live in different places. Only the first is required.

**1. The project folder** — copy everything *except* these, which are machine-specific or
regenerable:

```
.venv/            rebuilt by pip; contains absolute paths and platform binaries
.pytest_cache/    scratch
__pycache__/      scratch
*.wav  *.mp3      generated audio
```

**2. Voices and settings** (only if you want your cloned voices to come along) — copy the
whole folder:

```
%APPDATA%\DiaReader\        ->  same path on the new machine
```

It holds `settings.json` (HF token, voice list, speed, output folder), `voice_refs\`
(the reference WAVs), and `book_cache\` (each opened book's parsed chapters, keyed by
its file path — this is also where voice markers saved from the script editor live, so
bringing it along means a book you've already tagged doesn't need re-tagging on the new
machine). Voice entries are stored by filename, so a different Windows username on the
new machine is fine. **`settings.json` contains your Hugging Face token in plain
text** — treat the copy accordingly, or blank the field and re-enter it in Settings.

**3. The model cache** (optional, ~6.1 GB) — skip this and the first run just re-downloads.
Worth copying on a slow connection:

```
%APPDATA%\DiaReader\model_cache\    fine-tuned weights + the base config.json
```

That is the whole of it. Nothing under `%USERPROFILE%\.cache\huggingface\` is needed:
only the base repo's `config.json` (941 bytes) is ever read from it, and that is cached
alongside the weights above. If earlier runs left `models--nari-labs--Dia-1.6B` or
`models--nari-labs--Dia-1.6B-0626` there, they are dead weight now and can be deleted to
reclaim about 6.1 GB each.

### On the new machine

```bash
python -m venv .venv
```

```bash
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Then reinstall the GPU build of PyTorch over the one `nari-tts` pulled in — see section 2
above; this step is not optional, and `torch.cuda.is_available()` must print `True`.

```bash
.venv\Scripts\python.exe -m pytest tests/ -q
```

All tests run without a GPU or the model, so a green run confirms the copy is intact
before you spend time on a download. Finally:

```bash
.venv\Scripts\python.exe main.py
```

The new machine needs its own NVIDIA GPU with **at least 12 GB**. Measured during
generation, per `nvidia-smi`:

| | |
|---|---|
| CUDA context | 0.21 GB |
| Model + DAC weights | 6.29 GB |
| KV cache / activations while generating (CFG doubles the batch) | ~1.8 GB peak |
| **Driver-visible total** | **8.4 GB** |

`torch.cuda.memory_allocated()` reports 6.3 GB and is the wrong number to size a card
from: it sees neither the CUDA context nor anything the allocator is holding, and at rest
it misses the generation transients entirely.

## Notes

Everything below was measured against the real checkpoint, and several of these are traps that look
fine until you actually listen to the output.

- **The `dia` version and the base repo are pinned together, on purpose.** nari-labs' 2025-06-27 refactor
  changed how the model computes while keeping every parameter name identical, so cosrigel's weights load
  onto the newer `Dia-1.6B-0626` code with **zero missing or unexpected keys** — and then degenerate into
  looped fragments. A 68-char sentence took 1331 decode steps / 15.5s there versus 301 steps / 3.3s on the
  pinned pre-refactor commit. "All keys matched" is not evidence that a checkpoint is running correctly.
- **One sentence per generation call** (`chunking.chunk_text`). This fine-tune's training transcripts
  carry no sentence-internal punctuation, so it treats the first terminal period as end-of-utterance and
  silently drops everything after it. Grouped two-sentence chunks came out at 55-57% coverage, cutting off
  exactly at the first period, reproducibly. Do not "optimise" by batching sentences back together.
- **No `[S1]`/`[S2]` speaker tags.** Base Dia is a dialogue model, but 0 of 100 sampled training rows here
  contain any tag; feeding tagged text is out-of-distribution and makes the decoder repeat fragments.
- **float32, not float16.** All 343 tensors in the checkpoint are F32 and its `config_inference.json`
  records `training.dtype: float32`; running float16 leaves the model in a mixed-precision state.
- **The coverage check is a weak instrument, and it was measured that way.** Whisper large-v3
  transcribed 15 generated sentences from a real chapter and the syllables were compared against
  the source. Control first: on the six human voice reference clips Whisper itself scores 92.3%
  recall, which is the floor any generated audio is judged against. Results — sentences the
  pipeline **flagged** as short averaged **90.2%** real recall; sentences it **passed** averaged
  **88.9%**. The flagged ones were *better*, and both sit within ~3 points of Whisper's own error
  floor, so the audio was essentially complete either way and the retries were wasted GPU time.

  It also missed the one real failure in the sample: the bare heading `Quyển 4` produced gibberish
  (Whisper heard "Nguyễn Bốn Đông cho Mỗi Đồng", 0% recall) and scored **269% coverage** — a pass,
  because short text is never coverage-checked and the model padded it out.

  One concrete bug came out of this and is fixed: non-verbal tags were counted as characters the
  model must speak. `(clears throat)` is 15 characters, 0.74s of expected audio that is correctly
  never produced — two of the four flagged sentences were flagged for that reason alone.
  `chunking.spoken_text` now strips them before any duration estimate.

  **What ships instead is counting, not transcribing.** Vietnamese is monosyllabic, so one
  syllable is one vowel nucleus and one clear peak in the energy envelope. `verify.py` counts
  those peaks and compares against the syllable count of the text — no model, no VRAM, ~64ms per
  sentence of numpy. Calibrated on the six human reference clips it recovers the true syllable
  count to within **7.1%** (mean ratio 0.997). It is checked in **both** directions: too few
  sounds means dropped words, too many means the fragment-looping this decoder falls into, which
  is what the old one-sided length check could never see.

  Ordinary sentences are judged on the ratio, **0.85 to 1.60**. Short ones — six syllables or
  fewer — are judged on the count instead: within **±1 sound** of what the text asks for. A
  percentage means nothing at that length (one missed nucleus in three syllables is already 33%),
  and the rule it replaced rejected "2 sounds for ~1 syllable" as a 200% flood and regenerated it
  twice in a real export.

  Replayed over the 15 clips with Whisper's recall as ground truth, this disagrees on **one**:
  a sentence reading 0.83 that transcription rated fully complete. The asymmetry is why it is
  left there rather than loosened further — a false alarm costs one regeneration and nothing
  else, since the best attempt is kept either way, while a missed shortfall ships in the book.

  The parameters were swept before the threshold was tightened, not after. The original set read
  human speech at mean 0.997 but with 7.1% spread and a worst clip at 0.91: too loose a ruler to
  hang a 0.90 threshold on. The shipped set reads 1.017 with 4.4% spread and a worst clip at 0.96.
  Retries also go to **five** attempts rather than three, now that a rejection means the sounds
  really are absent — every failure seen in a real export was recovered by regenerating.

  A Whisper-based checker was built first and removed. It answered far more than the question
  needed, cost 4-6s per sentence, and loading it beside Dia crashed the app outright — CUDA OOM
  kills the process, so there is no exception to catch and no graceful degradation to write.

  The remaining limit is short text: below four syllables a ratio is too granular to threshold —
  one missed nucleus in three syllables is already 33% — so those chunks are failed only on
  silence or a flood.

  Treat the remaining warnings accordingly: a chunk at 79% against a threshold of 80% is noise,
  not evidence. The rate constant itself carries ~22% variation (measured across the reference
  clips, 2.80–5.29 syllables/sec — nearly 2x between speakers), so no threshold near 80% can be
  sharp. Counting syllables instead of characters does not help: over 238,593 sentences of the
  real book, characters per syllable is 4.458 with only 5.9% variation, so the two units are the
  same measurement scaled.

- **Adding a voice is one form, checked while you type.** It used to be three modal prompts in a
  row, with the transcript in a single-line box — awkward for text that runs 66 to 187 characters
  in the voices actually in use. The form now shows the file's rate, duration and channel count,
  whether the name can appear in a marker at all, and how the transcript's expected duration
  compares with the audio. Every one of those checks exists because the corresponding mistake was
  first noticed hours into an export, from a log line: a stereo file that made every sentence in
  that voice fail, a name with an accent that registered fine and then never matched a marker, and
  a clip holding 21.8s of speech behind a transcript worth 9.2s. A bad name blocks the form; a
  transcript that does not cover the audio only warns, since only the person who recorded it knows
  what is in it.

- **Loudness is levelled in two places, because one was not enough.** Six reference clips added by
  hand spanned **17.1 LU** of integrated loudness — the quietest was 7.17x smaller in amplitude
  than the loudest. `voice.add_voice` now normalises every reference to -20 LUFS on the way in
  (gain only, never limiting: squashing peaks would change the voice the clip exists to capture,
  so a clip that cannot reach the target without clipping is left quieter — NamMN1 lands at
  -22.7 for that reason).

  That is necessary but not sufficient, which only measurement showed. Generating the same
  sentence with every voice from references all sitting at -20 LUFS still produced narration
  spanning **10.1 LU** — NuMB2 came out at -13.9 against NuMN1's -19.1 from an identical prompt
  level. The model adds a per-voice offset of its own. So the pipeline also levels each generated
  sentence on the way out, which brings the same test to **0.0 LU**. Upward gain is capped at
  12 dB: a near-silent sentence is a generation failure, not a quiet delivery, and lifting it to
  the target would only make its noise floor audible.

- **Coverage is verified per chunk, not assumed.** The model can end a chunk early and drop text with no
  error at all — the audio simply comes out short, which is invisible across a multi-hour book.
  `pipeline._generate_chunk` compares each chunk's duration against what its text should take (~20.3
  chars/sec, measured on the fine-tune's own training clips), regenerates below 80%, and logs an error if
  it still falls short after 3 attempts. Watch the log on long exports.
- Voice conditioning is **opt-in**: training used plain text with no audio prompts, so that is the default.
  If the narrator voice drifts between sentences on your material, set a reference WAV + its transcript in
  Settings — note that Dia prefills the prompt audio into the decoder while the text conditions the
  encoder, so the prompt's transcript must be prepended to the text (`tts_engine.generate` does this).
- MP3 export uses a bundled ffmpeg binary (via `imageio-ffmpeg`) — no separate ffmpeg install needed.

### Performance, as measured

Generation runs at roughly **2.2-2.9x slower than realtime** on an RTX 5080 Laptop: about
17-20 seconds per ~150-character sentence. Run-to-run variance is large (43% between two
identical-length sentences in one sample), because decoding is stochastic.

- **VRAM: 8.4 GB during generation** (6.3 GB of that is weights). Two fixes got it there:
  the checkpoint is read into system RAM rather than straight onto the GPU — `DiaModel` is
  built on the CPU, so loading the state dict to CUDA left a second full copy of the
  weights there, 12.3 GB in total — and `tts_engine` releases cached blocks after each
  sentence, which the allocator otherwise holds at the generation peak for the whole
  session (9.65 GB versus 8.37 GB, at no measurable cost in speed).
- **`use_torch_compile=True` is now the biggest single win, and it is on by default.**
  Measured on RTX 5080 with `torch 2.14.0.dev20260803+cu130`, three sentences, seeded:
  **55.21s → 35.29s, a 1.56x speedup**, for ~28s of one-off compilation on the first
  sentence. Unlike TF32 it costs nothing in accuracy — compiled and eager produced audio
  of *identical length, sentence for sentence*, so the numerics never shifted enough to
  change what was sampled.

  This reverses an earlier finding, which is worth recording rather than quietly editing
  away. On the older `cu128` nightly the same flag aborted the process outright with
  `LLVM ERROR: Cannot select: intrinsic %llvm.nvvm.shfl.sync.bfly.i32` — Triton had no
  Blackwell backend — and this file said "don't spend time on it again". The environment
  moved; the note didn't. **A negative result about a fast-moving dependency has a shelf
  life**, and Dia's own README table (fp32: x0.9 → x1) predicted ~11%, so the measurement
  beat the documentation by a wide margin too.

  Keep the failure mode in mind on an older stack: compiling *aborts* rather than raising,
  so nothing can catch it. If the app dies at the first sentence, turn it off in Settings.

  One caveat visible in the compile logs: dynamo takes a graph break at
  `dia/state.py:100` (the KV cache update uses `Tensor.item()`), so the decoder is not
  compiled as one graph. `torch._dynamo.config.capture_scalar_outputs = True` is the
  documented lever if someone wants to chase the rest.

  **In the packaged build this needs `diareader/compile_support.py` to work at all.**
  Dynamo decides whether to trace *into* a function by matching its filename against
  the directory torch was imported from, and inside a PyInstaller bundle those two
  disagree: code objects carry a bundle-relative `torch/_dynamo/utils.py` while
  `torch.__file__` is an absolute `..._internal\torch\__init__.pyc`. Nothing matches,
  so dynamo stops recognising torch's internals as internals and traces into
  `get_current_stream`, which calls dynamo, which compiles it, which calls it again.

  It presents as `RecursionError: maximum recursion depth exceeded in
  __instancecheck__` and reads exactly like a depth problem. It is not one: raising
  the limit 10000 → 50000 changed nothing, because the recursion has no bottom. Two
  builds went into that guess before one build spent printing the traceback settled
  it in seconds. **When recursion dies at a limit that large, suspect a cycle, not a
  ceiling — and print the frames before touching the limit.**

  The selftest compiles an `nn.Module` and asserts **zero graph breaks**, not merely
  that compiling returns. Getting the path fix half-right — skipping all of torch
  rather than none of it — compiles cleanly and silently forfeits the whole 1.56x,
  and a bare `x * 2 + 1` never asks dynamo to trace into `torch.nn`, so it cannot
  tell the two apart.

  **The packaged build must also carry the toolchain the compile needs.** Inductor
  builds a small CPython extension per GPU kernel *at run time, on the user's
  machine*: it shells out to a C compiler, `#include`s `Python.h`, and links
  `pythonNNN.lib`. Triton locates all three through `sysconfig` and
  `sys.base_exec_prefix`, which inside a bundle still name the **build** machine's
  Python installation — and PyInstaller ships none of them, because nothing in an
  ordinary frozen app compiles C.

  Every one of those resolved happily on the machine that built the installer and
  was missing on the machine it was carried to, one layer at a time: first "Failed
  to find C compiler", then, once `compile_support` pointed at the bundled TinyCC,
  `include file 'Python.h' not found`. Measured against the bundled compiler
  directly — with the build machine's Python: builds; with none: fails on `Python.h`;
  with headers and import library copied into the bundle: builds. The spec now ships
  `pybuild/Include` (214 headers, 1.3 MB) and `pybuild/libs`.

  **The rule worth keeping: every input to a run-time compile must live inside the
  installed folder.** Anything resolved from the build machine works perfectly until
  the app is moved, which is the one condition local testing never covers. The
  selftest now checks each input for exactly that and fails if any of them points
  outside the app directory — a test that passes on the build machine is not
  evidence, so the check has to be about *where the path is*, not whether it worked.

  One related trap, in the error message rather than the code: it recommended
  installing MSVC, while `ensure_c_compiler` sets `CC` to the bundled compiler and
  therefore ignores MSVC entirely. Following that advice cost someone a multi-gigabyte
  install and produced the identical error. **Advice in an error message is part of
  the program and can be wrong in the same ways** — `CC` is now named as the override
  that actually takes effect.
- **TF32 (`use_tf32` in Settings) is ~1.22x faster and costs accuracy.** A 69-sentence
  chapter exported with it dropped words mid-sentence three times — heard by ear, and
  confirmed against the text at 68%, 75% and 78% coverage — where a 60-sentence fp32 run
  had none, and where fp32 retries recovered every short sentence. Off by default.

  Worth noting how nearly that was missed: an automated check that only looked for
  near-silent output reported the TF32 run as clean, because these sentences produce
  plausible-length audio and lose words inside it. Judge coverage per sentence against
  the source text, never just the presence of audio.
