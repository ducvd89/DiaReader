#!/bin/sh
# Install DiaReader's Python dependencies into the environment of the given python.
#
#   packaging/linux/install_deps.sh .venv/bin/python
#
# The order is the point, as on Windows (README section 2): nari-tts pins
# torch==2.6.0, so a plain `pip install -r requirements.txt` either fails to resolve
# (no 2.6.0 wheel for Python 3.14) or replaces the CUDA build with a CPU one. Here
# nari-tts goes in with --no-deps, and its runtime dependencies are listed by hand
# — minus gradio, which only its demo app imports.
set -eu

PY=${1:?usage: install_deps.sh PATH/TO/python}
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
INDEX=https://download.pytorch.org/whl/cu130
PIP="$PY -m pip --disable-pip-version-check --no-input"

$PIP install --upgrade pip
$PIP install -c "$HERE/constraints.txt" --index-url "$INDEX" torch torchaudio torchcodec

# Must stay the commit requirements.txt names — see the note there before changing it.
NARI=$(grep -E '^nari-tts @' "$ROOT/requirements.txt")
$PIP install --no-deps "$NARI"

# requirements.txt minus what is handled above, and minus PyInstaller, which only
# the Windows installer build uses.
REQS=$(mktemp)
trap 'rm -f "$REQS"' EXIT
grep -vE '^(nari-tts|torchcodec|pyinstaller)\b' "$ROOT/requirements.txt" > "$REQS"
$PIP install -c "$HERE/constraints.txt" --extra-index-url "$INDEX" -r "$REQS" \
    "descript-audio-codec>=1.0.0" "pydantic>=2.11.3"

# Install the optional ONNX engine last so its protobuf version wins. Both engines
# were verified to load and synthesize with that version.
$PIP install -c "$HERE/constraints.txt" -r "$ROOT/requirements-vieneu.txt"

"$PY" -c "import torch; print('torch', torch.__version__, '— CUDA available:', torch.cuda.is_available())"
