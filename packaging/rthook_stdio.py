"""Runtime hook: give a windowed (console=False) build working stdout/stderr.

PyInstaller sets sys.stdout/sys.stderr to None when the app has no console, since
there is nothing to write to. Any library that assumes a file-like object at these
names crashes the first time it actually prints - never seen during development,
since a source run always has a console. Seen for real: huggingface_hub's download
progress bar (tqdm) does sys.stderr.write(...) unconditionally rather than checking
for None, so the very first model download crashed the whole load.
"""
import os
import sys

if sys.stdout is None:
    sys.stdout = open(os.devnull, "w")
if sys.stderr is None:
    sys.stderr = open(os.devnull, "w")
