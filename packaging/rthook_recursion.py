"""Runtime hook: give TorchDynamo the stack depth it needs inside a frozen app.

torch.compile traces Python bytecode by recursing through it, and in a PyInstaller
build that trace also has to pass through the frozen import machinery. Measured on
the packaged build, that overruns the default limit and compilation dies with
"RecursionError: maximum recursion depth exceeded" — leaving the app running eager
and roughly 1.56x slower than the same code from source, silently.

Raised here rather than in main.py because dynamo can be reached before the app's
own code runs, and a runtime hook executes before anything else in the bundle.
"""
import sys

# 10000 is the value PyTorch's own CI uses for dynamo-heavy tests. The frames are
# small; the cost of the headroom is negligible next to a 3.8 GB bundle.
#
# Do not raise this further to chase a RecursionError. That was tried — 10000 to
# 50000 — and changed nothing, because the packaged build's RecursionError was not
# a depth problem at all: dynamo could not tell the bundle's copy of torch was torch
# and traced into its own internals, which recurses without a bottom. See
# diareader/compile_support.py. A higher limit only converts a catchable
# RecursionError into an uncatchable native stack overflow, which is strictly worse
# to diagnose.
if sys.getrecursionlimit() < 10000:
    sys.setrecursionlimit(10000)
