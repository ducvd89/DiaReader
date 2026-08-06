"""What torch.compile needs at run time, and where those things come from.

The rule these guard: every input to a run-time compile must live inside the
installed folder. Three separate bugs broke it — the C compiler, then CPython's
headers, then its import library — and each one worked perfectly on the machine that
built the app and failed on the next one.
"""
import sys

import pytest

from diareader import compile_support


def test_nothing_is_patched_when_running_from_source():
    """These fixes describe a bundle. From source the real paths are already right,
    and quietly rewriting them would break a working development install."""
    assert not getattr(sys, "frozen", False), "this test assumes a source run"
    assert compile_support.apply() is None
    assert compile_support.ensure_c_compiler() is None
    assert compile_support.bundled_python_build_dirs() is None
    assert compile_support.use_bundled_python_headers() is None


def test_a_bundle_without_headers_is_reported_as_such(tmp_path, monkeypatch):
    """A build that forgot to ship them must say so, not fail later at
    '#include <Python.h>' on somebody else's machine."""
    monkeypatch.setattr(sys, "_MEIPASS", str(tmp_path), raising=False)
    assert compile_support.bundled_python_build_dirs() is None


def test_headers_are_found_once_the_bundle_carries_them(tmp_path, monkeypatch):
    pybuild = tmp_path / compile_support.BUNDLED_PYBUILD
    (pybuild / "Include").mkdir(parents=True)
    (pybuild / "Include" / "Python.h").write_text("/* headers */")
    (pybuild / "libs").mkdir()
    (pybuild / "libs" / "python312.lib").write_bytes(b"")

    monkeypatch.setattr(sys, "_MEIPASS", str(tmp_path), raising=False)
    dirs = compile_support.bundled_python_build_dirs()
    assert dirs is not None
    include, libs = dirs
    assert (include / "Python.h").is_file()
    assert any(libs.glob("python*.lib"))


def test_headers_alone_are_not_enough(tmp_path, monkeypatch):
    """Compiling needs the headers; *linking* needs pythonNNN.lib. Shipping one
    without the other fails at the second step, which is harder to read."""
    pybuild = tmp_path / compile_support.BUNDLED_PYBUILD
    (pybuild / "Include").mkdir(parents=True)
    (pybuild / "Include" / "Python.h").write_text("/* headers */")
    (pybuild / "libs").mkdir()

    monkeypatch.setattr(sys, "_MEIPASS", str(tmp_path), raising=False)
    assert compile_support.bundled_python_build_dirs() is None


def test_a_compiler_set_by_hand_is_never_overridden(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setenv("CC", r"C:\my\own\cl.exe")
    assert compile_support.ensure_c_compiler() is None
    assert __import__("os").environ["CC"] == r"C:\my\own\cl.exe"


@pytest.mark.parametrize("detail", ["CalledProcessError: tcc.exe returned 1"])
def test_the_remedy_only_suggests_things_that_can_take_effect(detail):
    """It used to recommend installing MSVC, which this module then ignored because
    it sets CC itself. Someone installed gigabytes of build tools and got the same
    error back."""
    text = compile_support.remedy(detail)
    assert detail in text
    assert "use_torch_compile" in text, "turning it off must be offered"
    assert "CC" in text, "the override that actually works must be named"
    assert "winget install Microsoft.VisualStudio" not in text
