"""Reading round a huggingface_hub cache whose symlinks do not resolve.

The bug this guards against reported success at every step and failed several steps
later, in a message about the wrong thing: hf_hub_download returned a path, the path
existed as a directory entry, and the model load died with "Could not parse the
architecture config" — reading as corruption rather than as a file that could not be
opened at all.
"""
import os
import sys

import pytest

from diareader.hf_cache import resolve_download


def test_a_working_path_is_returned_untouched(tmp_path):
    real = tmp_path / "config.json"
    real.write_text("{}", encoding="utf-8")
    assert resolve_download(real) == real


def test_a_working_symlink_is_returned_untouched(tmp_path):
    """The normal cache layout. Only a *broken* link is worth rewriting — following
    working ones would hand callers blob paths for no reason."""
    blob = tmp_path / "blob"
    blob.write_text("hello", encoding="utf-8")
    link = tmp_path / "link"
    try:
        os.symlink(blob, link)
    except OSError:
        pytest.skip("this machine cannot create symlinks at all")
    if not link.exists():
        pytest.skip("symlinks are not resolvable in the temp directory either")
    assert resolve_download(link) == link


def test_a_broken_link_resolves_to_its_blob(tmp_path, monkeypatch):
    """Simulated rather than reproduced: the real failure needs a filesystem where
    creating a symlink succeeds and opening it does not, which is exactly the state
    that cannot be arranged on demand."""
    blob = tmp_path / "blobs" / "abc123"
    blob.parent.mkdir()
    blob.write_text("real content", encoding="utf-8")
    link = tmp_path / "snapshots" / "rev" / "config.json"
    link.parent.mkdir(parents=True)
    link.write_text("placeholder", encoding="utf-8")

    monkeypatch.setattr(os.path, "islink", lambda p: str(p) == str(link))
    monkeypatch.setattr(os, "readlink", lambda p: os.path.join("..", "..", "blobs", "abc123"))
    monkeypatch.setattr(
        type(link), "exists", lambda self: False if str(self) == str(link) else os.path.isfile(self)
    )

    assert resolve_download(link) == blob


def test_a_broken_link_with_no_blob_is_left_alone(tmp_path, monkeypatch):
    """Then the caller's own error is about the file it wanted, which says more than
    one about symlinks."""
    link = tmp_path / "snapshots" / "rev" / "config.json"
    link.parent.mkdir(parents=True)

    monkeypatch.setattr(os.path, "islink", lambda p: str(p) == str(link))
    monkeypatch.setattr(os, "readlink", lambda p: os.path.join("..", "..", "blobs", "missing"))

    assert resolve_download(link) == link


def test_a_plain_missing_path_is_left_alone(tmp_path):
    missing = tmp_path / "nothing-here.json"
    assert resolve_download(missing) == missing


@pytest.mark.skipif(sys.platform != "win32", reason="the failure is Windows-only")
def test_the_model_cache_is_checked_for_this(tmp_path):
    """Not an assertion about the machine — a record of how to tell. Creating a
    symlink succeeding proves nothing; opening it is the test that failed."""
    probe = tmp_path / "probe"
    probe.write_text("x", encoding="utf-8")
    link = tmp_path / "probe-link"
    try:
        os.symlink(probe, link)
    except OSError:
        pytest.skip("no symlink privilege here")
    # Whatever the answer, resolve_download must return something readable.
    resolved = resolve_download(link)
    assert resolved.exists()
