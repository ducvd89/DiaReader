"""Collect everything needed to move DiaReader to another machine.

    python tools/package_for_move.py D:\\transfer\\DiaReader
    python tools/package_for_move.py D:\\transfer\\DiaReader --with-model-cache

Copies the source tree without the machine-specific parts (a copied .venv is the usual
way this goes wrong — it holds absolute paths and platform binaries), plus your voices
and settings. The ~10 GB model cache is opt-in since it re-downloads on its own.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from diareader import config  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Rebuilt by pip, or regenerated output. Copying .venv to another machine produces an
# environment that looks installed but points at paths that do not exist there.
SKIP_DIRS = {".venv", ".pytest_cache", "__pycache__", ".git", "node_modules"}
SKIP_SUFFIXES = {".wav", ".mp3", ".pyc"}


def _ignore(directory: str, names: list[str]) -> set[str]:
    skipped = {n for n in names if n in SKIP_DIRS}
    skipped |= {n for n in names if Path(n).suffix.lower() in SKIP_SUFFIXES}
    return skipped


def _size_gb(path: Path) -> float:
    if not path.exists():
        return 0.0
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / (1024 ** 3)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", help="empty folder to write the bundle into")
    parser.add_argument("--with-model-cache", action="store_true",
                        help="also copy the ~10 GB of downloaded weights")
    parser.add_argument("--strip-token", action="store_true",
                        help="blank the Hugging Face token in the copied settings")
    args = parser.parse_args()

    dest = Path(args.destination)
    dest.mkdir(parents=True, exist_ok=True)

    print(f"Copying source tree -> {dest / 'DiaReader'}")
    shutil.copytree(PROJECT_ROOT, dest / "DiaReader", ignore=_ignore, dirs_exist_ok=True)

    # The token lives in two places; stripping only one of them would be worse than
    # stripping neither, because it looks like it was handled.
    env_copy = dest / "DiaReader" / ".env"
    if env_copy.exists():
        if args.strip_token:
            env_copy.write_text("HF_TOKEN=\n", encoding="utf-8")
            print("  token blanked in the copied .env")
        else:
            print("  NOTE: .env carries your Hugging Face token in plain text.")

    if config.APP_DIR.exists():
        target = dest / "AppData_DiaReader"
        print(f"Copying settings and voices -> {target}")
        shutil.copytree(
            config.APP_DIR, target,
            ignore=lambda d, n: {"model_cache"} if not args.with_model_cache else set(),
            dirs_exist_ok=True,
        )
        if args.strip_token:
            import json

            settings_file = target / "settings.json"
            if settings_file.exists():
                data = json.loads(settings_file.read_text(encoding="utf-8"))
                data["hf_token"] = ""
                settings_file.write_text(json.dumps(data, indent=2), encoding="utf-8")
                print("  token blanked in the copy")
        elif (target / "settings.json").exists():
            print("  NOTE: settings.json carries your Hugging Face token in plain text.")

    # Nothing else to fetch: model_cache (copied above with --with-model-cache) holds the
    # fine-tuned weights and the base repo's 941-byte config.json, which is all the model
    # loader reads. The shared Hugging Face cache is not involved.

    print()
    print(f"Done. Bundle is {_size_gb(dest):.2f} GB at {dest}")
    print()
    steps = [
        "Copy DiaReader\\ to wherever you want the project.",
        "Copy AppData_DiaReader\\ to %APPDATA%\\DiaReader",
    ]
    if args.with_model_cache:
        steps.append("Copy hf_hub contents into %USERPROFILE%\\.cache\\huggingface\\hub")
    steps += [
        "python -m venv .venv",
        ".venv\\Scripts\\python.exe -m pip install -r requirements.txt",
        "Reinstall the CUDA build of torch (see README section 2) — required.",
        ".venv\\Scripts\\python.exe -m pytest tests/ -q   (passes without a GPU)",
    ]
    print("On the new machine:")
    for i, step in enumerate(steps, start=1):
        print(f"  {i}. {step}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
