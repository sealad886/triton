from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PYTHON_ROOT = REPO_ROOT / "python"


def ensure_repo_imports() -> None:
    for path in (PYTHON_ROOT, REPO_ROOT):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)


def repo_pythonpath(environ: dict[str, str] | None = None) -> str:
    source_env = os.environ if environ is None else environ
    parts: list[str] = [str(PYTHON_ROOT), str(REPO_ROOT)]
    existing = source_env.get("PYTHONPATH", "")
    if existing:
        parts.extend(part for part in existing.split(os.pathsep) if part)

    deduped: list[str] = []
    seen: set[str] = set()
    for part in parts:
        if part in seen:
            continue
        deduped.append(part)
        seen.add(part)
    return os.pathsep.join(deduped)


def repo_subprocess_env() -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = repo_pythonpath(env)
    return env
