#!/usr/bin/env python3
"""Run release-readiness checks for the Triton Metal backend.

This script centralizes reproducible local release checks with structured
artifacts and optional soak runs.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _default_python_executable() -> str:
    venv_python = Path(".venv") / "bin" / "python"
    if venv_python.exists():
        return str(venv_python)
    return sys.executable


def _ensure_artifact_dir(root: str, tag: str) -> Path:
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = Path(root).expanduser().resolve() / f"{ts}_{tag}"
    path.mkdir(parents=True, exist_ok=False)
    return path


@dataclass
class CheckResult:
    name: str
    command: list[str]
    returncode: int
    start_utc: str
    end_utc: str
    duration_s: float
    stdout_path: str
    stderr_path: str
    cache_dir: str


def _run_one(
    name: str,
    cmd: Sequence[str],
    workdir: str,
    artifact_dir: Path,
    timeout_s: int,
) -> CheckResult:
    cache_dir = tempfile.mkdtemp(prefix=f"triton-metal-release-cache-{name}-")
    env = os.environ.copy()
    env["TRITON_CACHE_DIR"] = cache_dir
    start = datetime.now(timezone.utc)
    start_s = _utc_now_iso()
    proc = subprocess.run(
        list(cmd),
        cwd=workdir,
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout_s,
    )
    end = datetime.now(timezone.utc)
    end_s = _utc_now_iso()
    duration_s = (end - start).total_seconds()

    stdout_path = artifact_dir / f"{name}.stdout.log"
    stderr_path = artifact_dir / f"{name}.stderr.log"
    stdout_path.write_text(proc.stdout, encoding="utf-8")
    stderr_path.write_text(proc.stderr, encoding="utf-8")

    return CheckResult(
        name=name,
        command=list(cmd),
        returncode=proc.returncode,
        start_utc=start_s,
        end_utc=end_s,
        duration_s=duration_s,
        stdout_path=str(stdout_path),
        stderr_path=str(stderr_path),
        cache_dir=cache_dir,
    )


def _checks(python_exe: str, soak: bool) -> list[tuple[str, list[str], int]]:
    checks: list[tuple[str, list[str], int]] = [
        (
            "metal_backend_pytests",
            [
                python_exe,
                "-m",
                "pytest",
                "-q",
                "python/test/backend/test_metal_backend.py",
            ],
            1800,
        ),
        (
            "metal_smoke_script",
            [python_exe, "scripts/test_metal_smoke.py"],
            900,
        ),
        (
            "metal_aot_unit",
            [python_exe, "-m", "pytest", "-q", "python/test/unit/tools/test_aot_metal.py"],
            600,
        ),
        (
            "aot_unit_collection_gate",
            [python_exe, "-m", "pytest", "-q", "python/test/unit/tools/test_aot.py"],
            600,
        ),
    ]
    if soak:
        checks.extend(
            [
                (
                    "soak_transfer_mps",
                    [
                        python_exe,
                        "python/test/backend/metal_mps_transfer_stress.py",
                        "--mode",
                        "mps",
                        "--iters",
                        "2048",
                        "--shape",
                        "65536",
                        "--transfer-every",
                        "1",
                        "--sync-before-transfer",
                        "--sync-after-transfer",
                        "--tag",
                        "release-soak-transfer-mps",
                    ],
                    1800,
                ),
                (
                    "soak_project_flow_mps",
                    [
                        python_exe,
                        "python/test/backend/metal_mps_project_flow_stress.py",
                        "--mode",
                        "mps",
                        "--iters",
                        "1024",
                        "--shape",
                        "65536",
                        "--transfer-every",
                        "1",
                        "--sync-before-transfer",
                        "--sync-after-transfer",
                        "--tag",
                        "release-soak-project-mps",
                    ],
                    1800,
                ),
                (
                    "soak_training_loop_mps",
                    [
                        python_exe,
                        "python/test/backend/metal_mps_training_loop_stress.py",
                        "--mode",
                        "mps",
                        "--iters",
                        "1024",
                        "--shape",
                        "65536",
                        "--transfer-every",
                        "1",
                        "--sync-before-transfer",
                        "--sync-after-transfer",
                        "--tag",
                        "release-soak-training-mps",
                    ],
                    1800,
                ),
            ]
        )
    return checks


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Triton Metal release checks")
    parser.add_argument(
        "--python",
        default=_default_python_executable(),
        help="Python executable to use for checks (default: .venv/bin/python if present).",
    )
    parser.add_argument(
        "--artifact-root",
        default="artifacts/metal-release-checks",
        help="Root directory for release-check artifacts.",
    )
    parser.add_argument(
        "--tag",
        default="release-check",
        help="Tag suffix for artifact directory.",
    )
    parser.add_argument(
        "--soak",
        action="store_true",
        help="Run extended MPS soak checks in addition to the default suite.",
    )
    args = parser.parse_args()

    if shutil.which(args.python) is None and not Path(args.python).exists():
        print(f"ERROR: Python executable not found: {args.python}", file=sys.stderr)
        return 2

    artifact_dir = _ensure_artifact_dir(args.artifact_root, args.tag)
    summary_path = artifact_dir / "summary.json"
    results: list[CheckResult] = []
    failed = False

    print(f"artifact_dir={artifact_dir}")
    for name, cmd, timeout_s in _checks(args.python, args.soak):
        print(f"\n--- {name} ---")
        print("cmd:", " ".join(cmd))
        try:
            result = _run_one(
                name=name,
                cmd=cmd,
                workdir=os.getcwd(),
                artifact_dir=artifact_dir,
                timeout_s=timeout_s,
            )
            results.append(result)
            print(
                f"rc={result.returncode} duration_s={result.duration_s:.2f} "
                f"stdout={result.stdout_path}"
            )
            if result.returncode != 0:
                failed = True
                break
        except subprocess.TimeoutExpired as exc:
            failed = True
            timeout_result = CheckResult(
                name=name,
                command=list(cmd),
                returncode=124,
                start_utc=_utc_now_iso(),
                end_utc=_utc_now_iso(),
                duration_s=float(timeout_s),
                stdout_path=str(artifact_dir / f"{name}.stdout.log"),
                stderr_path=str(artifact_dir / f"{name}.stderr.log"),
                cache_dir="timeout",
            )
            Path(timeout_result.stdout_path).write_text(
                (exc.stdout or ""),
                encoding="utf-8",
            )
            Path(timeout_result.stderr_path).write_text(
                (exc.stderr or ""),
                encoding="utf-8",
            )
            results.append(timeout_result)
            print(f"rc=124 timeout_s={timeout_s} stdout={timeout_result.stdout_path}")
            break

    summary = {
        "ts_utc": _utc_now_iso(),
        "status": "failed" if failed else "success",
        "python": args.python,
        "soak": args.soak,
        "artifact_dir": str(artifact_dir),
        "results": [asdict(x) for x in results],
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"\nsummary={summary_path}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
