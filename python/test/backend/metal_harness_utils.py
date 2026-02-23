#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import platform
import random
import subprocess
import sys
import traceback
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from functools import reduce
from operator import mul
from pathlib import Path
from typing import Any

import numpy as np


def dtype_from_name(name: str):
    """Convert a dtype name string to a torch dtype.

    Import torch lazily so the module can be imported without torch installed.
    """
    import torch

    if name == "float16":
        return torch.float16
    if name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def numel(shape: tuple[int, ...]) -> int:
    """Return the number of elements for a given shape tuple."""
    return int(reduce(mul, shape, 1))


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_shape(shape: str) -> tuple[int, ...]:
    parts = [x.strip() for x in shape.split(",") if x.strip()]
    if not parts:
        raise ValueError(f"Invalid empty shape: {shape!r}")
    parsed = tuple(int(x) for x in parts)
    if any(x <= 0 for x in parsed):
        raise ValueError(f"Shape dimensions must be >0, got: {parsed}")
    return parsed


def parse_common_args(description: str, default_tag: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--mode",
        choices=("cpu", "mps"),
        default="mps",
        help="Execution mode.",
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=500,
        help="Number of deterministic loop iterations.",
    )
    parser.add_argument(
        "--shape",
        default="131072",
        help="Comma-separated tensor shape, e.g. '4096,256'.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1337,
        help="Deterministic seed.",
    )
    parser.add_argument(
        "--transfer-every",
        type=int,
        default=1,
        help="How often to force a device->host transfer.",
    )
    parser.add_argument(
        "--dtype",
        choices=("float16", "float32"),
        default="float32",
        help="Tensor dtype for the stress tensors.",
    )
    parser.add_argument(
        "--run-root",
        default="artifacts/metal-harness-runs",
        help="Root directory where timestamped run artifacts are stored.",
    )
    parser.add_argument(
        "--tag",
        default=default_tag,
        help="Tag used in the run directory name.",
    )
    parser.add_argument(
        "--sync-before-transfer",
        action="store_true",
        help="Insert explicit MPS synchronize() before transfer boundaries.",
    )
    parser.add_argument(
        "--sync-after-transfer",
        action="store_true",
        help="Insert explicit MPS synchronize() after transfer boundaries.",
    )
    args = parser.parse_args()
    args.shape = _parse_shape(args.shape)
    if args.iters <= 0:
        raise ValueError("--iters must be > 0")
    if args.transfer_every <= 0:
        raise ValueError("--transfer-every must be > 0")
    return args


def _macos_version() -> str:
    try:
        out = subprocess.check_output(["sw_vers", "-productVersion"], text=True).strip()
        if out:
            return out
    except Exception:
        pass
    return platform.mac_ver()[0] or "unknown"


def set_determinism(seed: int, torch_module: Any) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch_module.manual_seed(seed)


def require_mode(mode: str, torch_module: Any) -> None:
    if mode == "cpu":
        return
    if mode != "mps":
        raise RuntimeError(f"Unsupported mode: {mode!r}")
    if not torch_module.backends.mps.is_built():
        raise RuntimeError(
            "Requested mode 'mps' but this torch build has no MPS support"
        )
    if not torch_module.backends.mps.is_available():
        raise RuntimeError("Requested mode 'mps' but MPS is unavailable on this host")


def sync_mode(mode: str, torch_module: Any) -> None:
    if mode == "mps":
        torch_module.mps.synchronize()


def classify_from_stage(stage: str | None) -> str:
    if stage is None:
        return "unknown"
    lowered = stage.lower()
    if "cleanup" in lowered:
        return "cleanup/teardown-crash"
    if "transfer" in lowered or "to_cpu" in lowered or "copy" in lowered:
        return "transfer-sync-crash"
    if "compute" in lowered or "kernel" in lowered:
        return "compute-time-crash"
    return "unknown"


@dataclass
class RunConfig:
    mode: str
    iters: int
    shape: tuple[int, ...]
    seed: int
    transfer_every: int
    dtype: str
    sync_before_transfer: bool
    sync_after_transfer: bool
    run_root: str
    tag: str


class CrashSafeRunLogger:
    def __init__(self, config: RunConfig):
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.run_dir = (
            Path(config.run_root).expanduser().resolve()
            / f"{ts}_{config.tag}_{config.mode}"
        )
        self.run_dir.mkdir(parents=True, exist_ok=False)

        self.config_path = self.run_dir / "config.json"
        self.env_path = self.run_dir / "environment.json"
        self.events_path = self.run_dir / "events.jsonl"
        self.state_path = self.run_dir / "run_state.json"
        self.summary_path = self.run_dir / "summary.json"
        self.exception_path = self.run_dir / "python_exception.txt"
        self.last_checkpoint_path = self.run_dir / "last_successful_checkpoint.txt"

        self._events_fp = self.events_path.open("a", encoding="utf-8")
        self.last_stage: str | None = None
        self.last_iter: int = -1
        self._write_json_atomic(self.config_path, asdict(config))
        self.write_state("starting", -1, "startup")
        self.event("run_started")

    def _write_json_atomic(self, path: Path, payload: dict[str, Any]) -> None:
        tmp = path.with_suffix(path.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as fp:
            json.dump(payload, fp, indent=2, sort_keys=True)
            fp.write("\n")
            fp.flush()
            os.fsync(fp.fileno())
        tmp.replace(path)

    def _write_text_atomic(self, path: Path, text: str) -> None:
        tmp = path.with_suffix(path.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as fp:
            fp.write(text)
            fp.flush()
            os.fsync(fp.fileno())
        tmp.replace(path)

    def event(self, event: str, **payload: Any) -> None:
        record = {"ts_utc": utc_now_iso(), "event": event, **payload}
        self._events_fp.write(json.dumps(record, sort_keys=True) + "\n")
        self._events_fp.flush()
        os.fsync(self._events_fp.fileno())

    def write_environment(self, torch_module: Any, mode: str) -> None:
        env = {
            "python_version": sys.version,
            "python_executable": sys.executable,
            "torch_version": getattr(torch_module, "__version__", "unknown"),
            "platform": platform.platform(),
            "macos_version": _macos_version(),
            "selected_mode": mode,
            "mps_built": bool(torch_module.backends.mps.is_built()),
            "mps_available": bool(torch_module.backends.mps.is_available()),
        }
        self._write_json_atomic(self.env_path, env)
        self.print_banner(env)

    def print_banner(self, env: dict[str, Any]) -> None:
        lines = [
            "=== Triton Metal Harness Startup ===",
            f"python: {env['python_version'].splitlines()[0]}",
            f"torch: {env['torch_version']}",
            f"macOS: {env['macos_version']}",
            f"platform: {env['platform']}",
            f"mode: {env['selected_mode']}",
            f"mps_built: {env['mps_built']}",
            f"mps_available: {env['mps_available']}",
            f"run_dir: {self.run_dir}",
        ]
        print("\n".join(lines), flush=True)

    def mark_checkpoint(
        self,
        iteration: int,
        stage: str,
        *,
        extra: dict[str, Any] | None = None,
    ) -> None:
        self.last_stage = stage
        self.last_iter = iteration
        payload = {
            "ts_utc": utc_now_iso(),
            "iteration": iteration,
            "stage": stage,
            "extra": extra or {},
        }
        self._write_json_atomic(self.state_path, payload)
        self._write_text_atomic(
            self.last_checkpoint_path,
            json.dumps(payload, sort_keys=True),
        )
        self.event("checkpoint", iteration=iteration, stage=stage, extra=extra or {})

    def write_state(self, status: str, iteration: int, stage: str) -> None:
        payload = {
            "ts_utc": utc_now_iso(),
            "status": status,
            "iteration": iteration,
            "stage": stage,
        }
        self._write_json_atomic(self.state_path, payload)
        self.event("state", **payload)

    def finish_success(self, metrics: dict[str, Any]) -> None:
        summary = {
            "ts_utc": utc_now_iso(),
            "status": "success",
            "last_iteration": self.last_iter,
            "last_stage": self.last_stage,
            "classification": "none",
            "metrics": metrics,
        }
        self._write_json_atomic(self.summary_path, summary)
        self.write_state("success", self.last_iter, self.last_stage or "unknown")
        self.event("run_finished", summary=summary)
        self._events_fp.close()

    def finish_python_exception(self, exc: BaseException) -> None:
        tb = traceback.format_exc()
        self._write_text_atomic(self.exception_path, tb)
        classification = classify_from_stage(self.last_stage)
        summary = {
            "ts_utc": utc_now_iso(),
            "status": "python_exception",
            "last_iteration": self.last_iter,
            "last_stage": self.last_stage,
            "classification": classification,
            "exception_type": type(exc).__name__,
            "exception_message": str(exc),
        }
        self._write_json_atomic(self.summary_path, summary)
        self.write_state(
            "python_exception", self.last_iter, self.last_stage or "unknown"
        )
        self.event("run_failed", summary=summary)
        self._events_fp.close()
