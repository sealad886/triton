#!/usr/bin/env python3
"""Throughput guardrail checks for Triton Metal matmul kernels."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import torch
import triton
import triton.language as tl


@triton.jit
def _matmul_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    m,
    n,
    k,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for kk in range(0, k, BLOCK_K):
        a = tl.load(
            a_ptr + offs_m[:, None] * stride_am + (offs_k[None, :] + kk) * stride_ak,
            mask=(offs_m[:, None] < m) & (offs_k[None, :] + kk < k),
            other=0.0,
        )
        b = tl.load(
            b_ptr + (offs_k[:, None] + kk) * stride_bk + offs_n[None, :] * stride_bn,
            mask=(offs_k[:, None] + kk < k) & (offs_n[None, :] < n),
            other=0.0,
        )
        acc += tl.dot(a, b)

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc, mask=(offs_m[:, None] < m) & (offs_n[None, :] < n))


@dataclass
class ThroughputResult:
    workload: str
    dtype: str
    m: int
    n: int
    k: int
    median_s: float
    mean_s: float
    tflops: float
    baseline_tflops: float | None
    threshold_tflops: float | None
    passed: bool
    skipped: bool
    skip_reason: str | None


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _has_mps() -> bool:
    return bool(
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_built()
        and torch.backends.mps.is_available()
    )


def _target_arch() -> str:
    try:
        return triton.runtime.driver.active.get_current_target().arch
    except Exception:
        return "unknown"


def _dtype_supported_on_mps(dtype: torch.dtype) -> bool:
    try:
        torch.empty((1,), dtype=dtype, device="mps")
        return True
    except Exception:
        return False


def _artifact_dir(root: str, tag: str) -> Path:
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    out = Path(root).expanduser().resolve() / f"{ts}_{tag}"
    out.mkdir(parents=True, exist_ok=False)
    return out


def _run_workload(
    name: str,
    dtype: torch.dtype,
    m: int,
    n: int,
    k: int,
    warmup: int,
    reps: int,
    baseline_tflops: float | None,
    min_scale: float,
) -> ThroughputResult:
    dtype_name = str(dtype).split(".")[-1]
    if not _dtype_supported_on_mps(dtype):
        return ThroughputResult(
            workload=name,
            dtype=dtype_name,
            m=m,
            n=n,
            k=k,
            median_s=0.0,
            mean_s=0.0,
            tflops=0.0,
            baseline_tflops=baseline_tflops,
            threshold_tflops=(
                baseline_tflops * min_scale if baseline_tflops is not None else None
            ),
            passed=True,
            skipped=True,
            skip_reason=f"dtype {dtype_name} unsupported on MPS",
        )

    torch.manual_seed(1337)
    a = torch.randn((m, k), dtype=torch.float32).to(dtype=dtype, device="mps")
    b = torch.randn((k, n), dtype=torch.float32).to(dtype=dtype, device="mps")
    c = torch.empty((m, n), dtype=torch.float32, device="mps")

    grid = lambda meta: (triton.cdiv(m, meta["BLOCK_M"]), triton.cdiv(n, meta["BLOCK_N"]), 1)

    for _ in range(warmup):
        _matmul_kernel[grid](
            a,
            b,
            c,
            m,
            n,
            k,
            a.stride(0),
            a.stride(1),
            b.stride(0),
            b.stride(1),
            c.stride(0),
            c.stride(1),
            BLOCK_M=16,
            BLOCK_N=16,
            BLOCK_K=16,
        )
    torch.mps.synchronize()

    times: list[float] = []
    for _ in range(reps):
        t0 = time.perf_counter()
        _matmul_kernel[grid](
            a,
            b,
            c,
            m,
            n,
            k,
            a.stride(0),
            a.stride(1),
            b.stride(0),
            b.stride(1),
            c.stride(0),
            c.stride(1),
            BLOCK_M=16,
            BLOCK_N=16,
            BLOCK_K=16,
        )
        torch.mps.synchronize()
        times.append(time.perf_counter() - t0)

    median_s = statistics.median(times)
    mean_s = statistics.mean(times)
    tflops = (2.0 * m * n * k) / (median_s * 1e12)
    threshold = baseline_tflops * min_scale if baseline_tflops is not None else None
    passed = True if threshold is None else (tflops >= threshold)

    return ThroughputResult(
        workload=name,
        dtype=dtype_name,
        m=m,
        n=n,
        k=k,
        median_s=median_s,
        mean_s=mean_s,
        tflops=tflops,
        baseline_tflops=baseline_tflops,
        threshold_tflops=threshold,
        passed=passed,
        skipped=False,
        skip_reason=None,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Triton Metal matmul throughput guardrails")
    parser.add_argument(
        "--baseline",
        default="docs/metal-matmul-throughput-baselines.json",
        help="Path to throughput baseline JSON file.",
    )
    parser.add_argument(
        "--artifact-root",
        default="artifacts/metal-throughput-guard",
        help="Directory root for throughput run artifacts.",
    )
    parser.add_argument("--tag", default="throughput-guard", help="Artifact run tag.")
    parser.add_argument("--warmup", type=int, default=4, help="Warmup iterations.")
    parser.add_argument("--reps", type=int, default=12, help="Measured repetitions.")
    parser.add_argument(
        "--min-scale",
        type=float,
        default=0.6,
        help="Required fraction of baseline throughput to pass.",
    )
    parser.add_argument(
        "--write-baseline",
        action="store_true",
        help="Update baseline values for detected arch from this run.",
    )
    args = parser.parse_args()

    if not _has_mps():
        print("SKIP: MPS backend unavailable")
        return 0

    arch = _target_arch()
    baseline_path = Path(args.baseline).resolve()
    if baseline_path.exists():
        baseline_data = json.loads(baseline_path.read_text(encoding="utf-8"))
    else:
        baseline_data = {}

    workloads = [
        ("fp32_m128n128k128", torch.float32, 128, 128, 128),
        ("fp16_m256n256k256", torch.float16, 256, 256, 256),
        ("bf16_m256n256k256", torch.bfloat16, 256, 256, 256),
    ]

    per_arch_baseline = baseline_data.get(arch, {})
    results: list[ThroughputResult] = []
    failures: list[str] = []

    for name, dtype, m, n, k in workloads:
        baseline_tflops = per_arch_baseline.get(name)
        result = _run_workload(
            name=name,
            dtype=dtype,
            m=m,
            n=n,
            k=k,
            warmup=args.warmup,
            reps=args.reps,
            baseline_tflops=baseline_tflops,
            min_scale=args.min_scale,
        )
        results.append(result)
        if result.skipped:
            print(f"SKIP {name}: {result.skip_reason}")
            continue
        print(
            f"{name}: median={result.median_s:.6f}s tflops={result.tflops:.3f}",
            flush=True,
        )
        if not result.passed:
            failures.append(
                f"{name}: {result.tflops:.3f} < threshold {result.threshold_tflops:.3f}"
            )

    if args.write_baseline:
        baseline_data.setdefault(arch, {})
        for r in results:
            if not r.skipped:
                baseline_data[arch][r.workload] = r.tflops
        baseline_path.parent.mkdir(parents=True, exist_ok=True)
        baseline_path.write_text(
            json.dumps(baseline_data, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"Updated baseline: {baseline_path}")

    out_dir = _artifact_dir(args.artifact_root, args.tag)
    summary = {
        "ts_utc": _utc_now_iso(),
        "arch": arch,
        "baseline_path": str(baseline_path),
        "status": "failed" if failures else "success",
        "min_scale": args.min_scale,
        "warmup": args.warmup,
        "reps": args.reps,
        "results": [asdict(r) for r in results],
        "failures": failures,
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"artifact_dir={out_dir}")
    if failures:
        for f in failures:
            print(f"FAIL: {f}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
