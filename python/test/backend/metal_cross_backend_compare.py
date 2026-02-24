#!/usr/bin/env python3
"""Cross-backend numerical comparison harness for Triton Metal validation."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import torch
import triton
import triton.language as tl


@triton.jit
def _vadd(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    y = tl.load(y_ptr + offs, mask=mask, other=0.0)
    tl.store(out_ptr + offs, x + y, mask=mask)


@triton.jit
def _matmul_fp32(
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


@triton.jit
def _matmul_i8(
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
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
    for kk in range(0, k, BLOCK_K):
        for ki in range(0, BLOCK_K):
            k_idx = kk + ki
            a = tl.load(
                a_ptr + offs_m * stride_am + k_idx * stride_ak,
                mask=(offs_m < m) & (k_idx < k),
                other=0,
            ).to(tl.int32)
            b = tl.load(
                b_ptr + k_idx * stride_bk + offs_n * stride_bn,
                mask=(k_idx < k) & (offs_n < n),
                other=0,
            ).to(tl.int32)
            acc += a[:, None] * b[None, :]
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc, mask=(offs_m[:, None] < m) & (offs_n[None, :] < n))


@dataclass
class CompareResult:
    workload: str
    backend: str
    skipped: bool
    skip_reason: str | None
    max_abs_err: float | None
    mean_abs_err: float | None
    passed: bool
    atol: float
    rtol: float


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _is_backend_available(name: str) -> bool:
    if name == "cpu":
        return True
    if name == "mps":
        return bool(
            hasattr(torch.backends, "mps")
            and torch.backends.mps.is_built()
            and torch.backends.mps.is_available()
        )
    if name == "cuda":
        return bool(torch.cuda.is_available())
    return False


def _sync_backend(name: str) -> None:
    if name == "mps":
        torch.mps.synchronize()
    elif name == "cuda":
        torch.cuda.synchronize()


def _vadd_inputs() -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(11)
    n = 4096
    x_cpu = torch.randn((n,), dtype=torch.float32)
    y_cpu = torch.randn((n,), dtype=torch.float32)
    return x_cpu, y_cpu


def _run_vadd_backend(device: str, x_cpu: torch.Tensor, y_cpu: torch.Tensor) -> torch.Tensor:
    n = x_cpu.numel()
    x = x_cpu.to(device)
    y = y_cpu.to(device)
    out = torch.empty_like(x)
    _vadd[(triton.cdiv(n, 256),)](x, y, out, n, BLOCK=256)
    _sync_backend(device)
    return out.to("cpu")


def _matmul_fp32_inputs() -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(17)
    m = n = k = 32
    a_cpu = torch.randn((m, k), dtype=torch.float32)
    b_cpu = torch.randn((k, n), dtype=torch.float32)
    return a_cpu, b_cpu


def _run_matmul_fp32_backend(
    device: str, a_cpu: torch.Tensor, b_cpu: torch.Tensor
) -> torch.Tensor:
    m, k = a_cpu.shape
    n = b_cpu.shape[1]
    a = a_cpu.to(device)
    b = b_cpu.to(device)
    c = torch.empty((m, n), dtype=torch.float32, device=device)
    _matmul_fp32[(triton.cdiv(m, 16), triton.cdiv(n, 16), 1)](
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
    _sync_backend(device)
    return c.to("cpu")


def _matmul_i8_inputs() -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(23)
    m = n = k = 16
    a_cpu = torch.randint(-8, 8, (m, k), dtype=torch.int8)
    b_cpu = torch.randint(-8, 8, (k, n), dtype=torch.int8)
    return a_cpu, b_cpu


def _run_matmul_i8_backend(
    device: str, a_cpu: torch.Tensor, b_cpu: torch.Tensor
) -> torch.Tensor:
    m, k = a_cpu.shape
    n = b_cpu.shape[1]
    a = a_cpu.to(device)
    b = b_cpu.to(device)
    c = torch.empty((m, n), dtype=torch.int32, device=device)
    _matmul_i8[(triton.cdiv(m, 8), triton.cdiv(n, 8), 1)](
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
        BLOCK_M=8,
        BLOCK_N=8,
        BLOCK_K=8,
    )
    _sync_backend(device)
    return c.to("cpu")


def _compute_err(observed: torch.Tensor, expected: torch.Tensor) -> tuple[float, float]:
    diff = (observed.float() - expected.float()).abs()
    return float(diff.max().item()), float(diff.mean().item())


def _run_compare_for_workload(
    workload: str,
    make_inputs_fn,
    cpu_reference_fn,
    backend_run_fn,
    backends: list[str],
    atol: float,
    rtol: float,
    exact: bool = False,
) -> list[CompareResult]:
    inputs = make_inputs_fn()
    cpu_ref = cpu_reference_fn(*inputs)
    results: list[CompareResult] = []
    for backend in backends:
        if backend == "cpu":
            continue
        if not _is_backend_available(backend):
            results.append(
                CompareResult(
                    workload=workload,
                    backend=backend,
                    skipped=True,
                    skip_reason=f"{backend} unavailable",
                    max_abs_err=None,
                    mean_abs_err=None,
                    passed=True,
                    atol=atol,
                    rtol=rtol,
                )
            )
            continue
        observed = backend_run_fn(backend, *inputs)
        if exact:
            passed = torch.equal(observed, cpu_ref)
            max_abs_err, mean_abs_err = _compute_err(observed, cpu_ref)
        else:
            passed = torch.allclose(observed, cpu_ref, atol=atol, rtol=rtol)
            max_abs_err, mean_abs_err = _compute_err(observed, cpu_ref)
        results.append(
            CompareResult(
                workload=workload,
                backend=backend,
                skipped=False,
                skip_reason=None,
                max_abs_err=max_abs_err,
                mean_abs_err=mean_abs_err,
                passed=bool(passed),
                atol=atol,
                rtol=rtol,
            )
        )
    return results


def _artifact_dir(root: str, tag: str) -> Path:
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    out = Path(root).expanduser().resolve() / f"{ts}_{tag}"
    out.mkdir(parents=True, exist_ok=False)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Cross-backend numerical comparison harness")
    parser.add_argument(
        "--backends",
        default="mps,cuda",
        help="Comma-separated device backends to compare against CPU (default: mps,cuda).",
    )
    parser.add_argument(
        "--artifact-root",
        default="artifacts/metal-cross-backend",
        help="Root directory for comparison artifacts.",
    )
    parser.add_argument("--tag", default="cross-backend-compare", help="Artifact run tag.")
    args = parser.parse_args()

    backends = [b.strip() for b in args.backends.split(",") if b.strip()]
    out_dir = _artifact_dir(args.artifact_root, args.tag)

    results: list[CompareResult] = []
    results.extend(
        _run_compare_for_workload(
            workload="vector_add_fp32",
            make_inputs_fn=_vadd_inputs,
            cpu_reference_fn=lambda x, y: x + y,
            backend_run_fn=_run_vadd_backend,
            backends=backends,
            atol=1e-5,
            rtol=1e-5,
        )
    )
    results.extend(
        _run_compare_for_workload(
            workload="matmul_fp32",
            make_inputs_fn=_matmul_fp32_inputs,
            cpu_reference_fn=lambda a, b: a @ b,
            backend_run_fn=_run_matmul_fp32_backend,
            backends=backends,
            atol=3e-4,
            rtol=3e-4,
        )
    )
    results.extend(
        _run_compare_for_workload(
            workload="matmul_int8_i32acc",
            make_inputs_fn=_matmul_i8_inputs,
            cpu_reference_fn=lambda a, b: a.to(torch.int32) @ b.to(torch.int32),
            backend_run_fn=_run_matmul_i8_backend,
            backends=backends,
            atol=0.0,
            rtol=0.0,
            exact=True,
        )
    )

    failures = [
        r
        for r in results
        if not r.skipped and not r.passed
    ]

    summary = {
        "ts_utc": _utc_now_iso(),
        "status": "failed" if failures else "success",
        "backends": backends,
        "results": [asdict(r) for r in results],
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"artifact_dir={out_dir}")
    for r in results:
        if r.skipped:
            print(f"SKIP {r.workload} {r.backend}: {r.skip_reason}")
        else:
            print(
                f"{r.workload} {r.backend}: "
                f"max_abs_err={r.max_abs_err:.6e} mean_abs_err={r.mean_abs_err:.6e} "
                f"passed={r.passed}"
            )
    if failures:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
