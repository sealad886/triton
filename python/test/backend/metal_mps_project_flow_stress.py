#!/usr/bin/env python3

from __future__ import annotations

import time

import torch

from metal_harness_utils import (
    CrashSafeRunLogger,
    RunConfig,
    dtype_from_name,
    numel,
    parse_common_args,
    require_mode,
    set_determinism,
    sync_mode,
)

_MSL_SOURCE = """\
#include <metal_stdlib>
using namespace metal;

kernel void fused_axpy(
    device float* x [[buffer(0)]],
    device const float* y [[buffer(1)]],
    constant int& n [[buffer(2)]],
    constant float& alpha [[buffer(3)]],
    uint3 gid [[thread_position_in_grid]]
) {
    uint i = gid.x;
    if (i < (uint)n) {
        x[i] = fma(x[i], alpha, y[i]);
    }
}
"""


def _compute_cpu_step(x: torch.Tensor, y: torch.Tensor, alpha: float) -> torch.Tensor:
    return (x * alpha) + y


def main() -> int:
    args = parse_common_args(
        description=(
            "Deterministic project-flow harness that mirrors Triton's Metal runtime "
            "path using torch.mps.compile_shader for kernel launch, with matched CPU mode."
        ),
        default_tag="mps_project_flow",
    )
    require_mode(args.mode, torch)
    set_determinism(args.seed, torch)

    config = RunConfig(
        mode=args.mode,
        iters=args.iters,
        shape=args.shape,
        seed=args.seed,
        transfer_every=args.transfer_every,
        dtype=args.dtype,
        sync_before_transfer=args.sync_before_transfer,
        sync_after_transfer=args.sync_after_transfer,
        run_root=args.run_root,
        tag=args.tag,
    )
    logger = CrashSafeRunLogger(config)
    logger.write_environment(torch, args.mode)

    device = torch.device(args.mode)
    dtype = dtype_from_name(args.dtype)
    n = numel(args.shape)
    alpha = 1.0009765625
    launch_group = min(256, max(1, n))
    transfers = 0
    last_max_abs_err = 0.0
    started = time.perf_counter()

    try:
        logger.mark_checkpoint(-1, "tensor_init_start", extra={"numel": n})
        x_cpu = torch.linspace(-1.0, 1.0, n, dtype=torch.float32).reshape(args.shape)
        y_cpu = torch.full(args.shape, 0.015625, dtype=torch.float32)
        ref_cpu = x_cpu.clone()
        logger.mark_checkpoint(-1, "tensor_init_done")

        if args.mode == "mps":
            logger.mark_checkpoint(-1, "compile_shader_start")
            if not hasattr(torch.mps, "compile_shader"):
                raise RuntimeError("torch.mps.compile_shader is unavailable in this torch build")
            shader = torch.mps.compile_shader(_MSL_SOURCE)
            kernel = shader.fused_axpy
            logger.mark_checkpoint(-1, "compile_shader_done")
            x_dev = x_cpu.to(device=device, dtype=torch.float32)
            y_dev = y_cpu.to(device=device, dtype=torch.float32)
        else:
            x_dev = x_cpu.to(device=device, dtype=dtype)
            y_dev = y_cpu.to(device=device, dtype=dtype)
            kernel = None

        logger.write_state("running", 0, "compute_loop")
        for i in range(args.iters):
            logger.mark_checkpoint(i, "compute_start")
            if args.mode == "mps":
                kernel(
                    x_dev.reshape(-1),
                    y_dev.reshape(-1),
                    n,
                    float(alpha),
                    threads=(n, 1, 1),
                    group_size=(launch_group, 1, 1),
                )
            else:
                x_dev = _compute_cpu_step(x_dev, y_dev, alpha)
            ref_cpu = _compute_cpu_step(ref_cpu, y_cpu, alpha)
            logger.mark_checkpoint(i, "compute_done")

            if ((i + 1) % args.transfer_every) != 0:
                continue

            logger.mark_checkpoint(i, "transfer_to_cpu_pre")
            if args.sync_before_transfer:
                sync_mode(args.mode, torch)
            observed_cpu = x_dev.to(device="cpu", dtype=torch.float32)
            last_max_abs_err = float((observed_cpu - ref_cpu).abs().max().item())
            transfers += 1
            logger.event(
                "transfer_complete",
                iteration=i,
                transfer_count=transfers,
                max_abs_err=last_max_abs_err,
            )
            if args.sync_after_transfer:
                sync_mode(args.mode, torch)
            logger.mark_checkpoint(
                i,
                "transfer_to_cpu_post",
                extra={
                    "transfer_count": transfers,
                    "max_abs_err": last_max_abs_err,
                },
            )

        logger.mark_checkpoint(args.iters - 1, "cleanup_sync_pre")
        sync_mode(args.mode, torch)
        logger.mark_checkpoint(args.iters - 1, "cleanup_sync_post")

        elapsed_s = time.perf_counter() - started
        logger.finish_success(
            {
                "elapsed_s": elapsed_s,
                "iterations": args.iters,
                "transfers": transfers,
                "last_max_abs_err": last_max_abs_err,
                "alpha": alpha,
            }
        )
        print(
            f"SUCCESS mode={args.mode} iters={args.iters} transfers={transfers} "
            f"max_abs_err={last_max_abs_err:.6e}",
            flush=True,
        )
        return 0
    except BaseException as exc:
        logger.finish_python_exception(exc)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
