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


def main() -> int:
    args = parse_common_args(
        description=(
            "Deterministic stress harness for classifying MPS->CPU transfer crashes. "
            "The script intentionally alternates compute and explicit transfer phases."
        ),
        default_tag="mps_transfer_stress",
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
    transfer_count = 0
    last_checksum = 0.0
    started = time.perf_counter()

    try:
        logger.mark_checkpoint(-1, "tensor_init_start", extra={"numel": n})
        base = torch.linspace(0.0, 1.0, n, dtype=torch.float32).reshape(args.shape)
        delta = torch.full(args.shape, 0.03125, dtype=torch.float32)
        x = base.to(device=device, dtype=dtype)
        y = delta.to(device=device, dtype=dtype)
        logger.mark_checkpoint(-1, "tensor_init_done")
        logger.write_state("running", 0, "compute_loop")

        for i in range(args.iters):
            logger.mark_checkpoint(i, "compute_start")
            x = (x * 1.0009765625) + y
            logger.mark_checkpoint(i, "compute_done")

            should_transfer = ((i + 1) % args.transfer_every) == 0
            if not should_transfer:
                continue

            logger.mark_checkpoint(i, "transfer_to_cpu_pre")
            if args.sync_before_transfer:
                sync_mode(args.mode, torch)
            host = x.to(device="cpu", dtype=torch.float32)
            last_checksum = float(host.sum().item())
            transfer_count += 1
            logger.event(
                "transfer_complete",
                iteration=i,
                checksum=last_checksum,
                transfer_count=transfer_count,
            )
            if args.sync_after_transfer:
                sync_mode(args.mode, torch)
            logger.mark_checkpoint(
                i,
                "transfer_to_cpu_post",
                extra={"checksum": last_checksum, "transfer_count": transfer_count},
            )

        logger.mark_checkpoint(args.iters - 1, "cleanup_sync_pre")
        sync_mode(args.mode, torch)
        logger.mark_checkpoint(args.iters - 1, "cleanup_sync_post")

        elapsed_s = time.perf_counter() - started
        logger.finish_success(
            {
                "elapsed_s": elapsed_s,
                "iterations": args.iters,
                "transfers": transfer_count,
                "last_checksum": last_checksum,
            }
        )
        print(
            f"SUCCESS mode={args.mode} iters={args.iters} "
            f"transfers={transfer_count} checksum={last_checksum:.6f}",
            flush=True,
        )
        return 0
    except BaseException as exc:
        logger.finish_python_exception(exc)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
