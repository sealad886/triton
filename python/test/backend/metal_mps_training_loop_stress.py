#!/usr/bin/env python3

from __future__ import annotations

import time

import torch
import triton
import triton.language as tl
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


@triton.jit
def _sgd_step_kernel(
    param_ptr,
    momentum_ptr,
    lr,
    n_elements,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    param = tl.load(param_ptr + offs, mask=mask, other=0.0)
    momentum = tl.load(momentum_ptr + offs, mask=mask, other=0.0)
    tl.store(param_ptr + offs, param - (lr * momentum), mask=mask)


def _training_step_torch(
    param: torch.Tensor,
    momentum: torch.Tensor,
    inp: torch.Tensor,
    target: torch.Tensor,
    bias: torch.Tensor,
    lr: float,
    beta: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    n = float(param.numel())
    pred = param * inp + bias
    loss_grad = (pred - target) / n
    grad = loss_grad * inp
    momentum = (beta * momentum) + grad
    param = param - (lr * momentum)
    bias = bias - (lr * loss_grad.sum())
    loss = (pred - target).square().mean()
    return param, momentum, bias, loss


def main() -> int:
    args = parse_common_args(
        description=(
            "Deterministic training-style stress harness with optimizer-like "
            "updates, transfer checkpoints, and CPU/MPS mode split."
        ),
        default_tag="mps_training_loop_stress",
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
    lr = 0.03125
    beta = 0.9
    block = 256
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK"]),)
    transfers = 0
    last_max_abs_err = 0.0
    last_loss = 0.0
    started = time.perf_counter()

    try:
        logger.mark_checkpoint(-1, "tensor_init_start", extra={"numel": n})
        inp_cpu = torch.linspace(-1.0, 1.0, n, dtype=torch.float32).reshape(args.shape)
        target_cpu = torch.linspace(0.5, -0.5, n, dtype=torch.float32).reshape(
            args.shape
        )
        param_ref = torch.full(args.shape, 0.125, dtype=torch.float32)
        momentum_ref = torch.zeros(args.shape, dtype=torch.float32)
        bias_ref = torch.tensor(0.03125, dtype=torch.float32)

        param_dev = param_ref.to(device=device, dtype=dtype)
        momentum_dev = momentum_ref.to(device=device, dtype=dtype)
        inp_dev = inp_cpu.to(device=device, dtype=dtype)
        target_dev = target_cpu.to(device=device, dtype=dtype)
        bias_dev = bias_ref.to(device=device, dtype=dtype)
        logger.mark_checkpoint(-1, "tensor_init_done")
        logger.write_state("running", 0, "training_loop")

        for i in range(args.iters):
            logger.mark_checkpoint(i, "compute_forward_start")
            n_dev = float(n)
            pred_dev = (param_dev * inp_dev) + bias_dev
            loss_grad_dev = (pred_dev - target_dev) / n_dev
            logger.mark_checkpoint(i, "compute_forward_done")

            logger.mark_checkpoint(i, "compute_backward_start")
            grad_dev = loss_grad_dev * inp_dev
            momentum_dev = (beta * momentum_dev) + grad_dev
            logger.mark_checkpoint(i, "compute_backward_done")

            logger.mark_checkpoint(i, "optimizer_update_start")
            if args.mode == "mps":
                _sgd_step_kernel[grid](
                    param_dev.reshape(-1),
                    momentum_dev.reshape(-1),
                    lr,
                    n,
                    BLOCK=block,
                )
            else:
                param_dev = param_dev - (lr * momentum_dev)
            bias_dev = bias_dev - (lr * loss_grad_dev.sum())
            logger.mark_checkpoint(i, "optimizer_update_done")

            param_ref, momentum_ref, bias_ref, loss_ref = _training_step_torch(
                param_ref,
                momentum_ref,
                inp_cpu,
                target_cpu,
                bias_ref,
                lr,
                beta,
            )
            last_loss = float(loss_ref.item())

            if ((i + 1) % args.transfer_every) != 0:
                continue

            logger.mark_checkpoint(i, "transfer_to_cpu_pre")
            if args.sync_before_transfer:
                sync_mode(args.mode, torch)
            observed_cpu = param_dev.to(device="cpu", dtype=torch.float32)
            last_max_abs_err = float((observed_cpu - param_ref).abs().max().item())
            transfers += 1
            logger.event(
                "transfer_complete",
                iteration=i,
                transfer_count=transfers,
                max_abs_err=last_max_abs_err,
                loss=last_loss,
            )
            if args.sync_after_transfer:
                sync_mode(args.mode, torch)
            logger.mark_checkpoint(
                i,
                "transfer_to_cpu_post",
                extra={
                    "transfer_count": transfers,
                    "max_abs_err": last_max_abs_err,
                    "loss": last_loss,
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
                "last_loss": last_loss,
                "lr": lr,
                "beta": beta,
            }
        )
        print(
            f"SUCCESS mode={args.mode} iters={args.iters} transfers={transfers} "
            f"last_loss={last_loss:.6e} max_abs_err={last_max_abs_err:.6e}",
            flush=True,
        )
        return 0
    except BaseException as exc:
        logger.finish_python_exception(exc)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
