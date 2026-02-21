# Metal MPS Native Crash Incident Report

Date: 2026-02-21  
Branch: `feat/metal-support`

## Crash Signature Summary

- Observed signature (from prior failing runs): `EXC_BAD_ACCESS` / `SIGSEGV`
  in native PyTorch MPS/Metal stack.
- Reported timing: asynchronous Metal completion cleanup around explicit
  MPS-to-CPU synchronization/transfer boundaries (`.cpu()` / `.to("cpu")`).
- Impact: process-level native termination (no Python exception), creating
  risk of lost diagnostics and false attribution to compiler/runtime changes.

## Reproduction Harnesses Added

Two deterministic, standalone harnesses were added to isolate and classify
failures:

1. `python/test/backend/metal_mps_transfer_stress.py`
   - Minimal compute + forced MPS→CPU transfer stress.
   - Deterministic seed, shape, and iteration control.
   - CPU and MPS modes.
2. `python/test/backend/metal_mps_project_flow_stress.py`
   - Mirrors Triton runtime flow using `torch.mps.compile_shader`.
   - Deterministic CPU reference maintained in parallel for correctness drift.
   - CPU and MPS modes.

Both harnesses emit startup banners with:
- Python version
- torch version
- macOS version
- selected mode
- MPS built/available status

Both hard-fail early if the requested mode is unavailable.

## Crash-Safe Artifact Retention

Harness runs persist timestamped artifacts in:

- `artifacts/metal-harness-runs/<timestamp>_<tag>_<mode>/`

Each run stores:
- `config.json`
- `environment.json`
- `events.jsonl`
- `run_state.json`
- `last_successful_checkpoint.txt`
- `summary.json` (on graceful success/failure)
- `python_exception.txt` (on Python exceptions)

Checkpoints are written immediately before and after risky boundaries:
- `transfer_to_cpu_pre`
- `transfer_to_cpu_post`
- `cleanup_sync_pre`
- `cleanup_sync_post`

This allows post-mortem boundary localization even when native crashes bypass
Python exception handling.

## Reproduction Steps

Run inside workspace `.venv`:

```bash
source .venv/bin/activate

# CPU baselines
python python/test/backend/metal_mps_transfer_stress.py --mode cpu --iters 64 --shape 8192 --transfer-every 8 --tag cpu-transfer-baseline
python python/test/backend/metal_mps_project_flow_stress.py --mode cpu --iters 32 --shape 4096 --transfer-every 8 --tag cpu-project-baseline

# MPS stress variants
python python/test/backend/metal_mps_transfer_stress.py --mode mps --iters 1500 --shape 65536 --transfer-every 1 --sync-before-transfer --sync-after-transfer --tag mps-transfer-stress
python python/test/backend/metal_mps_project_flow_stress.py --mode mps --iters 800 --shape 65536 --transfer-every 1 --sync-before-transfer --sync-after-transfer --tag mps-project-stress

# Aggressive no-extra-sync transfer stress
python python/test/backend/metal_mps_transfer_stress.py --mode mps --iters 3000 --shape 131072 --transfer-every 1 --tag mps-transfer-nosync
```

## CPU vs MPS Behavior Comparison (Current Session)

| Run Tag | Mode | Result | Notes |
| --- | --- | --- | --- |
| `cpu-transfer-baseline` | CPU | Success | Deterministic transfer loop stable |
| `cpu-project-baseline` | CPU | Success | Deterministic project-flow baseline stable |
| `mps-transfer-stress` | MPS | Success | 1500 transfer boundaries, no native crash |
| `mps-project-stress` | MPS | Success | 800 shader-launch + transfer boundaries, no native crash |
| `mps-transfer-nosync` | MPS | Success | 3000 transfer boundaries, no extra syncs |

Representative summaries are available in `summary.json` files under each run
directory.

## First Identified Failing Boundary

- In this session: **no failure reproduced**.
- Instrumented boundaries now distinguish:
  - `compute_*` -> compute-time crash
  - `transfer_to_cpu_*` -> transfer/sync crash
  - `cleanup_sync_*` -> teardown crash
- For future native crashes, `last_successful_checkpoint.txt` and
  `run_state.json` provide the first failing boundary by exclusion.

## Confidence Assessment

- Current confidence (this session): **moderate** that the prior crash is
  runtime/backend instability rather than compiler/codegen logic.
  - Rationale:
    - Historical signature points to native async completion cleanup.
    - Crashes occur outside Python exception model.
    - New minimal harnesses can reproduce transfer/cleanup pressure without
      requiring full Triton compilation.
- Current confidence against immediate codegen regression: **moderate**.
  - Rationale:
    - CPU/MPS deterministic harnesses both complete successfully under high
      transfer pressure in this run.
    - No crash reproduced in project-flow mirror path (`torch.mps.compile_shader`).

## Workarounds Added

1. Deterministic mode-split harnesses (`cpu`/`mps`) to prevent false
   attribution and to separate backend-runtime failures from compiler issues.
2. Explicit optional synchronization flags around transfer boundaries:
   - `--sync-before-transfer`
   - `--sync-after-transfer`
3. Structured checkpointing before risky boundaries to preserve diagnostics when
   a native crash terminates the interpreter.
4. Smoke-level integration in `scripts/test_metal_smoke.py` to exercise both
   harnesses in CPU and MPS modes during routine validation.

Tradeoff:
- Frequent fsync/checkpoint writes increase harness overhead, but this is
  intentional for diagnostic resilience in crash scenarios.
