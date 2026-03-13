# Metal Backend Release Audit

Date: 2026-03-13
Branch: `feat/metal-support`

## Scope

This audit covers release-readiness for current Metal first-class work:

- Compiler/lowering correctness gates
- Runtime/harness stability gates
- AOT tooling gates for Metal paths
- Documentation consistency for user-facing behavior

## Validation Commands and Results

Executed in workspace `.venv`:

1. `PYTHONPATH=python .venv/bin/python -m pytest -q python/test/backend/test_metal_backend.py`
   - Result: `450 passed, 1 skipped`
2. `PYTHONPATH=python .venv/bin/python -m pytest -q python/test/backend/test_ir_types.py`
   - Result: `162 passed`
3. `PYTHONPATH=python .venv/bin/python scripts/test_metal_smoke.py`
   - Result: all checks passed
   - Includes CPU/MPS transfer, project-flow, and training-loop harness runs.
4. `PYTHONPATH=python .venv/bin/python scripts/test_metal_reduction.py`
   - Result: all reduction compilation checks passed
5. `PYTHONPATH=python .venv/bin/python scripts/metal_ci_compat_matrix.py --json`
   - Result: success (`overall_passed=true`)
6. `PYTHONPATH=python .venv/bin/python -m pytest -q python/test/unit/tools/test_aot_metal.py`
   - Result: `2 passed`
7. `PYTHONPATH=python .venv/bin/python -m pytest -q python/test/unit/tools/test_aot.py`
   - Result: `7 skipped` (expected non-CUDA/HIP environment)
8. `PYTHONPATH=python .venv/bin/python -m pytest -q python/test/backend/test_metal_backend.py python/test/backend/test_ir_types.py python/test/unit/tools/test_aot_metal.py`
   - Result: `614 passed, 1 skipped`
9. `PYTHONPATH=python .venv/bin/python scripts/metal_release_checks.py --profile hosted-ci --tag first-class-hosted-ci`
    - Result: success
    - Includes backend tests, `test_ir_types`, smoke, cross-backend numerics,
       AOT unit/runtime, and AOT collection gate.
10. `PYTHONPATH=python .venv/bin/python scripts/metal_matmul_throughput_guard.py --warmup 3 --reps 8 --tag first-class-recheck`
   - Result: success
   - Supplemental local throughput guardrail run for the current Apple Silicon lane.

Release-check artifacts:

- `artifacts/metal-release-checks/20260313-224758_first-class-hosted-ci/summary.json`
- `artifacts/metal-throughput-guard/20260313-225207_first-class-recheck/summary.json`

## Hardening Changes Included in This Audit Window

- Added consolidated release gate runner:
  - `scripts/metal_release_checks.py`
   - Structured logs, per-check cache isolation, timeout handling, optional
      soak, plus a hosted-ci profile for correctness-focused reusable gating.
- Wired a reusable hosted Metal correctness gate into primary CI and release
   creation:
   - `.github/workflows/metal-release-gate.yml`
   - `.github/workflows/ci.yml`
   - `.github/workflows/create_release.yml`
- Aligned shared Python backend/driver contracts with the actual compiler and
   benchmarking/runtime surfaces used across backends:
   - `python/triton/backends/compiler.py`
   - `python/triton/backends/driver.py`
- Fixed AOT tool test collection robustness:
  - `python/test/unit/tools/test_aot.py`
  - Initialized `test_utils_src` outside CUDA/HIP-only branches to avoid
    non-backend collection `NameError`.
- Updated and synchronized docs:
  - `docs/metal/backend.md`
  - `docs/metal/compatibility-matrix.md`
  - `docs/metal/first-class-support-plan.md`
  - `docs/metal/mps-crash-incident-report.md`

## Release Readiness Assessment

Current status: **supported on Apple Silicon for source builds** with a
mandatory hosted correctness gate and bounded, documented architecture
limitations.

This branch now meets the repo-controlled first-class support bar for Apple
Silicon correctness, tooling, and CI integration. It is **not** the same as
claiming strict universal fleet/performance parity across every Apple GPU
family, every backend-comparison lane, and every soak lane.

What is ready:

- End-to-end LLVM→MSL lowering is functional for broad ML kernel classes.
- Runtime contract coverage is in place for current Metal launch surface.
- Multi-output mean/variance reduction coverage is now part of the correctness
   contract, including odd-tail row coverage in isolated runtime checks.
- Deterministic crash-classification harnesses are implemented and integrated.
- Release-check workflows are reproducible with persisted artifacts.
- Primary CI and release creation now run a reusable hosted Metal correctness
   gate, while dedicated hosted/self-hosted Metal workflows remain available for
   richer validation.

What is still missing for a stricter "fully first-class" release bar:

1. Automated throughput baselines/regression gates across more Apple GPU
   families than the locally validated Apple M3 Pro lane.
2. HIP-backed cross-backend numerical comparison parity.
3. Always-on Metal GPU/throughput/soak coverage on provisioned Apple hardware,
   rather than the current supplemental self-hosted lanes.

Documented scope boundaries rather than current correctness blockers:

- Metal-native simdgroup/matrix-core acceleration is wired into the compiler
  pipeline for supported blocked layouts; the remaining work is cross-family
  tuning and performance guard coverage.
- `launch_cooperative_grid` is an explicit hard-fail on Metal, and
  `launch_pdl` is a compatibility no-op.
- `profile_scratch` metadata is preserved for future profiling flows, but the
  profiler-specific runtime path remains dormant on Metal.
- `fp8e4b15` is not advertised as a supported Metal dtype; the validated fp8
  path is `fp8e5`.

## Packaging/Release Recommendation

Logical stop point reached for this phase:

- Core correctness and stability guardrails are in place and passing.
- Release validation can be run via one command:
  - `PYTHONPATH=python .venv/bin/python scripts/metal_release_checks.py --soak`

Recommended release framing:

- Publish as a **supported Apple Silicon source-build backend** with explicit
   fleet/performance scope boundaries.
- Treat the reusable hosted correctness gate as the mandatory release bar.
- Use throughput guardrails and soak mode as supplemental pre-cut validation on
   target Apple hardware.
