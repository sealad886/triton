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
   - Result: `448 passed, 1 skipped`
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
8. `PYTHONPATH=python .venv/bin/python scripts/metal_release_checks.py --profile hosted-ci --tag local-hosted-ci`
    - Result: success
    - Includes backend tests, `test_ir_types`, smoke, cross-backend numerics,
       AOT unit/runtime, and AOT collection gate.
9. `PYTHONPATH=python .venv/bin/python scripts/metal_release_checks.py --python .venv/bin/python --tag prod-gap-closeout`
   - Result: success
    - Includes backend tests, `test_ir_types`, smoke, throughput guard,
       cross-backend numerics, AOT unit/runtime, and AOT collection gate.

Release-check artifacts:

- `artifacts/metal-release-checks/20260313-191740_local-hosted-ci/summary.json`
- `artifacts/metal-release-checks/20260313-192019_prod-gap-closeout/summary.json`

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

Current status: **ready for upstream review as a preview backend** with
bounded, documented architecture limitations.

What is ready:

- End-to-end LLVM→MSL lowering is functional for broad ML kernel classes.
- Runtime contract coverage is in place for current Metal launch surface.
- Deterministic crash-classification harnesses are implemented and integrated.
- Release-check workflows are reproducible with persisted artifacts.
- Primary CI and release creation now run a reusable hosted Metal correctness
   gate, while dedicated hosted/self-hosted Metal workflows remain available for
   richer validation.

What is still missing for a stricter "fully first-class" release bar:

1. Full Metal-native simdgroup/matrix-core matmul acceleration integration.
   The shared `accelerate_matmul` pass remains a non-CUDA no-op, while the
   Metal-specific strategy/lowering path exists but is not yet fully wired into
   the pass pipeline and performance-tuned across families.
2. Automated throughput baselines/regression gates across more Apple GPU
   families than the locally validated Apple M3 Pro lane.
3. fp8/int8 matmul-class runtime validation breadth.
4. HIP-backed cross-backend numerical comparison parity.
5. Always-on Metal GPU/throughput/soak coverage on provisioned Apple hardware,
   rather than the current supplemental self-hosted lanes.

## Packaging/Release Recommendation

Logical stop point reached for this phase:

- Core correctness and stability guardrails are in place and passing.
- Release validation can be run via one command:
  - `PYTHONPATH=python .venv/bin/python scripts/metal_release_checks.py --soak`

Recommended release framing:

- Publish as Metal backend **preview** with explicit known limitations.
- Gate release candidate updates on `metal_release_checks.py` default suite.
- Use soak mode for pre-cut validation on target Apple hardware before release
  cuts.
