# Metal Backend Release Audit

Date: 2026-03-06
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
   - Result: `449 passed, 1 skipped`
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
8. `PYTHONPATH=python .venv/bin/python scripts/metal_release_checks.py --python .venv/bin/python --tag merge-readiness-r3`
   - Result: success
   - Includes backend tests, smoke, throughput guard, cross-backend numerics,
     AOT unit/runtime, and AOT collection gate.

Release-check artifacts:

- `artifacts/metal-release-checks/20260306-173946_merge-readiness-r3/summary.json`

## Hardening Changes Included in This Audit Window

- Added consolidated release gate runner:
  - `scripts/metal_release_checks.py`
  - Structured logs, per-check cache isolation, timeout handling, optional soak.
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
- Dedicated hosted/self-hosted Metal CI lanes exist, but the primary
   integration workflow does not yet make every Metal validation lane a required
   gate.

What is still missing for a stricter "fully first-class" release bar:

1. Full Metal-native simdgroup/matrix-core matmul acceleration integration.
   The shared `accelerate_matmul` pass remains a non-CUDA no-op, while the
   Metal-specific strategy/lowering path exists but is not yet fully wired into
   the pass pipeline and performance-tuned across families.
2. Automated throughput baselines/regression gates across more Apple GPU
   families than the locally validated Apple M3 Pro lane.
3. fp8/int8 matmul-class runtime validation breadth.
4. HIP-backed cross-backend numerical comparison parity.
5. Always-on Metal CI gating for GPU/throughput/soak coverage on provisioned
   Apple hardware, rather than the current opt-in self-hosted lanes.

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
