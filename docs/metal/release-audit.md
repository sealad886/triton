# Metal Backend Release Audit

Date: 2026-02-24  
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
   - Result: `235 passed`
2. `PYTHONPATH=python .venv/bin/python scripts/test_metal_smoke.py`
   - Result: all checks passed
   - Includes CPU/MPS transfer, project-flow, and training-loop harness runs.
3. `PYTHONPATH=python .venv/bin/python -m pytest -q python/test/unit/tools/test_aot_metal.py`
   - Result: `1 passed`
4. `PYTHONPATH=python .venv/bin/python -m pytest -q python/test/unit/tools/test_aot.py`
   - Result: `7 skipped` (expected non-CUDA/HIP environment)
5. `PYTHONPATH=python .venv/bin/python scripts/metal_release_checks.py --python .venv/bin/python --tag local-release-gate`
   - Result: success
6. `PYTHONPATH=python .venv/bin/python scripts/metal_release_checks.py --python .venv/bin/python --soak --tag local-release-soak`
   - Result: success (includes sustained MPS transfer/project/training stress)

Release-check artifacts:

- `artifacts/metal-release-checks/20260224-070609_local-release-gate/summary.json`
- `artifacts/metal-release-checks/20260224-070639_local-release-soak/summary.json`

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

Current status: **conditionally ready** for Metal preview/experimental release.

What is ready:

- End-to-end LLVM→MSL lowering is functional for broad ML kernel classes.
- Runtime contract coverage is in place for current Metal launch surface.
- Deterministic crash-classification harnesses are implemented and integrated.
- Release-check and soak workflows are reproducible with persisted artifacts.

What is still missing for a stricter "fully first-class" release bar:

1. Metal-native simdgroup/matrix-core matmul acceleration strategy (current
   `accelerate_matmul` behavior is intentionally non-CUDA no-op for Metal).
2. Automated throughput baselines/regression gates across Apple GPU families.
3. fp8/int8 matmul-class runtime validation breadth.
4. Cross-backend CUDA/HIP numerical comparison harness.
5. CI automation for compatibility-matrix validation and soak gates.

## Packaging/Release Recommendation

Logical stop point reached for this phase:

- Core correctness and stability guardrails are in place and passing.
- Release validation can be run via one command:
  - `PYTHONPATH=python .venv/bin/python scripts/metal_release_checks.py --soak`

Recommended release framing:

- Publish as Metal backend **preview** with explicit known limitations.
- Gate release candidate updates on `metal_release_checks.py` default suite.
- Use soak mode for pre-cut validation on target Apple hardware.
