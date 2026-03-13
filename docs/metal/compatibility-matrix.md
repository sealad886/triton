# Metal Backend Compatibility Matrix

Last updated: 2026-03-13

## Supported Configurations

| Component | Minimum | Recommended | Notes |
|-----------|---------|-------------|-------|
| macOS | 14.0 (Sonoma) | 15.0+ | Metal 3.0 API required |
| Xcode CLI Tools | 15.0 | 16.0+ | `xcrun metal`/`xcrun metallib` required |
| Python | 3.10 | 3.12 | 3.9 deprecated |
| PyTorch | 2.1.0 | 2.10+ | `torch.mps.compile_shader` preferred path |
| Apple Silicon | M1 (apple7) | M3+ (apple9) | bf16 requires apple9+ |

## GPU Family Feature Support

| Feature | apple7 (M1) | apple8 (M2) | apple9 (M3/M4) |
|---------|-------------|-------------|----------------|
| fp32 compute | ✅ | ✅ | ✅ |
| fp16 compute | ✅ | ✅ | ✅ |
| bf16 compute | ❌ | ❌ | ✅ |
| simdgroup_matrix (f32) | ✅ | ✅ | ✅ |
| simdgroup_matrix (f16→f32) | ✅ | ✅ | ✅ |
| simdgroup_matrix (bf16→f32) | ❌ | ❌ | ✅ |
| Batched matmul (rank-3) | ✅ | ✅ | ✅ |
| Shared memory (32KB) | ✅ | ✅ | ✅ |
| Threadgroup size (1024) | ✅ | ✅ | ✅ |
| Device property reporting | ✅ | ✅ | ✅ |
| Atomic operations (add/max/min/xor/or/and/xchg) | ✅ | ✅ | ✅ |
| Scan operations (cumsum) | ✅ | ✅ | ✅ |
| Transpose (tl.trans) | ✅ | ✅ | ✅ |
| Barrier/fence lowering (C++) | ✅ | ✅ | ✅ |
| MetalGPU dialect ops | ✅ | ✅ | ✅ |
| FP sanitizer | ✅ | ✅ | ✅ |
| GPU profiling (timing) | ✅ | ✅ | ✅ |
| Register estimation | ✅ | ✅ | ✅ |
| Buffer pool reuse | ✅ | ✅ | ✅ |
| RNG (Philox CBRNG) | ✅ | ✅ | ✅ |
| Histogram | ✅ | ✅ | ✅ |
| Join/split/interleave | ✅ | ✅ | ✅ |
| Clamp + propagate_nan | ✅ | ✅ | ✅ |
| 3D grid launch | ✅ | ✅ | ✅ |
| Direct GPU→Metal lowering (no NVVM) | ✅ | ✅ | ✅ |
| Concurrency sanitizer (consan) | ✅ | ✅ | ✅ |
| Profile scratch metadata | ✅ | ✅ | ✅ |
| f32 dot TF32 opt-out | ✅ | ✅ | ✅ |
| Async copy guard assertion | ✅ | ✅ | ✅ |

## Known Limitations

| Limitation | Status | Workaround |
|-----------|--------|------------|
| Warp specialization (Hopper async-warp model) | **N/A** | No Metal hardware equivalent; architectural impossibility |
| TMA / Tensor Memory Access | **N/A** | NVIDIA Hopper-specific; no Metal DMA engine |
| Inline assembly | **N/A** | MSL has no inline assembly mechanism |
| Fence insertion pass (dual-proxy ordering) | **N/A** | NVIDIA Hopper-specific; Metal barriers are sufficient |
| Device-scope atomic ordering stronger than relaxed | **Limited by Metal** | Metal lowering collapses device atomics to relaxed ordering semantics |
| Throughput guardrails are automated in the default local release gate and self-hosted CI, but cross-family Apple7/8/9 coverage is still partial | **Partial** | Run `scripts/metal_release_checks.py` locally and on target-family self-hosted runners |
| Multi-output scalar-reduction reuse kernels are not yet part of the preview release gate | **Partial** | Prefer single-output reductions or vector-output formulations for release-critical kernels |
| fp8 and int8 matmul-class runtime validation | **Mitigated** | fp8e5m2 runtime matmul + int8 blocked matmul + boundary saturation tests added; FP8 software converters available |
| Cross-backend numerics cover MPS with optional CUDA, but HIP parity is still open | **Partial** | Use `python/test/backend/metal_cross_backend_compare.py` plus deterministic CPU-reference validation |

## Execution Modes

| Mode | Requirements | Status |
|------|-------------|--------|
| torch.mps (preferred) | PyTorch with MPS | ✅ Default |
| PyObjC metallib | `pyobjc-framework-Metal` | ✅ Fallback |
| Compile-only | Xcode CLI tools | ✅ Always works |

## Validated Branch Snapshot

Validated on branch `feat/metal-support` (2026-03-13):

- `PYTHONPATH=python .venv/bin/python -m pytest -q python/test/backend/test_metal_backend.py`
  - `448 passed, 1 skipped`
  - Skipped: 1× fp8e4b15 matmul pipeline (unsupported format)
- `PYTHONPATH=python .venv/bin/python -m pytest -q python/test/backend/test_ir_types.py`
  - `162 passed`
- `PYTHONPATH=python .venv/bin/python scripts/test_metal_smoke.py`
  - all smoke checks passed (including transfer/project/training harnesses in CPU and MPS modes)
- `PYTHONPATH=python .venv/bin/python scripts/test_metal_reduction.py`
  - all reduction checks passed
- `PYTHONPATH=python .venv/bin/python scripts/metal_ci_compat_matrix.py --json`
  - `overall_passed=true`
- `PYTHONPATH=python .venv/bin/python -m pytest -q python/test/unit/tools/test_aot_metal.py`
  - `2 passed`
- `PYTHONPATH=python .venv/bin/python -m pytest -q python/test/backend/test_metal_backend.py python/test/backend/test_ir_types.py python/test/unit/tools/test_aot_metal.py`
  - `613 passed, 1 skipped`
- `PYTHONPATH=python .venv/bin/python scripts/metal_release_checks.py --profile hosted-ci --tag final-hosted-ci`
  - success (backend tests, `test_ir_types`, smoke, cross-backend numerics, AOT checks)
- `PYTHONPATH=python .venv/bin/python scripts/metal_release_checks.py --python .venv/bin/python --tag final-default`
  - success (backend tests, `test_ir_types`, smoke, throughput guard, cross-backend numerics, AOT checks)

Primary CI and release creation now run a reusable hosted Metal correctness
gate (`.github/workflows/metal-release-gate.yml`). Self-hosted Metal GPU,
throughput, and soak lanes remain supplemental until dedicated Apple runners are
always available.

## Environment Variables

| Variable | Values | Description |
|----------|--------|-------------|
| `TRITON_METAL_DEBUG` | `1`, `true`, `yes` | Enable verbose compile diagnostics |
| `TRITON_CACHE_DIR` | path | Override default cache/artifact directory (~/.triton) |
| `TRITON_METAL_PREFER_TORCH_MPS` | `0` / `1` (default: `1`) | Prefer torch.mps execution path when available; set to `0` to force PyObjC fallback |
| `MTL_DEBUG_LAYER` | `1` | Enable Apple's Metal validation layer |
