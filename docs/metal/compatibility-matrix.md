# Metal Backend Compatibility Matrix

Last updated: 2026-07-03

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

## Known Limitations

| Limitation | Status | Workaround |
|-----------|--------|------------|
| Warp specialization (Hopper async-warp model) | **N/A** | No Metal hardware equivalent; architectural impossibility |
| TMA / Tensor Memory Access | **N/A** | NVIDIA Hopper-specific; no Metal DMA engine |
| Inline assembly | **N/A** | MSL has no inline assembly mechanism |
| Fence insertion pass (dual-proxy ordering) | **N/A** | NVIDIA Hopper-specific; Metal barriers are sufficient |
| Throughput guardrails across Apple7/8/9 are not yet automated | **Partial** | CI workflow includes throughput guardrail job stub; requires self-hosted Metal GPU runner |
| fp8 and int8 matmul-class runtime validation | **Mitigated** | fp8e5m2 runtime matmul + int8 blocked matmul + boundary saturation tests added; FP8 software converters available |
| Cross-backend CUDA/HIP numerical comparison harness is not yet in place | **Open** | Use deterministic CPU-reference validation |

## Execution Modes

| Mode | Requirements | Status |
|------|-------------|--------|
| torch.mps (preferred) | PyTorch with MPS | ✅ Default |
| PyObjC metallib | `pyobjc-framework-Metal` | ✅ Fallback |
| Compile-only | Xcode CLI tools | ✅ Always works |

## Validated Branch Snapshot

Validated on branch `feat/metal-support` (2026-07-03):

- `PYTHONPATH=python .venv/bin/python -m pytest -q python/test/backend/test_metal_backend.py python/test/backend/test_ir_types.py`
  - `609 passed, 1 skipped`
  - Skipped: 1× fp8e4b15 matmul pipeline (unsupported format)
- `PYTHONPATH=python .venv/bin/python scripts/test_metal_smoke.py`
  - all smoke checks passed (including transfer/project/training harnesses in CPU and MPS modes)
- `PYTHONPATH=python .venv/bin/python -m pytest -q python/test/unit/tools/test_aot_metal.py`
  - `1 passed`

## Environment Variables

| Variable | Values | Description |
|----------|--------|-------------|
| `TRITON_METAL_DEBUG` | `1`, `true`, `yes` | Enable verbose compile diagnostics |
| `TRITON_CACHE_DIR` | path | Override default cache/artifact directory (~/.triton) |
| `TRITON_METAL_PREFER_TORCH_MPS` | `0` / `1` (default: `1`) | Prefer torch.mps execution path when available; set to `0` to force PyObjC fallback |
| `MTL_DEBUG_LAYER` | `1` | Enable Apple's Metal validation layer |
