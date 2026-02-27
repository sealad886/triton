# Metal Backend Compatibility Matrix

Last updated: 2026-02-24

## Supported Configurations

| Component | Minimum | Recommended | Notes |
|-----------|---------|-------------|-------|
| macOS | 14.0 (Sonoma) | 15.0+ | Metal 3.0 API required |
| Xcode CLI Tools | 15.0 | 16.0+ | `xcrun metal`/`xcrun metallib` required |
| Python | 3.10 | 3.12 | 3.9 deprecated |
| PyTorch | 2.1.0 | 2.10+ | `torch.mps.compile_shader` preferred path |
| Apple Silicon | M1 (apple7) | M3+ (apple9) | bf16 requires apple9+ |

## GPU Family Feature Support

| Feature | apple7 (M1) | apple8 (M2) | apple9 (M3) |
|---------|-------------|-------------|-------------|
| fp32 compute | ✅ | ✅ | ✅ |
| fp16 compute | ✅ | ✅ | ✅ |
| bf16 compute | ❌ | ❌ | ✅ |
| simdgroup_matrix | ✅ | ✅ | ✅ |
| Shared memory (32KB) | ✅ | ✅ | ✅ |
| Threadgroup size (1024) | ✅ | ✅ | ✅ |

## Known Limitations

| Limitation | Status | Workaround |
|-----------|--------|------------|
| Matmul uses generic FMA path (no Metal-native simdgroup matmul integration yet) | Open | Use validated blocked matmul configurations; profile tile sizes |
| `accelerate_matmul` is currently CUDA-only optimization and no-ops on Metal | Open | Correctness path remains active via FMA lowering |
| Throughput guardrails across Apple7/8/9 are not yet automated | Open | Run `scripts/metal_release_checks.py --soak` on target hardware |
| fp8 and int8 matmul-class runtime validation are incomplete | Open | Use fp16/bf16/fp32 validated paths for production |
| Cross-backend CUDA/HIP numerical comparison harness is not yet in place | Open | Use deterministic CPU-reference validation |

## Execution Modes

| Mode | Requirements | Status |
|------|-------------|--------|
| torch.mps (preferred) | PyTorch with MPS | ✅ Default |
| PyObjC metallib | `pyobjc-framework-Metal` | ✅ Fallback |
| Compile-only | Xcode CLI tools | ✅ Always works |

## Validated Branch Snapshot

Validated on branch `feat/metal-support` (2026-02-24):

- `PYTHONPATH=python .venv/bin/python -m pytest -q python/test/backend/test_metal_backend.py`
  - `259 passed`
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
