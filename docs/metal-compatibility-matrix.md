# Metal Backend Compatibility Matrix

## Supported Configurations

| Component | Minimum | Recommended | Notes |
|-----------|---------|-------------|-------|
| macOS | 14.0 (Sonoma) | 15.0+ | Metal 3.0 API required |
| Xcode CLI Tools | 15.0 | 16.0+ | xcrun metal compiler |
| Python | 3.10 | 3.12 | 3.9 deprecated |
| PyTorch | 2.1.0 | 2.5+ | torch.mps.compile_shader support |
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
| FMA-only matmul (no simdgroup_matrix) | Planned | Use smaller tile sizes |
| accelerate_matmul pass disabled | Planned | FMA path works correctly |
| No async stream support | Planned | Sequential execution works |
| fp16 matmul xfail in FMA path | Known | Use fp32 matmul path |

## Execution Modes

| Mode | Requirements | Status |
|------|-------------|--------|
| torch.mps (preferred) | PyTorch with MPS | ✅ Default |
| PyObjC metallib | pyobjc-framework-Metal | ✅ Fallback |
| Compile-only | Xcode CLI tools | ✅ Always works |

## Environment Variables

| Variable | Values | Description |
|----------|--------|-------------|
| `TRITON_METAL_DEBUG` | `1`, `true`, `yes` | Enable verbose compile diagnostics |
| `TRITON_CACHE_DIR` | path | Override default cache/artifact directory (~/.triton) |
| `MTL_DEBUG_LAYER` | `1` | Enable Apple's Metal validation layer |
