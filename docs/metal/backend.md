# Metal Backend for Triton

This document describes the Metal backend for Triton, which enables running
Triton programs on Apple Silicon GPUs via the Metal API.

The current Metal path is a **supported, source-build backend** for Apple
Silicon, with mandatory hosted correctness gating in primary CI and release
workflows. It is not yet distributed as a general macOS wheel path.

The canonical machine-readable feature contract now lives in
`python/triton/backends/metal/capabilities.py` (mirrored through
`third_party/metal/backend/capabilities.py`). Use that snapshot for tooling,
CI, and tests instead of re-encoding launch/runtime support in multiple places.

## Overview

The Metal backend adds Apple GPU support to Triton by:

1. Lowering Triton IR through the standard TTIR → TTGIR → LLVM IR pipeline
2. Translating LLVM IR to Metal Shading Language (MSL)
3. Compiling MSL to `.metallib` binaries via `xcrun metal` / `xcrun metallib`
4. Loading and dispatching kernels through `torch.mps.compile_shader`
   (runtime path) and `.metallib` tooling/PyObjC utilities

## Requirements

- **macOS 14.0+** (Sonoma or later)
- **Apple Silicon** (M1 or later)
- **Xcode Command Line Tools** (provides `xcrun`, `metal`, `metallib`)
- **Python 3.10+**
- **PyObjC** (`pip install pyobjc-framework-Metal pyobjc-framework-Foundation`)

## Architecture

```text
┌──────────────┐
│  Triton IR   │
│   (TTIR)     │
└──────┬───────┘
       │ make_ttir()
┌──────┴───────┐
│  TritonGPU   │
│   (TTGIR)    │
└──────┬───────┘
       │ make_ttgir()
┌──────┴───────┐
│   LLVM IR    │
└──────┬───────┘
       │ make_metal_ir()
┌──────┴───────┐
│     MSL      │
│ (Metal SL)   │
└──────┬───────┘
       │ make_metallib() via xcrun
┌──────┴───────┐
│  .metallib   │
│   binary     │
└──────────────┘
```

### Compilation Pipeline

| Stage       | Input      | Output      | Tool            |
|-------------|-----------|-------------|-----------------|
| `ttir`      | Triton IR | Optimized IR| MLIR passes     |
| `ttgir`     | TTIR      | GPU IR      | MLIR passes     |
| `llir`      | TTGIR     | LLVM IR     | MLIR → LLVM     |
| `metal`     | LLVM IR   | MSL source  | Regex+codegen   |
| `metallib`  | MSL       | .metallib   | xcrun metal/lib |

### Runtime

The runtime has two execution/load paths:

- **Primary runtime path (`torch.mps.compile_shader`)**
  - Compiles generated MSL source at runtime and launches through torch MPS
    shader objects with tensor arguments bound directly.
- **Tooling/utility path (PyObjC + `.metallib`)**
  - Retained for backend smoke testing and lower-level validation flows.
  - Uses:
    - `MTLCreateSystemDefaultDevice()`
    - `newCommandQueue()`
    - `newLibraryWithURL:error:`
    - `newFunctionWithName:`
    - `newComputePipelineStateWithFunction:error:`

## File Structure

```text
third_party/metal/
├── CMakeLists.txt              # Build config for native extension
├── backend/
│   ├── __init__.py
│   ├── name.conf               # Backend name ("metal")
│   ├── compiler.py             # MetalBackend (BaseBackend impl)
│   ├── driver.py               # MetalDriver (DriverBase impl)
│   └── driver.c                # Native Metal utilities (Obj-C)
├── include/Dialect/MetalGPU/IR/
│   ├── MetalGPUDialect.td      # MetalGPU MLIR dialect definition
│   ├── MetalGPUOps.td          # Op definitions (barriers, shuffles)
│   └── MetalGPUAttrDefs.td     # Attribute definitions
└── lib/TritonMetalGPUToLLVM/
    ├── TritonGPUToLLVM.cpp     # Main conversion pass
    ├── GpuToMetalPatterns.cpp/.h # Direct GPU→Metal op lowering
    ├── TargetInfo.cpp/.h       # Metal target info (shuffles, atomics)
    ├── MetalGPUOpsToLLVM.cpp/.h # MetalGPU ops → LLVM lowering
    ├── NvidiaArtifactLowering.cpp/.h # Safety-net rewriter for residual NVVM artifacts
    └── Utility.cpp/.h          # Shared lowering utilities
```

## GPU Family Mapping

| Apple Silicon | GPU Family | Metal Feature Set | Notable Features |
|--------------|------------|-------------------|------------------|
| M1           | apple7     | Metal 2.4         | simdgroup matrix (f32, f16) |
| M2           | apple8     | Metal 2.5         | simdgroup matrix (f32, f16) |
| M3, M4       | apple9     | Metal 2.6+        | simdgroup matrix (f32, f16, bf16) |

## Usage

After building Triton with the Metal backend, it will be automatically
available when running on macOS with Apple Silicon:

```python
import triton
import triton.language as tl

@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)
```

## Current Limitations

- **Data type coverage is mostly complete**: common scalar/integer/float types
  are handled, including fp8e5m2 compile-path lowering for casts and dot/matmul
  kernels. fp8 runtime numerics and broader quantized-path validation still
  require additional coverage.
- **NVVM dialect decoupled**: GPU dialect ops (`gpu::ThreadIdOp`,
  `gpu::BarrierOp`) are now lowered directly to Metal extern calls via
  `GpuToMetalPatterns` without requiring the NVVM dialect as an intermediate.
  `NvidiaArtifactLowering` is retained as a safety net for residual
  `ttg.warp_id` and inline PTX asm patterns.
- **Concurrency sanitizer**: `consan` instrumentation mode is now wired
  through `parse_options` and `make_llir`. The pass emits device-side
  assertions; full runtime validation on Metal requires further testing.
- **Profiling scratch memory**: `profile_scratch_size` and
  `profile_scratch_align` metadata are now populated. The contract field is
  accepted as **metadata-only**; the driver currently discards the value until
  profiling instrumentation is implemented.
- **Launch contract extensions**: `launch_pdl` is accepted for ABI parity but
  remains a no-op. `launch_cooperative_grid` is accepted through the standard
  Metal dispatch path.
- **TF32 dot products**: Apple Silicon has no TF32 tensor cores;
  `add_f32_dot_tc(pm, False)` is explicitly called to opt out.
- **Warp specialization**: Metal has no hardware equivalent to NVIDIA's
  Hopper async-warp model. There is no Metal API for independent warp
  scheduling.
- **TMA / Tensor Memory Access**: NVIDIA-specific hardware (Hopper+). Metal
  has no equivalent DMA engine or tensor descriptor hardware.
- **Inline assembly**: MSL has no inline assembly mechanism.
- **Cross-backend numerics**: deterministic CPU-reference comparisons are
  implemented for MPS and optional CUDA via a dedicated harness and default
  release checks; HIP parity and broader multi-family CI coverage remain
  pending.

## Troubleshooting

### "torch.mps.compile_shader not available"

This error means your PyTorch version does not support MPS shader compilation.
- **Fix**: upgrade to PyTorch 2.1+ (`pip install --upgrade torch`).
- **Alternative**: the Metal backend will fall back to the PyObjC `.metallib`
  loading path automatically if `compile_shader` is unavailable.
- **Force the fallback path intentionally**: set
  `TRITON_METAL_PREFER_TORCH_MPS=0` to validate the PyObjC metallib runtime
  path even on hosts where `torch.mps.compile_shader` exists.

### "Unsupported LLVM IR" errors

The Metal translator handles a broad but not exhaustive set of LLVM IR
instructions. If you see unsupported-IR errors:
1. Set `TRITON_METAL_DEBUG=1` to get structured failure signatures and
   a detailed diagnostic report.
2. Check `~/.triton/metal_unsupported_ir.log` for the full list of
   unsupported instructions with context.
3. Consider using `best_effort=True` in `MetalOptions` to emit partial MSL
   with unsupported lines commented out.

### metallib compilation fails

Common causes:
- **Xcode CLI tools not installed**: run `xcode-select --install`.
- **Wrong SDK version**: ensure `xcrun metal --version` succeeds.
- **Syntax errors in generated MSL**: set `TRITON_METAL_DEBUG=1` and inspect
  the generated MSL source for issues.

### MPS instability / segfaults

- Use CPU mode (`--mode cpu`) for the stress harnesses to isolate whether
  the issue is in Metal dispatch or computation logic.
- Enable the Metal validation layer via Xcode's GPU diagnostics or by setting
  `MTL_DEBUG_LAYER=1` in your environment.
- Check `docs/metal/mps-crash-incident-report.md` for known patterns.

## Performance Tuning

### Block sizes

Start with `BLOCK_SIZE=128`. Tune downward for kernels with high register
pressure. Apple Silicon has 32KB shared memory per threadgroup; exceeding
this silently degrades performance via memory spilling.

### Memory access

- Use shared memory (`threadgroup` address space) for reductions (`tl.sum`,
  `tl.max`) — the Metal backend automatically maps `@global_smem` to
  threadgroup memory.
- Coalesce global memory loads: access patterns where consecutive threads
  read consecutive addresses perform best on Apple GPUs.

### Matrix Multiplication

The Metal backend uses Apple's simdgroup matrix intrinsics for
hardware-accelerated matrix multiplication. The `AccelerateMetalMatmul`
pass (`TritonMetalGPUAccelerateMatmul`) converts `tt.DotOp` with
`BlockedEncoding` to `MetalSimdgroupEncoding`, which the
`MetalSimdgroupDot` lowering then emits as simdgroup load/store/MMA
intrinsics.

**Supported configurations:**

| Operand Type | Accumulator | GPU Family | Notes |
|-------------|-------------|------------|-------|
| f32×f32     | f32         | apple7+    | Standard precision |
| f16×f16     | f32         | apple7+    | Mixed-precision, best performance |
| bf16×bf16   | f32         | apple9+    | M3/M4 only |

Operand A and B must have the same element type. The accumulator is
always f32. Matrices that don't meet the minimum size (16×16) or
alignment (multiples of 8 in M, N, K) fall back to the FMA path.

**Batched matmul** (rank-3 tensors) is supported: the lowering wraps
the 2D simdgroup MMA in an outer batch loop indexed by the batch
dimension.

**Tuning recommendations:**
- Use f16 operands with f32 accumulation for best throughput.
- Block sizes of 16×16 are a good starting point; 32×32 is supported
  given sufficient M, N dimensions.
- The pipeline includes a prefetch pass for loop-carried data movement
  optimization.

## Compatibility Notes

- **macOS 14+** required (Metal 3.0 API).
- **Xcode 15+** recommended for the `xcrun metal` compiler toolchain.
- **PyTorch 2.1+** for `torch.mps.compile_shader` support.
- **Apple GPU family**: `apple7` (M1) minimum. `apple9` (M3/M4) required
  for bf16 support.
- **Device properties**: per-chip memory bandwidth, clock rates, and GPU
  core counts are reported via `get_device_properties()` for all M1–M4
  and A-series chips.
- See [compatibility-matrix.md](compatibility-matrix.md) for the
  full compatibility matrix.

## Development

### Running tests

```bash
# Non-GPU tests (import/interface checks)
python -m pytest python/test/backend/test_metal_backend.py -v -k "not (Launch or Handle or Compilation)"

# Full tests (requires macOS + Apple Silicon)
python -m pytest python/test/backend/test_metal_backend.py -v

# Deterministic crash-classification harness (minimal transfer path)
python python/test/backend/metal_mps_transfer_stress.py --mode cpu --iters 256 --transfer-every 16
python python/test/backend/metal_mps_transfer_stress.py --mode mps --iters 4096 --transfer-every 1

# Deterministic crash-classification harness (runtime-flow mirror)
python python/test/backend/metal_mps_project_flow_stress.py --mode cpu --iters 256 --transfer-every 16
python python/test/backend/metal_mps_project_flow_stress.py --mode mps --iters 2048 --transfer-every 1

# Deterministic crash-classification harness (training-style optimizer loop)
python python/test/backend/metal_mps_training_loop_stress.py --mode cpu --iters 256 --transfer-every 16
python python/test/backend/metal_mps_training_loop_stress.py --mode mps --iters 2048 --transfer-every 1

# Optional: isolate cache state while validating to avoid stale-artifact reuse
TRITON_CACHE_DIR="$(mktemp -d /tmp/triton-metal-cache.XXXXXX)" python -m pytest -q python/test/backend/test_metal_backend.py

# Consolidated release checks (default suite)
python scripts/metal_release_checks.py

# Hosted correctness gate used by primary CI and release workflows
python scripts/metal_release_checks.py --profile hosted-ci

# Extended local soak checks (larger transfer/training stress)
python scripts/metal_release_checks.py --soak

# Throughput guardrails for representative GEMM shapes
python scripts/metal_matmul_throughput_guard.py

# Deterministic cross-backend numerics (CPU reference + MPS/CUDA where present)
python python/test/backend/metal_cross_backend_compare.py --backends mps,cuda
```

### Building the native extension

The `driver.c` file is compiled as an Objective-C source linking against
the Metal and Foundation frameworks. This happens automatically during
`pip install -e .` on macOS.
