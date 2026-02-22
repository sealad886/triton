# Metal Backend for Triton

This document describes the Metal backend for Triton, which enables running
Triton programs on Apple Silicon GPUs via the Metal API.

## Overview

The Metal backend adds Apple GPU support to Triton by:

1. Lowering Triton IR through the standard TTIR → TTGIR → LLVM IR pipeline
2. Translating LLVM IR to Metal Shading Language (MSL)
3. Compiling MSL to `.metallib` binaries via `xcrun metal` / `xcrun metallib`
4. Loading and dispatching kernels through `torch.mps.compile_shader`
   (runtime path) and `.metallib` tooling/PyObjC utilities

## Requirements

- **macOS 14.0+** (Sonoma or later)
- **Apple Silicon** (M1 or later) or discrete AMD GPU with Metal support
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
```

## GPU Family Mapping

| Apple Silicon | GPU Family | Metal Feature Set |
|--------------|------------|-------------------|
| M1           | apple7     | Metal 2.4         |
| M2           | apple8     | Metal 2.5         |
| M3, M4       | apple9     | Metal 2.6+        |

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
  are fully handled; fp8 variants and aggregate types remain under
  consideration for future work.
- **Tensor cores**: Apple's matrix multiply accelerator is not yet
  integrated into the pass pipeline.
- **CI automation**: automated CI artifact upload/reporting for crash harness
  runs is not yet wired.

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
```

### Building the native extension

The `driver.c` file is compiled as an Objective-C source linking against
the Metal and Foundation frameworks. This happens automatically during
`pip install -e .` on macOS.
