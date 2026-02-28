# Metal Backend Parity — Implementation Plan

**Date**: 2026-02-28
**Branch**: `feat/metal-support`
**Baseline**: 531 passed, 5 skipped
**Final**: 609 passed, 1 skipped (2026-07-03)
**Status**: ✅ COMPLETE — All 14 phases finished

---

## Scope Assessment

### In-Scope (22 items)

Every gap identified in the parity analysis is in-scope **except** items that
require hardware or upstream APIs that do not exist on the Metal platform. Being
difficult or requiring significant effort is **not** grounds for exclusion.

### Out of Scope (with justification)

| # | Item | Reason |
|---|------|--------|
| — | **Warp specialization** (`add_warp_specialize_to_llvm`) | Metal has no hardware equivalent to NVIDIA's warp-specialization (Hopper async-warp model). There is no Metal API for independent warp scheduling. This is an architectural impossibility, not a difficulty issue. |
| — | **TMA / Tensor Memory Access** (`add_tma_lowering`, `TensorPtrOpsToLLVM`) | TMA is NVIDIA-specific hardware (Hopper+). Metal has no equivalent DMA engine or tensor descriptor hardware. Block pointer lowering has no hardware target. |
| — | **Tensor Memory** (`add_allocate_tensor_memory`, `TensorMemoryToLLVM`) | Blackwell-specific tensor memory. No Metal equivalent. |
| — | **Cluster operations** (`ClusterOpsToLLVM`, `add_plan_cta`) | NVIDIA multi-SM cluster model. Metal threadgroups do not span multiple GPU cores cooperatively. |
| — | **FP4 / MXFP** (`Fp4ToFpOpToLLVM`, `UpcastMXFPToLLVM`) | Apple Silicon does not support FP4 natively. No hardware target. |
| — | **Cooperative grid launch** | Metal's `dispatchThreads` does not support inter-threadgroup synchronization. No hardware equivalent to CUDA cooperative groups. |
| — | **Inline assembly** | MSL has no inline assembly mechanism. |
| — | **NVVM dialect lowering** (`add_nvvm_to_llvm`) | NVIDIA-specific. Metal already has its own artifact lowering. |
| — | **AMD-specific passes** (buffer ops, in-thread transpose, block pingpong, hoist/sink layout, epilogue opt) | AMD CDNA/RDNA-specific. No Metal relevance. |
| — | **Occupancy query** (`cuOccupancyMaxActiveClusters`) | No Metal API for compute unit occupancy. |

### Decision rationale

The above items are excluded because **the target hardware lacks the required
capability**. Every other gap — including the structurally hard regex translator
issue — remains in scope for incremental improvement.

---

## Implementation Order

Tasks are ordered by: (1) unblocking skipped tests, (2) correctness, (3)
performance, (4) developer experience, (5) completeness.

### Phase 1 — Unblock Skipped Tests (translator fixes)

| Task | Description | Files | Est. |
|------|-------------|-------|------|
| **1.1** | Add `samesign` icmp predicate to MSL translator | `compiler.py` | S |
| **1.2** | Add vector select (`select <N x i1>`) to MSL translator | `compiler.py` | S |
| **1.3** | Update skipped tests → expect pass; add scan/transpose coverage | `test_metal_backend.py` | S |

**Gate**: All 5 previously-skipped tests should pass. Target: 535+ passed, ≤1 skipped.

### Phase 2 — Atomic Operations

| Task | Description | Files | Est. |
|------|-------------|-------|------|
| **2.1** | Implement `atomic_rmw` MSL translation (add, max, min, and, or, xor, xchg) | `compiler.py` | M |
| **2.2** | Verify `atomic_cmpxchg` translation works | `compiler.py` | S |
| **2.3** | Un-skip atomic tests; add broader atomic test coverage | `test_metal_backend.py` | S |

**Gate**: `tl.atomic_add`, `tl.atomic_max` work end-to-end. Atomic tests pass.

### Phase 3 — Memory Buffer Pool (driver performance)

| Task | Description | Files | Est. |
|------|-------------|-------|------|
| **3.1** | Implement `MetalBufferPool` class with size-bucketed reuse | `driver.py` | M |
| **3.2** | Integrate pool into launch path (replace per-launch `newBufferWithBytes`) | `driver.py` | M |
| **3.3** | Add pool stats / diagnostics | `driver.py` | S |
| **3.4** | Tests for buffer pool behavior | `test_metal_backend.py` | S |

**Gate**: Repeated kernel launches reuse buffers. No memory leaks.

### Phase 4 — Barrier & Fence Lowering (C++)

| Task | Description | Files | Est. |
|------|-------------|-------|------|
| **4.1** | Create `BarrierOpToLLVM.cpp` for Metal (threadgroup_barrier) | `TritonMetalGPUToLLVM/` | M |
| **4.2** | Register the new lowering pass and remove Python-side barrier workaround if applicable | `TritonGPUToLLVM.cpp`, `compiler.py` | M |
| **4.3** | Tests for barrier lowering correctness | `test_metal_backend.py` | S |

**Gate**: Barrier ops lower through C++ pass. Build + tests pass.

### Phase 5 — Backend-Specific ConvertLayoutOp & SPMDOp

| Task | Description | Files | Est. |
|------|-------------|-------|------|
| **5.1** | Create Metal-specific `ConvertLayoutOpToLLVM` (shared↔register for MetalSimdgroup) | `TritonMetalGPUToLLVM/` | L |
| **5.2** | Create Metal-specific `SPMDOpToLLVM` (program_id, num_programs) | `TritonMetalGPUToLLVM/` | M |
| **5.3** | Register passes and verify in pipeline | `TritonGPUToLLVM.cpp` | S |
| **5.4** | Tests | `test_metal_backend.py` | S |

**Gate**: Layout conversions and SPMD ops use Metal-specific lowering. Tests pass.

### Phase 6 — Loop Scheduling & Fusion

| Task | Description | Files | Est. |
|------|-------------|-------|------|
| **6.1** | Add `add_schedule_loops` pass to Metal pipeline (use generic TritonGPU pass) | `compiler.py` | S |
| **6.2** | Add `add_fuse_nested_loops` pass to Metal pipeline | `compiler.py` | S |
| **6.3** | Add `add_combine_tensor_select_and_if` pass | `compiler.py` | S |
| **6.4** | Verify no regressions; check generated IR quality | tests | S |

**Gate**: Passes run without error. All tests pass.

### Phase 7 — GPU Profiling

| Task | Description | Files | Est. |
|------|-------------|-------|------|
| **7.1** | Implement `MTLCounterSampleBuffer`-based GPU timing | `driver.py` or `driver.c` | L |
| **7.2** | Wire GPU timestamps into `do_bench` path | `driver.py` | M |
| **7.3** | Fallback to host-side timing when counter sampling unavailable | `driver.py` | S |
| **7.4** | Tests for profiling accuracy | `test_metal_backend.py` | S |

**Gate**: `do_bench` reports GPU-side timings on supported devices.

### Phase 8 — MetalGPU Dialect Operations

| Task | Description | Files | Est. |
|------|-------------|-------|------|
| **8.1** | Define initial MetalGPU ops in `MetalGPUOps.td` (e.g., `mtlg.barrier`, `mtlg.simdgroup_shuffle`) | `include/Dialect/MetalGPU/IR/` | M |
| **8.2** | Implement op definitions and verifiers | `lib/Dialect/MetalGPU/IR/` | M |
| **8.3** | Add lowering from MetalGPU ops to LLVM | `TritonMetalGPUToLLVM/` | M |
| **8.4** | Tests | tests | S |

**Gate**: At least 2-3 MetalGPU dialect ops defined, verified, and lowered.

### Phase 9 — FP Sanitizer & Debug Tooling

| Task | Description | Files | Est. |
|------|-------------|-------|------|
| **9.1** | Wire `add_fp_sanitizer` pass into Metal pipeline (when enabled) | `compiler.py` | S |
| **9.2** | Add `add_triton_licm` to TTIR phase (already in TTGIR) | `compiler.py` | S |
| **9.3** | Tests for sanitizer mode | `test_metal_backend.py` | S |

**Gate**: FP sanitizer mode works. LICM in both phases.

### Phase 10 — Multi-Device Support

| Task | Description | Files | Est. |
|------|-------------|-------|------|
| **10.1** | Remove single-device restriction; enumerate all Metal devices | `driver.py` | M |
| **10.2** | Route `device_id` to correct `MTLDevice` | `driver.py` | M |
| **10.3** | Update `get_device_properties` for multi-device | `driver.py` | S |
| **10.4** | Tests (may require mocking for single-GPU systems) | `test_metal_backend.py` | S |

**Gate**: `device_id=1` works on systems with external GPUs.

### Phase 11 — Register/Spill Reporting

| Task | Description | Files | Est. |
|------|-------------|-------|------|
| **11.1** | Parse `xcrun metal` compiler output for register/spill stats | `compiler.py` or `driver.py` | M |
| **11.2** | Return real values from `load_binary` instead of `(0, 0)` | `driver.py` | S |
| **11.3** | Tests | `test_metal_backend.py` | S |

**Gate**: `n_regs` and `n_spills` return non-zero values when available.

### Phase 12 — Fence Insertion Pass ✅ N/A

**Status**: Not applicable to Metal. No implementation needed.

**Research conclusion** (completed 2026-02-28):
- `TritonGPUFenceInsertion` and `TritonGPUProxyFenceInsertion` are NVIDIA
  Hopper-specific (compute capability ≥ 90). Both early-return for lower arches.
- They solve async-proxy ↔ generic-proxy ordering for TMA + WGMMA operations.
- Apple GPUs have **no dual-proxy memory model**, **no TMA**, **no WGMMA**.
- Metal's existing `barrier_pass.py` + MSL translator fence handling (`mem_none`,
  `mem_device`, `mem_threadgroup`) fully covers Metal's memory ordering needs.

**Gate**: Decision documented. No pass needed.

### Phase 13 — Test Coverage Expansion

| Task | Description | Files | Est. |
|------|-------------|-------|------|
| **13.1** | Add histogram tests | `test_metal_backend.py` | S |
| **13.2** | Add join/split/interleave tests | `test_metal_backend.py` | S |
| **13.3** | Add cat/cat_nd tests | `test_metal_backend.py` | S |
| **13.4** | Add 3D matmul runtime test | `test_metal_backend.py` | S |
| **13.5** | Add propagate_nan / clamp tests | `test_metal_backend.py` | S |
| **13.6** | Add num_programs / program_id tests | `test_metal_backend.py` | S |
| **13.7** | Add random number generation tests | `test_metal_backend.py` | S |
| **13.8** | Add type conversion coverage tests | `test_metal_backend.py` | S |

**Gate**: At least 15 new tests added. All pass or are explicitly documented as blocked.

### Phase 14 — Documentation & Final Validation

| Task | Description | Files | Est. |
|------|-------------|-------|------|
| **14.1** | Update `docs/metal/backend.md` with all new features | `docs/metal/backend.md` | M |
| **14.2** | Update `docs/metal/compatibility-matrix.md` with final test counts | `docs/metal/compatibility-matrix.md` | S |
| **14.3** | Update CONVENTIONS.md if conventions have changed | `docs/CONVENTIONS.md` | S |
| **14.4** | Final full test suite run + validation | all | S |

**Gate**: All docs current. Final test count documented.

---

## Size Estimates

| Size | Meaning |
|------|---------|
| **S** | < 1 hour: small change, clear implementation |
| **M** | 1-4 hours: moderate complexity, may need iteration |
| **L** | 4-8 hours: significant new code or deep investigation |

---

## Commit Strategy

- One commit per completed phase (or sub-phase for large ones)
- Conventional commit format: `feat(metal): <description>`
- Each commit must leave tests green
- Documentation updated inline with implementation

---

## Risk Register

| Risk | Mitigation |
|------|-----------|
| `samesign`/vector-select fix may cause regressions in other IR patterns | Run full test suite after each translator change |
| Atomic lowering may expose other missing IR patterns | Use best_effort mode for initial testing, then harden |
| Buffer pool may hide memory bugs | Add pool diagnostics and leak detection |
| GPU profiling API may not be available on all macOS versions | Graceful fallback to host-side timing |
| Loop scheduling/fusion passes may not apply cleanly to Metal encoding | Test with representative kernels; document if passes no-op |
| MetalGPU dialect ops may conflict with existing generic handling | Start with ops that have clear Metal-specific semantics only |
