# Metal First-Class Support Plan

Last updated: 2026-02-22

## Objective

Bring the Metal backend to first-class parity with existing in-tree GPU backends
(NVIDIA and AMD) for Triton core flows:

- JIT compilation and execution
- Backend runtime interface contracts
- AOT compile/link tooling integration
- Test and CI coverage patterns

## Scope

In:
- `third_party/metal/**/{lib,language,backend,tools,python}`
- Core runtime/compiler integration points consumed by all backends
- Metal-focused tests and backend utility helpers
- Crash-resilient MPS execution harness and artifact collection for triage

Out (for this phase):
- New Metal architecture-specific optimization passes beyond current baseline

## Backend Parity Audit

| Area | NVIDIA | AMD | Metal (current) | Gap |
| --- | --- | --- | --- | --- |
| `third_party/<backend>/backend` | mature runtime/compiler pair | mature runtime/compiler pair | present, but runtime contract mismatch | High |
| `third_party/<backend>/language` | `cuda` extras + libdevice | `hip` extras + libdevice | minimal `metal` extras only | Medium |
| `third_party/<backend>/lib` | large conversion stack | large conversion + transforms | minimal conversion stack | Medium |
| `third_party/<backend>/tools` | `compile.*` + `link.h` | `compile.*` + `link.h` | missing | High |
| `third_party/<backend>/python` | root binding (`triton_nvidia.cc`) | `python/triton_amd.cc` | `python/triton_metal.cc` present | Low |
| Runtime `utils.load_binary` contract | matches JIT expectations | matches JIT expectations | aligned for source+metallib payloads | Low |
| Runtime `utils.get_device_properties` schema | includes expected keys | includes expected keys | missing shared-memory key | High |
| LLVM IR -> MSL backend stage | mature backend-specific lowering | mature backend-specific lowering | placeholder stub generator | High |
| Backend stage inspection hook | implemented | implemented | missing | Medium |
| Test utility backend helpers | cuda/hip helpers | hip helpers | no `is_metal` helper | Medium |
| AOT unit test behavior | supported | supported | not handled cleanly | Medium |
| Crash diagnostics harness | mature sanitizer/profiler ecosystem | mature sanitizer/profiler ecosystem | no deterministic MPS crash triage harness | High |

## First-Class Definition

Metal is considered first-class for this project phase when:

1. Metal backend satisfies Triton core runtime interfaces used by
   `CompiledKernel` and JIT launch paths.
2. Metal backend participates in backend inspection and tooling flows where
   other backends already integrate.
3. AOT/link tool paths do not fail due to missing Metal backend templates.
4. Tests either validate Metal-specific behavior or explicitly gate
   unsupported paths without accidental failures.

## Execution Plan (Living Checklist)

### Phase 1: Runtime Contract Alignment
- [x] Enable Objective-C language support in top-level CMake project for Metal
      backend sources.
- [x] Align `MetalUtils.load_binary` with Triton JIT call contract and return
      shape.
- [x] Route runtime launch through `torch.mps.compile_shader` so Tensor
      arguments bind at source instead of downstream script adaptation.
- [x] Add `MetalUtils.unload_module` and ensure no-op safety semantics.
- [x] Align Metal device property schema with expected shared-memory keys used
      by compiler/runtime guards.
- [x] Verify `MetalLauncher` invocation path handles loaded function objects
      consistently.
- [x] Expand scalar-cast conformance coverage for edge types (`u64` high values,
      fp8 variants, tuple argument flattening in complex signatures).

Acceptance:
- JIT initialization can invoke Metal `utils.load_binary` path without schema
  mismatch errors.

### Phase 2: Compiler Integration Parity
- [x] Add stages inspection hook wiring in Metal backend stage construction.
- [x] Ensure Metal backend hash/versioning accounts for backend SDK/compiler
      version signals.
- [x] Validate options parsing defaults against shared Triton knobs where
      applicable.

Acceptance:
- Metal backend responds to stage inspection hooks similarly to CUDA/HIP.

### Phase 3: Tooling Parity (`third_party/metal/tools`)
- [x] Add Metal AOT template prelude (`link.h`).
- [x] Add Metal compile templates (`compile.h` + implementation template).
- [x] Ensure generated types are compatible with linker header parser.
- [x] Update any core compile tooling helpers required for Metal template
      generation.

Acceptance:
- `triton.tools.compile` and `triton.tools.link` no longer fail due to missing
  Metal backend tool templates.

### Phase 4: Test and Validation
- [x] Add Metal runtime-contract tests (interface/schema behavior).
- [x] Add Metal compiler stage hook test coverage.
- [x] Update AOT unit tests to gate unsupported backend coverage explicitly
      where needed.
- [x] Run targeted backend test set and record outcomes.

Acceptance:
- Targeted backend tests pass (or skip with intentional reasons) for updated
  Metal integration points.

### Phase 5: LLVM IR -> MSL Lowering
- [x] Replace placeholder MSL stub generation with translation from lowered
      LLVM IR.
- [x] Add helper-call aware lowering (`__metal_get_*`,
      `__metal_predicated_ld/st_*`) into generated MSL.
- [x] Add CFG/dataflow lowering primitives for `phi`, floating-point compare
      predicates, and floating binary ops.
- [x] Add coverage for translation correctness and metallib compilation from
      lowered LLVM IR snippets.
- [x] Add intrinsic/math lowering support (`llvm.fma`, `llvm.fabs`,
      `llvm.maximum/minimum`, etc.) and unary `fneg`.
- [x] Harden MSL SSA-name sanitization to avoid collisions with Metal builtins.
- [x] Fix source-pass lowering parity for float tensor elementwise ops
      (`arith.addf/subf/mulf/divf`) in `ConvertTritonMetalGPUToLLVM`.
- [x] Fix compiler artifact loading for mixed Metal outputs (`.metal` source +
      `.metallib` binary) to avoid text decode faults in `CompiledKernel`.
- [x] Stabilize phi-heavy CFG lowering by hoisting SSA declarations across
      switch-case blocks in generated MSL.
- [x] Lower CUDA/OCML libdevice math calls (`__nv_*`, `__ocml_*`) to MSL
      builtins during LLVM->MSL call translation.
- [x] Materialize `@global_smem` in Metal LLVM conversion to prevent missing
      shared-memory base symbols during `ttg.convert_layout` lowering.
- [x] Lower shared-memory `@global_smem` pointer arithmetic and typed
      load/store patterns in LLVM->MSL (`getelementptr inbounds`, vector
      element ops, address-space-aware pointer casts).
- [x] Extend translation coverage for complex control-flow constructs
      (phi-heavy CFGs, uncommon intrinsic patterns) used by advanced kernels.
- [x] Resolve remaining dynamic-loop reduction lowering gap where kernels with
      `scf.for`-shaped reductions fail control-flow legalization
      (`failed to legalize operation 'cf.br'`) in `make_llir`.

Acceptance:
- Kernels that lower through the Metal LLVM pipeline compile through
  `make_metal_ir` -> `make_metallib` without placeholder stubs.

### Phase 6: MPS Crash Diagnostics Harness (Track B)
- [x] Add deterministic transfer-stress repro script with CPU/MPS mode split
      and explicit transfer/cleanup checkpoints.
- [x] Add deterministic project-flow repro script that uses
      `torch.mps.compile_shader` to mirror Triton Metal runtime launch path.
- [x] Add startup environment banner + hard-fail mode gating for unavailable
      MPS runs.
- [x] Add crash-safe artifact retention (config/env/state/events/checkpoints)
      with frequent flush/fsync.
- [x] Add explicit transfer boundary markers (`transfer_to_cpu_pre/post`) and
      teardown markers (`cleanup_sync_pre/post`) for failure classification.
- [x] Integrate harness execution into `scripts/test_metal_smoke.py` for
      repeated validation in both CPU and MPS modes.
- [ ] Add automated CI artifact upload/reporting for harness run directories.

Acceptance:
- Native crashes can be localized to compute, transfer/sync, or cleanup
  boundary using persisted run artifacts, even without Python exceptions.

## Risks and Mitigations

- Risk: MSL generation remains a stub for many kernels.
  - Mitigation: Explicitly scope this plan to runtime/tooling first-class
    integration and document compiler limitations.
- Risk: Type mapping changes affect AOT code generation compatibility.
  - Mitigation: add focused tests for mapping/parser compatibility and keep
    mapping backend-specific.
- Risk: macOS-only runtime code paths are hard to validate in non-mac
  environments.
  - Mitigation: structure tests with deterministic non-mac assertions for
    interface contracts.

## Progress Log

- 2026-02-21: Branch renamed from `feat/mlx-support` to
  `feat/metal-support`. Stale local `feat/mlx-support` ref removed by rename.
- 2026-02-21: Completed backend parity audit across
  `third_party/**/{lib,language,backend,tools,python}` and captured gaps in
  this plan.
- 2026-02-21: Implemented runtime contract fixes in Metal driver
  (`load_binary` dual signature support, `unload_module`, shared-memory
  property schema, launcher strictness).
- 2026-02-21: Added Metal compiler parity integration for stage inspection hook
  and default option handling.
- 2026-02-21: Added initial Metal AOT tooling templates under
  `third_party/metal/tools/metal` and updated compile template generation for
  Metal argument binding code.
- 2026-02-21: Reworked `ConvertTritonMetalGPUToLLVM` to enforce strict
  Triton->LLVM legality, add explicit Metal load/store lowering, and replace
  NVVM/PTX artifacts at source with Metal helper calls.
- 2026-02-21: Replaced `make_metal_ir` stub generation with real LLVM-IR-to-MSL
  translation for lowered helper-call based kernels; added reserved-name
  sanitization and parser robustness for LLVM signatures containing attributes.
- 2026-02-21: Added regression tests for helper-call translation and direct
  metallib compilation from lowered LLVM IR snippets.
- 2026-02-21: Switched Metal runtime execution path to compile/load generated
  MSL via `torch.mps.compile_shader`, with signature-aware argument flattening
  and constexpr filtering in the launcher to fix Tensor argument binding at the
  source backend layer.
- 2026-02-21: Extended LLVM-IR-to-MSL lowering with predecessor-aware branch
  translation and `phi` node lowering using a switch-based CFG state machine
  (Metal does not support `goto`/labels), plus `fcmp` and floating arithmetic
  lowering coverage.
- 2026-02-21: Added deterministic crash-classification harnesses:
  `metal_mps_transfer_stress.py` (minimal transfer path) and
  `metal_mps_project_flow_stress.py` (runtime-flow mirror via
  `torch.mps.compile_shader`), both with CPU/MPS mode split and checkpointed
  artifact logging.
- 2026-02-21: Integrated harness checks into `scripts/test_metal_smoke.py`
  to run CPU and MPS variants with explicit synchronization boundaries.
- 2026-02-21: Extended LLVM-IR-to-MSL lowering with intrinsic math support
  (`llvm.fma`, `llvm.fabs`, `llvm.maximum/minimum`, etc.), unary `fneg`, and
  identifier sanitization for Metal builtin collisions (e.g. `%fma`).
- 2026-02-21: Fixed source pass gap in
  `ConvertTritonMetalGPUToLLVM` by adding explicit float elementwise lowering
  patterns and arith expand patterns, unblocking full Triton vector-add compile
  through Metal backend.
- 2026-02-21: Fixed `CompiledKernel` artifact loader to treat `.metallib` as a
  binary artifact even when `binary_ext` is `metal`, preventing UTF-8 decode
  crashes during mixed source/binary artifact reads.
- 2026-02-21: Fixed CFG-scoped SSA lifetime bugs in LLVM->MSL lowering by
  predeclaring inferred SSA temporaries at function scope; backedge-phi loop
  kernels now compile to valid `.metallib`.
- 2026-02-21: Added CUDA/OCML libdevice compatibility lowering in
  LLVM->MSL translation (e.g. `__nv_expf/__nv_logf/__nv_sqrtf` to
  `exp/log/sqrt`), unblocking math kernels that still reference shared
  libdevice symbols.
- 2026-02-21: Added explicit `@global_smem` symbol materialization in
  `ConvertTritonMetalGPUToLLVM`, fixing the prior native assertion in
  `getStackPointer` during reduction-related `ttg.convert_layout` lowering.
- 2026-02-21: Reworked Metal shared-memory predicated load/store lowering to
  branch-free `select`-based form, removing malformed CFG generation in
  reduction conversion.
- 2026-02-21: Extended LLVM->MSL lowering for reduction-generated IR
  (`getelementptr ... @global_smem`, typed address-space load/store, vector-1
  extract/insert handling), enabling Triton reduction kernels without dynamic
  loops to compile to valid `.metallib`.
- 2026-02-21: Isolated current reduction blocker to dynamic-loop control-flow
  legalization (`cf.br` illegal in `ConvertControlFlowToLLVMPass`) for kernels
  that retain `scf.for` structure through `make_llir`.
- 2026-02-22: Added `normalize_label()` to handle quoted LLVM IR label names
  (e.g. `%"loop.header"`) in MSL translator branch/phi/label parsing.
- 2026-02-22: **Resolved `scf.for` dynamic-loop reduction blocker** by moving
  `add_scf_to_cf` pass before `metal.passes.ttgpuir.add_to_llvmir` in
  `make_llir`, matching NVIDIA/AMD pass ordering. The backend pass already
  populates `cf→LLVM` patterns internally; running `scf→cf` after left
  newly-created `cf.br` ops with partially-lowered types that the standalone
  `ConvertControlFlowToLLVMPass` could not legalize. Also added
  `gluon.add_inliner` after `scf_to_cf` for parity with NVIDIA/AMD.
- 2026-02-22: Fixed label parsing to strip inline comments (e.g.
  `59: ; preds = %5`) before label detection.
- 2026-02-22: Fixed float binary op type inference — replaced greedy
  `(?:\s+[A-Za-z]+)*` flag regex with explicit LLVM flag enumeration to prevent
  type names (`float`, `half`) from being consumed as flags.
- 2026-02-22: Added `fptrunc` and `fpext` to cast-instruction regex so
  float-width conversions emit correct MSL casts.
- 2026-02-22: Added 21 new tests across 4 classes: TestMetalDynamicReduction
  (JIT reduction kernels), TestMetalComplexCFG (multi-block CFG patterns),
  TestMetalUncommonIntrinsics (ctpop/copysign/freeze/extract/insert),
  TestMetalScalarCastEdgeTypes (i64/half/double/i8/bool/mixed-addrspace/
  many-params). Total test count: 70 (all passing).
- 2026-02-22: All Phase 1, Phase 5, and Phase 6 items now complete. Only
  remaining item: Phase 6 CI artifact upload/reporting (infrastructure task).
