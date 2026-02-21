# Metal First-Class Support Plan

Last updated: 2026-02-21

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
| Runtime `utils.load_binary` contract | matches JIT expectations | matches JIT expectations | signature/return mismatch | High |
| Runtime `utils.get_device_properties` schema | includes expected keys | includes expected keys | missing shared-memory key | High |
| LLVM IR -> MSL backend stage | mature backend-specific lowering | mature backend-specific lowering | placeholder stub generator | High |
| Backend stage inspection hook | implemented | implemented | missing | Medium |
| Test utility backend helpers | cuda/hip helpers | hip helpers | no `is_metal` helper | Medium |
| AOT unit test behavior | supported | supported | not handled cleanly | Medium |

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
- [x] Add `MetalUtils.unload_module` and ensure no-op safety semantics.
- [x] Align Metal device property schema with expected shared-memory keys used
      by compiler/runtime guards.
- [x] Verify `MetalLauncher` invocation path handles loaded function objects
      consistently.

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
- [x] Add coverage for translation correctness and metallib compilation from
      lowered LLVM IR snippets.
- [ ] Extend translation coverage for complex control-flow constructs
      (phi-heavy CFGs, uncommon intrinsic patterns) used by advanced kernels.

Acceptance:
- Kernels that lower through the Metal LLVM pipeline compile through
  `make_metal_ir` -> `make_metallib` without placeholder stubs.

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
