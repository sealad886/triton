# Metal Backend Performance Audit Report

**Date**: 2025-07-24
**Scope**: `compiler.py` (LLVM IR → MSL translator), `driver.py` (Metal runtime)
**Baseline**: 203 tests passed, 1 xfailed. Synthetic benchmark: 1500 lines LLVM IR

## Executive Summary

Profiling identified regex pattern matching as the dominant bottleneck in the
LLVM IR → MSL compilation path. Six compiler optimizations and three driver
optimizations were implemented, achieving:

- **Compilation p50 latency**: 13.70ms → 12.62ms (**−7.9%**)
- **Regex match calls**: 201,050 → 96,550 (**−52%**)
- **Regex sub calls**: 76,560 → 18,520 (**−76%**)
- **Test suite**: 203 passed, 1 xfailed (unchanged)

## Methodology

### Profiling Infrastructure

Two scripts created in `scripts/`:
- `triton_metal_backend_compiler_perf.py`: Generates synthetic LLVM IR exercising
  all major code paths (binop, load, store, cast, GEP, icmp, fcmp, call, branch,
  phi, select, fneg, freeze)
- `profile_metal_ir.py`: cProfile-based benchmark measuring p50/p95/p99/mean at
  three workload sizes (50, 200, 500 instruction groups)

### Baseline Profile (1500-line workload, 10 runs × 500 groups)

| Function | Calls | Cumulative Time | % Total |
|----------|------:|----------------:|--------:|
| `re.Pattern.match` | 201,050 | 50ms | 17% |
| `record_ssa_decl` | 14,000 | 29ms | 10% |
| `msl_id` | 28,000 | 27ms | 9% |
| `to_expr` | 24,500 | 27ms | 9% |
| `re.Pattern.sub` | 76,560 | 24ms | 8% |
| `split_top_level` | 6,010 | 22ms | 7% |

## Implemented Optimizations

### Compiler (compiler.py)

| ID | Optimization | Impact | Evidence |
|----|-------------|--------|----------|
| OPT-1 | `to_expr()` SSA dict fast-path: skip `parse_gep_constexpr` when value is already in SSA dict (~60% of calls) | P1 | Avoids O(n) regex parsing on majority of lookups |
| OPT-2 | `msl_id()` clean-name check: skip `_RE_MSL_ID_SUB` when identifier already matches `[A-Za-z0-9_]+` | P1 | Most SSA names are clean; avoids 28K regex subs |
| OPT-3 | Combined `_RE_LINE_CLEAN`: merge 3 per-line regex subs (metadata, debug, attr-group) into single pattern | P0 | Sub calls: 76K → 18K (−76%) |
| OPT-4 | Declaration pass `%` skip: lines not starting with `%` cannot produce SSA decls | P1 | Skips ~30% of lines from all regex testing |
| OPT-5 | Pre-compiled fallback patterns: `_RE_PTR_SPEC`, `_RE_GEP_INSTR`, `_RE_GEP_FLAG_STRIP` at module level | P2 | Minor; Python regex cache handles most cases |
| OPT-6 | Opcode-based dispatch: extract opcode once per line via `_extract_ir_opcode()`, guard each handler with conditional (`_RE_BINOP.match(line) if _opc in _BINOP_OPCODES else None`) | P0 | Match calls: 201K → 96K (−52%) |

### Driver (driver.py)

| ID | Optimization | Impact | Evidence |
|----|-------------|--------|----------|
| OPT-D1 | Lazy numpy import: replace per-call `import numpy` in `_bind_argument()` with module-level `_get_numpy_module()` | P2 | Eliminates N import checks per launch |
| OPT-D2 | Module-level `_ceildiv()`: remove inline function definition from `launch_kernel()` hot path | P2 | Eliminates closure creation per dispatch |
| OPT-D3 | Module-level `_TORCH_DTYPE_MAP`: replace per-call dict creation in `_normalize_scalar_arg()` with constant | P2 | Eliminates dict allocation per scalar arg |

### Post-Optimization Profile (1500-line workload)

| Function | Calls (before → after) | Time (before → after) |
|----------|----------------------:|----------------------:|
| `re.Pattern.match` | 201,050 → 96,550 | 47ms → 24ms |
| `re.Pattern.sub` | 76,560 → 18,520 | 24ms → 18ms |
| `msl_id` | 28,000 (unchanged) | 27ms → 26ms |
| `to_expr` | 24,500 (unchanged) | 27ms → 26ms |
| `split_top_level` | 6,010 (unchanged) | 22ms → 21ms |

## Generated MSL Code Quality

Sample analysis of 30-group synthetic workload output:
- **Dead code**: Some SSA values computed but unused (e.g., identity casts,
  intermediate values). Harmless — Metal shader compiler optimizes these away.
- **Identity casts**: `(float)(1.0f)` from LLVM fpext/fptrunc of same-width types.
  Optimized away by the Metal compiler.
- **Freeze pass-through**: `v23 = v22;` from LLVM `freeze` instruction (semantic
  no-op in MSL). Optimized away by the Metal compiler.
- **Overall**: Generated MSL is structurally correct and functionally viable. The
  Metal compiler's optimization passes handle residual inefficiencies.

## Deferred Optimizations

These were identified but not implemented due to diminishing returns in pure Python:

| ID | Opportunity | Est. Impact | Reason Deferred |
|----|------------|-------------|-----------------|
| DEF-1 | C extension for `split_top_level()` | −1.5ms (12%) | Requires C build integration; function is inherently O(n) character scanning |
| DEF-2 | Fused declaration+codegen single pass | −2ms (16%) | High risk; two-pass design is architecturally intentional — decl pass determines types before codegen emits code |
| DEF-3 | Compiled IR parser replacing regex | −4ms (30%) | Complete rewrite; would use proper tokenizer/parser instead of 25+ regex patterns |
| DEF-4 | MLIR-level MSL emission | N/A | Would bypass LLVM IR text entirely; requires MLIR Metal dialect work |
| DEF-5 | Command buffer reuse in driver | Moderate | Requires careful lifecycle management; PyObjC overhead dominates |

## Verification

All optimizations validated with full test suite:
```
203 passed, 1 xfailed
```

Benchmark reproduced with:
```bash
source .venv/bin/activate
cd scripts && PYTHONPATH=$PWD/..:$PWD/../python:$PYTHONPATH python profile_metal_ir.py
```

## Git History

```
fccec27 test(metal): add make_metal_ir profiling scripts
77e48d6 perf(metal): reduce per-launch overhead in Metal driver
c1c4778 perf(metal): optimize make_metal_ir compilation hot path
```
