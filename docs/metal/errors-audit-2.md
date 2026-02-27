# Metal Backend — Final Errors Audit Report

**Date:** 2025-07-15  
**Branch:** `feat/metal-support`  
**Scope:** `third_party/metal/backend/compiler.py`, `third_party/metal/backend/driver.py`  
**Baseline:** 203 passed, 1 xfailed → **Final: 207 passed, 1 xfailed**  
**Commits:** `cb6cfc9018`, `0183cd209e`

---

## Context

This is the **final errors audit pass** following three previous audits:

| Audit | Scope | Key commits |
|-------|-------|------------|
| Duplication | compiler.py, driver.py, test utils | `16130beb70` |
| Errors pass 1 | compiler.py (ERR-001 through ERR-011) | `9f094d7e4c` |
| Performance | compiler.py hot paths | `c1c4778`, `77e48d6`, `fccec27`, `4af98f3` |

Goal: verify those changes introduced no regressions, and catch anything the first errors pass missed.

---

## Findings

### AUDIT2-001 — Combined regex masks attribute-group refs (P1, FIXED)

| Field | Value |
|-------|-------|
| **Severity** | P1 — silent mis-parse of LLVM IR lines |
| **Component** | `compiler.py` line-cleaning loop |
| **Evidence** | `_RE_LINE_CLEAN` alternation: when `#3` appears *before* `!dbg !42`, the debug-metadata alternative fires first and consumes the suffix, leaving `#3` in the cleaned line |
| **Reproduction** | `%r = tail call float @llvm.sqrt.f32(float %x) #3, !dbg !42` → cleaned to `%r = tail call float @llvm.sqrt.f32(float %x) #3` → `_RE_CALL_OUT` fails to match |
| **Root cause** | Single-pass combined regex with `|` alternatives. The attr-group ref `#\d+` occurs *before* the debug annotation `!dbg`, so the `!dbg` alternative matches first and gobbles the trailing portion, but `#3` persists |
| **Fix** | Added second-pass `_RE_ATTR_GROUP_STRIP = re.compile(r'\s+#\d+\s*$')` applied after `_RE_LINE_CLEAN`, guarded by `if "#" in line` |
| **Commit** | `cb6cfc9018` |
| **Tests added** | `TestMetalAudit2AttrGroupWithDebugMetadata` (2 tests) |
| **Regression risk** | Minimal — second pass only fires on lines containing `#`, which is rare after the first pass |

### AUDIT2-002 — `llvm.powi` lowered to `powr()` instead of `pown()` (P1, FIXED)

| Field | Value |
|-------|-------|
| **Severity** | P1 — undefined behavior for negative bases |
| **Component** | `compiler.py` `lower_intrinsic()`, line 1224 |
| **Evidence** | MSL spec: `powr(x, y)` requires `x ≥ 0`; `pown(x, n)` handles negative bases with integer exponents. `llvm.powi` takes an integer exponent, so the base can be negative |
| **Reproduction** | `powi(-2.0, 3)` → `powr(-2.0, 3)` → UB on Metal GPU |
| **Root cause** | Original code used `powr()` (probably confused with generic `pow()`). Also wrapped the integer exponent in `static_cast<float>`, which was unnecessary for `pown()` |
| **Fix** | Changed `powr({args[0]}, static_cast<float>({args[1]}))` → `pown({args[0]}, {args[1]})` |
| **Commit** | `0183cd209e` |
| **Tests added** | `TestMetalAudit2PowiIntrinsic` (2 tests); updated existing `test_powi_intrinsic` to assert `pown(` |
| **Regression risk** | None — `pown()` is a strict superset of `powr()` for integer exponents |

---

## Deferred Issues (P2)

### ERR-006 — `ashr` uses `>>` on signed types

`>>` on signed integers is implementation-defined in C/C++, but Apple GPU guarantees arithmetic right shift. Acceptable for Metal-only target. Would need explicit `arithmetic_shift_right()` helper if portability were required.

### ERR-011 — Vector select not supported

`_RE_SELECT` only matches `i1` condition, not `<N x i1>`. Vector selects are extremely rare in Triton's LLVM output. If encountered, the line falls through to `_unhandled()` and raises a clear error.

### musttail / notail call prefixes

`_RE_CALL_OUT` and `_RE_VOID_CALL` only handle `(?:tail\s+)?call`. The `musttail` and `notail` call prefixes are not matched. These are exceedingly rare in Triton's pipeline and would fail loudly at the regex-match stage rather than silently miscompiling.

---

## Verified Clean Areas

| Area | Verification method | Result |
|------|-------------------|--------|
| **Table-driven intrinsic dispatch** | Checked all 4 tables for prefix shadowing (e.g. `llvm.powi.` vs `llvm.pow.`). `startswith()` order is safe because `powi` is special-cased before table lookup | Clean |
| **Opcode-gated regex dispatch** | Confirmed `_extract_ir_opcode()` correctly routes all opcodes; `br` gates both `_RE_BR` and `_RE_BR_COND` | Clean |
| **Half / bfloat hex float parsing** | `constant_to_msl()` correctly decodes `0xH` (half) and `0xR` (bfloat16) hex literals via `struct.unpack` | Clean |
| **SSA name disambiguation** | `_msl_id_used` dict is local to `make_metal_ir()` — fresh per compilation, no stale state | Clean |
| **Memmove direction logic** | Backward copy when `dst > src`, forward otherwise. Loop bounds correct | Clean |
| **`_scale_grid_for_pyobjc()`** | Extracted helper used in both launch paths. Logic verified | Clean |
| **`_resolve_and_validate_kernel_name()`** | Extracted helper used in both launch paths. Logic verified | Clean |
| **driver.py overall** | Full review of device management, kernel handle classes, argument packing | Clean |

---

## Test Summary

```
207 passed, 1 xfailed in 1.70s
```

| Change | Tests before | Tests after |
|--------|-------------|-------------|
| Baseline | 203 | — |
| +AUDIT2-001 | — | 205 |
| +AUDIT2-002 | — | 207 |

All existing tests continue to pass. No xfail regressions.

---

## Commits (not pushed)

```
0183cd209e fix(metal): llvm.powi must lower to pown(), not powr()
cb6cfc9018 fix(metal): attr-group refs masked by debug metadata in combined regex
```
