"""
Metal backend compiler for Triton.

Implements the BaseBackend interface for Apple Metal, lowering Triton IR through
LLVM IR to AIR (Apple Intermediate Representation) and then to .metallib binaries
via xcrun.
"""

import functools
import hashlib
import math
import os
import re
import shutil
import struct
import subprocess
import tempfile
import warnings
from dataclasses import dataclass
from types import ModuleType
from typing import Any, Dict, Tuple

from triton import knobs
from triton._C.libtriton import ir, llvm, passes
from triton.backends.compiler import BaseBackend, GPUTarget, Language
from triton.backends.metal.translator_context import (
    _RE_ALIGN_STRIP,
    _UNSIGNED_MSL_MAP,
    TranslatorContext,
    extract_value_token,
    normalize_label,
    split_top_level,
    strip_operand_attrs,
    unsigned_msl,
    vector_alias_msl,
)

# ── Compile observability ───────────────────────────────────────────

_METAL_DEBUG = os.environ.get("TRITON_METAL_DEBUG", "").lower() in ("1", "true", "yes")


def _compile_provenance(options: "MetalOptions | None", src_hash: str) -> None:
    """Log compile provenance for debugging."""
    if not _METAL_DEBUG:
        return
    print("[TRITON_METAL_DEBUG] Compile provenance:")
    print(f"  Options hash: {options.hash() if options is not None else 'N/A'}")
    print(f"  Source hash: {src_hash[:16]}")
    print(f"  SDK version: {_get_metal_sdk_version()}")
    print(f"  Target arch: {getattr(options, 'arch', 'N/A')}")


# ── Unsupported IR diagnostics ──────────────────────────────────────


@dataclass
class UnsupportedIREntry:
    """A single LLVM IR line that the Metal translator could not lower."""

    line_number: int
    line: str
    context_before: list[str]
    context_after: list[str]
    category: str  # 'instruction', 'intrinsic', 'metadata', 'unknown'


def _classify_unsupported_ir(line: str) -> str:
    """Classify an unsupported LLVM IR line into a diagnostic category."""
    stripped = line.strip()
    if stripped.startswith("!") or stripped.startswith("attributes"):
        return "metadata"
    if "call" in stripped and "@llvm." in stripped:
        return "intrinsic"
    if any(
        stripped.startswith(op)
        for op in [
            "invoke",
            "resume",
            "landingpad",
            "indirectbr",
            "catchswitch",
            "catchret",
            "catchpad",
            "cleanupret",
            "cleanuppad",
        ]
    ):
        return "instruction"
    if stripped.startswith("%"):
        return "instruction"
    return "unknown"


# ── Shared LLVM IR regex constants ──────────────────────────────────
# Used by both the SSA declaration pass and the code generation pass
# inside make_metal_ir to keep the two passes in sync.

_SSA_NAME_RE = r"%[-A-Za-z0-9._]+"
_LLVM_FLAGS = (
    r"(?:\s+(?:nsw|nuw|nsz|nnan|ninf|arcp|contract|reassoc|afn|fast|exact|disjoint))*"
)

# ── Additional pre-compiled regex (PERF-001) ────────────────────────
# Combined line-cleaning regex: strips debug metadata, trailing
# comments, and attribute-group references in a single pass.
_RE_LINE_CLEAN = re.compile(
    r",\s*!\w+(?:\.\w+)*\s*![0-9]+.*$"  # LLVM metadata (!dbg, !tbaa, !range, …)
    r"|\s*;.*$"  # Trailing comments
    r"|\s+#\d+\s*$"  # Attribute-group references
)

# Structural / parsing patterns
_RE_LABEL = re.compile(r'^([A-Za-z0-9_."]+):$')
_RE_KERNEL_FUNC = re.compile(
    r"define\s+void\s+@([A-Za-z_][A-Za-z0-9_]*)\s*\(", re.MULTILINE
)
_RE_PARAM_NAME = re.compile(r"(%[-A-Za-z0-9._]+)\s*$")
_RE_PARAM_TYPE = re.compile(
    r"(ptr(?:\s+addrspace\(\d+\))?|i\d+|float|half|bfloat|double|i1)"
)

# Helper-function patterns (duplicates live in translator_context.py)
_RE_VEC_TYPE = re.compile(r"^<\s*(\d+)\s+x\s+(.+)\s*>$")
_RE_CONST_INT = re.compile(r"^-?[0-9]+$")

# LLVM IR attribute-group reference stripping (applied during line cleaning)
_RE_ATTR_GROUP_STRIP = re.compile(r"\s+#\d+\s*$")

# Codegen helper patterns
_RE_PHI_INCOMING = re.compile(r"^\[\s*(.+)\s*,\s*%(.+)\s*\]$")
_RE_SWITCH_CASE = re.compile(r"(\S+)\s+(-?\d+),\s*label\s+%(\S+)")

# ── Typed IR instruction imports ────────────────────────────────────
# ir_types.py defines its own regex patterns (no circular dependency).
from triton.backends.metal.ir_types import GEP as _GEP
from triton.backends.metal.ir_types import AggregateOp as _AggregateOp
from triton.backends.metal.ir_types import Alloca as _Alloca
from triton.backends.metal.ir_types import AtomicOp as _AtomicOp
from triton.backends.metal.ir_types import BinOp as _BinOp
from triton.backends.metal.ir_types import Call as _Call
from triton.backends.metal.ir_types import Cast as _Cast
from triton.backends.metal.ir_types import FCmp as _FCmp
from triton.backends.metal.ir_types import FNeg as _FNeg
from triton.backends.metal.ir_types import Freeze as _Freeze
from triton.backends.metal.ir_types import ICmp as _ICmp
from triton.backends.metal.ir_types import Load as _Load
from triton.backends.metal.ir_types import Phi as _Phi
from triton.backends.metal.ir_types import Select as _Select
from triton.backends.metal.ir_types import Store as _Store
from triton.backends.metal.ir_types import Terminator as _Terminator
from triton.backends.metal.ir_types import UnknownInstruction as _UnknownInstruction
from triton.backends.metal.ir_types import VectorOp as _VectorOp
from triton.backends.metal.ir_types import parse_block as _parse_block  # noqa: E402

# ── Phase 3: module-level codegen emit functions ────────────────────
# Each function mirrors the inline regex handler from the original
# codegen loop, operating on ``inst.raw_line`` (which is the cleaned
# line stored during block parsing).  All functions take the
# ``TranslatorContext`` as first argument.


def _emit_binop(ctx: "TranslatorContext", inst: _BinOp) -> bool:
    out_ssa = inst.out_ssa
    op = inst.op
    llvm_ty_binop = inst.llvm_ty
    lhs = inst.lhs
    rhs = inst.rhs
    out = ctx.msl_id(out_ssa)
    ctx.ssa[out_ssa] = out
    lhs_expr = ctx.to_expr(lhs)
    rhs_expr = ctx.to_expr(rhs)
    if op in _FLOAT_BIN_MAP:
        ctx.emit(f"{out} = {lhs_expr} {_FLOAT_BIN_MAP[op]} {rhs_expr};")
    elif op == "frem":
        ctx.emit(f"{out} = fmod({lhs_expr}, {rhs_expr});")
    elif op in ("lshr", "udiv", "urem"):
        vec_ty = _RE_VEC_TYPE.match(llvm_ty_binop.strip())
        if vec_ty:
            lanes = int(vec_ty.group(1))
            scalar_ty = ctx.llvm_scalar_to_msl(vec_ty.group(2))
            msl_ty = vector_alias_msl(scalar_ty, lanes) if lanes > 1 else scalar_ty
            u_scalar = unsigned_msl(scalar_ty)
            u_ty = vector_alias_msl(u_scalar, lanes) if lanes > 1 else u_scalar
        else:
            msl_ty = ctx.llvm_scalar_to_msl(llvm_ty_binop)
            u_ty = unsigned_msl(msl_ty)
        ctx.emit(
            f"{out} = ({msl_ty})(({u_ty}){lhs_expr} "
            f"{_BIN_MAP[op]} ({u_ty}){rhs_expr});"
        )
    else:
        ctx.emit(f"{out} = {lhs_expr} {_BIN_MAP[op]} {rhs_expr};")
    return False


def _emit_load(
    ctx: "TranslatorContext",
    inst: _Load,
    *,
    needs_tg_loop_sync: bool,
    inserted_tg_sync_before_load: bool,
) -> tuple[bool, bool]:
    out_ssa = inst.out_ssa
    llvm_ty = inst.llvm_ty
    addr_space = inst.addr_space
    ptr = inst.ptr
    out = ctx.msl_id(out_ssa)
    ctx.ssa[out_ssa] = out
    llvm_ty = llvm_ty.strip()
    if needs_tg_loop_sync and addr_space == "3" and not inserted_tg_sync_before_load:
        ctx.emit("threadgroup_barrier(mem_flags::mem_threadgroup);")
        inserted_tg_sync_before_load = True
    ptr_expr = ctx.to_expr(strip_operand_attrs(ptr))
    msl_ty = ctx.llvm_type_to_msl(llvm_ty)
    ctx.emit(f"{out} = *(({ctx.msl_addr_space(addr_space)} {msl_ty}*)({ptr_expr}));")
    return False, inserted_tg_sync_before_load


def _emit_store(ctx: "TranslatorContext", inst: _Store) -> bool:
    llvm_ty = inst.val_ty
    val_token = inst.val
    addr_space = inst.addr_space
    ptr = inst.ptr
    msl_ty = ctx.llvm_type_to_msl(llvm_ty)
    ptr_expr = ctx.to_expr(strip_operand_attrs(ptr))
    ctx.emit(
        f"*(({ctx.msl_addr_space(addr_space)} {msl_ty}*)({ptr_expr})) = {ctx.to_expr(val_token)};"
    )
    return False


def _emit_gep(ctx: "TranslatorContext", inst: _GEP) -> bool:
    out_ssa = inst.out_ssa
    elem_ty = inst.base_ty
    addr_space = inst.addr_space
    base = inst.ptr_operand
    # Try to extract single index from indices_raw (format: "iN value")
    idx_raw = inst.indices_raw.strip()
    idx_parts = idx_raw.split(None, 1)
    if len(idx_parts) == 2 and idx_parts[0].startswith("i"):
        idx = idx_parts[1].strip()
        out = ctx.msl_id(out_ssa)
        ctx.ssa[out_ssa] = out
        ptr_msl_ty = ctx.ptr_type_to_msl(elem_ty, addr_space=addr_space)
        ctx.emit(f"{out} = ({ptr_msl_ty})({ctx.to_expr(base)}) + {ctx.to_expr(idx)};")
        return False
    # Fallback for complex GEPs
    parsed_gep = ctx.parse_gep_instruction(inst.raw_line)
    if parsed_gep is not None:
        _, elem_ty, addr_space, base, idx = parsed_gep
        out = ctx.msl_id(out_ssa)
        ctx.ssa[out_ssa] = out
        ptr_msl_ty = ctx.ptr_type_to_msl(elem_ty, addr_space=addr_space)
        ctx.emit(f"{out} = ({ptr_msl_ty})({ctx.to_expr(base)}) + {ctx.to_expr(idx)};")
        return False
    raise RuntimeError(f"Unsupported GEP form in Metal lowering: '{inst.raw_line}'")


def _emit_cast(ctx: "TranslatorContext", inst: _Cast) -> bool:
    out_ssa = inst.out_ssa
    op = inst.cast_op
    dst_ty = inst.to_ty.strip()
    val = inst.val
    out = ctx.msl_id(out_ssa)
    ctx.ssa[out_ssa] = out
    if op in ("bitcast", "addrspacecast"):
        if dst_ty.startswith("ptr"):
            ctx.emit(f"{out} = {ctx.to_expr(val)};")
        else:
            ctx.emit(
                f"{out} = as_type<{ctx.llvm_type_to_msl(dst_ty)}>({ctx.to_expr(val)});"
            )
    elif op in ("ptrtoint", "inttoptr"):
        ctx.emit(f"{out} = {ctx.to_expr(val)};")
    else:
        ctx.emit(f"{out} = ({ctx.llvm_type_to_msl(dst_ty)})({ctx.to_expr(val)});")
    return False


def _emit_call(ctx: "TranslatorContext", inst: _Call) -> bool:
    out_ssa = inst.out_ssa
    fn = inst.fn_name
    args = [ctx.to_expr(v) for v in ctx.parse_call_args(inst.args_raw)]
    out = ctx.msl_id(out_ssa)
    ctx.ssa[out_ssa] = out
    if fn.startswith("llvm.sadd.with.overflow.") and len(args) == 2:
        ctx.emit(f"{out}.field0 = {args[0]} + {args[1]};")
        ctx.emit(
            f"{out}.field1 = (({args[0]} ^ {out}.field0) & ({args[1]} ^ {out}.field0)) < 0;"
        )
        return False
    if fn.startswith("llvm.uadd.with.overflow.") and len(args) == 2:
        ctx.emit(f"{out}.field0 = {args[0]} + {args[1]};")
        ctx.emit(f"{out}.field1 = {out}.field0 < {args[0]};")
        return False
    if fn.startswith("llvm.ssub.with.overflow.") and len(args) == 2:
        ctx.emit(f"{out}.field0 = {args[0]} - {args[1]};")
        ctx.emit(
            f"{out}.field1 = (({args[0]} ^ {args[1]}) & ({args[0]} ^ {out}.field0)) < 0;"
        )
        return False
    if fn.startswith("llvm.usub.with.overflow.") and len(args) == 2:
        ctx.emit(f"{out}.field0 = {args[0]} - {args[1]};")
        ctx.emit(f"{out}.field1 = {args[0]} < {args[1]};")
        return False
    lowered_intrinsic = ctx.lower_intrinsic(fn, args)
    if lowered_intrinsic is not None:
        ctx.emit(f"{out} = {lowered_intrinsic};")
    elif fn in _AXIS_HELPER_MAP:
        ctx.emit(f"{out} = {_AXIS_HELPER_MAP[fn]};")
    elif fn.startswith("__metal_predicated_ld_global_") and len(args) == 3:
        out_ty = ctx.ssa_decl_types.get(out)
        if out_ty is None:
            ctx.emit(f"{out} = ({args[2]} ? *{args[1]} : {args[0]});")
        else:
            ctx.emit(
                f"{out} = ({args[2]} ? ({out_ty})(*{args[1]}) : ({out_ty})({args[0]}));"
            )
    elif fn == "__metal_simd_shuffle_xor" and len(args) == 2:
        ctx.emit(f"{out} = simd_shuffle_xor({args[0]}, {args[1]});")
    elif fn == "__metal_simd_shuffle_up" and len(args) == 2:
        ctx.emit(f"{out} = simd_shuffle_up({args[0]}, {args[1]});")
    elif fn == "__metal_simd_shuffle" and len(args) == 2:
        ctx.emit(f"{out} = simd_shuffle({args[0]}, {args[1]});")
    elif fn.startswith("__metal_simdgroup_load_tg") and len(args) == 2:
        elem_ty = ctx.simdgroup_elem_for_msl_value(out)
        if ctx.use_native_simdgroup:
            ctx.emit(
                f"simdgroup_load({out}, "
                f"(const threadgroup {elem_ty}*){args[0]}, {args[1]});"
            )
        else:
            fn_tag = elem_ty.replace(" ", "_")
            ctx.emit(
                f"{out} = __metal_sg_load_{fn_tag}("
                f"(const threadgroup {elem_ty}*){args[0]}, {args[1]});"
            )
    elif fn == "__metal_simdgroup_load" and len(args) == 2:
        elem_ty = ctx.simdgroup_elem_for_msl_value(out)
        if ctx.use_native_simdgroup:
            ctx.emit(
                f"simdgroup_load({out}, "
                f"(const device {elem_ty}*){args[0]}, {args[1]});"
            )
        else:
            fn_tag = elem_ty.replace(" ", "_")
            ctx.emit(
                f"{out} = __metal_sg_load_{fn_tag}("
                f"(const device {elem_ty}*){args[0]}, {args[1]});"
            )
    elif fn.startswith("__metal_simdgroup_multiply_accumulate") and len(args) == 3:
        elem_ty = ctx.simdgroup_elem_for_msl_value(out)
        if ctx.use_native_simdgroup:
            ctx.emit(
                f"simdgroup_multiply_accumulate("
                f"{out}, {args[0]}, {args[1]}, {args[2]});"
            )
        else:
            fn_tag = elem_ty.replace(" ", "_")
            ctx.emit(
                f"{out} = __metal_sg_mma_{fn_tag}(" f"{args[0]}, {args[1]}, {args[2]});"
            )
    else:
        if fn.startswith("llvm."):
            raise RuntimeError(f"Unsupported LLVM intrinsic in Metal lowering: '{fn}'")
        ctx.emit(f"{out} = {fn}({', '.join(args)});")
    return False


def _emit_void_call(ctx: "TranslatorContext", inst: _Call) -> bool:
    fn = inst.fn_name
    args = [ctx.to_expr(v) for v in ctx.parse_call_args(inst.args_raw)]
    if fn.startswith("__metal_predicated_st_global_") and len(args) == 3:
        ctx.emit(f"if ({args[2]}) {{ *{args[1]} = {args[0]}; }}")
    elif fn == "__metal_simdgroup_barrier":
        barrier_flags = "mem_flags::mem_none"
        if len(args) >= 1:
            raw_flag = args[0].strip()
            if _RE_CONST_INT.match(raw_flag):
                flag_val = int(raw_flag)
                parts: list[str] = []
                if flag_val & 1:
                    parts.append("mem_flags::mem_threadgroup")
                if flag_val & 2:
                    parts.append("mem_flags::mem_device")
                if not parts:
                    barrier_flags = "mem_flags::mem_none"
                elif len(parts) == 1:
                    barrier_flags = parts[0]
                else:
                    barrier_flags = f"({parts[0]} | {parts[1]})"
            else:
                barrier_flags = "mem_flags::mem_threadgroup"
        ctx.emit(f"threadgroup_barrier({barrier_flags});")
    elif fn.startswith("__metal_simdgroup_store_tg") and len(args) == 3:
        elem_ty = ctx.simdgroup_elem_for_msl_value(args[0])
        if ctx.use_native_simdgroup:
            ctx.emit(
                f"simdgroup_store({args[0]}, "
                f"(threadgroup {elem_ty}*){args[1]}, {args[2]});"
            )
        else:
            fn_tag = elem_ty.replace(" ", "_")
            ctx.emit(
                f"__metal_sg_store_{fn_tag}({args[0]}, "
                f"(threadgroup {elem_ty}*){args[1]}, {args[2]});"
            )
    elif fn.startswith("__metal_simdgroup_store") and len(args) == 3:
        elem_ty = ctx.simdgroup_elem_for_msl_value(args[0])
        if ctx.use_native_simdgroup:
            ctx.emit(
                f"simdgroup_store({args[0]}, "
                f"(device {elem_ty}*){args[1]}, {args[2]});"
            )
        else:
            fn_tag = elem_ty.replace(" ", "_")
            ctx.emit(
                f"__metal_sg_store_{fn_tag}({args[0]}, "
                f"(device {elem_ty}*){args[1]}, {args[2]});"
            )
    elif fn.startswith("llvm.nvvm.barrier0"):
        ctx.emit("threadgroup_barrier(mem_flags::mem_threadgroup);")
    elif fn.startswith("llvm.assume"):
        ctx.emit("(void)0;")
    elif fn.startswith("llvm.lifetime.start") or fn.startswith("llvm.lifetime.end"):
        ctx.emit("(void)0;")
    elif fn.startswith("llvm.memcpy") and len(args) >= 3:
        ctx.emit(
            f"for (int __i = 0; __i < {args[2]}; __i++) "
            f"((device char*){args[0]})[__i] = ((device char*){args[1]})[__i];"
        )
    elif fn.startswith("llvm.memset") and len(args) >= 3:
        ctx.emit(
            f"for (int __i = 0; __i < {args[2]}; __i++) "
            f"((device char*){args[0]})[__i] = (char){args[1]};"
        )
    elif fn.startswith("llvm.memmove") and len(args) >= 3:
        ctx.emit(f"if ((uintptr_t){args[0]} > (uintptr_t){args[1]}) {{")
        ctx.emit(
            f"  for (int __i = {args[2]} - 1; __i >= 0; __i--) "
            f"((device char*){args[0]})[__i] = ((device char*){args[1]})[__i];"
        )
        ctx.emit(f"}} else {{")
        ctx.emit(
            f"  for (int __i = 0; __i < {args[2]}; __i++) "
            f"((device char*){args[0]})[__i] = ((device char*){args[1]})[__i];"
        )
        ctx.emit(f"}}")
    else:
        if fn.startswith("llvm."):
            raise RuntimeError(f"Unsupported LLVM intrinsic in Metal lowering: '{fn}'")
        ctx.emit(f"{fn}({', '.join(args)});")
    return False


def _emit_icmp(ctx: "TranslatorContext", inst: _ICmp) -> bool:
    out_ssa = inst.out_ssa
    pred = inst.pred
    llvm_ty_icmp = inst.llvm_ty
    lhs = inst.lhs
    rhs = inst.rhs
    out = ctx.msl_id(out_ssa)
    ctx.ssa[out_ssa] = out
    cmp_op = _CMP_MAP.get(pred)
    if cmp_op is None:
        raise RuntimeError(f"Unsupported icmp predicate '{pred}'")
    lhs_expr = ctx.to_expr(lhs)
    rhs_expr = ctx.to_expr(rhs)
    if pred.startswith("u") and pred not in ("eq", "ne"):
        vec_m = re.match(r"<\s*(\d+)\s+x\s+(\S+)\s*>", llvm_ty_icmp)
        if vec_m:
            width = int(vec_m.group(1))
            scalar_u = unsigned_msl(ctx.llvm_scalar_to_msl(vec_m.group(2)))
            u_ty = f"vec<{scalar_u}, {width}>"
        else:
            u_ty = unsigned_msl(ctx.llvm_scalar_to_msl(llvm_ty_icmp))
        lhs_expr = f"({u_ty})({lhs_expr})"
        rhs_expr = f"({u_ty})({rhs_expr})"
    ctx.emit(f"{out} = ({lhs_expr} {cmp_op} {rhs_expr});")
    return False


def _emit_fcmp(ctx: "TranslatorContext", inst: _FCmp) -> bool:
    out_ssa = inst.out_ssa
    out = ctx.msl_id(out_ssa)
    ctx.ssa[out_ssa] = out
    lhs_expr = ctx.to_expr(inst.lhs)
    rhs_expr = ctx.to_expr(inst.rhs)
    ctx.emit(f"{out} = {ctx.fcmp_expr(inst.pred, lhs_expr, rhs_expr)};")
    return False


def _emit_phi(ctx: "TranslatorContext", inst: _Phi) -> bool:
    out_ssa = inst.out_ssa
    out = ctx.msl_id(out_ssa)
    ctx.ssa[out_ssa] = out
    incoming_pairs = []
    for incoming in split_top_level(inst.incoming_raw):
        pair = incoming.strip()
        pm = _RE_PHI_INCOMING.match(pair)
        if pm is None:
            raise RuntimeError(
                f"Unsupported phi incoming value in Metal lowering: '{pair}'"
            )
        incoming_pairs.append((pm.group(1).strip(), normalize_label(pm.group(2))))
    if not incoming_pairs:
        raise RuntimeError("Malformed phi node with no incoming values")
    phi_expr = ctx.to_expr(incoming_pairs[-1][0])
    for val, pred in reversed(incoming_pairs[:-1]):
        pred_id = ctx.block_ids.get(pred, -1)
        phi_expr = (
            f"(__triton_pred_block == {pred_id} ? {ctx.to_expr(val)} : {phi_expr})"
        )
    ctx.emit(f"{out} = {phi_expr};")
    return False


def _emit_select(ctx: "TranslatorContext", inst: _Select) -> bool:
    out_ssa = inst.out_ssa
    out = ctx.msl_id(out_ssa)
    ctx.ssa[out_ssa] = out
    cond_ty = getattr(inst, "cond_ty", "i1")
    vec_m = re.match(r"<\s*(\d+)\s+x", cond_ty)
    if vec_m:
        cond_expr = ctx.to_expr(inst.cond)
        true_expr = ctx.to_expr(inst.true_val)
        false_expr = ctx.to_expr(inst.false_val)
        ctx.emit(f"{out} = select({false_expr}, {true_expr}, {cond_expr});")
    else:
        ctx.emit(
            f"{out} = ({ctx.to_expr(inst.cond)} ? {ctx.to_expr(inst.true_val)} : {ctx.to_expr(inst.false_val)});"
        )
    return False


def _emit_fneg(ctx: "TranslatorContext", inst: _FNeg) -> bool:
    out_ssa = inst.out_ssa
    out = ctx.msl_id(out_ssa)
    ctx.ssa[out_ssa] = out
    ctx.emit(f"{out} = -({ctx.to_expr(inst.operand)});")
    return False


def _emit_freeze(ctx: "TranslatorContext", inst: _Freeze) -> bool:
    out_ssa = inst.out_ssa
    out = ctx.msl_id(out_ssa)
    ctx.ssa[out_ssa] = out
    ctx.emit(f"{out} = {ctx.to_expr(inst.operand)};")
    return False


def _emit_vector_op(ctx: "TranslatorContext", inst: _VectorOp) -> bool:

    def _lane(vec_expr: str, width: int, lane: int) -> str:
        if width <= 1:
            return vec_expr
        return f"{vec_expr}[{lane}]"

    if inst.vector_op == "extractelement":
        if inst.width is None or inst.vec is None or inst.idx is None:
            raise RuntimeError(
                f"Unsupported extractelement form in Metal lowering: '{inst.raw_line}'"
            )
        out = ctx.msl_id(inst.out_ssa)
        ctx.ssa[inst.out_ssa] = out
        if inst.width == 1:
            ctx.emit(f"{out} = {ctx.to_expr(inst.vec)};")
        else:
            ctx.emit(f"{out} = {ctx.to_expr(inst.vec)}[{ctx.to_expr(inst.idx)}];")
        return False

    if inst.vector_op == "insertelement":
        if (
            inst.width is None
            or inst.insert_vec is None
            or inst.insert_val is None
            or inst.insert_idx is None
        ):
            raise RuntimeError(
                f"Unsupported insertelement form in Metal lowering: '{inst.raw_line}'"
            )
        out = ctx.msl_id(inst.out_ssa)
        ctx.ssa[inst.out_ssa] = out
        if inst.width == 1:
            ctx.emit(f"{out} = {ctx.to_expr(inst.insert_val)};")
        else:
            ctx.emit(f"{out} = {ctx.to_expr(inst.insert_vec)};")
            ctx.emit(
                f"{out}[{ctx.to_expr(inst.insert_idx)}] = {ctx.to_expr(inst.insert_val)};"
            )
        return False

    if inst.vector_op == "shufflevector":
        if (
            inst.lhs_width is None
            or inst.shuf_elem_ty is None
            or inst.lhs_vec is None
            or inst.rhs_width is None
            or inst.rhs_vec is None
            or inst.out_width is None
            or inst.mask_spec is None
        ):
            raise RuntimeError(
                f"Unsupported shufflevector form in Metal lowering: '{inst.raw_line}'"
            )
        out = ctx.msl_id(inst.out_ssa)
        ctx.ssa[inst.out_ssa] = out
        lhs_width = inst.lhs_width
        rhs_width = inst.rhs_width
        out_width = inst.out_width
        lhs_expr = ctx.to_expr(extract_value_token(inst.lhs_vec))
        rhs_expr = ctx.to_expr(extract_value_token(inst.rhs_vec))
        scalar_ty = ctx.llvm_scalar_to_msl(inst.shuf_elem_ty)

        mask = inst.mask_spec
        if mask == "zeroinitializer":
            mask_elems = ["0"] * out_width
        elif mask in ("undef", "poison"):
            mask_elems = [mask] * out_width
        else:
            if mask.startswith("<") and mask.endswith(">"):
                mask = mask[1:-1].strip()
            mask_elems = split_top_level(mask)

        shuffled: list[str] = []
        for mask_elem in mask_elems:
            _, lane_tok = ctx.split_typed_value(mask_elem)
            lane_tok = lane_tok.strip()
            if lane_tok in ("undef", "poison"):
                shuffled.append("0")
                continue
            if lane_tok == "zeroinitializer":
                lane_tok = "0"
            if not _RE_CONST_INT.match(lane_tok):
                raise RuntimeError(
                    f"Unsupported shufflevector lane token: '{lane_tok}'"
                )
            lane_val = int(lane_tok)
            if lane_val < lhs_width:
                shuffled.append(_lane(lhs_expr, lhs_width, lane_val))
            elif lane_val < lhs_width + rhs_width:
                shuffled.append(_lane(rhs_expr, rhs_width, lane_val - lhs_width))
            else:
                shuffled.append("0")

        if len(shuffled) < out_width:
            shuffled.extend(["0"] * (out_width - len(shuffled)))

        if out_width <= 1:
            ctx.emit(f"{out} = {shuffled[0] if shuffled else '0'};")
        else:
            ctx.emit(
                f"{out} = {scalar_ty}{out_width}({', '.join(shuffled[:out_width])});"
            )
        return False

    raise RuntimeError(
        f"Unsupported vector op form in Metal lowering: '{inst.raw_line}'"
    )


def _emit_aggregate_op(ctx: "TranslatorContext", inst: _AggregateOp) -> bool:
    if inst.agg_op == "extractvalue":
        if inst.src_val is None or inst.idx is None:
            raise RuntimeError(
                f"Unsupported extractvalue form in Metal lowering: '{inst.raw_line}'"
            )
        out_ssa = inst.out_ssa
        out = ctx.msl_id(out_ssa)
        ctx.ssa[out_ssa] = out
        ctx.emit(f"{out} = {ctx.to_expr(inst.src_val)}.field{inst.idx};")
        return False
    if inst.agg_op == "insertvalue":
        if inst.agg_val is None or inst.elem_val is None or inst.idx is None:
            raise RuntimeError(
                f"Unsupported insertvalue form in Metal lowering: '{inst.raw_line}'"
            )
        out_ssa = inst.out_ssa
        out = ctx.msl_id(out_ssa)
        ctx.ssa[out_ssa] = out
        ctx.emit(f"{out} = {ctx.to_expr(inst.agg_val)};")
        ctx.emit(f"{out}.field{inst.idx} = {ctx.to_expr(inst.elem_val)};")
        return False
    raise RuntimeError(
        f"Unsupported aggregate op '{inst.agg_op}' in Metal lowering: '{inst.raw_line}'"
    )


def _emit_atomic(ctx: "TranslatorContext", inst: _AtomicOp) -> bool:
    if inst.atomic_op == "cmpxchg":
        if (
            inst.ptr is None
            or inst.val_type is None
            or inst.expected is None
            or inst.desired is None
        ):
            raise RuntimeError(
                f"Unsupported cmpxchg form in Metal lowering: '{inst.raw_line}'"
            )
        out_ssa = inst.out_ssa
        out = ctx.msl_id(out_ssa)
        ctx.ssa[out_ssa] = out
        msl_ty = ctx.llvm_scalar_to_msl(inst.val_type.strip())
        msl_success = _MEMORY_ORDER_MAP.get(
            inst.success_order or inst.ordering, "memory_order_relaxed"
        )
        msl_fail = _MEMORY_ORDER_MAP.get(
            inst.fail_order or inst.ordering, "memory_order_relaxed"
        )
        ctx.emit(f"{out}.field0 = {ctx.to_expr(inst.expected)};")
        ctx.emit(
            f"{out}.field1 = atomic_compare_exchange_weak_explicit("
            f"reinterpret_cast<{ctx.msl_addr_space(inst.addr_space)} atomic_{msl_ty}*>({ctx.to_expr(inst.ptr)}), "
            f"&{out}.field0, {ctx.to_expr(inst.desired)}, {msl_success}, {msl_fail});"
        )
        return False
    # atomicrmw
    if inst.ptr is None or inst.val_type is None or inst.val is None:
        raise RuntimeError(
            f"Unsupported atomicrmw form in Metal lowering: '{inst.raw_line}'"
        )
    out_ssa = inst.out_ssa
    out = ctx.msl_id(out_ssa)
    ctx.ssa[out_ssa] = out
    msl_ty = ctx.llvm_scalar_to_msl(inst.val_type.strip())
    atomic_func = _ATOMIC_OP_MAP.get(inst.atomic_op, "atomic_fetch_add_explicit")
    msl_order = _MEMORY_ORDER_MAP.get(inst.ordering, "memory_order_relaxed")
    ctx.emit(
        f"{out} = {atomic_func}("
        f"reinterpret_cast<{ctx.msl_addr_space(inst.addr_space)} atomic_{msl_ty}*>({ctx.to_expr(inst.ptr)}), "
        f"{ctx.to_expr(inst.val)}, {msl_order});"
    )
    return False


def _emit_alloca(ctx: "TranslatorContext", inst: _Alloca) -> bool:
    out_ssa = inst.out_ssa
    out = ctx.msl_id(out_ssa)
    ctx.ssa[out_ssa] = out
    ctx.emit(f"{out} = &{out}_storage;")
    return False


def _emit_terminator(
    ctx: "TranslatorContext",
    inst: _Terminator,
    *,
    block_id: int,
    needs_tg_loop_sync: bool,
) -> bool:
    line = inst.raw_line

    if inst.term_kind == "br":
        if inst.target_label is None:
            raise RuntimeError(f"Unsupported br form in Metal lowering: '{line}'")
        target = normalize_label(inst.target_label)
        target_id = ctx.block_ids.get(target)
        if target_id is None:
            raise RuntimeError(f"Unknown branch target '{target}' in Metal lowering")
        if needs_tg_loop_sync and target_id <= block_id:
            ctx.emit("threadgroup_barrier(mem_flags::mem_threadgroup);")
        ctx.emit(f"__triton_pred_block = {block_id};")
        ctx.emit(f"__pc = {target_id};")
        ctx.emit("continue;")
        return True

    if inst.term_kind == "br_cond":
        if inst.cond is None or inst.true_label is None or inst.false_label is None:
            raise RuntimeError(f"Unsupported br_cond form in Metal lowering: '{line}'")
        cond = inst.cond
        t_lbl = normalize_label(inst.true_label)
        f_lbl = normalize_label(inst.false_label)
        t_id = ctx.block_ids.get(t_lbl)
        f_id = ctx.block_ids.get(f_lbl)
        if t_id is None or f_id is None:
            raise RuntimeError(
                f"Unknown branch targets '{t_lbl}'/'{f_lbl}' in Metal lowering"
            )
        if needs_tg_loop_sync and (t_id <= block_id or f_id <= block_id):
            ctx.emit("threadgroup_barrier(mem_flags::mem_threadgroup);")
        ctx.emit(
            f"if ({ctx.to_expr(cond)}) {{ __triton_pred_block = {block_id}; __pc = {t_id}; }} "
            f"else {{ __triton_pred_block = {block_id}; __pc = {f_id}; }}"
        )
        ctx.emit("continue;")
        return True

    if inst.term_kind == "switch":
        if inst.switch_val is None or inst.default_label is None:
            raise RuntimeError(f"Unsupported switch form in Metal lowering: '{line}'")
        default_label = normalize_label(inst.default_label)
        default_id = ctx.block_ids.get(default_label)
        ctx.emit(f"__triton_pred_block = {block_id};")
        ctx.emit(f"switch ({ctx.to_expr(inst.switch_val)}) {{")
        if inst.cases_raw is not None:
            for cm in _RE_SWITCH_CASE.finditer(inst.cases_raw):
                case_val = cm.group(2)
                case_label = normalize_label(cm.group(3))
                case_id = ctx.block_ids.get(case_label)
                if case_id is not None:
                    ctx.emit(f"  case {case_val}: __pc = {case_id}; break;")
        if default_id is not None:
            ctx.emit(f"  default: __pc = {default_id}; break;")
        ctx.emit("}")
        ctx.emit("continue;")
        return True

    if inst.term_kind == "fence":
        syncscope = inst.syncscope
        ordering = inst.ordering
        if ordering is None:
            raise RuntimeError(f"Unsupported fence form in Metal lowering: '{line}'")
        if syncscope in ("workgroup", "threadgroup"):
            ctx.emit("threadgroup_barrier(mem_flags::mem_threadgroup);")
        elif syncscope in ("subgroup", "wavefront"):
            ctx.emit("simdgroup_barrier(mem_flags::mem_threadgroup);")
        else:
            ctx.emit("threadgroup_barrier(mem_flags::mem_device);")
        return False

    if inst.term_kind == "unreachable":
        ctx.emit("return;")
        return True

    if inst.term_kind == "ret":
        ctx.emit("return;")
        return True

    return False


def _extract_ir_opcode(line: str) -> str:
    """Extract LLVM IR opcode from an instruction line for dispatch.

    Handles both SSA assignments (``%v = add ...``) and non-assignment
    instructions (``store ...``, ``br ...``).  Skips optional ``tail`` /
    ``musttail`` / ``notail`` call prefixes so that ``tail call`` returns
    ``'call'``.
    """
    eq_pos = line.find(" = ")
    rest = line[eq_pos + 3 :] if eq_pos >= 0 else line
    for prefix in ("tail ", "musttail ", "notail "):
        if rest.startswith(prefix):
            rest = rest[len(prefix) :]
            break
    sp = rest.find(" ")
    return rest[:sp] if sp >= 0 else rest


# ── Module-level constant data structures (PERF-003) ───────────────
_MSL_RESERVED_IDENTIFIERS = frozenset(
    {
        # MSL address space / function qualifiers
        "kernel",
        "vertex",
        "fragment",
        "compute",
        "thread",
        "threadgroup",
        "device",
        "constant",
        # MSL / C++ scalar types
        "bool",
        "char",
        "short",
        "int",
        "long",
        "half",
        "float",
        "double",
        "void",
        # MSL unsigned type aliases (used in generated code)
        "uint",
        "uchar",
        "ushort",
        "ulong",
        # C++ control-flow keywords
        "if",
        "else",
        "switch",
        "case",
        "default",
        "return",
        "continue",
        "break",
        "while",
        "for",
        "do",
        "goto",
        # C++ type / storage keywords
        "struct",
        "class",
        "union",
        "enum",
        "auto",
        "const",
        "volatile",
        "static",
        "extern",
        "inline",
        "sizeof",
        "namespace",
        "using",
        "template",
        "typename",
        "typedef",
        "new",
        "delete",
        # MSL math builtins (collision would shadow the builtin)
        "fma",
        "fabs",
        "sqrt",
        "floor",
        "ceil",
        "trunc",
        "rint",
        "exp",
        "exp2",
        "log",
        "log2",
        "sin",
        "cos",
        "tanh",
        "pow",
        "copysign",
        "max",
        "min",
        "isnan",
        "popcount",
        "select",
        "clamp",
        "abs",
        "sign",
        "saturate",
        "step",
        "mix",
    }
)

# _UNSIGNED_MSL_MAP: imported from translator_context

_CMP_MAP = {
    "eq": "==",
    "ne": "!=",
    "slt": "<",
    "sle": "<=",
    "sgt": ">",
    "sge": ">=",
    "ult": "<",
    "ule": "<=",
    "ugt": ">",
    "uge": ">=",
}

_FLOAT_BIN_MAP = {"fadd": "+", "fsub": "-", "fmul": "*", "fdiv": "/"}

_BIN_MAP = {
    "add": "+",
    "sub": "-",
    "mul": "*",
    "udiv": "/",
    "sdiv": "/",
    "urem": "%",
    "srem": "%",
    "shl": "<<",
    "lshr": ">>",
    "ashr": ">>",
    "and": "&",
    "or": "|",
    "xor": "^",
}

_AXIS_HELPER_MAP = {
    "__metal_get_thread_position_in_threadgroup_x": "thread_position_in_threadgroup.x",
    "__metal_get_thread_position_in_threadgroup_y": "thread_position_in_threadgroup.y",
    "__metal_get_thread_position_in_threadgroup_z": "thread_position_in_threadgroup.z",
    "__metal_get_threadgroup_position_in_grid_x": "threadgroup_position_in_grid.x",
    "__metal_get_threadgroup_position_in_grid_y": "threadgroup_position_in_grid.y",
    "__metal_get_threadgroup_position_in_grid_z": "threadgroup_position_in_grid.z",
    "__metal_get_threads_per_threadgroup_x": "threads_per_threadgroup.x",
    "__metal_get_threads_per_threadgroup_y": "threads_per_threadgroup.y",
    "__metal_get_threads_per_threadgroup_z": "threads_per_threadgroup.z",
    "__metal_get_threadgroups_per_grid_x": "threadgroups_per_grid.x",
    "__metal_get_threadgroups_per_grid_y": "threadgroups_per_grid.y",
    "__metal_get_threadgroups_per_grid_z": "threadgroups_per_grid.z",
}

_LLVM_SCALAR_TO_MSL = {
    "i1": "bool",
    "i8": "char",
    "i16": "short",
    "i32": "int",
    "i64": "long",
    "half": "half",
    "bfloat": "bfloat",
    "float": "float",
    "double": "double",
}

_MEMORY_ORDER_MAP = {
    "monotonic": "memory_order_relaxed",
    "acquire": "memory_order_acquire",
    "release": "memory_order_release",
    "acq_rel": "memory_order_acq_rel",
    "seq_cst": "memory_order_seq_cst",
}

_ATOMIC_OP_MAP = {
    "add": "atomic_fetch_add_explicit",
    "sub": "atomic_fetch_sub_explicit",
    "xchg": "atomic_exchange_explicit",
    "and": "atomic_fetch_and_explicit",
    "or": "atomic_fetch_or_explicit",
    "xor": "atomic_fetch_xor_explicit",
    "max": "atomic_fetch_max_explicit",
    "min": "atomic_fetch_min_explicit",
    "umax": "atomic_fetch_max_explicit",
    "umin": "atomic_fetch_min_explicit",
    "fadd": "atomic_fetch_add_explicit",
}

# ── Pre-compiled libdevice patterns (PERF-004) ─────────────────────
_LIBDEVICE_UNARY = tuple(
    (re.compile(p), b)
    for p, b in (
        (r"^__(?:nv|ocml)_fabs(?:f|_f32)?$", "fabs"),
        (r"^__(?:nv|ocml)_sqrt(?:f|_f32)?$", "sqrt"),
        (r"^__(?:nv|ocml)_floor(?:f|_f32)?$", "floor"),
        (r"^__(?:nv|ocml)_ceil(?:f|_f32)?$", "ceil"),
        (r"^__(?:nv|ocml)_trunc(?:f|_f32)?$", "trunc"),
        (r"^__(?:nv|ocml)_round(?:f|_f32)?$", "rint"),
        (r"^__(?:nv|ocml)_exp2(?:f|_f32)?$", "exp2"),
        (r"^__(?:nv|ocml)_exp(?:f|_f32)?$", "exp"),
        (r"^__(?:nv|ocml)_log2(?:f|_f32)?$", "log2"),
        (r"^__(?:nv|ocml)_log(?:f|_f32)?$", "log"),
        (r"^__(?:nv|ocml)_sin(?:f|_f32)?$", "sin"),
        (r"^__(?:nv|ocml)_cos(?:f|_f32)?$", "cos"),
        (r"^__(?:nv|ocml)_tanh(?:f|_f32)?$", "tanh"),
    )
)
_LIBDEVICE_BINARY = tuple(
    (re.compile(p), b)
    for p, b in (
        (r"^__(?:nv|ocml)_pow(?:f|_f32)?$", "pow"),
        (r"^__(?:nv|ocml)_copysign(?:f|_f32)?$", "copysign"),
        (r"^__(?:nv|ocml)_fmax(?:f|_f32)?$", "max"),
        (r"^__(?:nv|ocml)_fmin(?:f|_f32)?$", "min"),
    )
)
_LIBDEVICE_FMA = re.compile(r"^__(?:nv|ocml)_fma(?:f|_f32)?$")

# ── Table-driven LLVM intrinsic → MSL builtin mapping (DUP-001) ────
# Simple prefix→builtin tables replace repetitive if/startswith chains
# in lower_intrinsic().  Grouped by arity for dispatch.

_LLVM_INTRINSIC_UNARY: tuple[tuple[str, str], ...] = (
    ("llvm.fabs.", "fabs"),
    ("llvm.sqrt.", "sqrt"),
    ("llvm.floor.", "floor"),
    ("llvm.ceil.", "ceil"),
    ("llvm.trunc.", "trunc"),
    ("llvm.round.", "rint"),
    ("llvm.exp2.", "exp2"),
    ("llvm.exp.", "exp"),
    ("llvm.log2.", "log2"),
    ("llvm.log.", "log"),
    ("llvm.sin.", "sin"),
    ("llvm.cos.", "cos"),
    ("llvm.tanh.", "tanh"),
    ("llvm.ctpop.", "popcount"),
    ("llvm.bitreverse.", "reverse_bits"),
)

# These accept >=1 args (extra args like is_zero_undef are ignored).
_LLVM_INTRINSIC_UNARY_RELAXED: tuple[tuple[str, str], ...] = (
    ("llvm.ctlz.", "clz"),
    ("llvm.cttz.", "ctz"),
)

_LLVM_INTRINSIC_BINARY: tuple[tuple[str, str], ...] = (
    ("llvm.pow.", "pow"),
    ("llvm.copysign.", "copysign"),
)

_LLVM_INTRINSIC_TERNARY: tuple[tuple[str, str], ...] = (
    ("llvm.fma.", "fma"),
    ("llvm.fmuladd.", "fma"),
)

_LLVM_INTRINSIC_MAX_PREFIXES: tuple[str, ...] = (
    "llvm.maximum.",
    "llvm.maxnum.",
    "llvm.smax.",
    "llvm.umax.",
)
_LLVM_INTRINSIC_MIN_PREFIXES: tuple[str, ...] = (
    "llvm.minimum.",
    "llvm.minnum.",
    "llvm.smin.",
    "llvm.umin.",
)


@dataclass(frozen=True)
class MetalOptions:
    num_warps: int = 4
    num_stages: int = 2
    num_ctas: int = 1
    warp_size: int = 32
    enable_fp_fusion: bool = True
    extern_libs: dict = None
    debug: bool = False
    backend_name: str = "metal"
    arch: str = None
    supported_fp8_dtypes: Tuple[str] = ("fp8e5", "fp8e4b15")
    deprecated_fp8_dot_operand_dtypes: Tuple[str] = ()
    default_dot_input_precision: str = "ieee"
    allowed_dot_input_precisions: Tuple[str] = ("ieee",)
    max_num_imprecise_acc_default: int = 0
    sanitize_overflow: bool = True
    launch_cooperative_grid: bool = False
    instrumentation_mode: str = ""
    best_effort: bool = False
    simdgroup_matmul_strategy: str = "auto"

    def __post_init__(self):
        extern_libs = {} if self.extern_libs is None else dict(self.extern_libs)
        object.__setattr__(self, "extern_libs", tuple(extern_libs.items()))
        assert (
            self.num_warps > 0 and (self.num_warps & (self.num_warps - 1)) == 0
        ), "num_warps must be a power of 2"
        assert self.simdgroup_matmul_strategy in (
            "auto",
            "native",
            "fallback",
        ), "simdgroup_matmul_strategy must be one of: auto, native, fallback"

    def hash(self):
        key = "_".join([f"{name}-{val}" for name, val in sorted(self.__dict__.items())])
        return hashlib.sha256(key.encode("utf-8")).hexdigest()


def _get_metal_arch_version(arch) -> str:
    """Map a Metal GPU family (e.g. 'apple8') to an AIR version string."""
    mapping = {
        "apple7": "2.4",
        "apple8": "2.5",
        "apple9": "2.6",
    }
    return mapping.get(str(arch), "2.6")


@functools.lru_cache()
def _xcrun_path():
    path = shutil.which("xcrun")
    if path is None:
        raise RuntimeError(
            "'xcrun' not found in PATH; install Xcode Command Line Tools"
        )
    return path


@functools.lru_cache()
def _get_metal_sdk_version():
    """Get the Metal compiler version from xcrun."""
    try:
        result = subprocess.run(
            [_xcrun_path(), "metal", "--version"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return result.stderr.strip() or result.stdout.strip()
    except Exception:
        return "unknown"


def _metal_backend_hash_inputs() -> tuple[str, ...]:
    repo_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, os.pardir)
    )
    return (
        __file__,
        os.path.join(repo_root, "third_party", "metal", "backend", "driver.py"),
        os.path.join(
            repo_root,
            "third_party",
            "metal",
            "lib",
            "TritonMetalGPUToLLVM",
            "TritonGPUToLLVM.cpp",
        ),
        os.path.join(
            repo_root,
            "third_party",
            "metal",
            "lib",
            "TritonMetalGPUToLLVM",
            "TargetInfo.cpp",
        ),
        os.path.join(
            repo_root,
            "third_party",
            "metal",
            "lib",
            "TritonMetalGPUToLLVM",
            "TargetInfo.h",
        ),
        os.path.join(
            repo_root,
            "third_party",
            "metal",
            "lib",
            "TritonMetalGPUToLLVM",
            "Utility.cpp",
        ),
        os.path.join(
            repo_root,
            "third_party",
            "metal",
            "lib",
            "TritonMetalGPUToLLVM",
            "Utility.h",
        ),
        os.path.join(
            repo_root,
            "lib",
            "Conversion",
            "TritonGPUToLLVM",
            "DotOpToLLVM",
            "FMA.cpp",
        ),
        os.path.join(
            repo_root,
            "lib",
            "Conversion",
            "TritonGPUToLLVM",
            "DotOpToLLVM",
            "FMADotUtility.cpp",
        ),
    )


@functools.lru_cache()
def _get_metal_backend_source_hash() -> str:
    h = hashlib.sha256()
    for path in _metal_backend_hash_inputs():
        h.update(path.encode("utf-8"))
        try:
            with open(path, "rb") as f:
                h.update(f.read())
        except OSError:
            h.update(b"<missing>")
    return h.hexdigest()[:16]


class MetalBackend(BaseBackend):

    @staticmethod
    def supports_target(target: GPUTarget):
        return target.backend == "metal"

    def __init__(self, target: GPUTarget) -> None:
        super().__init__(target)
        # Runtime launches consume generated MSL source through
        # torch.mps.compile_shader while we still emit `.metallib` for tooling.
        self.binary_ext = "metal"

    def parse_options(self, opts) -> Any:
        args = {"arch": self.target.arch}
        if "enable_fp_fusion" not in opts:
            args["enable_fp_fusion"] = knobs.language.default_fp_fusion
        args.update(
            {
                k: opts[k]
                for k in MetalOptions.__dataclass_fields__.keys()
                if k in opts and opts[k] is not None
            }
        )
        return MetalOptions(**args)

    def pack_metadata(self, metadata):
        return (
            metadata.num_warps,
            metadata.num_ctas,
            metadata.shared,
        )

    @staticmethod
    def check_dot_compatibility(lhs_type, rhs_type):
        """Validate that *lhs_type* and *rhs_type* are supported for dot on Metal.

        Returns the minimum (M, N, K) tile supported.  Raises ``CompileError``
        for truly unsupported operand types (e.g. fp64).
        """
        lhs_bw = lhs_type.scalar.primitive_bitwidth
        rhs_bw = rhs_type.scalar.primitive_bitwidth
        if lhs_bw == 64 or rhs_bw == 64:
            raise ValueError(
                "Metal does not support fp64/i64 dot operands "
                f"(got lhs={lhs_bw}-bit, rhs={rhs_bw}-bit)"
            )
        return (1, 1, 1)

    def get_codegen_implementation(self, options):
        return {"min_dot_size": lambda lhs_type, rhs_type: (1, 1, 1)}

    def get_module_map(self) -> Dict[str, ModuleType]:
        from third_party.metal.language import libdevice

        return {"triton.language.extra.libdevice": libdevice}

    def load_dialects(self, ctx):
        import triton._C.libtriton.metal as metal

        if ctx is not None and hasattr(metal, "load_dialects"):
            metal.load_dialects(ctx)

    @staticmethod
    def make_ttir(mod, metadata, opt):
        pm = ir.pass_manager(mod.context)
        pm.enable_debug()
        passes.common.add_inliner(pm)
        passes.ttir.add_rewrite_tensor_pointer(pm)
        passes.ttir.add_rewrite_tensor_descriptor_to_pointer(pm)
        passes.common.add_canonicalizer(pm)
        passes.ttir.add_combine(pm)
        passes.ttir.add_reorder_broadcast(pm)
        passes.common.add_cse(pm)
        passes.common.add_symbol_dce(pm)
        passes.ttir.add_loop_unroll(pm)
        pm.run(mod, "make_ttir")
        return mod

    def gluon_to_ttgir(self, src, metadata, opt):
        """Lower Gluon dialect to TTGIR for Metal, mirroring NVIDIA's pipeline."""
        if not hasattr(passes, "gluon"):
            raise RuntimeError(
                "Gluon support requires passes.gluon (not available in this build)"
            )

        mod = src
        pm = ir.pass_manager(mod.context)
        pm.enable_debug()

        passes.gluon.add_inliner(pm)
        if hasattr(passes.gluon, "add_infer_coalesced_encodings"):
            passes.gluon.add_infer_coalesced_encodings(pm)
        if hasattr(passes.gluon, "add_resolve_auto_encodings"):
            passes.gluon.add_resolve_auto_encodings(pm)
        passes.gluon.add_canonicalizer(pm)
        passes.common.add_sccp(pm)
        passes.ttir.add_loop_aware_cse(pm)
        passes.gluon.add_canonicalizer(pm)
        if hasattr(passes.ttgpuir, "add_combine_tensor_select_and_if"):
            passes.ttgpuir.add_combine_tensor_select_and_if(pm)

        pm.run(mod, "gluon_to_ttgir")
        if hasattr(mod, "get_tensordesc_metadata"):
            metadata["tensordesc_meta"] = mod.get_tensordesc_metadata()
        return mod

    @staticmethod
    def make_ttgir(mod, metadata, opt):
        import triton._C.libtriton.metal as metal

        pm = ir.pass_manager(mod.context)
        pm.enable_debug()
        passes.ttir.add_convert_to_ttgpuir(
            pm, f"metal:{opt.arch}", opt.num_warps, 32, opt.num_ctas
        )
        passes.ttgpuir.add_coalesce(pm)
        passes.ttgpuir.add_remove_layout_conversions(pm)
        passes.ttgpuir.add_optimize_thread_locality(pm)
        metal.passes.ttgpuir.add_accelerate_matmul(pm, opt.arch, opt.num_warps)
        passes.ttgpuir.add_accelerate_matmul(pm)
        passes.ttgpuir.add_remove_layout_conversions(pm)
        passes.ttgpuir.add_optimize_dot_operands(pm, True)
        # Loop fusion and select/if combining (generic TritonGPU passes).
        # Note: add_schedule_loops and add_assign_latencies are excluded
        # because they cause add_pipeline to emit ttg.async_copy_global_to_local
        # which Metal cannot lower (no async copy hardware).
        passes.ttgpuir.add_fuse_nested_loops(pm)
        passes.common.add_canonicalizer(pm)
        passes.ttir.add_triton_licm(pm)
        passes.common.add_canonicalizer(pm)
        passes.ttgpuir.add_combine_tensor_select_and_if(pm)
        if opt.num_stages != 0:
            passes.ttgpuir.add_pipeline(pm, opt.num_stages, False)
        passes.ttir.add_loop_aware_cse(pm)
        passes.common.add_canonicalizer(pm)
        passes.ttgpuir.add_prefetch(pm)
        passes.ttgpuir.add_remove_layout_conversions(pm)
        passes.ttgpuir.add_reduce_data_duplication(pm)
        passes.ttgpuir.add_reorder_instructions(pm)
        passes.ttir.add_loop_aware_cse(pm)
        passes.common.add_symbol_dce(pm)
        passes.common.add_sccp(pm)
        passes.common.add_cse(pm)
        passes.common.add_canonicalizer(pm)
        pm.run(mod, "make_ttgir")
        return mod

    def make_llir(self, src, metadata, options):
        import time as _time

        mod = src
        pm = ir.pass_manager(mod.context)
        pm.enable_debug()

        def _run_pass(add_fn, name: str, *args: Any) -> None:
            """Add a pass and optionally time it when TRITON_METAL_DEBUG is set."""
            t0 = _time.monotonic()
            add_fn(*args) if args else add_fn(pm)
            if _METAL_DEBUG:
                elapsed = _time.monotonic() - t0
                print(f"[TRITON_METAL_DEBUG] Pass {name}: {elapsed:.3f}s")

        _run_pass(
            passes.ttgpuir.add_combine_tensor_select_and_if,
            "combine_tensor_select_and_if",
            pm,
        )
        _run_pass(passes.ttgpuir.add_allocate_warp_groups, "allocate_warp_groups", pm)

        # Lower structured control flow (scf.for/if) to cf dialect BEFORE
        # the backend-specific GPU→LLVM pass, matching NVIDIA/AMD ordering.
        _run_pass(passes.convert.add_scf_to_cf, "scf_to_cf", pm)

        if hasattr(passes, "gluon") and hasattr(passes.gluon, "add_inliner"):
            _run_pass(passes.gluon.add_inliner, "gluon_inliner", pm)

        if hasattr(passes.convert, "add_index_to_llvmir"):
            _run_pass(passes.convert.add_index_to_llvmir, "index_to_llvmir", pm)

        import triton._C.libtriton.metal as metal

        _run_pass(
            passes.ttgpuir.add_allocate_shared_memory, "allocate_shared_memory", pm
        )
        _run_pass(
            passes.ttgpuir.add_allocate_global_scratch_memory,
            "allocate_global_scratch_memory",
            pm,
        )

        _run_pass(metal.passes.ttgpuir.add_to_llvmir, "metal_to_llvmir", pm)
        _run_pass(passes.ttgpuir.add_canonicalize_llvm_ir, "canonicalize_llvm_ir", pm)
        _run_pass(passes.common.add_cse, "cse_1", pm)

        # Some kernels (for example blocked matmul with tt.dot in a K-loop)
        # can retain residual scf control-flow after backend conversion. Run a
        # second scf->cf sweep before cf->llvm to avoid cf.br legalization
        # failures on leftover structured branches.
        _run_pass(passes.convert.add_scf_to_cf, "scf_to_cf_2", pm)

        _run_pass(passes.convert.add_cf_to_llvmir, "cf_to_llvmir", pm)
        _run_pass(passes.convert.add_arith_to_llvmir, "arith_to_llvmir", pm)
        _run_pass(passes.common.add_canonicalizer, "canonicalizer", pm)
        _run_pass(passes.common.add_cse, "cse_2", pm)
        _run_pass(passes.common.add_symbol_dce, "symbol_dce", pm)

        if not hasattr(passes, "llvmir") or not hasattr(passes.llvmir, "add_di_scope"):
            pass
        else:
            _run_pass(passes.llvmir.add_di_scope, "di_scope", pm)

        t0 = _time.monotonic()
        pm.run(mod, "make_llir")
        if _METAL_DEBUG:
            elapsed = _time.monotonic() - t0
            print(f"[TRITON_METAL_DEBUG] pm.run(make_llir): {elapsed:.3f}s")

        # MLIR module -> LLVM IR text
        llvm.init_targets()
        context = llvm.context()
        llvm_mod = llvm.to_module(mod, context)
        # Use a generic AArch64 target triple for Metal/AIR
        triple = "aarch64-apple-macosx14.0.0"
        proc = ""
        features = ""
        llvm.attach_datalayout(llvm_mod, triple, proc, features)

        if options.extern_libs:
            paths = [path for (name, path) in options.extern_libs]
            llvm.link_extern_libs(llvm_mod, paths)

        llvm.optimize_module(llvm_mod, llvm.OPTIMIZE_O3)

        metadata["shared"] = src.get_int_attr("ttg.shared") or 0
        metadata["global_scratch_size"] = (
            src.get_int_attr("ttg.global_scratch_memory_size") or 0
        )
        metadata["global_scratch_align"] = (
            src.get_int_attr("ttg.global_scratch_memory_alignment") or 1
        )

        ret = str(llvm_mod)
        del llvm_mod
        del context

        # Run the dedicated barrier insertion pass on the LLVM IR text.
        # This analyzes shared-memory access patterns (addrspace(3)) and
        # inserts barrier intrinsics at synchronization points before the
        # MSL translator sees the IR.
        from third_party.metal.backend.barrier_pass import run_barrier_pass

        ret = run_barrier_pass(ret, debug=_METAL_DEBUG)

        return ret

    @staticmethod
    def make_metal_ir(src, metadata, opt):
        """
        Convert LLVM IR text to Metal Shading Language source.
        """
        if _METAL_DEBUG:
            src_hash = hashlib.sha256(src.encode()).hexdigest()
            _compile_provenance(opt, src_hash)

        uses_shared_smem = "@global_smem" in src
        shared_bytes = max(int(metadata.get("shared", 0) or 0), 1)
        simdgroup_strategy = (
            getattr(opt, "simdgroup_matmul_strategy", "auto")
            if opt is not None
            else "auto"
        )
        if simdgroup_strategy not in ("auto", "native", "fallback"):
            raise RuntimeError(
                "Unsupported simdgroup_matmul_strategy: "
                f"{simdgroup_strategy!r} (expected auto/native/fallback)"
            )
        # `fallback` forces portable software lowering for simdgroup matrix
        # intrinsics. `auto`/`native` use MSL simdgroup intrinsics directly.
        use_native_simdgroup = simdgroup_strategy != "fallback"

        func_header = _RE_KERNEL_FUNC.search(src)
        if not func_header:
            raise RuntimeError("No kernel function found in LLVM IR")

        kernel_name = func_header.group(1)
        reserved = {"kernel", "vertex", "fragment", "compute"}
        msl_kernel_name = kernel_name
        if kernel_name in reserved:
            msl_kernel_name = f"triton_{kernel_name}"
        metadata["name"] = msl_kernel_name

        ctx = TranslatorContext(
            uses_shared_smem=uses_shared_smem,
            shared_bytes=shared_bytes,
            use_native_simdgroup=use_native_simdgroup,
            kernel_name=kernel_name,
        )

        sig_l = src.find("(", func_header.start())
        depth = 0
        sig_r = -1
        for i in range(sig_l, len(src)):
            ch = src[i]
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth == 0:
                    sig_r = i
                    break
        if sig_r == -1:
            raise RuntimeError(f"Failed to parse signature for kernel '{kernel_name}'")

        params_str = src[sig_l + 1 : sig_r].strip()
        func_body_l = src.find("{", sig_r)
        if func_body_l == -1:
            raise RuntimeError(f"Failed to parse body for kernel '{kernel_name}'")
        brace_depth = 0
        func_body_r = -1
        for i in range(func_body_l, len(src)):
            ch = src[i]
            if ch == "{":
                brace_depth += 1
            elif ch == "}":
                brace_depth -= 1
                if brace_depth == 0:
                    func_body_r = i
                    break
        if func_body_r == -1:
            raise RuntimeError(f"Failed to parse body for kernel '{kernel_name}'")

        params = []
        for idx, raw in enumerate(split_top_level(params_str)):
            name_match = _RE_PARAM_NAME.search(raw)
            llvm_name = name_match.group(1) if name_match else f"%arg{idx}"
            prefix = raw[: name_match.start()].strip() if name_match else raw.strip()
            type_match = _RE_PARAM_TYPE.match(prefix)
            llvm_ty = type_match.group(1) if type_match else "i32"
            params.append(
                {
                    "index": idx,
                    "llvm_name": llvm_name,
                    "llvm_type": llvm_ty,
                    "is_ptr": llvm_ty.startswith("ptr"),
                }
            )

        ptr_elem = {}
        for p in params:
            if not p["is_ptr"]:
                continue
            llvm_name = re.escape(p["llvm_name"])
            m = re.findall(
                rf"getelementptr\s+([A-Za-z0-9_]+),\s+ptr(?:\s+addrspace\(\d+\))?\s+{llvm_name}",
                src,
            )
            pointee = m[-1] if m else "float"
            ptr_elem[p["llvm_name"]] = ctx.llvm_scalar_to_msl(pointee)

        ssa = {}
        param_lines = []
        for p in params:
            arg_name = f"arg{p['index']}"
            ssa[p["llvm_name"]] = arg_name
            if p["is_ptr"]:
                elem_ty = ptr_elem.get(p["llvm_name"], "uint")
                param_lines.append(
                    f"    device {elem_ty}* {arg_name} [[buffer({p['index']})]]"
                )
            else:
                scalar_ty = ctx.llvm_scalar_to_msl(p["llvm_type"])
                param_lines.append(
                    f"    constant {scalar_ty}& {arg_name} [[buffer({p['index']})]]"
                )

        param_lines.extend(
            [
                "    uint3 thread_position_in_threadgroup [[thread_position_in_threadgroup]]",
                "    uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]]",
                "    uint3 threads_per_threadgroup [[threads_per_threadgroup]]",
                "    uint3 threadgroups_per_grid [[threadgroups_per_grid]]",
            ]
        )

        body = src[func_body_l + 1 : func_body_r]
        cleaned_lines = []
        for raw_line in body.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            # Combined line-cleaning: strip debug metadata, trailing
            # comments, and attribute-group references in one pass.
            line = _RE_LINE_CLEAN.sub("", line).rstrip()
            # Second pass: catch attribute-group refs (e.g. " #3") that
            # were masked by debug metadata or comments on the same line.
            if "#" in line:
                line = _RE_ATTR_GROUP_STRIP.sub("", line).rstrip()
            if not line:
                continue
            cleaned_lines.append(line)

        joined_lines: list[str] = []
        i_join = 0
        while i_join < len(cleaned_lines):
            ln = cleaned_lines[i_join]
            if ln.startswith("switch ") and "[" in ln and "]" not in ln:
                parts = [ln]
                while i_join + 1 < len(cleaned_lines):
                    i_join += 1
                    parts.append(cleaned_lines[i_join])
                    if "]" in cleaned_lines[i_join]:
                        break
                joined_lines.append(" ".join(parts))
            else:
                joined_lines.append(ln)
            i_join += 1
        cleaned_lines = joined_lines

        blocks = {"entry": []}
        block_order = ["entry"]
        current_block = "entry"
        for line in cleaned_lines:
            label_match = _RE_LABEL.match(line)
            if label_match:
                label = label_match.group(1).replace("%", "")
                label = normalize_label(label)
                current_block = label
                if label not in blocks:
                    blocks[label] = []
                    block_order.append(label)
                continue
            blocks[current_block].append(line)

        block_ids = {label: idx for idx, label in enumerate(block_order)}
        param_ids = set(ssa.values())

        # Transfer locally-built state into the translator context.
        ctx.ssa = ssa
        ctx.ptr_elem = ptr_elem
        ctx.blocks = blocks
        ctx.block_order = block_order
        ctx.block_ids = block_ids
        ctx.param_ids = param_ids

        # ── Phase 2: parse blocks into typed instruction objects ─────────
        parsed_blocks: dict[str, list] = {}
        for label in ctx.block_order:
            parsed_blocks[label] = _parse_block(ctx.blocks.get(label, []))
        # Store on ctx so the codegen pass (Phase 3) can reuse it later.
        ctx.parsed_blocks = parsed_blocks

        # ── SSA declaration pass (isinstance dispatch) ──────────────────
        for block_label in ctx.block_order:
            for inst in parsed_blocks[block_label]:
                # Only instructions with out_ssa need SSA declaration
                if (
                    not hasattr(inst, "out_ssa")
                    or getattr(inst, "out_ssa", None) is None
                ):
                    continue

                if isinstance(inst, _BinOp):
                    ctx.record_ssa_decl(inst.out_ssa, llvm_ty=inst.llvm_ty)

                elif isinstance(inst, _Load):
                    ctx.record_ssa_decl(inst.out_ssa, llvm_ty=inst.llvm_ty)

                elif isinstance(inst, _Cast):
                    ctx.record_ssa_decl(inst.out_ssa, llvm_ty=inst.to_ty.strip())

                elif isinstance(inst, _Call):
                    if inst.is_void:
                        continue
                    ret_type = ctx.extract_call_ret_type(inst.ret_type)
                    fn_name = inst.fn_name
                    if fn_name.startswith(
                        "__metal_simdgroup_load"
                    ) or fn_name.startswith("__metal_simdgroup_multiply_accumulate"):
                        elem_ty = "float"
                        vec_m = _RE_VEC_TYPE.match(ret_type)
                        if vec_m:
                            elem_ty = ctx.llvm_scalar_to_msl(vec_m.group(2))
                        if ctx.use_native_simdgroup:
                            ctx.record_ssa_decl(
                                inst.out_ssa,
                                msl_ty=f"simdgroup_matrix<{elem_ty}, 8, 8>",
                            )
                        else:
                            if elem_ty not in ("float", "half", "bfloat"):
                                raise RuntimeError(
                                    "Software simdgroup fallback only supports "
                                    f"float/half/bfloat elements, got {elem_ty!r}"
                                )
                            ctx.fallback_simdgroup_elem_types.add(elem_ty)
                            ctx.record_ssa_decl(
                                inst.out_ssa, msl_ty=f"__metal_sgmat_{elem_ty}"
                            )
                    elif ret_type.startswith("{"):
                        struct_name, _ = ctx.get_aggregate_struct_name(ret_type)
                        ctx.record_ssa_decl(inst.out_ssa, msl_ty=struct_name)
                    else:
                        ctx.record_ssa_decl(inst.out_ssa, llvm_ty=ret_type)

                elif isinstance(inst, _GEP):
                    ctx.record_ssa_decl(
                        inst.out_ssa,
                        msl_ty=ctx.ptr_type_to_msl(
                            inst.base_ty, addr_space=inst.addr_space
                        ),
                    )

                elif isinstance(inst, _ICmp):
                    vec_m = re.match(r"<\s*(\d+)\s+x", inst.llvm_ty)
                    if vec_m:
                        width = int(vec_m.group(1))
                        ctx.record_ssa_decl(inst.out_ssa, msl_ty=f"bool{width}" if width > 1 else "bool")
                    else:
                        ctx.record_ssa_decl(inst.out_ssa, msl_ty="bool")

                elif isinstance(inst, _FCmp):
                    ctx.record_ssa_decl(inst.out_ssa, msl_ty="bool")

                elif isinstance(inst, _Phi):
                    ctx.record_ssa_decl(inst.out_ssa, llvm_ty=inst.llvm_ty)

                elif isinstance(inst, _Select):
                    ctx.record_ssa_decl(inst.out_ssa, llvm_ty=inst.result_ty)

                elif isinstance(inst, _FNeg):
                    ctx.record_ssa_decl(inst.out_ssa, llvm_ty=inst.llvm_ty)

                elif isinstance(inst, _Freeze):
                    ctx.record_ssa_decl(inst.out_ssa, llvm_ty=inst.llvm_ty)

                elif isinstance(inst, _VectorOp):
                    if inst.vector_op == "extractelement":
                        if inst.elem_ty is not None:
                            ctx.record_ssa_decl(inst.out_ssa, llvm_ty=inst.elem_ty)
                    elif inst.vector_op == "insertelement":
                        if inst.vec_ty is not None:
                            ctx.record_ssa_decl(inst.out_ssa, llvm_ty=inst.vec_ty)
                    elif inst.vector_op == "shufflevector":
                        if inst.shuf_elem_ty is not None and inst.out_width is not None:
                            scalar_ty = ctx.llvm_scalar_to_msl(inst.shuf_elem_ty)
                            if inst.out_width <= 1:
                                ctx.record_ssa_decl(inst.out_ssa, msl_ty=scalar_ty)
                            else:
                                ctx.record_ssa_decl(
                                    inst.out_ssa,
                                    msl_ty=f"{scalar_ty}{inst.out_width}",
                                )

                elif isinstance(inst, _AggregateOp):
                    if inst.agg_op == "extractvalue":
                        if inst.agg_type is not None and inst.idx is not None:
                            _, field_types = ctx.get_aggregate_struct_name(
                                inst.agg_type
                            )
                            ft = (
                                field_types[inst.idx]
                                if inst.idx < len(field_types)
                                else "int"
                            )
                            ctx.record_ssa_decl(inst.out_ssa, msl_ty=ft)
                    elif inst.agg_op == "insertvalue":
                        if inst.agg_type is not None:
                            struct_name, _ = ctx.get_aggregate_struct_name(
                                inst.agg_type
                            )
                            ctx.record_ssa_decl(inst.out_ssa, msl_ty=struct_name)

                elif isinstance(inst, _AtomicOp):
                    if inst.atomic_op == "cmpxchg":
                        if inst.val_type is not None:
                            agg_type = "{" + inst.val_type.strip() + ", i1}"
                            struct_name, _ = ctx.get_aggregate_struct_name(agg_type)
                            ctx.record_ssa_decl(inst.out_ssa, msl_ty=struct_name)
                    elif inst.val_type is not None:
                        ctx.record_ssa_decl(inst.out_ssa, llvm_ty=inst.val_type.strip())

                elif isinstance(inst, _Alloca):
                    msl_ty = ctx.llvm_scalar_to_msl(inst.alloc_ty.strip())
                    ctx.record_ssa_decl(inst.out_ssa, msl_ty=f"thread {msl_ty}*")
                    storage_name = f"{ctx.msl_id(inst.out_ssa)}_storage"
                    if storage_name not in ctx.ssa_decl_types:
                        ctx.ssa_decl_types[storage_name] = msl_ty

        ctx.body_lines = [
            "  int __triton_pred_block = -1;",
            f"  int __pc = {ctx.block_ids['entry']};",
        ]
        if uses_shared_smem:
            ctx.body_lines.insert(
                0, f"  threadgroup char __triton_shared[{shared_bytes}];"
            )
        ctx.body_lines.extend(
            [f"  {msl_ty} {name};" for name, msl_ty in ctx.ssa_decl_types.items()]
        )
        ctx.body_lines.extend(
            [
                "  while (true) {",
                "    switch (__pc) {",
            ]
        )

        best_effort = getattr(opt, "best_effort", False) if opt is not None else False
        unsupported_lines: list[UnsupportedIREntry] = []
        all_codegen_lines: list[str] = []
        for blk in ctx.block_order:
            for inst in parsed_blocks.get(blk, []):
                all_codegen_lines.append(inst.raw_line)

        for block in ctx.block_order:
            block_id = ctx.block_ids[block]
            block_insts = parsed_blocks.get(block, [])
            ctx.body_lines.append(f"    case {block_id}: {{")

            # Shared-memory dot/staging loops lowered from TTGIR can arrive
            # without explicit barrier ops in LLIR. Detect the canonical
            # pattern (shared stores + shared loads + loop backedge) and
            # conservatively inject threadgroup barriers at the translation
            # boundary to preserve correctness across simdgroups.
            has_tg_store = False
            has_tg_load = False
            has_backedge = False
            for scan_inst in block_insts:
                if isinstance(scan_inst, _Store):
                    if scan_inst.addr_space == "3":
                        has_tg_store = True
                elif isinstance(scan_inst, _Load):
                    if scan_inst.addr_space == "3":
                        has_tg_load = True
                elif isinstance(scan_inst, _Terminator) and scan_inst.opcode == "br":
                    if scan_inst.target_label is not None:
                        target = normalize_label(scan_inst.target_label)
                        target_id = ctx.block_ids.get(target)
                        if target_id is not None and target_id <= block_id:
                            has_backedge = True
                    if (
                        scan_inst.true_label is not None
                        and scan_inst.false_label is not None
                    ):
                        t_lbl = normalize_label(scan_inst.true_label)
                        f_lbl = normalize_label(scan_inst.false_label)
                        t_id = ctx.block_ids.get(t_lbl)
                        f_id = ctx.block_ids.get(f_lbl)
                        if (t_id is not None and t_id <= block_id) or (
                            f_id is not None and f_id <= block_id
                        ):
                            has_backedge = True
            needs_tg_loop_sync = has_tg_store and has_tg_load and has_backedge
            inserted_tg_sync_before_load = False

            terminated = False
            for inst in block_insts:
                line = inst.raw_line

                if (
                    isinstance(inst, _Terminator)
                    and inst.term_kind == "ret"
                    and inst.ret_val is None
                ):
                    ctx.emit("return;")
                    terminated = True
                    break

                if isinstance(inst, _BinOp):
                    _emit_binop(ctx, inst)
                elif isinstance(inst, _Load):
                    _, inserted_tg_sync_before_load = _emit_load(
                        ctx,
                        inst,
                        needs_tg_loop_sync=needs_tg_loop_sync,
                        inserted_tg_sync_before_load=inserted_tg_sync_before_load,
                    )
                elif isinstance(inst, _Store):
                    _emit_store(ctx, inst)
                elif isinstance(inst, _GEP):
                    _emit_gep(ctx, inst)
                elif isinstance(inst, _Cast):
                    _emit_cast(ctx, inst)
                elif isinstance(inst, _Call):
                    if inst.is_void:
                        _emit_void_call(ctx, inst)
                    else:
                        _emit_call(ctx, inst)
                elif isinstance(inst, _ICmp):
                    _emit_icmp(ctx, inst)
                elif isinstance(inst, _FCmp):
                    _emit_fcmp(ctx, inst)
                elif isinstance(inst, _Phi):
                    _emit_phi(ctx, inst)
                elif isinstance(inst, _Select):
                    _emit_select(ctx, inst)
                elif isinstance(inst, _FNeg):
                    _emit_fneg(ctx, inst)
                elif isinstance(inst, _Freeze):
                    _emit_freeze(ctx, inst)
                elif isinstance(inst, _VectorOp):
                    _emit_vector_op(ctx, inst)
                elif isinstance(inst, _AggregateOp):
                    _emit_aggregate_op(ctx, inst)
                elif isinstance(inst, _AtomicOp):
                    _emit_atomic(ctx, inst)
                elif isinstance(inst, _Alloca):
                    _emit_alloca(ctx, inst)
                elif isinstance(inst, _Terminator):
                    terminated = _emit_terminator(
                        ctx,
                        inst,
                        block_id=block_id,
                        needs_tg_loop_sync=needs_tg_loop_sync,
                    )
                    if terminated:
                        break
                else:
                    # UnknownInstruction or unmatched — unsupported line
                    line_idx = -1
                    for _i, _l in enumerate(all_codegen_lines):
                        if _l is line:
                            line_idx = _i
                            break
                    ctx_before = (
                        all_codegen_lines[max(0, line_idx - 2) : line_idx]
                        if line_idx > 0
                        else []
                    )
                    ctx_after = (
                        all_codegen_lines[line_idx + 1 : line_idx + 3]
                        if line_idx >= 0
                        else []
                    )
                    category = _classify_unsupported_ir(line)
                    entry = UnsupportedIREntry(
                        line_number=line_idx + 1,
                        line=line,
                        context_before=list(ctx_before),
                        context_after=list(ctx_after),
                        category=category,
                    )
                    unsupported_lines.append(entry)
                    if _METAL_DEBUG:
                        opcode = line.strip().split()[0] if line.strip() else "UNKNOWN"
                        print(
                            f"[TRITON_METAL_DEBUG] Failure signature: UNSUPPORTED_IR_{category}_{opcode}"
                        )
                    if best_effort:
                        ctx.emit(f"// UNSUPPORTED: {line}")
                        continue

            if not terminated:
                ctx.emit("return;")
            ctx.body_lines.append("    }")

        ctx.body_lines.append("    default: return;")
        ctx.body_lines.append("    }")
        ctx.body_lines.append("  }")

        if unsupported_lines:
            from collections import Counter

            counts = Counter(e.category for e in unsupported_lines)
            total = len(unsupported_lines)
            cat_summary = ", ".join(f"{n} {cat}" for cat, n in sorted(counts.items()))

            artifact_dir = os.environ.get(
                "TRITON_CACHE_DIR", os.path.expanduser("~/.triton")
            )
            os.makedirs(artifact_dir, exist_ok=True)
            artifact_path = os.path.join(artifact_dir, "metal_unsupported_ir.log")
            with open(artifact_path, "w") as flog:
                flog.write(f"Metal IR Translation Diagnostic Report\n")
                flog.write(f"======================================\n\n")
                flog.write(f"Total unsupported lines: {total}\n")
                for cat, n in sorted(counts.items()):
                    flog.write(f"  {cat}: {n}\n")
                flog.write(f"\nDetails:\n")
                flog.write(f"--------\n\n")
                for i, e in enumerate(unsupported_lines, 1):
                    flog.write(f"[{i}] Line {e.line_number} ({e.category}):\n")
                    for cb in e.context_before:
                        flog.write(f"    | {cb}\n")
                    flog.write(f"  > | {e.line}\n")
                    for ca in e.context_after:
                        flog.write(f"    | {ca}\n")
                    flog.write(f"\n")

            preview = "; ".join(e.line for e in unsupported_lines[:3])
            msg = (
                f"{total} unsupported LLVM IR lines ({cat_summary}). "
                f"See {artifact_path} for details. "
                f"First unsupported: {preview}"
            )
            if best_effort:
                warnings.warn(msg, stacklevel=2)
            else:
                raise RuntimeError(msg)

        uses_fp8e5_helpers = (
            "__metal_fp8e5m2_to_fp32" in src or "__metal_fp32_to_fp8e5m2_rn" in src
        )

        msl_lines = [
            "#include <metal_stdlib>",
            "using namespace metal;",
            "",
        ]
        if uses_fp8e5_helpers:
            msl_lines.extend(
                [
                    "inline float __metal_fp8e5m2_to_fp32(char bits) {",
                    "  uchar ub = as_type<uchar>(bits);",
                    "  uint sign = (uint)(ub >> 7);",
                    "  uint exp = (uint)((ub >> 2) & 0x1Fu);",
                    "  uint mant = (uint)(ub & 0x3u);",
                    "  float mag = 0.0f;",
                    "  if (exp == 0u) {",
                    "    if (mant != 0u) {",
                    "      mag = ldexp((float)mant, -16);",
                    "    }",
                    "  } else if (exp == 0x1Fu) {",
                    "    mag = (mant == 0u) ? INFINITY : NAN;",
                    "  } else {",
                    "    mag = ldexp(1.0f + ((float)mant * 0.25f), (int)exp - 15);",
                    "  }",
                    "  return sign ? -mag : mag;",
                    "}",
                    "",
                    "inline char __metal_fp32_to_fp8e5m2_rn(float x) {",
                    "  if (isnan(x)) return as_type<char>((uchar)0x7Fu);",
                    "  uint sign = signbit(x) ? 0x80u : 0u;",
                    "  float ax = fabs(x);",
                    "  if (isinf(ax)) return as_type<char>((uchar)(sign | 0x7Cu));",
                    "  if (ax == 0.0f) return as_type<char>((uchar)sign);",
                    "  int exp2 = 0;",
                    "  float m = frexp(ax, exp2);",
                    "  int e = exp2 - 1 + 15;",
                    "  uint mant = 0u;",
                    "  if (e <= 0) {",
                    "    int sm = (int)rint(ax * 65536.0f);",
                    "    if (sm <= 0) return as_type<char>((uchar)sign);",
                    "    if (sm >= 4) {",
                    "      e = 1;",
                    "      mant = 0u;",
                    "    } else {",
                    "      mant = (uint)sm;",
                    "      e = 0;",
                    "    }",
                    "  } else if (e >= 0x1F) {",
                    "    return as_type<char>((uchar)(sign | 0x7Cu));",
                    "  } else {",
                    "    float frac = (m * 2.0f) - 1.0f;",
                    "    int m2 = (int)rint(frac * 4.0f);",
                    "    if (m2 == 4) {",
                    "      m2 = 0;",
                    "      e += 1;",
                    "      if (e >= 0x1F) return as_type<char>((uchar)(sign | 0x7Cu));",
                    "    }",
                    "    if (m2 < 0) m2 = 0;",
                    "    mant = (uint)m2 & 0x3u;",
                    "  }",
                    "  uchar out = (uchar)(sign | (((uint)e & 0x1Fu) << 2) | mant);",
                    "  return as_type<char>(out);",
                    "}",
                    "",
                ]
            )
        msl_lines.extend(ctx.struct_defs)
        if not use_native_simdgroup and ctx.fallback_simdgroup_elem_types:
            if ctx.struct_defs:
                msl_lines.append("")
            for elem_ty in sorted(ctx.fallback_simdgroup_elem_types):
                tag = elem_ty.replace(" ", "_")
                acc_ty = "float" if elem_ty in ("half", "float", "bfloat") else elem_ty
                msl_lines.extend(
                    [
                        f"struct __metal_sgmat_{tag} {{",
                        f"  {elem_ty} e[64];",
                        "};",
                        "",
                        f"inline __metal_sgmat_{tag} __metal_sg_load_{tag}(",
                        f"    const device {elem_ty}* base, int stride_bytes) {{",
                        f"  __metal_sgmat_{tag} out;",
                        f"  int __row_stride = stride_bytes / (int)sizeof({elem_ty});",
                        "  if (__row_stride <= 0) __row_stride = 8;",
                        "  for (int r = 0; r < 8; ++r) {",
                        "    for (int c = 0; c < 8; ++c) {",
                        "      out.e[r * 8 + c] = base[r * __row_stride + c];",
                        "    }",
                        "  }",
                        "  return out;",
                        "}",
                        "",
                        f"inline void __metal_sg_store_{tag}(",
                        f"    __metal_sgmat_{tag} mat, device {elem_ty}* base, int stride_bytes) {{",
                        f"  int __row_stride = stride_bytes / (int)sizeof({elem_ty});",
                        "  if (__row_stride <= 0) __row_stride = 8;",
                        "  for (int r = 0; r < 8; ++r) {",
                        "    for (int c = 0; c < 8; ++c) {",
                        "      base[r * __row_stride + c] = mat.e[r * 8 + c];",
                        "    }",
                        "  }",
                        "}",
                        "",
                        f"inline __metal_sgmat_{tag} __metal_sg_mma_{tag}(",
                        f"    __metal_sgmat_{tag} a, __metal_sgmat_{tag} b, __metal_sgmat_{tag} c) {{",
                        f"  __metal_sgmat_{tag} out;",
                        "  for (int r = 0; r < 8; ++r) {",
                        "    for (int n = 0; n < 8; ++n) {",
                        f"      {acc_ty} acc = ({acc_ty})c.e[r * 8 + n];",
                        "      for (int k = 0; k < 8; ++k) {",
                        f"        acc += ({acc_ty})a.e[r * 8 + k] * ({acc_ty})b.e[k * 8 + n];",
                        "      }",
                        f"      out.e[r * 8 + n] = ({elem_ty})acc;",
                        "    }",
                        "  }",
                        "  return out;",
                        "}",
                        "",
                    ]
                )
        if ctx.struct_defs:
            msl_lines.append("")
        msl_lines.extend(
            [
                f"kernel void {msl_kernel_name}(",
                ",\n".join(param_lines),
                ") {",
            ]
        )
        msl_lines.extend(ctx.body_lines)
        msl_lines.append("}")
        msl_lines.append("")
        msl_source = "\n".join(msl_lines)

        # ── Optional matmul acceleration pass ──
        if simdgroup_strategy in ("auto", "native"):
            from third_party.metal.backend.matmul_accel import (
                optimize_matmul_msl,
                select_matmul_strategy,
            )

            gpu_family = getattr(opt, "arch", None) or "apple8"
            strategy = select_matmul_strategy(
                M=32,
                N=32,
                K=32,
                dtype="float",
                gpu_family=str(gpu_family),
                strategy_hint=simdgroup_strategy,
            )
            msl_source = optimize_matmul_msl(msl_source, [strategy])

        return msl_source

    @staticmethod
    def make_metallib(src, metadata, opt):
        """
        Compile Metal source to .metallib binary using xcrun.

        Takes MSL source code and produces a compiled .metallib binary
        suitable for loading on Apple GPU hardware.
        """
        xcrun = _xcrun_path()
        src_path = None
        air_path = None
        metallib_path = None

        try:
            # Write MSL source to temp file
            with tempfile.NamedTemporaryFile(
                suffix=".metal", delete=False, mode="w"
            ) as f:
                f.write(src)
                src_path = f.name

            # Compile .metal -> .air (Apple Intermediate Representation)
            air_path = src_path.replace(".metal", ".air")
            compile_cmd = [xcrun, "metal", "-c", src_path, "-o", air_path]
            if opt.debug:
                compile_cmd.append("-gline-tables-only")
            result = subprocess.run(compile_cmd, capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError(
                    f"Metal compilation failed:\n"
                    f"stdout: {result.stdout}\n"
                    f"stderr: {result.stderr}\n"
                    f"command: {' '.join(compile_cmd)}"
                )

            # Link .air -> .metallib
            metallib_path = src_path.replace(".metal", ".metallib")
            link_cmd = [xcrun, "metallib", air_path, "-o", metallib_path]
            result = subprocess.run(link_cmd, capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError(
                    f"Metal linking failed:\n"
                    f"stdout: {result.stdout}\n"
                    f"stderr: {result.stderr}\n"
                    f"command: {' '.join(link_cmd)}"
                )

            with open(metallib_path, "rb") as f:
                return f.read()

        finally:
            for path in [src_path, air_path, metallib_path]:
                if path and os.path.exists(path):
                    try:
                        os.remove(path)
                    except OSError:
                        pass

    def add_stages(self, stages, options, language):
        if language == Language.TRITON:
            stages["ttir"] = lambda src, metadata: self.make_ttir(
                src, metadata, options
            )
            stages["ttgir"] = lambda src, metadata: self.make_ttgir(
                src, metadata, options
            )
        elif language == Language.GLUON:
            stages["ttgir"] = lambda src, metadata: self.gluon_to_ttgir(
                src, metadata, options
            )
        stages["llir"] = lambda src, metadata: self.make_llir(src, metadata, options)
        stages["metal"] = lambda src, metadata: self.make_metal_ir(
            src, metadata, options
        )
        stages["metallib"] = lambda src, metadata: self.make_metallib(
            src, metadata, options
        )
        if knobs.runtime.add_stages_inspection_hook is not None:
            knobs.runtime.add_stages_inspection_hook(
                self, stages, options, language, None
            )

    @functools.lru_cache()
    def hash(self):
        version = _get_metal_sdk_version()

        try:
            import triton

            triton_version = triton.__version__
        except (ImportError, AttributeError):
            triton_version = "dev"
        backend_hash = _get_metal_backend_source_hash()
        return f"{version}-{self.target.arch}-{triton_version}-{backend_hash}"
