"""
Metal backend compiler for Triton.

Implements the BaseBackend interface for Apple Metal, lowering Triton IR through
LLVM IR to AIR (Apple Intermediate Representation) and then to .metallib binaries
via xcrun.
"""

import dataclasses
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
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, Tuple

from triton import knobs
from triton._C.libtriton import ir, llvm, passes
from triton.backends.compiler import BaseBackend, GPUTarget, Language

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


@dataclasses.dataclass
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

_RE_CALL_OUT = re.compile(
    r"^("
    + _SSA_NAME_RE
    + r")\s*=\s*(?:tail\s+)?call\s+(.+?)\s+@([A-Za-z0-9_.$-]+)\((.*)\)$"
)
_RE_BINOP = re.compile(
    r"^("
    + _SSA_NAME_RE
    + r")\s*=\s*(add|sub|mul|udiv|sdiv|urem|srem|shl|lshr|ashr|and|or|xor|fadd|fsub|fmul|fdiv|frem)"
    + _LLVM_FLAGS
    + r"\s+(.+)$"
)
_RE_ICMP = re.compile(
    r"^(" + _SSA_NAME_RE + r")\s*=\s*icmp\s+(\w+)\s+([^ ]+)\s+([^,]+),\s*(.+)$"
)
_RE_FCMP = re.compile(
    r"^(" + _SSA_NAME_RE + r")\s*=\s*fcmp\s+(\w+)\s+[^ ]+\s+([^,]+),\s*(.+)$"
)
_RE_CAST = re.compile(
    r"^("
    + _SSA_NAME_RE
    + r")\s*=\s*(sext|zext|trunc|fptrunc|fpext|sitofp|uitofp|fptosi|fptoui|bitcast|addrspacecast|ptrtoint|inttoptr)\s+(.+)\s+to\s+(.+)$"
)

# ── Additional pre-compiled regex (PERF-001) ────────────────────────
# Line-cleaning patterns (applied to every raw LLVM IR line)
_RE_DBG_STRIP = re.compile(r",\s*!dbg\s*![0-9]+.*$")
_RE_COMMENT_STRIP = re.compile(r"\s*;.*$")

# Combined line-cleaning regex: strips debug metadata, trailing
# comments, and attribute-group references in a single pass.
_RE_LINE_CLEAN = re.compile(
    r",\s*!dbg\s*![0-9]+.*$"  # Debug metadata
    r"|\s*;.*$"  # Trailing comments
    r"|\s+#\d+\s*$"  # Attribute-group references
)
_RE_LLVM_VECTOR_REDUCE = re.compile(
    r"^llvm\.vector\.reduce\.([a-z]+)\.v(\d+)([A-Za-z0-9]+)$"
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

# Helper-function patterns
_RE_MSL_ID_CLEAN = re.compile(r"[^A-Za-z0-9_]")
_RE_PTR_TYPE = re.compile(r"^ptr(?:\s+addrspace\((\d+)\))?$")
_RE_VEC_TYPE = re.compile(r"^<\s*(\d+)\s+x\s+(.+)\s*>$")
_RE_ALIGN_STRIP = re.compile(r",\s*align\s+\d+$")
_RE_CONST_INT = re.compile(r"^-?[0-9]+$")
_RE_CONST_HEX_FLOAT = re.compile(r"^0x([0-9A-Fa-f]{16})$")
_RE_CONST_HEX_HALF = re.compile(r"^0xH([0-9A-Fa-f]{4})$")
_RE_CONST_HEX_BFLOAT = re.compile(r"^0xR([0-9A-Fa-f]{4})$")
_RE_CONST_FLOAT = re.compile(r"^-?[0-9]*\.?[0-9]+([eE][+-]?[0-9]+)?$")
_RE_CALL_RET_VEC = re.compile(r"<\s*\d+\s+x\s+[^>]+\s*>")
_RE_CALL_RET_PTR = re.compile(r"ptr(?:\s+addrspace\(\d+\))?")
_RE_CALL_RET_SCALAR = re.compile(
    r"\bi\d+\b|\bi1\b|\bhalf\b|\bbfloat\b|\bfloat\b|\bdouble\b"
)

# LLVM IR attribute-group reference stripping (applied during line cleaning)
_RE_ATTR_GROUP_STRIP = re.compile(r"\s+#\d+\s*$")

# SSA declaration pass patterns (types needed but not full codegen)
_RE_PHI_DECL = re.compile(r"^(" + _SSA_NAME_RE + r")\s*=\s*phi\s+(.+?)\s+\[")
_RE_FNEG_DECL = re.compile(
    r"^(" + _SSA_NAME_RE + r")\s*=\s*fneg" + _LLVM_FLAGS + r"\s+(.+?)\s+(.+)$"
)
_RE_FREEZE_DECL = re.compile(r"^(" + _SSA_NAME_RE + r")\s*=\s*freeze\s+(.+?)\s+(.+)$")
_RE_SELECT_DECL = re.compile(
    r"^(" + _SSA_NAME_RE + r")\s*=\s*select\s+i1\s+[^,]+,\s+(.+?)\s+[^,]+,\s+.+$"
)
_RE_GEP_DECL = re.compile(
    r"^(" + _SSA_NAME_RE + r")\s*=\s*getelementptr(?:\s+\w+)*\s+([A-Za-z0-9_]+),"
    r"\s+ptr(?:\s+addrspace\((\d+)\))?\s+([^,]+),\s+i\d+\s+(.+)$"
)
_RE_EXTRACTELEM_DECL = re.compile(
    r"^(" + _SSA_NAME_RE + r")\s*=\s*extractelement\s+<\s*\d+\s+x\s+(.+?)\s*>"
    r"\s+([^,]+),\s+i\d+\s+(.+)$"
)
_RE_INSERTELEM_DECL = re.compile(
    r"^(" + _SSA_NAME_RE + r")\s*=\s*insertelement\s+(<\s*\d+\s+x\s+.+\s*>)"
    r"\s+([^,]+),\s+.+\s+([^,]+),\s+i\d+\s+(.+)$"
)
_RE_SHUFFLEVECTOR_DECL = re.compile(
    r"^("
    + _SSA_NAME_RE
    + r")\s*=\s*shufflevector\s+<\s*(\d+)\s+x\s+(.+?)\s*>\s+([^,]+),\s*"
    r"<\s*(\d+)\s+x\s+.+?\s*>\s+([^,]+),\s*<\s*(\d+)\s+x\s+i\d+\s*>\s+(.+)$"
)
_RE_LOAD_DECL = re.compile(
    r"^(" + _SSA_NAME_RE + r")\s*=\s*load\s+([^,]+),"
    r"\s+ptr(?:\s+addrspace\(\d+\))?\s+(.+)$"
)

# Code-generation pass patterns
_RE_PHI = re.compile(r"^(" + _SSA_NAME_RE + r")\s*=\s*phi\s+.+?\s+(\[.+)$")
_RE_VOID_CALL = re.compile(
    r"^(?:tail\s+)?call(?:\s+\w+)*\s+void\s+@([A-Za-z0-9_.$-]+)\((.*)\)$"
)
_RE_FNEG = re.compile(
    r"^(" + _SSA_NAME_RE + r")\s*=\s*fneg" + _LLVM_FLAGS + r"\s+[^ ]+\s+(.+)$"
)
_RE_FREEZE = re.compile(r"^(" + _SSA_NAME_RE + r")\s*=\s*freeze\s+[^ ]+\s+(.+)$")
_RE_SELECT = re.compile(
    r"^(" + _SSA_NAME_RE + r")\s*=\s*select\s+i1\s+([^,]+),"
    r"\s+[^ ]+\s+([^,]+),\s+[^ ]+\s+(.+)$"
)
_RE_GEP = re.compile(
    r"^(" + _SSA_NAME_RE + r")\s*=\s*getelementptr(?:\s+\w+)*\s+[A-Za-z0-9_]+,"
    r"\s+ptr(?:\s+addrspace\((\d+)\))?\s+([^,]+),\s+i\d+\s+(.+)$"
)
_RE_EXTRACTELEM = re.compile(
    r"^(" + _SSA_NAME_RE + r")\s*=\s*extractelement\s+<\s*(\d+)\s+x\s+.+\s*>"
    r"\s+([^,]+),\s+i\d+\s+(.+)$"
)
_RE_INSERTELEM = re.compile(
    r"^(" + _SSA_NAME_RE + r")\s*=\s*insertelement\s+<\s*(\d+)\s+x\s+.+\s*>"
    r"\s+([^,]+),\s+.+\s+([^,]+),\s+i\d+\s+(.+)$"
)
_RE_SHUFFLEVECTOR = re.compile(
    r"^("
    + _SSA_NAME_RE
    + r")\s*=\s*shufflevector\s+<\s*(\d+)\s+x\s+(.+?)\s*>\s+([^,]+),\s*"
    r"<\s*(\d+)\s+x\s+.+?\s*>\s+([^,]+),\s*<\s*(\d+)\s+x\s+i\d+\s*>\s+(.+)$"
)
_RE_LOAD = re.compile(
    r"^(" + _SSA_NAME_RE + r")\s*=\s*load\s+(.+?),"
    r"\s+ptr(?:\s+addrspace\((\d+)\))?\s+(.+)$"
)
_RE_STORE = re.compile(r"^store\s+(.+?),\s+ptr(?:\s+addrspace\((\d+)\))?\s+(.+)$")
_RE_BR = re.compile(r"^br\s+label\s+%(.+)$")
_RE_BR_COND = re.compile(r"^br\s+i1\s+([^,]+),\s+label\s+%([^,]+),\s+label\s+%(.+)$")
_RE_PHI_INCOMING = re.compile(r"^\[\s*(.+)\s*,\s*%(.+)\s*\]$")

# ── Phase 7: LLVM surface generalization patterns ──────────────────
_RE_EXTRACTVALUE = re.compile(
    r"^(" + _SSA_NAME_RE + r")\s*=\s*extractvalue\s+(\{[^}]+\})\s+(\S+),\s*(\d+)$"
)
_RE_INSERTVALUE = re.compile(
    r"^("
    + _SSA_NAME_RE
    + r")\s*=\s*insertvalue\s+(\{[^}]+\})\s+(\S+),\s+(\S+)\s+(\S+),\s*(\d+)$"
)
_RE_ATOMICRMW = re.compile(
    r"^(" + _SSA_NAME_RE + r")\s*=\s*atomicrmw\s+"
    r"(add|sub|xchg|and|or|xor|max|min|umax|umin|fadd)\s+"
    r"ptr(?:\s+addrspace\((\d+)\))?\s+(\S+),\s+"
    r"(\S+)\s+(\S+)\s+"
    r"(monotonic|acquire|release|acq_rel|seq_cst)"
)
_RE_CMPXCHG = re.compile(
    r"^(" + _SSA_NAME_RE + r")\s*=\s*cmpxchg(?:\s+weak)?\s+"
    r"ptr(?:\s+addrspace\((\d+)\))?\s+(\S+),\s+"
    r"(\S+)\s+(\S+),\s+"
    r"\S+\s+(\S+)\s+"
    r"(monotonic|acquire|release|acq_rel|seq_cst)\s+"
    r"(monotonic|acquire|release|acq_rel|seq_cst)"
)
_RE_ALLOCA = re.compile(
    r"^(" + _SSA_NAME_RE + r")\s*=\s*alloca\s+(\S+)(?:,\s*align\s+\d+)?$"
)
_RE_SWITCH = re.compile(r"^switch\s+(\S+)\s+(\S+),\s*label\s+%(\S+)\s*\[(.+)\]$")
_RE_SWITCH_CASE = re.compile(r"(\S+)\s+(-?\d+),\s*label\s+%(\S+)")
_RE_FENCE = re.compile(
    r"^fence\s+(?:syncscope\(\"(\w+)\"\)\s+)?(monotonic|acquire|release|acq_rel|seq_cst)$"
)

# ── Fallback GEP / ptr patterns (pre-compiled for hot helpers) ──────
_RE_GEP_INSTRUCTION_FALLBACK = re.compile(
    r"^(" + _SSA_NAME_RE + r")\s*=\s*getelementptr(?:\s+\w+)*\s+(.+)$"
)
_RE_PTR_SPEC = re.compile(r"^ptr(?:\s+addrspace\((\d+)\))?\s+(.+)$")
_RE_GEP_FLAG_STRIP = re.compile(r"^(?:inbounds|nuw|nsw|inrange)\s+")

# ── Opcode dispatch sets for codegen fast-path (PERF-005) ───────────
_BINOP_OPCODES = frozenset(
    {
        "add",
        "sub",
        "mul",
        "udiv",
        "sdiv",
        "urem",
        "srem",
        "shl",
        "lshr",
        "ashr",
        "and",
        "or",
        "xor",
        "fadd",
        "fsub",
        "fmul",
        "fdiv",
        "frem",
    }
)
_CAST_OPCODES = frozenset(
    {
        "sext",
        "zext",
        "trunc",
        "fptrunc",
        "fpext",
        "sitofp",
        "uitofp",
        "fptosi",
        "fptoui",
        "bitcast",
        "addrspacecast",
        "ptrtoint",
        "inttoptr",
    }
)


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
        "kernel",
        "vertex",
        "fragment",
        "compute",
        "thread",
        "threadgroup",
        "device",
        "constant",
        "bool",
        "char",
        "short",
        "int",
        "long",
        "half",
        "float",
        "double",
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
    }
)

_UNSIGNED_MSL_MAP = {
    "bool": "bool",
    "char": "unsigned char",
    "short": "unsigned short",
    "int": "unsigned int",
    "long": "unsigned long",
}

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

    def __post_init__(self):
        extern_libs = {} if self.extern_libs is None else dict(self.extern_libs)
        object.__setattr__(self, "extern_libs", tuple(extern_libs.items()))
        assert (
            self.num_warps > 0 and (self.num_warps & (self.num_warps - 1)) == 0
        ), "num_warps must be a power of 2"

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

    def get_codegen_implementation(self, options):
        return {"min_dot_size": lambda lhs_type, rhs_type: (1, 1, 1)}

    def get_module_map(self) -> Dict[str, ModuleType]:
        return {}

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

    @staticmethod
    def make_ttgir(mod, metadata, opt):
        # Metal GPU family arch string -> numeric for pass config
        # Apple Silicon uses SIMD width 32
        pm = ir.pass_manager(mod.context)
        pm.enable_debug()
        passes.ttir.add_convert_to_ttgpuir(
            pm, f"metal:{opt.arch}", opt.num_warps, 32, opt.num_ctas
        )
        passes.ttgpuir.add_coalesce(pm)
        passes.ttgpuir.add_remove_layout_conversions(pm)
        passes.ttgpuir.add_optimize_thread_locality(pm)
        # passes.ttgpuir.add_accelerate_matmul(pm)
        passes.ttgpuir.add_remove_layout_conversions(pm)
        passes.ttgpuir.add_optimize_dot_operands(pm, True)
        if opt.num_stages != 0:
            passes.ttgpuir.add_pipeline(pm, opt.num_stages, False)
        passes.ttir.add_loop_aware_cse(pm)
        passes.ttir.add_triton_licm(pm)
        passes.common.add_canonicalizer(pm)
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

        def split_top_level(text: str, sep: str = ",") -> list[str]:
            parts = []
            cur = []
            depth = 0
            for ch in text:
                if ch in "([{<":
                    depth += 1
                elif ch in ")]}>":
                    depth = max(0, depth - 1)
                if ch == sep and depth == 0:
                    part = "".join(cur).strip()
                    if part:
                        parts.append(part)
                    cur = []
                    continue
                cur.append(ch)
            tail = "".join(cur).strip()
            if tail:
                parts.append(tail)
            return parts

        _msl_id_used: dict[str, str] = {}  # msl_name -> llvm_name that claimed it

        def msl_id(llvm_name: str) -> str:
            raw = llvm_name.lstrip("%")
            # Only invoke regex sub when there are chars that need cleaning;
            # most SSA names (e.g. "0", "v1", "arg0") are already clean.
            if _RE_MSL_ID_CLEAN.search(raw):
                raw = _RE_MSL_ID_CLEAN.sub("_", raw)
            if not raw:
                raw = "tmp"
            if raw[0].isdigit():
                raw = f"v{raw}"
            if raw in _MSL_RESERVED_IDENTIFIERS:
                raw = f"v_{raw}"
            # Disambiguate collisions: different LLVM names (e.g. %foo.bar
            # vs %foo_bar) can map to the same MSL identifier after
            # character replacement.
            owner = _msl_id_used.get(raw)
            if owner is not None and owner != llvm_name:
                suffix = 2
                candidate = f"{raw}_{suffix}"
                while candidate in _msl_id_used:
                    suffix += 1
                    candidate = f"{raw}_{suffix}"
                raw = candidate
            _msl_id_used[raw] = llvm_name
            return raw

        def llvm_scalar_to_msl(llvm_ty: str) -> str:
            return _LLVM_SCALAR_TO_MSL.get(llvm_ty.strip(), "int")

        def unsigned_msl(msl_ty: str) -> str:
            return _UNSIGNED_MSL_MAP.get(msl_ty, f"unsigned {msl_ty}")

        def vector_alias_msl(msl_scalar_ty: str, width: int) -> str:
            aliases = {
                "bool": "bool",
                "char": "char",
                "short": "short",
                "int": "int",
                "long": "long",
                "half": "half",
                "float": "float",
                "double": "double",
                "unsigned char": "uchar",
                "unsigned short": "ushort",
                "unsigned int": "uint",
                "unsigned long": "ulong",
            }
            base = aliases.get(msl_scalar_ty)
            if base is None:
                return f"vec<{msl_scalar_ty}, {width}>"
            return f"{base}{width}"

        def llvm_type_to_msl(llvm_ty: str) -> str:
            llvm_ty = llvm_ty.strip()
            ptr_match = _RE_PTR_TYPE.match(llvm_ty)
            if ptr_match:
                addr_space = ptr_match.group(1)
                if addr_space == "3":
                    return "threadgroup uint*"
                return "device uint*"
            vec_match = _RE_VEC_TYPE.match(llvm_ty)
            if vec_match:
                width = int(vec_match.group(1))
                scalar = llvm_scalar_to_msl(vec_match.group(2))
                if width == 1:
                    return scalar
                return f"vec<{scalar}, {width}>"
            return llvm_scalar_to_msl(llvm_ty)

        def msl_addr_space(addr_space: str | None) -> str:
            return "threadgroup" if addr_space == "3" else "device"

        def ptr_type_to_msl(pointee_llvm_ty: str, addr_space: str | None = None) -> str:
            space = msl_addr_space(addr_space)
            return f"{space} {llvm_scalar_to_msl(pointee_llvm_ty)}*"

        def extract_call_ret_type(ret_spec: str) -> str:
            ret_spec = ret_spec.strip()
            if not ret_spec:
                return "void"
            if "void" in ret_spec.split():
                return "void"
            if "{" in ret_spec:
                start = ret_spec.index("{")
                depth = 0
                for i in range(start, len(ret_spec)):
                    if ret_spec[i] == "{":
                        depth += 1
                    elif ret_spec[i] == "}":
                        depth -= 1
                    if depth == 0:
                        return ret_spec[start : i + 1]
                return ret_spec[start:]
            vec_match = _RE_CALL_RET_VEC.search(ret_spec)
            if vec_match:
                return vec_match.group(0)
            ptr_match = _RE_CALL_RET_PTR.search(ret_spec)
            if ptr_match:
                return ptr_match.group(0)
            scalar_matches = _RE_CALL_RET_SCALAR.findall(ret_spec)
            if scalar_matches:
                return scalar_matches[-1]
            return "i32"

        def constant_to_msl(token: str) -> str:
            token = token.strip()
            if token in ("undef", "poison", "zeroinitializer"):
                return "0"
            if token in ("true", "false", "nullptr", "null"):
                return "nullptr" if token == "null" else token
            if token.startswith("<") and token.endswith(">"):
                inner = token[1:-1].strip()
                elems = split_top_level(inner)
                if elems:
                    elem_vals = []
                    elem_msl_ty = None
                    vector_ok = True
                    for elem in elems:
                        llvm_ty, val = split_typed_value(elem)
                        llvm_ty = llvm_ty.strip()
                        if not llvm_ty:
                            vector_ok = False
                            break
                        cur_msl_ty = llvm_scalar_to_msl(llvm_ty)
                        if elem_msl_ty is None:
                            elem_msl_ty = cur_msl_ty
                        elif elem_msl_ty != cur_msl_ty:
                            vector_ok = False
                            break
                        elem_vals.append(constant_to_msl(val))
                    if vector_ok and elem_msl_ty is not None:
                        return f"{elem_msl_ty}{len(elem_vals)}({', '.join(elem_vals)})"
            if _RE_CONST_INT.match(token):
                return token
            # LLVM IR hex float: 0x followed by 16 hex digits encoding an
            # IEEE-754 double. Convert to the actual floating-point value
            # so MSL receives a numeric literal, not a huge integer.
            hex_m = _RE_CONST_HEX_FLOAT.match(token)
            if hex_m:
                raw = int(hex_m.group(1), 16)
                dval = struct.unpack("d", struct.pack("Q", raw))[0]
                if math.isinf(dval):
                    return "-INFINITY" if dval < 0 else "INFINITY"
                if math.isnan(dval):
                    return "NAN"
                return f"{dval!r}f"
            # LLVM IR half-precision hex float: 0xH followed by 4 hex digits
            # encoding an IEEE-754 binary16 value.
            hex_h = _RE_CONST_HEX_HALF.match(token)
            if hex_h:
                raw16 = int(hex_h.group(1), 16)
                # Decode IEEE-754 binary16 → Python float
                sign = (raw16 >> 15) & 1
                exp = (raw16 >> 10) & 0x1F
                frac = raw16 & 0x3FF
                if exp == 0:
                    hval = (-1) ** sign * (2**-14) * (frac / 1024.0)
                elif exp == 0x1F:
                    if frac:
                        return "NAN"
                    return "-INFINITY" if sign else "INFINITY"
                else:
                    hval = (-1) ** sign * (2 ** (exp - 15)) * (1.0 + frac / 1024.0)
                if hval == 0.0 and sign:
                    return "(-0.0h)"
                return f"(half)({hval!r}f)"
            # LLVM IR bfloat16 hex float: 0xR followed by 4 hex digits.
            hex_bf = _RE_CONST_HEX_BFLOAT.match(token)
            if hex_bf:
                raw_bf = int(hex_bf.group(1), 16)
                # bfloat16 is the upper 16 bits of an IEEE-754 float32
                f32_bits = raw_bf << 16
                fval = struct.unpack("f", struct.pack("I", f32_bits))[0]
                if math.isinf(fval):
                    return "-INFINITY" if fval < 0 else "INFINITY"
                if math.isnan(fval):
                    return "NAN"
                return f"{fval!r}f"
            if _RE_CONST_FLOAT.match(token):
                return token if token.endswith("f") else f"{token}f"
            return token

        def parse_call_args(arg_list: str) -> list[str]:
            values = []
            for arg in split_top_level(arg_list):
                arg = arg.strip()
                if not arg:
                    continue
                values.append(extract_value_token(arg))
            return values

        def extract_value_token(spec: str) -> str:
            spec = spec.strip()
            if spec.startswith("<") and spec.endswith(">"):
                return spec
            if spec.startswith("{") and spec.endswith("}"):
                return spec
            if "%" in spec:
                return spec[spec.rfind("%") :].strip()
            if "@" in spec:
                return spec[spec.rfind("@") :].strip()
            return spec.split()[-1].strip()

        def split_typed_value(spec: str) -> tuple[str, str]:
            spec = spec.strip()
            value = extract_value_token(spec)
            idx = spec.rfind(value)
            llvm_ty = spec[:idx].strip() if idx >= 0 else spec
            return llvm_ty, value

        def strip_operand_attrs(spec: str) -> str:
            spec = spec.strip()
            spec = _RE_ALIGN_STRIP.sub("", spec)
            return spec.strip()

        def parse_ptr_spec(spec: str) -> tuple[str | None, str] | None:
            m = _RE_PTR_SPEC.match(spec.strip())
            if not m:
                return None
            return m.group(1), m.group(2).strip()

        def parse_gep_components(spec: str) -> tuple[str, str | None, str, str] | None:
            parts = split_top_level(spec)
            if len(parts) < 3:
                return None
            elem_ty = parts[0].strip()
            ptr_info = parse_ptr_spec(parts[1])
            if ptr_info is None:
                return None
            addr_space, base = ptr_info
            _, idx_token = split_typed_value(parts[2])
            return elem_ty, addr_space, base, idx_token

        def parse_gep_instruction(
            line: str,
        ) -> tuple[str, str, str | None, str, str] | None:
            m = _RE_GEP_INSTRUCTION_FALLBACK.match(line)
            if not m:
                return None
            out_ssa = m.group(1)
            comps = parse_gep_components(m.group(2))
            if comps is None:
                return None
            elem_ty, addr_space, base, idx_token = comps
            return out_ssa, elem_ty, addr_space, base, idx_token

        def parse_gep_constexpr(token: str) -> tuple[str, str | None, str, str] | None:
            if not token.startswith("getelementptr"):
                return None
            rest = token[len("getelementptr") :].strip()
            while True:
                stripped = _RE_GEP_FLAG_STRIP.sub("", rest, count=1)
                if stripped == rest:
                    break
                rest = stripped
            if rest.startswith("(") and rest.endswith(")"):
                rest = rest[1:-1].strip()
            comps = parse_gep_components(rest)
            if comps is None:
                return None
            elem_ty, addr_space, base, idx_token = comps
            return elem_ty, addr_space, base, idx_token

        def normalize_label(label: str) -> str:
            label = label.strip()
            if label.startswith('"') and label.endswith('"'):
                return label[1:-1]
            return label

        func_header = _RE_KERNEL_FUNC.search(src)
        if not func_header:
            raise RuntimeError("No kernel function found in LLVM IR")

        kernel_name = func_header.group(1)
        reserved = {"kernel", "vertex", "fragment", "compute"}
        msl_kernel_name = kernel_name
        if kernel_name in reserved:
            msl_kernel_name = f"triton_{kernel_name}"
        metadata["name"] = msl_kernel_name

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
            ptr_elem[p["llvm_name"]] = llvm_scalar_to_msl(pointee)

        aggregate_type_structs: dict[str, tuple[str, list[str]]] = {}
        struct_defs: list[str] = []

        def get_aggregate_struct_name(agg_type_str: str) -> tuple[str, list[str]]:
            agg_type_str = agg_type_str.strip()
            if agg_type_str in aggregate_type_structs:
                return aggregate_type_structs[agg_type_str]
            idx = len(aggregate_type_structs)
            name = f"__triton_aggr_{idx}"
            inner = agg_type_str.strip("{ }")
            field_types = [llvm_type_to_msl(t.strip()) for t in inner.split(",")]
            aggregate_type_structs[agg_type_str] = (name, field_types)
            fields = "".join(f"  {ft} field{i};\n" for i, ft in enumerate(field_types))
            struct_defs.append(f"struct {name} {{\n{fields}}};")
            return name, field_types

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
                scalar_ty = llvm_scalar_to_msl(p["llvm_type"])
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

        def to_expr(token: str) -> str:
            token = token.strip()
            # Fast path: SSA name lookup (most common case, ~60% of calls)
            cached = ssa.get(token)
            if cached is not None:
                return cached
            if token.startswith("%"):
                out = msl_id(token)
                ssa[token] = out
                return out
            if token == "@global_smem":
                return "((threadgroup char*)__triton_shared)"
            gep_cexpr = parse_gep_constexpr(token)
            if gep_cexpr is not None:
                _, _, base, idx_token = gep_cexpr
                return f"({to_expr(base)} + {to_expr(idx_token)})"
            return constant_to_msl(token)

        cmp_map = _CMP_MAP
        float_bin_map = _FLOAT_BIN_MAP

        def lower_intrinsic(fn: str, args: list[str]) -> str | None:
            nargs = len(args)

            def fold_infix(terms: list[str], op: str) -> str:
                expr = terms[0]
                for term in terms[1:]:
                    expr = f"({expr} {op} {term})"
                return expr

            def fold_func(terms: list[str], fn_name: str) -> str:
                expr = terms[0]
                for term in terms[1:]:
                    expr = f"{fn_name}({expr}, {term})"
                return expr

            def lower_vector_reduce() -> str | None:
                m = _RE_LLVM_VECTOR_REDUCE.match(fn)
                if not m:
                    return None
                reduce_op = m.group(1)
                lanes = int(m.group(2))
                elem_ty = m.group(3)
                if lanes <= 0:
                    return None

                init: str | None = None
                vec_arg: str | None = None
                if reduce_op in (
                    "fadd",
                    "fmul",
                    "fmax",
                    "fmin",
                    "fmaximum",
                    "fminimum",
                ):
                    if nargs == 1:
                        vec_arg = args[0]
                    elif nargs == 2:
                        init, vec_arg = args
                    else:
                        return None
                else:
                    if nargs != 1:
                        return None
                    vec_arg = args[0]

                terms = [f"({vec_arg}[{i}])" for i in range(lanes)]
                if reduce_op in ("umax", "umin") and elem_ty.startswith("i"):
                    u_ty = unsigned_msl(llvm_scalar_to_msl(elem_ty))
                    terms = [f"(({u_ty}){term})" for term in terms]
                    if init is not None:
                        init = f"(({u_ty})({init}))"

                if reduce_op in ("or", "and", "xor", "add", "mul", "fadd", "fmul"):
                    op_map = {
                        "or": "|",
                        "and": "&",
                        "xor": "^",
                        "add": "+",
                        "mul": "*",
                        "fadd": "+",
                        "fmul": "*",
                    }
                    expr = fold_infix(terms, op_map[reduce_op])
                    if init is not None:
                        expr = fold_infix([f"({init})", f"({expr})"], op_map[reduce_op])
                    return expr

                if reduce_op in ("smax", "umax", "fmax", "fmaximum"):
                    expr = fold_func(terms, "max")
                    if init is not None:
                        expr = f"max(({init}), ({expr}))"
                    return expr

                if reduce_op in ("smin", "umin", "fmin", "fminimum"):
                    expr = fold_func(terms, "min")
                    if init is not None:
                        expr = f"min(({init}), ({expr}))"
                    return expr

                return None

            reduced = lower_vector_reduce()
            if reduced is not None:
                return reduced

            # Table-driven simple intrinsics (DUP-001 consolidation)
            if nargs == 1:
                for prefix, builtin in _LLVM_INTRINSIC_UNARY:
                    if fn.startswith(prefix):
                        return f"{builtin}({args[0]})"

            if nargs >= 1:
                for prefix, builtin in _LLVM_INTRINSIC_UNARY_RELAXED:
                    if fn.startswith(prefix):
                        return f"{builtin}({args[0]})"

            if nargs == 2:
                for prefix, builtin in _LLVM_INTRINSIC_BINARY:
                    if fn.startswith(prefix):
                        return f"{builtin}({args[0]}, {args[1]})"
                if any(fn.startswith(p) for p in _LLVM_INTRINSIC_MAX_PREFIXES):
                    return f"max({args[0]}, {args[1]})"
                if any(fn.startswith(p) for p in _LLVM_INTRINSIC_MIN_PREFIXES):
                    return f"min({args[0]}, {args[1]})"

            if nargs == 3:
                for prefix, builtin in _LLVM_INTRINSIC_TERNARY:
                    if fn.startswith(prefix):
                        return f"{builtin}({args[0]}, {args[1]}, {args[2]})"

            # Special-case intrinsics that need inline expansion
            if fn == "llvm.bswap.i32" and nargs == 1:
                a = args[0]
                return (
                    f"((({a}) >> 24) | ((({a}) >> 8) & 0xFF00) | "
                    f"((({a}) << 8) & 0xFF0000) | (({a}) << 24))"
                )
            if fn == "llvm.bswap.i64" and nargs == 1:
                a = args[0]
                return (
                    f"(((unsigned long)({a}) >> 56) | "
                    f"(((unsigned long)({a}) >> 40) & 0xFF00UL) | "
                    f"(((unsigned long)({a}) >> 24) & 0xFF0000UL) | "
                    f"(((unsigned long)({a}) >> 8) & 0xFF000000UL) | "
                    f"(((unsigned long)({a}) << 8) & 0xFF00000000UL) | "
                    f"(((unsigned long)({a}) << 24) & 0xFF0000000000UL) | "
                    f"(((unsigned long)({a}) << 40) & 0xFF000000000000UL) | "
                    f"((unsigned long)({a}) << 56))"
                )
            if fn.startswith("llvm.fshr.") and nargs == 3:
                bits = "32" if "i32" in fn else "64"
                u_ty = "unsigned int" if "i32" in fn else "unsigned long"
                a, b, c = args[0], args[1], args[2]
                # Guard against UB: when shift % bits == 0, shifting by
                # the full bit width is undefined in C/MSL.  Use a
                # ternary so the complementary shift is only evaluated
                # when the amount is non-zero.
                return (
                    f"(({c} & ({bits} - 1)) == 0 ? ({u_ty})({b}) : "
                    f"(({u_ty})({b}) >> ({c} & ({bits} - 1))) | "
                    f"(({u_ty})({a}) << ({bits} - ({c} & ({bits} - 1)))))"
                )
            if fn.startswith("llvm.fshl.") and nargs == 3:
                bits = "32" if "i32" in fn else "64"
                u_ty = "unsigned int" if "i32" in fn else "unsigned long"
                a, b, c = args[0], args[1], args[2]
                return (
                    f"(({c} & ({bits} - 1)) == 0 ? ({u_ty})({a}) : "
                    f"(({u_ty})({a}) << ({c} & ({bits} - 1))) | "
                    f"(({u_ty})({b}) >> ({bits} - ({c} & ({bits} - 1)))))"
                )
            if fn.startswith("llvm.powi.") and nargs == 2:
                return f"pown({args[0]}, {args[1]})"

            # LLVM IR emitted by shared Triton pipelines can still reference
            # CUDA/OCML-style libdevice symbols. Lower these to equivalent MSL
            # math builtins so Metal compilation remains backend-agnostic.
            if nargs == 1:
                for pat, builtin in _LIBDEVICE_UNARY:
                    if pat.match(fn):
                        return f"{builtin}({args[0]})"

            if nargs == 2:
                for pat, builtin in _LIBDEVICE_BINARY:
                    if pat.match(fn):
                        return f"{builtin}({args[0]}, {args[1]})"

            if nargs == 3 and _LIBDEVICE_FMA.match(fn):
                return f"fma({args[0]}, {args[1]}, {args[2]})"
            return None

        def fcmp_expr(pred: str, lhs: str, rhs: str) -> str:
            ordered = f"(!isnan({lhs}) && !isnan({rhs}))"
            unordered = f"(isnan({lhs}) || isnan({rhs}))"
            table = {
                "false": "false",
                "true": "true",
                "oeq": f"({ordered} && ({lhs} == {rhs}))",
                "ogt": f"({ordered} && ({lhs} > {rhs}))",
                "oge": f"({ordered} && ({lhs} >= {rhs}))",
                "olt": f"({ordered} && ({lhs} < {rhs}))",
                "ole": f"({ordered} && ({lhs} <= {rhs}))",
                "one": f"({ordered} && ({lhs} != {rhs}))",
                "ord": ordered,
                "ueq": f"({unordered} || ({lhs} == {rhs}))",
                "ugt": f"({unordered} || ({lhs} > {rhs}))",
                "uge": f"({unordered} || ({lhs} >= {rhs}))",
                "ult": f"({unordered} || ({lhs} < {rhs}))",
                "ule": f"({unordered} || ({lhs} <= {rhs}))",
                "une": f"({unordered} || ({lhs} != {rhs}))",
                "uno": unordered,
            }
            if pred not in table:
                raise RuntimeError(f"Unsupported fcmp predicate '{pred}'")
            return table[pred]

        bin_map = _BIN_MAP
        axis_helper_map = _AXIS_HELPER_MAP

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
        ssa_decl_types: dict[str, str] = {}

        def record_ssa_decl(
            out_ssa: str, llvm_ty: str | None = None, msl_ty: str | None = None
        ) -> None:
            out = msl_id(out_ssa)
            ssa[out_ssa] = out
            if out in param_ids or out in ssa_decl_types:
                return
            resolved = (
                msl_ty if msl_ty is not None else llvm_type_to_msl(llvm_ty or "i32")
            )
            ssa_decl_types[out] = resolved

        for block in block_order:
            for line in blocks.get(block, []):
                # Lines that don't start with '%' cannot produce SSA
                # declarations — skip all regex testing for them.
                if not line.startswith("%"):
                    continue

                _opc = _extract_ir_opcode(line)

                # Reordered by frequency: binop > load > cast > call > GEP >
                # icmp/fcmp > phi > select > fneg > freeze > extract/insert
                m = _RE_BINOP.match(line) if _opc in _BINOP_OPCODES else None
                if m:
                    out_ssa, _, operands_spec = m.groups()
                    parts = split_top_level(operands_spec)
                    if len(parts) != 2:
                        continue
                    llvm_ty, _ = split_typed_value(parts[0])
                    record_ssa_decl(out_ssa, llvm_ty=llvm_ty)
                    continue

                m = _RE_LOAD_DECL.match(line) if _opc == "load" else None
                if m:
                    out_ssa, llvm_ty, _ = m.groups()
                    record_ssa_decl(out_ssa, llvm_ty=llvm_ty)
                    continue

                m = _RE_CAST.match(line) if _opc in _CAST_OPCODES else None
                if m:
                    out_ssa, _, _, dst_ty = m.groups()
                    record_ssa_decl(out_ssa, llvm_ty=dst_ty.strip())
                    continue

                m = _RE_CALL_OUT.match(line) if _opc == "call" else None
                if m:
                    out_ssa, ret_spec, fn_name, _ = m.groups()
                    ret_type = extract_call_ret_type(ret_spec)
                    if fn_name in (
                        "__metal_simdgroup_load",
                        "__metal_simdgroup_multiply_accumulate",
                    ):
                        elem_ty = "float"
                        vec_m = _RE_VEC_TYPE.match(ret_type)
                        if vec_m:
                            elem_ty = llvm_scalar_to_msl(vec_m.group(2))
                        record_ssa_decl(
                            out_ssa,
                            msl_ty=f"simdgroup_matrix<{elem_ty}, 8, 8>",
                        )
                    elif ret_type.startswith("{"):
                        struct_name, _ = get_aggregate_struct_name(ret_type)
                        record_ssa_decl(out_ssa, msl_ty=struct_name)
                    else:
                        record_ssa_decl(out_ssa, llvm_ty=ret_type)
                    continue

                m = _RE_GEP_DECL.match(line) if _opc == "getelementptr" else None
                if m:
                    out_ssa, elem_ty, addr_space, _, _ = m.groups()
                    record_ssa_decl(
                        out_ssa, msl_ty=ptr_type_to_msl(elem_ty, addr_space=addr_space)
                    )
                    continue

                parsed_gep = parse_gep_instruction(line)
                if parsed_gep is not None:
                    out_ssa, elem_ty, addr_space, _, _ = parsed_gep
                    record_ssa_decl(
                        out_ssa, msl_ty=ptr_type_to_msl(elem_ty, addr_space=addr_space)
                    )
                    continue

                m = _RE_ICMP.match(line) if _opc == "icmp" else None
                if m:
                    out_ssa, _, _, _, _ = m.groups()
                    record_ssa_decl(out_ssa, msl_ty="bool")
                    continue

                m = _RE_FCMP.match(line) if _opc == "fcmp" else None
                if m:
                    out_ssa, _, _, _ = m.groups()
                    record_ssa_decl(out_ssa, msl_ty="bool")
                    continue

                m = _RE_PHI_DECL.match(line) if _opc == "phi" else None
                if m:
                    out_ssa, llvm_ty = m.groups()
                    record_ssa_decl(out_ssa, llvm_ty=llvm_ty)
                    continue

                m = _RE_SELECT_DECL.match(line) if _opc == "select" else None
                if m:
                    out_ssa, llvm_ty = m.groups()
                    record_ssa_decl(out_ssa, llvm_ty=llvm_ty)
                    continue

                m = _RE_FNEG_DECL.match(line) if _opc == "fneg" else None
                if m:
                    out_ssa, llvm_ty, _ = m.groups()
                    record_ssa_decl(out_ssa, llvm_ty=llvm_ty)
                    continue

                m = _RE_FREEZE_DECL.match(line) if _opc == "freeze" else None
                if m:
                    out_ssa, llvm_ty, _ = m.groups()
                    record_ssa_decl(out_ssa, llvm_ty=llvm_ty)
                    continue

                m = (
                    _RE_EXTRACTELEM_DECL.match(line)
                    if _opc == "extractelement"
                    else None
                )
                if m:
                    out_ssa, elem_ty, _, _ = m.groups()
                    record_ssa_decl(out_ssa, llvm_ty=elem_ty)
                    continue

                m = _RE_INSERTELEM_DECL.match(line) if _opc == "insertelement" else None
                if m:
                    out_ssa, vec_ty, _, _, _ = m.groups()
                    record_ssa_decl(out_ssa, llvm_ty=vec_ty)
                    continue

                m = (
                    _RE_SHUFFLEVECTOR_DECL.match(line)
                    if _opc == "shufflevector"
                    else None
                )
                if m:
                    out_ssa, _, elem_ty, _, _, _, out_width_s, _ = m.groups()
                    out_width = int(out_width_s)
                    scalar_ty = llvm_scalar_to_msl(elem_ty.strip())
                    if out_width <= 1:
                        record_ssa_decl(out_ssa, msl_ty=scalar_ty)
                    else:
                        record_ssa_decl(out_ssa, msl_ty=f"{scalar_ty}{out_width}")
                    continue

                m = _RE_EXTRACTVALUE.match(line) if _opc == "extractvalue" else None
                if m:
                    out_ssa, agg_type, _, idx_str = m.groups()
                    _, field_types = get_aggregate_struct_name(agg_type)
                    idx = int(idx_str)
                    ft = field_types[idx] if idx < len(field_types) else "int"
                    record_ssa_decl(out_ssa, msl_ty=ft)
                    continue

                m = _RE_INSERTVALUE.match(line) if _opc == "insertvalue" else None
                if m:
                    out_ssa, agg_type, _, _, _, _ = m.groups()
                    struct_name, _ = get_aggregate_struct_name(agg_type)
                    record_ssa_decl(out_ssa, msl_ty=struct_name)
                    continue

                m = _RE_ATOMICRMW.match(line) if _opc == "atomicrmw" else None
                if m:
                    out_ssa = m.group(1)
                    val_type = m.group(5)
                    record_ssa_decl(out_ssa, llvm_ty=val_type.strip())
                    continue

                m = _RE_CMPXCHG.match(line) if _opc == "cmpxchg" else None
                if m:
                    out_ssa = m.group(1)
                    val_type = m.group(4)
                    agg_type = "{" + val_type.strip() + ", i1}"
                    struct_name, _ = get_aggregate_struct_name(agg_type)
                    record_ssa_decl(out_ssa, msl_ty=struct_name)
                    continue

                m = _RE_ALLOCA.match(line) if _opc == "alloca" else None
                if m:
                    out_ssa, elem_type = m.groups()
                    msl_ty = llvm_scalar_to_msl(elem_type.strip())
                    record_ssa_decl(out_ssa, msl_ty=f"thread {msl_ty}*")
                    storage_name = f"{msl_id(out_ssa)}_storage"
                    if storage_name not in ssa_decl_types:
                        ssa_decl_types[storage_name] = msl_ty
                    continue

        body_lines = [
            "  int __triton_pred_block = -1;",
            f"  int __pc = {block_ids['entry']};",
        ]
        if uses_shared_smem:
            body_lines.insert(0, f"  threadgroup char __triton_shared[{shared_bytes}];")
        body_lines.extend(
            [f"  {msl_ty} {name};" for name, msl_ty in ssa_decl_types.items()]
        )
        body_lines.extend(
            [
                "  while (true) {",
                "    switch (__pc) {",
            ]
        )

        best_effort = getattr(opt, "best_effort", False) if opt is not None else False
        unsupported_lines: list[UnsupportedIREntry] = []
        all_codegen_lines: list[str] = []
        for blk in block_order:
            all_codegen_lines.extend(blocks.get(blk, []))

        for block in block_order:
            block_id = block_ids[block]
            instrs = blocks.get(block, [])
            body_lines.append(f"    case {block_id}: {{")

            # Shared-memory dot/staging loops lowered from TTGIR can arrive
            # without explicit barrier ops in LLIR. Detect the canonical
            # pattern (shared stores + shared loads + loop backedge) and
            # conservatively inject threadgroup barriers at the translation
            # boundary to preserve correctness across simdgroups.
            has_tg_store = False
            has_tg_load = False
            has_backedge = False
            for scan_line in instrs:
                scan_opc = _extract_ir_opcode(scan_line)
                scan_store = _RE_STORE.match(scan_line) if scan_opc == "store" else None
                if scan_store and scan_store.group(2) == "3":
                    has_tg_store = True
                scan_load = _RE_LOAD.match(scan_line) if scan_opc == "load" else None
                if scan_load and scan_load.group(3) == "3":
                    has_tg_load = True
                if scan_opc == "br":
                    scan_br = _RE_BR.match(scan_line)
                    if scan_br is not None:
                        target = normalize_label(scan_br.group(1))
                        target_id = block_ids.get(target)
                        if target_id is not None and target_id <= block_id:
                            has_backedge = True
                    scan_cond = _RE_BR_COND.match(scan_line)
                    if scan_cond is not None:
                        t_lbl = normalize_label(scan_cond.group(2))
                        f_lbl = normalize_label(scan_cond.group(3))
                        t_id = block_ids.get(t_lbl)
                        f_id = block_ids.get(f_lbl)
                        if (t_id is not None and t_id <= block_id) or (
                            f_id is not None and f_id <= block_id
                        ):
                            has_backedge = True
            needs_tg_loop_sync = has_tg_store and has_tg_load and has_backedge
            inserted_tg_sync_before_load = False

            def emit(stmt: str):
                body_lines.append(f"      {stmt}")

            terminated = False
            for line in instrs:
                if line == "ret void":
                    emit("return;")
                    terminated = True
                    break

                _opc = _extract_ir_opcode(line)

                # Reordered by frequency: binop > load > store > GEP > cast >
                # call > icmp > fcmp > br > phi > void_call > select > fneg >
                # freeze > extract/insert > unreachable

                m = _RE_BINOP.match(line) if _opc in _BINOP_OPCODES else None
                if m:
                    out_ssa, op, operands_spec = m.groups()
                    parts = split_top_level(operands_spec)
                    if len(parts) != 2:
                        raise RuntimeError(
                            f"Unsupported binary operand form in Metal lowering: '{line}'"
                        )
                    llvm_ty_binop, lhs = split_typed_value(parts[0])
                    _, rhs = split_typed_value(parts[1])
                    out = msl_id(out_ssa)
                    ssa[out_ssa] = out
                    lhs_expr = to_expr(lhs)
                    rhs_expr = to_expr(rhs)
                    if op in float_bin_map:
                        emit(f"{out} = {lhs_expr} {float_bin_map[op]} {rhs_expr};")
                    elif op == "frem":
                        emit(f"{out} = fmod({lhs_expr}, {rhs_expr});")
                    elif op in ("lshr", "udiv", "urem"):
                        # These LLVM IR ops have unsigned semantics but MSL
                        # integer types are signed.  Cast to unsigned before
                        # the operation to preserve correctness.
                        vec_ty = _RE_VEC_TYPE.match(llvm_ty_binop.strip())
                        if vec_ty:
                            lanes = int(vec_ty.group(1))
                            scalar_ty = llvm_scalar_to_msl(vec_ty.group(2))
                            msl_ty = (
                                vector_alias_msl(scalar_ty, lanes)
                                if lanes > 1
                                else scalar_ty
                            )
                            u_scalar = unsigned_msl(scalar_ty)
                            u_ty = (
                                vector_alias_msl(u_scalar, lanes)
                                if lanes > 1
                                else u_scalar
                            )
                        else:
                            msl_ty = llvm_scalar_to_msl(llvm_ty_binop)
                            u_ty = unsigned_msl(msl_ty)
                        emit(
                            f"{out} = ({msl_ty})(({u_ty}){lhs_expr} "
                            f"{bin_map[op]} ({u_ty}){rhs_expr});"
                        )
                    else:
                        emit(f"{out} = {lhs_expr} {bin_map[op]} {rhs_expr};")
                    continue

                m = _RE_LOAD.match(line) if _opc == "load" else None
                if m:
                    out_ssa, llvm_ty, addr_space, ptr = m.groups()
                    out = msl_id(out_ssa)
                    ssa[out_ssa] = out
                    llvm_ty = llvm_ty.strip()
                    if (
                        needs_tg_loop_sync
                        and addr_space == "3"
                        and not inserted_tg_sync_before_load
                    ):
                        emit("threadgroup_barrier(mem_flags::mem_threadgroup);")
                        inserted_tg_sync_before_load = True
                    ptr_expr = to_expr(strip_operand_attrs(ptr))
                    msl_ty = llvm_type_to_msl(llvm_ty)
                    emit(
                        f"{out} = *(({msl_addr_space(addr_space)} {msl_ty}*)({ptr_expr}));"
                    )
                    continue

                m = _RE_STORE.match(line) if _opc == "store" else None
                if m:
                    val_spec, addr_space, ptr = m.groups()
                    llvm_ty, val_token = split_typed_value(val_spec)
                    msl_ty = llvm_type_to_msl(llvm_ty)
                    ptr_expr = to_expr(strip_operand_attrs(ptr))
                    emit(
                        f"*(({msl_addr_space(addr_space)} {msl_ty}*)({ptr_expr})) = {to_expr(val_token)};"
                    )
                    continue

                m = _RE_GEP.match(line) if _opc == "getelementptr" else None
                if m:
                    out_ssa, _, base, idx = m.groups()
                    out = msl_id(out_ssa)
                    ssa[out_ssa] = out
                    emit(f"{out} = {to_expr(base)} + {to_expr(idx)};")
                    continue

                parsed_gep = (
                    parse_gep_instruction(line) if _opc == "getelementptr" else None
                )
                if parsed_gep is not None:
                    out_ssa, _, _, base, idx = parsed_gep
                    out = msl_id(out_ssa)
                    ssa[out_ssa] = out
                    emit(f"{out} = {to_expr(base)} + {to_expr(idx)};")
                    continue

                m = _RE_CAST.match(line) if _opc in _CAST_OPCODES else None
                if m:
                    out_ssa, op, src_spec, dst_ty = m.groups()
                    out = msl_id(out_ssa)
                    ssa[out_ssa] = out
                    dst_ty = dst_ty.strip()
                    val = extract_value_token(src_spec)
                    if op in ("bitcast", "addrspacecast"):
                        if dst_ty.startswith("ptr"):
                            emit(f"{out} = {to_expr(val)};")
                        else:
                            emit(
                                f"{out} = as_type<{llvm_type_to_msl(dst_ty)}>({to_expr(val)});"
                            )
                    elif op in ("ptrtoint", "inttoptr"):
                        emit(f"{out} = {to_expr(val)};")
                    else:
                        emit(f"{out} = ({llvm_type_to_msl(dst_ty)})({to_expr(val)});")
                    continue

                m = _RE_CALL_OUT.match(line) if _opc == "call" else None
                if m:
                    out_ssa, _, fn, args_raw = m.groups()
                    args = [to_expr(v) for v in parse_call_args(args_raw)]
                    out = msl_id(out_ssa)
                    ssa[out_ssa] = out
                    if fn.startswith("llvm.sadd.with.overflow.") and len(args) == 2:
                        emit(f"{out}.field0 = {args[0]} + {args[1]};")
                        emit(
                            f"{out}.field1 = (({args[0]} ^ {out}.field0) & ({args[1]} ^ {out}.field0)) < 0;"
                        )
                        continue
                    if fn.startswith("llvm.uadd.with.overflow.") and len(args) == 2:
                        emit(f"{out}.field0 = {args[0]} + {args[1]};")
                        emit(f"{out}.field1 = {out}.field0 < {args[0]};")
                        continue
                    if fn.startswith("llvm.ssub.with.overflow.") and len(args) == 2:
                        emit(f"{out}.field0 = {args[0]} - {args[1]};")
                        emit(
                            f"{out}.field1 = (({args[0]} ^ {args[1]}) & ({args[0]} ^ {out}.field0)) < 0;"
                        )
                        continue
                    if fn.startswith("llvm.usub.with.overflow.") and len(args) == 2:
                        emit(f"{out}.field0 = {args[0]} - {args[1]};")
                        emit(f"{out}.field1 = {args[0]} < {args[1]};")
                        continue
                    lowered_intrinsic = lower_intrinsic(fn, args)
                    if lowered_intrinsic is not None:
                        emit(f"{out} = {lowered_intrinsic};")
                    elif fn in axis_helper_map:
                        emit(f"{out} = {axis_helper_map[fn]};")
                    elif (
                        fn.startswith("__metal_predicated_ld_global_")
                        and len(args) == 3
                    ):
                        out_ty = ssa_decl_types.get(out)
                        if out_ty is None:
                            emit(f"{out} = ({args[2]} ? *{args[1]} : {args[0]});")
                        else:
                            emit(
                                f"{out} = ({args[2]} ? ({out_ty})(*{args[1]}) : ({out_ty})({args[0]}));"
                            )
                    elif fn == "__metal_simd_shuffle_xor" and len(args) == 2:
                        emit(f"{out} = simd_shuffle_xor({args[0]}, {args[1]});")
                    elif fn == "__metal_simd_shuffle_up" and len(args) == 2:
                        emit(f"{out} = simd_shuffle_up({args[0]}, {args[1]});")
                    elif fn == "__metal_simd_shuffle" and len(args) == 2:
                        emit(f"{out} = simd_shuffle({args[0]}, {args[1]});")
                    elif fn == "__metal_simdgroup_load" and len(args) == 2:
                        emit(
                            f"simdgroup_load({out}, "
                            f"(const device float*){args[0]}, {args[1]});"
                        )
                    elif (
                        fn == "__metal_simdgroup_multiply_accumulate" and len(args) == 3
                    ):
                        emit(
                            f"simdgroup_multiply_accumulate("
                            f"{out}, {args[0]}, {args[1]}, {args[2]});"
                        )
                    else:
                        if fn.startswith("llvm."):
                            raise RuntimeError(
                                f"Unsupported LLVM intrinsic in Metal lowering: '{fn}'"
                            )
                        emit(f"{out} = {fn}({', '.join(args)});")
                    continue

                m = _RE_ICMP.match(line) if _opc == "icmp" else None
                if m:
                    out_ssa, pred, llvm_ty_icmp, lhs, rhs = m.groups()
                    out = msl_id(out_ssa)
                    ssa[out_ssa] = out
                    cmp_op = cmp_map.get(pred)
                    if cmp_op is None:
                        raise RuntimeError(f"Unsupported icmp predicate '{pred}'")
                    lhs_expr = to_expr(lhs)
                    rhs_expr = to_expr(rhs)
                    if pred.startswith("u") and pred not in ("eq", "ne"):
                        u_ty = unsigned_msl(llvm_scalar_to_msl(llvm_ty_icmp))
                        lhs_expr = f"({u_ty}){lhs_expr}"
                        rhs_expr = f"({u_ty}){rhs_expr}"
                    emit(f"{out} = ({lhs_expr} {cmp_op} {rhs_expr});")
                    continue

                m = _RE_FCMP.match(line) if _opc == "fcmp" else None
                if m:
                    out_ssa, pred, lhs, rhs = m.groups()
                    out = msl_id(out_ssa)
                    ssa[out_ssa] = out
                    lhs_expr = to_expr(lhs)
                    rhs_expr = to_expr(rhs)
                    emit(f"{out} = {fcmp_expr(pred, lhs_expr, rhs_expr)};")
                    continue

                m = _RE_BR.match(line) if _opc == "br" else None
                if m:
                    target = normalize_label(m.group(1))
                    target_id = block_ids.get(target)
                    if target_id is None:
                        raise RuntimeError(
                            f"Unknown branch target '{target}' in Metal lowering"
                        )
                    if needs_tg_loop_sync and target_id <= block_id:
                        emit("threadgroup_barrier(mem_flags::mem_threadgroup);")
                    emit(f"__triton_pred_block = {block_id};")
                    emit(f"__pc = {target_id};")
                    emit("continue;")
                    terminated = True
                    break

                m = _RE_BR_COND.match(line) if _opc == "br" else None
                if m:
                    cond, t_lbl, f_lbl = m.groups()
                    t_lbl = normalize_label(t_lbl)
                    f_lbl = normalize_label(f_lbl)
                    t_id = block_ids.get(t_lbl)
                    f_id = block_ids.get(f_lbl)
                    if t_id is None or f_id is None:
                        raise RuntimeError(
                            f"Unknown branch targets '{t_lbl}'/'{f_lbl}' in Metal lowering"
                        )
                    if needs_tg_loop_sync and (t_id <= block_id or f_id <= block_id):
                        emit("threadgroup_barrier(mem_flags::mem_threadgroup);")
                    emit(
                        f"if ({to_expr(cond)}) {{ __triton_pred_block = {block_id}; __pc = {t_id}; }} "
                        f"else {{ __triton_pred_block = {block_id}; __pc = {f_id}; }}"
                    )
                    emit("continue;")
                    terminated = True
                    break

                m = _RE_PHI.match(line) if _opc == "phi" else None
                if m:
                    out_ssa, incoming_raw = m.groups()
                    out = msl_id(out_ssa)
                    ssa[out_ssa] = out
                    incoming_pairs = []
                    for incoming in split_top_level(incoming_raw):
                        pair = incoming.strip()
                        pm = _RE_PHI_INCOMING.match(pair)
                        if pm is None:
                            raise RuntimeError(
                                f"Unsupported phi incoming value in Metal lowering: '{pair}'"
                            )
                        incoming_pairs.append(
                            (pm.group(1).strip(), normalize_label(pm.group(2)))
                        )
                    if not incoming_pairs:
                        raise RuntimeError("Malformed phi node with no incoming values")
                    phi_expr = to_expr(incoming_pairs[-1][0])
                    for val, pred in reversed(incoming_pairs[:-1]):
                        pred_id = block_ids.get(pred, -1)
                        phi_expr = f"(__triton_pred_block == {pred_id} ? {to_expr(val)} : {phi_expr})"
                    emit(f"{out} = {phi_expr};")
                    continue

                m = _RE_VOID_CALL.match(line) if _opc == "call" else None
                if m:
                    fn, args_raw = m.groups()
                    args = [to_expr(v) for v in parse_call_args(args_raw)]
                    if (
                        fn.startswith("__metal_predicated_st_global_")
                        and len(args) == 3
                    ):
                        emit(f"if ({args[2]}) {{ *{args[1]} = {args[0]}; }}")
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
                                # Conservative fallback when flag folding is
                                # not possible.
                                barrier_flags = "mem_flags::mem_threadgroup"
                        emit(f"threadgroup_barrier({barrier_flags});")
                    elif fn == "__metal_simdgroup_store" and len(args) == 3:
                        emit(
                            f"simdgroup_store({args[0]}, "
                            f"(device float*){args[1]}, {args[2]});"
                        )
                    elif fn.startswith("llvm.assume"):
                        emit("(void)0;")
                    elif fn.startswith("llvm.lifetime.start") or fn.startswith(
                        "llvm.lifetime.end"
                    ):
                        emit("(void)0;")
                    elif fn.startswith("llvm.memcpy") and len(args) >= 3:
                        emit(
                            f"for (int __i = 0; __i < {args[2]}; __i++) "
                            f"((device char*){args[0]})[__i] = ((device char*){args[1]})[__i];"
                        )
                    elif fn.startswith("llvm.memset") and len(args) >= 3:
                        emit(
                            f"for (int __i = 0; __i < {args[2]}; __i++) "
                            f"((device char*){args[0]})[__i] = (char){args[1]};"
                        )
                    elif fn.startswith("llvm.memmove") and len(args) >= 3:
                        # Correct memmove semantics: copy backward when
                        # dst > src to handle overlapping regions safely.
                        emit(f"if ((uintptr_t){args[0]} > (uintptr_t){args[1]}) {{")
                        emit(
                            f"  for (int __i = {args[2]} - 1; __i >= 0; __i--) "
                            f"((device char*){args[0]})[__i] = ((device char*){args[1]})[__i];"
                        )
                        emit(f"}} else {{")
                        emit(
                            f"  for (int __i = 0; __i < {args[2]}; __i++) "
                            f"((device char*){args[0]})[__i] = ((device char*){args[1]})[__i];"
                        )
                        emit(f"}}")
                    else:
                        if fn.startswith("llvm."):
                            raise RuntimeError(
                                f"Unsupported LLVM intrinsic in Metal lowering: '{fn}'"
                            )
                        emit(f"{fn}({', '.join(args)});")
                    continue

                m = _RE_SELECT.match(line) if _opc == "select" else None
                if m:
                    out_ssa, cond, lhs, rhs = m.groups()
                    out = msl_id(out_ssa)
                    ssa[out_ssa] = out
                    emit(
                        f"{out} = ({to_expr(cond)} ? {to_expr(lhs)} : {to_expr(rhs)});"
                    )
                    continue

                m = _RE_FNEG.match(line) if _opc == "fneg" else None
                if m:
                    out_ssa, val = m.groups()
                    out = msl_id(out_ssa)
                    ssa[out_ssa] = out
                    emit(f"{out} = -({to_expr(val)});")
                    continue

                m = _RE_FREEZE.match(line) if _opc == "freeze" else None
                if m:
                    out_ssa, val = m.groups()
                    out = msl_id(out_ssa)
                    ssa[out_ssa] = out
                    emit(f"{out} = {to_expr(val)};")
                    continue

                m = _RE_EXTRACTELEM.match(line) if _opc == "extractelement" else None
                if m:
                    out_ssa, width_s, vec, idx = m.groups()
                    out = msl_id(out_ssa)
                    ssa[out_ssa] = out
                    width = int(width_s)
                    if width == 1:
                        emit(f"{out} = {to_expr(vec)};")
                    else:
                        emit(f"{out} = {to_expr(vec)}[{to_expr(idx)}];")
                    continue

                m = _RE_INSERTELEM.match(line) if _opc == "insertelement" else None
                if m:
                    out_ssa, width_s, vec, val, idx = m.groups()
                    out = msl_id(out_ssa)
                    ssa[out_ssa] = out
                    width = int(width_s)
                    if width == 1:
                        emit(f"{out} = {to_expr(val)};")
                    else:
                        emit(f"{out} = {to_expr(vec)};")
                        emit(f"{out}[{to_expr(idx)}] = {to_expr(val)};")
                    continue

                m = _RE_SHUFFLEVECTOR.match(line) if _opc == "shufflevector" else None
                if m:
                    (
                        out_ssa,
                        lhs_width_s,
                        elem_ty,
                        lhs_vec,
                        rhs_width_s,
                        rhs_vec,
                        out_width_s,
                        mask_spec,
                    ) = m.groups()
                    out = msl_id(out_ssa)
                    ssa[out_ssa] = out
                    lhs_width = int(lhs_width_s)
                    rhs_width = int(rhs_width_s)
                    out_width = int(out_width_s)
                    lhs_expr = to_expr(extract_value_token(lhs_vec))
                    rhs_expr = to_expr(extract_value_token(rhs_vec))
                    scalar_ty = llvm_scalar_to_msl(elem_ty.strip())

                    mask = mask_spec.strip()
                    if mask == "zeroinitializer":
                        mask_elems = ["0"] * out_width
                    elif mask in ("undef", "poison"):
                        mask_elems = [mask] * out_width
                    else:
                        if mask.startswith("<") and mask.endswith(">"):
                            mask = mask[1:-1].strip()
                        mask_elems = split_top_level(mask)

                    def _lane(vec_expr: str, width: int, lane: int) -> str:
                        if width <= 1:
                            return vec_expr
                        return f"{vec_expr}[{lane}]"

                    shuffled: list[str] = []
                    for mask_elem in mask_elems:
                        _, lane_tok = split_typed_value(mask_elem)
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
                        lane = int(lane_tok)
                        if lane < lhs_width:
                            shuffled.append(_lane(lhs_expr, lhs_width, lane))
                        elif lane < lhs_width + rhs_width:
                            shuffled.append(_lane(rhs_expr, rhs_width, lane - lhs_width))
                        else:
                            shuffled.append("0")

                    if len(shuffled) < out_width:
                        shuffled.extend(["0"] * (out_width - len(shuffled)))

                    if out_width <= 1:
                        emit(f"{out} = {shuffled[0] if shuffled else '0'};")
                    else:
                        emit(f"{out} = {scalar_ty}{out_width}({', '.join(shuffled[:out_width])});")
                    continue

                m = _RE_EXTRACTVALUE.match(line) if _opc == "extractvalue" else None
                if m:
                    out_ssa, _, src_val, idx_str = m.groups()
                    out = msl_id(out_ssa)
                    ssa[out_ssa] = out
                    emit(f"{out} = {to_expr(src_val)}.field{idx_str};")
                    continue

                m = _RE_INSERTVALUE.match(line) if _opc == "insertvalue" else None
                if m:
                    out_ssa, _, agg_val, _, elem_val, idx_str = m.groups()
                    out = msl_id(out_ssa)
                    ssa[out_ssa] = out
                    emit(f"{out} = {to_expr(agg_val)};")
                    emit(f"{out}.field{idx_str} = {to_expr(elem_val)};")
                    continue

                m = _RE_ATOMICRMW.match(line) if _opc == "atomicrmw" else None
                if m:
                    out_ssa, op, addr_space, ptr, val_type, val, ordering = m.groups()
                    out = msl_id(out_ssa)
                    ssa[out_ssa] = out
                    msl_ty = llvm_scalar_to_msl(val_type.strip())
                    atomic_func = _ATOMIC_OP_MAP.get(op, "atomic_fetch_add_explicit")
                    msl_order = _MEMORY_ORDER_MAP.get(ordering, "memory_order_relaxed")
                    emit(
                        f"{out} = {atomic_func}("
                        f"reinterpret_cast<{msl_addr_space(addr_space)} atomic_{msl_ty}*>({to_expr(ptr)}), "
                        f"{to_expr(val)}, {msl_order});"
                    )
                    continue

                m = _RE_CMPXCHG.match(line) if _opc == "cmpxchg" else None
                if m:
                    (
                        out_ssa,
                        addr_space,
                        ptr,
                        val_type,
                        expected,
                        desired,
                        success_order,
                        fail_order,
                    ) = m.groups()
                    out = msl_id(out_ssa)
                    ssa[out_ssa] = out
                    msl_ty = llvm_scalar_to_msl(val_type.strip())
                    msl_success = _MEMORY_ORDER_MAP.get(
                        success_order, "memory_order_relaxed"
                    )
                    msl_fail = _MEMORY_ORDER_MAP.get(fail_order, "memory_order_relaxed")
                    emit(f"{out}.field0 = {to_expr(expected)};")
                    emit(
                        f"{out}.field1 = atomic_compare_exchange_weak_explicit("
                        f"reinterpret_cast<{msl_addr_space(addr_space)} atomic_{msl_ty}*>({to_expr(ptr)}), "
                        f"&{out}.field0, {to_expr(desired)}, {msl_success}, {msl_fail});"
                    )
                    continue

                m = _RE_ALLOCA.match(line) if _opc == "alloca" else None
                if m:
                    out_ssa, elem_type = m.groups()
                    out = msl_id(out_ssa)
                    ssa[out_ssa] = out
                    emit(f"{out} = &{out}_storage;")
                    continue

                m = _RE_SWITCH.match(line) if _opc == "switch" else None
                if m:
                    val_type, val, default_label, cases_str = m.groups()
                    default_label = normalize_label(default_label)
                    default_id = block_ids.get(default_label)
                    emit(f"__triton_pred_block = {block_id};")
                    emit(f"switch ({to_expr(val)}) {{")
                    for cm in _RE_SWITCH_CASE.finditer(cases_str):
                        case_val = cm.group(2)
                        case_label = normalize_label(cm.group(3))
                        case_id = block_ids.get(case_label)
                        if case_id is not None:
                            emit(f"  case {case_val}: __pc = {case_id}; break;")
                    if default_id is not None:
                        emit(f"  default: __pc = {default_id}; break;")
                    emit("}")
                    emit("continue;")
                    terminated = True
                    break

                m = _RE_FENCE.match(line) if _opc == "fence" else None
                if m:
                    syncscope, ordering = m.groups()
                    if syncscope in ("workgroup", "threadgroup"):
                        emit("threadgroup_barrier(mem_flags::mem_threadgroup);")
                    elif syncscope in ("subgroup", "wavefront"):
                        emit("simdgroup_barrier(mem_flags::mem_threadgroup);")
                    else:
                        emit("threadgroup_barrier(mem_flags::mem_device);")
                    continue

                if line.startswith("unreachable"):
                    emit("return;")
                    terminated = True
                    break

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
                    emit(f"// UNSUPPORTED: {line}")
                    continue

            if not terminated:
                emit("return;")
            body_lines.append("    }")

        body_lines.append("    default: return;")
        body_lines.append("    }")
        body_lines.append("  }")

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

        msl_lines = [
            "#include <metal_stdlib>",
            "using namespace metal;",
            "",
        ]
        msl_lines.extend(struct_defs)
        if struct_defs:
            msl_lines.append("")
        msl_lines.extend(
            [
                f"kernel void {msl_kernel_name}(",
                ",\n".join(param_lines),
                ") {",
            ]
        )
        msl_lines.extend(body_lines)
        msl_lines.append("}")
        msl_lines.append("")
        return "\n".join(msl_lines)

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
