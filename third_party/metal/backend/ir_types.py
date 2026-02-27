"""
Typed LLVM IR instruction AST and parser for the Metal backend.

Provides a structured, type-safe representation of LLVM IR instructions
as frozen dataclasses with a deterministic parser that maps raw IR lines
to typed instruction nodes.  This is Phase 1 of replacing the regex-driven
translator with typed IR dispatch.

Usage:
    from triton.third_party.metal.backend.ir_types import parse_instruction, parse_block

    inst = parse_instruction("%5 = add nsw i32 %3, %4")
    assert isinstance(inst, BinOp)
    assert inst.op == "add"
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Union

from triton.backends.metal.compiler import (
    _BINOP_OPCODES,
    _CAST_OPCODES,
    _LLVM_FLAGS,
    _RE_ALLOCA,
    _RE_ATOMICRMW,
    _RE_BINOP,
    _RE_BR,
    _RE_BR_COND,
    _RE_CALL_OUT,
    _RE_CAST,
    _RE_CMPXCHG,
    _RE_EXTRACTELEM_DECL,
    _RE_EXTRACTVALUE,
    _RE_FCMP,
    _RE_FENCE,
    _RE_FNEG_DECL,
    _RE_FREEZE_DECL,
    _RE_GEP,
    _RE_ICMP,
    _RE_INSERTELEM_DECL,
    _RE_INSERTVALUE,
    _RE_LINE_CLEAN,
    _RE_LOAD,
    _RE_PHI,
    _RE_SELECT,
    _RE_SHUFFLEVECTOR,
    _RE_STORE,
    _RE_SWITCH,
    _RE_VOID_CALL,
    _SSA_NAME_RE,
)

# ── Base class ──────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class LLVMInstruction:
    """Base class for all typed LLVM IR instruction nodes."""

    raw_line: str
    opcode: str


# ── Typed instruction dataclasses ───────────────────────────────────


@dataclass(frozen=True, slots=True)
class BinOp(LLVMInstruction):
    """Binary arithmetic/logic operation (add, sub, mul, fadd, and, or, xor, …)."""

    out_ssa: str
    op: str
    llvm_ty: str
    lhs: str
    rhs: str
    flags: str


@dataclass(frozen=True, slots=True)
class Load(LLVMInstruction):
    """Memory load instruction."""

    out_ssa: str
    llvm_ty: str
    addr_space: str | None
    ptr: str
    alignment: str | None


@dataclass(frozen=True, slots=True)
class Store(LLVMInstruction):
    """Memory store instruction."""

    val_ty: str
    val: str
    addr_space: str | None
    ptr: str
    alignment: str | None


@dataclass(frozen=True, slots=True)
class Cast(LLVMInstruction):
    """Type cast instruction (sext, zext, trunc, fptrunc, bitcast, …)."""

    out_ssa: str
    cast_op: str
    from_ty: str
    val: str
    to_ty: str


@dataclass(frozen=True, slots=True)
class ICmp(LLVMInstruction):
    """Integer comparison."""

    out_ssa: str
    pred: str
    llvm_ty: str
    lhs: str
    rhs: str


@dataclass(frozen=True, slots=True)
class FCmp(LLVMInstruction):
    """Floating-point comparison."""

    out_ssa: str
    pred: str
    llvm_ty: str
    lhs: str
    rhs: str


@dataclass(frozen=True, slots=True)
class Call(LLVMInstruction):
    """Function call (void and non-void)."""

    out_ssa: str | None
    ret_type: str
    fn_name: str
    args_raw: str
    is_void: bool


@dataclass(frozen=True, slots=True)
class GEP(LLVMInstruction):
    """getelementptr instruction."""

    out_ssa: str
    inbounds: bool
    base_ty: str
    ptr_operand: str
    indices_raw: str


@dataclass(frozen=True, slots=True)
class Phi(LLVMInstruction):
    """phi node."""

    out_ssa: str
    llvm_ty: str
    incoming_raw: str


@dataclass(frozen=True, slots=True)
class Select(LLVMInstruction):
    """select instruction."""

    out_ssa: str
    cond: str
    true_val: str
    false_val: str
    result_ty: str


@dataclass(frozen=True, slots=True)
class FNeg(LLVMInstruction):
    """Floating-point negation."""

    out_ssa: str
    llvm_ty: str
    operand: str


@dataclass(frozen=True, slots=True)
class Freeze(LLVMInstruction):
    """freeze instruction."""

    out_ssa: str
    llvm_ty: str
    operand: str


@dataclass(frozen=True, slots=True)
class VectorOp(LLVMInstruction):
    """Vector operations: extractelement, insertelement, shufflevector."""

    out_ssa: str
    vector_op: str
    operands_raw: str


@dataclass(frozen=True, slots=True)
class AggregateOp(LLVMInstruction):
    """Aggregate operations: extractvalue, insertvalue."""

    out_ssa: str
    agg_op: str
    operands_raw: str


@dataclass(frozen=True, slots=True)
class AtomicOp(LLVMInstruction):
    """Atomic operations: atomicrmw, cmpxchg."""

    out_ssa: str
    atomic_op: str
    ordering: str
    operands_raw: str


@dataclass(frozen=True, slots=True)
class Alloca(LLVMInstruction):
    """Stack allocation."""

    out_ssa: str
    alloc_ty: str
    num_elements: str | None
    alignment: str | None


@dataclass(frozen=True, slots=True)
class Terminator(LLVMInstruction):
    """Block terminator: br, br_cond, switch, ret, unreachable, fence."""

    term_kind: str
    operands_raw: str


@dataclass(frozen=True, slots=True)
class UnknownInstruction(LLVMInstruction):
    """Fallback for lines that don't match any known instruction pattern."""


# ── Union type alias ────────────────────────────────────────────────

Instruction = (
    BinOp
    | Load
    | Store
    | Cast
    | ICmp
    | FCmp
    | Call
    | GEP
    | Phi
    | Select
    | FNeg
    | Freeze
    | VectorOp
    | AggregateOp
    | AtomicOp
    | Alloca
    | Terminator
    | UnknownInstruction
)


# ── Internal helpers ────────────────────────────────────────────────

# Alignment regex used for load/store/alloca field extraction
_RE_ALIGN = re.compile(r",\s*align\s+(\d+)")

# Addrspace extractor for load/store raw operand strings
_RE_ADDRSPACE = re.compile(r"addrspace\((\d+)\)")

# Store pattern with explicit val type capture
_RE_STORE_FULL = re.compile(
    r"^store\s+(.+?)\s+([^,]+),\s+ptr(?:\s+addrspace\((\d+)\))?\s+(.+)$"
)

# Alloca with optional num elements and alignment
# Format: %r = alloca <type>[, <ty> <num>][, align <n>]
_RE_ALLOCA_FULL = re.compile(
    r"^(" + _SSA_NAME_RE + r")\s*=\s*alloca\s+(\S+)"
    r"(?:,\s*(?!align)(\S+)\s+(\S+))?"
    r"(?:,\s*align\s+(\d+))?$"
)

# GEP with inbounds detection
_RE_GEP_FULL = re.compile(
    r"^(" + _SSA_NAME_RE + r")\s*=\s*getelementptr\s+"
    r"(inbounds\s+)?"
    r"(.+?),"
    r"\s+ptr(?:\s+addrspace\(\d+\))?\s+(.+)$"
)

# Phi with type capture
_RE_PHI_FULL = re.compile(r"^(" + _SSA_NAME_RE + r")\s*=\s*phi\s+(.+?)\s+(\[.+)$")

# Select with result type
_RE_SELECT_FULL = re.compile(
    r"^(" + _SSA_NAME_RE + r")\s*=\s*select\s+i1\s+([^,]+),"
    r"\s+(\S+)\s+([^,]+),\s+\S+\s+(.+)$"
)

# FNeg with type capture
_RE_FNEG_FULL = re.compile(
    r"^(" + _SSA_NAME_RE + r")\s*=\s*fneg" + _LLVM_FLAGS + r"\s+(\S+)\s+(.+)$"
)

# BinOp with type+operand split
_RE_BINOP_DETAIL = re.compile(
    r"^("
    + _SSA_NAME_RE
    + r")\s*=\s*(add|sub|mul|udiv|sdiv|urem|srem|shl|lshr|ashr|and|or|xor|fadd|fsub|fmul|fdiv|frem)"
    + _LLVM_FLAGS
    + r"\s+(\S+)\s+([^,]+),\s*(.+)$"
)

# FCmp with type capture (the compiler _RE_FCMP doesn't capture the type)
_RE_FCMP_FULL = re.compile(
    r"^(" + _SSA_NAME_RE + r")\s*=\s*fcmp\s+(\w+)\s+(\S+)\s+([^,]+),\s*(.+)$"
)


def _extract_opcode(line: str) -> str:
    """Extract the LLVM IR opcode from an instruction line.

    Mirrors ``_extract_ir_opcode`` in compiler.py.
    """
    eq_pos = line.find(" = ")
    rest = line[eq_pos + 3 :] if eq_pos >= 0 else line
    for prefix in ("tail ", "musttail ", "notail "):
        if rest.startswith(prefix):
            rest = rest[len(prefix) :]
            break
    sp = rest.find(" ")
    return rest[:sp] if sp >= 0 else rest


def _clean_line(line: str) -> str:
    """Strip metadata, comments, and attribute-group references."""
    return _RE_LINE_CLEAN.sub("", line).strip()


def _extract_flags(line: str, opcode: str) -> str:
    """Extract LLVM fast-math / arithmetic flags between opcode and type."""
    eq_pos = line.find(" = ")
    if eq_pos < 0:
        return ""
    after_eq = line[eq_pos + 3 :]
    op_pos = after_eq.find(opcode)
    if op_pos < 0:
        return ""
    after_op = after_eq[op_pos + len(opcode) :]
    known_flags = {
        "nsw",
        "nuw",
        "nsz",
        "nnan",
        "ninf",
        "arcp",
        "contract",
        "reassoc",
        "afn",
        "fast",
        "exact",
        "disjoint",
    }
    flags = []
    rest = after_op.lstrip()
    while True:
        sp = rest.find(" ")
        if sp < 0:
            break
        token = rest[:sp]
        if token in known_flags:
            flags.append(token)
            rest = rest[sp + 1 :]
        else:
            break
    return " ".join(flags)


# ── Parse function ──────────────────────────────────────────────────


def parse_instruction(line: str) -> LLVMInstruction:
    """Parse a single LLVM IR instruction line into a typed AST node.

    Returns ``UnknownInstruction`` for unrecognized lines (never raises).
    The ``raw_line`` field always stores the *original* input line.
    """
    raw = line
    cleaned = _clean_line(line)
    if not cleaned:
        return UnknownInstruction(raw_line=raw, opcode="")

    opcode = _extract_opcode(cleaned)

    # ── BinOp ───────────────────────────────────────────────────────
    if opcode in _BINOP_OPCODES:
        m = _RE_BINOP_DETAIL.match(cleaned)
        if m:
            flags = _extract_flags(cleaned, m.group(2))
            return BinOp(
                raw_line=raw,
                opcode=opcode,
                out_ssa=m.group(1),
                op=m.group(2),
                llvm_ty=m.group(3),
                lhs=m.group(4).strip(),
                rhs=m.group(5).strip(),
                flags=flags,
            )

    # ── Cast ────────────────────────────────────────────────────────
    if opcode in _CAST_OPCODES:
        m = _RE_CAST.match(cleaned)
        if m:
            # m.group(2)=cast_op, m.group(3)=from_ty+val, m.group(4)=to_ty
            from_part = m.group(3).strip()
            # Split "from_ty val" — type is first token(s) up to last SSA/literal
            parts = from_part.rsplit(None, 1)
            from_ty = parts[0] if len(parts) == 2 else from_part
            val = parts[1] if len(parts) == 2 else ""
            return Cast(
                raw_line=raw,
                opcode=opcode,
                out_ssa=m.group(1),
                cast_op=m.group(2),
                from_ty=from_ty,
                val=val,
                to_ty=m.group(4).strip(),
            )

    # ── ICmp ────────────────────────────────────────────────────────
    if opcode == "icmp":
        m = _RE_ICMP.match(cleaned)
        if m:
            return ICmp(
                raw_line=raw,
                opcode="icmp",
                out_ssa=m.group(1),
                pred=m.group(2),
                llvm_ty=m.group(3),
                lhs=m.group(4).strip(),
                rhs=m.group(5).strip(),
            )

    # ── FCmp ────────────────────────────────────────────────────────
    if opcode == "fcmp":
        m = _RE_FCMP_FULL.match(cleaned)
        if m:
            return FCmp(
                raw_line=raw,
                opcode="fcmp",
                out_ssa=m.group(1),
                pred=m.group(2),
                llvm_ty=m.group(3),
                lhs=m.group(4).strip(),
                rhs=m.group(5).strip(),
            )

    # ── Load ────────────────────────────────────────────────────────
    if opcode == "load":
        align_m = _RE_ALIGN.search(cleaned)
        alignment = align_m.group(1) if align_m else None
        stripped = _RE_ALIGN.sub("", cleaned)
        m = _RE_LOAD.match(stripped)
        if m:
            addrspace = m.group(3)
            return Load(
                raw_line=raw,
                opcode="load",
                out_ssa=m.group(1),
                llvm_ty=m.group(2).strip(),
                addr_space=addrspace,
                ptr=m.group(4).strip(),
                alignment=alignment,
            )

    # ── Store ───────────────────────────────────────────────────────
    if opcode == "store":
        align_m = _RE_ALIGN.search(cleaned)
        alignment = align_m.group(1) if align_m else None
        stripped = _RE_ALIGN.sub("", cleaned)
        m = _RE_STORE_FULL.match(stripped)
        if m:
            return Store(
                raw_line=raw,
                opcode="store",
                val_ty=m.group(1).strip(),
                val=m.group(2).strip(),
                addr_space=m.group(3),
                ptr=m.group(4).strip(),
                alignment=alignment,
            )

    # ── Call (non-void, then void) ──────────────────────────────────
    if opcode == "call":
        m = _RE_CALL_OUT.match(cleaned)
        if m:
            return Call(
                raw_line=raw,
                opcode="call",
                out_ssa=m.group(1),
                ret_type=m.group(2).strip(),
                fn_name=m.group(3),
                args_raw=m.group(4),
                is_void=False,
            )
        m = _RE_VOID_CALL.match(cleaned)
        if m:
            return Call(
                raw_line=raw,
                opcode="call",
                out_ssa=None,
                ret_type="void",
                fn_name=m.group(1),
                args_raw=m.group(2),
                is_void=True,
            )

    # ── GEP ─────────────────────────────────────────────────────────
    if opcode == "getelementptr":
        m = _RE_GEP_FULL.match(cleaned)
        if m:
            inbounds = m.group(2) is not None
            # Separate the base pointer from index operands
            remaining = m.group(4).strip()
            # remaining has "ptr_operand, index_type index_val, ..."
            parts = remaining.split(",", 1)
            ptr_operand = parts[0].strip()
            indices_raw = parts[1].strip() if len(parts) > 1 else ""
            return GEP(
                raw_line=raw,
                opcode="getelementptr",
                out_ssa=m.group(1),
                inbounds=inbounds,
                base_ty=m.group(3).strip(),
                ptr_operand=ptr_operand,
                indices_raw=indices_raw,
            )

    # ── Phi ─────────────────────────────────────────────────────────
    if opcode == "phi":
        m = _RE_PHI_FULL.match(cleaned)
        if m:
            return Phi(
                raw_line=raw,
                opcode="phi",
                out_ssa=m.group(1),
                llvm_ty=m.group(2).strip(),
                incoming_raw=m.group(3),
            )

    # ── Select ──────────────────────────────────────────────────────
    if opcode == "select":
        m = _RE_SELECT_FULL.match(cleaned)
        if m:
            return Select(
                raw_line=raw,
                opcode="select",
                out_ssa=m.group(1),
                cond=m.group(2).strip(),
                true_val=m.group(4).strip(),
                false_val=m.group(5).strip(),
                result_ty=m.group(3).strip(),
            )

    # ── FNeg ────────────────────────────────────────────────────────
    if opcode == "fneg":
        m = _RE_FNEG_FULL.match(cleaned)
        if m:
            return FNeg(
                raw_line=raw,
                opcode="fneg",
                out_ssa=m.group(1),
                llvm_ty=m.group(2),
                operand=m.group(3).strip(),
            )

    # ── Freeze ──────────────────────────────────────────────────────
    if opcode == "freeze":
        m = _RE_FREEZE_DECL.match(cleaned)
        if m:
            return Freeze(
                raw_line=raw,
                opcode="freeze",
                out_ssa=m.group(1),
                llvm_ty=m.group(2),
                operand=m.group(3).strip(),
            )

    # ── Vector operations ───────────────────────────────────────────
    if opcode == "extractelement":
        m = _RE_EXTRACTELEM_DECL.match(cleaned)
        if m:
            return VectorOp(
                raw_line=raw,
                opcode="extractelement",
                out_ssa=m.group(1),
                vector_op="extractelement",
                operands_raw=cleaned[
                    cleaned.find("extractelement") + len("extractelement") :
                ].strip(),
            )

    if opcode == "insertelement":
        m = _RE_INSERTELEM_DECL.match(cleaned)
        if m:
            return VectorOp(
                raw_line=raw,
                opcode="insertelement",
                out_ssa=m.group(1),
                vector_op="insertelement",
                operands_raw=cleaned[
                    cleaned.find("insertelement") + len("insertelement") :
                ].strip(),
            )

    if opcode == "shufflevector":
        m = _RE_SHUFFLEVECTOR.match(cleaned)
        if m:
            return VectorOp(
                raw_line=raw,
                opcode="shufflevector",
                out_ssa=m.group(1),
                vector_op="shufflevector",
                operands_raw=cleaned[
                    cleaned.find("shufflevector") + len("shufflevector") :
                ].strip(),
            )

    # ── Aggregate operations ────────────────────────────────────────
    if opcode == "extractvalue":
        m = _RE_EXTRACTVALUE.match(cleaned)
        if m:
            return AggregateOp(
                raw_line=raw,
                opcode="extractvalue",
                out_ssa=m.group(1),
                agg_op="extractvalue",
                operands_raw=cleaned[
                    cleaned.find("extractvalue") + len("extractvalue") :
                ].strip(),
            )

    if opcode == "insertvalue":
        m = _RE_INSERTVALUE.match(cleaned)
        if m:
            return AggregateOp(
                raw_line=raw,
                opcode="insertvalue",
                out_ssa=m.group(1),
                agg_op="insertvalue",
                operands_raw=cleaned[
                    cleaned.find("insertvalue") + len("insertvalue") :
                ].strip(),
            )

    # ── Atomic operations ───────────────────────────────────────────
    if opcode == "atomicrmw":
        m = _RE_ATOMICRMW.match(cleaned)
        if m:
            return AtomicOp(
                raw_line=raw,
                opcode="atomicrmw",
                out_ssa=m.group(1),
                atomic_op=m.group(2),
                ordering=m.group(7),
                operands_raw=cleaned[
                    cleaned.find("atomicrmw") + len("atomicrmw") :
                ].strip(),
            )

    if opcode == "cmpxchg":
        m = _RE_CMPXCHG.match(cleaned)
        if m:
            return AtomicOp(
                raw_line=raw,
                opcode="cmpxchg",
                out_ssa=m.group(1),
                atomic_op="cmpxchg",
                ordering=m.group(7),
                operands_raw=cleaned[
                    cleaned.find("cmpxchg") + len("cmpxchg") :
                ].strip(),
            )

    # ── Alloca ──────────────────────────────────────────────────────
    if opcode == "alloca":
        m = _RE_ALLOCA_FULL.match(cleaned)
        if m:
            return Alloca(
                raw_line=raw,
                opcode="alloca",
                out_ssa=m.group(1),
                alloc_ty=m.group(2),
                num_elements=m.group(4),  # group(3) is element_ty, group(4) is count
                alignment=m.group(5),
            )

    # ── Terminators ─────────────────────────────────────────────────
    if opcode == "br":
        m = _RE_BR_COND.match(cleaned)
        if m:
            return Terminator(
                raw_line=raw,
                opcode="br",
                term_kind="br_cond",
                operands_raw=cleaned[2:].strip(),
            )
        m = _RE_BR.match(cleaned)
        if m:
            return Terminator(
                raw_line=raw,
                opcode="br",
                term_kind="br",
                operands_raw=cleaned[2:].strip(),
            )

    if opcode == "switch":
        m = _RE_SWITCH.match(cleaned)
        if m:
            return Terminator(
                raw_line=raw,
                opcode="switch",
                term_kind="switch",
                operands_raw=cleaned[6:].strip(),
            )

    if opcode == "ret":
        return Terminator(
            raw_line=raw,
            opcode="ret",
            term_kind="ret",
            operands_raw=cleaned[3:].strip(),
        )

    if opcode == "unreachable":
        return Terminator(
            raw_line=raw,
            opcode="unreachable",
            term_kind="unreachable",
            operands_raw="",
        )

    if opcode == "fence":
        m = _RE_FENCE.match(cleaned)
        if m:
            return Terminator(
                raw_line=raw,
                opcode="fence",
                term_kind="fence",
                operands_raw=cleaned[5:].strip(),
            )

    # ── Fallback ────────────────────────────────────────────────────
    return UnknownInstruction(raw_line=raw, opcode=opcode)


# ── Batch helper ────────────────────────────────────────────────────


def parse_block(lines: list[str]) -> list[LLVMInstruction]:
    """Parse a sequence of LLVM IR lines into typed instruction nodes."""
    return [parse_instruction(line) for line in lines]
