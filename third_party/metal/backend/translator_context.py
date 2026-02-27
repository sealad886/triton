"""TranslatorContext: typed state object for LLVM IR → MSL translation.

Extracted from the closure scope of ``make_metal_ir()`` in ``compiler.py``
to enable typed instruction dispatch and improve testability.
"""

from __future__ import annotations

import math
import re
import struct

# ── Constants used by migrated utility functions ────────────────────
# Authoritative definitions; compiler.py imports these to avoid duplication.

_UNSIGNED_MSL_MAP = {
    "bool": "bool",
    "char": "unsigned char",
    "short": "unsigned short",
    "int": "unsigned int",
    "long": "unsigned long",
}

_RE_ALIGN_STRIP = re.compile(r",\s*align\s+\d+$")


# ── Constants needed by TranslatorContext methods ───────────────────
# These mirror definitions in compiler.py.  They live here to avoid a
# circular import (compiler.py already imports from this module).  When
# compiler.py is wired to delegate to TranslatorContext (Task 1.6), these
# will become the single source of truth for both files.

_SSA_NAME_RE = r"%[-A-Za-z0-9._]+"

_MSL_RESERVED_IDENTIFIERS = frozenset(
    {
        "kernel", "vertex", "fragment", "compute", "thread", "threadgroup",
        "device", "constant",
        "bool", "char", "short", "int", "long", "half", "float", "double", "void",
        "uint", "uchar", "ushort", "ulong",
        "if", "else", "switch", "case", "default", "return", "continue", "break",
        "while", "for", "do", "goto",
        "struct", "class", "union", "enum", "auto", "const", "volatile", "static",
        "extern", "inline", "sizeof", "namespace", "using", "template", "typename",
        "typedef", "new", "delete",
        "fma", "fabs", "sqrt", "floor", "ceil", "trunc", "rint", "exp", "exp2",
        "log", "log2", "sin", "cos", "tanh", "pow", "copysign", "max", "min",
        "isnan", "popcount", "select", "clamp", "abs", "sign", "saturate", "step",
        "mix",
    }
)

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

# ── Regex patterns for type conversion / constant parsing ───────────

_RE_MSL_ID_CLEAN = re.compile(r"[^A-Za-z0-9_]")
_RE_PTR_TYPE = re.compile(r"^ptr(?:\s+addrspace\((\d+)\))?$")
_RE_VEC_TYPE = re.compile(r"^<\s*(\d+)\s+x\s+(.+)\s*>$")
_RE_CONST_INT = re.compile(r"^-?[0-9]+$")
_RE_CONST_HEX_FLOAT = re.compile(r"^0x([0-9A-Fa-f]{16})$")
_RE_CONST_HEX_HALF = re.compile(r"^0xH([0-9A-Fa-f]{4})$")
_RE_CONST_HEX_BFLOAT = re.compile(r"^0xR([0-9A-Fa-f]{4})$")
_RE_CONST_FLOAT = re.compile(r"^-?[0-9]*\.?[0-9]+([eE][+-]?[0-9]+)?$")

# ── Regex patterns for call return type extraction ──────────────────

_RE_CALL_RET_VEC = re.compile(r"<\s*\d+\s+x\s+[^>]+\s*>")
_RE_CALL_RET_PTR = re.compile(r"ptr(?:\s+addrspace\(\d+\))?")
_RE_CALL_RET_SCALAR = re.compile(
    r"\bi\d+\b|\bi1\b|\bhalf\b|\bbfloat\b|\bfloat\b|\bdouble\b"
)

# ── Regex patterns for GEP / ptr parsing ────────────────────────────

_RE_PTR_SPEC = re.compile(r"^ptr(?:\s+addrspace\((\d+)\))?\s+(.+)$")
_RE_GEP_INSTRUCTION_FALLBACK = re.compile(
    r"^(" + _SSA_NAME_RE + r")\s*=\s*getelementptr(?:\s+\w+)*\s+(.+)$"
)
_RE_GEP_FLAG_STRIP = re.compile(r"^(?:inbounds|nuw|nsw|inrange)\s+")

# ── Regex / tables for intrinsic lowering ───────────────────────────

_RE_LLVM_VECTOR_REDUCE = re.compile(
    r"^llvm\.vector\.reduce\.([a-z]+)\.v(\d+)([A-Za-z0-9]+)$"
)

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


# ── Stateless utility functions ─────────────────────────────────────
# Migrated from nested defs inside make_metal_ir() in compiler.py.
# These are pure functions with no closure-variable dependencies.


def split_top_level(text: str, sep: str = ",") -> list[str]:
    """Split *text* on *sep* respecting nested brackets ``([{<…>}])``."""
    parts: list[str] = []
    cur: list[str] = []
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


def normalize_label(label: str) -> str:
    """Strip surrounding quotes from a basic-block label."""
    label = label.strip()
    if label.startswith('"') and label.endswith('"'):
        return label[1:-1]
    return label


def unsigned_msl(msl_ty: str) -> str:
    """Map a signed MSL scalar type to its unsigned counterpart."""
    return _UNSIGNED_MSL_MAP.get(msl_ty, f"unsigned {msl_ty}")


def vector_alias_msl(msl_scalar_ty: str, width: int) -> str:
    """Return the MSL vector type alias for *msl_scalar_ty* × *width*."""
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


def strip_operand_attrs(spec: str) -> str:
    """Strip LLVM alignment attributes from an operand spec string."""
    spec = spec.strip()
    spec = _RE_ALIGN_STRIP.sub("", spec)
    return spec.strip()


def extract_value_token(spec: str) -> str:
    """Extract the value token from a typed LLVM operand specification."""
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


class TranslatorContext:
    """Holds all mutable state for a single LLVM IR → MSL translation pass."""

    __slots__ = (
        "ssa",
        "ssa_decl_types",
        "_msl_id_used",
        "blocks",
        "block_order",
        "block_ids",
        "params",
        "ptr_elem",
        "aggregate_type_structs",
        "struct_defs",
        "body_lines",
        "unsupported",
        "use_native_simdgroup",
        "fallback_simdgroup_elem_types",
        "uses_shared_smem",
        "shared_bytes",
        "kernel_name",
        "param_ids",
    )

    def __init__(
        self,
        *,
        uses_shared_smem: bool = False,
        shared_bytes: int = 1,
        use_native_simdgroup: bool = True,
        kernel_name: str = "kernel_",
    ) -> None:
        self.ssa: dict[str, str] = {}
        self.ssa_decl_types: dict[str, str] = {}
        self._msl_id_used: dict[str, str] = {}
        self.blocks: dict[str, list[str]] = {}
        self.block_order: list[str] = []
        self.block_ids: dict[str, int] = {}
        self.params: list[dict] = []
        self.ptr_elem: dict[str, str] = {}
        self.aggregate_type_structs: dict[str, tuple[str, list[str]]] = {}
        self.struct_defs: list[str] = []
        self.body_lines: list[str] = []
        self.unsupported: list = []
        self.use_native_simdgroup: bool = use_native_simdgroup
        self.fallback_simdgroup_elem_types: set[str] = set()
        self.uses_shared_smem: bool = uses_shared_smem
        self.shared_bytes: int = shared_bytes
        self.kernel_name: str = kernel_name
        self.param_ids: set[str] = set()

    # ── Batch 1: Type Conversion (Task 1.3) ─────────────────────────

    def msl_id(self, llvm_name: str) -> str:
        """Map an LLVM SSA name to a unique, legal MSL identifier."""
        raw = llvm_name.lstrip("%")
        if _RE_MSL_ID_CLEAN.search(raw):
            raw = _RE_MSL_ID_CLEAN.sub("_", raw)
        if not raw:
            raw = "tmp"
        if raw[0].isdigit():
            raw = f"v{raw}"
        if raw in _MSL_RESERVED_IDENTIFIERS:
            raw = f"v_{raw}"
        owner = self._msl_id_used.get(raw)
        if owner is not None and owner != llvm_name:
            suffix = 2
            candidate = f"{raw}_{suffix}"
            while candidate in self._msl_id_used:
                suffix += 1
                candidate = f"{raw}_{suffix}"
            raw = candidate
        self._msl_id_used[raw] = llvm_name
        return raw

    def llvm_scalar_to_msl(self, llvm_ty: str) -> str:
        """Return the MSL scalar type for an LLVM scalar type string."""
        return _LLVM_SCALAR_TO_MSL.get(llvm_ty.strip(), "int")

    def llvm_type_to_msl(self, llvm_ty: str) -> str:
        """Convert an LLVM type string (scalar, ptr, vector) to MSL."""
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
            scalar = self.llvm_scalar_to_msl(vec_match.group(2))
            if width == 1:
                return scalar
            return f"vec<{scalar}, {width}>"
        return self.llvm_scalar_to_msl(llvm_ty)

    def msl_addr_space(self, addr_space: str | None) -> str:
        """Return MSL address-space qualifier for an LLVM addrspace number."""
        return "threadgroup" if addr_space == "3" else "device"

    def ptr_type_to_msl(
        self, pointee_llvm_ty: str, addr_space: str | None = None
    ) -> str:
        """Return an MSL pointer type string for a pointee LLVM type."""
        space = self.msl_addr_space(addr_space)
        return f"{space} {self.llvm_scalar_to_msl(pointee_llvm_ty)}*"

    def constant_to_msl(self, token: str) -> str:
        """Convert an LLVM IR constant token to an MSL literal expression."""
        token = token.strip()
        if token in ("undef", "poison", "zeroinitializer"):
            return "0"
        if token in ("true", "false", "nullptr", "null"):
            return "nullptr" if token == "null" else token
        if token.startswith("<") and token.endswith(">"):
            inner = token[1:-1].strip()
            elems = split_top_level(inner)
            if elems:
                elem_vals: list[str] = []
                elem_msl_ty: str | None = None
                vector_ok = True
                for elem in elems:
                    llvm_ty, val = self.split_typed_value(elem)
                    llvm_ty = llvm_ty.strip()
                    if not llvm_ty:
                        vector_ok = False
                        break
                    cur_msl_ty = self.llvm_scalar_to_msl(llvm_ty)
                    if elem_msl_ty is None:
                        elem_msl_ty = cur_msl_ty
                    elif elem_msl_ty != cur_msl_ty:
                        vector_ok = False
                        break
                    elem_vals.append(self.constant_to_msl(val))
                if vector_ok and elem_msl_ty is not None:
                    return (
                        f"{elem_msl_ty}{len(elem_vals)}"
                        f"({', '.join(elem_vals)})"
                    )
        if _RE_CONST_INT.match(token):
            return token
        hex_m = _RE_CONST_HEX_FLOAT.match(token)
        if hex_m:
            raw = int(hex_m.group(1), 16)
            dval = struct.unpack("d", struct.pack("Q", raw))[0]
            if math.isinf(dval):
                return "-INFINITY" if dval < 0 else "INFINITY"
            if math.isnan(dval):
                return "NAN"
            return f"{dval!r}f"
        hex_h = _RE_CONST_HEX_HALF.match(token)
        if hex_h:
            raw16 = int(hex_h.group(1), 16)
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
        hex_bf = _RE_CONST_HEX_BFLOAT.match(token)
        if hex_bf:
            raw_bf = int(hex_bf.group(1), 16)
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

    def to_expr(self, token: str) -> str:
        """Resolve an LLVM IR value token to an MSL expression string."""
        token = token.strip()
        cached = self.ssa.get(token)
        if cached is not None:
            return cached
        if token.startswith("%"):
            out = self.msl_id(token)
            self.ssa[token] = out
            return out
        if token == "@global_smem":
            return "((threadgroup char*)__triton_shared)"
        gep_cexpr = self.parse_gep_constexpr(token)
        if gep_cexpr is not None:
            _, _, base, idx_token = gep_cexpr
            return f"({self.to_expr(base)} + {self.to_expr(idx_token)})"
        return self.constant_to_msl(token)

    # ── Batch 2: Parsing & Emission (Task 1.4) ─────────────────────

    def record_ssa_decl(
        self,
        out_ssa: str,
        llvm_ty: str | None = None,
        msl_ty: str | None = None,
    ) -> None:
        """Register an SSA declaration with its resolved MSL type."""
        out = self.msl_id(out_ssa)
        self.ssa[out_ssa] = out
        if out in self.param_ids or out in self.ssa_decl_types:
            return
        resolved = (
            msl_ty if msl_ty is not None else self.llvm_type_to_msl(llvm_ty or "i32")
        )
        self.ssa_decl_types[out] = resolved

    def emit(self, stmt: str) -> None:
        """Append an indented MSL statement to the body output."""
        self.body_lines.append(f"      {stmt}")

    def simdgroup_elem_from_decl(self, msl_decl_ty: str | None) -> str | None:
        """Extract the element type from a simdgroup_matrix<…> declaration."""
        if not msl_decl_ty:
            return None
        msl_decl_ty = msl_decl_ty.strip()
        if msl_decl_ty.startswith("simdgroup_matrix<") and msl_decl_ty.endswith(
            ">"
        ):
            inner = msl_decl_ty[len("simdgroup_matrix<") : -1]
            return inner.split(",", 1)[0].strip()
        if msl_decl_ty.startswith("__metal_sgmat_"):
            return msl_decl_ty[len("__metal_sgmat_") :].replace("_", " ")
        return None

    def simdgroup_elem_for_msl_value(self, msl_value: str) -> str:
        """Look up the simdgroup element type for a declared MSL value."""
        msl_ty = self.ssa_decl_types.get(msl_value)
        elem_ty = self.simdgroup_elem_from_decl(msl_ty)
        return elem_ty if elem_ty is not None else "float"

    def extract_call_ret_type(self, ret_spec: str) -> str:
        """Extract the LLVM return type from a call instruction's ret spec."""
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

    def parse_call_args(self, arg_list: str) -> list[str]:
        """Parse a call instruction's argument list into value tokens."""
        values: list[str] = []
        for arg in split_top_level(arg_list):
            arg = arg.strip()
            if not arg:
                continue
            values.append(extract_value_token(arg))
        return values

    def split_typed_value(self, spec: str) -> tuple[str, str]:
        """Split a typed LLVM operand into ``(llvm_type, value_token)``."""
        spec = spec.strip()
        value = extract_value_token(spec)
        idx = spec.rfind(value)
        llvm_ty = spec[:idx].strip() if idx >= 0 else spec
        return llvm_ty, value

    def parse_ptr_spec(self, spec: str) -> tuple[str | None, str] | None:
        """Parse ``ptr [addrspace(N)] <value>`` into ``(addr_space, value)``."""
        m = _RE_PTR_SPEC.match(spec.strip())
        if not m:
            return None
        return m.group(1), m.group(2).strip()

    def parse_gep_components(
        self, spec: str
    ) -> tuple[str, str | None, str, str] | None:
        """Parse GEP component spec into ``(elem_ty, addr_space, base, idx)``."""
        parts = split_top_level(spec)
        if len(parts) < 3:
            return None
        elem_ty = parts[0].strip()
        ptr_info = self.parse_ptr_spec(parts[1])
        if ptr_info is None:
            return None
        addr_space, base = ptr_info
        _, idx_token = self.split_typed_value(parts[2])
        return elem_ty, addr_space, base, idx_token

    def parse_gep_instruction(
        self, line: str
    ) -> tuple[str, str, str | None, str, str] | None:
        """Parse a full GEP instruction line via fallback regex."""
        m = _RE_GEP_INSTRUCTION_FALLBACK.match(line)
        if not m:
            return None
        out_ssa = m.group(1)
        comps = self.parse_gep_components(m.group(2))
        if comps is None:
            return None
        elem_ty, addr_space, base, idx_token = comps
        return out_ssa, elem_ty, addr_space, base, idx_token

    def parse_gep_constexpr(
        self, token: str
    ) -> tuple[str, str | None, str, str] | None:
        """Parse an inline ``getelementptr`` constant expression."""
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
        comps = self.parse_gep_components(rest)
        if comps is None:
            return None
        elem_ty, addr_space, base, idx_token = comps
        return elem_ty, addr_space, base, idx_token

    def fcmp_expr(self, pred: str, lhs: str, rhs: str) -> str:
        """Generate an MSL expression for an ``fcmp`` predicate."""
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

    def get_aggregate_struct_name(
        self, agg_type_str: str
    ) -> tuple[str, list[str]]:
        """Get or create a struct definition for an LLVM aggregate type."""
        agg_type_str = agg_type_str.strip()
        if agg_type_str in self.aggregate_type_structs:
            return self.aggregate_type_structs[agg_type_str]
        idx = len(self.aggregate_type_structs)
        name = f"__triton_aggr_{idx}"
        inner = agg_type_str.strip("{ }")
        field_types = [self.llvm_type_to_msl(t.strip()) for t in inner.split(",")]
        self.aggregate_type_structs[agg_type_str] = (name, field_types)
        fields = "".join(
            f"  {ft} field{i};\n" for i, ft in enumerate(field_types)
        )
        self.struct_defs.append(f"struct {name} {{\n{fields}}};")
        return name, field_types

    # ── Batch 3: Intrinsic Lowering (Task 1.5) ─────────────────────

    def lower_intrinsic(self, fn: str, args: list[str]) -> str | None:
        """Lower an LLVM intrinsic / libdevice call to an MSL expression."""
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
                u_ty = unsigned_msl(self.llvm_scalar_to_msl(elem_ty))
                terms = [f"(({u_ty}){term})" for term in terms]
                if init is not None:
                    init = f"(({u_ty})({init}))"

            if reduce_op in (
                "or", "and", "xor", "add", "mul", "fadd", "fmul",
            ):
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
                    expr = fold_infix(
                        [f"({init})", f"({expr})"], op_map[reduce_op]
                    )
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

        # CUDA/OCML-style libdevice symbols → MSL math builtins
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
