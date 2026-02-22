"""
Metal backend compiler for Triton.

Implements the BaseBackend interface for Apple Metal, lowering Triton IR through
LLVM IR to AIR (Apple Intermediate Representation) and then to .metallib binaries
via xcrun.
"""

import functools
import hashlib
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, Tuple

from triton import knobs
from triton._C.libtriton import ir, llvm, passes
from triton.backends.compiler import BaseBackend, GPUTarget, Language


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

    def __post_init__(self):
        extern_libs = {} if self.extern_libs is None else dict(self.extern_libs)
        object.__setattr__(self, "extern_libs", tuple(extern_libs.items()))
        assert self.num_warps > 0 and (
            self.num_warps & (self.num_warps - 1)
        ) == 0, "num_warps must be a power of 2"

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
        # passes.ttgpuir.add_optimize_dot_operands(pm, True)
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
        mod = src
        pm = ir.pass_manager(mod.context)
        pm.enable_debug()
        passes.ttgpuir.add_combine_tensor_select_and_if(pm)
        passes.ttgpuir.add_allocate_warp_groups(pm)
        if hasattr(passes.convert, "add_index_to_llvmir"):
            passes.convert.add_index_to_llvmir(pm)

        import triton._C.libtriton.metal as metal

        passes.ttgpuir.add_allocate_shared_memory(pm)
        passes.ttgpuir.add_allocate_global_scratch_memory(pm)

        metal.passes.ttgpuir.add_to_llvmir(pm)
        passes.ttgpuir.add_canonicalize_llvm_ir(pm)
        passes.common.add_cse(pm)

        passes.convert.add_scf_to_cf(pm)
        passes.convert.add_cf_to_llvmir(pm)
        passes.convert.add_arith_to_llvmir(pm)
        passes.common.add_canonicalizer(pm)
        passes.common.add_cse(pm)
        passes.common.add_symbol_dce(pm)

        if not hasattr(passes, "llvmir") or not hasattr(passes.llvmir, "add_di_scope"):
            pass
        else:
            passes.llvmir.add_di_scope(pm)

        pm.run(mod, "make_llir")

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
        uses_shared_smem = "@global_smem" in src
        shared_bytes = max(int(metadata.get("shared", 0) or 0), 1)

        def split_top_level(text: str, sep: str = ",") -> list[str]:
            parts = []
            cur = []
            depth = 0
            for ch in text:
                if ch in "([":
                    depth += 1
                elif ch in ")]":
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

        msl_reserved_identifiers = {
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

        def msl_id(llvm_name: str) -> str:
            raw = llvm_name.lstrip("%")
            raw = re.sub(r"[^A-Za-z0-9_]", "_", raw)
            if not raw:
                raw = "tmp"
            if raw[0].isdigit():
                raw = f"v{raw}"
            if raw in msl_reserved_identifiers:
                raw = f"v_{raw}"
            return raw

        def llvm_scalar_to_msl(llvm_ty: str) -> str:
            llvm_ty = llvm_ty.strip()
            table = {
                "i1": "bool",
                "i8": "char",
                "i16": "short",
                "i32": "int",
                "i64": "long",
                "half": "half",
                "float": "float",
                "double": "double",
            }
            return table.get(llvm_ty, "int")

        def llvm_type_to_msl(llvm_ty: str) -> str:
            llvm_ty = llvm_ty.strip()
            ptr_match = re.match(r"^ptr(?:\s+addrspace\((\d+)\))?$", llvm_ty)
            if ptr_match:
                addr_space = ptr_match.group(1)
                if addr_space == "3":
                    return "threadgroup uint*"
                return "device uint*"
            vec_match = re.match(r"^<\s*(\d+)\s+x\s+(.+)\s*>$", llvm_ty)
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
            vec_match = re.search(r"<\s*\d+\s+x\s+[^>]+\s*>", ret_spec)
            if vec_match:
                return vec_match.group(0)
            ptr_match = re.search(r"ptr(?:\s+addrspace\(\d+\))?", ret_spec)
            if ptr_match:
                return ptr_match.group(0)
            scalar_matches = re.findall(
                r"\bi\d+\b|\bi1\b|\bhalf\b|\bfloat\b|\bdouble\b", ret_spec
            )
            if scalar_matches:
                return scalar_matches[-1]
            return "i32"

        def constant_to_msl(token: str) -> str:
            token = token.strip()
            if token in ("undef", "poison", "zeroinitializer"):
                return "0"
            if token in ("true", "false", "nullptr", "null"):
                return "nullptr" if token == "null" else token
            if re.match(r"^-?[0-9]+$", token):
                return token
            if re.match(r"^-?[0-9]*\\.?[0-9]+([eE][+-]?[0-9]+)?$", token):
                return token if token.endswith("f") else f"{token}f"
            return token

        def parse_call_args(arg_list: str) -> list[str]:
            values = []
            for arg in split_top_level(arg_list):
                arg = arg.strip()
                if not arg:
                    continue
                if "%" in arg:
                    values.append(arg[arg.rfind("%"):].strip())
                else:
                    values.append(arg.split()[-1].strip())
            return values

        def extract_value_token(spec: str) -> str:
            spec = spec.strip()
            if "%" in spec:
                return spec[spec.rfind("%"):].strip()
            if "@" in spec:
                return spec[spec.rfind("@"):].strip()
            return spec.split()[-1].strip()

        def split_typed_value(spec: str) -> tuple[str, str]:
            spec = spec.strip()
            value = extract_value_token(spec)
            idx = spec.rfind(value)
            llvm_ty = spec[:idx].strip() if idx >= 0 else spec
            return llvm_ty, value

        def strip_operand_attrs(spec: str) -> str:
            spec = spec.strip()
            spec = re.sub(r",\s*align\s+\d+$", "", spec)
            return spec.strip()

        def normalize_label(label: str) -> str:
            label = label.strip()
            if label.startswith('"') and label.endswith('"'):
                return label[1:-1]
            return label

        func_header = re.search(
            r"define\s+void\s+@([A-Za-z_][A-Za-z0-9_]*)\s*\(",
            src,
            flags=re.MULTILINE,
        )
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

        params_str = src[sig_l + 1:sig_r].strip()
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
            name_match = re.search(r"(%[-A-Za-z0-9._]+)\s*$", raw)
            llvm_name = name_match.group(1) if name_match else f"%arg{idx}"
            prefix = raw[:name_match.start()].strip() if name_match else raw.strip()
            type_match = re.match(
                r"(ptr(?:\s+addrspace\(\d+\))?|i\d+|float|half|double|i1)", prefix
            )
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
            if token == "@global_smem":
                return "((threadgroup char*)__triton_shared)"
            if token in ssa:
                return ssa[token]
            if token.startswith("%"):
                out = msl_id(token)
                ssa[token] = out
                return out
            return constant_to_msl(token)

        cmp_map = {
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
        float_bin_map = {
            "fadd": "+",
            "fsub": "-",
            "fmul": "*",
            "fdiv": "/",
        }

        def lower_intrinsic(fn: str, args: list[str]) -> str | None:
            if fn.startswith("llvm.fabs.") and len(args) == 1:
                return f"fabs({args[0]})"
            if fn.startswith("llvm.sqrt.") and len(args) == 1:
                return f"sqrt({args[0]})"
            if fn.startswith("llvm.floor.") and len(args) == 1:
                return f"floor({args[0]})"
            if fn.startswith("llvm.ceil.") and len(args) == 1:
                return f"ceil({args[0]})"
            if fn.startswith("llvm.trunc.") and len(args) == 1:
                return f"trunc({args[0]})"
            if fn.startswith("llvm.round.") and len(args) == 1:
                return f"rint({args[0]})"
            if fn.startswith("llvm.exp2.") and len(args) == 1:
                return f"exp2({args[0]})"
            if fn.startswith("llvm.exp.") and len(args) == 1:
                return f"exp({args[0]})"
            if fn.startswith("llvm.log2.") and len(args) == 1:
                return f"log2({args[0]})"
            if fn.startswith("llvm.log.") and len(args) == 1:
                return f"log({args[0]})"
            if fn.startswith("llvm.sin.") and len(args) == 1:
                return f"sin({args[0]})"
            if fn.startswith("llvm.cos.") and len(args) == 1:
                return f"cos({args[0]})"
            if fn.startswith("llvm.tanh.") and len(args) == 1:
                return f"tanh({args[0]})"
            if fn.startswith("llvm.pow.") and len(args) == 2:
                return f"pow({args[0]}, {args[1]})"
            if fn.startswith("llvm.copysign.") and len(args) == 2:
                return f"copysign({args[0]}, {args[1]})"
            if fn.startswith("llvm.fma.") and len(args) == 3:
                return f"fma({args[0]}, {args[1]}, {args[2]})"
            if (
                fn.startswith("llvm.maximum.")
                or fn.startswith("llvm.maxnum.")
                or fn.startswith("llvm.smax.")
                or fn.startswith("llvm.umax.")
            ) and len(args) == 2:
                return f"max({args[0]}, {args[1]})"
            if (
                fn.startswith("llvm.minimum.")
                or fn.startswith("llvm.minnum.")
                or fn.startswith("llvm.smin.")
                or fn.startswith("llvm.umin.")
            ) and len(args) == 2:
                return f"min({args[0]}, {args[1]})"
            if fn.startswith("llvm.ctpop.") and len(args) == 1:
                return f"popcount({args[0]})"

            # LLVM IR emitted by shared Triton pipelines can still reference
            # CUDA/OCML-style libdevice symbols. Lower these to equivalent MSL
            # math builtins so Metal compilation remains backend-agnostic.
            libdevice_unary = (
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
            if len(args) == 1:
                for pattern, builtin in libdevice_unary:
                    if re.match(pattern, fn):
                        return f"{builtin}({args[0]})"

            libdevice_binary = (
                (r"^__(?:nv|ocml)_pow(?:f|_f32)?$", "pow"),
                (r"^__(?:nv|ocml)_copysign(?:f|_f32)?$", "copysign"),
                (r"^__(?:nv|ocml)_fmax(?:f|_f32)?$", "max"),
                (r"^__(?:nv|ocml)_fmin(?:f|_f32)?$", "min"),
            )
            if len(args) == 2:
                for pattern, builtin in libdevice_binary:
                    if re.match(pattern, fn):
                        return f"{builtin}({args[0]}, {args[1]})"

            if len(args) == 3 and re.match(r"^__(?:nv|ocml)_fma(?:f|_f32)?$", fn):
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

        bin_map = {
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
        axis_helper_map = {
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

        body = src[func_body_l + 1:func_body_r]
        cleaned_lines = []
        for raw_line in body.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            line = re.sub(r",\s*!dbg\s*![0-9]+.*$", "", line)
            if line.startswith(";"):
                continue
            cleaned_lines.append(line)

        blocks = {"entry": []}
        block_order = ["entry"]
        current_block = "entry"
        for line in cleaned_lines:
            if line.endswith(":"):
                label = line[:-1].strip().replace("%", "")
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

        def record_ssa_decl(out_ssa: str, llvm_ty: str | None = None, msl_ty: str | None = None) -> None:
            out = msl_id(out_ssa)
            ssa[out_ssa] = out
            if out in param_ids or out in ssa_decl_types:
                return
            resolved = msl_ty if msl_ty is not None else llvm_type_to_msl(llvm_ty or "i32")
            ssa_decl_types[out] = resolved

        for block in block_order:
            for line in blocks.get(block, []):
                m = re.match(r"^(%[-A-Za-z0-9._]+)\s*=\s*phi\s+(.+?)\s+\[", line)
                if m:
                    out_ssa, llvm_ty = m.groups()
                    record_ssa_decl(out_ssa, llvm_ty=llvm_ty)
                    continue

                m = re.match(
                    r"^(%[-A-Za-z0-9._]+)\s*=\s*(?:tail\s+)?call\s+(.+?)\s+@([A-Za-z0-9_.$-]+)\((.*)\)$",
                    line,
                )
                if m:
                    out_ssa, ret_spec, _, _ = m.groups()
                    record_ssa_decl(out_ssa, llvm_ty=extract_call_ret_type(ret_spec))
                    continue

                m = re.match(
                    r"^(%[-A-Za-z0-9._]+)\s*=\s*(add|sub|mul|udiv|sdiv|urem|srem|shl|lshr|ashr|and|or|xor|fadd|fsub|fmul|fdiv|frem)(?:\s+[A-Za-z]+)*\s+(.+)$",
                    line,
                )
                if m:
                    out_ssa, _, operands_spec = m.groups()
                    parts = split_top_level(operands_spec)
                    if len(parts) != 2:
                        continue
                    llvm_ty, _ = split_typed_value(parts[0])
                    record_ssa_decl(out_ssa, llvm_ty=llvm_ty)
                    continue

                m = re.match(
                    r"^(%[-A-Za-z0-9._]+)\s*=\s*fneg(?:\s+[A-Za-z]+)*\s+(.+?)\s+(.+)$",
                    line,
                )
                if m:
                    out_ssa, llvm_ty, _ = m.groups()
                    record_ssa_decl(out_ssa, llvm_ty=llvm_ty)
                    continue

                m = re.match(r"^(%[-A-Za-z0-9._]+)\s*=\s*icmp\s+(\w+)\s+[^ ]+\s+([^,]+),\s*(.+)$", line)
                if m:
                    out_ssa, _, _, _ = m.groups()
                    record_ssa_decl(out_ssa, msl_ty="bool")
                    continue

                m = re.match(r"^(%[-A-Za-z0-9._]+)\s*=\s*fcmp\s+(\w+)\s+[^ ]+\s+([^,]+),\s*(.+)$", line)
                if m:
                    out_ssa, _, _, _ = m.groups()
                    record_ssa_decl(out_ssa, msl_ty="bool")
                    continue

                m = re.match(
                    r"^(%[-A-Za-z0-9._]+)\s*=\s*(sext|zext|trunc|sitofp|uitofp|fptosi|fptoui|bitcast|addrspacecast|ptrtoint|inttoptr)\s+(.+)\s+to\s+(.+)$",
                    line,
                )
                if m:
                    out_ssa, _, _, dst_ty = m.groups()
                    record_ssa_decl(out_ssa, llvm_ty=dst_ty.strip())
                    continue

                m = re.match(r"^(%[-A-Za-z0-9._]+)\s*=\s*freeze\s+(.+?)\s+(.+)$", line)
                if m:
                    out_ssa, llvm_ty, _ = m.groups()
                    record_ssa_decl(out_ssa, llvm_ty=llvm_ty)
                    continue

                m = re.match(
                    r"^(%[-A-Za-z0-9._]+)\s*=\s*select\s+i1\s+[^,]+,\s+(.+?)\s+[^,]+,\s+.+$",
                    line,
                )
                if m:
                    out_ssa, llvm_ty = m.groups()
                    record_ssa_decl(out_ssa, llvm_ty=llvm_ty)
                    continue

                m = re.match(
                    r"^(%[-A-Za-z0-9._]+)\s*=\s*getelementptr(?:\s+\w+)*\s+([A-Za-z0-9_]+),\s+ptr(?:\s+addrspace\((\d+)\))?\s+([^,]+),\s+i\d+\s+(.+)$",
                    line,
                )
                if m:
                    out_ssa, elem_ty, addr_space, _, _ = m.groups()
                    record_ssa_decl(
                        out_ssa, msl_ty=ptr_type_to_msl(elem_ty, addr_space=addr_space)
                    )
                    continue

                m = re.match(
                    r"^(%[-A-Za-z0-9._]+)\s*=\s*extractelement\s+<\s*\d+\s+x\s+(.+?)\s*>\s+([^,]+),\s+i\d+\s+(.+)$",
                    line,
                )
                if m:
                    out_ssa, elem_ty, _, _ = m.groups()
                    record_ssa_decl(out_ssa, llvm_ty=elem_ty)
                    continue

                m = re.match(
                    r"^(%[-A-Za-z0-9._]+)\s*=\s*insertelement\s+(<\s*\d+\s+x\s+.+\s*>)\s+([^,]+),\s+.+\s+([^,]+),\s+i\d+\s+(.+)$",
                    line,
                )
                if m:
                    out_ssa, vec_ty, _, _, _ = m.groups()
                    record_ssa_decl(out_ssa, llvm_ty=vec_ty)
                    continue

                m = re.match(
                    r"^(%[-A-Za-z0-9._]+)\s*=\s*load\s+([^,]+),\s+ptr(?:\s+addrspace\(\d+\))?\s+(.+)$",
                    line,
                )
                if m:
                    out_ssa, llvm_ty, _ = m.groups()
                    record_ssa_decl(out_ssa, llvm_ty=llvm_ty)
                    continue

        body_lines = [
            "  int __triton_pred_block = -1;",
            f"  int __pc = {block_ids['entry']};",
        ]
        if uses_shared_smem:
            body_lines.insert(
                0, f"  threadgroup char __triton_shared[{shared_bytes}];"
            )
        body_lines.extend(
            [f"  {msl_ty} {name};" for name, msl_ty in ssa_decl_types.items()]
        )
        body_lines.extend([
            "  while (true) {",
            "    switch (__pc) {",
        ])

        for block in block_order:
            block_id = block_ids[block]
            instrs = blocks.get(block, [])
            body_lines.append(f"    case {block_id}: {{")

            def emit(stmt: str):
                body_lines.append(f"      {stmt}")

            terminated = False
            for line in instrs:
                if line == "ret void":
                    emit("return;")
                    terminated = True
                    break

                m = re.match(r"^(%[-A-Za-z0-9._]+)\s*=\s*phi\s+[^ ]+\s+(.+)$", line)
                if m:
                    out_ssa, incoming_raw = m.groups()
                    out = msl_id(out_ssa)
                    ssa[out_ssa] = out
                    incoming_pairs = []
                    for incoming in split_top_level(incoming_raw):
                        pair = incoming.strip()
                        pm = re.match(r'^\[\s*(.+)\s*,\s*%(.+)\s*\]$', pair)
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
                        phi_expr = (
                            f"(__triton_pred_block == {pred_id} ? {to_expr(val)} : {phi_expr})"
                        )
                    emit(f"{out} = {phi_expr};")
                    continue

                m = re.match(r"^(%[-A-Za-z0-9._]+)\s*=\s*(?:tail\s+)?call\s+(.+?)\s+@([A-Za-z0-9_.$-]+)\((.*)\)$", line)
                if m:
                    out_ssa, _, fn, args_raw = m.groups()
                    args = [to_expr(v) for v in parse_call_args(args_raw)]
                    out = msl_id(out_ssa)
                    ssa[out_ssa] = out
                    lowered_intrinsic = lower_intrinsic(fn, args)
                    if lowered_intrinsic is not None:
                        emit(f"{out} = {lowered_intrinsic};")
                    elif fn in axis_helper_map:
                        emit(f"{out} = {axis_helper_map[fn]};")
                    elif fn.startswith("__metal_predicated_ld_global_") and len(args) == 3:
                        emit(f"{out} = ({args[2]} ? *{args[1]} : {args[0]});")
                    elif fn == "__metal_simd_shuffle_xor" and len(args) == 2:
                        emit(f"{out} = simd_shuffle_xor({args[0]}, {args[1]});")
                    elif fn == "__metal_simd_shuffle_up" and len(args) == 2:
                        emit(f"{out} = simd_shuffle_up({args[0]}, {args[1]});")
                    elif fn == "__metal_simd_shuffle" and len(args) == 2:
                        emit(f"{out} = simd_shuffle({args[0]}, {args[1]});")
                    else:
                        emit(f"{out} = {fn}({', '.join(args)});")
                    continue

                m = re.match(r"^(?:tail\s+)?call(?:\s+\w+)*\s+void\s+@([A-Za-z0-9_.$-]+)\((.*)\)$", line)
                if m:
                    fn, args_raw = m.groups()
                    args = [to_expr(v) for v in parse_call_args(args_raw)]
                    if fn.startswith("__metal_predicated_st_global_") and len(args) == 3:
                        emit(f"if ({args[2]}) {{ *{args[1]} = {args[0]}; }}")
                    elif fn == "__metal_simdgroup_barrier":
                        emit("threadgroup_barrier(mem_flags::mem_none);")
                    elif fn.startswith("llvm.assume"):
                        emit("(void)0;")
                    else:
                        emit(f"{fn}({', '.join(args)});")
                    continue

                m = re.match(
                    r"^(%[-A-Za-z0-9._]+)\s*=\s*(add|sub|mul|udiv|sdiv|urem|srem|shl|lshr|ashr|and|or|xor|fadd|fsub|fmul|fdiv|frem)(?:\s+[A-Za-z]+)*\s+(.+)$",
                    line,
                )
                if m:
                    out_ssa, op, operands_spec = m.groups()
                    parts = split_top_level(operands_spec)
                    if len(parts) != 2:
                        raise RuntimeError(
                            f"Unsupported binary operand form in Metal lowering: '{line}'"
                        )
                    _, lhs = split_typed_value(parts[0])
                    _, rhs = split_typed_value(parts[1])
                    out = msl_id(out_ssa)
                    ssa[out_ssa] = out
                    lhs_expr = to_expr(lhs)
                    rhs_expr = to_expr(rhs)
                    if op in float_bin_map:
                        emit(f"{out} = {lhs_expr} {float_bin_map[op]} {rhs_expr};")
                    elif op == "frem":
                        emit(f"{out} = fmod({lhs_expr}, {rhs_expr});")
                    else:
                        emit(f"{out} = {lhs_expr} {bin_map[op]} {rhs_expr};")
                    continue

                m = re.match(
                    r"^(%[-A-Za-z0-9._]+)\s*=\s*fneg(?:\s+[A-Za-z]+)*\s+[^ ]+\s+(.+)$",
                    line,
                )
                if m:
                    out_ssa, val = m.groups()
                    out = msl_id(out_ssa)
                    ssa[out_ssa] = out
                    emit(f"{out} = -({to_expr(val)});")
                    continue

                m = re.match(r"^(%[-A-Za-z0-9._]+)\s*=\s*icmp\s+(\w+)\s+[^ ]+\s+([^,]+),\s*(.+)$", line)
                if m:
                    out_ssa, pred, lhs, rhs = m.groups()
                    out = msl_id(out_ssa)
                    ssa[out_ssa] = out
                    cmp_op = cmp_map.get(pred)
                    if cmp_op is None:
                        raise RuntimeError(f"Unsupported icmp predicate '{pred}'")
                    emit(f"{out} = ({to_expr(lhs)} {cmp_op} {to_expr(rhs)});")
                    continue

                m = re.match(r"^(%[-A-Za-z0-9._]+)\s*=\s*fcmp\s+(\w+)\s+[^ ]+\s+([^,]+),\s*(.+)$", line)
                if m:
                    out_ssa, pred, lhs, rhs = m.groups()
                    out = msl_id(out_ssa)
                    ssa[out_ssa] = out
                    lhs_expr = to_expr(lhs)
                    rhs_expr = to_expr(rhs)
                    emit(f"{out} = {fcmp_expr(pred, lhs_expr, rhs_expr)};")
                    continue

                m = re.match(
                    r"^(%[-A-Za-z0-9._]+)\s*=\s*(sext|zext|trunc|sitofp|uitofp|fptosi|fptoui|bitcast|addrspacecast|ptrtoint|inttoptr)\s+(.+)\s+to\s+(.+)$",
                    line,
                )
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
                            emit(f"{out} = as_type<{llvm_type_to_msl(dst_ty)}>({to_expr(val)});")
                    elif op in ("ptrtoint", "inttoptr"):
                        emit(f"{out} = {to_expr(val)};")
                    else:
                        emit(f"{out} = ({llvm_type_to_msl(dst_ty)})({to_expr(val)});")
                    continue

                m = re.match(r"^(%[-A-Za-z0-9._]+)\s*=\s*freeze\s+[^ ]+\s+(.+)$", line)
                if m:
                    out_ssa, val = m.groups()
                    out = msl_id(out_ssa)
                    ssa[out_ssa] = out
                    emit(f"{out} = {to_expr(val)};")
                    continue

                m = re.match(r"^(%[-A-Za-z0-9._]+)\s*=\s*select\s+i1\s+([^,]+),\s+[^ ]+\s+([^,]+),\s+[^ ]+\s+(.+)$", line)
                if m:
                    out_ssa, cond, lhs, rhs = m.groups()
                    out = msl_id(out_ssa)
                    ssa[out_ssa] = out
                    emit(f"{out} = ({to_expr(cond)} ? {to_expr(lhs)} : {to_expr(rhs)});")
                    continue

                m = re.match(
                    r"^(%[-A-Za-z0-9._]+)\s*=\s*getelementptr(?:\s+\w+)*\s+[A-Za-z0-9_]+,\s+ptr(?:\s+addrspace\(\d+\))?\s+([^,]+),\s+i\d+\s+(.+)$",
                    line,
                )
                if m:
                    out_ssa, base, idx = m.groups()
                    out = msl_id(out_ssa)
                    ssa[out_ssa] = out
                    emit(f"{out} = {to_expr(base)} + {to_expr(idx)};")
                    continue

                m = re.match(
                    r"^(%[-A-Za-z0-9._]+)\s*=\s*extractelement\s+<\s*(\d+)\s+x\s+.+\s*>\s+([^,]+),\s+i\d+\s+(.+)$",
                    line,
                )
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

                m = re.match(
                    r"^(%[-A-Za-z0-9._]+)\s*=\s*insertelement\s+<\s*(\d+)\s+x\s+.+\s*>\s+([^,]+),\s+.+\s+([^,]+),\s+i\d+\s+(.+)$",
                    line,
                )
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

                m = re.match(
                    r"^(%[-A-Za-z0-9._]+)\s*=\s*load\s+(.+?),\s+ptr(?:\s+addrspace\((\d+)\))?\s+(.+)$",
                    line,
                )
                if m:
                    out_ssa, llvm_ty, addr_space, ptr = m.groups()
                    out = msl_id(out_ssa)
                    ssa[out_ssa] = out
                    llvm_ty = llvm_ty.strip()
                    ptr_expr = to_expr(strip_operand_attrs(ptr))
                    msl_ty = llvm_type_to_msl(llvm_ty)
                    emit(
                        f"{out} = *(({msl_addr_space(addr_space)} {msl_ty}*)({ptr_expr}));"
                    )
                    continue

                m = re.match(
                    r"^store\s+(.+?),\s+ptr(?:\s+addrspace\((\d+)\))?\s+(.+)$",
                    line,
                )
                if m:
                    val_spec, addr_space, ptr = m.groups()
                    llvm_ty, val_token = split_typed_value(val_spec)
                    msl_ty = llvm_type_to_msl(llvm_ty)
                    ptr_expr = to_expr(strip_operand_attrs(ptr))
                    emit(
                        f"*(({msl_addr_space(addr_space)} {msl_ty}*)({ptr_expr})) = {to_expr(val_token)};"
                    )
                    continue

                m = re.match(r"^br\s+label\s+%(.+)$", line)
                if m:
                    target = normalize_label(m.group(1))
                    target_id = block_ids.get(target)
                    if target_id is None:
                        raise RuntimeError(
                            f"Unknown branch target '{target}' in Metal lowering"
                        )
                    emit(f"__triton_pred_block = {block_id};")
                    emit(f"__pc = {target_id};")
                    emit("continue;")
                    terminated = True
                    break

                m = re.match(
                    r"^br\s+i1\s+([^,]+),\s+label\s+%([^,]+),\s+label\s+%(.+)$",
                    line,
                )
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
                    emit(
                        f"if ({to_expr(cond)}) {{ __triton_pred_block = {block_id}; __pc = {t_id}; }} "
                        f"else {{ __triton_pred_block = {block_id}; __pc = {f_id}; }}"
                    )
                    emit("continue;")
                    terminated = True
                    break

                if line.startswith("unreachable"):
                    emit("return;")
                    terminated = True
                    break

                raise RuntimeError(f"Unsupported LLVM IR in Metal lowering: '{line}'")

            if not terminated:
                emit("return;")
            body_lines.append("    }")

        body_lines.append("    default: return;")
        body_lines.append("    }")
        body_lines.append("  }")

        msl_lines = [
            "#include <metal_stdlib>",
            "using namespace metal;",
            "",
            f"kernel void {msl_kernel_name}(",
            ",\n".join(param_lines),
            ") {",
        ]
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
        return f"{version}-{self.target.arch}"
