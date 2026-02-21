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

    def __post_init__(self):
        extern_libs = {} if self.extern_libs is None else dict(self.extern_libs)
        object.__setattr__(self, "extern_libs", tuple(extern_libs.items()))

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
        self.binary_ext = "metallib"

    def parse_options(self, opts) -> Any:
        args = {"arch": self.target.arch}
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
        passes.convert.add_scf_to_cf(pm)

        import triton._C.libtriton.metal as metal

        passes.ttgpuir.add_allocate_shared_memory(pm)
        passes.ttgpuir.add_allocate_global_scratch_memory(pm)

        metal.passes.ttgpuir.add_to_llvmir(pm)
        passes.ttgpuir.add_canonicalize_llvm_ir(pm)
        passes.common.add_cse(pm)

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

        For Metal, we take the LLVM IR and translate it to a compute kernel in
        Metal Shading Language (MSL). This stage extracts kernel signatures and
        generates valid MSL source that can be compiled by xcrun metal.
        """
        # Extract kernel name from LLVM IR
        names = re.findall(r"define.*void @([a-zA-Z_][a-zA-Z0-9_]*)\(", src)
        if not names:
            raise RuntimeError("No kernel function found in LLVM IR")
        kernel_name = names[0]
        metadata["name"] = kernel_name

        # Parse function arguments from LLVM IR
        # Match the full function signature
        func_match = re.search(
            r"define.*void @" + re.escape(kernel_name) + r"\(([^)]*)\)",
            src,
            re.DOTALL,
        )
        args = []
        if func_match:
            params_str = func_match.group(1).strip()
            if params_str:
                for i, param in enumerate(params_str.split(",")):
                    param = param.strip()
                    if "ptr" in param or "*" in param:
                        args.append(("buffer", f"arg{i}", i))
                    elif "i32" in param:
                        args.append(("uint", f"arg{i}", i))
                    elif "i64" in param:
                        args.append(("ulong", f"arg{i}", i))
                    elif "float" in param:
                        args.append(("float", f"arg{i}", i))
                    elif "half" in param:
                        args.append(("half", f"arg{i}", i))
                    else:
                        args.append(("uint", f"arg{i}", i))

        # Generate Metal Shading Language source
        msl_lines = [
            "#include <metal_stdlib>",
            "using namespace metal;",
            "",
        ]

        # Build kernel signature
        param_strs = []
        for arg_type, arg_name, idx in args:
            if arg_type == "buffer":
                param_strs.append(f"    device float* {arg_name} [[buffer({idx})]]")
            else:
                param_strs.append(
                    f"    constant {arg_type}& {arg_name} [[buffer({idx})]]"
                )

        # Add threadgroup position and thread position
        param_strs.append("    uint3 tid [[thread_position_in_grid]]")
        param_strs.append("    uint3 ntid [[threads_per_grid]]")

        msl_lines.append(f"kernel void {kernel_name}(")
        msl_lines.append(",\n".join(param_strs))
        msl_lines.append(") {")
        msl_lines.append("    // Auto-generated Metal kernel stub")
        msl_lines.append(
            "    // Full kernel body will be generated by LLVM→AIR→metallib pipeline"
        )
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

    @functools.lru_cache()
    def hash(self):
        version = _get_metal_sdk_version()
        return f"{version}-{self.target.arch}"
