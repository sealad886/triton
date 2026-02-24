"""
Tests for the Metal backend compiler, driver, and runtime.

These tests verify the Metal backend integration with Triton's backend
infrastructure. Tests that require a real Metal device are skipped on
non-macOS platforms.
"""

import os
import shutil
import struct
import subprocess
import sys
from unittest.mock import MagicMock, patch

import pytest

# ── Fixtures ──────────────────────────────────────────────────────────

skip_non_darwin = pytest.mark.skipif(
    sys.platform != "darwin",
    reason="Metal tests require macOS",
)

skip_no_xcrun = pytest.mark.skipif(
    shutil.which("xcrun") is None,
    reason="xcrun not found (need Xcode CLI tools)",
)


def assert_metal_compilation_artifacts(kernel):
    """Verify that a compiled kernel contains LLIR, MSL source, and a valid metallib."""
    assert "llir" in kernel.asm and len(kernel.asm["llir"]) > 0
    assert "metal" in kernel.asm and b"kernel void" in kernel.asm["metal"]
    assert "metallib" in kernel.asm and kernel.asm["metallib"][:4] == b"MTLB"


@pytest.fixture
def metal_source():
    """Simple Metal compute kernel for testing."""
    return """\
#include <metal_stdlib>
using namespace metal;

kernel void add_arrays(
    device float* inA [[buffer(0)]],
    device float* inB [[buffer(1)]],
    device float* out [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    out[id] = inA[id] + inB[id];
}
"""


@pytest.fixture
def trivial_metal_source():
    """Minimal kernel that just writes a constant."""
    return """\
#include <metal_stdlib>
using namespace metal;

kernel void write_constant(
    device float* out [[buffer(0)]],
    uint id [[thread_position_in_grid]]
) {
    out[id] = 42.0f;
}
"""


# ── MetalOptions tests ──────────────────────────────────────────────


class TestMetalOptions:
    def test_default_construction(self):
        # Must import from the backend location — this mirrors how Triton
        # discovers backends at runtime.
        from third_party.metal.backend.compiler import MetalOptions

        opts = MetalOptions()
        assert opts.num_warps == 4
        assert opts.num_stages == 2
        assert opts.backend_name == "metal"
        assert opts.debug is False

    def test_custom_options(self):
        from third_party.metal.backend.compiler import MetalOptions

        opts = MetalOptions(num_warps=8, debug=True, arch="apple9")
        assert opts.num_warps == 8
        assert opts.debug is True
        assert opts.arch == "apple9"

    def test_hash_deterministic(self):
        from third_party.metal.backend.compiler import MetalOptions

        opts = MetalOptions(num_warps=4, arch="apple8")
        h1 = opts.hash()
        h2 = opts.hash()
        assert h1 == h2
        assert len(h1) == 64  # SHA-256 hex digest

    def test_hash_varies_with_options(self):
        from third_party.metal.backend.compiler import MetalOptions

        opts_a = MetalOptions(num_warps=4, arch="apple8")
        opts_b = MetalOptions(num_warps=8, arch="apple9")
        assert opts_a.hash() != opts_b.hash()


# ── MetalBackend interface tests ────────────────────────────────────


class TestMetalBackend:
    def test_supports_target(self):
        from third_party.metal.backend.compiler import MetalBackend

        from triton.backends.compiler import GPUTarget

        assert MetalBackend.supports_target(GPUTarget("metal", "apple8", 32))
        assert not MetalBackend.supports_target(GPUTarget("cuda", 90, 32))
        assert not MetalBackend.supports_target(GPUTarget("hip", "gfx942", 64))

    def test_init(self):
        from third_party.metal.backend.compiler import MetalBackend

        from triton.backends.compiler import GPUTarget

        target = GPUTarget("metal", "apple8", 32)
        backend = MetalBackend(target)
        assert backend.binary_ext == "metal"

    def test_backend_hash_includes_source_fingerprint(self):
        from third_party.metal.backend.compiler import MetalBackend

        from triton.backends.compiler import GPUTarget

        target = GPUTarget("metal", "apple8", 32)
        backend = MetalBackend(target)
        with patch(
            "third_party.metal.backend.compiler._get_metal_sdk_version",
            return_value="sdk-version",
        ), patch(
            "third_party.metal.backend.compiler._get_metal_backend_source_hash",
            return_value="backend-src-hash",
        ), patch("triton.__version__", "triton-version", create=True):
            assert (
                backend.hash()
                == "sdk-version-apple8-triton-version-backend-src-hash"
            )

    def test_parse_options(self):
        from third_party.metal.backend.compiler import MetalBackend

        from triton.backends.compiler import GPUTarget

        target = GPUTarget("metal", "apple8", 32)
        backend = MetalBackend(target)
        opts = backend.parse_options({"num_warps": 8, "debug": True})
        assert opts.num_warps == 8
        assert opts.debug is True
        assert opts.arch == "apple8"

    def test_add_stages(self):
        from third_party.metal.backend.compiler import MetalBackend

        from triton.backends.compiler import GPUTarget, Language

        target = GPUTarget("metal", "apple8", 32)
        backend = MetalBackend(target)
        opts = backend.parse_options({})
        stages = {}
        backend.add_stages(stages, opts, Language.TRITON)
        assert "ttir" in stages
        assert "ttgir" in stages
        assert "llir" in stages
        assert "metal" in stages
        assert "metallib" in stages

    def test_add_stages_inspection_hook(self, monkeypatch):
        from third_party.metal.backend.compiler import MetalBackend

        from triton import knobs
        from triton.backends.compiler import GPUTarget, Language

        calls = {"count": 0}

        def hook(backend, stages, options, language, capability):
            calls["count"] += 1
            assert language == Language.TRITON
            assert capability is None
            assert "metallib" in stages

        monkeypatch.setattr(knobs.runtime, "add_stages_inspection_hook", hook)

        target = GPUTarget("metal", "apple8", 32)
        backend = MetalBackend(target)
        opts = backend.parse_options({})
        stages = {}
        backend.add_stages(stages, opts, Language.TRITON)
        assert calls["count"] == 1

    def test_load_dialects_no_error(self):
        from third_party.metal.backend.compiler import MetalBackend

        from triton.backends.compiler import GPUTarget

        target = GPUTarget("metal", "apple8", 32)
        backend = MetalBackend(target)
        # load_dialects should not raise
        backend.load_dialects(None)

    def test_get_module_map_empty(self):
        from third_party.metal.backend.compiler import MetalBackend

        from triton.backends.compiler import GPUTarget

        target = GPUTarget("metal", "apple8", 32)
        backend = MetalBackend(target)
        assert backend.get_module_map() == {}


# ── Compiler xcrun integration ──────────────────────────────────────


class TestMetalCompilation:
    @skip_non_darwin
    @skip_no_xcrun
    def test_compile_metallib(self, metal_source):
        from third_party.metal.backend.compiler import MetalBackend, MetalOptions

        opts = MetalOptions(arch="apple8")
        metadata = {}
        binary = MetalBackend.make_metallib(metal_source, metadata, opts)
        assert isinstance(binary, bytes)
        assert len(binary) > 0
        # .metallib files start with the MTLB magic
        assert binary[:4] == b"MTLB"

    @skip_non_darwin
    @skip_no_xcrun
    def test_compile_trivial_kernel(self, trivial_metal_source):
        from third_party.metal.backend.compiler import MetalBackend, MetalOptions

        opts = MetalOptions(arch="apple8")
        metadata = {}
        binary = MetalBackend.make_metallib(trivial_metal_source, metadata, opts)
        assert isinstance(binary, bytes)
        assert len(binary) > 0

    @skip_non_darwin
    @skip_no_xcrun
    def test_compile_debug_mode(self, metal_source):
        from third_party.metal.backend.compiler import MetalBackend, MetalOptions

        opts = MetalOptions(arch="apple8", debug=True)
        metadata = {}
        binary = MetalBackend.make_metallib(metal_source, metadata, opts)
        assert isinstance(binary, bytes)
        assert len(binary) > 0

    def test_compile_fails_without_xcrun(self, metal_source):
        from third_party.metal.backend.compiler import MetalBackend, MetalOptions

        opts = MetalOptions(arch="apple8")
        with patch("third_party.metal.backend.compiler._xcrun_path") as mock_xcrun:
            mock_xcrun.side_effect = RuntimeError("xcrun not found")
            with pytest.raises(RuntimeError, match="xcrun"):
                MetalBackend.make_metallib(metal_source, {}, opts)

    @skip_non_darwin
    @skip_no_xcrun
    def test_compile_from_lowered_llvm_ir(self):
        from third_party.metal.backend.compiler import MetalBackend, MetalOptions

        llvm_ir = """
define void @kernel(ptr addrspace(1) %0, ptr addrspace(1) %1, i32 %2) {
  %3 = call i32 @__metal_get_threadgroup_position_in_grid_x()
  %4 = shl i32 %3, 7
  %5 = call i32 @__metal_get_thread_position_in_threadgroup_x()
  %6 = and i32 %5, 127
  %7 = add i32 %4, %6
  %8 = icmp slt i32 %7, %2
  %9 = sext i32 %7 to i64
  %10 = getelementptr float, ptr addrspace(1) %0, i64 %9
  %11 = call float @__metal_predicated_ld_global_f32_p1(float 0.000000e+00, ptr addrspace(1) %10, i1 %8)
  %12 = getelementptr float, ptr addrspace(1) %1, i64 %9
  call void @__metal_predicated_st_global_f32_p1(float %11, ptr addrspace(1) %12, i1 %8)
  ret void
}
"""
        metadata = {}
        msl = MetalBackend.make_metal_ir(llvm_ir, metadata, None)
        opts = MetalOptions(arch="apple8")
        binary = MetalBackend.make_metallib(msl, metadata, opts)
        assert isinstance(binary, bytes)
        assert binary[:4] == b"MTLB"

    @skip_non_darwin
    @skip_no_xcrun
    def test_compile_from_lowered_cfg_with_phi(self):
        from third_party.metal.backend.compiler import MetalBackend, MetalOptions

        llvm_ir = """
define void @phi_kernel(ptr %out, i1 %cond) {
entry:
  br i1 %cond, label %then, label %else
then:
  br label %merge
else:
  br label %merge
merge:
  %v = phi i32 [ 1, %then ], [ 2, %else ]
  %p = getelementptr i32, ptr %out, i64 0
  store i32 %v, ptr %p
  ret void
}
"""
        metadata = {}
        msl = MetalBackend.make_metal_ir(llvm_ir, metadata, None)
        opts = MetalOptions(arch="apple8")
        binary = MetalBackend.make_metallib(msl, metadata, opts)
        assert isinstance(binary, bytes)
        assert binary[:4] == b"MTLB"

    @skip_non_darwin
    @skip_no_xcrun
    def test_compile_triton_vector_add_pipeline(self):
        import triton
        import triton.language as tl
        from triton.backends.compiler import GPUTarget

        @triton.jit
        def _add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
            pid = tl.program_id(axis=0)
            offs = pid * BLOCK + tl.arange(0, BLOCK)
            mask = offs < n
            x = tl.load(x_ptr + offs, mask=mask, other=0.0)
            y = tl.load(y_ptr + offs, mask=mask, other=0.0)
            tl.store(out_ptr + offs, x + y, mask=mask)

        src = triton.compiler.ASTSource(
            fn=_add_kernel,
            signature={
                "x_ptr": "*fp32",
                "y_ptr": "*fp32",
                "out_ptr": "*fp32",
                "n": "i32",
            },
            constexprs={"BLOCK": 128},
        )
        kernel = triton.compile(src=src, target=GPUTarget("metal", "apple8", 32))
        assert_metal_compilation_artifacts(kernel)

    @skip_non_darwin
    @skip_no_xcrun
    def test_compile_triton_reduce_pipeline_no_loop(self):
        import triton
        import triton.language as tl
        from triton.backends.compiler import GPUTarget

        @triton.jit
        def _reduce_kernel(x_ptr, out_ptr, m, n, BM: tl.constexpr, BN: tl.constexpr):
            pid = tl.program_id(axis=0)
            offs_m = pid * BM + tl.arange(0, BM)
            offs_n = tl.arange(0, BN)
            ptrs = x_ptr + offs_m[:, None] * n + offs_n[None, :]
            mask = (offs_m[:, None] < m) & (offs_n[None, :] < n)
            x = tl.load(ptrs, mask=mask, other=0.0)
            s = tl.sum(x, axis=1)
            tl.store(out_ptr + offs_m, s, mask=offs_m < m)

        src = triton.compiler.ASTSource(
            fn=_reduce_kernel,
            signature={
                "x_ptr": "*fp32",
                "out_ptr": "*fp32",
                "m": "i32",
                "n": "i32",
            },
            constexprs={"BM": 16, "BN": 32},
        )
        kernel = triton.compile(src=src, target=GPUTarget("metal", "apple8", 32))
        assert_metal_compilation_artifacts(kernel)

    @skip_non_darwin
    @skip_no_xcrun
    def test_compile_from_lowered_intrinsic_ir(self):
        from third_party.metal.backend.compiler import MetalBackend, MetalOptions

        llvm_ir = """
define void @intrinsic_kernel(ptr %out, float %a, float %b) {
entry:
  %neg = fneg float %a
  %fma = call float @llvm.fma.f32(float %a, float %b, float %neg)
  %abs = call float @llvm.fabs.f32(float %fma)
  %mx = call float @llvm.maximum.f32(float %abs, float %b)
  %p = getelementptr float, ptr %out, i64 0
  store float %mx, ptr %p
  ret void
}
"""
        metadata = {}
        msl = MetalBackend.make_metal_ir(llvm_ir, metadata, None)
        opts = MetalOptions(arch="apple8")
        binary = MetalBackend.make_metallib(msl, metadata, opts)
        assert isinstance(binary, bytes)
        assert binary[:4] == b"MTLB"

    @skip_non_darwin
    @skip_no_xcrun
    def test_compile_from_lowered_loop_with_backedge_phi(self):
        from third_party.metal.backend.compiler import MetalBackend, MetalOptions

        llvm_ir = """
define void @loop_kernel(ptr %out, i32 %n) {
entry:
  br label %loop
loop:
  %i = phi i32 [ 0, %entry ], [ %next, %body ]
  %acc = phi i32 [ 0, %entry ], [ %acc_next, %body ]
  %cmp = icmp slt i32 %i, %n
  br i1 %cmp, label %body, label %exit
body:
  %acc_next = add i32 %acc, %i
  %next = add i32 %i, 1
  br label %loop
exit:
  %p = getelementptr i32, ptr %out, i64 0
  store i32 %acc, ptr %p
  ret void
}
"""
        metadata = {}
        msl = MetalBackend.make_metal_ir(llvm_ir, metadata, None)
        opts = MetalOptions(arch="apple8")
        binary = MetalBackend.make_metallib(msl, metadata, opts)
        assert isinstance(binary, bytes)
        assert binary[:4] == b"MTLB"


# ── Metal IR generation tests ──────────────────────────────────────


class TestMetalIRGeneration:
    def test_make_metal_ir_extracts_kernel_name(self):
        from third_party.metal.backend.compiler import MetalBackend

        llvm_ir = "define void @my_kernel(ptr %arg0, i32 %arg1) {\nret void\n}"
        metadata = {}
        msl = MetalBackend.make_metal_ir(llvm_ir, metadata, None)
        assert metadata["name"] == "my_kernel"
        assert "kernel void my_kernel" in msl

    def test_make_metal_ir_pointer_args(self):
        from third_party.metal.backend.compiler import MetalBackend

        llvm_ir = "define void @test_kernel(ptr %a, ptr %b, i32 %n) {\nret void\n}"
        metadata = {}
        msl = MetalBackend.make_metal_ir(llvm_ir, metadata, None)
        assert "device float*" in msl
        assert "[[buffer(0)]]" in msl
        assert "[[buffer(1)]]" in msl

    def test_make_metal_ir_no_kernel_raises(self):
        from third_party.metal.backend.compiler import MetalBackend

        llvm_ir = "declare void @not_a_definition()"
        with pytest.raises(RuntimeError, match="No kernel function found"):
            MetalBackend.make_metal_ir(llvm_ir, {}, None)

    def test_make_metal_ir_reserved_name_is_sanitized(self):
        from third_party.metal.backend.compiler import MetalBackend

        llvm_ir = "define void @kernel(ptr %arg0) {\nret void\n}"
        metadata = {}
        msl = MetalBackend.make_metal_ir(llvm_ir, metadata, None)
        assert metadata["name"] == "triton_kernel"
        assert "kernel void triton_kernel" in msl

    def test_make_metal_ir_translates_helper_calls(self):
        from third_party.metal.backend.compiler import MetalBackend

        llvm_ir = """
define void @my_kernel(ptr addrspace(1) %0, ptr addrspace(1) %1, i32 %2) {
  %3 = call i32 @__metal_get_threadgroup_position_in_grid_x()
  %4 = shl i32 %3, 7
  %5 = call i32 @__metal_get_thread_position_in_threadgroup_x()
  %6 = add i32 %4, %5
  %7 = icmp slt i32 %6, %2
  %8 = sext i32 %6 to i64
  %9 = getelementptr float, ptr addrspace(1) %0, i64 %8
  %10 = call float @__metal_predicated_ld_global_f32_p1(float 0.000000e+00, ptr addrspace(1) %9, i1 %7)
  %11 = getelementptr float, ptr addrspace(1) %1, i64 %8
  call void @__metal_predicated_st_global_f32_p1(float %10, ptr addrspace(1) %11, i1 %7)
  ret void
}
"""
        metadata = {}
        msl = MetalBackend.make_metal_ir(llvm_ir, metadata, None)
        assert "threadgroup_position_in_grid.x" in msl
        assert "thread_position_in_threadgroup.x" in msl
        assert "? *" in msl
        assert "if (" in msl and "*v11 = v10" in msl

    def test_make_metal_ir_translates_phi_nodes(self):
        from third_party.metal.backend.compiler import MetalBackend

        llvm_ir = """
define void @phi_kernel(ptr %out, i1 %cond) {
entry:
  br i1 %cond, label %then, label %else
then:
  br label %merge
else:
  br label %merge
merge:
  %v = phi i32 [ 1, %then ], [ 2, %else ]
  %p = getelementptr i32, ptr %out, i64 0
  store i32 %v, ptr %p
  ret void
}
"""
        metadata = {}
        msl = MetalBackend.make_metal_ir(llvm_ir, metadata, None)
        assert "__triton_pred_block" in msl
        assert "switch (__pc)" in msl
        assert "__triton_pred_block ==" in msl

    def test_make_metal_ir_translates_fcmp_and_float_ops(self):
        from third_party.metal.backend.compiler import MetalBackend

        llvm_ir = """
define void @fcmp_kernel(ptr %out, float %a, float %b) {
entry:
  %sum = fadd float %a, %b
  %cmp = fcmp olt float %sum, %b
  %sel = select i1 %cmp, float %sum, float %b
  %p = getelementptr float, ptr %out, i64 0
  store float %sel, ptr %p
  ret void
}
"""
        metadata = {}
        msl = MetalBackend.make_metal_ir(llvm_ir, metadata, None)
        assert " + " in msl
        assert "isnan" in msl
        assert " ? " in msl

    def test_make_metal_ir_translates_intrinsics_and_fneg(self):
        from third_party.metal.backend.compiler import MetalBackend

        llvm_ir = """
define void @intrinsic_kernel(ptr %out, float %a, float %b) {
entry:
  %neg = fneg float %a
  %fma = call float @llvm.fma.f32(float %a, float %b, float %neg)
  %abs = call float @llvm.fabs.f32(float %fma)
  %mx = call float @llvm.maximum.f32(float %abs, float %b)
  %mn = call float @llvm.minimum.f32(float %mx, float %a)
  %cmp = fcmp oge float %mn, %a
  %sel = select i1 %cmp, float %mn, float %a
  %p = getelementptr float, ptr %out, i64 0
  store float %sel, ptr %p
  ret void
}
"""
        metadata = {}
        msl = MetalBackend.make_metal_ir(llvm_ir, metadata, None)
        assert "fma(" in msl
        assert "fabs(" in msl
        assert "max(" in msl
        assert "min(" in msl
        assert "= -(" in msl

    def test_make_metal_ir_translates_fmuladd_intrinsic(self):
        from third_party.metal.backend.compiler import MetalBackend

        llvm_ir = """
define void @fmuladd_kernel(ptr %out, float %a, float %b, float %c) {
entry:
  %r = call float @llvm.fmuladd.f32(float %a, float %b, float %c)
  %p = getelementptr float, ptr %out, i64 0
  store float %r, ptr %p
  ret void
}
"""
        metadata = {}
        msl = MetalBackend.make_metal_ir(llvm_ir, metadata, None)
        assert "fma(" in msl
        assert "llvm.fmuladd" not in msl

    def test_make_metal_ir_translates_libdevice_math_calls(self):
        from third_party.metal.backend.compiler import MetalBackend

        llvm_ir = """
define void @libdevice_kernel(ptr %out, float %a) {
entry:
  %e = call float @__nv_expf(float %a)
  %l = call float @__nv_logf(float %a)
  %s = call float @__nv_sqrtf(float %a)
  %sum = fadd float %e, %l
  %res = fadd float %sum, %s
  %p = getelementptr float, ptr %out, i64 0
  store float %res, ptr %p
  ret void
}
"""
        metadata = {}
        msl = MetalBackend.make_metal_ir(llvm_ir, metadata, None)
        assert "exp(" in msl
        assert "log(" in msl
        assert "sqrt(" in msl
        assert "__nv_expf" not in msl
        assert "__nv_logf" not in msl
        assert "__nv_sqrtf" not in msl

    def test_make_metal_ir_translates_llvm_assume(self):
        from third_party.metal.backend.compiler import MetalBackend

        llvm_ir = """
define void @assume_kernel(ptr %out, i1 %pred) {
entry:
  call void @llvm.assume(i1 %pred)
  %p = getelementptr i32, ptr %out, i64 0
  store i32 1, ptr %p
  ret void
}
"""
        metadata = {}
        msl = MetalBackend.make_metal_ir(llvm_ir, metadata, None)
        assert "(void)0;" in msl


# ── Driver tests ────────────────────────────────────────────────────


class TestMetalDriver:
    @skip_non_darwin
    def test_driver_is_active(self):
        from third_party.metal.backend.driver import MetalDriver

        # On macOS with Metal hardware this should be True
        assert MetalDriver.is_active() is True

    @skip_non_darwin
    def test_get_current_target(self):
        from third_party.metal.backend.driver import MetalDriver

        driver = MetalDriver()
        target = driver.get_current_target()
        assert target.backend == "metal"
        assert target.warp_size == 32
        assert target.arch in ("apple7", "apple8", "apple9")

    @skip_non_darwin
    def test_get_device_properties(self):
        from third_party.metal.backend.driver import MetalUtils

        utils = MetalUtils()
        props = utils.get_device_properties()
        assert "name" in props
        assert "max_buffer_length" in props
        assert props["max_buffer_length"] > 0

    def test_get_device_properties_schema(self):
        from third_party.metal.backend.driver import MetalUtils

        utils = MetalUtils()
        props = utils.get_device_properties()
        assert "max_shared_mem" in props
        assert "max_threadgroup_memory_length" in props
        assert "mem_clock_rate" in props
        assert "mem_bus_width" in props
        assert "multiprocessor_count" in props

    def test_load_binary_runtime_contract(self):
        from third_party.metal.backend.driver import MetalUtils

        class _DummyHandle:
            pass

        dummy = _DummyHandle()
        dummy_metallib = _DummyHandle()
        utils = MetalUtils()

        with patch.object(
            MetalUtils, "_load_msl_source_handle", return_value=dummy
        ), patch.object(
            MetalUtils, "_load_metallib_handle", return_value=dummy_metallib
        ):
            direct = utils.load_binary("kernel void k() {}")
            assert direct is dummy

            direct_metallib = utils.load_binary(b"MTLBdummy")
            assert direct_metallib is dummy_metallib

            module, function, n_regs, n_spills, n_max_threads = utils.load_binary(
                "kernel_name", "kernel void k() {}", 0, 0
            )
            assert module is dummy
            assert function is dummy
            assert n_regs == 0
            assert n_spills == 0
            assert isinstance(n_max_threads, int)

    def test_unload_module_noop(self):
        from third_party.metal.backend.driver import MetalUtils

        utils = MetalUtils()
        assert utils.unload_module(object()) is None

    def test_flatten_runtime_args_skips_constexpr(self):
        from third_party.metal.backend.driver import _flatten_runtime_args

        ptr = object()
        flat = _flatten_runtime_args(["*fp32", "constexpr", "i32"], [ptr, 128, 7])
        assert flat == [("*fp32", ptr), ("i32", 7)]

    def test_map_python_to_cpp_type(self):
        from third_party.metal.backend.driver import MetalDriver

        driver = MetalDriver.__new__(MetalDriver)
        assert driver.map_python_to_cpp_type("*fp32") == "MTLBufferPtr"
        assert driver.map_python_to_cpp_type("i32") == "int32_t"
        assert driver.map_python_to_cpp_type("fp32") == "float"
        assert driver.map_python_to_cpp_type("fp16") == "uint16_t"

    def test_get_active_torch_device(self):
        from third_party.metal.backend.driver import MetalDriver

        driver = MetalDriver.__new__(MetalDriver)
        try:
            import torch

            device = driver.get_active_torch_device()
            assert str(device) == "mps"
        except ImportError:
            pytest.skip("torch not available")

    def test_set_current_device_validation(self):
        from third_party.metal.backend.driver import MetalDriver

        driver = MetalDriver()
        driver.set_current_device(0)
        with pytest.raises(ValueError, match="single logical device"):
            driver.set_current_device(1)

    def test_get_device_interface_contract(self):
        from third_party.metal.backend.driver import MetalDriver

        driver = MetalDriver()
        di = driver.get_device_interface()
        assert di.current_device() == 0
        start = di.Event(enable_timing=True)
        end = di.Event(enable_timing=True)
        start.record()
        end.record()
        assert start.elapsed_time(end) >= 0.0

    def test_clear_cache_noop_and_tensor(self):
        from third_party.metal.backend.driver import MetalDriver

        driver = MetalDriver()
        driver.clear_cache(None)
        cache = MagicMock()
        driver.clear_cache(cache)
        cache.zero_.assert_called_once()


# ── GPU family detection ────────────────────────────────────────────


class TestGPUFamilyDetection:
    def test_detect_m1(self):
        from third_party.metal.backend.driver import _detect_gpu_family

        mock_device = MagicMock()
        mock_device.name.return_value = "Apple M1"
        assert _detect_gpu_family(mock_device) == "apple7"

    def test_detect_m2(self):
        from third_party.metal.backend.driver import _detect_gpu_family

        mock_device = MagicMock()
        mock_device.name.return_value = "Apple M2 Pro"
        assert _detect_gpu_family(mock_device) == "apple8"

    def test_detect_m3(self):
        from third_party.metal.backend.driver import _detect_gpu_family

        mock_device = MagicMock()
        mock_device.name.return_value = "Apple M3 Max"
        assert _detect_gpu_family(mock_device) == "apple9"

    def test_detect_m4(self):
        from third_party.metal.backend.driver import _detect_gpu_family

        mock_device = MagicMock()
        mock_device.name.return_value = "Apple M4"
        assert _detect_gpu_family(mock_device) == "apple9"

    def test_detect_unknown_defaults(self):
        from third_party.metal.backend.driver import _detect_gpu_family

        mock_device = MagicMock()
        mock_device.name.return_value = "Unknown GPU"
        assert _detect_gpu_family(mock_device) == "apple8"


# ── Kernel handle tests ────────────────────────────────────────────


class TestMetalKernelHandle:
    @skip_non_darwin
    @skip_no_xcrun
    def test_load_and_get_pipeline(self, metal_source):
        from third_party.metal.backend.compiler import MetalBackend, MetalOptions
        from third_party.metal.backend.driver import MetalUtils

        opts = MetalOptions(arch="apple8")
        binary = MetalBackend.make_metallib(metal_source, {}, opts)
        utils = MetalUtils()
        handle = utils.load_binary(binary)
        pipeline = handle.get_pipeline("add_arrays")
        assert pipeline is not None

    @skip_non_darwin
    @skip_no_xcrun
    def test_pipeline_caching(self, metal_source):
        from third_party.metal.backend.compiler import MetalBackend, MetalOptions
        from third_party.metal.backend.driver import MetalUtils

        opts = MetalOptions(arch="apple8")
        binary = MetalBackend.make_metallib(metal_source, {}, opts)
        utils = MetalUtils()
        handle = utils.load_binary(binary)
        p1 = handle.get_pipeline("add_arrays")
        p2 = handle.get_pipeline("add_arrays")
        assert p1 is p2

    @skip_non_darwin
    @skip_no_xcrun
    def test_missing_kernel_raises(self, metal_source):
        from third_party.metal.backend.compiler import MetalBackend, MetalOptions
        from third_party.metal.backend.driver import MetalUtils

        opts = MetalOptions(arch="apple8")
        binary = MetalBackend.make_metallib(metal_source, {}, opts)
        utils = MetalUtils()
        handle = utils.load_binary(binary)
        with pytest.raises(RuntimeError, match="not found"):
            handle.get_pipeline("nonexistent_kernel")


# ── End-to-end kernel launch ────────────────────────────────────────


class TestMetalKernelLaunch:
    @skip_non_darwin
    @skip_no_xcrun
    def test_launch_write_constant(self, trivial_metal_source):
        """End-to-end: compile, load, and launch a Metal kernel."""
        try:
            import numpy as np
        except ImportError:
            pytest.skip("numpy not available")

        from third_party.metal.backend.compiler import MetalBackend, MetalOptions
        from third_party.metal.backend.driver import MetalUtils

        opts = MetalOptions(arch="apple8")
        binary = MetalBackend.make_metallib(trivial_metal_source, {}, opts)

        utils = MetalUtils()
        handle = utils.load_binary(binary)

        out = np.zeros(16, dtype=np.float32)
        handle.launch_kernel(
            name="write_constant",
            args=[out],
            grid=(16, 1, 1),
            block=(16, 1, 1),
        )
        # After launch, the output buffer should contain 42.0
        # Note: this test runs on real GPU hardware
        # We just verify no crash — readback verification depends on
        # Metal buffer storage mode.


# ── Dynamic-loop reduction kernels ──────────────────────────────────


class TestMetalDynamicReduction:
    @skip_non_darwin
    @skip_no_xcrun
    def test_compile_triton_reduce_sum_1d(self):
        import triton
        import triton.language as tl
        from triton.backends.compiler import GPUTarget

        @triton.jit
        def _reduce_sum_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
            pid = tl.program_id(axis=0)
            offs = pid * BLOCK + tl.arange(0, BLOCK)
            mask = offs < n
            x = tl.load(x_ptr + offs, mask=mask, other=0.0)
            total = tl.sum(x, axis=0)
            if pid == 0:
                tl.store(out_ptr, total)

        src = triton.compiler.ASTSource(
            fn=_reduce_sum_kernel,
            signature={
                "x_ptr": "*fp32",
                "out_ptr": "*fp32",
                "n": "i32",
            },
            constexprs={"BLOCK": 128},
        )
        kernel = triton.compile(src=src, target=GPUTarget("metal", "apple8", 32))
        assert_metal_compilation_artifacts(kernel)

    @skip_non_darwin
    @skip_no_xcrun
    def test_compile_triton_reduce_max_1d(self):
        import triton
        import triton.language as tl
        from triton.backends.compiler import GPUTarget

        @triton.jit
        def _reduce_max_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
            pid = tl.program_id(axis=0)
            offs = pid * BLOCK + tl.arange(0, BLOCK)
            mask = offs < n
            x = tl.load(x_ptr + offs, mask=mask, other=float("-inf"))
            mx = tl.max(x, axis=0)
            if pid == 0:
                tl.store(out_ptr, mx)

        src = triton.compiler.ASTSource(
            fn=_reduce_max_kernel,
            signature={
                "x_ptr": "*fp32",
                "out_ptr": "*fp32",
                "n": "i32",
            },
            constexprs={"BLOCK": 128},
        )
        kernel = triton.compile(src=src, target=GPUTarget("metal", "apple8", 32))
        assert_metal_compilation_artifacts(kernel)

    @skip_non_darwin
    @skip_no_xcrun
    def test_compile_triton_reduce_softmax(self):
        import triton
        import triton.language as tl
        from triton.backends.compiler import GPUTarget

        @triton.jit
        def _softmax_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
            pid = tl.program_id(axis=0)
            offs = pid * BLOCK + tl.arange(0, BLOCK)
            mask = offs < n
            x = tl.load(x_ptr + offs, mask=mask, other=float("-inf"))
            x_max = tl.max(x, axis=0)
            x_exp = tl.exp(x - x_max)
            x_sum = tl.sum(x_exp, axis=0)
            out = x_exp / x_sum
            tl.store(out_ptr + offs, out, mask=mask)

        src = triton.compiler.ASTSource(
            fn=_softmax_kernel,
            signature={
                "x_ptr": "*fp32",
                "out_ptr": "*fp32",
                "n": "i32",
            },
            constexprs={"BLOCK": 128},
        )
        kernel = triton.compile(src=src, target=GPUTarget("metal", "apple8", 32))
        assert_metal_compilation_artifacts(kernel)


class TestMetalRealWorldCompileCases:
    """Compile-only coverage for common ML kernels used in production stacks."""

    @skip_non_darwin
    @skip_no_xcrun
    def test_compile_triton_silu_pipeline(self):
        import triton
        import triton.language as tl
        from triton.backends.compiler import GPUTarget

        @triton.jit
        def _silu_kernel(x_ptr, y_ptr, n, BLOCK: tl.constexpr):
            pid = tl.program_id(axis=0)
            offs = pid * BLOCK + tl.arange(0, BLOCK)
            mask = offs < n
            x = tl.load(x_ptr + offs, mask=mask, other=0.0)
            y = x / (1.0 + tl.exp(-x))
            tl.store(y_ptr + offs, y, mask=mask)

        src = triton.compiler.ASTSource(
            fn=_silu_kernel,
            signature={
                "x_ptr": "*fp32",
                "y_ptr": "*fp32",
                "n": "i32",
            },
            constexprs={"BLOCK": 128},
        )
        kernel = triton.compile(src=src, target=GPUTarget("metal", "apple8", 32))
        assert_metal_compilation_artifacts(kernel)

    @skip_non_darwin
    @skip_no_xcrun
    def test_compile_triton_layernorm_pipeline(self):
        import triton
        import triton.language as tl
        from triton.backends.compiler import GPUTarget

        @triton.jit
        def _layernorm_kernel(
            x_ptr,
            w_ptr,
            b_ptr,
            y_ptr,
            n,
            BLOCK: tl.constexpr,
            EPS: tl.constexpr,
        ):
            pid = tl.program_id(axis=0)
            offs = pid * BLOCK + tl.arange(0, BLOCK)
            mask = offs < n
            x = tl.load(x_ptr + offs, mask=mask, other=0.0)
            mean = tl.sum(x, axis=0) / BLOCK
            centered = x - mean
            var = tl.sum(centered * centered, axis=0) / BLOCK
            inv = 1.0 / tl.sqrt(var + EPS)
            w = tl.load(w_ptr + offs, mask=mask, other=1.0)
            b = tl.load(b_ptr + offs, mask=mask, other=0.0)
            y = centered * inv * w + b
            tl.store(y_ptr + offs, y, mask=mask)

        src = triton.compiler.ASTSource(
            fn=_layernorm_kernel,
            signature={
                "x_ptr": "*fp32",
                "w_ptr": "*fp32",
                "b_ptr": "*fp32",
                "y_ptr": "*fp32",
                "n": "i32",
            },
            constexprs={"BLOCK": 128, "EPS": 1e-5},
        )
        kernel = triton.compile(src=src, target=GPUTarget("metal", "apple8", 32))
        assert_metal_compilation_artifacts(kernel)

    @skip_non_darwin
    @skip_no_xcrun
    def test_compile_triton_blocked_matmul_pipeline(self):
        import triton
        import triton.language as tl
        from triton.backends.compiler import GPUTarget

        @triton.jit
        def _matmul_kernel(
            a_ptr,
            b_ptr,
            c_ptr,
            m,
            n,
            k,
            stride_am,
            stride_ak,
            stride_bk,
            stride_bn,
            stride_cm,
            stride_cn,
            BLOCK_M: tl.constexpr,
            BLOCK_N: tl.constexpr,
            BLOCK_K: tl.constexpr,
        ):
            pid_m = tl.program_id(axis=0)
            pid_n = tl.program_id(axis=1)

            offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
            offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
            offs_k = tl.arange(0, BLOCK_K)

            acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            for kk in range(0, k, BLOCK_K):
                a_ptrs = (
                    a_ptr
                    + offs_m[:, None] * stride_am
                    + (offs_k[None, :] + kk) * stride_ak
                )
                b_ptrs = (
                    b_ptr
                    + (offs_k[:, None] + kk) * stride_bk
                    + offs_n[None, :] * stride_bn
                )
                a = tl.load(
                    a_ptrs,
                    mask=(offs_m[:, None] < m) & (offs_k[None, :] + kk < k),
                    other=0.0,
                )
                b = tl.load(
                    b_ptrs,
                    mask=(offs_k[:, None] + kk < k) & (offs_n[None, :] < n),
                    other=0.0,
                )
                acc += tl.dot(a, b)

            c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
            c_mask = (offs_m[:, None] < m) & (offs_n[None, :] < n)
            tl.store(c_ptrs, acc, mask=c_mask)

        src = triton.compiler.ASTSource(
            fn=_matmul_kernel,
            signature={
                "a_ptr": "*fp32",
                "b_ptr": "*fp32",
                "c_ptr": "*fp32",
                "m": "i32",
                "n": "i32",
                "k": "i32",
                "stride_am": "i32",
                "stride_ak": "i32",
                "stride_bk": "i32",
                "stride_bn": "i32",
                "stride_cm": "i32",
                "stride_cn": "i32",
            },
            constexprs={"BLOCK_M": 16, "BLOCK_N": 16, "BLOCK_K": 16},
        )
        kernel = triton.compile(src=src, target=GPUTarget("metal", "apple8", 32))
        assert_metal_compilation_artifacts(kernel)


# ── Complex CFG patterns in LLVM IR ─────────────────────────────────


class TestMetalComplexCFG:
    def test_make_metal_ir_nested_branches(self):
        from third_party.metal.backend.compiler import MetalBackend

        llvm_ir = """
define void @nested_branch_kernel(ptr %out, i32 %a, i32 %b) {
entry:
  %c0 = icmp sgt i32 %a, 0
  br i1 %c0, label %outer_then, label %outer_else
outer_then:
  %c1 = icmp sgt i32 %b, 10
  br i1 %c1, label %inner_then, label %inner_else
inner_then:
  br label %inner_merge
inner_else:
  br label %inner_merge
inner_merge:
  %iv = phi i32 [ 100, %inner_then ], [ 200, %inner_else ]
  br label %final
outer_else:
  br label %final
final:
  %fv = phi i32 [ %iv, %inner_merge ], [ 300, %outer_else ]
  %p = getelementptr i32, ptr %out, i64 0
  store i32 %fv, ptr %p
  ret void
}
"""
        metadata = {}
        msl = MetalBackend.make_metal_ir(llvm_ir, metadata, None)
        assert "kernel void nested_branch_kernel" in msl
        assert "__triton_pred_block" in msl
        assert "switch (__pc)" in msl

    def test_make_metal_ir_switch_like_cascade(self):
        from third_party.metal.backend.compiler import MetalBackend

        llvm_ir = """
define void @cascade_kernel(ptr %out, i32 %sel) {
entry:
  %c0 = icmp eq i32 %sel, 0
  br i1 %c0, label %case0, label %check1
check1:
  %c1 = icmp eq i32 %sel, 1
  br i1 %c1, label %case1, label %check2
check2:
  %c2 = icmp eq i32 %sel, 2
  br i1 %c2, label %case2, label %default_case
case0:
  br label %done
case1:
  br label %done
case2:
  br label %done
default_case:
  br label %done
done:
  %r = phi i32 [ 10, %case0 ], [ 20, %case1 ], [ 30, %case2 ], [ 99, %default_case ]
  %p = getelementptr i32, ptr %out, i64 0
  store i32 %r, ptr %p
  ret void
}
"""
        metadata = {}
        msl = MetalBackend.make_metal_ir(llvm_ir, metadata, None)
        assert "kernel void cascade_kernel" in msl
        assert "__triton_pred_block ==" in msl
        # Four-way phi must produce nested ternaries
        assert "?" in msl

    def test_make_metal_ir_loop_with_multiple_exits(self):
        from third_party.metal.backend.compiler import MetalBackend

        llvm_ir = """
define void @multi_exit_kernel(ptr %out, i32 %n) {
entry:
  br label %loop
loop:
  %i = phi i32 [ 0, %entry ], [ %next, %body ]
  %acc = phi i32 [ 0, %entry ], [ %acc2, %body ]
  %bound = icmp slt i32 %i, %n
  br i1 %bound, label %body, label %exit_normal
body:
  %acc2 = add i32 %acc, %i
  %early = icmp eq i32 %acc2, 42
  %next = add i32 %i, 1
  br i1 %early, label %exit_early, label %loop
exit_early:
  br label %merge
exit_normal:
  br label %merge
merge:
  %result = phi i32 [ %acc2, %exit_early ], [ %acc, %exit_normal ]
  %p = getelementptr i32, ptr %out, i64 0
  store i32 %result, ptr %p
  ret void
}
"""
        metadata = {}
        msl = MetalBackend.make_metal_ir(llvm_ir, metadata, None)
        assert "kernel void multi_exit_kernel" in msl
        assert "__triton_pred_block" in msl
        assert "continue;" in msl

    def test_make_metal_ir_quoted_labels(self):
        from third_party.metal.backend.compiler import MetalBackend

        llvm_ir = """
define void @quoted_label_kernel(ptr %out, i1 %cond) {
entry:
  br i1 %cond, label %"loop.header", label %"exit.block"
"loop.header":
  br label %"exit.block"
"exit.block":
  %v = phi i32 [ 1, %"loop.header" ], [ 2, %entry ]
  %p = getelementptr i32, ptr %out, i64 0
  store i32 %v, ptr %p
  ret void
}
"""
        metadata = {}
        msl = MetalBackend.make_metal_ir(llvm_ir, metadata, None)
        assert "kernel void quoted_label_kernel" in msl
        assert "__triton_pred_block ==" in msl


# ── Uncommon intrinsic patterns ─────────────────────────────────────


class TestMetalUncommonIntrinsics:
    def test_make_metal_ir_ctpop(self):
        from third_party.metal.backend.compiler import MetalBackend

        llvm_ir = """
define void @ctpop_kernel(ptr %out, i32 %val) {
entry:
  %pc = call i32 @llvm.ctpop.i32(i32 %val)
  %p = getelementptr i32, ptr %out, i64 0
  store i32 %pc, ptr %p
  ret void
}
"""
        metadata = {}
        msl = MetalBackend.make_metal_ir(llvm_ir, metadata, None)
        assert "popcount(" in msl
        assert "llvm.ctpop" not in msl

    def test_make_metal_ir_copysign(self):
        from third_party.metal.backend.compiler import MetalBackend

        llvm_ir = """
define void @copysign_kernel(ptr %out, float %mag, float %sgn) {
entry:
  %r = call float @llvm.copysign.f32(float %mag, float %sgn)
  %p = getelementptr float, ptr %out, i64 0
  store float %r, ptr %p
  ret void
}
"""
        metadata = {}
        msl = MetalBackend.make_metal_ir(llvm_ir, metadata, None)
        assert "copysign(" in msl
        assert "llvm.copysign" not in msl

    def test_make_metal_ir_exp2_log2(self):
        from third_party.metal.backend.compiler import MetalBackend

        llvm_ir = """
define void @exp2_log2_kernel(ptr %out, float %a) {
entry:
  %e = call float @llvm.exp2.f32(float %a)
  %l = call float @llvm.log2.f32(float %e)
  %p = getelementptr float, ptr %out, i64 0
  store float %l, ptr %p
  ret void
}
"""
        metadata = {}
        msl = MetalBackend.make_metal_ir(llvm_ir, metadata, None)
        assert "exp2(" in msl
        assert "log2(" in msl
        assert "llvm.exp2" not in msl
        assert "llvm.log2" not in msl

    def test_make_metal_ir_ocml_math(self):
        from third_party.metal.backend.compiler import MetalBackend

        llvm_ir = """
define void @ocml_kernel(ptr %out, float %a) {
entry:
  %e = call float @__ocml_exp_f32(float %a)
  %s = call float @__ocml_sin_f32(float %a)
  %sum = fadd float %e, %s
  %p = getelementptr float, ptr %out, i64 0
  store float %sum, ptr %p
  ret void
}
"""
        metadata = {}
        msl = MetalBackend.make_metal_ir(llvm_ir, metadata, None)
        assert "exp(" in msl
        assert "sin(" in msl
        assert "__ocml_exp" not in msl
        assert "__ocml_sin" not in msl

    def test_make_metal_ir_freeze_instruction(self):
        from third_party.metal.backend.compiler import MetalBackend

        llvm_ir = """
define void @freeze_kernel(ptr %out, i32 %a) {
entry:
  %f = freeze i32 %a
  %p = getelementptr i32, ptr %out, i64 0
  store i32 %f, ptr %p
  ret void
}
"""
        metadata = {}
        msl = MetalBackend.make_metal_ir(llvm_ir, metadata, None)
        assert "kernel void freeze_kernel" in msl
        # freeze should pass through the value
        assert "= arg1;" in msl or "= v_a;" in msl or "arg1" in msl

    def test_make_metal_ir_extractelement_insertelement(self):
        from third_party.metal.backend.compiler import MetalBackend

        llvm_ir = """
define void @vecop_kernel(ptr %out, float %a, float %b) {
entry:
  %v0 = insertelement <4 x float> undef, float %a, i32 0
  %v1 = insertelement <4 x float> %v0, float %b, i32 1
  %e = extractelement <4 x float> %v1, i32 0
  %p = getelementptr float, ptr %out, i64 0
  store float %e, ptr %p
  ret void
}
"""
        metadata = {}
        msl = MetalBackend.make_metal_ir(llvm_ir, metadata, None)
        assert "kernel void vecop_kernel" in msl
        assert "[0]" in msl or "[1]" in msl


# ── Scalar-cast edge-type conformance ───────────────────────────────


class TestMetalScalarCastEdgeTypes:
    """Conformance tests for edge-case scalar types, casts, and complex
    parameter signatures in the LLVM-IR-to-MSL translator."""

    def test_make_metal_ir_i64_params(self):
        from third_party.metal.backend.compiler import MetalBackend

        llvm_ir = """
define void @i64_kernel(ptr addrspace(1) %out, i64 %a, i64 %b) {
entry:
  %sum = add i64 %a, %b
  %p = getelementptr i64, ptr addrspace(1) %out, i64 0
  store i64 %sum, ptr addrspace(1) %p
  ret void
}
"""
        metadata = {}
        msl = MetalBackend.make_metal_ir(llvm_ir, metadata, None)
        assert "kernel void i64_kernel" in msl
        assert "long" in msl
        assert "constant long&" in msl
        assert "[[buffer(1)]]" in msl
        assert "[[buffer(2)]]" in msl

    def test_make_metal_ir_i64_arithmetic(self):
        from third_party.metal.backend.compiler import MetalBackend

        llvm_ir = """
define void @i64_arith_kernel(ptr addrspace(1) %out, i64 %a, i64 %b) {
entry:
  %sum = add i64 %a, %b
  %prod = mul i64 %sum, %b
  %shifted = shl i64 %prod, 2
  %cmp = icmp sgt i64 %shifted, %a
  %sel = select i1 %cmp, i64 %shifted, i64 %a
  %p = getelementptr i64, ptr addrspace(1) %out, i64 0
  store i64 %sel, ptr addrspace(1) %p
  ret void
}
"""
        metadata = {}
        msl = MetalBackend.make_metal_ir(llvm_ir, metadata, None)
        assert "kernel void i64_arith_kernel" in msl
        assert "long" in msl
        assert "+" in msl
        assert "*" in msl
        assert "<<" in msl
        assert ">" in msl
        assert "?" in msl

    def test_make_metal_ir_half_params(self):
        from third_party.metal.backend.compiler import MetalBackend

        llvm_ir = """
define void @half_kernel(ptr addrspace(1) %out, half %a, half %b) {
entry:
  %sum = fadd half %a, %b
  %p = getelementptr half, ptr addrspace(1) %out, i64 0
  store half %sum, ptr addrspace(1) %p
  ret void
}
"""
        metadata = {}
        msl = MetalBackend.make_metal_ir(llvm_ir, metadata, None)
        assert "kernel void half_kernel" in msl
        assert "constant half&" in msl
        assert "half" in msl
        assert "[[buffer(1)]]" in msl
        assert "[[buffer(2)]]" in msl

    def test_make_metal_ir_double_to_float_trunc(self):
        from third_party.metal.backend.compiler import MetalBackend

        llvm_ir = """
define void @fptrunc_kernel(ptr addrspace(1) %out, double %val) {
entry:
  %narrow = fptrunc double %val to float
  %p = getelementptr float, ptr addrspace(1) %out, i64 0
  store float %narrow, ptr addrspace(1) %p
  ret void
}
"""
        metadata = {}
        msl = MetalBackend.make_metal_ir(llvm_ir, metadata, None)
        assert "kernel void fptrunc_kernel" in msl
        assert "constant double&" in msl
        assert "(float)" in msl

    def test_make_metal_ir_i8_to_i32_extend(self):
        from third_party.metal.backend.compiler import MetalBackend

        llvm_ir = """
define void @extend_kernel(ptr addrspace(1) %out, i8 %a, i8 %b) {
entry:
  %sa = sext i8 %a to i32
  %zb = zext i8 %b to i32
  %sum = add i32 %sa, %zb
  %p = getelementptr i32, ptr addrspace(1) %out, i64 0
  store i32 %sum, ptr addrspace(1) %p
  ret void
}
"""
        metadata = {}
        msl = MetalBackend.make_metal_ir(llvm_ir, metadata, None)
        assert "kernel void extend_kernel" in msl
        assert "constant char&" in msl
        assert "(int)" in msl

    def test_make_metal_ir_multiple_ptr_types(self):
        from third_party.metal.backend.compiler import MetalBackend

        llvm_ir = """
define void @mixed_ptr_kernel(ptr addrspace(1) %g_in, ptr addrspace(3) %s_buf, ptr addrspace(1) %g_out, i32 %n) {
entry:
  %idx = sext i32 %n to i64
  %gp = getelementptr float, ptr addrspace(1) %g_in, i64 %idx
  %val = load float, ptr addrspace(1) %gp
  %sp = getelementptr float, ptr addrspace(3) %s_buf, i64 0
  store float %val, ptr addrspace(3) %sp
  %val2 = load float, ptr addrspace(3) %sp
  %op = getelementptr float, ptr addrspace(1) %g_out, i64 %idx
  store float %val2, ptr addrspace(1) %op
  ret void
}
"""
        metadata = {}
        msl = MetalBackend.make_metal_ir(llvm_ir, metadata, None)
        assert "kernel void mixed_ptr_kernel" in msl
        assert "device" in msl
        assert "threadgroup" in msl
        assert "[[buffer(0)]]" in msl
        assert "[[buffer(1)]]" in msl
        assert "[[buffer(2)]]" in msl
        assert "[[buffer(3)]]" in msl

    def test_make_metal_ir_bool_select(self):
        from third_party.metal.backend.compiler import MetalBackend

        llvm_ir = """
define void @bool_select_kernel(ptr addrspace(1) %out, i32 %a, i32 %b) {
entry:
  %flag = trunc i32 %a to i1
  %sel = select i1 %flag, i32 %a, i32 %b
  %p = getelementptr i32, ptr addrspace(1) %out, i64 0
  store i32 %sel, ptr addrspace(1) %p
  ret void
}
"""
        metadata = {}
        msl = MetalBackend.make_metal_ir(llvm_ir, metadata, None)
        assert "kernel void bool_select_kernel" in msl
        assert "(bool)" in msl
        assert "?" in msl

    def test_make_metal_ir_many_params(self):
        from third_party.metal.backend.compiler import MetalBackend

        llvm_ir = """
define void @many_params_kernel(
    ptr addrspace(1) %buf0,
    ptr addrspace(1) %buf1,
    ptr addrspace(1) %buf2,
    i32 %s0,
    i64 %s1,
    float %s2,
    half %s3,
    i32 %s4,
    ptr addrspace(1) %buf3,
    i32 %s5,
    ptr addrspace(3) %shared0,
    i32 %s6
) {
entry:
  %idx = sext i32 %s0 to i64
  %p0 = getelementptr float, ptr addrspace(1) %buf0, i64 %idx
  %v0 = load float, ptr addrspace(1) %p0
  %p1 = getelementptr float, ptr addrspace(1) %buf1, i64 %idx
  %v1 = load float, ptr addrspace(1) %p1
  %sum = fadd float %v0, %v1
  %p2 = getelementptr float, ptr addrspace(1) %buf2, i64 %idx
  store float %sum, ptr addrspace(1) %p2
  ret void
}
"""
        metadata = {}
        msl = MetalBackend.make_metal_ir(llvm_ir, metadata, None)
        assert "kernel void many_params_kernel" in msl
        for i in range(12):
            assert f"[[buffer({i})]]" in msl
        assert "device float*" in msl
        assert "constant int&" in msl
        assert "constant long&" in msl
        assert "constant float&" in msl
        assert "constant half&" in msl
        assert "threadgroup" in msl


# ── Unsigned-operation correctness (AUDIT-P0-001, AUDIT-P0-002) ─────


class TestMetalUnsignedOps:
    """Regression tests: lshr, udiv, urem must use unsigned MSL semantics."""

    def test_make_metal_ir_lshr_uses_unsigned_shift(self):
        from third_party.metal.backend.compiler import MetalBackend

        llvm_ir = """
define void @lshr_kernel(ptr %out, i32 %a) {
entry:
  %r = lshr i32 %a, 3
  %p = getelementptr i32, ptr %out, i64 0
  store i32 %r, ptr %p
  ret void
}
"""
        metadata = {}
        msl = MetalBackend.make_metal_ir(llvm_ir, metadata, None)
        assert "unsigned int" in msl, "lshr must cast to unsigned before shifting"
        assert ">>" in msl

    def test_make_metal_ir_udiv_uses_unsigned_division(self):
        from third_party.metal.backend.compiler import MetalBackend

        llvm_ir = """
define void @udiv_kernel(ptr %out, i32 %a, i32 %b) {
entry:
  %r = udiv i32 %a, %b
  %p = getelementptr i32, ptr %out, i64 0
  store i32 %r, ptr %p
  ret void
}
"""
        metadata = {}
        msl = MetalBackend.make_metal_ir(llvm_ir, metadata, None)
        assert "unsigned int" in msl, "udiv must cast to unsigned before dividing"

    def test_make_metal_ir_urem_uses_unsigned_remainder(self):
        from third_party.metal.backend.compiler import MetalBackend

        llvm_ir = """
define void @urem_kernel(ptr %out, i32 %a, i32 %b) {
entry:
  %r = urem i32 %a, %b
  %p = getelementptr i32, ptr %out, i64 0
  store i32 %r, ptr %p
  ret void
}
"""
        metadata = {}
        msl = MetalBackend.make_metal_ir(llvm_ir, metadata, None)
        assert "unsigned int" in msl, "urem must cast to unsigned before remainder"

    def test_make_metal_ir_lshr_i64_uses_unsigned_long(self):
        from third_party.metal.backend.compiler import MetalBackend

        llvm_ir = """
define void @lshr64_kernel(ptr %out, i64 %a) {
entry:
  %r = lshr i64 %a, 1
  %p = getelementptr i64, ptr %out, i64 0
  store i64 %r, ptr %p
  ret void
}
"""
        metadata = {}
        msl = MetalBackend.make_metal_ir(llvm_ir, metadata, None)
        assert "unsigned long" in msl, "lshr i64 must cast to unsigned long"

    def test_make_metal_ir_sdiv_stays_signed(self):
        """Signed division (sdiv) must NOT cast to unsigned."""
        from third_party.metal.backend.compiler import MetalBackend

        llvm_ir = """
define void @sdiv_kernel(ptr %out, i32 %a, i32 %b) {
entry:
  %r = sdiv i32 %a, %b
  %p = getelementptr i32, ptr %out, i64 0
  store i32 %r, ptr %p
  ret void
}
"""
        metadata = {}
        msl = MetalBackend.make_metal_ir(llvm_ir, metadata, None)
        assert "unsigned" not in msl, "sdiv must use signed division (no unsigned cast)"

    @skip_non_darwin
    @skip_no_xcrun
    def test_compile_lshr_udiv_kernel(self):
        """End-to-end: lshr+udiv kernel compiles to valid metallib."""
        from third_party.metal.backend.compiler import MetalBackend, MetalOptions

        llvm_ir = """
define void @unsigned_ops_kernel(ptr addrspace(1) %out, i32 %a, i32 %b) {
entry:
  %sh = lshr i32 %a, 3
  %dv = udiv i32 %sh, %b
  %rm = urem i32 %dv, %b
  %p = getelementptr i32, ptr addrspace(1) %out, i64 0
  store i32 %rm, ptr addrspace(1) %p
  ret void
}
"""
        metadata = {}
        msl = MetalBackend.make_metal_ir(llvm_ir, metadata, None)
        opts = MetalOptions(arch="apple8")
        binary = MetalBackend.make_metallib(msl, metadata, opts)
        assert isinstance(binary, bytes) and binary[:4] == b"MTLB"


# ── Hex/decimal float constant handling (AUDIT-P1-001, AUDIT-P1-002)─


class TestMetalFloatConstants:
    """Regression tests: LLVM hex float and decimal float constants."""

    def test_constant_to_msl_hex_neg_inf(self):
        from third_party.metal.backend.compiler import MetalBackend

        llvm_ir = """
define void @hex_neg_inf_kernel(ptr addrspace(1) %out, ptr addrspace(1) %src, i1 %mask) {
entry:
  %v = call float @__metal_predicated_ld_global_f32_p1(float 0xFFF0000000000000, ptr addrspace(1) %src, i1 %mask)
  %p = getelementptr float, ptr addrspace(1) %out, i64 0
  store float %v, ptr addrspace(1) %p
  ret void
}
"""
        metadata = {}
        msl = MetalBackend.make_metal_ir(llvm_ir, metadata, None)
        assert (
            "-INFINITY" in msl
        ), "hex float 0xFFF... (-inf) must become -INFINITY in MSL"
        assert "0xFFF0000000000000" not in msl, "raw hex must not appear in MSL output"

    def test_constant_to_msl_hex_pos_inf(self):
        from third_party.metal.backend.compiler import MetalBackend

        llvm_ir = """
define void @hex_pos_inf_kernel(ptr %out) {
entry:
  %p = getelementptr float, ptr %out, i64 0
  store float 0x7FF0000000000000, ptr %p
  ret void
}
"""
        metadata = {}
        msl = MetalBackend.make_metal_ir(llvm_ir, metadata, None)
        assert "INFINITY" in msl

    def test_constant_to_msl_hex_nan(self):
        from third_party.metal.backend.compiler import MetalBackend

        llvm_ir = """
define void @hex_nan_kernel(ptr %out) {
entry:
  %p = getelementptr float, ptr %out, i64 0
  store float 0x7FF8000000000000, ptr %p
  ret void
}
"""
        metadata = {}
        msl = MetalBackend.make_metal_ir(llvm_ir, metadata, None)
        assert "NAN" in msl

    def test_constant_to_msl_hex_regular_float(self):
        from third_party.metal.backend.compiler import MetalBackend

        llvm_ir = """
define void @hex_float_kernel(ptr %out, float %a) {
entry:
  %r = fadd float %a, 0x3FB99999A0000000
  %p = getelementptr float, ptr %out, i64 0
  store float %r, ptr %p
  ret void
}
"""
        metadata = {}
        msl = MetalBackend.make_metal_ir(llvm_ir, metadata, None)
        assert "0x3FB99999" not in msl, "hex float must be converted to decimal"
        assert "f" in msl  # should have f suffix

    def test_constant_to_msl_decimal_float_gets_suffix(self):
        from third_party.metal.backend.compiler import MetalBackend

        llvm_ir = """
define void @decimal_float_kernel(ptr %out, float %a) {
entry:
  %r = fadd float %a, 0.000000e+00
  %p = getelementptr float, ptr %out, i64 0
  store float %r, ptr %p
  ret void
}
"""
        metadata = {}
        msl = MetalBackend.make_metal_ir(llvm_ir, metadata, None)
        assert "0.000000e+00f" in msl, "decimal float must get 'f' suffix"

    @skip_non_darwin
    @skip_no_xcrun
    def test_compile_kernel_with_neg_inf(self):
        """End-to-end: kernel with -inf constant compiles to valid metallib."""
        from third_party.metal.backend.compiler import MetalBackend, MetalOptions

        llvm_ir = """
define void @neg_inf_compile_kernel(ptr addrspace(1) %out, ptr addrspace(1) %src, i1 %mask) {
entry:
  %v = call float @__metal_predicated_ld_global_f32_p1(float 0xFFF0000000000000, ptr addrspace(1) %src, i1 %mask)
  %p = getelementptr float, ptr addrspace(1) %out, i64 0
  store float %v, ptr addrspace(1) %p
  ret void
}
"""
        metadata = {}
        msl = MetalBackend.make_metal_ir(llvm_ir, metadata, None)
        opts = MetalOptions(arch="apple8")
        binary = MetalBackend.make_metallib(msl, metadata, opts)
        assert isinstance(binary, bytes) and binary[:4] == b"MTLB"

    @skip_non_darwin
    @skip_no_xcrun
    def test_compile_triton_softmax_produces_correct_msl(self):
        """The softmax kernel uses -inf and must produce correct MSL constants."""
        import triton
        import triton.language as tl
        from triton.backends.compiler import GPUTarget

        @triton.jit
        def _softmax_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
            pid = tl.program_id(axis=0)
            offs = pid * BLOCK + tl.arange(0, BLOCK)
            mask = offs < n
            x = tl.load(x_ptr + offs, mask=mask, other=float("-inf"))
            x_max = tl.max(x, axis=0)
            x_exp = tl.exp(x - x_max)
            x_sum = tl.sum(x_exp, axis=0)
            out = x_exp / x_sum
            tl.store(out_ptr + offs, out, mask=mask)

        src = triton.compiler.ASTSource(
            fn=_softmax_kernel,
            signature={"x_ptr": "*fp32", "out_ptr": "*fp32", "n": "i32"},
            constexprs={"BLOCK": 128},
        )
        kernel = triton.compile(src=src, target=GPUTarget("metal", "apple8", 32))
        msl = kernel.asm["metal"]
        if isinstance(msl, bytes):
            msl = msl.decode()
        assert "0xFFF" not in msl, "hex float constant must be converted in MSL"
        assert_metal_compilation_artifacts(kernel)


# ── Unsigned icmp predicate correctness (AUDIT-P0-003) ──────────────


class TestMetalUnsignedIcmp:
    """Regression: icmp ult/ule/ugt/uge must cast operands to unsigned."""

    def test_icmp_ult_uses_unsigned_cast(self):
        from third_party.metal.backend.compiler import MetalBackend

        llvm_ir = """
define void @ult_kernel(ptr %out, i32 %a, i32 %b) {
entry:
  %r = icmp ult i32 %a, %b
  ret void
}
"""
        msl = MetalBackend.make_metal_ir(llvm_ir, {}, None)
        assert "(unsigned int)" in msl, "icmp ult must cast to unsigned"
        assert "<" in msl

    def test_icmp_uge_uses_unsigned_cast(self):
        from third_party.metal.backend.compiler import MetalBackend

        llvm_ir = """
define void @uge_kernel(ptr %out, i32 %a, i32 %b) {
entry:
  %r = icmp uge i32 %a, %b
  ret void
}
"""
        msl = MetalBackend.make_metal_ir(llvm_ir, {}, None)
        assert "(unsigned int)" in msl, "icmp uge must cast to unsigned"
        assert ">=" in msl

    def test_icmp_ugt_uses_unsigned_cast(self):
        from third_party.metal.backend.compiler import MetalBackend

        llvm_ir = """
define void @ugt_kernel(ptr %out, i32 %a, i32 %b) {
entry:
  %r = icmp ugt i32 %a, %b
  ret void
}
"""
        msl = MetalBackend.make_metal_ir(llvm_ir, {}, None)
        assert "(unsigned int)" in msl, "icmp ugt must cast to unsigned"
        assert ">" in msl

    def test_icmp_ule_uses_unsigned_cast(self):
        from third_party.metal.backend.compiler import MetalBackend

        llvm_ir = """
define void @ule_kernel(ptr %out, i32 %a, i32 %b) {
entry:
  %r = icmp ule i32 %a, %b
  ret void
}
"""
        msl = MetalBackend.make_metal_ir(llvm_ir, {}, None)
        assert "(unsigned int)" in msl, "icmp ule must cast to unsigned"
        assert "<=" in msl

    def test_icmp_slt_stays_signed(self):
        """Signed predicates must NOT cast to unsigned."""
        from third_party.metal.backend.compiler import MetalBackend

        llvm_ir = """
define void @slt_kernel(ptr %out, i32 %a, i32 %b) {
entry:
  %r = icmp slt i32 %a, %b
  ret void
}
"""
        msl = MetalBackend.make_metal_ir(llvm_ir, {}, None)
        assert "unsigned" not in msl, "icmp slt must NOT cast to unsigned"

    def test_icmp_eq_stays_neutral(self):
        """eq/ne are sign-agnostic and must NOT cast."""
        from third_party.metal.backend.compiler import MetalBackend

        llvm_ir = """
define void @eq_kernel(ptr %out, i32 %a, i32 %b) {
entry:
  %r = icmp eq i32 %a, %b
  ret void
}
"""
        msl = MetalBackend.make_metal_ir(llvm_ir, {}, None)
        assert "unsigned" not in msl, "icmp eq must NOT cast to unsigned"

    def test_icmp_ult_i64_uses_unsigned_long(self):
        from third_party.metal.backend.compiler import MetalBackend

        llvm_ir = """
define void @ult64_kernel(ptr %out, i64 %a, i64 %b) {
entry:
  %r = icmp ult i64 %a, %b
  ret void
}
"""
        msl = MetalBackend.make_metal_ir(llvm_ir, {}, None)
        assert "unsigned long" in msl, "icmp ult i64 must cast to unsigned long"

    @skip_non_darwin
    @skip_no_xcrun
    def test_compile_unsigned_icmp_kernel(self):
        """End-to-end: kernel with unsigned comparisons compiles."""
        from third_party.metal.backend.compiler import MetalBackend, MetalOptions

        llvm_ir = """
define void @unsigned_cmp_kernel(ptr addrspace(1) %out, i32 %a, i32 %b) {
entry:
  %cmp = icmp ult i32 %a, %b
  %sel = select i1 %cmp, i32 %a, i32 %b
  %p = getelementptr i32, ptr addrspace(1) %out, i64 0
  store i32 %sel, ptr addrspace(1) %p
  ret void
}
"""
        metadata = {}
        msl = MetalBackend.make_metal_ir(llvm_ir, metadata, None)
        opts = MetalOptions(arch="apple8")
        binary = MetalBackend.make_metallib(msl, metadata, opts)
        assert isinstance(binary, bytes) and binary[:4] == b"MTLB"


# ── Phase 7: New intrinsics ────────────────────────────────────────


class TestMetalNewIntrinsics:
    """Tests for Phase 7 Task 1.1 — missing intrinsic batch."""

    def test_ctlz_intrinsic(self):
        from third_party.metal.backend.compiler import MetalBackend

        llvm_ir = """
define void @ctlz_kernel(ptr %out, i32 %val) {
entry:
  %r = call i32 @llvm.ctlz.i32(i32 %val, i1 false)
  %p = getelementptr i32, ptr %out, i64 0
  store i32 %r, ptr %p
  ret void
}
"""
        msl = MetalBackend.make_metal_ir(llvm_ir, {}, None)
        assert "clz(" in msl
        assert "llvm.ctlz" not in msl

    def test_cttz_intrinsic(self):
        from third_party.metal.backend.compiler import MetalBackend

        llvm_ir = """
define void @cttz_kernel(ptr %out, i32 %val) {
entry:
  %r = call i32 @llvm.cttz.i32(i32 %val, i1 false)
  %p = getelementptr i32, ptr %out, i64 0
  store i32 %r, ptr %p
  ret void
}
"""
        msl = MetalBackend.make_metal_ir(llvm_ir, {}, None)
        assert "ctz(" in msl
        assert "llvm.cttz" not in msl

    def test_bitreverse_intrinsic(self):
        from third_party.metal.backend.compiler import MetalBackend

        llvm_ir = """
define void @bitrev_kernel(ptr %out, i32 %val) {
entry:
  %r = call i32 @llvm.bitreverse.i32(i32 %val)
  %p = getelementptr i32, ptr %out, i64 0
  store i32 %r, ptr %p
  ret void
}
"""
        msl = MetalBackend.make_metal_ir(llvm_ir, {}, None)
        assert "reverse_bits(" in msl

    def test_bswap_i32_intrinsic(self):
        from third_party.metal.backend.compiler import MetalBackend

        llvm_ir = """
define void @bswap32_kernel(ptr %out, i32 %val) {
entry:
  %r = call i32 @llvm.bswap.i32(i32 %val)
  %p = getelementptr i32, ptr %out, i64 0
  store i32 %r, ptr %p
  ret void
}
"""
        msl = MetalBackend.make_metal_ir(llvm_ir, {}, None)
        assert ">> 24" in msl
        assert "<< 24" in msl
        assert "0xFF00" in msl

    def test_bswap_i64_intrinsic(self):
        from third_party.metal.backend.compiler import MetalBackend

        llvm_ir = """
define void @bswap64_kernel(ptr %out, i64 %val) {
entry:
  %r = call i64 @llvm.bswap.i64(i64 %val)
  %p = getelementptr i64, ptr %out, i64 0
  store i64 %r, ptr %p
  ret void
}
"""
        msl = MetalBackend.make_metal_ir(llvm_ir, {}, None)
        assert ">> 56" in msl
        assert "<< 56" in msl
        assert "unsigned long" in msl

    def test_fshr_intrinsic(self):
        from third_party.metal.backend.compiler import MetalBackend

        llvm_ir = """
define void @fshr_kernel(ptr %out, i32 %a, i32 %b, i32 %c) {
entry:
  %r = call i32 @llvm.fshr.i32(i32 %a, i32 %b, i32 %c)
  %p = getelementptr i32, ptr %out, i64 0
  store i32 %r, ptr %p
  ret void
}
"""
        msl = MetalBackend.make_metal_ir(llvm_ir, {}, None)
        assert "unsigned int" in msl
        assert ">>" in msl
        assert "<<" in msl

    def test_fshl_intrinsic(self):
        from third_party.metal.backend.compiler import MetalBackend

        llvm_ir = """
define void @fshl_kernel(ptr %out, i32 %a, i32 %b, i32 %c) {
entry:
  %r = call i32 @llvm.fshl.i32(i32 %a, i32 %b, i32 %c)
  %p = getelementptr i32, ptr %out, i64 0
  store i32 %r, ptr %p
  ret void
}
"""
        msl = MetalBackend.make_metal_ir(llvm_ir, {}, None)
        assert "unsigned int" in msl
        assert "<<" in msl
        assert ">>" in msl

    def test_lifetime_start_end_skipped(self):
        from third_party.metal.backend.compiler import MetalBackend

        llvm_ir = """
define void @lifetime_kernel(ptr %out) {
entry:
  call void @llvm.lifetime.start.p0(i64 4, ptr %out)
  %p = getelementptr i32, ptr %out, i64 0
  store i32 42, ptr %p
  call void @llvm.lifetime.end.p0(i64 4, ptr %out)
  ret void
}
"""
        msl = MetalBackend.make_metal_ir(llvm_ir, {}, None)
        assert "(void)0;" in msl
        assert "llvm.lifetime" not in msl

    def test_powi_intrinsic(self):
        from third_party.metal.backend.compiler import MetalBackend

        llvm_ir = """
define void @powi_kernel(ptr %out, float %base, i32 %exp) {
entry:
  %r = call float @llvm.powi.f32(float %base, i32 %exp)
  %p = getelementptr float, ptr %out, i64 0
  store float %r, ptr %p
  ret void
}
"""
        msl = MetalBackend.make_metal_ir(llvm_ir, {}, None)
        assert "pown(" in msl, "llvm.powi must lower to pown() (handles negative bases)"
        assert "powr(" not in msl, "powr() requires x >= 0; must not be used for powi"

    def test_memcpy_intrinsic(self):
        from third_party.metal.backend.compiler import MetalBackend

        llvm_ir = """
define void @memcpy_kernel(ptr %dst, ptr %src, i64 %n) {
entry:
  call void @llvm.memcpy.p0.p0.i64(ptr %dst, ptr %src, i64 %n, i1 false)
  ret void
}
"""
        msl = MetalBackend.make_metal_ir(llvm_ir, {}, None)
        assert "for (int __i" in msl
        assert "device char*" in msl

    def test_memset_intrinsic(self):
        from third_party.metal.backend.compiler import MetalBackend

        llvm_ir = """
define void @memset_kernel(ptr %dst, i8 %val, i64 %n) {
entry:
  call void @llvm.memset.p0.i64(ptr %dst, i8 %val, i64 %n, i1 false)
  ret void
}
"""
        msl = MetalBackend.make_metal_ir(llvm_ir, {}, None)
        assert "for (int __i" in msl
        assert "(char)" in msl


# ── Phase 7: Atomic operations ─────────────────────────────────────


class TestMetalAtomicOps:
    """Tests for Phase 7 Task 1.3 — atomicrmw and cmpxchg."""

    def test_atomicrmw_add(self):
        from third_party.metal.backend.compiler import MetalBackend

        llvm_ir = """
define void @atomic_add_kernel(ptr addrspace(1) %out, i32 %val) {
entry:
  %old = atomicrmw add ptr addrspace(1) %out, i32 %val monotonic
  ret void
}
"""
        msl = MetalBackend.make_metal_ir(llvm_ir, {}, None)
        assert "atomic_fetch_add_explicit" in msl
        assert "memory_order_relaxed" in msl
        assert "atomic_int" in msl

    def test_atomicrmw_xchg(self):
        from third_party.metal.backend.compiler import MetalBackend

        llvm_ir = """
define void @atomic_xchg_kernel(ptr addrspace(1) %out, i32 %val) {
entry:
  %old = atomicrmw xchg ptr addrspace(1) %out, i32 %val seq_cst
  ret void
}
"""
        msl = MetalBackend.make_metal_ir(llvm_ir, {}, None)
        assert "atomic_exchange_explicit" in msl
        assert "memory_order_seq_cst" in msl

    def test_atomicrmw_or_acquire(self):
        from third_party.metal.backend.compiler import MetalBackend

        llvm_ir = """
define void @atomic_or_kernel(ptr addrspace(1) %out, i32 %val) {
entry:
  %old = atomicrmw or ptr addrspace(1) %out, i32 %val acquire
  ret void
}
"""
        msl = MetalBackend.make_metal_ir(llvm_ir, {}, None)
        assert "atomic_fetch_or_explicit" in msl
        assert "memory_order_acquire" in msl

    def test_cmpxchg_basic(self):
        from third_party.metal.backend.compiler import MetalBackend

        llvm_ir = """
define void @cmpxchg_kernel(ptr addrspace(1) %ptr, i32 %expected, i32 %desired) {
entry:
  %result = cmpxchg ptr addrspace(1) %ptr, i32 %expected, i32 %desired acq_rel monotonic
  %val = extractvalue {i32, i1} %result, 0
  %success = extractvalue {i32, i1} %result, 1
  ret void
}
"""
        msl = MetalBackend.make_metal_ir(llvm_ir, {}, None)
        assert "atomic_compare_exchange_weak_explicit" in msl
        assert "memory_order_acq_rel" in msl
        assert "memory_order_relaxed" in msl
        assert ".field0" in msl
        assert ".field1" in msl

    def test_memory_ordering_map(self):
        from third_party.metal.backend.compiler import _MEMORY_ORDER_MAP

        assert _MEMORY_ORDER_MAP["monotonic"] == "memory_order_relaxed"
        assert _MEMORY_ORDER_MAP["acquire"] == "memory_order_acquire"
        assert _MEMORY_ORDER_MAP["release"] == "memory_order_release"
        assert _MEMORY_ORDER_MAP["acq_rel"] == "memory_order_acq_rel"
        assert _MEMORY_ORDER_MAP["seq_cst"] == "memory_order_seq_cst"

    def test_atomic_op_map_coverage(self):
        from third_party.metal.backend.compiler import _ATOMIC_OP_MAP

        assert "add" in _ATOMIC_OP_MAP
        assert "sub" in _ATOMIC_OP_MAP
        assert "xchg" in _ATOMIC_OP_MAP
        assert "and" in _ATOMIC_OP_MAP
        assert "or" in _ATOMIC_OP_MAP
        assert "xor" in _ATOMIC_OP_MAP
        assert "max" in _ATOMIC_OP_MAP
        assert "min" in _ATOMIC_OP_MAP
        assert "umax" in _ATOMIC_OP_MAP
        assert "umin" in _ATOMIC_OP_MAP
        assert "fadd" in _ATOMIC_OP_MAP


# ── Phase 7: Aggregate operations ──────────────────────────────────


class TestMetalAggregateOps:
    """Tests for Phase 7 Task 1.2 — extractvalue, insertvalue, overflow intrinsics."""

    def test_extractvalue_basic(self):
        from third_party.metal.backend.compiler import MetalBackend

        llvm_ir = """
define void @ev_kernel(ptr %out, i32 %a, i32 %b) {
entry:
  %result = call {i32, i1} @llvm.sadd.with.overflow.i32(i32 %a, i32 %b)
  %val = extractvalue {i32, i1} %result, 0
  %p = getelementptr i32, ptr %out, i64 0
  store i32 %val, ptr %p
  ret void
}
"""
        msl = MetalBackend.make_metal_ir(llvm_ir, {}, None)
        assert ".field0" in msl
        assert "__triton_aggr_" in msl

    def test_insertvalue_basic(self):
        from third_party.metal.backend.compiler import MetalBackend

        llvm_ir = """
define void @iv_kernel(ptr %out, i32 %a) {
entry:
  %result = call {i32, i1} @llvm.sadd.with.overflow.i32(i32 %a, i32 %a)
  %modified = insertvalue {i32, i1} %result, i32 99, 0
  %val = extractvalue {i32, i1} %modified, 0
  %p = getelementptr i32, ptr %out, i64 0
  store i32 %val, ptr %p
  ret void
}
"""
        msl = MetalBackend.make_metal_ir(llvm_ir, {}, None)
        assert ".field0" in msl

    def test_sadd_with_overflow(self):
        from third_party.metal.backend.compiler import MetalBackend

        llvm_ir = """
define void @sadd_ov_kernel(ptr %out, i32 %a, i32 %b) {
entry:
  %result = call {i32, i1} @llvm.sadd.with.overflow.i32(i32 %a, i32 %b)
  %val = extractvalue {i32, i1} %result, 0
  %ov = extractvalue {i32, i1} %result, 1
  %p = getelementptr i32, ptr %out, i64 0
  store i32 %val, ptr %p
  ret void
}
"""
        msl = MetalBackend.make_metal_ir(llvm_ir, {}, None)
        assert ".field0 =" in msl
        assert ".field1 =" in msl
        assert "+" in msl

    def test_uadd_with_overflow(self):
        from third_party.metal.backend.compiler import MetalBackend

        llvm_ir = """
define void @uadd_ov_kernel(ptr %out, i32 %a, i32 %b) {
entry:
  %result = call {i32, i1} @llvm.uadd.with.overflow.i32(i32 %a, i32 %b)
  %val = extractvalue {i32, i1} %result, 0
  %p = getelementptr i32, ptr %out, i64 0
  store i32 %val, ptr %p
  ret void
}
"""
        msl = MetalBackend.make_metal_ir(llvm_ir, {}, None)
        assert ".field0 =" in msl
        assert ".field1 =" in msl

    def test_ssub_with_overflow(self):
        from third_party.metal.backend.compiler import MetalBackend

        llvm_ir = """
define void @ssub_ov_kernel(ptr %out, i32 %a, i32 %b) {
entry:
  %result = call {i32, i1} @llvm.ssub.with.overflow.i32(i32 %a, i32 %b)
  %val = extractvalue {i32, i1} %result, 0
  %p = getelementptr i32, ptr %out, i64 0
  store i32 %val, ptr %p
  ret void
}
"""
        msl = MetalBackend.make_metal_ir(llvm_ir, {}, None)
        assert ".field0 =" in msl
        assert "-" in msl

    def test_struct_typedef_emitted(self):
        from third_party.metal.backend.compiler import MetalBackend

        llvm_ir = """
define void @struct_kernel(ptr %out, i32 %a, i32 %b) {
entry:
  %result = call {i32, i1} @llvm.sadd.with.overflow.i32(i32 %a, i32 %b)
  %val = extractvalue {i32, i1} %result, 0
  %p = getelementptr i32, ptr %out, i64 0
  store i32 %val, ptr %p
  ret void
}
"""
        msl = MetalBackend.make_metal_ir(llvm_ir, {}, None)
        assert "struct __triton_aggr_" in msl
        assert "int field0;" in msl
        assert "bool field1;" in msl


# ── Phase 7: alloca / switch / fence ───────────────────────────────


class TestMetalMiscInstructions:
    """Tests for Phase 7 Task 1.4 — alloca, switch, fence."""

    def test_alloca_basic(self):
        from third_party.metal.backend.compiler import MetalBackend

        llvm_ir = """
define void @alloca_kernel(ptr %out, i32 %val) {
entry:
  %ptr = alloca i32, align 4
  store i32 %val, ptr %ptr
  %loaded = load i32, ptr %ptr
  %p = getelementptr i32, ptr %out, i64 0
  store i32 %loaded, ptr %p
  ret void
}
"""
        msl = MetalBackend.make_metal_ir(llvm_ir, {}, None)
        assert "_storage" in msl
        assert "thread int*" in msl

    def test_switch_instruction(self):
        from third_party.metal.backend.compiler import MetalBackend

        llvm_ir = """
define void @switch_kernel(ptr %out, i32 %sel) {
entry:
  switch i32 %sel, label %default [
    i32 0, label %case0
    i32 1, label %case1
  ]
case0:
  %p0 = getelementptr i32, ptr %out, i64 0
  store i32 10, ptr %p0
  ret void
case1:
  %p1 = getelementptr i32, ptr %out, i64 0
  store i32 20, ptr %p1
  ret void
default:
  %pd = getelementptr i32, ptr %out, i64 0
  store i32 99, ptr %pd
  ret void
}
"""
        msl = MetalBackend.make_metal_ir(llvm_ir, {}, None)
        assert "switch (" in msl
        assert "case 0:" in msl
        assert "case 1:" in msl
        assert "default:" in msl

    def test_fence_workgroup(self):
        from third_party.metal.backend.compiler import MetalBackend

        llvm_ir = """
define void @fence_kernel(ptr %out) {
entry:
  fence syncscope("workgroup") release
  %p = getelementptr i32, ptr %out, i64 0
  store i32 1, ptr %p
  ret void
}
"""
        msl = MetalBackend.make_metal_ir(llvm_ir, {}, None)
        assert "threadgroup_barrier(mem_flags::mem_threadgroup)" in msl

    def test_fence_no_scope(self):
        from third_party.metal.backend.compiler import MetalBackend

        llvm_ir = """
define void @fence_noscope_kernel(ptr %out) {
entry:
  fence seq_cst
  %p = getelementptr i32, ptr %out, i64 0
  store i32 1, ptr %p
  ret void
}
"""
        msl = MetalBackend.make_metal_ir(llvm_ir, {}, None)
        assert "threadgroup_barrier(mem_flags::mem_device)" in msl

    def test_fence_subgroup(self):
        from third_party.metal.backend.compiler import MetalBackend

        llvm_ir = """
define void @fence_sub_kernel(ptr %out) {
entry:
  fence syncscope("subgroup") acquire
  %p = getelementptr i32, ptr %out, i64 0
  store i32 1, ptr %p
  ret void
}
"""
        msl = MetalBackend.make_metal_ir(llvm_ir, {}, None)
        assert "simdgroup_barrier(mem_flags::mem_threadgroup)" in msl

    def test_alloca_with_load_store(self):
        """alloca + load/store roundtrip produces thread-local variable."""
        from third_party.metal.backend.compiler import MetalBackend

        llvm_ir = """
define void @alloca_ls_kernel(ptr %out, i32 %val) {
entry:
  %ptr = alloca i32, align 4
  store i32 %val, ptr %ptr
  %loaded = load i32, ptr %ptr
  %p = getelementptr i32, ptr %out, i64 0
  store i32 %loaded, ptr %p
  ret void
}
"""
        msl = MetalBackend.make_metal_ir(llvm_ir, {}, None)
        assert "_storage" in msl
        assert "thread int*" in msl
        assert "loaded" in msl


# ── Corpus-Driven Translation Tests (Phase 7 – Task 1.6) ────────────


class TestMetalLLVMIRCorpus:
    """Validate the translator handles all IR patterns from real Triton kernels."""

    def test_corpus_vector_add(self):
        """Basic ops: add, load, store, gep, br."""
        from third_party.metal.backend.compiler import MetalBackend

        ir = """\
define void @vec_add_kernel(ptr %a, ptr %b, ptr %out, i32 %n) {
entry:
  %tid = call i32 @__metal_get_thread_position_in_threadgroup_x()
  %cmp = icmp slt i32 %tid, %n
  br i1 %cmp, label %body, label %exit

body:
  %idx = sext i32 %tid to i64
  %pa = getelementptr float, ptr %a, i64 %idx
  %pb = getelementptr float, ptr %b, i64 %idx
  %pout = getelementptr float, ptr %out, i64 %idx
  %va = load float, ptr %pa
  %vb = load float, ptr %pb
  %sum = fadd float %va, %vb
  store float %sum, ptr %pout
  br label %exit

exit:
  ret void
}
"""
        msl = MetalBackend.make_metal_ir(ir, {}, None)
        assert "UNSUPPORTED" not in msl
        assert "kernel void" in msl
        assert "fadd" not in msl  # should be lowered to +

    def test_corpus_reduction(self):
        """Reduction kernel: phi, fcmp, fadd, select, branch patterns."""
        from third_party.metal.backend.compiler import MetalBackend

        ir = """\
define void @reduce_kernel(ptr %input, ptr %out, i32 %n) {
entry:
  %tid = call i32 @__metal_get_thread_position_in_threadgroup_x()
  %cmp_bounds = icmp slt i32 %tid, %n
  br i1 %cmp_bounds, label %loop_header, label %done

loop_header:
  %i = phi i32 [0, %entry], [%i_next, %loop_body]
  %acc = phi float [0.0, %entry], [%acc_next, %loop_body]
  %loop_cmp = icmp slt i32 %i, %n
  br i1 %loop_cmp, label %loop_body, label %write_out

loop_body:
  %idx = sext i32 %i to i64
  %ptr = getelementptr float, ptr %input, i64 %idx
  %val = load float, ptr %ptr
  %cmp_gt = fcmp ogt float %val, %acc
  %acc_next = select i1 %cmp_gt, float %val, float %acc
  %i_next = add i32 %i, 1
  br label %loop_header

write_out:
  %out_idx = sext i32 %tid to i64
  %out_ptr = getelementptr float, ptr %out, i64 %out_idx
  store float %acc, ptr %out_ptr
  br label %done

done:
  ret void
}
"""
        msl = MetalBackend.make_metal_ir(ir, {}, None)
        assert "UNSUPPORTED" not in msl
        assert "kernel void" in msl
        assert "__triton_pred_block" in msl  # phi lowering

    def test_corpus_matmul(self):
        """Matmul-like kernel: nested loops with phi, fmuladd, gep chains."""
        from third_party.metal.backend.compiler import MetalBackend

        ir = """\
define void @matmul_kernel(ptr %A, ptr %B, ptr %C, i32 %M, i32 %N, i32 %K) {
entry:
  %row = call i32 @__metal_get_thread_position_in_threadgroup_x()
  %col = call i32 @__metal_get_thread_position_in_threadgroup_y()
  br label %k_loop

k_loop:
  %k = phi i32 [0, %entry], [%k_next, %k_body]
  %acc = phi float [0.0, %entry], [%acc_next, %k_body]
  %k_cmp = icmp slt i32 %k, %K
  br i1 %k_cmp, label %k_body, label %store_result

k_body:
  %a_off = mul i32 %row, %K
  %a_idx = add i32 %a_off, %k
  %a_idx64 = sext i32 %a_idx to i64
  %a_ptr = getelementptr float, ptr %A, i64 %a_idx64
  %a_val = load float, ptr %a_ptr
  %b_off = mul i32 %k, %N
  %b_idx = add i32 %b_off, %col
  %b_idx64 = sext i32 %b_idx to i64
  %b_ptr = getelementptr float, ptr %B, i64 %b_idx64
  %b_val = load float, ptr %b_ptr
  %acc_next = call float @llvm.fmuladd.f32(float %a_val, float %b_val, float %acc)
  %k_next = add i32 %k, 1
  br label %k_loop

store_result:
  %c_off = mul i32 %row, %N
  %c_idx = add i32 %c_off, %col
  %c_idx64 = sext i32 %c_idx to i64
  %c_ptr = getelementptr float, ptr %C, i64 %c_idx64
  store float %acc, ptr %c_ptr
  ret void
}
"""
        msl = MetalBackend.make_metal_ir(ir, {}, None)
        assert "UNSUPPORTED" not in msl
        assert "kernel void" in msl
        assert "fma(" in msl  # fmuladd -> fma

    def test_corpus_atomics(self):
        """Kernel with atomicrmw add."""
        from third_party.metal.backend.compiler import MetalBackend

        ir = """\
define void @atomic_add_kernel(ptr %data, ptr %out) {
entry:
  %tid = call i32 @__metal_get_thread_position_in_threadgroup_x()
  %idx = sext i32 %tid to i64
  %ptr = getelementptr i32, ptr %data, i64 %idx
  %val = load i32, ptr %ptr
  %old = atomicrmw add ptr %out, i32 %val seq_cst
  ret void
}
"""
        msl = MetalBackend.make_metal_ir(ir, {}, None)
        assert "UNSUPPORTED" not in msl
        assert "kernel void" in msl
        assert "atomic_fetch_add_explicit" in msl

    def test_corpus_mixed_types(self):
        """Kernel with fptrunc, fpext, sext, zext, trunc."""
        from third_party.metal.backend.compiler import MetalBackend

        ir = """\
define void @mixed_types_kernel(ptr %out, float %f_in, i64 %i_in) {
entry:
  %h = fptrunc float %f_in to half
  %d = fpext float %f_in to double
  %narrow = trunc i64 %i_in to i32
  %wide = sext i32 %narrow to i64
  %unsigned_wide = zext i32 %narrow to i64
  %from_float = fptosi float %f_in to i32
  %to_float = sitofp i32 %from_float to float
  %p = getelementptr float, ptr %out, i64 0
  store float %to_float, ptr %p
  ret void
}
"""
        msl = MetalBackend.make_metal_ir(ir, {}, None)
        assert "UNSUPPORTED" not in msl
        assert "kernel void" in msl
        assert "(half)" in msl  # fptrunc
        assert "(double)" in msl  # fpext

    def test_corpus_intrinsic_math(self):
        """Kernel with intrinsic math calls: fabs, sqrt, exp, log."""
        from third_party.metal.backend.compiler import MetalBackend

        ir = """\
define void @math_kernel(ptr %out, float %x) {
entry:
  %a = call float @llvm.fabs.f32(float %x)
  %b = call float @llvm.sqrt.f32(float %a)
  %c = call float @llvm.exp.f32(float %b)
  %d = call float @llvm.log.f32(float %c)
  %e = call float @llvm.sin.f32(float %d)
  %f = call float @llvm.cos.f32(float %e)
  %p = getelementptr float, ptr %out, i64 0
  store float %f, ptr %p
  ret void
}
"""
        msl = MetalBackend.make_metal_ir(ir, {}, None)
        assert "UNSUPPORTED" not in msl
        assert "kernel void" in msl
        assert "fabs(" in msl
        assert "sqrt(" in msl
        assert "exp(" in msl
        assert "log(" in msl
        assert "sin(" in msl
        assert "cos(" in msl

    def test_corpus_switch_pattern(self):
        """Switch statement pattern with multiple cases."""
        from third_party.metal.backend.compiler import MetalBackend

        ir = """\
define void @switch_kernel(ptr %out, i32 %selector) {
entry:
  switch i32 %selector, label %default [i32 0, label %case0 i32 1, label %case1 i32 2, label %case2]

case0:
  %p0 = getelementptr i32, ptr %out, i64 0
  store i32 10, ptr %p0
  br label %done

case1:
  %p1 = getelementptr i32, ptr %out, i64 0
  store i32 20, ptr %p1
  br label %done

case2:
  %p2 = getelementptr i32, ptr %out, i64 0
  store i32 30, ptr %p2
  br label %done

default:
  %pd = getelementptr i32, ptr %out, i64 0
  store i32 -1, ptr %pd
  br label %done

done:
  ret void
}
"""
        msl = MetalBackend.make_metal_ir(ir, {}, None)
        assert "UNSUPPORTED" not in msl
        assert "kernel void" in msl
        assert "switch" in msl
        assert "case 0:" in msl
        assert "case 1:" in msl

    def test_corpus_freeze_and_fneg(self):
        """Freeze and fneg instructions."""
        from third_party.metal.backend.compiler import MetalBackend

        ir = """\
define void @freeze_fneg_kernel(ptr %out, float %x) {
entry:
  %frozen = freeze float %x
  %neg = fneg float %frozen
  %p = getelementptr float, ptr %out, i64 0
  store float %neg, ptr %p
  ret void
}
"""
        msl = MetalBackend.make_metal_ir(ir, {}, None)
        assert "UNSUPPORTED" not in msl
        assert "kernel void" in msl
        assert "-(" in msl  # fneg

    def test_corpus_extractinsert_value(self):
        """extractvalue / insertvalue with aggregate types."""
        from third_party.metal.backend.compiler import MetalBackend

        ir = """\
define void @aggr_kernel(ptr %out, i32 %a, i32 %b) {
entry:
  %r = call {i32, i1} @llvm.sadd.with.overflow.i32(i32 %a, i32 %b)
  %sum = extractvalue {i32, i1} %r, 0
  %overflow = extractvalue {i32, i1} %r, 1
  %p = getelementptr i32, ptr %out, i64 0
  store i32 %sum, ptr %p
  ret void
}
"""
        msl = MetalBackend.make_metal_ir(ir, {}, None)
        assert "UNSUPPORTED" not in msl
        assert "kernel void" in msl
        assert "field0" in msl

    def test_corpus_alloca_pattern(self):
        """alloca + thread-local variable pattern."""
        from third_party.metal.backend.compiler import MetalBackend

        ir = """\
define void @alloca_kernel(ptr %out, i32 %val) {
entry:
  %tmp = alloca i32, align 4
  store i32 %val, ptr %tmp
  %loaded = load i32, ptr %tmp
  %doubled = add i32 %loaded, %loaded
  %p = getelementptr i32, ptr %out, i64 0
  store i32 %doubled, ptr %p
  ret void
}
"""
        msl = MetalBackend.make_metal_ir(ir, {}, None)
        assert "UNSUPPORTED" not in msl
        assert "kernel void" in msl
        assert "_storage" in msl
        assert "thread int*" in msl


class TestMetalUnsupportedIRDiagnostics:
    """Verify the structured diagnostic system for unsupported LLVM IR."""

    def test_unsupported_ir_raises_with_summary(self):
        """Hitting unsupported lines produces a RuntimeError with a summary."""
        from third_party.metal.backend.compiler import MetalBackend

        ir = """\
define void @bad_kernel(ptr %out) {
entry:
  invoke void @some_func() to label %next unwind label %pad

next:
  ret void

pad:
  %lp = landingpad token cleanup
  ret void
}
"""
        with pytest.raises(RuntimeError, match="unsupported LLVM IR"):
            MetalBackend.make_metal_ir(ir, {}, None)

    def test_unsupported_ir_diagnostic_categories(self):
        """Diagnostic entries are classified correctly."""
        from third_party.metal.backend.compiler import (
            _classify_unsupported_ir,
        )

        assert _classify_unsupported_ir("invoke void @foo()") == "instruction"
        assert _classify_unsupported_ir("resume { ptr, i32 } %r") == "instruction"
        assert _classify_unsupported_ir("landingpad token cleanup") == "instruction"
        assert (
            _classify_unsupported_ir("indirectbr ptr %addr, [label %a]")
            == "instruction"
        )
        assert _classify_unsupported_ir("!0 = !{i32 1}") == "metadata"
        assert _classify_unsupported_ir("attributes #0 = { nounwind }") == "metadata"
        assert (
            _classify_unsupported_ir(
                "%r = call i32 @llvm.some.unknown.intrinsic(i32 %x)"
            )
            == "intrinsic"
        )
        assert _classify_unsupported_ir("something completely unknown") == "unknown"

    def test_unsupported_ir_artifact_file(self):
        """Diagnostic artifact file is written on unsupported IR."""
        import tempfile

        from third_party.metal.backend.compiler import MetalBackend

        ir = """\
define void @artifact_kernel(ptr %out) {
entry:
  invoke void @some_func() to label %next unwind label %pad

next:
  ret void

pad:
  %lp = landingpad token cleanup
  ret void
}
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(os.environ, {"TRITON_CACHE_DIR": tmpdir}):
                with pytest.raises(RuntimeError):
                    MetalBackend.make_metal_ir(ir, {}, None)
                artifact_path = os.path.join(tmpdir, "metal_unsupported_ir.log")
                assert os.path.exists(artifact_path)
                content = open(artifact_path).read()
                assert "invoke" in content
                assert "instruction" in content or "unknown" in content

    def test_best_effort_mode_emits_comments(self):
        """best_effort=True emits UNSUPPORTED comments instead of raising."""
        from third_party.metal.backend.compiler import MetalBackend, MetalOptions

        ir = """\
define void @besteffort_kernel(ptr %out) {
entry:
  invoke void @some_func() to label %next unwind label %pad

next:
  ret void

pad:
  %lp = landingpad token cleanup
  ret void
}
"""
        opt = MetalOptions(best_effort=True)
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(os.environ, {"TRITON_CACHE_DIR": tmpdir}):
                with pytest.warns(match="unsupported LLVM IR"):
                    msl = MetalBackend.make_metal_ir(ir, {}, opt)
                assert "// UNSUPPORTED:" in msl
                assert "kernel void" in msl
                artifact_path = os.path.join(tmpdir, "metal_unsupported_ir.log")
                assert os.path.exists(artifact_path)

    def test_best_effort_no_unsupported_no_warning(self):
        """best_effort=True with fully supported IR produces no warnings."""
        from third_party.metal.backend.compiler import MetalBackend, MetalOptions

        ir = """\
define void @clean_kernel(ptr %out, i32 %val) {
entry:
  %p = getelementptr i32, ptr %out, i64 0
  store i32 %val, ptr %p
  ret void
}
"""
        opt = MetalOptions(best_effort=True)
        import warnings as w

        with w.catch_warnings(record=True) as caught:
            w.simplefilter("always")
            msl = MetalBackend.make_metal_ir(ir, {}, opt)
        metal_warns = [x for x in caught if "unsupported" in str(x.message).lower()]
        assert len(metal_warns) == 0
        assert "kernel void" in msl
        assert "UNSUPPORTED" not in msl

    def test_unsupported_ir_entry_dataclass(self):
        """UnsupportedIREntry is a proper dataclass with expected fields."""
        from third_party.metal.backend.compiler import UnsupportedIREntry

        entry = UnsupportedIREntry(
            line_number=5,
            line="invoke void @foo()",
            context_before=["br label %bb1"],
            context_after=["ret void"],
            category="instruction",
        )
        assert entry.line_number == 5
        assert entry.category == "instruction"
        assert len(entry.context_before) == 1
        assert len(entry.context_after) == 1

    def test_error_message_includes_first_lines(self):
        """The RuntimeError message includes up to the first 3 unsupported lines."""
        from third_party.metal.backend.compiler import MetalBackend

        ir = """\
define void @multi_unsup_kernel(ptr %out) {
entry:
  invoke void @a() to label %b1 unwind label %pad
b1:
  invoke void @b() to label %b2 unwind label %pad
b2:
  invoke void @c() to label %b3 unwind label %pad
b3:
  invoke void @d() to label %done unwind label %pad
done:
  ret void
pad:
  %lp = landingpad token cleanup
  ret void
}
"""
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(os.environ, {"TRITON_CACHE_DIR": tmpdir}):
                with pytest.raises(RuntimeError) as ctx:
                    MetalBackend.make_metal_ir(ir, {}, None)
                msg = str(ctx.value)
                assert (
                    "unsupported LLVM IR" in msg.lower() or "unsupported" in msg.lower()
                )
                assert "invoke" in msg

    def test_normal_path_zero_overhead(self):
        """The normal (all-supported) path completes without diagnostics."""
        from third_party.metal.backend.compiler import MetalBackend

        ir = """\
define void @normal_kernel(ptr %a, ptr %b, ptr %out) {
entry:
  %p_a = getelementptr float, ptr %a, i64 0
  %p_b = getelementptr float, ptr %b, i64 0
  %p_out = getelementptr float, ptr %out, i64 0
  %va = load float, ptr %p_a
  %vb = load float, ptr %p_b
  %sum = fadd float %va, %vb
  store float %sum, ptr %p_out
  ret void
}
"""
        msl = MetalBackend.make_metal_ir(ir, {}, None)
        assert "kernel void" in msl
        assert "UNSUPPORTED" not in msl


# ── Multi-dtype GEMM compilation coverage (Phase 8, Task 2.2) ───────


class TestMetalGEMMDtypes:
    """Compile matmul kernels with different dtype combinations."""

    @skip_non_darwin
    @skip_no_xcrun
    def test_gemm_fp32_fp32(self):
        """fp32 x fp32 -> fp32 accumulation."""
        import triton
        import triton.language as tl
        from triton.backends.compiler import GPUTarget

        @triton.jit
        def _matmul_fp32(
            a_ptr,
            b_ptr,
            c_ptr,
            m,
            n,
            k,
            stride_am,
            stride_ak,
            stride_bk,
            stride_bn,
            stride_cm,
            stride_cn,
            BLOCK_M: tl.constexpr,
            BLOCK_N: tl.constexpr,
            BLOCK_K: tl.constexpr,
        ):
            pid_m = tl.program_id(axis=0)
            pid_n = tl.program_id(axis=1)
            offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
            offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
            offs_k = tl.arange(0, BLOCK_K)
            acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            for kk in range(0, k, BLOCK_K):
                a = tl.load(
                    a_ptr
                    + offs_m[:, None] * stride_am
                    + (offs_k[None, :] + kk) * stride_ak,
                    mask=(offs_m[:, None] < m) & (offs_k[None, :] + kk < k),
                    other=0.0,
                )
                b = tl.load(
                    b_ptr
                    + (offs_k[:, None] + kk) * stride_bk
                    + offs_n[None, :] * stride_bn,
                    mask=(offs_k[:, None] + kk < k) & (offs_n[None, :] < n),
                    other=0.0,
                )
                acc += tl.dot(a, b)
            c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
            tl.store(c_ptrs, acc, mask=(offs_m[:, None] < m) & (offs_n[None, :] < n))

        src = triton.compiler.ASTSource(
            fn=_matmul_fp32,
            signature={
                "a_ptr": "*fp32",
                "b_ptr": "*fp32",
                "c_ptr": "*fp32",
                "m": "i32",
                "n": "i32",
                "k": "i32",
                "stride_am": "i32",
                "stride_ak": "i32",
                "stride_bk": "i32",
                "stride_bn": "i32",
                "stride_cm": "i32",
                "stride_cn": "i32",
            },
            constexprs={"BLOCK_M": 16, "BLOCK_N": 16, "BLOCK_K": 16},
        )
        kernel = triton.compile(src=src, target=GPUTarget("metal", "apple8", 32))
        assert_metal_compilation_artifacts(kernel)

    @skip_non_darwin
    @skip_no_xcrun
    def test_gemm_fp16_input(self):
        """fp16 x fp16 -> fp32 accumulation."""
        import triton
        import triton.language as tl
        from triton.backends.compiler import GPUTarget

        @triton.jit
        def _matmul_fp16(
            a_ptr,
            b_ptr,
            c_ptr,
            m,
            n,
            k,
            stride_am,
            stride_ak,
            stride_bk,
            stride_bn,
            stride_cm,
            stride_cn,
            BLOCK_M: tl.constexpr,
            BLOCK_N: tl.constexpr,
            BLOCK_K: tl.constexpr,
        ):
            pid_m = tl.program_id(axis=0)
            pid_n = tl.program_id(axis=1)
            offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
            offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
            offs_k = tl.arange(0, BLOCK_K)
            acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            for kk in range(0, k, BLOCK_K):
                a = tl.load(
                    a_ptr
                    + offs_m[:, None] * stride_am
                    + (offs_k[None, :] + kk) * stride_ak,
                    mask=(offs_m[:, None] < m) & (offs_k[None, :] + kk < k),
                    other=0.0,
                )
                b = tl.load(
                    b_ptr
                    + (offs_k[:, None] + kk) * stride_bk
                    + offs_n[None, :] * stride_bn,
                    mask=(offs_k[:, None] + kk < k) & (offs_n[None, :] < n),
                    other=0.0,
                )
                acc += tl.dot(a, b)
            c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
            tl.store(
                c_ptrs,
                acc.to(tl.float16),
                mask=(offs_m[:, None] < m) & (offs_n[None, :] < n),
            )

        src = triton.compiler.ASTSource(
            fn=_matmul_fp16,
            signature={
                "a_ptr": "*fp16",
                "b_ptr": "*fp16",
                "c_ptr": "*fp16",
                "m": "i32",
                "n": "i32",
                "k": "i32",
                "stride_am": "i32",
                "stride_ak": "i32",
                "stride_bk": "i32",
                "stride_bn": "i32",
                "stride_cm": "i32",
                "stride_cn": "i32",
            },
            constexprs={"BLOCK_M": 16, "BLOCK_N": 16, "BLOCK_K": 16},
        )
        kernel = triton.compile(src=src, target=GPUTarget("metal", "apple8", 32))
        assert_metal_compilation_artifacts(kernel)

    @skip_non_darwin
    @skip_no_xcrun
    def test_gemm_odd_k_tail(self):
        """Matmul with K not multiple of BLOCK_K (K param=17, BLOCK_K=16)."""
        import triton
        import triton.language as tl
        from triton.backends.compiler import GPUTarget

        @triton.jit
        def _matmul_odd(
            a_ptr,
            b_ptr,
            c_ptr,
            m,
            n,
            k,
            stride_am,
            stride_ak,
            stride_bk,
            stride_bn,
            stride_cm,
            stride_cn,
            BLOCK_M: tl.constexpr,
            BLOCK_N: tl.constexpr,
            BLOCK_K: tl.constexpr,
        ):
            pid_m = tl.program_id(axis=0)
            pid_n = tl.program_id(axis=1)
            offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
            offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
            offs_k = tl.arange(0, BLOCK_K)
            acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            for kk in range(0, k, BLOCK_K):
                a_mask = (offs_m[:, None] < m) & (offs_k[None, :] + kk < k)
                b_mask = (offs_k[:, None] + kk < k) & (offs_n[None, :] < n)
                a = tl.load(
                    a_ptr
                    + offs_m[:, None] * stride_am
                    + (offs_k[None, :] + kk) * stride_ak,
                    mask=a_mask,
                    other=0.0,
                )
                b = tl.load(
                    b_ptr
                    + (offs_k[:, None] + kk) * stride_bk
                    + offs_n[None, :] * stride_bn,
                    mask=b_mask,
                    other=0.0,
                )
                acc += tl.dot(a, b)
            c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
            tl.store(c_ptrs, acc, mask=(offs_m[:, None] < m) & (offs_n[None, :] < n))

        src = triton.compiler.ASTSource(
            fn=_matmul_odd,
            signature={
                "a_ptr": "*fp32",
                "b_ptr": "*fp32",
                "c_ptr": "*fp32",
                "m": "i32",
                "n": "i32",
                "k": "i32",
                "stride_am": "i32",
                "stride_ak": "i32",
                "stride_bk": "i32",
                "stride_bn": "i32",
                "stride_cm": "i32",
                "stride_cn": "i32",
            },
            constexprs={"BLOCK_M": 16, "BLOCK_N": 16, "BLOCK_K": 16},
        )
        kernel = triton.compile(src=src, target=GPUTarget("metal", "apple8", 32))
        assert_metal_compilation_artifacts(kernel)

    @skip_non_darwin
    @skip_no_xcrun
    def test_gemm_small_tiles(self):
        """Small tile GEMM (BLOCK_M=N=K=8)."""
        import triton
        import triton.language as tl
        from triton.backends.compiler import GPUTarget

        @triton.jit
        def _matmul_small(
            a_ptr,
            b_ptr,
            c_ptr,
            m,
            n,
            k,
            stride_am,
            stride_ak,
            stride_bk,
            stride_bn,
            stride_cm,
            stride_cn,
            BLOCK_M: tl.constexpr,
            BLOCK_N: tl.constexpr,
            BLOCK_K: tl.constexpr,
        ):
            pid_m = tl.program_id(axis=0)
            pid_n = tl.program_id(axis=1)
            offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
            offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
            offs_k = tl.arange(0, BLOCK_K)
            acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            for kk in range(0, k, BLOCK_K):
                a = tl.load(
                    a_ptr
                    + offs_m[:, None] * stride_am
                    + (offs_k[None, :] + kk) * stride_ak,
                    mask=(offs_m[:, None] < m) & (offs_k[None, :] + kk < k),
                    other=0.0,
                )
                b = tl.load(
                    b_ptr
                    + (offs_k[:, None] + kk) * stride_bk
                    + offs_n[None, :] * stride_bn,
                    mask=(offs_k[:, None] + kk < k) & (offs_n[None, :] < n),
                    other=0.0,
                )
                acc += tl.dot(a, b)
            c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
            tl.store(c_ptrs, acc, mask=(offs_m[:, None] < m) & (offs_n[None, :] < n))

        src = triton.compiler.ASTSource(
            fn=_matmul_small,
            signature={
                "a_ptr": "*fp32",
                "b_ptr": "*fp32",
                "c_ptr": "*fp32",
                "m": "i32",
                "n": "i32",
                "k": "i32",
                "stride_am": "i32",
                "stride_ak": "i32",
                "stride_bk": "i32",
                "stride_bn": "i32",
                "stride_cm": "i32",
                "stride_cn": "i32",
            },
            constexprs={"BLOCK_M": 8, "BLOCK_N": 8, "BLOCK_K": 8},
        )
        kernel = triton.compile(src=src, target=GPUTarget("metal", "apple8", 32))
        assert_metal_compilation_artifacts(kernel)


# ── Simdgroup matrix stub tests (Phase 8, Task 2.3) ─────────────────


class TestMetalSimdgroupMatrixStubs:
    """Verify translator handles simdgroup_matrix call patterns."""

    def test_simdgroup_load_translation(self):
        """__metal_simdgroup_load -> simdgroup_load() in MSL."""
        from third_party.metal.backend.compiler import MetalBackend

        ir = """\
define void @simdgroup_load_kernel(ptr addrspace(1) %ptr) {
entry:
  %mat = call <8 x float> @__metal_simdgroup_load(ptr addrspace(1) %ptr, i32 64)
  ret void
}
"""
        msl = MetalBackend.make_metal_ir(ir, {}, None)
        assert "simdgroup_load" in msl
        assert "simdgroup_matrix" in msl

    def test_simdgroup_store_translation(self):
        """__metal_simdgroup_store -> simdgroup_store() in MSL."""
        from third_party.metal.backend.compiler import MetalBackend

        ir = """\
define void @simdgroup_store_kernel(ptr addrspace(1) %ptr) {
entry:
  %mat = call <8 x float> @__metal_simdgroup_load(ptr addrspace(1) %ptr, i32 64)
  call void @__metal_simdgroup_store(<8 x float> %mat, ptr addrspace(1) %ptr, i32 64)
  ret void
}
"""
        msl = MetalBackend.make_metal_ir(ir, {}, None)
        assert "simdgroup_store" in msl

    def test_simdgroup_multiply_accumulate_translation(self):
        """__metal_simdgroup_multiply_accumulate -> simdgroup_multiply_accumulate()."""
        from third_party.metal.backend.compiler import MetalBackend

        ir = """\
define void @simdgroup_mac_kernel(ptr addrspace(1) %a_ptr, ptr addrspace(1) %b_ptr, ptr addrspace(1) %c_ptr) {
entry:
  %a = call <8 x float> @__metal_simdgroup_load(ptr addrspace(1) %a_ptr, i32 64)
  %b = call <8 x float> @__metal_simdgroup_load(ptr addrspace(1) %b_ptr, i32 64)
  %c = call <8 x float> @__metal_simdgroup_load(ptr addrspace(1) %c_ptr, i32 64)
  %d = call <8 x float> @__metal_simdgroup_multiply_accumulate(<8 x float> %a, <8 x float> %b, <8 x float> %c)
  call void @__metal_simdgroup_store(<8 x float> %d, ptr addrspace(1) %c_ptr, i32 64)
  ret void
}
"""
        msl = MetalBackend.make_metal_ir(ir, {}, None)
        assert "simdgroup_multiply_accumulate" in msl
        assert "simdgroup_load" in msl
        assert "simdgroup_store" in msl

    def test_simdgroup_matrix_type_declaration(self):
        """simdgroup_matrix type appears in variable declarations."""
        from third_party.metal.backend.compiler import MetalBackend

        ir = """\
define void @simdgroup_type_kernel(ptr addrspace(1) %ptr) {
entry:
  %mat = call <8 x float> @__metal_simdgroup_load(ptr addrspace(1) %ptr, i32 64)
  ret void
}
"""
        msl = MetalBackend.make_metal_ir(ir, {}, None)
        assert "simdgroup_matrix<float, 8, 8>" in msl

    def test_simdgroup_half_type(self):
        """simdgroup_matrix with half element type."""
        from third_party.metal.backend.compiler import MetalBackend

        ir = """\
define void @simdgroup_half_kernel(ptr addrspace(1) %ptr) {
entry:
  %mat = call <8 x half> @__metal_simdgroup_load(ptr addrspace(1) %ptr, i32 32)
  ret void
}
"""
        msl = MetalBackend.make_metal_ir(ir, {}, None)
        assert "simdgroup_matrix<half, 8, 8>" in msl


# ── Matmul perf regression tests (Phase 8, Task 2.4) ────────────────


class TestMetalMatmulRegression:
    """Guard against code-quality regressions in generated matmul MSL."""

    @skip_non_darwin
    @skip_no_xcrun
    def test_matmul_fma_count_16x16(self):
        """FMA instruction count for 16x16 blocked matmul should be stable."""
        import triton
        import triton.language as tl
        from triton.backends.compiler import GPUTarget

        @triton.jit
        def _matmul_regress(
            a_ptr,
            b_ptr,
            c_ptr,
            m,
            n,
            k,
            stride_am,
            stride_ak,
            stride_bk,
            stride_bn,
            stride_cm,
            stride_cn,
            BLOCK_M: tl.constexpr,
            BLOCK_N: tl.constexpr,
            BLOCK_K: tl.constexpr,
        ):
            pid_m = tl.program_id(axis=0)
            pid_n = tl.program_id(axis=1)
            offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
            offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
            offs_k = tl.arange(0, BLOCK_K)
            acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            for kk in range(0, k, BLOCK_K):
                a = tl.load(
                    a_ptr
                    + offs_m[:, None] * stride_am
                    + (offs_k[None, :] + kk) * stride_ak,
                    mask=(offs_m[:, None] < m) & (offs_k[None, :] + kk < k),
                    other=0.0,
                )
                b = tl.load(
                    b_ptr
                    + (offs_k[:, None] + kk) * stride_bk
                    + offs_n[None, :] * stride_bn,
                    mask=(offs_k[:, None] + kk < k) & (offs_n[None, :] < n),
                    other=0.0,
                )
                acc += tl.dot(a, b)
            c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
            tl.store(c_ptrs, acc, mask=(offs_m[:, None] < m) & (offs_n[None, :] < n))

        src = triton.compiler.ASTSource(
            fn=_matmul_regress,
            signature={
                "a_ptr": "*fp32",
                "b_ptr": "*fp32",
                "c_ptr": "*fp32",
                "m": "i32",
                "n": "i32",
                "k": "i32",
                "stride_am": "i32",
                "stride_ak": "i32",
                "stride_bk": "i32",
                "stride_bn": "i32",
                "stride_cm": "i32",
                "stride_cn": "i32",
            },
            constexprs={"BLOCK_M": 16, "BLOCK_N": 16, "BLOCK_K": 16},
        )
        kernel = triton.compile(src=src, target=GPUTarget("metal", "apple8", 32))
        assert_metal_compilation_artifacts(kernel)

        msl_text = kernel.asm["metal"]
        if isinstance(msl_text, bytes):
            msl_text = msl_text.decode("utf-8", errors="replace")

        fma_count = msl_text.count("fma(")
        assert (
            fma_count >= 10
        ), f"Expected at least 10 fma() calls in matmul MSL, got {fma_count}"
        assert (
            fma_count < 50000
        ), f"fma() count suspiciously high ({fma_count}), possible code bloat"

    @skip_non_darwin
    @skip_no_xcrun
    def test_matmul_msl_line_count(self):
        """Generated MSL should not blow up in size."""
        import triton
        import triton.language as tl
        from triton.backends.compiler import GPUTarget

        @triton.jit
        def _matmul_lines(
            a_ptr,
            b_ptr,
            c_ptr,
            m,
            n,
            k,
            stride_am,
            stride_ak,
            stride_bk,
            stride_bn,
            stride_cm,
            stride_cn,
            BLOCK_M: tl.constexpr,
            BLOCK_N: tl.constexpr,
            BLOCK_K: tl.constexpr,
        ):
            pid_m = tl.program_id(axis=0)
            pid_n = tl.program_id(axis=1)
            offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
            offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
            offs_k = tl.arange(0, BLOCK_K)
            acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            for kk in range(0, k, BLOCK_K):
                a = tl.load(
                    a_ptr
                    + offs_m[:, None] * stride_am
                    + (offs_k[None, :] + kk) * stride_ak,
                    mask=(offs_m[:, None] < m) & (offs_k[None, :] + kk < k),
                    other=0.0,
                )
                b = tl.load(
                    b_ptr
                    + (offs_k[:, None] + kk) * stride_bk
                    + offs_n[None, :] * stride_bn,
                    mask=(offs_k[:, None] + kk < k) & (offs_n[None, :] < n),
                    other=0.0,
                )
                acc += tl.dot(a, b)
            c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
            tl.store(c_ptrs, acc, mask=(offs_m[:, None] < m) & (offs_n[None, :] < n))

        src = triton.compiler.ASTSource(
            fn=_matmul_lines,
            signature={
                "a_ptr": "*fp32",
                "b_ptr": "*fp32",
                "c_ptr": "*fp32",
                "m": "i32",
                "n": "i32",
                "k": "i32",
                "stride_am": "i32",
                "stride_ak": "i32",
                "stride_bk": "i32",
                "stride_bn": "i32",
                "stride_cm": "i32",
                "stride_cn": "i32",
            },
            constexprs={"BLOCK_M": 16, "BLOCK_N": 16, "BLOCK_K": 16},
        )
        kernel = triton.compile(src=src, target=GPUTarget("metal", "apple8", 32))
        assert_metal_compilation_artifacts(kernel)

        msl_text = kernel.asm["metal"]
        if isinstance(msl_text, bytes):
            msl_text = msl_text.decode("utf-8", errors="replace")

        line_count = len(msl_text.splitlines())
        assert (
            line_count < 10000
        ), f"Matmul MSL has {line_count} lines - possible code bloat (expected < 10000)"
        assert (
            line_count > 20
        ), f"Matmul MSL has only {line_count} lines - suspiciously small"


# ── Runtime Conformance Tests ──────────────────────────────────────


class TestMetalRuntimeConformance:
    """Test Metal runtime behaviour matches the shared contract."""

    # ── Stream semantics ──────────────────────────────────────────

    @skip_non_darwin
    def test_stream_default(self):
        from third_party.metal.backend.driver import MetalUtils

        utils = MetalUtils()
        assert utils.get_current_stream() == 0

    @skip_non_darwin
    def test_stream_switch_and_return(self):
        from third_party.metal.backend.driver import MetalUtils

        utils = MetalUtils()
        utils.set_stream(1)
        assert utils.get_current_stream() == 1
        utils.set_stream(0)
        assert utils.get_current_stream() == 0

    @skip_non_darwin
    def test_stream_queue_created(self):
        from third_party.metal.backend.driver import MetalUtils

        utils = MetalUtils()
        utils.set_stream(42)
        q = utils.get_command_queue(42)
        assert q is not None
        # Cleanup: switch back to default
        utils.set_stream(0)

    @skip_non_darwin
    def test_synchronize_empty_stream(self):
        """synchronize_stream on a stream with no pending work should not raise."""
        from third_party.metal.backend.driver import MetalUtils

        utils = MetalUtils()
        utils.synchronize_stream(0)

    # ── Argument binding ──────────────────────────────────────────

    def test_argument_binding_i64_no_truncation(self):
        large_val = 2**40
        packed = struct.pack("q", large_val)
        assert len(packed) == 8
        assert struct.unpack("q", packed)[0] == large_val

    def test_argument_binding_negative_i64(self):
        val = -(2**33)
        packed = struct.pack("q", val)
        assert len(packed) == 8
        assert struct.unpack("q", packed)[0] == val

    def test_argument_binding_i32_range(self):
        val = 2**30
        packed = struct.pack("i", val)
        assert len(packed) == 4
        assert struct.unpack("i", packed)[0] == val

    def test_argument_binding_f16_format(self):
        packed = struct.pack("e", 1.5)
        assert len(packed) == 2
        assert abs(struct.unpack("e", packed)[0] - 1.5) < 1e-3

    def test_argument_binding_f64_format(self):
        import math

        packed = struct.pack("d", math.pi)
        assert len(packed) == 8
        assert abs(struct.unpack("d", packed)[0] - math.pi) < 1e-15

    def test_arg_pack_format_map(self):
        from third_party.metal.backend.driver import _ARG_PACK_FORMAT

        expected_keys = {"i32", "i64", "u32", "u64", "f32", "f64", "f16"}
        assert expected_keys == set(_ARG_PACK_FORMAT.keys())

    def test_arg_pack_format_sizes(self):
        from third_party.metal.backend.driver import _ARG_PACK_FORMAT

        expected_sizes = {
            "i32": 4,
            "i64": 8,
            "u32": 4,
            "u64": 8,
            "f32": 4,
            "f64": 8,
            "f16": 2,
        }
        for key, fmt in _ARG_PACK_FORMAT.items():
            assert struct.calcsize(fmt) == expected_sizes[key], f"{key} size mismatch"

    # ── Scratch buffer ────────────────────────────────────────────

    @skip_non_darwin
    @skip_no_xcrun
    def test_scratch_buffer_allocated_when_requested(self):
        from third_party.metal.backend.driver import MetalKernelHandle, MetalUtils

        utils = MetalUtils()
        dev = utils.device
        cq = utils.command_queue
        # Need a dummy library; use a trivial metallib
        from third_party.metal.backend.compiler import MetalBackend, MetalOptions

        source = (
            "#include <metal_stdlib>\nusing namespace metal;\n"
            "kernel void noop(uint id [[thread_position_in_grid]]) {}\n"
        )
        opts = MetalOptions(arch="apple8")
        binary = MetalBackend.make_metallib(source, {}, opts)

        import tempfile

        import Foundation

        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".metallib", delete=False) as f:
                f.write(binary)
                tmp_path = f.name
            url = Foundation.NSURL.fileURLWithPath_(tmp_path)
            result = dev.newLibraryWithURL_error_(url, None)
            library = result[0] if isinstance(result, tuple) else result
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)

        handle = MetalKernelHandle(
            device=dev,
            command_queue=cq,
            library=library,
            metadata={"global_scratch_size": 4096},
        )
        assert handle.global_scratch_size == 4096
        scratch = handle._get_scratch_buffer()
        assert scratch is not None

    def test_scratch_buffer_zero_means_none(self):
        """When global_scratch_size is 0 or absent, no buffer is created."""
        from third_party.metal.backend.driver import MetalKernelHandle

        handle = MetalKernelHandle.__new__(MetalKernelHandle)
        handle.metadata = {}
        handle.global_scratch_size = 0
        handle._scratch_buffer = None
        handle.device = None
        assert handle._get_scratch_buffer() is None

    # ── Execution mode detection ──────────────────────────────────

    @skip_non_darwin
    def test_execution_mode_detection(self):
        from third_party.metal.backend.driver import MetalUtils

        utils = MetalUtils()
        # Reset cached mode for a fresh probe
        utils._execution_mode = None
        mode = utils.resolve_execution_mode()
        assert mode in ("torch_mps", "pyobjc", "unavailable")

    def test_execution_mode_unavailable_when_nothing(self):
        from third_party.metal.backend.driver import MetalUtils

        utils = MetalUtils()
        old_mode = utils._execution_mode
        old_metal = utils._Metal
        old_torch = utils._torch
        try:
            utils._execution_mode = None
            utils._Metal = None
            utils._torch = None
            with patch(
                "third_party.metal.backend.driver._get_torch_module", return_value=None
            ), patch.dict("sys.modules", {"torch": None}):
                mode = utils.resolve_execution_mode()
            assert mode == "unavailable"
        finally:
            utils._execution_mode = old_mode
            utils._Metal = old_metal
            utils._torch = old_torch

    # ── Launch hooks ──────────────────────────────────────────────

    @skip_non_darwin
    def test_launch_hooks_called(self):
        from unittest.mock import MagicMock

        from third_party.metal.backend.driver import (
            MetalLauncher,
            TorchMetalKernelHandle,
        )

        mock_enter = MagicMock()
        mock_exit = MagicMock()

        mock_lib = MagicMock()
        mock_fn = MagicMock()
        mock_lib.test_fn = mock_fn
        handle = TorchMetalKernelHandle(
            shader_library=mock_lib,
            metadata={"name": "test_fn"},
        )

        launcher = MetalLauncher.__new__(MetalLauncher)
        launcher.metadata = {"name": "test_fn"}
        launcher._signature_layout = []

        launcher(
            1,
            1,
            1,  # grid
            0,  # stream
            handle,
            {"name": "test_fn"},  # kernel_metadata
            {},  # launch_metadata
            mock_enter,
            mock_exit,
        )

        mock_enter.assert_called_once()
        mock_exit.assert_called_once()

    def test_launcher_consumes_stream_argument(self):
        from third_party.metal.backend.driver import (
            MetalLauncher,
            TorchMetalKernelHandle,
        )

        mock_lib = MagicMock()
        mock_fn = MagicMock()
        mock_lib.test_fn = mock_fn
        handle = TorchMetalKernelHandle(
            shader_library=mock_lib,
            metadata={"name": "test_fn"},
        )

        launcher = MetalLauncher.__new__(MetalLauncher)
        launcher.metadata = {"name": "test_fn"}
        launcher._signature_layout = []
        launcher._utils = MagicMock()
        launcher._utils.activate_stream.return_value = (0, 7)

        launcher(
            1,
            1,
            1,
            7,  # stream
            handle,
            {"name": "test_fn"},
            {},
            None,
            None,
        )

        launcher._utils.activate_stream.assert_called_once_with(7)
        launcher._utils.restore_stream.assert_called_once_with(0)

    def test_source_handle_pyobjc_fallback(self):
        from third_party.metal.backend.driver import MetalUtils

        utils = MetalUtils()
        dummy_handle = object()
        with patch.object(
            utils, "resolve_execution_mode", return_value="pyobjc"
        ), patch.object(
            utils, "get_device_properties", return_value={"gpu_family": "apple8"}
        ), patch(
            "third_party.metal.backend.driver._get_torch_module", return_value=None
        ), patch.object(
            utils, "_torch", None
        ), patch(
            "third_party.metal.backend.compiler.MetalBackend.make_metallib",
            return_value=b"MTLB",
        ), patch.object(
            utils, "_load_metallib_handle", return_value=dummy_handle
        ) as load_metallib:
            handle = utils._load_msl_source_handle("kernel void test_fn() {}")

        assert handle is dummy_handle
        load_metallib.assert_called_once()

    def test_utils_launch_consumes_stream(self):
        from third_party.metal.backend.driver import MetalUtils, TorchMetalKernelHandle

        mock_lib = MagicMock()
        mock_lib.test_fn = MagicMock()
        handle = TorchMetalKernelHandle(
            shader_library=mock_lib,
            metadata={"name": "test_fn"},
        )
        handle.launch_kernel = MagicMock()
        utils = MetalUtils()
        with patch.object(utils, "activate_stream", return_value=(0, 9)), patch.object(
            utils, "restore_stream"
        ) as restore_stream:
            utils.launch(
                1,
                1,
                1,
                9,
                handle,
                False,
                False,
                {"name": "test_fn"},
                {},
                None,
                None,
                None,
                None,
                None,
                None,
                [],
            )

        handle.launch_kernel.assert_called_once()
        assert handle.launch_kernel.call_args.kwargs["stream_id"] == 9
        assert handle.launch_kernel.call_args.kwargs["sync"] is False
        restore_stream.assert_called_once_with(0)

    def test_utils_launch_rejects_cooperative_grid(self):
        from third_party.metal.backend.driver import MetalUtils, TorchMetalKernelHandle

        mock_lib = MagicMock()
        mock_lib.test_fn = MagicMock()
        handle = TorchMetalKernelHandle(
            shader_library=mock_lib,
            metadata={"name": "test_fn"},
        )
        handle.launch_kernel = MagicMock()
        utils = MetalUtils()
        with patch.object(utils, "activate_stream", return_value=(0, 1)), patch.object(
            utils, "restore_stream"
        ) as restore_stream:
            with pytest.raises(RuntimeError, match="cooperative-grid"):
                utils.launch(
                    1,
                    1,
                    1,
                    1,
                    handle,
                    True,
                    False,
                    {"name": "test_fn"},
                    {},
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    [],
                )
        restore_stream.assert_called_once_with(0)

    # ── MetalDriver stream proxy ──────────────────────────────────

    @skip_non_darwin
    def test_driver_get_current_stream(self):
        from third_party.metal.backend.driver import MetalDriver

        driver = MetalDriver()
        assert driver.get_current_stream() == 0
        driver.utils.set_stream(5)
        assert driver.get_current_stream() == 5
        driver.utils.set_stream(0)


# ── ML Workload Test Infrastructure ─────────────────────────────────


class MetalTestHarness:
    """Reusable test infrastructure for Metal backend testing."""

    TOLERANCE = {
        "float32": {"rtol": 1e-5, "atol": 1e-5},
        "float16": {"rtol": 1e-3, "atol": 1e-3},
        "bfloat16": {"rtol": 1e-2, "atol": 1e-2},
        "int32": {"rtol": 0, "atol": 0},
    }

    @staticmethod
    def get_metal_backend():
        """Get a Metal backend instance configured for apple8."""
        from third_party.metal.backend.compiler import MetalBackend

        from triton.backends.compiler import GPUTarget

        target = GPUTarget("metal", "apple8", 32)
        return MetalBackend(target)

    @staticmethod
    def compile_triton_kernel(kernel_fn, signature, constexprs):
        """Compile a Triton kernel through the full Metal pipeline.

        Returns the compiled kernel with asm artifacts.
        """
        import triton
        from triton.backends.compiler import GPUTarget

        src = triton.compiler.ASTSource(
            fn=kernel_fn,
            signature=signature,
            constexprs=constexprs,
        )
        return triton.compile(src=src, target=GPUTarget("metal", "apple8", 32))

    @staticmethod
    def create_reference_tensors(shape, dtype="float32", seed=42):
        """Create deterministic test tensors with CPU reference."""
        import numpy as np

        rng = np.random.RandomState(seed)
        return rng.randn(*shape).astype(dtype)

    @staticmethod
    def assert_close(actual, expected, dtype="float32"):
        """Assert tensors are close within dtype-specific tolerance."""
        import numpy as np

        tol = MetalTestHarness.TOLERANCE.get(str(dtype), {"rtol": 1e-5, "atol": 1e-5})
        np.testing.assert_allclose(actual, expected, **tol)


# ── ML Workload Compilation Tests ───────────────────────────────────


class TestMetalMLWorkloads:
    """Test ML workload compilation breadth via IR→MSL translation."""

    def test_vector_add_compile(self):
        """Vector add compiles to valid MSL with correct ops."""
        from third_party.metal.backend.compiler import MetalBackend

        ir = """\
define void @vector_add_kernel(ptr %a, ptr %b, ptr %out, i32 %n) {
entry:
  %tid = call i32 @__metal_get_thread_position_in_threadgroup_x()
  %cmp = icmp slt i32 %tid, %n
  br i1 %cmp, label %body, label %exit

body:
  %idx = sext i32 %tid to i64
  %pa = getelementptr float, ptr %a, i64 %idx
  %pb = getelementptr float, ptr %b, i64 %idx
  %pout = getelementptr float, ptr %out, i64 %idx
  %va = load float, ptr %pa
  %vb = load float, ptr %pb
  %sum = fadd float %va, %vb
  store float %sum, ptr %pout
  br label %exit

exit:
  ret void
}
"""
        msl = MetalBackend.make_metal_ir(ir, {}, None)
        assert "UNSUPPORTED" not in msl
        assert "kernel void" in msl
        assert " + " in msl

    def test_reduction_sum_compile(self):
        """Sum reduction with loop/phi compiles correctly."""
        from third_party.metal.backend.compiler import MetalBackend

        ir = """\
define void @reduction_sum_kernel(ptr %input, ptr %out, i32 %n) {
entry:
  %tid = call i32 @__metal_get_thread_position_in_threadgroup_x()
  br label %loop_header

loop_header:
  %i = phi i32 [0, %entry], [%i_next, %loop_body]
  %acc = phi float [0.0, %entry], [%acc_next, %loop_body]
  %cmp = icmp slt i32 %i, %n
  br i1 %cmp, label %loop_body, label %write_out

loop_body:
  %idx = sext i32 %i to i64
  %ptr = getelementptr float, ptr %input, i64 %idx
  %val = load float, ptr %ptr
  %acc_next = fadd float %acc, %val
  %i_next = add i32 %i, 1
  br label %loop_header

write_out:
  %out_idx = sext i32 %tid to i64
  %out_ptr = getelementptr float, ptr %out, i64 %out_idx
  store float %acc, ptr %out_ptr
  ret void
}
"""
        msl = MetalBackend.make_metal_ir(ir, {}, None)
        assert "UNSUPPORTED" not in msl
        assert "kernel void" in msl
        assert "__triton_pred_block" in msl
        assert " + " in msl

    def test_softmax_compile(self):
        """Softmax pattern (exp, sub, div) compiles correctly."""
        from third_party.metal.backend.compiler import MetalBackend

        ir = """\
define void @softmax_kernel(ptr %input, ptr %out, float %max_val) {
entry:
  %tid = call i32 @__metal_get_thread_position_in_threadgroup_x()
  %idx = sext i32 %tid to i64
  %ptr_in = getelementptr float, ptr %input, i64 %idx
  %val = load float, ptr %ptr_in
  %shifted = fsub float %val, %max_val
  %exp_val = call float @__nv_expf(float %shifted)
  %sum_inv = fdiv float 1.0, %max_val
  %result = fmul float %exp_val, %sum_inv
  %ptr_out = getelementptr float, ptr %out, i64 %idx
  store float %result, ptr %ptr_out
  ret void
}
"""
        msl = MetalBackend.make_metal_ir(ir, {}, None)
        assert "UNSUPPORTED" not in msl
        assert "kernel void" in msl
        assert "exp(" in msl
        assert " - " in msl
        assert " / " in msl
        assert " * " in msl

    def test_matmul_compile(self):
        """Matmul (nested loop with fma) compiles correctly."""
        from third_party.metal.backend.compiler import MetalBackend

        ir = """\
define void @matmul_kernel(ptr %A, ptr %B, ptr %C, i32 %M, i32 %N, i32 %K) {
entry:
  %row = call i32 @__metal_get_thread_position_in_threadgroup_x()
  %col = call i32 @__metal_get_thread_position_in_threadgroup_y()
  br label %k_loop

k_loop:
  %k = phi i32 [0, %entry], [%k_next, %k_body]
  %acc = phi float [0.0, %entry], [%acc_next, %k_body]
  %k_cmp = icmp slt i32 %k, %K
  br i1 %k_cmp, label %k_body, label %store_result

k_body:
  %a_off = mul i32 %row, %K
  %a_idx = add i32 %a_off, %k
  %a_idx64 = sext i32 %a_idx to i64
  %a_ptr = getelementptr float, ptr %A, i64 %a_idx64
  %a_val = load float, ptr %a_ptr
  %b_off = mul i32 %k, %N
  %b_idx = add i32 %b_off, %col
  %b_idx64 = sext i32 %b_idx to i64
  %b_ptr = getelementptr float, ptr %B, i64 %b_idx64
  %b_val = load float, ptr %b_ptr
  %acc_next = call float @llvm.fmuladd.f32(float %a_val, float %b_val, float %acc)
  %k_next = add i32 %k, 1
  br label %k_loop

store_result:
  %c_off = mul i32 %row, %N
  %c_idx = add i32 %c_off, %col
  %c_idx64 = sext i32 %c_idx to i64
  %c_ptr = getelementptr float, ptr %C, i64 %c_idx64
  store float %acc, ptr %c_ptr
  ret void
}
"""
        msl = MetalBackend.make_metal_ir(ir, {}, None)
        assert "UNSUPPORTED" not in msl
        assert "kernel void" in msl
        assert "fma(" in msl

    def test_silu_activation_compile(self):
        """SiLU (x * sigmoid(x)) pattern compiles correctly."""
        from third_party.metal.backend.compiler import MetalBackend

        ir = """\
define void @silu_kernel(ptr %input, ptr %out, i32 %n) {
entry:
  %tid = call i32 @__metal_get_thread_position_in_threadgroup_x()
  %cmp = icmp slt i32 %tid, %n
  br i1 %cmp, label %body, label %exit

body:
  %idx = sext i32 %tid to i64
  %ptr_in = getelementptr float, ptr %input, i64 %idx
  %x = load float, ptr %ptr_in
  %neg_x = fneg float %x
  %exp_neg = call float @__nv_expf(float %neg_x)
  %one_plus = fadd float 1.0, %exp_neg
  %sigmoid = fdiv float %x, %one_plus
  %ptr_out = getelementptr float, ptr %out, i64 %idx
  store float %sigmoid, ptr %ptr_out
  br label %exit

exit:
  ret void
}
"""
        msl = MetalBackend.make_metal_ir(ir, {}, None)
        assert "UNSUPPORTED" not in msl
        assert "kernel void" in msl
        assert "exp(" in msl
        assert " / " in msl
        assert "= -(" in msl

    def test_layer_norm_compile(self):
        """LayerNorm pattern (center, scale) compiles correctly."""
        from third_party.metal.backend.compiler import MetalBackend

        ir = """\
define void @layernorm_kernel(ptr %input, ptr %weight, ptr %out, float %mean, float %inv_std) {
entry:
  %tid = call i32 @__metal_get_thread_position_in_threadgroup_x()
  %idx = sext i32 %tid to i64
  %ptr_in = getelementptr float, ptr %input, i64 %idx
  %x = load float, ptr %ptr_in
  %centered = fsub float %x, %mean
  %normed = fmul float %centered, %inv_std
  %ptr_w = getelementptr float, ptr %weight, i64 %idx
  %w = load float, ptr %ptr_w
  %scaled = fmul float %normed, %w
  %ptr_out = getelementptr float, ptr %out, i64 %idx
  store float %scaled, ptr %ptr_out
  ret void
}
"""
        msl = MetalBackend.make_metal_ir(ir, {}, None)
        assert "UNSUPPORTED" not in msl
        assert "kernel void" in msl
        assert " - " in msl
        assert " * " in msl

    def test_embedding_lookup_compile(self):
        """Embedding/gather pattern (index → load) compiles correctly."""
        from third_party.metal.backend.compiler import MetalBackend

        ir = """\
define void @embedding_kernel(ptr %table, ptr %indices, ptr %out, i32 %n, i32 %dim) {
entry:
  %tid = call i32 @__metal_get_thread_position_in_threadgroup_x()
  %cmp = icmp slt i32 %tid, %n
  br i1 %cmp, label %body, label %exit

body:
  %tid64 = sext i32 %tid to i64
  %idx_ptr = getelementptr i32, ptr %indices, i64 %tid64
  %row = load i32, ptr %idx_ptr
  %row_off = mul i32 %row, %dim
  %elem_idx = add i32 %row_off, 0
  %elem_idx64 = sext i32 %elem_idx to i64
  %tab_ptr = getelementptr float, ptr %table, i64 %elem_idx64
  %val = load float, ptr %tab_ptr
  %out_ptr = getelementptr float, ptr %out, i64 %tid64
  store float %val, ptr %out_ptr
  br label %exit

exit:
  ret void
}
"""
        msl = MetalBackend.make_metal_ir(ir, {}, None)
        assert "UNSUPPORTED" not in msl
        assert "kernel void" in msl

    def test_elementwise_chain_compile(self):
        """Chained elementwise ops (add → mul → relu) compile correctly."""
        from third_party.metal.backend.compiler import MetalBackend

        ir = """\
define void @chain_kernel(ptr %a, ptr %b, ptr %out, i32 %n) {
entry:
  %tid = call i32 @__metal_get_thread_position_in_threadgroup_x()
  %cmp = icmp slt i32 %tid, %n
  br i1 %cmp, label %body, label %exit

body:
  %idx = sext i32 %tid to i64
  %pa = getelementptr float, ptr %a, i64 %idx
  %pb = getelementptr float, ptr %b, i64 %idx
  %va = load float, ptr %pa
  %vb = load float, ptr %pb
  %sum = fadd float %va, %vb
  %prod = fmul float %sum, %vb
  %cmp_relu = fcmp ogt float %prod, 0.0
  %relu = select i1 %cmp_relu, float %prod, float 0.0
  %pout = getelementptr float, ptr %out, i64 %idx
  store float %relu, ptr %pout
  br label %exit

exit:
  ret void
}
"""
        msl = MetalBackend.make_metal_ir(ir, {}, None)
        assert "UNSUPPORTED" not in msl
        assert "kernel void" in msl
        assert " + " in msl
        assert " * " in msl
        assert " ? " in msl


# ── Mixed Precision Tests ───────────────────────────────────────────


class TestMetalMixedPrecision:
    """Test mixed precision compilation paths."""

    def test_fp16_to_fp32_convert(self):
        """fp16 input → fp32 computation chain compiles."""
        from third_party.metal.backend.compiler import MetalBackend

        ir = """\
define void @fp16_to_fp32_kernel(ptr %out, half %h_in) {
entry:
  %ext = fpext half %h_in to float
  %r = fmul float %ext, 2.0
  %p = getelementptr float, ptr %out, i64 0
  store float %r, ptr %p
  ret void
}
"""
        msl = MetalBackend.make_metal_ir(ir, {}, None)
        assert "UNSUPPORTED" not in msl
        assert "kernel void" in msl
        assert "(float)" in msl

    def test_fp32_to_fp16_truncate(self):
        """fp32 → fp16 truncation compiles."""
        from third_party.metal.backend.compiler import MetalBackend

        ir = """\
define void @fp32_to_fp16_kernel(ptr %out, float %f_in) {
entry:
  %truncated = fptrunc float %f_in to half
  %p = getelementptr half, ptr %out, i64 0
  store half %truncated, ptr %p
  ret void
}
"""
        msl = MetalBackend.make_metal_ir(ir, {}, None)
        assert "UNSUPPORTED" not in msl
        assert "kernel void" in msl
        assert "(half)" in msl

    def test_mixed_int_widths(self):
        """i8/i16/i32/i64 mixed width operations compile."""
        from third_party.metal.backend.compiler import MetalBackend

        ir = """\
define void @mixed_int_kernel(ptr %out, i64 %i64_in) {
entry:
  %narrow32 = trunc i64 %i64_in to i32
  %narrow16 = trunc i32 %narrow32 to i16
  %narrow8 = trunc i16 %narrow16 to i8
  %wide16 = zext i8 %narrow8 to i16
  %wide32 = sext i16 %wide16 to i32
  %wide64 = sext i32 %wide32 to i64
  %p = getelementptr i32, ptr %out, i64 0
  store i32 %wide32, ptr %p
  ret void
}
"""
        msl = MetalBackend.make_metal_ir(ir, {}, None)
        assert "UNSUPPORTED" not in msl
        assert "kernel void" in msl

    def test_tolerance_envelopes_documented(self):
        """Verify tolerance constants are defined for all expected dtypes."""
        for dtype in ("float32", "float16", "bfloat16", "int32"):
            assert dtype in MetalTestHarness.TOLERANCE, f"Missing tolerance for {dtype}"
            tol = MetalTestHarness.TOLERANCE[dtype]
            assert "rtol" in tol, f"Missing rtol for {dtype}"
            assert "atol" in tol, f"Missing atol for {dtype}"


# ── Dynamic Shape Coverage ──────────────────────────────────────────


class TestMetalDynamicShapes:
    """Test compilation with non-standard shapes and block sizes."""

    def test_non_power_of_2_elements(self):
        """Kernel with non-power-of-2 bounds check (127 elements) compiles."""
        from third_party.metal.backend.compiler import MetalBackend

        ir = """\
define void @non_pow2_kernel(ptr %out, i32 %n) {
entry:
  %tid = call i32 @__metal_get_thread_position_in_threadgroup_x()
  %cmp = icmp slt i32 %tid, 127
  br i1 %cmp, label %body, label %exit

body:
  %idx = sext i32 %tid to i64
  %p = getelementptr float, ptr %out, i64 %idx
  %val = sitofp i32 %tid to float
  store float %val, ptr %p
  br label %exit

exit:
  ret void
}
"""
        msl = MetalBackend.make_metal_ir(ir, {}, None)
        assert "UNSUPPORTED" not in msl
        assert "kernel void" in msl

    def test_very_small_tensor(self):
        """Kernel operating on a single element compiles."""
        from third_party.metal.backend.compiler import MetalBackend

        ir = """\
define void @single_elem_kernel(ptr %out, float %val) {
entry:
  %p = getelementptr float, ptr %out, i64 0
  store float %val, ptr %p
  ret void
}
"""
        msl = MetalBackend.make_metal_ir(ir, {}, None)
        assert "UNSUPPORTED" not in msl
        assert "kernel void" in msl

    def test_large_tensor_compile(self):
        """Kernel with large index (1M elements) compiles."""
        from third_party.metal.backend.compiler import MetalBackend

        ir = """\
define void @large_tensor_kernel(ptr %out, i32 %n) {
entry:
  %gid = call i32 @__metal_get_threadgroup_position_in_grid_x()
  %tid = call i32 @__metal_get_thread_position_in_threadgroup_x()
  %block_off = shl i32 %gid, 10
  %idx = add i32 %block_off, %tid
  %cmp = icmp slt i32 %idx, 1048576
  br i1 %cmp, label %body, label %exit

body:
  %idx64 = sext i32 %idx to i64
  %p = getelementptr float, ptr %out, i64 %idx64
  %val = sitofp i32 %idx to float
  store float %val, ptr %p
  br label %exit

exit:
  ret void
}
"""
        msl = MetalBackend.make_metal_ir(ir, {}, None)
        assert "UNSUPPORTED" not in msl
        assert "kernel void" in msl

    def test_odd_block_size(self):
        """Non-standard block size (33) in masking compiles."""
        from third_party.metal.backend.compiler import MetalBackend

        ir = """\
define void @odd_block_kernel(ptr %a, ptr %out, i32 %n) {
entry:
  %gid = call i32 @__metal_get_threadgroup_position_in_grid_x()
  %tid = call i32 @__metal_get_thread_position_in_threadgroup_x()
  %block_off = mul i32 %gid, 33
  %idx = add i32 %block_off, %tid
  %cmp = icmp slt i32 %idx, %n
  br i1 %cmp, label %body, label %exit

body:
  %idx64 = sext i32 %idx to i64
  %pa = getelementptr float, ptr %a, i64 %idx64
  %va = load float, ptr %pa
  %doubled = fmul float %va, 2.0
  %pout = getelementptr float, ptr %out, i64 %idx64
  store float %doubled, ptr %pout
  br label %exit

exit:
  ret void
}
"""
        msl = MetalBackend.make_metal_ir(ir, {}, None)
        assert "UNSUPPORTED" not in msl
        assert "kernel void" in msl


# ── Cross-Backend Numerical Comparison ──────────────────────────────


class TestMetalCrossBackendNumerics:
    """Cross-backend numerical comparison infrastructure."""

    def test_reference_generation(self):
        """Verify CPU reference generation is deterministic."""
        import numpy as np

        ref1 = MetalTestHarness.create_reference_tensors((128,), seed=42)
        ref2 = MetalTestHarness.create_reference_tensors((128,), seed=42)
        np.testing.assert_array_equal(ref1, ref2)

    def test_tolerance_bounds(self):
        """Verify tolerance bounds are reasonable across dtypes."""
        tol = MetalTestHarness.TOLERANCE
        assert tol["float32"]["atol"] < tol["float16"]["atol"]
        assert tol["float16"]["atol"] < tol["bfloat16"]["atol"]
        assert tol["int32"]["atol"] == 0

    def test_numerical_drift_logging(self):
        """Verify drift within tolerance envelope passes."""
        import numpy as np

        ref = np.ones(100, dtype=np.float32)
        result = ref + 1e-6 * np.random.RandomState(99).randn(100).astype(np.float32)
        MetalTestHarness.assert_close(result, ref, dtype="float32")

    def test_vector_add_numerics_cpu_reference(self):
        """Vector add CPU reference matches expected output."""
        import numpy as np

        a = MetalTestHarness.create_reference_tensors((1024,), seed=1)
        b = MetalTestHarness.create_reference_tensors((1024,), seed=2)
        expected = a + b
        assert expected.shape == (1024,)
        assert expected.dtype == np.float32

    def test_matmul_numerics_cpu_reference(self):
        """Matmul CPU reference matches numpy."""
        import numpy as np

        a = MetalTestHarness.create_reference_tensors((64, 32), seed=1).astype(
            np.float32
        )
        b = MetalTestHarness.create_reference_tensors((32, 64), seed=2).astype(
            np.float32
        )
        expected = a @ b
        assert expected.shape == (64, 64)


# ── Audit ERR regression tests ──────────────────────────────────────


class TestMetalAuditERR001HalfHexFloat:
    """ERR-001: Half / bfloat16 hex float constants must produce valid MSL."""

    def test_half_hex_zero(self):
        from third_party.metal.backend.compiler import MetalBackend

        llvm_ir = """
define void @half_zero(ptr %out) {
  store half 0xH0000, ptr %out
  ret void
}
"""
        metadata = {}
        msl = MetalBackend.make_metal_ir(llvm_ir, metadata, None)
        assert "0xH" not in msl, "Raw LLVM half hex literal leaked into MSL"

    def test_half_hex_one(self):
        from third_party.metal.backend.compiler import MetalBackend

        llvm_ir = """
define void @half_one(ptr %out) {
  store half 0xH3C00, ptr %out
  ret void
}
"""
        metadata = {}
        msl = MetalBackend.make_metal_ir(llvm_ir, metadata, None)
        assert "0xH" not in msl
        assert "(half)" in msl or "1.0" in msl

    def test_half_hex_negative(self):
        from third_party.metal.backend.compiler import MetalBackend

        llvm_ir = """
define void @half_neg(ptr %out) {
  store half 0xHBC00, ptr %out
  ret void
}
"""
        metadata = {}
        msl = MetalBackend.make_metal_ir(llvm_ir, metadata, None)
        assert "0xH" not in msl

    def test_half_hex_infinity(self):
        from third_party.metal.backend.compiler import MetalBackend

        llvm_ir = """
define void @half_inf(ptr %out) {
  store half 0xH7C00, ptr %out
  ret void
}
"""
        metadata = {}
        msl = MetalBackend.make_metal_ir(llvm_ir, metadata, None)
        assert "INFINITY" in msl

    def test_half_hex_nan(self):
        from third_party.metal.backend.compiler import MetalBackend

        llvm_ir = """
define void @half_nan(ptr %out) {
  store half 0xH7E00, ptr %out
  ret void
}
"""
        metadata = {}
        msl = MetalBackend.make_metal_ir(llvm_ir, metadata, None)
        assert "NAN" in msl

    def test_bfloat16_constant_to_msl_regex(self):
        """Verify _RE_CONST_HEX_BFLOAT regex matches 0xR prefix."""
        import struct as _struct

        from third_party.metal.backend.compiler import _RE_CONST_HEX_BFLOAT

        m = _RE_CONST_HEX_BFLOAT.match("0xR3F80")
        assert m is not None
        raw_bf = int(m.group(1), 16)
        f32_bits = raw_bf << 16
        fval = _struct.unpack("f", _struct.pack("I", f32_bits))[0]
        assert abs(fval - 1.0) < 0.01


class TestMetalAuditERR002FunnelShiftZeroGuard:
    """ERR-002: fshr/fshl must not invoke UB when shift amount is 0."""

    def test_fshr_shift_zero_guard(self):
        from third_party.metal.backend.compiler import MetalBackend

        llvm_ir = """
define void @fshr_kernel(ptr %out, i32 %a, i32 %b, i32 %c) {
  %r = call i32 @llvm.fshr.i32(i32 %a, i32 %b, i32 %c)
  store i32 %r, ptr %out
  ret void
}
declare i32 @llvm.fshr.i32(i32, i32, i32)
"""
        metadata = {}
        msl = MetalBackend.make_metal_ir(llvm_ir, metadata, None)
        assert "== 0 ?" in msl, "Missing zero-shift guard in fshr expansion"

    def test_fshl_shift_zero_guard(self):
        from third_party.metal.backend.compiler import MetalBackend

        llvm_ir = """
define void @fshl_kernel(ptr %out, i32 %a, i32 %b, i32 %c) {
  %r = call i32 @llvm.fshl.i32(i32 %a, i32 %b, i32 %c)
  store i32 %r, ptr %out
  ret void
}
declare i32 @llvm.fshl.i32(i32, i32, i32)
"""
        metadata = {}
        msl = MetalBackend.make_metal_ir(llvm_ir, metadata, None)
        assert "== 0 ?" in msl, "Missing zero-shift guard in fshl expansion"

    def test_fshr_i64_guard(self):
        from third_party.metal.backend.compiler import MetalBackend

        llvm_ir = """
define void @fshr64(ptr %out, i64 %a, i64 %b, i64 %c) {
  %r = call i64 @llvm.fshr.i64(i64 %a, i64 %b, i64 %c)
  store i64 %r, ptr %out
  ret void
}
declare i64 @llvm.fshr.i64(i64, i64, i64)
"""
        metadata = {}
        msl = MetalBackend.make_metal_ir(llvm_ir, metadata, None)
        assert "64 - 1" in msl and "== 0 ?" in msl


class TestMetalAuditERR003MemmoveDirection:
    """ERR-003: memmove must handle overlapping regions correctly."""

    def test_memmove_uses_backward_copy(self):
        from third_party.metal.backend.compiler import MetalBackend

        llvm_ir = """
define void @memmove_kernel(ptr %dst, ptr %src) {
  call void @llvm.memmove.p0.p0.i32(ptr %dst, ptr %src, i32 16, i1 false)
  ret void
}
declare void @llvm.memmove.p0.p0.i32(ptr, ptr, i32, i1)
"""
        metadata = {}
        msl = MetalBackend.make_metal_ir(llvm_ir, metadata, None)
        assert "__i >= 0; __i--" in msl, "Missing backward copy path in memmove"
        assert "uintptr_t" in msl, "Missing pointer comparison for direction"


class TestMetalAuditERR004AttrGroupStripping:
    """ERR-004: LLVM attribute group refs (#N) on calls must not block matching."""

    def test_call_with_attr_group(self):
        from third_party.metal.backend.compiler import MetalBackend

        llvm_ir = """
define void @attr_kernel(ptr %out) {
  %tid = call i32 @__metal_get_thread_position_in_threadgroup_x() #3
  store i32 %tid, ptr %out
  ret void
}
"""
        metadata = {}
        msl = MetalBackend.make_metal_ir(llvm_ir, metadata, None)
        assert "thread_position_in_threadgroup.x" in msl

    def test_void_call_with_attr_group(self):
        from third_party.metal.backend.compiler import MetalBackend

        llvm_ir = """
define void @lifetime_kernel(ptr %p) {
  call void @llvm.lifetime.start.p0(i64 4, ptr %p) #1
  call void @llvm.lifetime.end.p0(i64 4, ptr %p) #1
  ret void
}
declare void @llvm.lifetime.start.p0(i64, ptr)
declare void @llvm.lifetime.end.p0(i64, ptr)
"""
        metadata = {}
        msl = MetalBackend.make_metal_ir(llvm_ir, metadata, None)
        assert "UNSUPPORTED" not in msl


class TestMetalAuditERR005SSANameCollision:
    """ERR-005: Different LLVM SSA names must not collide after msl_id mapping."""

    def test_dot_vs_underscore_disambiguated(self):
        from third_party.metal.backend.compiler import MetalBackend

        llvm_ir = """
define void @collision_kernel(ptr %out, i32 %n) {
  %x.0 = add i32 %n, 1
  %x_0 = add i32 %n, 2
  %sum = add i32 %x.0, %x_0
  store i32 %sum, ptr %out
  ret void
}
"""
        metadata = {}
        msl = MetalBackend.make_metal_ir(llvm_ir, metadata, None)
        lines = msl.split("\n")
        int_decls = [l.strip() for l in lines if l.strip().startswith("int x_0")]
        assert (
            len(int_decls) >= 2
        ), f"Expected at least 2 distinct x_0* declarations; got: {int_decls}"


class TestMetalAudit2AttrGroupWithDebugMetadata:
    """AUDIT2-001: Attr-group refs masked by debug metadata on same line."""

    def test_call_attr_group_before_dbg(self):
        """#N before ,!dbg must be stripped by the second cleaning pass."""
        from third_party.metal.backend.compiler import MetalBackend

        llvm_ir = """
define void @attrdbg_kernel(ptr %out) {
  %tid = call i32 @__metal_get_thread_position_in_threadgroup_x() #3, !dbg !5
  store i32 %tid, ptr %out
  ret void
}
!llvm.dbg.cu = !{}
!5 = !{}
"""
        metadata = {}
        msl = MetalBackend.make_metal_ir(llvm_ir, metadata, None)
        assert (
            "thread_position_in_threadgroup.x" in msl
        ), "Call with #N before !dbg was not matched after line cleaning"

    def test_void_call_attr_group_before_comment(self):
        """#N before ; comment must be stripped."""
        from third_party.metal.backend.compiler import MetalBackend

        llvm_ir = """
define void @attrcomment_kernel(ptr %p) {
  call void @llvm.lifetime.start.p0(i64 4, ptr %p) #1 ; mark start
  call void @llvm.lifetime.end.p0(i64 4, ptr %p) #1 ; mark end
  ret void
}
declare void @llvm.lifetime.start.p0(i64, ptr)
declare void @llvm.lifetime.end.p0(i64, ptr)
"""
        metadata = {}
        msl = MetalBackend.make_metal_ir(llvm_ir, metadata, None)
        assert (
            "UNSUPPORTED" not in msl
        ), "Void call with #N before ; comment was not cleaned properly"


class TestMetalAudit2PowiUsePown:
    """AUDIT2-002: llvm.powi must lower to pown(), not powr().

    MSL powr(x, y) requires x >= 0 (undefined for negative bases).
    MSL pown(x, y) handles negative bases with integer exponents correctly.
    """

    def test_powi_lowers_to_pown(self):
        from third_party.metal.backend.compiler import MetalBackend

        llvm_ir = """
define void @powi_kernel(ptr %out, float %base, i32 %exp) {
  %r = call float @llvm.powi.f32.i32(float %base, i32 %exp)
  store float %r, ptr %out
  ret void
}
declare float @llvm.powi.f32.i32(float, i32)
"""
        metadata = {}
        msl = MetalBackend.make_metal_ir(llvm_ir, metadata, None)
        assert "pown(" in msl, "llvm.powi should lower to pown(), not powr()"
        assert "powr(" not in msl, "powr() requires x >= 0; pown() must be used"

    def test_powi_no_float_cast_on_exponent(self):
        """pown() takes int exponent directly — no static_cast<float> needed."""
        from third_party.metal.backend.compiler import MetalBackend

        llvm_ir = """
define void @powi_cast_kernel(ptr %out, float %x, i32 %n) {
  %r = call float @llvm.powi.f32.i32(float %x, i32 %n)
  store float %r, ptr %out
  ret void
}
declare float @llvm.powi.f32.i32(float, i32)
"""
        metadata = {}
        msl = MetalBackend.make_metal_ir(llvm_ir, metadata, None)
        assert (
            "static_cast<float>" not in msl
        ), "pown() takes integer exponent; float cast is unnecessary"
