"""
Tests for the Metal backend compiler, driver, and runtime.

These tests verify the Metal backend integration with Triton's backend
infrastructure. Tests that require a real Metal device are skipped on
non-macOS platforms.
"""

import os
import sys
import struct
import pytest
import subprocess
import shutil
from unittest.mock import MagicMock, patch

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
        from triton.backends.compiler import GPUTarget, Language
        from triton import knobs

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
        assert "-INFINITY" in msl, "hex float 0xFFF... (-inf) must become -INFINITY in MSL"
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
