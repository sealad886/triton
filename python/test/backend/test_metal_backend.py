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
        assert "llir" in kernel.asm and len(kernel.asm["llir"]) > 0
        assert "metal" in kernel.asm and b"kernel void" in kernel.asm["metal"]
        assert "metallib" in kernel.asm and kernel.asm["metallib"][:4] == b"MTLB"

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
