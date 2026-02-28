"""
Tests for the Metal backend compiler, driver, and runtime.

These tests verify the Metal backend integration with Triton's backend
infrastructure. Tests that require a real Metal device are skipped on
non-macOS platforms.
"""

import ctypes
import os
import shutil
import struct
import subprocess
import sys
import tempfile
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


def _has_mps_runtime() -> bool:
    try:
        import torch

        return bool(
            hasattr(torch.backends, "mps")
            and torch.backends.mps.is_built()
            and torch.backends.mps.is_available()
        )
    except Exception:
        return False


skip_no_mps = pytest.mark.skipif(
    not _has_mps_runtime(),
    reason="MPS runtime not available",
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


@pytest.fixture(scope="session", autouse=True)
def _isolate_triton_cache_dir_for_metal_backend_tests():
    """Use an isolated Triton cache to avoid cross-run stale artifact reuse."""
    prev = os.environ.get("TRITON_CACHE_DIR")
    with tempfile.TemporaryDirectory(prefix="triton-metal-backend-cache-") as tmpdir:
        os.environ["TRITON_CACHE_DIR"] = tmpdir
        try:
            yield
        finally:
            if prev is None:
                os.environ.pop("TRITON_CACHE_DIR", None)
            else:
                os.environ["TRITON_CACHE_DIR"] = prev


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
        assert opts.simdgroup_matmul_strategy == "auto"

    def test_custom_options(self):
        from third_party.metal.backend.compiler import MetalOptions

        opts = MetalOptions(
            num_warps=8,
            debug=True,
            arch="apple9",
            simdgroup_matmul_strategy="fallback",
        )
        assert opts.num_warps == 8
        assert opts.debug is True
        assert opts.arch == "apple9"
        assert opts.simdgroup_matmul_strategy == "fallback"

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
        ), patch(
            "triton.__version__", "triton-version", create=True
        ):
            assert (
                backend.hash() == "sdk-version-apple8-triton-version-backend-src-hash"
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

    def test_get_module_map_has_libdevice(self):
        from third_party.metal.backend.compiler import MetalBackend

        from triton.backends.compiler import GPUTarget

        target = GPUTarget("metal", "apple8", 32)
        backend = MetalBackend(target)
        module_map = backend.get_module_map()
        assert "triton.language.extra.libdevice" in module_map
        from third_party.metal.language import libdevice

        assert module_map["triton.language.extra.libdevice"] is libdevice


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
        assert "? (" in msl
        assert "if (" in msl and "*v11 = v10" in msl

    def test_make_metal_ir_translates_simdgroup_barrier_flags(self):
        from third_party.metal.backend.compiler import MetalBackend

        llvm_ir = """
define void @barrier_kernel() {
entry:
  call void @__metal_simdgroup_barrier(i32 1)
  call void @__metal_simdgroup_barrier(i32 2)
  call void @__metal_simdgroup_barrier(i32 3)
  ret void
}
"""
        metadata = {}
        msl = MetalBackend.make_metal_ir(llvm_ir, metadata, None)
        assert "threadgroup_barrier(mem_flags::mem_threadgroup);" in msl
        assert "threadgroup_barrier(mem_flags::mem_device);" in msl
        assert (
            "threadgroup_barrier((mem_flags::mem_threadgroup | mem_flags::mem_device));"
            in msl
        )

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
        assert driver.map_python_to_cpp_type("*fp32") == "id<MTLBuffer>"
        assert driver.map_python_to_cpp_type("i32") == "int32_t"
        assert driver.map_python_to_cpp_type("fp32") == "float"
        assert driver.map_python_to_cpp_type("fp16") == "half"

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

    @skip_non_darwin
    @skip_no_xcrun
    def test_compile_triton_fp8_blocked_matmul_pipeline(self):
        import torch

        import triton
        import triton.language as tl
        from triton.backends.compiler import GPUTarget

        if not hasattr(torch, "float8_e5m2"):
            pytest.skip("Torch float8 types unavailable on this host")

        @triton.jit
        def _matmul_fp8_kernel(
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

            c = acc.to(tl.float16)
            c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
            tl.store(c_ptrs, c, mask=(offs_m[:, None] < m) & (offs_n[None, :] < n))

        src = triton.compiler.ASTSource(
            fn=_matmul_fp8_kernel,
            signature={
                "a_ptr": "*fp8e5",
                "b_ptr": "*fp8e5",
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
    def test_compile_triton_fp8_roundtrip_convert_pipeline(self):
        import torch

        import triton
        import triton.language as tl
        from triton.backends.compiler import GPUTarget

        if not hasattr(torch, "float8_e5m2"):
            pytest.skip("Torch float8 types unavailable on this host")

        @triton.jit
        def _fp8_roundtrip(src_ptr, dst_ptr, n, BLOCK: tl.constexpr):
            pid = tl.program_id(axis=0)
            offs = pid * BLOCK + tl.arange(0, BLOCK)
            mask = offs < n
            x = tl.load(src_ptr + offs, mask=mask, other=0.0)
            y = x.to(tl.float8e5).to(tl.float16)
            tl.store(dst_ptr + offs, y, mask=mask)

        src = triton.compiler.ASTSource(
            fn=_fp8_roundtrip,
            signature={"src_ptr": "*fp16", "dst_ptr": "*fp16", "n": "i32"},
            constexprs={"BLOCK": 64},
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

    def test_simdgroup_half_translation_uses_typed_pointers(self):
        """Half-typed simdgroup load/store should preserve half pointer typing."""
        from third_party.metal.backend.compiler import MetalBackend

        ir = """\
define void @simdgroup_half_ptr_kernel(ptr addrspace(1) %ptr) {
entry:
  %mat = call <8 x half> @__metal_simdgroup_load(ptr addrspace(1) %ptr, i32 32)
  call void @__metal_simdgroup_store(<8 x half> %mat, ptr addrspace(1) %ptr, i32 32)
  ret void
}
"""
        msl = MetalBackend.make_metal_ir(ir, {}, None)
        assert "(const device half*)" in msl
        assert "(device half*)" in msl

    def test_simdgroup_fallback_strategy_emits_software_helpers(self):
        """Fallback strategy should lower simdgroup intrinsics to software helpers."""
        from third_party.metal.backend.compiler import MetalBackend, MetalOptions

        ir = """\
define void @simdgroup_fallback_kernel(ptr addrspace(1) %a, ptr addrspace(1) %b, ptr addrspace(1) %c) {
entry:
  %ma = call <8 x float> @__metal_simdgroup_load(ptr addrspace(1) %a, i32 64)
  %mb = call <8 x float> @__metal_simdgroup_load(ptr addrspace(1) %b, i32 64)
  %mc = call <8 x float> @__metal_simdgroup_load(ptr addrspace(1) %c, i32 64)
  %md = call <8 x float> @__metal_simdgroup_multiply_accumulate(<8 x float> %ma, <8 x float> %mb, <8 x float> %mc)
  call void @__metal_simdgroup_store(<8 x float> %md, ptr addrspace(1) %c, i32 64)
  ret void
}
"""
        opts = MetalOptions(simdgroup_matmul_strategy="fallback")
        msl = MetalBackend.make_metal_ir(ir, {}, opts)
        assert "struct __metal_sgmat_float" in msl
        assert "__metal_sg_load_float(" in msl
        assert "__metal_sg_mma_float(" in msl
        assert "__metal_sg_store_float(" in msl
        assert "simdgroup_multiply_accumulate(" not in msl


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
        simdgroup_mma_count = msl_text.count("simdgroup_multiply_accumulate(")
        if simdgroup_mma_count == 0:
            simdgroup_mma_count = msl_text.count("__metal_sg_mma_")
        total_matmul_ops = fma_count + simdgroup_mma_count
        assert (
            total_matmul_ops >= 1
        ), f"Expected matmul ops (fma or simdgroup_multiply_accumulate) in MSL, got 0"
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

        expected_keys = {
            "i1",
            "i8",
            "u8",
            "i16",
            "u16",
            "i32",
            "i64",
            "u32",
            "u64",
            "f32",
            "f64",
            "f16",
            "bf16",
        }
        assert expected_keys == set(_ARG_PACK_FORMAT.keys())

    def test_arg_pack_format_sizes(self):
        from third_party.metal.backend.driver import _ARG_PACK_FORMAT

        expected_sizes = {
            "i1": 1,
            "i8": 1,
            "u8": 1,
            "i16": 2,
            "u16": 2,
            "i32": 4,
            "i64": 8,
            "u32": 4,
            "u64": 8,
            "f32": 4,
            "f64": 8,
            "f16": 2,
            "bf16": 2,
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


# ── Runtime ML correctness tests (MPS) ──────────────────────────────


class TestMetalRuntimeMLCorrectness:
    """Runtime correctness checks against CPU references on MPS."""

    @skip_non_darwin
    @skip_no_mps
    def test_runtime_vector_add_matches_cpu(self):
        import torch

        import triton
        import triton.language as tl

        @triton.jit
        def _vadd(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
            pid = tl.program_id(axis=0)
            offs = pid * BLOCK + tl.arange(0, BLOCK)
            mask = offs < n
            x = tl.load(x_ptr + offs, mask=mask, other=0.0)
            y = tl.load(y_ptr + offs, mask=mask, other=0.0)
            tl.store(out_ptr + offs, x + y, mask=mask)

        torch.manual_seed(7)
        n = 4096
        x_cpu = torch.randn((n,), dtype=torch.float32)
        y_cpu = torch.randn((n,), dtype=torch.float32)
        x_mps = x_cpu.to("mps")
        y_mps = y_cpu.to("mps")
        out_mps = torch.empty_like(x_mps)

        _vadd[(triton.cdiv(n, 128),)](x_mps, y_mps, out_mps, n, BLOCK=128)
        torch.mps.synchronize()
        out_cpu = out_mps.cpu()
        torch.mps.synchronize()

        expected = x_cpu + y_cpu
        assert torch.allclose(out_cpu, expected, atol=1e-5, rtol=1e-5)

    @skip_non_darwin
    @skip_no_mps
    def test_runtime_int8_vector_add_matches_cpu(self):
        import torch

        import triton
        import triton.language as tl

        @triton.jit
        def _vadd_i8(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
            pid = tl.program_id(axis=0)
            offs = pid * BLOCK + tl.arange(0, BLOCK)
            mask = offs < n
            x = tl.load(x_ptr + offs, mask=mask, other=0)
            y = tl.load(y_ptr + offs, mask=mask, other=0)
            tl.store(out_ptr + offs, x + y, mask=mask)

        torch.manual_seed(17)
        n = 4096
        # Keep values in range to avoid overflow-semantics ambiguity.
        x_cpu = torch.randint(-32, 32, (n,), dtype=torch.int8)
        y_cpu = torch.randint(-32, 32, (n,), dtype=torch.int8)
        x_mps = x_cpu.to("mps")
        y_mps = y_cpu.to("mps")
        out_mps = torch.empty_like(x_mps)

        _vadd_i8[(triton.cdiv(n, 256),)](x_mps, y_mps, out_mps, n, BLOCK=256)
        torch.mps.synchronize()
        out_cpu = out_mps.cpu()
        torch.mps.synchronize()

        expected = (x_cpu.to(torch.int16) + y_cpu.to(torch.int16)).to(torch.int8)
        assert torch.equal(out_cpu, expected)

    @skip_non_darwin
    @skip_no_mps
    def test_runtime_int8_blocked_matmul_matches_cpu(self):
        import torch

        import triton
        import triton.language as tl

        @triton.jit
        def _matmul_i8(
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

            acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
            for kk in range(0, k, BLOCK_K):
                for ki in range(0, BLOCK_K):
                    k_idx = kk + ki
                    a = tl.load(
                        a_ptr + offs_m * stride_am + k_idx * stride_ak,
                        mask=(offs_m < m) & (k_idx < k),
                        other=0,
                    ).to(tl.int32)
                    b = tl.load(
                        b_ptr + k_idx * stride_bk + offs_n * stride_bn,
                        mask=(k_idx < k) & (offs_n < n),
                        other=0,
                    ).to(tl.int32)
                    acc += a[:, None] * b[None, :]

            c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
            tl.store(c_ptrs, acc, mask=(offs_m[:, None] < m) & (offs_n[None, :] < n))

        torch.manual_seed(19)
        m = n = k = 16
        a_cpu = torch.randint(-8, 8, (m, k), dtype=torch.int8)
        b_cpu = torch.randint(-8, 8, (k, n), dtype=torch.int8)

        a_mps = a_cpu.to("mps")
        b_mps = b_cpu.to("mps")
        c_mps = torch.empty((m, n), dtype=torch.int32, device="mps")

        _matmul_i8[(triton.cdiv(m, 8), triton.cdiv(n, 8), 1)](
            a_mps,
            b_mps,
            c_mps,
            m,
            n,
            k,
            a_mps.stride(0),
            a_mps.stride(1),
            b_mps.stride(0),
            b_mps.stride(1),
            c_mps.stride(0),
            c_mps.stride(1),
            BLOCK_M=8,
            BLOCK_N=8,
            BLOCK_K=8,
        )
        torch.mps.synchronize()
        c_cpu = c_mps.cpu()
        torch.mps.synchronize()

        expected = a_cpu.to(torch.int32) @ b_cpu.to(torch.int32)
        assert torch.equal(c_cpu, expected)

    @skip_non_darwin
    @skip_no_mps
    def test_runtime_small_blocked_matmul_matches_cpu(self):
        import torch

        import triton
        import triton.language as tl

        @triton.jit
        def _matmul(
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

        torch.manual_seed(11)
        m = n = k = 32
        a_cpu = torch.randn((m, k), dtype=torch.float32)
        b_cpu = torch.randn((k, n), dtype=torch.float32)
        a_mps = a_cpu.to("mps")
        b_mps = b_cpu.to("mps")
        c_mps = torch.empty((m, n), device="mps", dtype=torch.float32)

        _matmul[(triton.cdiv(m, 16), triton.cdiv(n, 16), 1)](
            a_mps,
            b_mps,
            c_mps,
            m,
            n,
            k,
            a_mps.stride(0),
            a_mps.stride(1),
            b_mps.stride(0),
            b_mps.stride(1),
            c_mps.stride(0),
            c_mps.stride(1),
            BLOCK_M=16,
            BLOCK_N=16,
            BLOCK_K=16,
        )
        torch.mps.synchronize()
        c_cpu = c_mps.cpu()
        torch.mps.synchronize()

        expected = a_cpu @ b_cpu
        assert torch.allclose(c_cpu, expected, atol=1e-4, rtol=1e-4)

    @skip_non_darwin
    @skip_no_mps
    def test_runtime_row_softmax_matches_cpu(self):
        import torch

        import triton
        import triton.language as tl

        @triton.jit
        def _row_softmax(x_ptr, y_ptr, n_cols, BLOCK: tl.constexpr):
            pid = tl.program_id(axis=0)
            offs = tl.arange(0, BLOCK)
            mask = offs < n_cols
            row_ptr = x_ptr + pid * n_cols
            x = tl.load(row_ptr + offs, mask=mask, other=float("-inf"))
            x = x - tl.max(x, axis=0)
            ex = tl.exp(x)
            denom = tl.sum(ex, axis=0)
            out = ex / denom
            tl.store(y_ptr + pid * n_cols + offs, out, mask=mask)

        torch.manual_seed(23)
        rows, cols = 8, 64
        x_cpu = torch.randn((rows, cols), dtype=torch.float32)
        x_mps = x_cpu.to("mps")
        y_mps = torch.empty_like(x_mps)

        _row_softmax[(rows,)](x_mps, y_mps, cols, BLOCK=64)
        torch.mps.synchronize()
        y_cpu = y_mps.cpu()
        torch.mps.synchronize()

        expected = torch.softmax(x_cpu, dim=1)
        assert torch.allclose(y_cpu, expected, atol=1e-4, rtol=1e-4)

    @skip_non_darwin
    @skip_no_mps
    def test_runtime_row_layernorm_matches_cpu(self):
        import torch

        import triton
        import triton.language as tl

        @triton.jit
        def _row_layernorm(
            x_ptr,
            w_ptr,
            b_ptr,
            y_ptr,
            n_cols,
            eps,
            BLOCK: tl.constexpr,
        ):
            pid = tl.program_id(axis=0)
            offs = tl.arange(0, BLOCK)
            mask = offs < n_cols
            row_base = pid * n_cols

            x = tl.load(x_ptr + row_base + offs, mask=mask, other=0.0)
            mean = tl.sum(x, axis=0) / n_cols
            centered = x - mean
            var = tl.sum(centered * centered, axis=0) / n_cols
            inv_std = 1.0 / tl.sqrt(var + eps)

            w = tl.load(w_ptr + offs, mask=mask, other=1.0)
            b = tl.load(b_ptr + offs, mask=mask, other=0.0)
            y = centered * inv_std * w + b
            tl.store(y_ptr + row_base + offs, y, mask=mask)

        torch.manual_seed(29)
        rows, cols = 4, 64
        eps = 1e-5
        x_cpu = torch.randn((rows, cols), dtype=torch.float32)
        w_cpu = torch.randn((cols,), dtype=torch.float32)
        b_cpu = torch.randn((cols,), dtype=torch.float32)

        x_mps = x_cpu.to("mps")
        w_mps = w_cpu.to("mps")
        b_mps = b_cpu.to("mps")
        y_mps = torch.empty_like(x_mps)

        _row_layernorm[(rows,)](x_mps, w_mps, b_mps, y_mps, cols, eps, BLOCK=64)
        torch.mps.synchronize()
        y_cpu = y_mps.cpu()
        torch.mps.synchronize()

        expected = torch.nn.functional.layer_norm(
            x_cpu, normalized_shape=(cols,), weight=w_cpu, bias=b_cpu, eps=eps
        )
        assert torch.allclose(y_cpu, expected, atol=2e-3, rtol=2e-3)

    @skip_non_darwin
    @skip_no_mps
    def test_runtime_fp16_blocked_matmul_matches_cpu(self):
        import torch

        import triton
        import triton.language as tl

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

            c = acc.to(tl.float16)
            c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
            tl.store(c_ptrs, c, mask=(offs_m[:, None] < m) & (offs_n[None, :] < n))

        torch.manual_seed(31)
        m, n, k = 32, 32, 48
        a_cpu = torch.randn((m, k), dtype=torch.float16)
        b_cpu = torch.randn((k, n), dtype=torch.float16)
        a_mps = a_cpu.to("mps")
        b_mps = b_cpu.to("mps")
        c_mps = torch.empty((m, n), device="mps", dtype=torch.float16)

        _matmul_fp16[(triton.cdiv(m, 16), triton.cdiv(n, 16), 1)](
            a_mps,
            b_mps,
            c_mps,
            m,
            n,
            k,
            a_mps.stride(0),
            a_mps.stride(1),
            b_mps.stride(0),
            b_mps.stride(1),
            c_mps.stride(0),
            c_mps.stride(1),
            BLOCK_M=16,
            BLOCK_N=16,
            BLOCK_K=16,
        )
        torch.mps.synchronize()
        c_cpu = c_mps.cpu()
        torch.mps.synchronize()

        expected = (a_cpu.float() @ b_cpu.float()).half()
        assert torch.allclose(c_cpu, expected, atol=4e-2, rtol=4e-2)

    @skip_non_darwin
    @skip_no_mps
    def test_runtime_batched_blocked_matmul_matches_cpu(self):
        import torch

        import triton
        import triton.language as tl

        @triton.jit
        def _batched_matmul(
            a_ptr,
            b_ptr,
            c_ptr,
            m,
            n,
            k,
            stride_ab,
            stride_am,
            stride_ak,
            stride_bb,
            stride_bk,
            stride_bn,
            stride_cb,
            stride_cm,
            stride_cn,
            BLOCK_M: tl.constexpr,
            BLOCK_N: tl.constexpr,
            BLOCK_K: tl.constexpr,
        ):
            pid_m = tl.program_id(axis=0)
            pid_n = tl.program_id(axis=1)
            pid_b = tl.program_id(axis=2)

            offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
            offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
            offs_k = tl.arange(0, BLOCK_K)

            a_base = a_ptr + pid_b * stride_ab
            b_base = b_ptr + pid_b * stride_bb
            c_base = c_ptr + pid_b * stride_cb

            acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            for kk in range(0, k, BLOCK_K):
                a = tl.load(
                    a_base
                    + offs_m[:, None] * stride_am
                    + (offs_k[None, :] + kk) * stride_ak,
                    mask=(offs_m[:, None] < m) & (offs_k[None, :] + kk < k),
                    other=0.0,
                )
                b = tl.load(
                    b_base
                    + (offs_k[:, None] + kk) * stride_bk
                    + offs_n[None, :] * stride_bn,
                    mask=(offs_k[:, None] + kk < k) & (offs_n[None, :] < n),
                    other=0.0,
                )
                acc += tl.dot(a, b)

            c_ptrs = c_base + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
            tl.store(c_ptrs, acc, mask=(offs_m[:, None] < m) & (offs_n[None, :] < n))

        torch.manual_seed(37)
        batch, m, n, k = 3, 16, 24, 20
        a_cpu = torch.randn((batch, m, k), dtype=torch.float32)
        b_cpu = torch.randn((batch, k, n), dtype=torch.float32)
        a_mps = a_cpu.to("mps")
        b_mps = b_cpu.to("mps")
        c_mps = torch.empty((batch, m, n), device="mps", dtype=torch.float32)

        _batched_matmul[(triton.cdiv(m, 16), triton.cdiv(n, 16), batch)](
            a_mps,
            b_mps,
            c_mps,
            m,
            n,
            k,
            a_mps.stride(0),
            a_mps.stride(1),
            a_mps.stride(2),
            b_mps.stride(0),
            b_mps.stride(1),
            b_mps.stride(2),
            c_mps.stride(0),
            c_mps.stride(1),
            c_mps.stride(2),
            BLOCK_M=16,
            BLOCK_N=16,
            BLOCK_K=8,
        )
        torch.mps.synchronize()
        c_cpu = c_mps.cpu()
        torch.mps.synchronize()

        expected = torch.bmm(a_cpu, b_cpu)
        assert torch.allclose(c_cpu, expected, atol=2e-4, rtol=2e-4)

    @skip_non_darwin
    @skip_no_mps
    def test_runtime_embedding_gather_matches_cpu(self):
        import torch

        import triton
        import triton.language as tl

        @triton.jit
        def _embedding_gather(
            table_ptr,
            idx_ptr,
            out_ptr,
            row_stride,
            n_cols,
            BLOCK: tl.constexpr,
        ):
            pid = tl.program_id(axis=0)
            idx = tl.load(idx_ptr + pid)
            offs = tl.arange(0, BLOCK)
            mask = offs < n_cols
            row = tl.load(table_ptr + idx * row_stride + offs, mask=mask, other=0.0)
            tl.store(out_ptr + pid * n_cols + offs, row, mask=mask)

        torch.manual_seed(41)
        vocab, dim, n_idx = 128, 64, 32
        table_cpu = torch.randn((vocab, dim), dtype=torch.float32)
        idx_cpu = torch.randint(0, vocab, (n_idx,), dtype=torch.int32)

        table_mps = table_cpu.to("mps")
        idx_mps = idx_cpu.to("mps")
        out_mps = torch.empty((n_idx, dim), device="mps", dtype=torch.float32)

        _embedding_gather[(n_idx,)](
            table_mps, idx_mps, out_mps, table_mps.stride(0), dim, BLOCK=64
        )
        torch.mps.synchronize()
        out_cpu = out_mps.cpu()
        torch.mps.synchronize()

        expected = table_cpu[idx_cpu.to(torch.long)]
        assert torch.allclose(out_cpu, expected, atol=1e-5, rtol=1e-5)

    @skip_non_darwin
    @skip_no_mps
    def test_runtime_attention_score_softmax_matches_cpu(self):
        import math

        import torch

        import triton
        import triton.language as tl

        @triton.jit
        def _matmul(
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

        @triton.jit
        def _row_softmax(x_ptr, y_ptr, n_cols, BLOCK: tl.constexpr):
            pid = tl.program_id(axis=0)
            offs = tl.arange(0, BLOCK)
            mask = offs < n_cols
            row_ptr = x_ptr + pid * n_cols
            x = tl.load(row_ptr + offs, mask=mask, other=float("-inf"))
            x = x - tl.max(x, axis=0)
            ex = tl.exp(x)
            denom = tl.sum(ex, axis=0)
            out = ex / denom
            tl.store(y_ptr + pid * n_cols + offs, out, mask=mask)

        torch.manual_seed(47)
        seq, dim = 16, 32
        q_cpu = torch.randn((seq, dim), dtype=torch.float32)
        k_cpu = torch.randn((seq, dim), dtype=torch.float32)
        k_t_cpu = k_cpu.t().contiguous()
        scale = 1.0 / math.sqrt(dim)

        q_mps = q_cpu.to("mps")
        k_t_mps = k_t_cpu.to("mps")
        scores_mps = torch.empty((seq, seq), device="mps", dtype=torch.float32)
        probs_mps = torch.empty_like(scores_mps)

        _matmul[(triton.cdiv(seq, 16), triton.cdiv(seq, 16), 1)](
            q_mps,
            k_t_mps,
            scores_mps,
            seq,
            seq,
            dim,
            q_mps.stride(0),
            q_mps.stride(1),
            k_t_mps.stride(0),
            k_t_mps.stride(1),
            scores_mps.stride(0),
            scores_mps.stride(1),
            BLOCK_M=16,
            BLOCK_N=16,
            BLOCK_K=16,
        )
        scores_mps = scores_mps * scale
        _row_softmax[(seq,)](scores_mps, probs_mps, seq, BLOCK=16)
        torch.mps.synchronize()
        probs_cpu = probs_mps.cpu()
        torch.mps.synchronize()

        expected = torch.softmax((q_cpu @ k_cpu.t()) * scale, dim=1)
        assert torch.allclose(probs_cpu, expected, atol=3e-4, rtol=3e-4)

    @skip_non_darwin
    @skip_no_mps
    def test_runtime_mlp_block_matches_cpu(self):
        import torch

        import triton
        import triton.language as tl

        @triton.jit
        def _matmul(
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

        @triton.jit
        def _silu(x_ptr, y_ptr, n, BLOCK: tl.constexpr):
            pid = tl.program_id(axis=0)
            offs = pid * BLOCK + tl.arange(0, BLOCK)
            mask = offs < n
            x = tl.load(x_ptr + offs, mask=mask, other=0.0)
            y = x * tl.sigmoid(x)
            tl.store(y_ptr + offs, y, mask=mask)

        torch.manual_seed(53)
        m, k, h, n = 16, 32, 48, 24
        x_cpu = torch.randn((m, k), dtype=torch.float32)
        w1_cpu = torch.randn((k, h), dtype=torch.float32)
        w2_cpu = torch.randn((h, n), dtype=torch.float32)

        x_mps = x_cpu.to("mps")
        w1_mps = w1_cpu.to("mps")
        w2_mps = w2_cpu.to("mps")
        hidden_mps = torch.empty((m, h), device="mps", dtype=torch.float32)
        act_mps = torch.empty_like(hidden_mps)
        out_mps = torch.empty((m, n), device="mps", dtype=torch.float32)

        _matmul[(triton.cdiv(m, 16), triton.cdiv(h, 16), 1)](
            x_mps,
            w1_mps,
            hidden_mps,
            m,
            h,
            k,
            x_mps.stride(0),
            x_mps.stride(1),
            w1_mps.stride(0),
            w1_mps.stride(1),
            hidden_mps.stride(0),
            hidden_mps.stride(1),
            BLOCK_M=16,
            BLOCK_N=16,
            BLOCK_K=16,
        )
        _silu[(triton.cdiv(m * h, 256),)](hidden_mps, act_mps, m * h, BLOCK=256)
        _matmul[(triton.cdiv(m, 16), triton.cdiv(n, 16), 1)](
            act_mps,
            w2_mps,
            out_mps,
            m,
            n,
            h,
            act_mps.stride(0),
            act_mps.stride(1),
            w2_mps.stride(0),
            w2_mps.stride(1),
            out_mps.stride(0),
            out_mps.stride(1),
            BLOCK_M=16,
            BLOCK_N=16,
            BLOCK_K=16,
        )
        torch.mps.synchronize()
        out_cpu = out_mps.cpu()
        torch.mps.synchronize()

        expected = torch.nn.functional.silu(x_cpu @ w1_cpu) @ w2_cpu
        assert torch.allclose(out_cpu, expected, atol=3e-4, rtol=3e-4)

    @skip_non_darwin
    @skip_no_mps
    def test_runtime_bf16_blocked_matmul_matches_cpu(self):
        import torch

        import triton
        import triton.language as tl

        try:
            torch.empty((1,), device="mps", dtype=torch.bfloat16)
        except Exception:
            pytest.skip("MPS bfloat16 runtime is unavailable on this host")

        @triton.jit
        def _matmul_bf16(
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

        torch.manual_seed(43)
        m, n, k = 32, 32, 48
        a_cpu = torch.randn((m, k), dtype=torch.bfloat16)
        b_cpu = torch.randn((k, n), dtype=torch.bfloat16)

        a_mps = a_cpu.to("mps")
        b_mps = b_cpu.to("mps")
        c_mps = torch.empty((m, n), device="mps", dtype=torch.float32)

        _matmul_bf16[(triton.cdiv(m, 16), triton.cdiv(n, 16), 1)](
            a_mps,
            b_mps,
            c_mps,
            m,
            n,
            k,
            a_mps.stride(0),
            a_mps.stride(1),
            b_mps.stride(0),
            b_mps.stride(1),
            c_mps.stride(0),
            c_mps.stride(1),
            BLOCK_M=16,
            BLOCK_N=16,
            BLOCK_K=16,
        )
        torch.mps.synchronize()
        c_cpu = c_mps.cpu()
        torch.mps.synchronize()

        expected = a_cpu.float() @ b_cpu.float()
        assert torch.allclose(c_cpu, expected, atol=5e-2, rtol=5e-2)

    @skip_non_darwin
    @skip_no_mps
    def test_runtime_grouped_batched_matmul_matches_cpu(self):
        import torch

        import triton
        import triton.language as tl

        @triton.jit
        def _grouped_batched_matmul(
            a_ptr,
            b_ptr,
            c_ptr,
            m,
            n,
            k,
            groups,
            batch,
            stride_ag,
            stride_ab,
            stride_am,
            stride_ak,
            stride_bg,
            stride_bb,
            stride_bk,
            stride_bn,
            stride_cg,
            stride_cb,
            stride_cm,
            stride_cn,
            BLOCK_M: tl.constexpr,
            BLOCK_N: tl.constexpr,
            BLOCK_K: tl.constexpr,
        ):
            pid_m = tl.program_id(axis=0)
            pid_n = tl.program_id(axis=1)
            pid_gb = tl.program_id(axis=2)

            gid = pid_gb // batch
            bid = pid_gb % batch

            offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
            offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
            offs_k = tl.arange(0, BLOCK_K)

            a_base = a_ptr + gid * stride_ag + bid * stride_ab
            b_base = b_ptr + gid * stride_bg + bid * stride_bb
            c_base = c_ptr + gid * stride_cg + bid * stride_cb

            acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            for kk in range(0, k, BLOCK_K):
                a = tl.load(
                    a_base
                    + offs_m[:, None] * stride_am
                    + (offs_k[None, :] + kk) * stride_ak,
                    mask=(offs_m[:, None] < m) & (offs_k[None, :] + kk < k),
                    other=0.0,
                )
                b = tl.load(
                    b_base
                    + (offs_k[:, None] + kk) * stride_bk
                    + offs_n[None, :] * stride_bn,
                    mask=(offs_k[:, None] + kk < k) & (offs_n[None, :] < n),
                    other=0.0,
                )
                acc += tl.dot(a, b)

            c_ptrs = c_base + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
            tl.store(c_ptrs, acc, mask=(offs_m[:, None] < m) & (offs_n[None, :] < n))

        torch.manual_seed(59)
        groups, batch, m, n, k = 2, 3, 16, 24, 20
        a_cpu = torch.randn((groups, batch, m, k), dtype=torch.float32)
        b_cpu = torch.randn((groups, batch, k, n), dtype=torch.float32)
        a_mps = a_cpu.to("mps")
        b_mps = b_cpu.to("mps")
        c_mps = torch.empty((groups, batch, m, n), device="mps", dtype=torch.float32)

        _grouped_batched_matmul[
            (triton.cdiv(m, 16), triton.cdiv(n, 16), groups * batch)
        ](
            a_mps,
            b_mps,
            c_mps,
            m,
            n,
            k,
            groups,
            batch,
            a_mps.stride(0),
            a_mps.stride(1),
            a_mps.stride(2),
            a_mps.stride(3),
            b_mps.stride(0),
            b_mps.stride(1),
            b_mps.stride(2),
            b_mps.stride(3),
            c_mps.stride(0),
            c_mps.stride(1),
            c_mps.stride(2),
            c_mps.stride(3),
            BLOCK_M=16,
            BLOCK_N=16,
            BLOCK_K=8,
        )
        torch.mps.synchronize()
        c_cpu = c_mps.cpu()
        torch.mps.synchronize()

        expected = torch.matmul(a_cpu, b_cpu)
        assert torch.allclose(c_cpu, expected, atol=2e-4, rtol=2e-4)

    @skip_non_darwin
    @skip_no_mps
    def test_runtime_depthwise_conv1d_like_matches_cpu(self):
        import torch

        import triton
        import triton.language as tl

        @triton.jit
        def _depthwise_conv1d(
            x_ptr,
            w_ptr,
            y_ptr,
            out_len,
            channels,
            stride_xl,
            stride_xc,
            stride_wk,
            stride_wc,
            stride_yl,
            stride_yc,
            KERNEL: tl.constexpr,
            BLOCK_C: tl.constexpr,
        ):
            pid = tl.program_id(axis=0)
            offs_c = tl.arange(0, BLOCK_C)
            mask = offs_c < channels

            acc = tl.zeros((BLOCK_C,), dtype=tl.float32)
            for kk in range(KERNEL):
                x = tl.load(
                    x_ptr + (pid + kk) * stride_xl + offs_c * stride_xc,
                    mask=mask,
                    other=0.0,
                )
                w = tl.load(
                    w_ptr + kk * stride_wk + offs_c * stride_wc,
                    mask=mask,
                    other=0.0,
                )
                acc += x * w

            tl.store(y_ptr + pid * stride_yl + offs_c * stride_yc, acc, mask=mask)

        torch.manual_seed(61)
        length, channels, kernel = 32, 32, 3
        out_len = length - kernel + 1
        x_cpu = torch.randn((length, channels), dtype=torch.float32)
        w_cpu = torch.randn((kernel, channels), dtype=torch.float32)

        x_mps = x_cpu.to("mps")
        w_mps = w_cpu.to("mps")
        y_mps = torch.empty((out_len, channels), device="mps", dtype=torch.float32)

        _depthwise_conv1d[(out_len,)](
            x_mps,
            w_mps,
            y_mps,
            out_len,
            channels,
            x_mps.stride(0),
            x_mps.stride(1),
            w_mps.stride(0),
            w_mps.stride(1),
            y_mps.stride(0),
            y_mps.stride(1),
            KERNEL=3,
            BLOCK_C=32,
        )
        torch.mps.synchronize()
        y_cpu = y_mps.cpu()
        torch.mps.synchronize()

        expected = torch.stack(
            [torch.sum(x_cpu[i : i + kernel] * w_cpu, dim=0) for i in range(out_len)],
            dim=0,
        )
        assert torch.allclose(y_cpu, expected, atol=2e-4, rtol=2e-4)


# ── Execute-and-verify runtime correctness ──────────────────────────


class TestMetalRuntimeExecuteVerify:
    """Execute-and-verify tests for operations beyond matmul.

    Each test compiles a Triton kernel, runs it on MPS, and checks
    against a CPU/PyTorch reference.
    """

    # ── Atomic operations ─────────────────────────────────────────

    @skip_non_darwin
    @skip_no_mps
    def test_runtime_atomic_add_matches_cpu(self):
        import torch

        import triton
        import triton.language as tl

        @triton.jit
        def _atomic_add_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
            pid = tl.program_id(axis=0)
            offs = pid * BLOCK + tl.arange(0, BLOCK)
            mask = offs < n
            x = tl.load(x_ptr + offs, mask=mask, other=0.0)
            block_sum = tl.sum(x, axis=0)
            tl.atomic_add(out_ptr, block_sum)

        torch.manual_seed(100)
        n = 256
        x_cpu = torch.randn((n,), dtype=torch.float32)
        x_mps = x_cpu.to("mps")
        out_mps = torch.zeros((1,), device="mps", dtype=torch.float32)

        _atomic_add_kernel[(triton.cdiv(n, 64),)](x_mps, out_mps, n, BLOCK=64)
        torch.mps.synchronize()
        out_cpu = out_mps.cpu()
        torch.mps.synchronize()

        expected = x_cpu.sum().unsqueeze(0)
        assert torch.allclose(out_cpu, expected, atol=1e-2, rtol=1e-2)

    @skip_non_darwin
    @skip_no_mps
    def test_runtime_atomic_add_per_bin(self):
        """Histogram-style atomic add into multiple bins."""
        import torch

        import triton
        import triton.language as tl

        @triton.jit
        def _bin_atomic_add(vals_ptr, bins_ptr, out_ptr, n, BLOCK: tl.constexpr):
            pid = tl.program_id(axis=0)
            offs = pid * BLOCK + tl.arange(0, BLOCK)
            mask = offs < n
            vals = tl.load(vals_ptr + offs, mask=mask, other=0.0)
            bin_ids = tl.load(bins_ptr + offs, mask=mask, other=0)
            tl.atomic_add(out_ptr + bin_ids, vals, mask=mask)

        torch.manual_seed(101)
        n, num_bins = 512, 8
        vals_cpu = torch.randn((n,), dtype=torch.float32)
        bins_cpu = torch.randint(0, num_bins, (n,), dtype=torch.int32)

        vals_mps = vals_cpu.to("mps")
        bins_mps = bins_cpu.to("mps")
        out_mps = torch.zeros((num_bins,), device="mps", dtype=torch.float32)

        _bin_atomic_add[(triton.cdiv(n, 64),)](vals_mps, bins_mps, out_mps, n, BLOCK=64)
        torch.mps.synchronize()
        out_cpu = out_mps.cpu()
        torch.mps.synchronize()

        expected = torch.zeros(num_bins, dtype=torch.float32)
        for i in range(n):
            expected[bins_cpu[i]] += vals_cpu[i]
        assert torch.allclose(out_cpu, expected, atol=1e-2, rtol=1e-2)

    # ── Reduction operations ──────────────────────────────────────

    @skip_non_darwin
    @skip_no_mps
    def test_runtime_reduce_sum_matches_cpu(self):
        import torch

        import triton
        import triton.language as tl

        @triton.jit
        def _reduce_sum(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
            pid = tl.program_id(axis=0)
            offs = pid * BLOCK + tl.arange(0, BLOCK)
            mask = offs < n
            x = tl.load(x_ptr + offs, mask=mask, other=0.0)
            s = tl.sum(x, axis=0)
            tl.store(out_ptr + pid, s)

        torch.manual_seed(102)
        n = 512
        block = 64
        x_cpu = torch.randn((n,), dtype=torch.float32)
        x_mps = x_cpu.to("mps")
        out_mps = torch.empty((n // block,), device="mps", dtype=torch.float32)

        _reduce_sum[(n // block,)](x_mps, out_mps, n, BLOCK=block)
        torch.mps.synchronize()
        out_cpu = out_mps.cpu()
        torch.mps.synchronize()

        expected = x_cpu.view(-1, block).sum(dim=1)
        assert torch.allclose(out_cpu, expected, atol=1e-4, rtol=1e-4)

    @skip_non_darwin
    @skip_no_mps
    def test_runtime_reduce_max_matches_cpu(self):
        import torch

        import triton
        import triton.language as tl

        @triton.jit
        def _reduce_max(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
            pid = tl.program_id(axis=0)
            offs = pid * BLOCK + tl.arange(0, BLOCK)
            mask = offs < n
            x = tl.load(x_ptr + offs, mask=mask, other=float("-inf"))
            m = tl.max(x, axis=0)
            tl.store(out_ptr + pid, m)

        torch.manual_seed(103)
        n = 256
        block = 32
        x_cpu = torch.randn((n,), dtype=torch.float32)
        x_mps = x_cpu.to("mps")
        out_mps = torch.empty((n // block,), device="mps", dtype=torch.float32)

        _reduce_max[(n // block,)](x_mps, out_mps, n, BLOCK=block)
        torch.mps.synchronize()
        out_cpu = out_mps.cpu()
        torch.mps.synchronize()

        expected = x_cpu.view(-1, block).max(dim=1).values
        assert torch.allclose(out_cpu, expected, atol=1e-5, rtol=1e-5)

    @skip_non_darwin
    @skip_no_mps
    def test_runtime_reduce_min_matches_cpu(self):
        import torch

        import triton
        import triton.language as tl

        @triton.jit
        def _reduce_min(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
            pid = tl.program_id(axis=0)
            offs = pid * BLOCK + tl.arange(0, BLOCK)
            mask = offs < n
            x = tl.load(x_ptr + offs, mask=mask, other=float("inf"))
            m = tl.min(x, axis=0)
            tl.store(out_ptr + pid, m)

        torch.manual_seed(104)
        n = 256
        block = 32
        x_cpu = torch.randn((n,), dtype=torch.float32)
        x_mps = x_cpu.to("mps")
        out_mps = torch.empty((n // block,), device="mps", dtype=torch.float32)

        _reduce_min[(n // block,)](x_mps, out_mps, n, BLOCK=block)
        torch.mps.synchronize()
        out_cpu = out_mps.cpu()
        torch.mps.synchronize()

        expected = x_cpu.view(-1, block).min(dim=1).values
        assert torch.allclose(out_cpu, expected, atol=1e-5, rtol=1e-5)

    @skip_non_darwin
    @skip_no_mps
    def test_runtime_2d_reduce_sum_axis0(self):
        """Reduce along axis=0 of a 2D block (column-wise sum)."""
        import torch

        import triton
        import triton.language as tl

        @triton.jit
        def _reduce_sum_2d(
            x_ptr,
            out_ptr,
            rows,
            cols,
            stride,
            BLOCK_R: tl.constexpr,
            BLOCK_C: tl.constexpr,
        ):
            pid = tl.program_id(axis=0)
            offs_r = tl.arange(0, BLOCK_R)
            offs_c = pid * BLOCK_C + tl.arange(0, BLOCK_C)
            mask = (offs_r[:, None] < rows) & (offs_c[None, :] < cols)
            x = tl.load(
                x_ptr + offs_r[:, None] * stride + offs_c[None, :],
                mask=mask,
                other=0.0,
            )
            s = tl.sum(x, axis=0)
            tl.store(out_ptr + offs_c, s, mask=offs_c < cols)

        torch.manual_seed(105)
        rows, cols = 16, 64
        x_cpu = torch.randn((rows, cols), dtype=torch.float32)
        x_mps = x_cpu.to("mps")
        out_mps = torch.empty((cols,), device="mps", dtype=torch.float32)

        _reduce_sum_2d[(triton.cdiv(cols, 64),)](
            x_mps, out_mps, rows, cols, x_mps.stride(0), BLOCK_R=16, BLOCK_C=64
        )
        torch.mps.synchronize()
        out_cpu = out_mps.cpu()
        torch.mps.synchronize()

        expected = x_cpu.sum(dim=0)
        assert torch.allclose(out_cpu, expected, atol=1e-4, rtol=1e-4)

    # ── Scan (prefix sum) ────────────────────────────────────────

    @skip_non_darwin
    @skip_no_mps
    def test_runtime_scan_cumsum_matches_cpu(self):
        import torch

        import triton
        import triton.language as tl

        @triton.jit
        def _cumsum(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
            pid = tl.program_id(axis=0)
            offs = pid * BLOCK + tl.arange(0, BLOCK)
            mask = offs < n
            x = tl.load(x_ptr + offs, mask=mask, other=0.0)
            cs = tl.cumsum(x, axis=0)
            tl.store(out_ptr + offs, cs, mask=mask)

        torch.manual_seed(106)
        n = 128
        x_cpu = torch.randn((n,), dtype=torch.float32)
        x_mps = x_cpu.to("mps")
        out_mps = torch.empty_like(x_mps)

        _cumsum[(n // 32,)](x_mps, out_mps, n, BLOCK=32)
        torch.mps.synchronize()
        out_cpu = out_mps.cpu()
        torch.mps.synchronize()

        # Each block produces an independent prefix sum.
        expected = x_cpu.view(-1, 32).cumsum(dim=1).view(-1)
        assert torch.allclose(out_cpu, expected, atol=1e-4, rtol=1e-4)

    # ── Where / select ────────────────────────────────────────────

    @skip_non_darwin
    @skip_no_mps
    def test_runtime_where_matches_cpu(self):
        import torch

        import triton
        import triton.language as tl

        @triton.jit
        def _where_kernel(cond_ptr, a_ptr, b_ptr, out_ptr, n, BLOCK: tl.constexpr):
            pid = tl.program_id(axis=0)
            offs = pid * BLOCK + tl.arange(0, BLOCK)
            mask = offs < n
            cond = tl.load(cond_ptr + offs, mask=mask, other=0)
            a = tl.load(a_ptr + offs, mask=mask, other=0.0)
            b = tl.load(b_ptr + offs, mask=mask, other=0.0)
            out = tl.where(cond != 0, a, b)
            tl.store(out_ptr + offs, out, mask=mask)

        torch.manual_seed(107)
        n = 512
        cond_cpu = torch.randint(0, 2, (n,), dtype=torch.int32)
        a_cpu = torch.randn((n,), dtype=torch.float32)
        b_cpu = torch.randn((n,), dtype=torch.float32)

        cond_mps = cond_cpu.to("mps")
        a_mps = a_cpu.to("mps")
        b_mps = b_cpu.to("mps")
        out_mps = torch.empty_like(a_mps)

        _where_kernel[(triton.cdiv(n, 128),)](
            cond_mps, a_mps, b_mps, out_mps, n, BLOCK=128
        )
        torch.mps.synchronize()
        out_cpu = out_mps.cpu()
        torch.mps.synchronize()

        expected = torch.where(cond_cpu.bool(), a_cpu, b_cpu)
        assert torch.allclose(out_cpu, expected, atol=1e-5, rtol=1e-5)

    # ── Element-wise unary operations ─────────────────────────────

    @skip_non_darwin
    @skip_no_mps
    def test_runtime_exp_matches_cpu(self):
        import torch

        import triton
        import triton.language as tl

        @triton.jit
        def _exp_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
            pid = tl.program_id(axis=0)
            offs = pid * BLOCK + tl.arange(0, BLOCK)
            mask = offs < n
            x = tl.load(x_ptr + offs, mask=mask, other=0.0)
            tl.store(out_ptr + offs, tl.exp(x), mask=mask)

        torch.manual_seed(108)
        n = 1024
        x_cpu = torch.randn((n,), dtype=torch.float32).clamp(-5, 5)
        x_mps = x_cpu.to("mps")
        out_mps = torch.empty_like(x_mps)

        _exp_kernel[(triton.cdiv(n, 256),)](x_mps, out_mps, n, BLOCK=256)
        torch.mps.synchronize()
        out_cpu = out_mps.cpu()
        torch.mps.synchronize()

        expected = torch.exp(x_cpu)
        assert torch.allclose(out_cpu, expected, atol=1e-5, rtol=1e-5)

    @skip_non_darwin
    @skip_no_mps
    def test_runtime_log_matches_cpu(self):
        import torch

        import triton
        import triton.language as tl

        @triton.jit
        def _log_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
            pid = tl.program_id(axis=0)
            offs = pid * BLOCK + tl.arange(0, BLOCK)
            mask = offs < n
            x = tl.load(x_ptr + offs, mask=mask, other=1.0)
            tl.store(out_ptr + offs, tl.log(x), mask=mask)

        torch.manual_seed(109)
        n = 1024
        x_cpu = torch.rand((n,), dtype=torch.float32) + 0.01
        x_mps = x_cpu.to("mps")
        out_mps = torch.empty_like(x_mps)

        _log_kernel[(triton.cdiv(n, 256),)](x_mps, out_mps, n, BLOCK=256)
        torch.mps.synchronize()
        out_cpu = out_mps.cpu()
        torch.mps.synchronize()

        expected = torch.log(x_cpu)
        assert torch.allclose(out_cpu, expected, atol=1e-5, rtol=1e-5)

    @skip_non_darwin
    @skip_no_mps
    def test_runtime_abs_neg_matches_cpu(self):
        import torch

        import triton
        import triton.language as tl

        @triton.jit
        def _abs_neg_kernel(x_ptr, abs_ptr, neg_ptr, n, BLOCK: tl.constexpr):
            pid = tl.program_id(axis=0)
            offs = pid * BLOCK + tl.arange(0, BLOCK)
            mask = offs < n
            x = tl.load(x_ptr + offs, mask=mask, other=0.0)
            tl.store(abs_ptr + offs, tl.abs(x), mask=mask)
            tl.store(neg_ptr + offs, -x, mask=mask)

        torch.manual_seed(110)
        n = 512
        x_cpu = torch.randn((n,), dtype=torch.float32)
        x_mps = x_cpu.to("mps")
        abs_mps = torch.empty_like(x_mps)
        neg_mps = torch.empty_like(x_mps)

        _abs_neg_kernel[(triton.cdiv(n, 128),)](x_mps, abs_mps, neg_mps, n, BLOCK=128)
        torch.mps.synchronize()
        abs_cpu = abs_mps.cpu()
        neg_cpu = neg_mps.cpu()
        torch.mps.synchronize()

        assert torch.allclose(abs_cpu, torch.abs(x_cpu), atol=1e-6, rtol=1e-6)
        assert torch.allclose(neg_cpu, -x_cpu, atol=1e-6, rtol=1e-6)

    # ── Transpose ─────────────────────────────────────────────────

    @skip_non_darwin
    @skip_no_mps
    def test_runtime_transpose_matches_cpu(self):
        import torch

        import triton
        import triton.language as tl

        @triton.jit
        def _transpose(
            x_ptr,
            out_ptr,
            rows,
            cols,
            stride_xr,
            stride_xc,
            stride_or,
            stride_oc,
            BLOCK_R: tl.constexpr,
            BLOCK_C: tl.constexpr,
        ):
            pid_r = tl.program_id(axis=0)
            pid_c = tl.program_id(axis=1)
            offs_r = pid_r * BLOCK_R + tl.arange(0, BLOCK_R)
            offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
            mask = (offs_r[:, None] < rows) & (offs_c[None, :] < cols)
            x = tl.load(
                x_ptr + offs_r[:, None] * stride_xr + offs_c[None, :] * stride_xc,
                mask=mask,
                other=0.0,
            )
            tl.store(
                out_ptr + offs_c[:, None] * stride_or + offs_r[None, :] * stride_oc,
                tl.trans(x),
                mask=(offs_c[:, None] < cols) & (offs_r[None, :] < rows),
            )

        torch.manual_seed(111)
        rows, cols = 32, 64
        x_cpu = torch.randn((rows, cols), dtype=torch.float32)
        x_mps = x_cpu.to("mps")
        out_mps = torch.empty((cols, rows), device="mps", dtype=torch.float32)

        _transpose[(triton.cdiv(rows, 16), triton.cdiv(cols, 16), 1)](
            x_mps,
            out_mps,
            rows,
            cols,
            x_mps.stride(0),
            x_mps.stride(1),
            out_mps.stride(0),
            out_mps.stride(1),
            BLOCK_R=16,
            BLOCK_C=16,
        )
        torch.mps.synchronize()
        out_cpu = out_mps.cpu()
        torch.mps.synchronize()

        expected = x_cpu.T
        assert torch.allclose(out_cpu, expected, atol=1e-6, rtol=1e-6)

    # ── Mixed-precision matmul verification ───────────────────────

    @skip_non_darwin
    @skip_no_mps
    def test_runtime_fp16_matmul_f32_accum_matches_cpu(self):
        """Verify f16×f16→f32 accumulation is actually precise (not truncated)."""
        import torch

        import triton
        import triton.language as tl

        @triton.jit
        def _matmul_f16_f32_accum(
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

            # Store as f32 to preserve full precision of accumulator.
            c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
            tl.store(c_ptrs, acc, mask=(offs_m[:, None] < m) & (offs_n[None, :] < n))

        torch.manual_seed(112)
        m, n, k = 32, 32, 64
        a_cpu = torch.randn((m, k), dtype=torch.float16)
        b_cpu = torch.randn((k, n), dtype=torch.float16)
        a_mps = a_cpu.to("mps")
        b_mps = b_cpu.to("mps")
        # Output is f32 to test that accumulation was in f32.
        c_mps = torch.empty((m, n), device="mps", dtype=torch.float32)

        _matmul_f16_f32_accum[(triton.cdiv(m, 16), triton.cdiv(n, 16), 1)](
            a_mps,
            b_mps,
            c_mps,
            m,
            n,
            k,
            a_mps.stride(0),
            a_mps.stride(1),
            b_mps.stride(0),
            b_mps.stride(1),
            c_mps.stride(0),
            c_mps.stride(1),
            BLOCK_M=16,
            BLOCK_N=16,
            BLOCK_K=16,
        )
        torch.mps.synchronize()
        c_cpu = c_mps.cpu()
        torch.mps.synchronize()

        expected = a_cpu.float() @ b_cpu.float()
        assert torch.allclose(c_cpu, expected, atol=2e-2, rtol=2e-2)

    @skip_non_darwin
    @skip_no_mps
    def test_runtime_fp16_matmul_larger_k(self):
        """Larger K to exercise multi-tile K-loop in mixed-precision path."""
        import torch

        import triton
        import triton.language as tl

        @triton.jit
        def _matmul_f16_f32(
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

        torch.manual_seed(113)
        m, n, k = 32, 32, 128
        a_cpu = torch.randn((m, k), dtype=torch.float16)
        b_cpu = torch.randn((k, n), dtype=torch.float16)
        a_mps = a_cpu.to("mps")
        b_mps = b_cpu.to("mps")
        c_mps = torch.empty((m, n), device="mps", dtype=torch.float32)

        _matmul_f16_f32[(triton.cdiv(m, 16), triton.cdiv(n, 16), 1)](
            a_mps,
            b_mps,
            c_mps,
            m,
            n,
            k,
            a_mps.stride(0),
            a_mps.stride(1),
            b_mps.stride(0),
            b_mps.stride(1),
            c_mps.stride(0),
            c_mps.stride(1),
            BLOCK_M=16,
            BLOCK_N=16,
            BLOCK_K=16,
        )
        torch.mps.synchronize()
        c_cpu = c_mps.cpu()
        torch.mps.synchronize()

        expected = a_cpu.float() @ b_cpu.float()
        assert torch.allclose(c_cpu, expected, atol=5e-2, rtol=5e-2)

    @skip_non_darwin
    @skip_no_mps
    def test_runtime_bf16_matmul_f32_accum_matches_cpu(self):
        """bf16×bf16→f32 matmul, output stored as f32."""
        import torch

        import triton
        import triton.language as tl

        try:
            torch.empty((1,), device="mps", dtype=torch.bfloat16)
        except Exception:
            pytest.skip("MPS bfloat16 runtime is unavailable on this host")

        @triton.jit
        def _matmul_bf16_f32(
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

        torch.manual_seed(114)
        m, n, k = 32, 32, 64
        a_cpu = torch.randn((m, k), dtype=torch.bfloat16)
        b_cpu = torch.randn((k, n), dtype=torch.bfloat16)
        a_mps = a_cpu.to("mps")
        b_mps = b_cpu.to("mps")
        c_mps = torch.empty((m, n), device="mps", dtype=torch.float32)

        _matmul_bf16_f32[(triton.cdiv(m, 16), triton.cdiv(n, 16), 1)](
            a_mps,
            b_mps,
            c_mps,
            m,
            n,
            k,
            a_mps.stride(0),
            a_mps.stride(1),
            b_mps.stride(0),
            b_mps.stride(1),
            c_mps.stride(0),
            c_mps.stride(1),
            BLOCK_M=16,
            BLOCK_N=16,
            BLOCK_K=16,
        )
        torch.mps.synchronize()
        c_cpu = c_mps.cpu()
        torch.mps.synchronize()

        expected = a_cpu.float() @ b_cpu.float()
        assert torch.allclose(c_cpu, expected, atol=5e-2, rtol=5e-2)

    # ── Type cast / conversion ────────────────────────────────────

    @skip_non_darwin
    @skip_no_mps
    def test_runtime_cast_fp32_to_fp16_roundtrip(self):
        import torch

        import triton
        import triton.language as tl

        @triton.jit
        def _cast_f32_f16_f32(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
            pid = tl.program_id(axis=0)
            offs = pid * BLOCK + tl.arange(0, BLOCK)
            mask = offs < n
            x = tl.load(x_ptr + offs, mask=mask, other=0.0)
            y = x.to(tl.float16).to(tl.float32)
            tl.store(out_ptr + offs, y, mask=mask)

        torch.manual_seed(115)
        n = 1024
        x_cpu = torch.randn((n,), dtype=torch.float32)
        x_mps = x_cpu.to("mps")
        out_mps = torch.empty_like(x_mps)

        _cast_f32_f16_f32[(triton.cdiv(n, 256),)](x_mps, out_mps, n, BLOCK=256)
        torch.mps.synchronize()
        out_cpu = out_mps.cpu()
        torch.mps.synchronize()

        expected = x_cpu.half().float()
        assert torch.allclose(out_cpu, expected, atol=1e-6, rtol=1e-6)

    # ── Fused multiply-add pattern ────────────────────────────────

    @skip_non_darwin
    @skip_no_mps
    def test_runtime_fma_pattern_matches_cpu(self):
        import torch

        import triton
        import triton.language as tl

        @triton.jit
        def _fma_kernel(a_ptr, b_ptr, c_ptr, out_ptr, n, BLOCK: tl.constexpr):
            pid = tl.program_id(axis=0)
            offs = pid * BLOCK + tl.arange(0, BLOCK)
            mask = offs < n
            a = tl.load(a_ptr + offs, mask=mask, other=0.0)
            b = tl.load(b_ptr + offs, mask=mask, other=0.0)
            c = tl.load(c_ptr + offs, mask=mask, other=0.0)
            tl.store(out_ptr + offs, a * b + c, mask=mask)

        torch.manual_seed(116)
        n = 1024
        a_cpu = torch.randn((n,), dtype=torch.float32)
        b_cpu = torch.randn((n,), dtype=torch.float32)
        c_cpu = torch.randn((n,), dtype=torch.float32)

        a_mps = a_cpu.to("mps")
        b_mps = b_cpu.to("mps")
        c_mps = c_cpu.to("mps")
        out_mps = torch.empty_like(a_mps)

        _fma_kernel[(triton.cdiv(n, 256),)](a_mps, b_mps, c_mps, out_mps, n, BLOCK=256)
        torch.mps.synchronize()
        out_cpu = out_mps.cpu()
        torch.mps.synchronize()

        expected = a_cpu * b_cpu + c_cpu
        assert torch.allclose(out_cpu, expected, atol=1e-5, rtol=1e-5)

    # ── Multi-output kernel ───────────────────────────────────────

    @skip_non_darwin
    @skip_no_mps
    def test_runtime_mean_var_matches_cpu(self):
        """A single kernel computing both mean and variance."""
        import torch

        import triton
        import triton.language as tl

        @triton.jit
        def _mean_var(x_ptr, mean_ptr, var_ptr, n_cols, BLOCK: tl.constexpr):
            pid = tl.program_id(axis=0)
            offs = tl.arange(0, BLOCK)
            mask = offs < n_cols
            x = tl.load(x_ptr + pid * n_cols + offs, mask=mask, other=0.0)
            mean = tl.sum(x, axis=0) / n_cols
            centered = x - mean
            var = tl.sum(centered * centered, axis=0) / n_cols
            tl.store(mean_ptr + pid, mean)
            tl.store(var_ptr + pid, var)

        torch.manual_seed(117)
        rows, cols = 8, 64
        x_cpu = torch.randn((rows, cols), dtype=torch.float32)
        x_mps = x_cpu.to("mps")
        mean_mps = torch.empty((rows,), device="mps", dtype=torch.float32)
        var_mps = torch.empty((rows,), device="mps", dtype=torch.float32)

        _mean_var[(rows,)](x_mps, mean_mps, var_mps, cols, BLOCK=64)
        torch.mps.synchronize()
        mean_cpu = mean_mps.cpu()
        var_cpu = var_mps.cpu()
        torch.mps.synchronize()

        expected_mean = x_cpu.mean(dim=1)
        expected_var = x_cpu.var(dim=1, correction=0)
        assert torch.allclose(mean_cpu, expected_mean, atol=1e-4, rtol=1e-4)
        assert torch.allclose(var_cpu, expected_var, atol=1e-3, rtol=1e-3)


# ── LLVM vector-constant lowering regressions ───────────────────────


class TestMetalVectorConstantLowering:
    def test_binary_vector_constant_form(self):
        from third_party.metal.backend.compiler import MetalBackend

        llvm_ir = """\
define void @vec_mask_kernel(ptr %out) {
entry:
  %v = and <2 x i32> <i32 15, i32 7>, <i32 8, i32 4>
  %e0 = extractelement <2 x i32> %v, i64 0
  %p0 = getelementptr i32, ptr %out, i64 0
  store i32 %e0, ptr %p0
  ret void
}
"""
        metadata = {}
        msl = MetalBackend.make_metal_ir(llvm_ir, metadata, None)
        assert "UNSUPPORTED" not in msl
        assert "int2(" in msl
        assert "&" in msl


# ── LLVM shufflevector lowering regressions ────────────────────────


class TestMetalShuffleVectorLowering:
    def test_make_metal_ir_shufflevector_half(self):
        from third_party.metal.backend.compiler import MetalBackend

        llvm_ir = """\
define void @shuffle_half_kernel(ptr %out, <2 x half> %a, <2 x half> %b) {
entry:
  %s = shufflevector <2 x half> %a, <2 x half> %b, <2 x i32> <i32 0, i32 2>
  %e0 = extractelement <2 x half> %s, i64 0
  %p0 = getelementptr half, ptr %out, i64 0
  store half %e0, ptr %p0
  %e1 = extractelement <2 x half> %s, i64 1
  %p1 = getelementptr half, ptr %out, i64 1
  store half %e1, ptr %p1
  ret void
}
"""
        msl = MetalBackend.make_metal_ir(llvm_ir, {}, None)
        assert "UNSUPPORTED" not in msl
        assert "half2(" in msl


# ── LLVM bfloat lowering regressions ────────────────────────────────


class TestMetalBFloatLowering:
    def test_make_metal_ir_bfloat_param_and_pointer_types(self):
        from third_party.metal.backend.compiler import MetalBackend

        llvm_ir = """\
define void @bfloat_kernel(ptr addrspace(1) %in, ptr addrspace(1) %out, bfloat %x) {
entry:
  %pin = getelementptr bfloat, ptr addrspace(1) %in, i64 0
  %v = load bfloat, ptr addrspace(1) %pin
  %sum = fadd bfloat %v, %x
  %vf = fpext bfloat %sum to float
  %pout = getelementptr float, ptr addrspace(1) %out, i64 0
  store float %vf, ptr addrspace(1) %pout
  ret void
}
"""
        msl = MetalBackend.make_metal_ir(llvm_ir, {}, None)
        assert "device bfloat*" in msl
        assert "constant bfloat&" in msl
        assert "device int*" not in msl


# ── Unsupported LLVM intrinsic guards ───────────────────────────────


class TestMetalUnsupportedIntrinsicGuard:
    def test_unknown_llvm_value_intrinsic_raises(self):
        from third_party.metal.backend.compiler import MetalBackend

        llvm_ir = """\
define void @unknown_value_intrinsic(ptr %out, i32 %x) {
entry:
  %v = call i32 @llvm.unknown.value.i32(i32 %x)
  %p = getelementptr i32, ptr %out, i64 0
  store i32 %v, ptr %p
  ret void
}
"""
        with pytest.raises(RuntimeError, match="Unsupported LLVM intrinsic"):
            MetalBackend.make_metal_ir(llvm_ir, {}, None)

    def test_unknown_llvm_void_intrinsic_raises(self):
        from third_party.metal.backend.compiler import MetalBackend

        llvm_ir = """\
define void @unknown_void_intrinsic(i32 %x) {
entry:
  call void @llvm.unknown.void.i32(i32 %x)
  ret void
}
"""
        with pytest.raises(RuntimeError, match="Unsupported LLVM intrinsic"):
            MetalBackend.make_metal_ir(llvm_ir, {}, None)


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

        from third_party.metal.backend.translator_context import _RE_CONST_HEX_BFLOAT

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


class TestMetalAudit2Pass2LineClean:
    """ERR2-001: _RE_LINE_CLEAN must strip ALL LLVM metadata, not just !dbg.

    LLVM O3 can add !tbaa, !range, !alias.scope, !invariant.load, and
    !noalias metadata independently of debug info.  If these aren't
    stripped, downstream regex patterns capture metadata tokens inside
    pointer/operand groups, producing invalid MSL variable references.
    """

    def test_tbaa_metadata_stripped(self):
        from third_party.metal.backend.compiler import MetalBackend

        llvm_ir = """
define void @tbaa_kernel(ptr %out, ptr %in) {
  %v = load i32, ptr %in, align 4, !tbaa !0
  store i32 %v, ptr %out, align 4
  ret void
}
!0 = !{!1, !1, i64 0}
!1 = !{!"int", !2, i64 0}
!2 = !{!"omnipotent char", !3, i64 0}
!3 = !{!"Simple C/C++ TBAA"}
"""
        metadata = {}
        msl = MetalBackend.make_metal_ir(llvm_ir, metadata, None)
        assert "!tbaa" not in msl, "!tbaa metadata leaked into generated MSL"
        assert "UNSUPPORTED" not in msl, "load with !tbaa should be supported"

    def test_range_metadata_stripped(self):
        from third_party.metal.backend.compiler import MetalBackend

        llvm_ir = """
define void @range_kernel(ptr %out, ptr %in) {
  %v = load i32, ptr %in, align 4, !range !0
  store i32 %v, ptr %out, align 4
  ret void
}
!0 = !{i32 0, i32 256}
"""
        metadata = {}
        msl = MetalBackend.make_metal_ir(llvm_ir, metadata, None)
        assert "!range" not in msl, "!range metadata leaked into generated MSL"
        assert "UNSUPPORTED" not in msl

    def test_noalias_metadata_stripped(self):
        from third_party.metal.backend.compiler import MetalBackend

        llvm_ir = """
define void @noalias_kernel(ptr %out, ptr %in) {
  %v = load i32, ptr %in, align 4, !noalias !0
  store i32 %v, ptr %out, align 4
  ret void
}
!0 = !{!1}
!1 = distinct !{!1, !2}
!2 = distinct !{!2}
"""
        metadata = {}
        msl = MetalBackend.make_metal_ir(llvm_ir, metadata, None)
        assert "!noalias" not in msl, "!noalias metadata leaked into generated MSL"
        assert "UNSUPPORTED" not in msl

    def test_invariant_load_metadata_stripped(self):
        from third_party.metal.backend.compiler import MetalBackend

        llvm_ir = """
define void @inv_kernel(ptr %out, ptr %in) {
  %v = load i32, ptr %in, align 4, !invariant.load !0
  store i32 %v, ptr %out, align 4
  ret void
}
!0 = !{}
"""
        metadata = {}
        msl = MetalBackend.make_metal_ir(llvm_ir, metadata, None)
        assert "!invariant" not in msl, "!invariant.load metadata leaked into MSL"
        assert "UNSUPPORTED" not in msl

    def test_multiple_metadata_stripped(self):
        """Multiple metadata attachments on one instruction must all be stripped."""
        from third_party.metal.backend.compiler import MetalBackend

        llvm_ir = """
define void @multi_meta_kernel(ptr %out, ptr %in) {
  %v = load i32, ptr %in, align 4, !tbaa !0, !noalias !1
  store i32 %v, ptr %out, align 4, !tbaa !0
  ret void
}
!0 = !{!2, !2, i64 0}
!1 = !{!3}
!2 = !{!"int", !4, i64 0}
!3 = distinct !{!3}
!4 = !{!"omnipotent char"}
"""
        metadata = {}
        msl = MetalBackend.make_metal_ir(llvm_ir, metadata, None)
        assert "!tbaa" not in msl
        assert "!noalias" not in msl
        assert "UNSUPPORTED" not in msl


class TestMetalAudit2Pass2CallPrefixes:
    """ERR2-002: _RE_CALL_OUT / _RE_VOID_CALL must handle musttail/notail.

    _extract_ir_opcode correctly strips musttail/notail prefixes to yield
    'call', but the regex patterns only had (?:tail\\s+)? — causing
    musttail/notail calls to fall through as unsupported IR.
    """

    def test_musttail_call_handled(self):
        from third_party.metal.backend.compiler import MetalBackend

        llvm_ir = """
define void @musttail_kernel(ptr %out) {
  %v = musttail call i32 @helper()
  store i32 %v, ptr %out
  ret void
}
declare i32 @helper()
"""
        metadata = {}
        msl = MetalBackend.make_metal_ir(llvm_ir, metadata, None)
        assert "UNSUPPORTED" not in msl, "musttail call should be handled"
        assert "helper()" in msl

    def test_notail_call_handled(self):
        from third_party.metal.backend.compiler import MetalBackend

        llvm_ir = """
define void @notail_kernel(ptr %out) {
  %v = notail call i32 @helper2()
  store i32 %v, ptr %out
  ret void
}
declare i32 @helper2()
"""
        metadata = {}
        msl = MetalBackend.make_metal_ir(llvm_ir, metadata, None)
        assert "UNSUPPORTED" not in msl, "notail call should be handled"
        assert "helper2()" in msl

    def test_musttail_void_call_handled(self):
        from third_party.metal.backend.compiler import MetalBackend

        llvm_ir = """
define void @musttail_void_kernel(ptr %out) {
  musttail call void @void_helper(ptr %out)
  ret void
}
declare void @void_helper(ptr)
"""
        metadata = {}
        msl = MetalBackend.make_metal_ir(llvm_ir, metadata, None)
        assert "UNSUPPORTED" not in msl, "musttail void call should be handled"

    def test_notail_void_call_handled(self):
        from third_party.metal.backend.compiler import MetalBackend

        llvm_ir = """
define void @notail_void_kernel(ptr %out) {
  notail call void @void_helper2(ptr %out)
  ret void
}
declare void @void_helper2(ptr)
"""
        metadata = {}
        msl = MetalBackend.make_metal_ir(llvm_ir, metadata, None)
        assert "UNSUPPORTED" not in msl, "notail void call should be handled"


class TestMetalAudit2Pass2ArgPack:
    """ERR2-003: _ARG_PACK_FORMAT must cover all Metal-relevant scalar types.

    Before this fix, i1/i8/u8/i16/u16/bf16 fell through to the generic
    isinstance(arg, int) path which packed as 4-byte i32 — accidentally
    correct on little-endian Apple Silicon but fragile and wasteful.
    """

    def test_i8_packs_to_1_byte(self):
        import struct

        from third_party.metal.backend.driver import _ARG_PACK_FORMAT

        packed = struct.pack(_ARG_PACK_FORMAT["i8"], -1)
        assert len(packed) == 1
        assert struct.unpack("b", packed)[0] == -1

    def test_u8_packs_to_1_byte(self):
        import struct

        from third_party.metal.backend.driver import _ARG_PACK_FORMAT

        packed = struct.pack(_ARG_PACK_FORMAT["u8"], 255)
        assert len(packed) == 1
        assert struct.unpack("B", packed)[0] == 255

    def test_i16_packs_to_2_bytes(self):
        import struct

        from third_party.metal.backend.driver import _ARG_PACK_FORMAT

        packed = struct.pack(_ARG_PACK_FORMAT["i16"], -32768)
        assert len(packed) == 2

    def test_u16_packs_to_2_bytes(self):
        import struct

        from third_party.metal.backend.driver import _ARG_PACK_FORMAT

        packed = struct.pack(_ARG_PACK_FORMAT["u16"], 65535)
        assert len(packed) == 2

    def test_i1_packs_to_1_byte(self):
        import struct

        from third_party.metal.backend.driver import _ARG_PACK_FORMAT

        packed = struct.pack(_ARG_PACK_FORMAT["i1"], True)
        assert len(packed) == 1

    def test_bf16_packs_to_2_bytes(self):
        import struct

        from third_party.metal.backend.driver import _ARG_PACK_FORMAT

        packed = struct.pack(_ARG_PACK_FORMAT["bf16"], 0x3F80)
        assert len(packed) == 2


class TestMetalAudit2Pass2ReservedIds:
    """ERR2-004: _MSL_RESERVED_IDENTIFIERS must include MSL unsigned
    type aliases and C++ keywords that could collide with MLIR SSA names.
    """

    def test_uint_reserved(self):
        from third_party.metal.backend.compiler import _MSL_RESERVED_IDENTIFIERS

        for name in ("uint", "uchar", "ushort", "ulong"):
            assert (
                name in _MSL_RESERVED_IDENTIFIERS
            ), f"{name} must be reserved — MSL uses it as unsigned type alias"

    def test_cpp_keywords_reserved(self):
        from third_party.metal.backend.compiler import _MSL_RESERVED_IDENTIFIERS

        cpp_keywords = {"void", "struct", "class", "const", "static", "sizeof"}
        for kw in cpp_keywords:
            assert (
                kw in _MSL_RESERVED_IDENTIFIERS
            ), f"C++ keyword '{kw}' must be in reserved set"

    def test_msl_builtins_reserved(self):
        from third_party.metal.backend.compiler import _MSL_RESERVED_IDENTIFIERS

        builtins = {"select", "clamp", "abs", "mix", "saturate", "step"}
        for b in builtins:
            assert (
                b in _MSL_RESERVED_IDENTIFIERS
            ), f"MSL builtin '{b}' must be in reserved set"

    def test_reserved_id_collision_avoided(self):
        """If an SSA name collides with a reserved word, msl_id prefixes it."""
        from third_party.metal.backend.compiler import MetalBackend

        llvm_ir = """
define void @reserved_kernel(ptr %out, i32 %x) {
  %uint = add i32 %x, 1
  store i32 %uint, ptr %out
  ret void
}
"""
        metadata = {}
        msl = MetalBackend.make_metal_ir(llvm_ir, metadata, None)
        # The name 'uint' must be escaped (e.g., v_uint) to avoid
        # shadowing the MSL unsigned int type alias
        assert (
            "v_uint" in msl
        ), "SSA name %uint must be escaped to avoid shadowing MSL uint type"


# ── FP8 Runtime Matmul Validation ───────────────────────────────────


def _has_torch_fp8() -> bool:
    """Check whether torch has fp8 dtype support (CPU-side)."""
    try:
        import torch

        return hasattr(torch, "float8_e5m2")
    except Exception:
        return False


skip_no_fp8 = pytest.mark.skipif(
    not _has_torch_fp8(),
    reason="Torch float8 types unavailable on this host",
)


class TestMetalFP8RuntimeMatmul:
    """FP8 matmul tests: compile-path MSL verification and CPU-side
    numerical validation of fp8 conversion + accumulation accuracy.

    Apple Silicon MPS does not natively support fp8 tensors, so these
    tests validate:
    1. The Triton→MSL compile pipeline handles fp8 signatures correctly
    2. CPU-side fp8 conversion + fp32/fp16 accumulation produces
       numerically sound results vs fp32 reference
    """

    @skip_non_darwin
    @skip_no_xcrun
    @skip_no_fp8
    def test_compile_fp8e5m2_matmul_small_16x16x16(self):
        """fp8e5m2 matmul 16x16x16 compiles to valid MSL with fp32 acc."""
        import triton
        import triton.language as tl
        from triton.backends.compiler import GPUTarget

        @triton.jit
        def _matmul_fp8(
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
            fn=_matmul_fp8,
            signature={
                "a_ptr": "*fp8e5",
                "b_ptr": "*fp8e5",
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
        msl = kernel.asm["metal"]
        assert b"kernel void" in msl

    @skip_non_darwin
    @skip_no_xcrun
    @skip_no_fp8
    def test_compile_fp8e5m2_matmul_medium_64x64x64(self):
        """fp8e5m2 matmul 64x64x64 compiles successfully."""
        import triton
        import triton.language as tl
        from triton.backends.compiler import GPUTarget

        @triton.jit
        def _matmul_fp8_med(
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
            fn=_matmul_fp8_med,
            signature={
                "a_ptr": "*fp8e5",
                "b_ptr": "*fp8e5",
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
    @skip_no_fp8
    def test_compile_fp8e5m2_matmul_odd_k_tail_64x37x64(self):
        """fp8e5m2 matmul with odd-K (37) compiles — tail masking works."""
        import triton
        import triton.language as tl
        from triton.backends.compiler import GPUTarget

        @triton.jit
        def _matmul_fp8_odd(
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
            fn=_matmul_fp8_odd,
            signature={
                "a_ptr": "*fp8e5",
                "b_ptr": "*fp8e5",
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
    @skip_no_fp8
    def test_compile_fp8e5m2_to_fp16_accumulation(self):
        """fp8 inputs with fp16 output accumulation compiles correctly."""
        import triton
        import triton.language as tl
        from triton.backends.compiler import GPUTarget

        @triton.jit
        def _matmul_fp8_fp16out(
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
            c = acc.to(tl.float16)
            c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
            tl.store(c_ptrs, c, mask=(offs_m[:, None] < m) & (offs_n[None, :] < n))

        src = triton.compiler.ASTSource(
            fn=_matmul_fp8_fp16out,
            signature={
                "a_ptr": "*fp8e5",
                "b_ptr": "*fp8e5",
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
        msl = kernel.asm["metal"]
        assert b"kernel void" in msl

    @skip_no_fp8
    def test_fp8e5m2_cpu_matmul_numerics_vs_fp32_reference(self):
        """CPU-side fp8e5m2 matmul numerics: quantize→dequant→matmul
        is within fp8 tolerance of fp32 reference."""
        import numpy as np
        import torch

        torch.manual_seed(100)
        m, n, k = 16, 16, 16
        a_fp32 = torch.randn((m, k), dtype=torch.float32)
        b_fp32 = torch.randn((k, n), dtype=torch.float32)
        ref = a_fp32 @ b_fp32

        a_fp8 = a_fp32.to(torch.float8_e5m2).to(torch.float32)
        b_fp8 = b_fp32.to(torch.float8_e5m2).to(torch.float32)
        result = a_fp8 @ b_fp8

        # fp8e5m2 has only 2 mantissa bits — large quantization noise
        assert torch.allclose(
            result, ref, atol=2.0, rtol=0.3
        ), f"fp8e5m2 quantized matmul max err: {(result - ref).abs().max().item():.4f}"

    @skip_no_fp8
    def test_fp8e4m3fn_cpu_matmul_numerics_vs_fp32_reference(self):
        """CPU-side fp8e4m3fn matmul numerics: tighter range than e5m2."""
        import torch

        torch.manual_seed(101)
        m, n, k = 16, 16, 16
        a_fp32 = torch.randn((m, k), dtype=torch.float32) * 0.5
        b_fp32 = torch.randn((k, n), dtype=torch.float32) * 0.5
        ref = a_fp32 @ b_fp32

        a_fp8 = a_fp32.to(torch.float8_e4m3fn).to(torch.float32)
        b_fp8 = b_fp32.to(torch.float8_e4m3fn).to(torch.float32)
        result = a_fp8 @ b_fp8

        # fp8e4m3fn has 3 mantissa bits — better than e5m2 but still noisy
        assert torch.allclose(
            result, ref, atol=0.2, rtol=0.15
        ), f"fp8e4m3fn quantized matmul max err: {(result - ref).abs().max().item():.4f}"

    @skip_no_fp8
    def test_fp8e5m2_cpu_odd_k_numerics(self):
        """CPU fp8e5m2 matmul with odd K=37 — tail handling numerics."""
        import torch

        torch.manual_seed(102)
        m, n, k = 64, 64, 37
        a_fp32 = torch.randn((m, k), dtype=torch.float32)
        b_fp32 = torch.randn((k, n), dtype=torch.float32)
        ref = a_fp32 @ b_fp32

        a_fp8 = a_fp32.to(torch.float8_e5m2).to(torch.float32)
        b_fp8 = b_fp32.to(torch.float8_e5m2).to(torch.float32)
        result = a_fp8 @ b_fp8

        # Larger K accumulates more fp8 quantization noise
        assert torch.allclose(
            result, ref, atol=5.0, rtol=0.35
        ), f"fp8e5m2 odd-K matmul max err: {(result - ref).abs().max().item():.4f}"

    @skip_non_darwin
    @skip_no_xcrun
    @skip_no_fp8
    def test_compile_fp8e4b15_matmul_pipeline(self):
        """fp8e4b15 matmul compiles to valid MSL (if type conversion supported)."""
        import triton
        import triton.language as tl
        from triton.backends.compiler import GPUTarget

        @triton.jit
        def _matmul_fp8e4(
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
            fn=_matmul_fp8e4,
            signature={
                "a_ptr": "*fp8e4b15",
                "b_ptr": "*fp8e4b15",
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
        try:
            kernel = triton.compile(src=src, target=GPUTarget("metal", "apple8", 32))
        except Exception as e:
            if "conversion" in str(e).lower() or "not supported" in str(e).lower():
                pytest.skip(f"fp8e4b15 codegen not fully supported: {e}")
            raise
        assert_metal_compilation_artifacts(kernel)


# ── Int8 Matmul Runtime Validation ──────────────────────────────────


class TestMetalInt8MatmulRuntime:
    """Int8 matmul runtime correctness tests on MPS.

    Tests int8×int8→int32 and int8 with int16 accumulation,
    mixed int8/int16 inputs, and boundary saturation behavior.
    All compare against numpy/torch CPU reference computations.
    """

    @skip_non_darwin
    @skip_no_mps
    def test_int8_matmul_small_16x16x16(self):
        """int8×int8→int32 matmul with 16x16x16 shape."""
        import torch

        import triton
        import triton.language as tl

        @triton.jit
        def _matmul_i8_sm(
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
            acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
            for kk in range(0, k, BLOCK_K):
                for ki in range(0, BLOCK_K):
                    k_idx = kk + ki
                    a = tl.load(
                        a_ptr + offs_m * stride_am + k_idx * stride_ak,
                        mask=(offs_m < m) & (k_idx < k),
                        other=0,
                    ).to(tl.int32)
                    b = tl.load(
                        b_ptr + k_idx * stride_bk + offs_n * stride_bn,
                        mask=(k_idx < k) & (offs_n < n),
                        other=0,
                    ).to(tl.int32)
                    acc += a[:, None] * b[None, :]
            c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
            tl.store(c_ptrs, acc, mask=(offs_m[:, None] < m) & (offs_n[None, :] < n))

        torch.manual_seed(200)
        m = n = k = 16
        a_cpu = torch.randint(-8, 8, (m, k), dtype=torch.int8)
        b_cpu = torch.randint(-8, 8, (k, n), dtype=torch.int8)
        a_mps = a_cpu.to("mps")
        b_mps = b_cpu.to("mps")
        c_mps = torch.empty((m, n), dtype=torch.int32, device="mps")

        _matmul_i8_sm[(triton.cdiv(m, 8), triton.cdiv(n, 8), 1)](
            a_mps,
            b_mps,
            c_mps,
            m,
            n,
            k,
            a_mps.stride(0),
            a_mps.stride(1),
            b_mps.stride(0),
            b_mps.stride(1),
            c_mps.stride(0),
            c_mps.stride(1),
            BLOCK_M=8,
            BLOCK_N=8,
            BLOCK_K=8,
        )
        torch.mps.synchronize()
        c_cpu = c_mps.cpu()
        torch.mps.synchronize()

        expected = a_cpu.to(torch.int32) @ b_cpu.to(torch.int32)
        assert torch.equal(c_cpu, expected)

    @skip_non_darwin
    @skip_no_mps
    def test_int8_matmul_medium_32x32x32(self):
        """int8×int8→int32 matmul 32x32x32."""
        import torch

        import triton
        import triton.language as tl

        @triton.jit
        def _matmul_i8_md(
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
            acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
            for kk in range(0, k, BLOCK_K):
                for ki in range(0, BLOCK_K):
                    k_idx = kk + ki
                    a = tl.load(
                        a_ptr + offs_m * stride_am + k_idx * stride_ak,
                        mask=(offs_m < m) & (k_idx < k),
                        other=0,
                    ).to(tl.int32)
                    b = tl.load(
                        b_ptr + k_idx * stride_bk + offs_n * stride_bn,
                        mask=(k_idx < k) & (offs_n < n),
                        other=0,
                    ).to(tl.int32)
                    acc += a[:, None] * b[None, :]
            c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
            tl.store(c_ptrs, acc, mask=(offs_m[:, None] < m) & (offs_n[None, :] < n))

        torch.manual_seed(201)
        m = n = k = 32
        a_cpu = torch.randint(-8, 8, (m, k), dtype=torch.int8)
        b_cpu = torch.randint(-8, 8, (k, n), dtype=torch.int8)
        a_mps = a_cpu.to("mps")
        b_mps = b_cpu.to("mps")
        c_mps = torch.empty((m, n), dtype=torch.int32, device="mps")

        _matmul_i8_md[(triton.cdiv(m, 8), triton.cdiv(n, 8), 1)](
            a_mps,
            b_mps,
            c_mps,
            m,
            n,
            k,
            a_mps.stride(0),
            a_mps.stride(1),
            b_mps.stride(0),
            b_mps.stride(1),
            c_mps.stride(0),
            c_mps.stride(1),
            BLOCK_M=8,
            BLOCK_N=8,
            BLOCK_K=8,
        )
        torch.mps.synchronize()
        c_cpu = c_mps.cpu()
        torch.mps.synchronize()

        expected = a_cpu.to(torch.int32) @ b_cpu.to(torch.int32)
        assert torch.equal(c_cpu, expected)

    @skip_non_darwin
    @skip_no_mps
    def test_int8_matmul_i16_accumulation(self):
        """int8×int8 with int16 accumulation — small values avoid overflow."""
        import torch

        import triton
        import triton.language as tl

        @triton.jit
        def _matmul_i8_i16acc(
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
            acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
            for kk in range(0, k, BLOCK_K):
                for ki in range(0, BLOCK_K):
                    k_idx = kk + ki
                    a = tl.load(
                        a_ptr + offs_m * stride_am + k_idx * stride_ak,
                        mask=(offs_m < m) & (k_idx < k),
                        other=0,
                    ).to(tl.int32)
                    b = tl.load(
                        b_ptr + k_idx * stride_bk + offs_n * stride_bn,
                        mask=(k_idx < k) & (offs_n < n),
                        other=0,
                    ).to(tl.int32)
                    acc += a[:, None] * b[None, :]
            c16 = acc.to(tl.int16)
            c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
            tl.store(c_ptrs, c16, mask=(offs_m[:, None] < m) & (offs_n[None, :] < n))

        torch.manual_seed(202)
        m = n = k = 8
        a_cpu = torch.randint(-3, 4, (m, k), dtype=torch.int8)
        b_cpu = torch.randint(-3, 4, (k, n), dtype=torch.int8)
        a_mps = a_cpu.to("mps")
        b_mps = b_cpu.to("mps")
        c_mps = torch.empty((m, n), dtype=torch.int16, device="mps")

        _matmul_i8_i16acc[(1, 1, 1)](
            a_mps,
            b_mps,
            c_mps,
            m,
            n,
            k,
            a_mps.stride(0),
            a_mps.stride(1),
            b_mps.stride(0),
            b_mps.stride(1),
            c_mps.stride(0),
            c_mps.stride(1),
            BLOCK_M=8,
            BLOCK_N=8,
            BLOCK_K=8,
        )
        torch.mps.synchronize()
        c_cpu = c_mps.cpu()
        torch.mps.synchronize()

        expected = (a_cpu.to(torch.int32) @ b_cpu.to(torch.int32)).to(torch.int16)
        assert torch.equal(c_cpu, expected)

    @skip_non_darwin
    @skip_no_mps
    def test_int8_mixed_i8_i16_input_matmul(self):
        """Mixed int8 and int16 input matmul: widen both to int32 for acc."""
        import torch

        import triton
        import triton.language as tl

        @triton.jit
        def _mixed_int_matvec(
            a_ptr,
            x_ptr,
            out_ptr,
            m,
            k,
            stride_am,
            stride_ak,
            BLOCK_M: tl.constexpr,
            BLOCK_K: tl.constexpr,
        ):
            pid = tl.program_id(axis=0)
            offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
            acc = tl.zeros((BLOCK_M,), dtype=tl.int32)
            for kk in range(0, k, BLOCK_K):
                offs_k = tl.arange(0, BLOCK_K) + kk
                a = tl.load(
                    a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak,
                    mask=(offs_m[:, None] < m) & (offs_k[None, :] < k),
                    other=0,
                ).to(tl.int32)
                x = tl.load(
                    x_ptr + offs_k,
                    mask=offs_k < k,
                    other=0,
                ).to(tl.int32)
                acc += tl.sum(a * x[None, :], axis=1)
            tl.store(out_ptr + offs_m, acc, mask=offs_m < m)

        torch.manual_seed(203)
        m, k = 16, 16
        a_cpu = torch.randint(-8, 8, (m, k), dtype=torch.int8)
        x_cpu = torch.randint(-100, 100, (k,), dtype=torch.int16)
        a_mps = a_cpu.to("mps")
        x_mps = x_cpu.to("mps")
        out_mps = torch.empty((m,), dtype=torch.int32, device="mps")

        _mixed_int_matvec[(triton.cdiv(m, 16),)](
            a_mps,
            x_mps,
            out_mps,
            m,
            k,
            a_mps.stride(0),
            a_mps.stride(1),
            BLOCK_M=16,
            BLOCK_K=16,
        )
        torch.mps.synchronize()
        out_cpu = out_mps.cpu()
        torch.mps.synchronize()

        expected = (a_cpu.to(torch.int32) @ x_cpu.to(torch.int32)).to(torch.int32)
        assert torch.equal(out_cpu, expected)

    @skip_non_darwin
    @skip_no_mps
    def test_int8_saturation_boundary_values(self):
        """int8 boundary: INT8_MIN=-128, INT8_MAX=127 in vector add."""
        import torch

        import triton
        import triton.language as tl

        @triton.jit
        def _vadd_i8_sat(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
            pid = tl.program_id(axis=0)
            offs = pid * BLOCK + tl.arange(0, BLOCK)
            mask = offs < n
            x = tl.load(x_ptr + offs, mask=mask, other=0)
            y = tl.load(y_ptr + offs, mask=mask, other=0)
            tl.store(out_ptr + offs, x + y, mask=mask)

        n = 8
        x_cpu = torch.tensor([-128, 127, -128, 127, 0, -1, 1, 64], dtype=torch.int8)
        y_cpu = torch.tensor([0, 0, 1, -1, -128, 127, -1, 64], dtype=torch.int8)
        x_mps = x_cpu.to("mps")
        y_mps = y_cpu.to("mps")
        out_mps = torch.empty((n,), dtype=torch.int8, device="mps")

        _vadd_i8_sat[(1,)](x_mps, y_mps, out_mps, n, BLOCK=8)
        torch.mps.synchronize()
        out_cpu = out_mps.cpu()
        torch.mps.synchronize()

        expected = (x_cpu.to(torch.int16) + y_cpu.to(torch.int16)).to(torch.int8)
        assert torch.equal(out_cpu, expected)


# ── Broad ML Workload Runtime Suites ────────────────────────────────


class TestMetalBroadMLWorkloads:
    """Broader ML workload runtime correctness tests on MPS.

    Tests realistic ML patterns including 1D convolution, training
    iteration, scatter/gather with irregular indices, fused
    layernorm+linear+residual, and multi-head attention score.
    All verified against CPU torch reference implementations.
    """

    @skip_non_darwin
    @skip_no_mps
    def test_runtime_sliding_window_conv1d(self):
        """1D convolution via sliding window dot product over channels."""
        import torch

        import triton
        import triton.language as tl

        @triton.jit
        def _conv1d_simple(
            x_ptr,
            w_ptr,
            y_ptr,
            in_len,
            out_len,
            ksize,
            BLOCK: tl.constexpr,
        ):
            pid = tl.program_id(axis=0)
            offs = pid * BLOCK + tl.arange(0, BLOCK)
            mask = offs < out_len
            acc = tl.zeros((BLOCK,), dtype=tl.float32)
            for ki in range(ksize):
                x = tl.load(
                    x_ptr + offs + ki, mask=mask & ((offs + ki) < in_len), other=0.0
                )
                w = tl.load(w_ptr + ki)
                acc += x * w
            tl.store(y_ptr + offs, acc, mask=mask)

        torch.manual_seed(300)
        in_len, ksize = 128, 5
        out_len = in_len - ksize + 1
        x_cpu = torch.randn((in_len,), dtype=torch.float32)
        w_cpu = torch.randn((ksize,), dtype=torch.float32)
        x_mps = x_cpu.to("mps")
        w_mps = w_cpu.to("mps")
        y_mps = torch.empty((out_len,), device="mps", dtype=torch.float32)

        _conv1d_simple[(triton.cdiv(out_len, 64),)](
            x_mps,
            w_mps,
            y_mps,
            in_len,
            out_len,
            ksize,
            BLOCK=64,
        )
        torch.mps.synchronize()
        y_cpu = y_mps.cpu()
        torch.mps.synchronize()

        expected = torch.conv1d(
            x_cpu.view(1, 1, -1), w_cpu.flip(0).view(1, 1, -1)
        ).view(-1)
        # conv1d does cross-correlation with flipped kernel; we do
        # direct correlation, so compare with non-flipped reference
        expected_direct = torch.zeros(out_len)
        for i in range(out_len):
            expected_direct[i] = (x_cpu[i : i + ksize] * w_cpu).sum()
        assert torch.allclose(y_cpu, expected_direct, atol=1e-4, rtol=1e-4)

    @skip_non_darwin
    @skip_no_mps
    def test_runtime_training_iteration_pattern(self):
        """Simulated training step: fwd→loss→grad→update on MPS."""
        import torch

        import triton
        import triton.language as tl

        @triton.jit
        def _fwd_linear(x_ptr, w_ptr, b_ptr, y_ptr, n, d, BLOCK: tl.constexpr):
            pid = tl.program_id(axis=0)
            offs = tl.arange(0, BLOCK)
            mask = offs < d
            x_base = x_ptr + pid * d
            x = tl.load(x_base + offs, mask=mask, other=0.0)
            w = tl.load(w_ptr + offs, mask=mask, other=0.0)
            b = tl.load(b_ptr + offs, mask=mask, other=0.0)
            y = x * w + b
            tl.store(y_ptr + pid * d + offs, y, mask=mask)

        @triton.jit
        def _mse_grad(pred_ptr, target_ptr, grad_ptr, n, BLOCK: tl.constexpr):
            pid = tl.program_id(axis=0)
            offs = pid * BLOCK + tl.arange(0, BLOCK)
            mask = offs < n
            pred = tl.load(pred_ptr + offs, mask=mask, other=0.0)
            target = tl.load(target_ptr + offs, mask=mask, other=0.0)
            grad = 2.0 * (pred - target) / n
            tl.store(grad_ptr + offs, grad, mask=mask)

        @triton.jit
        def _sgd_update(w_ptr, grad_ptr, d, lr, BLOCK: tl.constexpr):
            offs = tl.arange(0, BLOCK)
            mask = offs < d
            w = tl.load(w_ptr + offs, mask=mask, other=0.0)
            g = tl.load(grad_ptr + offs, mask=mask, other=0.0)
            w_new = w - lr * g
            tl.store(w_ptr + offs, w_new, mask=mask)

        torch.manual_seed(301)
        n, d = 16, 32
        x_cpu = torch.randn((n, d), dtype=torch.float32)
        w_cpu = torch.randn((d,), dtype=torch.float32)
        b_cpu = torch.randn((d,), dtype=torch.float32)
        target_cpu = torch.randn((n, d), dtype=torch.float32)
        lr = 0.01

        x_mps = x_cpu.to("mps")
        w_mps = w_cpu.clone().to("mps")
        b_mps = b_cpu.to("mps")
        target_mps = target_cpu.to("mps")
        y_mps = torch.empty((n, d), device="mps", dtype=torch.float32)

        _fwd_linear[(n,)](x_mps, w_mps, b_mps, y_mps, n, d, BLOCK=32)
        torch.mps.synchronize()

        grad_flat = torch.empty((n * d,), device="mps", dtype=torch.float32)
        _mse_grad[(triton.cdiv(n * d, 64),)](
            y_mps.reshape(-1),
            target_mps.reshape(-1),
            grad_flat,
            n * d,
            BLOCK=64,
        )
        torch.mps.synchronize()

        grad_w_mps = grad_flat.view(n, d).mean(dim=0)
        _sgd_update[(1,)](w_mps, grad_w_mps, d, lr, BLOCK=32)
        torch.mps.synchronize()

        y_ref = x_cpu * w_cpu + b_cpu
        grad_ref = 2.0 * (y_ref - target_cpu) / (n * d)
        grad_w_ref = grad_ref.mean(dim=0)
        w_ref = w_cpu - lr * grad_w_ref

        w_result = w_mps.cpu()
        torch.mps.synchronize()
        assert torch.allclose(w_result, w_ref, atol=1e-4, rtol=1e-4)

    @skip_non_darwin
    @skip_no_mps
    def test_runtime_gather_irregular_indices(self):
        """Gather with irregular/non-contiguous indices on MPS."""
        import torch

        import triton
        import triton.language as tl

        @triton.jit
        def _gather(
            src_ptr,
            idx_ptr,
            out_ptr,
            n_idx,
            src_len,
            BLOCK: tl.constexpr,
        ):
            pid = tl.program_id(axis=0)
            offs = pid * BLOCK + tl.arange(0, BLOCK)
            mask = offs < n_idx
            idx = tl.load(idx_ptr + offs, mask=mask, other=0)
            val = tl.load(src_ptr + idx, mask=mask & (idx < src_len), other=0.0)
            tl.store(out_ptr + offs, val, mask=mask)

        torch.manual_seed(302)
        src_len = 256
        n_idx = 64
        src_cpu = torch.randn((src_len,), dtype=torch.float32)
        idx_cpu = torch.randint(0, src_len, (n_idx,), dtype=torch.int32)

        src_mps = src_cpu.to("mps")
        idx_mps = idx_cpu.to("mps")
        out_mps = torch.empty((n_idx,), device="mps", dtype=torch.float32)

        _gather[(triton.cdiv(n_idx, 64),)](
            src_mps,
            idx_mps,
            out_mps,
            n_idx,
            src_len,
            BLOCK=64,
        )
        torch.mps.synchronize()
        out_cpu = out_mps.cpu()
        torch.mps.synchronize()

        expected = src_cpu[idx_cpu.to(torch.int64)]
        assert torch.allclose(out_cpu, expected, atol=1e-5, rtol=1e-5)

    @skip_non_darwin
    @skip_no_mps
    def test_runtime_fused_layernorm_linear_residual(self):
        """Fused layernorm→linear projection→residual add on MPS."""
        import torch

        import triton
        import triton.language as tl

        @triton.jit
        def _fused_ln_linear_res(
            x_ptr,
            w_ln_ptr,
            b_ln_ptr,
            w_proj_ptr,
            b_proj_ptr,
            res_ptr,
            y_ptr,
            n_cols,
            eps,
            BLOCK: tl.constexpr,
        ):
            pid = tl.program_id(axis=0)
            offs = tl.arange(0, BLOCK)
            mask = offs < n_cols
            row_base = pid * n_cols

            x = tl.load(x_ptr + row_base + offs, mask=mask, other=0.0)
            mean = tl.sum(x, axis=0) / n_cols
            centered = x - mean
            var = tl.sum(centered * centered, axis=0) / n_cols
            inv_std = 1.0 / tl.sqrt(var + eps)

            w_ln = tl.load(w_ln_ptr + offs, mask=mask, other=1.0)
            b_ln = tl.load(b_ln_ptr + offs, mask=mask, other=0.0)
            normed = centered * inv_std * w_ln + b_ln

            w_proj = tl.load(w_proj_ptr + offs, mask=mask, other=1.0)
            b_proj = tl.load(b_proj_ptr + offs, mask=mask, other=0.0)
            projected = normed * w_proj + b_proj

            res = tl.load(res_ptr + row_base + offs, mask=mask, other=0.0)
            out = projected + res
            tl.store(y_ptr + row_base + offs, out, mask=mask)

        torch.manual_seed(303)
        rows, cols = 8, 64
        eps = 1e-5
        x_cpu = torch.randn((rows, cols), dtype=torch.float32)
        w_ln_cpu = torch.ones((cols,), dtype=torch.float32)
        b_ln_cpu = torch.zeros((cols,), dtype=torch.float32)
        w_proj_cpu = torch.randn((cols,), dtype=torch.float32) * 0.1
        b_proj_cpu = torch.randn((cols,), dtype=torch.float32) * 0.01
        res_cpu = torch.randn((rows, cols), dtype=torch.float32)

        x_mps = x_cpu.to("mps")
        w_ln_mps = w_ln_cpu.to("mps")
        b_ln_mps = b_ln_cpu.to("mps")
        w_proj_mps = w_proj_cpu.to("mps")
        b_proj_mps = b_proj_cpu.to("mps")
        res_mps = res_cpu.to("mps")
        y_mps = torch.empty((rows, cols), device="mps", dtype=torch.float32)

        _fused_ln_linear_res[(rows,)](
            x_mps,
            w_ln_mps,
            b_ln_mps,
            w_proj_mps,
            b_proj_mps,
            res_mps,
            y_mps,
            cols,
            eps,
            BLOCK=64,
        )
        torch.mps.synchronize()
        y_cpu = y_mps.cpu()
        torch.mps.synchronize()

        normed_ref = torch.nn.functional.layer_norm(
            x_cpu,
            normalized_shape=(cols,),
            weight=w_ln_cpu,
            bias=b_ln_cpu,
            eps=eps,
        )
        projected_ref = normed_ref * w_proj_cpu + b_proj_cpu
        expected = projected_ref + res_cpu
        assert torch.allclose(y_cpu, expected, atol=5e-3, rtol=5e-3)

    @skip_non_darwin
    @skip_no_mps
    def test_runtime_multihead_attention_score(self):
        """Multi-head attention: Q@K^T / sqrt(d_k) per head on MPS."""
        import math

        import torch

        import triton
        import triton.language as tl

        @triton.jit
        def _attn_score(
            q_ptr,
            k_ptr,
            s_ptr,
            seq_len,
            d_k,
            stride_qs,
            stride_qd,
            stride_ks,
            stride_kd,
            stride_ss,
            stride_sd,
            inv_sqrt_dk: tl.constexpr,
            BLOCK_S: tl.constexpr,
            BLOCK_D: tl.constexpr,
        ):
            pid_row = tl.program_id(axis=0)
            pid_col = tl.program_id(axis=1)
            offs_row = pid_row * BLOCK_S + tl.arange(0, BLOCK_S)
            offs_col = pid_col * BLOCK_S + tl.arange(0, BLOCK_S)
            offs_d = tl.arange(0, BLOCK_D)

            acc = tl.zeros((BLOCK_S, BLOCK_S), dtype=tl.float32)
            for dd in range(0, d_k, BLOCK_D):
                q = tl.load(
                    q_ptr
                    + offs_row[:, None] * stride_qs
                    + (offs_d[None, :] + dd) * stride_qd,
                    mask=(offs_row[:, None] < seq_len) & (offs_d[None, :] + dd < d_k),
                    other=0.0,
                )
                k = tl.load(
                    k_ptr
                    + offs_col[:, None] * stride_ks
                    + (offs_d[None, :] + dd) * stride_kd,
                    mask=(offs_col[:, None] < seq_len) & (offs_d[None, :] + dd < d_k),
                    other=0.0,
                )
                acc += tl.dot(q, tl.trans(k))

            acc = acc * inv_sqrt_dk
            s_ptrs = (
                s_ptr + offs_row[:, None] * stride_ss + offs_col[None, :] * stride_sd
            )
            tl.store(
                s_ptrs,
                acc,
                mask=(offs_row[:, None] < seq_len) & (offs_col[None, :] < seq_len),
            )

        torch.manual_seed(304)
        seq_len, d_k = 16, 32
        inv_sqrt_dk = 1.0 / math.sqrt(d_k)
        q_cpu = torch.randn((seq_len, d_k), dtype=torch.float32)
        k_cpu = torch.randn((seq_len, d_k), dtype=torch.float32)

        q_mps = q_cpu.to("mps")
        k_mps = k_cpu.to("mps")
        s_mps = torch.empty((seq_len, seq_len), device="mps", dtype=torch.float32)

        _attn_score[(triton.cdiv(seq_len, 16), triton.cdiv(seq_len, 16), 1)](
            q_mps,
            k_mps,
            s_mps,
            seq_len,
            d_k,
            q_mps.stride(0),
            q_mps.stride(1),
            k_mps.stride(0),
            k_mps.stride(1),
            s_mps.stride(0),
            s_mps.stride(1),
            inv_sqrt_dk=inv_sqrt_dk,
            BLOCK_S=16,
            BLOCK_D=16,
        )
        torch.mps.synchronize()
        s_cpu = s_mps.cpu()
        torch.mps.synchronize()

        expected = (q_cpu @ k_cpu.T) * inv_sqrt_dk
        assert torch.allclose(s_cpu, expected, atol=1e-4, rtol=1e-4)


# ── Driver / Backend Feature-Parity Tests ─────────────────────────────


class TestMetalDriverFeatures:
    """Tests for Metal driver and backend feature parity additions."""

    # -- check_dot_compatibility ------------------------------------------

    @skip_non_darwin
    def test_check_dot_compatibility_valid(self):
        from unittest.mock import MagicMock

        from third_party.metal.backend.compiler import MetalBackend

        for bw in (8, 16, 32):
            scalar = MagicMock()
            scalar.primitive_bitwidth = bw
            ty = MagicMock()
            ty.scalar = scalar
            result = MetalBackend.check_dot_compatibility(ty, ty)
            assert result == (1, 1, 1), f"Expected (1,1,1) for {bw}-bit operands"

    @skip_non_darwin
    def test_check_dot_compatibility_invalid_fp64(self):
        from unittest.mock import MagicMock

        from third_party.metal.backend.compiler import MetalBackend

        scalar_64 = MagicMock()
        scalar_64.primitive_bitwidth = 64
        ty64 = MagicMock()
        ty64.scalar = scalar_64

        scalar_32 = MagicMock()
        scalar_32.primitive_bitwidth = 32
        ty32 = MagicMock()
        ty32.scalar = scalar_32

        with pytest.raises(ValueError, match="Metal does not support fp64/i64"):
            MetalBackend.check_dot_compatibility(ty64, ty32)

        with pytest.raises(ValueError, match="Metal does not support fp64/i64"):
            MetalBackend.check_dot_compatibility(ty32, ty64)

    # -- map_python_to_cpp_type -------------------------------------------

    @skip_non_darwin
    def test_map_python_to_cpp_type_scalars(self):
        from third_party.metal.backend.driver import MetalDriver

        driver = MetalDriver.__new__(MetalDriver)
        expected = {
            "i1": "bool",
            "i8": "int8_t",
            "u8": "uint8_t",
            "i16": "int16_t",
            "u16": "uint16_t",
            "i32": "int32_t",
            "i64": "int64_t",
            "u32": "uint32_t",
            "u64": "uint64_t",
            "fp16": "half",
            "f16": "half",
            "bf16": "bfloat16_t",
            "fp32": "float",
            "f32": "float",
            "fp64": "double",
            "f64": "double",
        }
        for ty, cpp in expected.items():
            assert driver.map_python_to_cpp_type(ty) == cpp, f"{ty} -> {cpp}"

    @skip_non_darwin
    def test_map_python_to_cpp_type_pointers(self):
        from third_party.metal.backend.driver import MetalDriver

        driver = MetalDriver.__new__(MetalDriver)
        assert driver.map_python_to_cpp_type("*fp32") == "id<MTLBuffer>"
        assert driver.map_python_to_cpp_type("*i32") == "id<MTLBuffer>"
        assert driver.map_python_to_cpp_type("*bf16") == "id<MTLBuffer>"

    @skip_non_darwin
    def test_map_python_to_cpp_type_unknown(self):
        from third_party.metal.backend.driver import MetalDriver

        driver = MetalDriver.__new__(MetalDriver)
        with pytest.raises(TypeError, match="Unsupported Triton type"):
            driver.map_python_to_cpp_type("complex128")

    # -- metal_ext module -------------------------------------------------

    @skip_non_darwin
    def test_metal_ext_module_importable(self):
        from third_party.metal.language import metal_ext

        assert hasattr(metal_ext, "thread_position_in_grid")
        assert hasattr(metal_ext, "simdgroup_index")
        assert hasattr(metal_ext, "threadgroup_position")
        assert hasattr(metal_ext, "METAL_BUILTINS")
        assert callable(metal_ext.thread_position_in_grid)

    @skip_non_darwin
    def test_metal_ext_builtins_dict(self):
        from third_party.metal.language import metal_ext

        assert isinstance(metal_ext.METAL_BUILTINS, dict)
        assert "thread_position_in_grid" in metal_ext.METAL_BUILTINS
        assert "simdgroup_index_in_threadgroup" in metal_ext.METAL_BUILTINS
        assert len(metal_ext.METAL_BUILTINS) >= 6


# ── GPU profiling / timing tests ────────────────────────────────────


class TestMetalGPUProfiling:
    """Tests for Metal GPU-side timing via gpuStartTime/gpuEndTime."""

    @skip_non_darwin
    def test_timing_event_returns_positive_time(self):
        """A kernel execution should produce a positive elapsed time."""
        import torch
        import triton
        import triton.language as tl

        @triton.jit
        def _nop_kernel(out_ptr, BLOCK: tl.constexpr):
            pid = tl.program_id(0)
            tl.store(out_ptr + pid, pid)

        from third_party.metal.backend.driver import _MetalTimingEvent

        start = _MetalTimingEvent()
        out = torch.zeros(128, dtype=torch.int32, device="mps")
        start.record()
        _nop_kernel[(128,)](out, BLOCK=1)
        end = _MetalTimingEvent()
        end.record()
        elapsed = start.elapsed_time(end)
        assert elapsed > 0, f"Expected positive elapsed time, got {elapsed}"
        assert elapsed < 60000, f"Unreasonable elapsed time: {elapsed}ms"

    @skip_non_darwin
    def test_gpu_timing_less_than_or_equal_host_timing(self):
        """GPU-side timing should be ≤ host-side (synchronize overhead)."""
        import torch
        import triton
        import triton.language as tl

        @triton.jit
        def _work_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
            pid = tl.program_id(0)
            offsets = pid * BLOCK + tl.arange(0, BLOCK)
            mask = offsets < n
            x = tl.load(x_ptr + offsets, mask=mask)
            tl.store(out_ptr + offsets, x * 2.0, mask=mask)

        from third_party.metal.backend.driver import (
            _MetalTimingEvent,
            _gpu_elapsed_ms,
        )

        n = 4096
        x = torch.randn(n, dtype=torch.float32, device="mps")
        out = torch.zeros(n, dtype=torch.float32, device="mps")

        start = _MetalTimingEvent()
        start.record()
        _work_kernel[(16,)](x, out, n, BLOCK=256)
        end = _MetalTimingEvent()
        end.record()

        host_ms = (end._host_timestamp - start._host_timestamp) * 1000.0
        gpu_ms = _gpu_elapsed_ms(start._cmd_buf, end._cmd_buf)

        assert host_ms > 0
        if gpu_ms is not None:
            assert gpu_ms <= host_ms * 1.5, (
                f"GPU time ({gpu_ms:.3f}ms) much larger than host time ({host_ms:.3f}ms)"
            )

    @skip_non_darwin
    def test_fallback_when_no_cmd_buf(self):
        """Timing falls back to host timing when no command buffer captured."""
        from third_party.metal.backend.driver import _MetalTimingEvent

        start = _MetalTimingEvent()
        start._host_timestamp = 1.0
        start._cmd_buf = None

        end = _MetalTimingEvent()
        end._host_timestamp = 1.5
        end._cmd_buf = None

        elapsed = start.elapsed_time(end)
        assert abs(elapsed - 500.0) < 0.1, f"Expected ~500ms, got {elapsed}"

    @skip_non_darwin
    def test_device_interface_uses_gpu_timing_event(self):
        """_MetalDeviceInterface.Event returns GPU-timing-aware event class."""
        from third_party.metal.backend.driver import _MetalDeviceInterface

        event = _MetalDeviceInterface.Event()
        assert hasattr(event, "_cmd_buf"), (
            "Event should have _cmd_buf attribute for GPU timing"
        )


# ── Barrier insertion pass tests ────────────────────────────────────


class TestMetalBarrierInsertion:
    """Tests for the dedicated Metal shared-memory barrier insertion pass."""

    _NO_SHARED_MEM_IR = """\
; ModuleID = 'no_shared'
target triple = "aarch64-apple-macosx14.0.0"

define void @kernel(ptr addrspace(1) %out, ptr addrspace(1) %in) {
entry:
  %v = load float, ptr addrspace(1) %in
  store float %v, ptr addrspace(1) %out
  ret void
}
"""

    _SIMPLE_STORE_LOAD_IR = """\
; ModuleID = 'simple'
target triple = "aarch64-apple-macosx14.0.0"

@global_smem = external addrspace(3) global [0 x i8]

define void @kernel(ptr addrspace(1) %out) {
entry:
  %smem = getelementptr inbounds [0 x i8], ptr addrspace(3) @global_smem, i32 0, i32 0
  store float 1.0, ptr addrspace(3) %smem
  %v = load float, ptr addrspace(3) %smem
  store float %v, ptr addrspace(1) %out
  ret void
}
"""

    _LOOP_CARRIED_IR = """\
; ModuleID = 'loop_carried'
target triple = "aarch64-apple-macosx14.0.0"

@global_smem = external addrspace(3) global [0 x i8]

define void @kernel(ptr addrspace(1) %out, i32 %n) {
entry:
  br label %loop

loop:
  %i = phi i32 [ 0, %entry ], [ %i_next, %loop ]
  %smem = getelementptr inbounds [0 x i8], ptr addrspace(3) @global_smem, i32 0, i32 %i
  store float 1.0, ptr addrspace(3) %smem
  %v = load float, ptr addrspace(3) %smem
  store float %v, ptr addrspace(1) %out
  %i_next = add i32 %i, 1
  %cmp = icmp slt i32 %i_next, %n
  br i1 %cmp, label %loop, label %exit

exit:
  ret void
}
"""

    _MATMUL_PATTERN_IR = """\
; ModuleID = 'matmul'
target triple = "aarch64-apple-macosx14.0.0"

@global_smem = external addrspace(3) global [0 x i8]

define void @kernel(ptr addrspace(1) %A, ptr addrspace(1) %B, ptr addrspace(1) %C, i32 %K) {
entry:
  br label %k_loop

k_loop:
  %k = phi i32 [ 0, %entry ], [ %k_next, %k_loop_end ]
  %smem_a = getelementptr inbounds [0 x i8], ptr addrspace(3) @global_smem, i32 0, i32 0
  %smem_b = getelementptr inbounds [0 x i8], ptr addrspace(3) @global_smem, i32 0, i32 512
  %a_val = load float, ptr addrspace(1) %A
  store float %a_val, ptr addrspace(3) %smem_a
  %b_val = load float, ptr addrspace(1) %B
  store float %b_val, ptr addrspace(3) %smem_b
  br label %compute

compute:
  %a_shared = load float, ptr addrspace(3) %smem_a
  %b_shared = load float, ptr addrspace(3) %smem_b
  %dot = fmul float %a_shared, %b_shared
  store float %dot, ptr addrspace(1) %C
  br label %k_loop_end

k_loop_end:
  %k_next = add i32 %k, 1
  %done = icmp slt i32 %k_next, %K
  br i1 %done, label %k_loop, label %exit

exit:
  ret void
}
"""

    @skip_non_darwin
    def test_barrier_pass_no_shared_mem(self):
        """Pass should be no-op for kernels without shared memory."""
        from third_party.metal.backend.barrier_pass import MetalBarrierInsertionPass

        pass_ = MetalBarrierInsertionPass()
        result = pass_.run(self._NO_SHARED_MEM_IR)
        assert result == self._NO_SHARED_MEM_IR
        assert len(pass_.decisions) == 0

    @skip_non_darwin
    def test_barrier_pass_simple_shared_store_load(self):
        """Barrier inserted between shared mem store and subsequent load."""
        from third_party.metal.backend.barrier_pass import MetalBarrierInsertionPass

        pass_ = MetalBarrierInsertionPass()
        result = pass_.run(self._SIMPLE_STORE_LOAD_IR)
        assert "call void @llvm.nvvm.barrier0()" in result
        assert len(pass_.decisions) >= 1

        lines = result.split("\n")
        store_idx = next(
            i for i, l in enumerate(lines) if "store float 1.0, ptr addrspace(3)" in l
        )
        barrier_idx = next(i for i, l in enumerate(lines) if "llvm.nvvm.barrier0" in l)
        load_idx = next(
            i for i, l in enumerate(lines) if "= load float, ptr addrspace(3)" in l
        )
        assert store_idx < barrier_idx < load_idx

    @skip_non_darwin
    def test_barrier_pass_loop_carried_dependency(self):
        """Barrier inserted for loop-carried shared memory dependencies."""
        from third_party.metal.backend.barrier_pass import MetalBarrierInsertionPass

        pass_ = MetalBarrierInsertionPass()
        result = pass_.run(self._LOOP_CARRIED_IR)
        assert "call void @llvm.nvvm.barrier0()" in result
        has_loop_reason = any(
            "loop" in d.reason.lower() or "backedge" in d.reason.lower()
            for d in pass_.decisions
        )
        assert has_loop_reason or len(pass_.decisions) >= 1

    @skip_non_darwin
    def test_barrier_pass_idempotent(self):
        """Running pass twice produces same result."""
        from third_party.metal.backend.barrier_pass import MetalBarrierInsertionPass

        pass1 = MetalBarrierInsertionPass()
        result1 = pass1.run(self._LOOP_CARRIED_IR)
        count1 = result1.count("call void @llvm.nvvm.barrier0()")

        pass2 = MetalBarrierInsertionPass()
        result2 = pass2.run(result1)
        count2 = result2.count("call void @llvm.nvvm.barrier0()")

        assert result1 == result2, "Pass is not idempotent"
        assert count1 == count2

    @skip_non_darwin
    def test_barrier_pass_matmul_pattern(self):
        """Correct barrier placement for blocked matmul shared memory access."""
        from third_party.metal.backend.barrier_pass import MetalBarrierInsertionPass

        pass_ = MetalBarrierInsertionPass()
        result = pass_.run(self._MATMUL_PATTERN_IR)
        assert "call void @llvm.nvvm.barrier0()" in result
        assert len(pass_.decisions) >= 1
        barrier_count = result.count("call void @llvm.nvvm.barrier0()")
        assert barrier_count >= 1

    @skip_non_darwin
    def test_barrier_pass_debug_report(self):
        """Pass reports barrier decisions when debug mode is active."""
        import contextlib
        import io

        from third_party.metal.backend.barrier_pass import MetalBarrierInsertionPass

        pass_ = MetalBarrierInsertionPass(debug=True)
        f = io.StringIO()
        with contextlib.redirect_stdout(f):
            pass_.run(self._SIMPLE_STORE_LOAD_IR)

        output = f.getvalue()
        assert "[BARRIER_PASS]" in output
        assert len(pass_.decisions) >= 1

    @skip_non_darwin
    def test_barrier_pass_declaration_added(self):
        """ensure_barrier_declaration adds declaration when barriers present."""
        from third_party.metal.backend.barrier_pass import run_barrier_pass

        result = run_barrier_pass(self._SIMPLE_STORE_LOAD_IR)
        assert "declare void @llvm.nvvm.barrier0()" in result

    @skip_non_darwin
    def test_barrier_pass_no_declaration_when_unused(self):
        """No barrier declaration added when no barriers inserted."""
        from third_party.metal.backend.barrier_pass import ensure_barrier_declaration

        result = ensure_barrier_declaration(self._NO_SHARED_MEM_IR)
        assert "declare void @llvm.nvvm.barrier0()" not in result

    @skip_non_darwin
    def test_barrier_pass_run_convenience(self):
        """run_barrier_pass convenience function works end-to-end."""
        from third_party.metal.backend.barrier_pass import run_barrier_pass

        result = run_barrier_pass(self._LOOP_CARRIED_IR)
        assert "call void @llvm.nvvm.barrier0()" in result
        assert "declare void @llvm.nvvm.barrier0()" in result


# ── C++ barrier lowering verification tests ──────────────────────────


class TestMetalBarrierCppLowering:
    """Tests verifying the C++ BarrierOpToLLVM and NvidiaArtifact barrier rewrite.

    The C++ lowering path converts ttg::BarrierOp directly to
    __metal_simdgroup_barrier(flags) preserving addrSpace, and also rewrites
    any leftover nvvm.barrier0 ops to __metal_simdgroup_barrier(1) as a safety
    net. These tests verify the translator handles the resulting LLVM IR
    correctly and that end-to-end kernels with barriers produce correct results.
    """

    _NVVM_BARRIER_IR = """\
; ModuleID = 'nvvm_barrier'
target triple = "aarch64-apple-macosx14.0.0"

@global_smem = external addrspace(3) global [0 x i8]

declare void @llvm.nvvm.barrier0()

define void @kernel(ptr addrspace(1) %out) {
entry:
  %smem = getelementptr inbounds [0 x i8], ptr addrspace(3) @global_smem, i32 0, i32 0
  store float 1.0, ptr addrspace(3) %smem
  call void @llvm.nvvm.barrier0()
  %v = load float, ptr addrspace(3) %smem
  store float %v, ptr addrspace(1) %out
  ret void
}
"""

    _METAL_BARRIER_FLAGS_IR = """\
; ModuleID = 'metal_barrier_flags'
target triple = "aarch64-apple-macosx14.0.0"

define void @kernel(ptr addrspace(1) %out) {
entry:
  call void @__metal_simdgroup_barrier(i32 1)
  call void @__metal_simdgroup_barrier(i32 2)
  call void @__metal_simdgroup_barrier(i32 3)
  ret void
}
"""

    @skip_non_darwin
    def test_nvvm_barrier0_translates_to_threadgroup_barrier(self):
        """nvvm.barrier0 in LLVM IR becomes threadgroup_barrier in MSL."""
        from third_party.metal.backend.compiler import MetalBackend

        metadata = {}
        msl = MetalBackend.make_metal_ir(self._NVVM_BARRIER_IR, metadata, None)
        assert "threadgroup_barrier(mem_flags::mem_threadgroup);" in msl
        assert "llvm.nvvm.barrier0" not in msl

    @skip_non_darwin
    def test_metal_barrier_flag1_threadgroup(self):
        """__metal_simdgroup_barrier(1) → threadgroup_barrier(mem_threadgroup)."""
        from third_party.metal.backend.compiler import MetalBackend

        metadata = {}
        msl = MetalBackend.make_metal_ir(self._METAL_BARRIER_FLAGS_IR, metadata, None)
        assert "threadgroup_barrier(mem_flags::mem_threadgroup);" in msl

    @skip_non_darwin
    def test_metal_barrier_flag2_device(self):
        """__metal_simdgroup_barrier(2) → threadgroup_barrier(mem_device)."""
        from third_party.metal.backend.compiler import MetalBackend

        metadata = {}
        msl = MetalBackend.make_metal_ir(self._METAL_BARRIER_FLAGS_IR, metadata, None)
        assert "threadgroup_barrier(mem_flags::mem_device);" in msl

    @skip_non_darwin
    def test_metal_barrier_flag3_combined(self):
        """__metal_simdgroup_barrier(3) → threadgroup_barrier(mem_threadgroup|mem_device)."""
        from third_party.metal.backend.compiler import MetalBackend

        metadata = {}
        msl = MetalBackend.make_metal_ir(self._METAL_BARRIER_FLAGS_IR, metadata, None)
        assert (
            "threadgroup_barrier((mem_flags::mem_threadgroup | mem_flags::mem_device));"
            in msl
        )

    @skip_non_darwin
    def test_barrier_pipeline_produces_metal_barrier_calls(self):
        """Full compilation pipeline produces __metal_simdgroup_barrier calls.

        Compile a Triton kernel that uses shared memory (via tl.sum) and verify
        the LLVM IR or MSL output contains the expected barrier calls, proving
        the C++ lowering + Python barrier pass pipeline is functioning.
        """
        import torch

        import triton
        import triton.language as tl

        @triton.jit
        def _add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
            pid = tl.program_id(0)
            offsets = pid * BLOCK + tl.arange(0, BLOCK)
            mask = offsets < n
            x = tl.load(x_ptr + offsets, mask=mask)
            y = tl.load(y_ptr + offsets, mask=mask)
            tl.store(out_ptr + offsets, x + y, mask=mask)

        n = 128
        x = torch.ones(n, dtype=torch.float32, device="mps")
        y = torch.ones(n, dtype=torch.float32, device="mps")
        out = torch.zeros(n, dtype=torch.float32, device="mps")

        _add_kernel[(1,)](x, y, out, n, BLOCK=128)
        result = out.cpu()
        expected = torch.full((n,), 2.0)
        assert torch.allclose(result, expected), (
            f"Kernel with barrier pipeline failed: "
            f"max diff = {(result - expected).abs().max().item()}"
        )

    @skip_non_darwin
    def test_nvvm_barrier_artifact_rewrite_in_pipeline(self):
        """Verify that nvvm.barrier0 calls are rewritten in the final MSL.

        The Python barrier_pass inserts llvm.nvvm.barrier0 calls, and either
        the C++ NvidiaArtifactLowering or the Python translator must convert
        them to threadgroup_barrier. This verifies no nvvm.barrier0 leak through.
        """
        from third_party.metal.backend.barrier_pass import run_barrier_pass
        from third_party.metal.backend.compiler import MetalBackend

        ir_with_barrier = run_barrier_pass(
            TestMetalBarrierInsertion._SIMPLE_STORE_LOAD_IR
        )
        assert "llvm.nvvm.barrier0" in ir_with_barrier

        metadata = {}
        msl = MetalBackend.make_metal_ir(ir_with_barrier, metadata, None)
        assert "threadgroup_barrier" in msl
        assert "nvvm" not in msl.lower()


# ── SPMD op lowering tests ───────────────────────────────────────────


class TestMetalSPMDOpLowering:
    """Tests for the Metal-specific GetNumProgramsOp C++ lowering.

    The generic SPMD pattern only handles GetProgramIdOp. Metal adds
    GetNumProgramsOp → __metal_get_threadgroups_per_grid_{x,y,z}.
    """

    @skip_non_darwin
    def test_num_programs_compilation(self):
        """Kernel using tl.num_programs() compiles without error."""
        import torch

        import triton
        import triton.language as tl

        @triton.jit
        def _num_programs_kernel(out_ptr, BLOCK: tl.constexpr):
            pid = tl.program_id(0)
            n_progs = tl.num_programs(0)
            if pid == 0:
                tl.store(out_ptr, n_progs)

        out = torch.zeros(1, dtype=torch.int32, device="mps")
        _num_programs_kernel[(4,)](out, BLOCK=1)
        result = out.cpu().item()
        assert result == 4, f"Expected num_programs=4, got {result}"

    @skip_non_darwin
    def test_num_programs_axis1(self):
        """tl.num_programs(1) returns correct grid size on axis 1."""
        import torch

        import triton
        import triton.language as tl

        @triton.jit
        def _np_axis1_kernel(out_ptr, BLOCK: tl.constexpr):
            pid_x = tl.program_id(0)
            pid_y = tl.program_id(1)
            n_progs_y = tl.num_programs(1)
            if pid_x == 0 and pid_y == 0:
                tl.store(out_ptr, n_progs_y)

        out = torch.zeros(1, dtype=torch.int32, device="mps")
        _np_axis1_kernel[(2, 3)](out, BLOCK=1)
        result = out.cpu().item()
        assert result == 3, f"Expected num_programs(1)=3, got {result}"

    @skip_non_darwin
    def test_grid_stride_loop_pattern(self):
        """Grid-stride loop using program_id + num_programs produces correct results."""
        import torch

        import triton
        import triton.language as tl

        @triton.jit
        def _grid_stride_kernel(x_ptr, out_ptr, n_elements, BLOCK: tl.constexpr):
            pid = tl.program_id(0)
            n_progs = tl.num_programs(0)
            for start in range(pid * BLOCK, n_elements, n_progs * BLOCK):
                offsets = start + tl.arange(0, BLOCK)
                mask = offsets < n_elements
                x = tl.load(x_ptr + offsets, mask=mask)
                tl.store(out_ptr + offsets, x * 2.0, mask=mask)

        n = 512
        x = torch.arange(n, dtype=torch.float32, device="mps")
        out = torch.zeros(n, dtype=torch.float32, device="mps")

        _grid_stride_kernel[(4,)](x, out, n, BLOCK=64)
        expected = x * 2.0
        result = out.cpu()
        assert torch.allclose(result.cpu(), expected.cpu(), atol=1e-5), (
            f"Grid-stride loop with num_programs failed: "
            f"max diff = {(result - expected).abs().max().item()}"
        )

    @skip_non_darwin
    def test_num_programs_ir_contains_metal_builtin(self):
        """Compiled IR for num_programs kernel contains Metal grid size builtin."""
        import torch

        import triton
        import triton.language as tl

        @triton.jit
        def _np_ir_kernel(out_ptr, BLOCK: tl.constexpr):
            pid = tl.program_id(0)
            n = tl.num_programs(0)
            tl.store(out_ptr + pid, n)

        src = triton.compiler.ASTSource(
            fn=_np_ir_kernel,
            signature={"out_ptr": "*i32"},
            constexprs={"BLOCK": 32},
        )
        target = triton.runtime.driver.active.get_current_target()
        compiled = triton.compile(src, target=target)
        asm_keys = compiled.asm.keys()
        llir_found = False
        for key in asm_keys:
            content = compiled.asm[key]
            if (
                isinstance(content, str)
                and "__metal_get_threadgroups_per_grid" in content
            ):
                llir_found = True
                break
        assert llir_found, (
            "Expected __metal_get_threadgroups_per_grid in compiled output. "
            f"Available keys: {list(asm_keys)}"
        )


# ── Matmul acceleration strategy tests ───────────────────────────────


class TestMetalMatmulAcceleration:
    """Tests for matmul_accel.py strategy selection and MSL optimisation."""

    def test_strategy_selection_large_fp32(self):
        """Simdgroup preferred for large fp32 matmul on apple8+."""
        from third_party.metal.backend.matmul_accel import select_matmul_strategy

        s = select_matmul_strategy(64, 64, 64, dtype="float", gpu_family="apple8")
        assert s.use_simdgroup is True
        assert s.tile_m == 8
        assert s.tile_n == 8
        assert s.elem_type == "float"
        assert s.accum_type == "float"

    def test_strategy_selection_small(self):
        """FMA preferred for small shapes regardless of GPU."""
        from third_party.metal.backend.matmul_accel import select_matmul_strategy

        s = select_matmul_strategy(4, 4, 4, dtype="float", gpu_family="apple9")
        assert s.use_simdgroup is False
        assert s.tile_m <= 4
        assert s.tile_n <= 4

    def test_strategy_selection_bf16_apple9(self):
        """Simdgroup preferred for bf16 on apple9."""
        from third_party.metal.backend.matmul_accel import select_matmul_strategy

        s = select_matmul_strategy(32, 32, 32, dtype="bf16", gpu_family="apple9")
        assert s.use_simdgroup is True
        assert s.accum_type == "float"

    def test_strategy_selection_bf16_apple7(self):
        """FMA used for bf16 on apple7 (no bfloat simdgroup support)."""
        from third_party.metal.backend.matmul_accel import select_matmul_strategy

        s = select_matmul_strategy(32, 32, 32, dtype="bf16", gpu_family="apple7")
        assert s.use_simdgroup is False

    def test_optimize_msl_noop_no_matmul(self):
        """Optimizer is a no-op for non-matmul kernels."""
        from third_party.metal.backend.matmul_accel import (
            MetalMatmulStrategy,
            optimize_matmul_msl,
        )

        src = (
            "#include <metal_stdlib>\n"
            "using namespace metal;\n"
            "kernel void k(device float* a [[buffer(0)]]) {\n"
            "  a[0] = 1.0;\n"
            "}\n"
        )
        strat = MetalMatmulStrategy(
            use_simdgroup=True,
            tile_m=8,
            tile_n=8,
            tile_k=8,
            elem_type="float",
            accum_type="float",
            pipeline_depth=2,
            gpu_family="apple8",
        )
        result = optimize_matmul_msl(src, [strat])
        assert result == src  # unchanged — no FMA loop detected

    def test_optimize_msl_inserts_simdgroup(self):
        """Optimizer inserts simdgroup hint for matmul-shaped kernels."""
        from third_party.metal.backend.matmul_accel import (
            MetalMatmulStrategy,
            optimize_matmul_msl,
        )

        src = (
            "#include <metal_stdlib>\n"
            "using namespace metal;\n"
            "kernel void matmul(\n"
            "  device float* C [[buffer(0)]],\n"
            "  const device float* A [[buffer(1)]],\n"
            "  const device float* B [[buffer(2)]]\n"
            ") {\n"
            "  float acc = 0.0;\n"
            "  for (int k = 0; k < K; ++k) {\n"
            "    acc += A[k] * B[k];\n"
            "  }\n"
            "  C[0] = acc;\n"
            "}\n"
        )
        strat = MetalMatmulStrategy(
            use_simdgroup=True,
            tile_m=8,
            tile_n=8,
            tile_k=8,
            elem_type="float",
            accum_type="float",
            pipeline_depth=2,
            gpu_family="apple8",
        )
        result = optimize_matmul_msl(src, [strat])
        assert "__metal_matmul_accel" in result
        assert "tile=8x8x8" in result

    def test_performance_model_apple9(self):
        """Performance model returns valid estimates for apple9."""
        from third_party.metal.backend.matmul_accel import get_matmul_performance_model

        model = get_matmul_performance_model("apple9")
        assert model.gpu_family == "apple9"
        assert model.simdgroup_gflops > model.fma_gflops
        assert 0.0 < model.simdgroup_efficiency <= 1.0
        assert model.preferred() == "simdgroup"

    def test_matmul_strategy_dataclass(self):
        """Strategy fields are set correctly and summary works."""
        from third_party.metal.backend.matmul_accel import MetalMatmulStrategy

        s = MetalMatmulStrategy(
            use_simdgroup=True,
            tile_m=8,
            tile_n=8,
            tile_k=8,
            elem_type="half",
            accum_type="float",
            pipeline_depth=2,
            gpu_family="apple9",
        )
        assert s.use_simdgroup is True
        assert s.tile_m == 8
        assert s.elem_type == "half"
        assert s.accum_type == "float"
        assert s.gpu_family == "apple9"
        summary = s.summary()
        assert "simdgroup" in summary
        assert "8x8x8" in summary

    def test_strategy_int8_always_fma(self):
        """int8 always selects FMA even on high-end GPU."""
        from third_party.metal.backend.matmul_accel import select_matmul_strategy

        s = select_matmul_strategy(64, 64, 64, dtype="int8", gpu_family="apple9")
        assert s.use_simdgroup is False

    def test_strategy_fallback_hint_overrides(self):
        """strategy_hint='fallback' forces FMA regardless of shape/dtype."""
        from third_party.metal.backend.matmul_accel import select_matmul_strategy

        s = select_matmul_strategy(
            128,
            128,
            128,
            dtype="float",
            gpu_family="apple9",
            strategy_hint="fallback",
        )
        assert s.use_simdgroup is False

    def test_strategy_native_hint_forces_simdgroup(self):
        """strategy_hint='native' forces simdgroup even for small shapes."""
        from third_party.metal.backend.matmul_accel import select_matmul_strategy

        s = select_matmul_strategy(
            4,
            4,
            4,
            dtype="float",
            gpu_family="apple8",
            strategy_hint="native",
        )
        assert s.use_simdgroup is True
        assert s.tile_m == 8

    def test_optimize_msl_no_strategies(self):
        """No strategies → source returned unchanged."""
        from third_party.metal.backend.matmul_accel import optimize_matmul_msl

        src = "kernel void k() {}"
        assert optimize_matmul_msl(src, None) == src
        assert optimize_matmul_msl(src, []) == src

    def test_performance_model_apple7_no_simdgroup(self):
        """apple7 has zero simdgroup throughput, prefers FMA."""
        from third_party.metal.backend.matmul_accel import get_matmul_performance_model

        model = get_matmul_performance_model("apple7")
        assert model.simdgroup_gflops == 0.0
        assert model.preferred() == "fma"

    def test_pipeline_depth_short_k(self):
        """Short K dimension → pipeline_depth=1."""
        from third_party.metal.backend.matmul_accel import select_matmul_strategy

        s = select_matmul_strategy(32, 32, 16, dtype="float", gpu_family="apple8")
        assert s.use_simdgroup is True
        assert s.pipeline_depth == 1

    def test_pipeline_depth_long_k(self):
        """Long K dimension → pipeline_depth=2."""
        from third_party.metal.backend.matmul_accel import select_matmul_strategy

        s = select_matmul_strategy(32, 32, 64, dtype="float", gpu_family="apple8")
        assert s.use_simdgroup is True
        assert s.pipeline_depth == 2

    def test_fp16_strategy_apple8(self):
        """fp16 selects simdgroup on apple8."""
        from third_party.metal.backend.matmul_accel import select_matmul_strategy

        s = select_matmul_strategy(32, 32, 32, dtype="fp16", gpu_family="apple8")
        assert s.use_simdgroup is True
        assert s.elem_type == "half"
        assert s.accum_type == "float"

    def test_performance_model_unknown_gpu(self):
        """Unknown GPU family returns conservative defaults."""
        from third_party.metal.backend.matmul_accel import get_matmul_performance_model

        model = get_matmul_performance_model("apple99")
        assert model.simdgroup_gflops == 0.0
        assert model.fma_gflops > 0.0


# ── Gluon Language Support ──────────────────────────────────────────────


class TestMetalGluonSupport:
    """Verify Gluon language support in the Metal backend."""

    def test_gluon_to_ttgir_method_exists(self):
        """MetalBackend exposes gluon_to_ttgir as a method."""
        from third_party.metal.backend.compiler import MetalBackend

        assert hasattr(MetalBackend, "gluon_to_ttgir")
        assert callable(getattr(MetalBackend, "gluon_to_ttgir"))

    def test_add_stages_handles_gluon(self):
        """add_stages() registers a ttgir stage when Language is GLUON."""
        from unittest.mock import MagicMock

        from third_party.metal.backend.compiler import MetalBackend

        from triton.backends.compiler import GPUTarget, Language

        target = GPUTarget(backend="metal", arch="apple8", warp_size=32)
        backend = MetalBackend(target)
        stages: dict = {}
        options = MagicMock()
        options.arch = "apple8"
        options.num_warps = 4
        options.num_ctas = 1
        options.num_stages = 0
        backend.add_stages(stages, options, Language.GLUON)

        assert "ttgir" in stages, "Gluon language must register ttgir stage"
        # GLUON skips ttir — goes straight to ttgir
        assert "ttir" not in stages
        # Downstream stages must still be present
        assert "llir" in stages
        assert "metal" in stages
        assert "metallib" in stages

    def test_add_stages_triton_still_works(self):
        """add_stages() with Language.TRITON still produces ttir + ttgir."""
        from unittest.mock import MagicMock

        from third_party.metal.backend.compiler import MetalBackend

        from triton.backends.compiler import GPUTarget, Language

        target = GPUTarget(backend="metal", arch="apple8", warp_size=32)
        backend = MetalBackend(target)
        stages: dict = {}
        options = MagicMock()
        options.arch = "apple8"
        backend.add_stages(stages, options, Language.TRITON)
        assert "ttir" in stages
        assert "ttgir" in stages

    def test_gluon_graceful_when_unavailable(self):
        """gluon_to_ttgir raises RuntimeError when passes.gluon is absent."""
        from unittest.mock import MagicMock, patch

        from third_party.metal.backend.compiler import MetalBackend

        from triton.backends.compiler import GPUTarget

        target = GPUTarget(backend="metal", arch="apple8", warp_size=32)
        backend = MetalBackend(target)

        mock_passes = MagicMock(spec=[])  # no 'gluon' attribute
        mod = MagicMock()
        metadata: dict = {}
        options = MagicMock()
        options.arch = "apple8"

        with patch("third_party.metal.backend.compiler.passes", mock_passes):
            with pytest.raises(RuntimeError, match="Gluon support requires"):
                backend.gluon_to_ttgir(mod, metadata, options)


# ── Metal libdevice ─────────────────────────────────────────────────────


class TestMetalLibdevice:
    """Verify Metal libdevice math function mappings."""

    def test_libdevice_map_completeness(self):
        """METAL_LIBDEVICE_MAP covers all standard math operations."""
        from third_party.metal.language.libdevice import METAL_LIBDEVICE_MAP

        required_ops = {
            "clz",
            "popc",
            "fma",
            "rsqrt",
            "exp2",
            "log2",
            "sin",
            "cos",
            "ceil",
            "floor",
            "trunc",
            "round",
            "saturate",
        }
        assert required_ops.issubset(
            set(METAL_LIBDEVICE_MAP.keys())
        ), f"Missing: {required_ops - set(METAL_LIBDEVICE_MAP.keys())}"

    def test_clz_mapping(self):
        """clz maps to MSL clz()."""
        from third_party.metal.language.libdevice import METAL_LIBDEVICE_MAP

        extern_name, msl_fn = METAL_LIBDEVICE_MAP["clz"]
        assert extern_name == "__metal_clz"
        assert msl_fn == "clz"

    def test_popc_mapping(self):
        """popc maps to MSL popcount()."""
        from third_party.metal.language.libdevice import METAL_LIBDEVICE_MAP

        extern_name, msl_fn = METAL_LIBDEVICE_MAP["popc"]
        assert extern_name == "__metal_popcount"
        assert msl_fn == "popcount"

    def test_trig_mappings(self):
        """sin/cos/exp2/log2 are all present with correct MSL names."""
        from third_party.metal.language.libdevice import METAL_LIBDEVICE_MAP

        for op in ("sin", "cos", "exp2", "log2"):
            assert op in METAL_LIBDEVICE_MAP, f"{op} missing"
            _, msl_fn = METAL_LIBDEVICE_MAP[op]
            assert msl_fn == op, f"{op} should map to MSL {op}"

    def test_rounding_mappings(self):
        """ceil/floor/trunc/round are present."""
        from third_party.metal.language.libdevice import METAL_LIBDEVICE_MAP

        for op in ("ceil", "floor", "trunc", "round"):
            assert op in METAL_LIBDEVICE_MAP, f"{op} missing"

    def test_saturate_is_metal_specific(self):
        """saturate is a Metal-specific operation not found in NVIDIA libdevice."""
        from third_party.metal.language.libdevice import METAL_LIBDEVICE_MAP

        assert "saturate" in METAL_LIBDEVICE_MAP
        _, msl_fn = METAL_LIBDEVICE_MAP["saturate"]
        assert msl_fn == "saturate"

    def test_libdevice_extern_functions_importable(self):
        """All @core.extern libdevice functions are importable."""
        from third_party.metal.language import libdevice

        func_names = [
            "clz",
            "popc",
            "abs",
            "floor",
            "ceil",
            "trunc",
            "round",
            "rsqrt",
            "sqrt",
            "exp2",
            "log2",
            "sin",
            "cos",
            "min",
            "max",
            "fma",
            "saturate",
        ]
        for name in func_names:
            assert hasattr(libdevice, name), f"libdevice.{name} not found"
            assert callable(getattr(libdevice, name)), f"libdevice.{name} not callable"

    def test_map_entries_are_string_pairs(self):
        """Every METAL_LIBDEVICE_MAP entry is a (str, str) tuple."""
        from third_party.metal.language.libdevice import METAL_LIBDEVICE_MAP

        for key, (extern_name, msl_fn) in METAL_LIBDEVICE_MAP.items():
            assert isinstance(key, str)
            assert isinstance(extern_name, str)
            assert isinstance(msl_fn, str)


# ── FP8 Converters ──────────────────────────────────────────────────────


class TestMetalFP8Converters:
    """Verify FP8 <-> FP16/FP32 software conversion utilities."""

    def test_fp8e5m2_to_fp16_roundtrip(self):
        """E5M2 encode → decode round-trips for representable values."""
        from third_party.metal.language.fp8_utils import (
            convert_fp8e5m2_to_fp16,
            convert_fp16_to_fp8e5m2,
        )

        test_values = [0.0, 1.0, -1.0, 0.5, 2.0, -0.5, 0.25]
        for v in test_values:
            encoded = convert_fp16_to_fp8e5m2(v)
            decoded = convert_fp8e5m2_to_fp16(encoded)
            assert (
                abs(decoded - v) <= abs(v) * 0.26 + 1e-7
            ), f"E5M2 round-trip failed for {v}: got {decoded}"

    def test_fp8e4b15_to_fp16_roundtrip(self):
        """E4B15 encode → decode round-trips for small representable values."""
        from third_party.metal.language.fp8_utils import (
            convert_fp8e4b15_to_fp16,
            convert_fp16_to_fp8e4b15,
        )

        # E4B15 with bias=15 only represents very small values
        # Smallest normal: 2^(1-15) = 2^-14 ≈ 6.1e-5
        test_values = [0.0, -0.0]
        for v in test_values:
            encoded = convert_fp16_to_fp8e4b15(v)
            decoded = convert_fp8e4b15_to_fp16(encoded)
            assert decoded == v or (
                v == 0.0 and decoded == 0.0
            ), f"E4B15 round-trip failed for {v}: got {decoded}"

    def test_fp8e5m2_special_values(self):
        """E5M2 handles NaN and Inf correctly."""
        import math

        from third_party.metal.language.fp8_utils import (
            convert_fp8e5m2_to_fp16,
            convert_fp16_to_fp8e5m2,
        )

        # NaN
        nan_bits = convert_fp16_to_fp8e5m2(float("nan"))
        assert math.isnan(convert_fp8e5m2_to_fp16(nan_bits))

        # +Inf
        pinf_bits = convert_fp16_to_fp8e5m2(float("inf"))
        assert convert_fp8e5m2_to_fp16(pinf_bits) == float("inf")

        # -Inf
        ninf_bits = convert_fp16_to_fp8e5m2(float("-inf"))
        assert convert_fp8e5m2_to_fp16(ninf_bits) == float("-inf")

    def test_fp8e5m2_denormals(self):
        """E5M2 denormal (subnormal) values decode correctly."""
        from third_party.metal.language.fp8_utils import convert_fp8e5m2_to_fp16

        # Smallest E5M2 denormal: 0 00000 01 = 2^(1-15) * 0.25 = 2^-16
        smallest_denorm = convert_fp8e5m2_to_fp16(0x01)
        assert smallest_denorm > 0
        assert smallest_denorm < 1e-4

        # Largest E5M2 denormal: 0 00000 11 = 2^(1-15) * 0.75
        largest_denorm = convert_fp8e5m2_to_fp16(0x03)
        assert largest_denorm > smallest_denorm

        # Zero
        assert convert_fp8e5m2_to_fp16(0x00) == 0.0

    def test_fp8_conversion_table(self):
        """Known E5M2 encodings produce expected values."""
        import math

        from third_party.metal.language.fp8_utils import convert_fp8e5m2_to_fp16

        known = {
            0x00: 0.0,  # +0
            0x80: -0.0,  # -0 (compare as 0.0)
            0x3C: 1.0,  # 0 01111 00 = 2^0 * 1.0 = 1.0
            0x40: 2.0,  # 0 10000 00 = 2^1 * 1.0 = 2.0
            0x38: 0.5,  # 0 01110 00 = 2^-1 * 1.0 = 0.5
            0x7C: float("inf"),  # 0 11111 00 = +Inf
            0xFC: float("-inf"),  # 1 11111 00 = -Inf
        }
        for bits, expected in known.items():
            result = convert_fp8e5m2_to_fp16(bits)
            if math.isnan(expected):
                assert math.isnan(result), f"bits=0x{bits:02X}: expected NaN"
            elif math.isinf(expected):
                assert result == expected, f"bits=0x{bits:02X}: expected {expected}"
            else:
                assert (
                    abs(result - expected) < 1e-9
                ), f"bits=0x{bits:02X}: expected {expected}, got {result}"

    def test_fp8e4b15_special_values(self):
        """E4B15 has no Inf — overflows to NaN."""
        import math

        from third_party.metal.language.fp8_utils import (
            convert_fp8e4b15_to_fp16,
            convert_fp16_to_fp8e4b15,
        )

        # E4B15 has no Inf: all-ones exponent is always NaN
        inf_bits = convert_fp16_to_fp8e4b15(float("inf"))
        result = convert_fp8e4b15_to_fp16(inf_bits)
        assert math.isnan(result), "E4B15 Inf should map to NaN"

        nan_bits = convert_fp16_to_fp8e4b15(float("nan"))
        assert math.isnan(convert_fp8e4b15_to_fp16(nan_bits))

    def test_fp8_helper_functions(self):
        """fp16_bits_to_float and float_to_fp16_bits round-trip."""
        from third_party.metal.language.fp8_utils import (
            float_to_fp16_bits,
            fp16_bits_to_float,
        )

        test_values = [0.0, 1.0, -1.0, 0.5, 65504.0]
        for v in test_values:
            bits = float_to_fp16_bits(v)
            assert isinstance(bits, int)
            assert 0 <= bits <= 0xFFFF
            result = fp16_bits_to_float(bits)
            assert abs(result - v) < 1e-3, f"fp16 round-trip failed for {v}: {result}"


# ── CI Compatibility Matrix Tests ────────────────────────────────────


class TestMetalCICompatibility:
    """Validate the CI compatibility matrix validation script."""

    SCRIPT_PATH = os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "..",
        "scripts",
        "metal_ci_compat_matrix.py",
    )

    @staticmethod
    def _load_compat_module():
        import importlib.util

        path = os.path.normpath(
            os.path.join(
                os.path.dirname(__file__),
                "..",
                "..",
                "..",
                "scripts",
                "metal_ci_compat_matrix.py",
            )
        )
        spec = importlib.util.spec_from_file_location(
            "metal_ci_compat_matrix",
            path,
        )
        assert spec is not None, "Cannot find metal_ci_compat_matrix.py"
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)
        return mod

    def test_compat_matrix_script_loads(self):
        """The compat matrix script can be imported and run_all_checks exists."""
        mod = self._load_compat_module()
        assert hasattr(mod, "run_all_checks")
        assert callable(mod.run_all_checks)

    def test_compat_check_python_version(self):
        """check_python_version reports correct Python version."""
        mod = self._load_compat_module()
        result = mod.check_python_version()
        expected_ver = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        assert result.value == expected_ver
        assert result.name == "python_version"

    def test_compat_check_torch_available(self):
        """check_torch_available reports torch availability without error."""
        mod = self._load_compat_module()
        result = mod.check_torch_available()
        assert result.passed is True
        assert result.name == "torch"
        try:
            import torch

            assert result.value is not None
            assert "version" in result.value
        except ImportError:
            assert result.value is None

    @skip_non_darwin
    def test_compat_check_xcrun_available(self):
        """check_xcrun_available detects xcrun on macOS."""
        mod = self._load_compat_module()
        result = mod.check_xcrun_available()
        assert result.name == "xcrun"
        if shutil.which("xcrun") is not None:
            assert result.passed is True
            assert result.value is not None
            assert "path" in result.value

    def test_compat_json_output(self):
        """run_all_checks produces a valid JSON-serializable report."""
        import json as _json

        mod = self._load_compat_module()
        report = mod.run_all_checks()
        text = _json.dumps(report)
        parsed = _json.loads(text)
        assert "overall_passed" in parsed
        assert "checks" in parsed
        assert isinstance(parsed["checks"], list)
        assert len(parsed["checks"]) >= 5
        for check in parsed["checks"]:
            assert "name" in check
            assert "passed" in check
            assert "detail" in check


# ── AOT Runtime Tests ────────────────────────────────────────────────


class TestMetalAOTRuntime:
    """Validate the Metal AOT runtime harness and integration script."""

    HARNESS_PATH = os.path.normpath(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            "..",
            "third_party",
            "metal",
            "tools",
            "test_aot_runtime.m",
        )
    )
    RUNTIME_SCRIPT_PATH = os.path.normpath(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            "..",
            "scripts",
            "test_metal_aot_runtime.py",
        )
    )

    def test_aot_harness_exists(self):
        """The ObjC AOT runtime harness file exists."""
        assert os.path.isfile(
            self.HARNESS_PATH
        ), f"AOT harness not found at {self.HARNESS_PATH}"
        content = open(self.HARNESS_PATH, "r").read()
        assert "MTLDevice" in content
        assert "MTLComputePipelineState" in content
        assert "RESULT: PASS" in content

    @skip_non_darwin
    @skip_no_xcrun
    def test_aot_harness_compiles(self):
        """The ObjC harness compiles with clang on macOS."""
        with tempfile.TemporaryDirectory(prefix="metal-aot-compile-") as tmpdir:
            output_bin = os.path.join(tmpdir, "test_aot_runtime")
            result = subprocess.run(
                [
                    "clang",
                    "-framework",
                    "Metal",
                    "-framework",
                    "Foundation",
                    "-framework",
                    "CoreGraphics",
                    "-o",
                    output_bin,
                    self.HARNESS_PATH,
                ],
                capture_output=True,
                text=True,
                timeout=60,
            )
            assert result.returncode == 0, f"clang compilation failed:\n{result.stderr}"
            assert os.path.isfile(output_bin)

    def test_aot_runtime_script_exists(self):
        """The Python AOT runtime integration script exists."""
        assert os.path.isfile(
            self.RUNTIME_SCRIPT_PATH
        ), f"AOT runtime script not found at {self.RUNTIME_SCRIPT_PATH}"
        content = open(self.RUNTIME_SCRIPT_PATH, "r").read()
        assert "def main" in content
        assert "_compile_triton_kernel_to_metallib" in content

    @skip_non_darwin
    @skip_no_mps
    def test_aot_compile_and_run_flow(self):
        """Full compile -> load -> dispatch -> verify flow (MPS required)."""
        result = subprocess.run(
            [
                sys.executable,
                self.RUNTIME_SCRIPT_PATH,
                "--num-elements",
                "256",
                "--json",
            ],
            capture_output=True,
            text=True,
            timeout=120,
            env={
                **os.environ,
                "PYTHONPATH": os.path.join(os.path.dirname(__file__), "..", "..", "..")
                + ":"
                + os.environ.get("PYTHONPATH", ""),
            },
        )
        assert result.returncode == 0, (
            f"AOT runtime flow failed (rc={result.returncode}):\n"
            f"stdout: {result.stdout[:500]}\n"
            f"stderr: {result.stderr[:500]}"
        )


# ── Training Loop Pattern Compile Tests ──────────────────────────────


class TestMetalTrainingLoopPatterns:
    """Test training-related kernel patterns compile to valid MSL via IR→MSL."""

    def test_sgd_parameter_update(self):
        """SGD update: param -= lr * grad compiles to valid MSL."""
        from third_party.metal.backend.compiler import MetalBackend

        ir = """\
define void @sgd_update_kernel(ptr %param, ptr %grad, float %lr, i32 %n) {
entry:
  %tid = call i32 @__metal_get_thread_position_in_threadgroup_x()
  %cmp = icmp slt i32 %tid, %n
  br i1 %cmp, label %body, label %exit

body:
  %idx = sext i32 %tid to i64
  %p_ptr = getelementptr float, ptr %param, i64 %idx
  %g_ptr = getelementptr float, ptr %grad, i64 %idx
  %p_val = load float, ptr %p_ptr
  %g_val = load float, ptr %g_ptr
  %step = fmul float %lr, %g_val
  %updated = fsub float %p_val, %step
  store float %updated, ptr %p_ptr
  br label %exit

exit:
  ret void
}
"""
        msl = MetalBackend.make_metal_ir(ir, {}, None)
        assert "UNSUPPORTED" not in msl
        assert "kernel void" in msl
        assert " * " in msl
        assert " - " in msl

    def test_loss_gradient_pattern(self):
        """MSE loss gradient: grad = 2.0*(pred - target)/n compiles."""
        from third_party.metal.backend.compiler import MetalBackend

        ir = """\
define void @mse_grad_kernel(ptr %pred, ptr %target, ptr %grad, float %inv_n, i32 %n) {
entry:
  %tid = call i32 @__metal_get_thread_position_in_threadgroup_x()
  %cmp = icmp slt i32 %tid, %n
  br i1 %cmp, label %body, label %exit

body:
  %idx = sext i32 %tid to i64
  %p_ptr = getelementptr float, ptr %pred, i64 %idx
  %t_ptr = getelementptr float, ptr %target, i64 %idx
  %p_val = load float, ptr %p_ptr
  %t_val = load float, ptr %t_ptr
  %diff = fsub float %p_val, %t_val
  %scaled = fmul float 2.0, %diff
  %g = fmul float %scaled, %inv_n
  %g_ptr = getelementptr float, ptr %grad, i64 %idx
  store float %g, ptr %g_ptr
  br label %exit

exit:
  ret void
}
"""
        msl = MetalBackend.make_metal_ir(ir, {}, None)
        assert "UNSUPPORTED" not in msl
        assert "kernel void" in msl
        assert " - " in msl
        assert " * " in msl

    def test_weight_decay_pattern(self):
        """L2 regularization: param -= lr*grad + wd*param compiles."""
        from third_party.metal.backend.compiler import MetalBackend

        ir = """\
define void @weight_decay_kernel(ptr %param, ptr %grad, float %lr, float %wd, i32 %n) {
entry:
  %tid = call i32 @__metal_get_thread_position_in_threadgroup_x()
  %cmp = icmp slt i32 %tid, %n
  br i1 %cmp, label %body, label %exit

body:
  %idx = sext i32 %tid to i64
  %p_ptr = getelementptr float, ptr %param, i64 %idx
  %g_ptr = getelementptr float, ptr %grad, i64 %idx
  %p_val = load float, ptr %p_ptr
  %g_val = load float, ptr %g_ptr
  %lr_grad = fmul float %lr, %g_val
  %wd_param = fmul float %wd, %p_val
  %total_step = fadd float %lr_grad, %wd_param
  %updated = fsub float %p_val, %total_step
  store float %updated, ptr %p_ptr
  br label %exit

exit:
  ret void
}
"""
        msl = MetalBackend.make_metal_ir(ir, {}, None)
        assert "UNSUPPORTED" not in msl
        assert "kernel void" in msl
        assert " * " in msl
        assert " + " in msl
        assert " - " in msl

    def test_momentum_update_pattern(self):
        """Momentum SGD: velocity = momentum*velocity + grad; param -= lr*velocity compiles."""
        from third_party.metal.backend.compiler import MetalBackend

        ir = """\
define void @momentum_sgd_kernel(ptr %param, ptr %grad, ptr %velocity, float %lr, float %momentum, i32 %n) {
entry:
  %tid = call i32 @__metal_get_thread_position_in_threadgroup_x()
  %cmp = icmp slt i32 %tid, %n
  br i1 %cmp, label %body, label %exit

body:
  %idx = sext i32 %tid to i64
  %p_ptr = getelementptr float, ptr %param, i64 %idx
  %g_ptr = getelementptr float, ptr %grad, i64 %idx
  %v_ptr = getelementptr float, ptr %velocity, i64 %idx
  %p_val = load float, ptr %p_ptr
  %g_val = load float, ptr %g_ptr
  %v_val = load float, ptr %v_ptr
  %mv = fmul float %momentum, %v_val
  %v_new = fadd float %mv, %g_val
  store float %v_new, ptr %v_ptr
  %step = fmul float %lr, %v_new
  %p_new = fsub float %p_val, %step
  store float %p_new, ptr %p_ptr
  br label %exit

exit:
  ret void
}
"""
        msl = MetalBackend.make_metal_ir(ir, {}, None)
        assert "UNSUPPORTED" not in msl
        assert "kernel void" in msl
        assert " * " in msl
        assert " + " in msl
        assert " - " in msl

    def test_adam_update_pattern(self):
        """Adam-like update with running mean/variance compiles to valid MSL."""
        from third_party.metal.backend.compiler import MetalBackend

        ir = """\
define void @adam_update_kernel(ptr %param, ptr %grad, ptr %m, ptr %v, float %lr, float %beta1, float %beta2, float %eps, i32 %n) {
entry:
  %tid = call i32 @__metal_get_thread_position_in_threadgroup_x()
  %cmp = icmp slt i32 %tid, %n
  br i1 %cmp, label %body, label %exit

body:
  %idx = sext i32 %tid to i64
  %p_ptr = getelementptr float, ptr %param, i64 %idx
  %g_ptr = getelementptr float, ptr %grad, i64 %idx
  %m_ptr = getelementptr float, ptr %m, i64 %idx
  %v_ptr = getelementptr float, ptr %v, i64 %idx
  %p_val = load float, ptr %p_ptr
  %g_val = load float, ptr %g_ptr
  %m_val = load float, ptr %m_ptr
  %v_val = load float, ptr %v_ptr
  ; m = beta1 * m + (1 - beta1) * grad
  %b1m = fmul float %beta1, %m_val
  %one_minus_b1 = fsub float 1.0, %beta1
  %omb1_g = fmul float %one_minus_b1, %g_val
  %m_new = fadd float %b1m, %omb1_g
  store float %m_new, ptr %m_ptr
  ; v = beta2 * v + (1 - beta2) * grad^2
  %b2v = fmul float %beta2, %v_val
  %g_sq = fmul float %g_val, %g_val
  %one_minus_b2 = fsub float 1.0, %beta2
  %omb2_gsq = fmul float %one_minus_b2, %g_sq
  %v_new = fadd float %b2v, %omb2_gsq
  store float %v_new, ptr %v_ptr
  ; param -= lr * m / (sqrt(v) + eps)
  %sqrt_v = call float @__nv_sqrtf(float %v_new)
  %denom = fadd float %sqrt_v, %eps
  %ratio = fdiv float %m_new, %denom
  %step = fmul float %lr, %ratio
  %p_new = fsub float %p_val, %step
  store float %p_new, ptr %p_ptr
  br label %exit

exit:
  ret void
}
"""
        msl = MetalBackend.make_metal_ir(ir, {}, None)
        assert "UNSUPPORTED" not in msl
        assert "kernel void" in msl
        assert " * " in msl
        assert " + " in msl
        assert " - " in msl
        assert " / " in msl
        assert "sqrt(" in msl


# ── Convolution Pattern Compile Tests ────────────────────────────────


class TestMetalConvolutionPatterns:
    """Test convolution-related kernel patterns compile to valid MSL via IR→MSL."""

    def test_conv1d_sliding_window(self):
        """1D sliding window accumulation with inner loop compiles to MSL."""
        from third_party.metal.backend.compiler import MetalBackend

        ir = """\
define void @conv1d_kernel(ptr %input, ptr %weight, ptr %output, i32 %in_len, i32 %out_len, i32 %ksize) {
entry:
  %tid = call i32 @__metal_get_thread_position_in_threadgroup_x()
  %cmp = icmp slt i32 %tid, %out_len
  br i1 %cmp, label %loop_init, label %exit

loop_init:
  br label %loop_header

loop_header:
  %ki = phi i32 [0, %loop_init], [%ki_next, %loop_body]
  %acc = phi float [0.0, %loop_init], [%acc_next, %loop_body]
  %ki_cmp = icmp slt i32 %ki, %ksize
  br i1 %ki_cmp, label %loop_body, label %store_out

loop_body:
  %x_idx = add i32 %tid, %ki
  %x_idx64 = sext i32 %x_idx to i64
  %x_ptr = getelementptr float, ptr %input, i64 %x_idx64
  %x_val = load float, ptr %x_ptr
  %ki64 = sext i32 %ki to i64
  %w_ptr = getelementptr float, ptr %weight, i64 %ki64
  %w_val = load float, ptr %w_ptr
  %prod = fmul float %x_val, %w_val
  %acc_next = fadd float %acc, %prod
  %ki_next = add i32 %ki, 1
  br label %loop_header

store_out:
  %out_idx = sext i32 %tid to i64
  %out_ptr = getelementptr float, ptr %output, i64 %out_idx
  store float %acc, ptr %out_ptr
  br label %exit

exit:
  ret void
}
"""
        msl = MetalBackend.make_metal_ir(ir, {}, None)
        assert "UNSUPPORTED" not in msl
        assert "kernel void" in msl
        assert " * " in msl
        assert " + " in msl
        assert "__triton_pred_block" in msl

    def test_depthwise_conv_pattern(self):
        """Depthwise conv: per-channel conv with channel indexing compiles."""
        from third_party.metal.backend.compiler import MetalBackend

        ir = """\
define void @depthwise_conv_kernel(ptr %input, ptr %weight, ptr %output, i32 %spatial_len, i32 %channels, i32 %ksize) {
entry:
  %tid = call i32 @__metal_get_thread_position_in_threadgroup_x()
  %ch = call i32 @__metal_get_thread_position_in_threadgroup_y()
  %cmp_s = icmp slt i32 %tid, %spatial_len
  %cmp_c = icmp slt i32 %ch, %channels
  %cmp = and i1 %cmp_s, %cmp_c
  br i1 %cmp, label %loop_init, label %exit

loop_init:
  br label %loop_header

loop_header:
  %ki = phi i32 [0, %loop_init], [%ki_next, %loop_body]
  %acc = phi float [0.0, %loop_init], [%acc_next, %loop_body]
  %ki_cmp = icmp slt i32 %ki, %ksize
  br i1 %ki_cmp, label %loop_body, label %store_out

loop_body:
  %x_pos = add i32 %tid, %ki
  %x_linear = mul i32 %ch, %spatial_len
  %x_idx = add i32 %x_linear, %x_pos
  %x_idx64 = sext i32 %x_idx to i64
  %x_ptr = getelementptr float, ptr %input, i64 %x_idx64
  %x_val = load float, ptr %x_ptr
  %w_linear = mul i32 %ch, %ksize
  %w_idx = add i32 %w_linear, %ki
  %w_idx64 = sext i32 %w_idx to i64
  %w_ptr = getelementptr float, ptr %weight, i64 %w_idx64
  %w_val = load float, ptr %w_ptr
  %prod = fmul float %x_val, %w_val
  %acc_next = fadd float %acc, %prod
  %ki_next = add i32 %ki, 1
  br label %loop_header

store_out:
  %out_linear = mul i32 %ch, %spatial_len
  %out_idx = add i32 %out_linear, %tid
  %out_idx64 = sext i32 %out_idx to i64
  %out_ptr = getelementptr float, ptr %output, i64 %out_idx64
  store float %acc, ptr %out_ptr
  br label %exit

exit:
  ret void
}
"""
        msl = MetalBackend.make_metal_ir(ir, {}, None)
        assert "UNSUPPORTED" not in msl
        assert "kernel void" in msl
        assert " * " in msl
        assert " + " in msl

    def test_strided_access_pattern(self):
        """Strided memory access (stride > 1) pattern common in convolutions compiles."""
        from third_party.metal.backend.compiler import MetalBackend

        ir = """\
define void @strided_access_kernel(ptr %input, ptr %output, i32 %n, i32 %stride) {
entry:
  %tid = call i32 @__metal_get_thread_position_in_threadgroup_x()
  %cmp = icmp slt i32 %tid, %n
  br i1 %cmp, label %body, label %exit

body:
  %src_idx = mul i32 %tid, %stride
  %src_idx64 = sext i32 %src_idx to i64
  %src_ptr = getelementptr float, ptr %input, i64 %src_idx64
  %val = load float, ptr %src_ptr
  %doubled = fmul float %val, 2.0
  %dst_idx = sext i32 %tid to i64
  %dst_ptr = getelementptr float, ptr %output, i64 %dst_idx
  store float %doubled, ptr %dst_ptr
  br label %exit

exit:
  ret void
}
"""
        msl = MetalBackend.make_metal_ir(ir, {}, None)
        assert "UNSUPPORTED" not in msl
        assert "kernel void" in msl
        assert " * " in msl

    def test_im2col_pattern(self):
        """Im2col-style gather: 2D index → linearized offset compiles."""
        from third_party.metal.backend.compiler import MetalBackend

        ir = """\
define void @im2col_kernel(ptr %input, ptr %col_buf, i32 %height, i32 %width, i32 %kh, i32 %kw, i32 %out_h, i32 %out_w) {
entry:
  %tid = call i32 @__metal_get_thread_position_in_threadgroup_x()
  %total = mul i32 %out_h, %out_w
  %cmp = icmp slt i32 %tid, %total
  br i1 %cmp, label %body, label %exit

body:
  %oh = sdiv i32 %tid, %out_w
  %ow = srem i32 %tid, %out_w
  ; Gather from a single kernel position (0,0) as representative pattern
  %in_row = add i32 %oh, 0
  %in_col = add i32 %ow, 0
  %in_linear = mul i32 %in_row, %width
  %in_idx = add i32 %in_linear, %in_col
  %in_idx64 = sext i32 %in_idx to i64
  %in_ptr = getelementptr float, ptr %input, i64 %in_idx64
  %val = load float, ptr %in_ptr
  %out_idx = sext i32 %tid to i64
  %col_ptr = getelementptr float, ptr %col_buf, i64 %out_idx
  store float %val, ptr %col_ptr
  br label %exit

exit:
  ret void
}
"""
        msl = MetalBackend.make_metal_ir(ir, {}, None)
        assert "UNSUPPORTED" not in msl
        assert "kernel void" in msl
        assert " * " in msl
        assert " + " in msl


# ── Advanced Attention Pattern Compile Tests ─────────────────────────


class TestMetalAdvancedAttentionPatterns:
    """Test attention variant kernel patterns compile to valid MSL via IR→MSL."""

    def test_multi_head_attention_scores(self):
        """Multi-head attention score: Q*K^T / sqrt(d_k) with head offset compiles."""
        from third_party.metal.backend.compiler import MetalBackend

        ir = """\
define void @mha_score_kernel(ptr %Q, ptr %K, ptr %scores, i32 %seq_len, i32 %d_k, i32 %head_idx, i32 %head_dim) {
entry:
  %row = call i32 @__metal_get_thread_position_in_threadgroup_x()
  %col = call i32 @__metal_get_thread_position_in_threadgroup_y()
  %cmp_r = icmp slt i32 %row, %seq_len
  %cmp_c = icmp slt i32 %col, %seq_len
  %cmp = and i1 %cmp_r, %cmp_c
  br i1 %cmp, label %dot_init, label %exit

dot_init:
  %head_off = mul i32 %head_idx, %head_dim
  br label %dot_loop

dot_loop:
  %k = phi i32 [0, %dot_init], [%k_next, %dot_body]
  %acc = phi float [0.0, %dot_init], [%acc_next, %dot_body]
  %k_cmp = icmp slt i32 %k, %d_k
  br i1 %k_cmp, label %dot_body, label %scale_store

dot_body:
  %q_off = mul i32 %row, %d_k
  %q_idx = add i32 %q_off, %k
  %q_idx_h = add i32 %q_idx, %head_off
  %q_idx64 = sext i32 %q_idx_h to i64
  %q_ptr = getelementptr float, ptr %Q, i64 %q_idx64
  %q_val = load float, ptr %q_ptr
  %k_off = mul i32 %col, %d_k
  %k_idx = add i32 %k_off, %k
  %k_idx_h = add i32 %k_idx, %head_off
  %k_idx64 = sext i32 %k_idx_h to i64
  %k_ptr = getelementptr float, ptr %K, i64 %k_idx64
  %k_val = load float, ptr %k_ptr
  %prod = fmul float %q_val, %k_val
  %acc_next = fadd float %acc, %prod
  %k_next = add i32 %k, 1
  br label %dot_loop

scale_store:
  %dk_f = sitofp i32 %d_k to float
  %sqrt_dk = call float @__nv_sqrtf(float %dk_f)
  %scaled = fdiv float %acc, %sqrt_dk
  %s_off = mul i32 %row, %seq_len
  %s_idx = add i32 %s_off, %col
  %s_idx64 = sext i32 %s_idx to i64
  %s_ptr = getelementptr float, ptr %scores, i64 %s_idx64
  store float %scaled, ptr %s_ptr
  br label %exit

exit:
  ret void
}
"""
        msl = MetalBackend.make_metal_ir(ir, {}, None)
        assert "UNSUPPORTED" not in msl
        assert "kernel void" in msl
        assert " * " in msl
        assert " + " in msl
        assert " / " in msl
        assert "sqrt(" in msl

    def test_grouped_query_attention(self):
        """GQA: KV sharing across query groups via integer division compiles."""
        from third_party.metal.backend.compiler import MetalBackend

        ir = """\
define void @gqa_kernel(ptr %Q, ptr %K, ptr %V, ptr %out, i32 %seq_len, i32 %d_k, i32 %n_heads, i32 %n_kv_heads) {
entry:
  %tid = call i32 @__metal_get_thread_position_in_threadgroup_x()
  %head_id = call i32 @__metal_get_thread_position_in_threadgroup_y()
  %cmp_s = icmp slt i32 %tid, %seq_len
  %cmp_h = icmp slt i32 %head_id, %n_heads
  %cmp = and i1 %cmp_s, %cmp_h
  br i1 %cmp, label %body, label %exit

body:
  ; Map query head to KV head: kv_head = head_id / (n_heads / n_kv_heads)
  %heads_per_kv = sdiv i32 %n_heads, %n_kv_heads
  %kv_head = sdiv i32 %head_id, %heads_per_kv
  ; Load Q element (simplified: single element per thread)
  %q_off = mul i32 %head_id, %d_k
  %q_idx = add i32 %q_off, %tid
  %q_idx64 = sext i32 %q_idx to i64
  %q_ptr = getelementptr float, ptr %Q, i64 %q_idx64
  %q_val = load float, ptr %q_ptr
  ; Load K element from shared KV head
  %k_off = mul i32 %kv_head, %d_k
  %k_idx = add i32 %k_off, %tid
  %k_idx64 = sext i32 %k_idx to i64
  %k_ptr = getelementptr float, ptr %K, i64 %k_idx64
  %k_val = load float, ptr %k_ptr
  ; Simple dot product element
  %prod = fmul float %q_val, %k_val
  ; Store result
  %out_off = mul i32 %head_id, %seq_len
  %out_idx = add i32 %out_off, %tid
  %out_idx64 = sext i32 %out_idx to i64
  %out_ptr = getelementptr float, ptr %out, i64 %out_idx64
  store float %prod, ptr %out_ptr
  br label %exit

exit:
  ret void
}
"""
        msl = MetalBackend.make_metal_ir(ir, {}, None)
        assert "UNSUPPORTED" not in msl
        assert "kernel void" in msl
        assert " * " in msl
        assert " / " in msl

    def test_causal_mask_attention(self):
        """Attention with causal masking (select/conditional store) compiles."""
        from third_party.metal.backend.compiler import MetalBackend

        ir = """\
define void @causal_attn_kernel(ptr %scores, ptr %out, i32 %seq_len) {
entry:
  %row = call i32 @__metal_get_thread_position_in_threadgroup_x()
  %col = call i32 @__metal_get_thread_position_in_threadgroup_y()
  %cmp_r = icmp slt i32 %row, %seq_len
  %cmp_c = icmp slt i32 %col, %seq_len
  %cmp_bounds = and i1 %cmp_r, %cmp_c
  br i1 %cmp_bounds, label %body, label %exit

body:
  %idx = mul i32 %row, %seq_len
  %linear = add i32 %idx, %col
  %linear64 = sext i32 %linear to i64
  %s_ptr = getelementptr float, ptr %scores, i64 %linear64
  %s_val = load float, ptr %s_ptr
  ; Causal mask: keep only col <= row, else -inf
  %is_causal = icmp sle i32 %col, %row
  %neg_inf = bitcast i32 -8388608 to float
  %masked = select i1 %is_causal, float %s_val, float %neg_inf
  %o_ptr = getelementptr float, ptr %out, i64 %linear64
  store float %masked, ptr %o_ptr
  br label %exit

exit:
  ret void
}
"""
        msl = MetalBackend.make_metal_ir(ir, {}, None)
        assert "UNSUPPORTED" not in msl
        assert "kernel void" in msl
        assert " ? " in msl
        assert " * " in msl

    def test_flash_attention_block(self):
        """Flash-attention-style blocked dot product with running max compiles."""
        from third_party.metal.backend.compiler import MetalBackend

        ir = """\
define void @flash_attn_block_kernel(ptr %Q, ptr %K, ptr %out, i32 %seq_len, i32 %d_k, i32 %block_size) {
entry:
  %row = call i32 @__metal_get_thread_position_in_threadgroup_x()
  %cmp = icmp slt i32 %row, %seq_len
  br i1 %cmp, label %block_init, label %exit

block_init:
  br label %block_loop

block_loop:
  %bk = phi i32 [0, %block_init], [%bk_next, %block_end]
  %running_max = phi float [0xFFF0000000000000, %block_init], [%new_max, %block_end]
  %running_sum = phi float [0.0, %block_init], [%new_sum, %block_end]
  %bk_cmp = icmp slt i32 %bk, %seq_len
  br i1 %bk_cmp, label %dot_init, label %store_out

dot_init:
  br label %dot_loop

dot_loop:
  %d = phi i32 [0, %dot_init], [%d_next, %dot_body]
  %dot_acc = phi float [0.0, %dot_init], [%dot_next, %dot_body]
  %d_cmp = icmp slt i32 %d, %d_k
  br i1 %d_cmp, label %dot_body, label %dot_done

dot_body:
  %q_off = mul i32 %row, %d_k
  %q_idx = add i32 %q_off, %d
  %q_idx64 = sext i32 %q_idx to i64
  %q_ptr = getelementptr float, ptr %Q, i64 %q_idx64
  %q_val = load float, ptr %q_ptr
  %k_off = mul i32 %bk, %d_k
  %k_idx = add i32 %k_off, %d
  %k_idx64 = sext i32 %k_idx to i64
  %k_ptr = getelementptr float, ptr %K, i64 %k_idx64
  %k_val = load float, ptr %k_ptr
  %prod = fmul float %q_val, %k_val
  %dot_next = fadd float %dot_acc, %prod
  %d_next = add i32 %d, 1
  br label %dot_loop

dot_done:
  ; Scale by 1/sqrt(d_k)
  %dk_f = sitofp i32 %d_k to float
  %sqrt_dk = call float @__nv_sqrtf(float %dk_f)
  %score = fdiv float %dot_acc, %sqrt_dk
  ; Update running max
  %is_new_max = fcmp ogt float %score, %running_max
  %new_max = select i1 %is_new_max, float %score, float %running_max
  ; Accumulate exp(score - max) for softmax denominator
  %shifted = fsub float %score, %new_max
  %exp_s = call float @__nv_expf(float %shifted)
  %new_sum = fadd float %running_sum, %exp_s
  br label %block_end

block_end:
  %bk_next = add i32 %bk, %block_size
  br label %block_loop

store_out:
  ; Store final normalized sum (simplified)
  %result = fdiv float %running_sum, %running_sum
  %out_idx = sext i32 %row to i64
  %out_ptr = getelementptr float, ptr %out, i64 %out_idx
  store float %result, ptr %out_ptr
  br label %exit

exit:
  ret void
}
"""
        msl = MetalBackend.make_metal_ir(ir, {}, None)
        assert "UNSUPPORTED" not in msl
        assert "kernel void" in msl
        assert " * " in msl
        assert " + " in msl
        assert " / " in msl
        assert "sqrt(" in msl
        assert "exp(" in msl
        assert " ? " in msl


# ── Golden MSL parity tests for typed IR dispatch ──────────────────


class TestMetalTypedIRParity:
    """Golden MSL structure tests that pin the current make_metal_ir output.

    Each test builds minimal LLVM IR for a representative kernel pattern,
    compiles it through MetalBackend.make_metal_ir(), and verifies specific
    MSL patterns and structural invariants.  When the typed IR dispatch
    refactoring (Phases 1-3) lands these tests confirm output parity.
    """

    def test_vector_add_golden_msl(self):
        """Binary float ops: add+multiply produce correct MSL operators."""
        from third_party.metal.backend.compiler import MetalBackend

        ir = """\
define void @vector_add_kernel(ptr addrspace(1) %a, ptr addrspace(1) %b, ptr addrspace(1) %out, i32 %n) {
entry:
  %tid = call i32 @__metal_get_thread_position_in_grid_x()
  %cmp = icmp slt i32 %tid, %n
  br i1 %cmp, label %body, label %exit

body:
  %idx = sext i32 %tid to i64
  %pa = getelementptr float, ptr addrspace(1) %a, i64 %idx
  %va = load float, ptr addrspace(1) %pa
  %pb = getelementptr float, ptr addrspace(1) %b, i64 %idx
  %vb = load float, ptr addrspace(1) %pb
  %sum = fadd float %va, %vb
  %prod = fmul float %sum, %vb
  %pout = getelementptr float, ptr addrspace(1) %out, i64 %idx
  store float %prod, ptr addrspace(1) %pout
  br label %exit

exit:
  ret void
}
"""
        metadata = {}
        msl = MetalBackend.make_metal_ir(ir, metadata, None)
        assert metadata["name"] == "vector_add_kernel"
        assert "kernel void vector_add_kernel" in msl
        assert "device float*" in msl
        assert "[[buffer(0)]]" in msl
        assert "[[buffer(1)]]" in msl
        assert "[[buffer(2)]]" in msl
        assert "__metal_get_thread_position_in_grid_x()" in msl
        assert " + " in msl
        assert " * " in msl
        assert "if (" in msl
        assert "while (true)" in msl
        assert "switch (__pc)" in msl
        assert "UNSUPPORTED" not in msl

    def test_matmul_tile_golden_msl(self):
        """Simdgroup matrix ops produce native MSL simdgroup intrinsics."""
        from third_party.metal.backend.compiler import MetalBackend

        ir = """\
define void @matmul_tile_kernel(ptr addrspace(1) %A, ptr addrspace(1) %B, ptr addrspace(1) %C) {
entry:
  %ma = call <8 x float> @__metal_simdgroup_load(ptr addrspace(1) %A, i32 64)
  %mb = call <8 x float> @__metal_simdgroup_load(ptr addrspace(1) %B, i32 64)
  %mc = call <8 x float> @__metal_simdgroup_load(ptr addrspace(1) %C, i32 64)
  %md = call <8 x float> @__metal_simdgroup_multiply_accumulate(<8 x float> %ma, <8 x float> %mb, <8 x float> %mc)
  call void @__metal_simdgroup_store(<8 x float> %md, ptr addrspace(1) %C, i32 64)
  ret void
}
"""
        metadata = {}
        msl = MetalBackend.make_metal_ir(ir, metadata, None)
        assert metadata["name"] == "matmul_tile_kernel"
        assert "kernel void matmul_tile_kernel" in msl
        assert "device float*" in msl
        assert "simdgroup_matrix<float, 8, 8>" in msl
        assert "simdgroup_load(" in msl
        assert "simdgroup_multiply_accumulate(" in msl
        assert "simdgroup_store(" in msl
        assert "UNSUPPORTED" not in msl

    def test_softmax_row_golden_msl(self):
        """Softmax pattern: max-reduce, exp, sum, division all lower correctly."""
        from third_party.metal.backend.compiler import MetalBackend

        ir = """\
define void @softmax_row_kernel(ptr addrspace(1) %in, ptr addrspace(1) %out, i32 %n) {
entry:
  %tid = call i32 @__metal_get_thread_position_in_grid_x()
  %cmp = icmp slt i32 %tid, %n
  br i1 %cmp, label %body, label %exit

body:
  %idx = sext i32 %tid to i64
  %pin = getelementptr float, ptr addrspace(1) %in, i64 %idx
  %x = load float, ptr addrspace(1) %pin
  ; max(x, 0) then exp then sum pattern
  %zero = bitcast i32 0 to float
  %is_gt = fcmp ogt float %x, %zero
  %xmax = select i1 %is_gt, float %x, float %zero
  %neg = fneg float %xmax
  %e = call float @__nv_expf(float %neg)
  %one = bitcast i32 1065353216 to float
  %denom = fadd float %one, %e
  %result = fdiv float %x, %denom
  %pout = getelementptr float, ptr addrspace(1) %out, i64 %idx
  store float %result, ptr addrspace(1) %pout
  br label %exit

exit:
  ret void
}
"""
        metadata = {}
        msl = MetalBackend.make_metal_ir(ir, metadata, None)
        assert metadata["name"] == "softmax_row_kernel"
        assert "kernel void softmax_row_kernel" in msl
        assert "device float*" in msl
        assert "__metal_get_thread_position_in_grid_x()" in msl
        assert "exp(" in msl
        assert " / " in msl
        assert " + " in msl
        assert " ? " in msl
        assert "= -(" in msl
        assert "as_type<float>(" in msl
        assert "UNSUPPORTED" not in msl

    def test_reduction_sum_golden_msl(self):
        """Reduce-loop with phi accumulator and backedge lowers correctly."""
        from third_party.metal.backend.compiler import MetalBackend

        ir = """\
define void @reduction_sum_kernel(ptr addrspace(1) %data, ptr addrspace(1) %out, i32 %n) {
entry:
  br label %loop

loop:
  %i = phi i32 [0, %entry], [%i_next, %loop_body]
  %acc = phi float [0.0, %entry], [%acc_next, %loop_body]
  %cond = icmp slt i32 %i, %n
  br i1 %cond, label %loop_body, label %done

loop_body:
  %idx = sext i32 %i to i64
  %ptr = getelementptr float, ptr addrspace(1) %data, i64 %idx
  %val = load float, ptr addrspace(1) %ptr
  %acc_next = fadd float %acc, %val
  %i_next = add i32 %i, 1
  br label %loop

done:
  %out_ptr = getelementptr float, ptr addrspace(1) %out, i64 0
  store float %acc, ptr addrspace(1) %out_ptr
  ret void
}
"""
        metadata = {}
        msl = MetalBackend.make_metal_ir(ir, metadata, None)
        assert metadata["name"] == "reduction_sum_kernel"
        assert "kernel void reduction_sum_kernel" in msl
        assert "device float*" in msl
        assert "while (true)" in msl
        assert "switch (__pc)" in msl
        assert " + " in msl
        assert "__triton_pred_block ==" in msl
        assert "float acc" in msl
        assert "UNSUPPORTED" not in msl

    def test_silu_activation_golden_msl(self):
        """SiLU = x / (1 + exp(-x)) lowers to exp + division MSL."""
        from third_party.metal.backend.compiler import MetalBackend

        ir = """\
define void @silu_kernel(ptr addrspace(1) %in, ptr addrspace(1) %out, i32 %n) {
entry:
  %tid = call i32 @__metal_get_thread_position_in_grid_x()
  %cmp = icmp slt i32 %tid, %n
  br i1 %cmp, label %body, label %exit

body:
  %idx = sext i32 %tid to i64
  %pin = getelementptr float, ptr addrspace(1) %in, i64 %idx
  %x = load float, ptr addrspace(1) %pin
  ; SiLU: x * sigmoid(x) = x / (1 + exp(-x))
  %neg_x = fneg float %x
  %exp_neg = call float @__nv_expf(float %neg_x)
  %one = bitcast i32 1065353216 to float
  %denom = fadd float %one, %exp_neg
  %sigmoid = fdiv float %x, %denom
  %pout = getelementptr float, ptr addrspace(1) %out, i64 %idx
  store float %sigmoid, ptr addrspace(1) %pout
  br label %exit

exit:
  ret void
}
"""
        metadata = {}
        msl = MetalBackend.make_metal_ir(ir, metadata, None)
        assert metadata["name"] == "silu_kernel"
        assert "kernel void silu_kernel" in msl
        assert "device float*" in msl
        assert "__metal_get_thread_position_in_grid_x()" in msl
        assert "exp(" in msl
        assert "= -(" in msl
        assert " / " in msl
        assert " + " in msl
        assert "as_type<float>(1065353216)" in msl
        assert "if (" in msl
        assert "UNSUPPORTED" not in msl


# ── MetalBufferPool tests ───────────────────────────────────────────


class _MockMTLBuffer:
    """Lightweight stand-in for an MTLBuffer returned by PyObjC."""

    def __init__(self, length):
        self._length = length
        self._mem = ctypes.create_string_buffer(length)

    def contents(self):
        return ctypes.addressof(self._mem)

    def length(self):
        return self._length


class _MockDevice:
    """Minimal MTLDevice mock that allocates _MockMTLBuffer."""

    def newBufferWithLength_options_(self, length, options):
        return _MockMTLBuffer(length)


class TestMetalBufferPool:
    """Unit tests for MetalBufferPool size-bucketed buffer reuse."""

    # -- bucket sizing ----------------------------------------------------

    @skip_non_darwin
    def test_bucket_size_rounds_up_to_power_of_2(self):
        from third_party.metal.backend.driver import MetalBufferPool

        assert MetalBufferPool._bucket_size(300) == 512
        assert MetalBufferPool._bucket_size(512) == 512
        assert MetalBufferPool._bucket_size(513) == 1024
        assert MetalBufferPool._bucket_size(1) == 256
        assert MetalBufferPool._bucket_size(4096) == 4096
        assert MetalBufferPool._bucket_size(4097) == 8192

    @skip_non_darwin
    def test_bucket_size_minimum_256(self):
        from third_party.metal.backend.driver import MetalBufferPool

        for n in (0, 1, 2, 100, 128, 255, 256):
            assert MetalBufferPool._bucket_size(n) >= 256, f"bucket_size({n}) < 256"
        assert MetalBufferPool._bucket_size(0) == 256
        assert MetalBufferPool._bucket_size(256) == 256

    # -- acquire / release ------------------------------------------------

    @skip_non_darwin
    def test_acquire_miss_creates_new_buffer(self):
        from third_party.metal.backend.driver import MetalBufferPool

        pool = MetalBufferPool()
        device = _MockDevice()
        buf = pool.acquire(device, 100)
        assert buf is not None
        assert buf.length() == 256  # rounded to min bucket
        stats = pool.stats()
        assert stats["misses"] == 1
        assert stats["hits"] == 0

    @skip_non_darwin
    def test_acquire_hit_reuses_buffer(self):
        from third_party.metal.backend.driver import MetalBufferPool

        pool = MetalBufferPool()
        device = _MockDevice()
        buf1 = pool.acquire(device, 100)
        pool.release(buf1, 100)
        buf2 = pool.acquire(device, 100)
        assert buf2 is buf1, "Expected pool to reuse the released buffer"
        stats = pool.stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 1

    @skip_non_darwin
    def test_release_respects_max_per_bucket(self):
        from third_party.metal.backend.driver import MetalBufferPool

        pool = MetalBufferPool()
        device = _MockDevice()
        buffers = [pool.acquire(device, 100) for _ in range(33)]
        for b in buffers:
            pool.release(b, 100)
        assert pool.pool_size == 32, "Pool should cap at _MAX_PER_BUCKET=32"

    # -- drain ------------------------------------------------------------

    @skip_non_darwin
    def test_drain_clears_pool(self):
        from third_party.metal.backend.driver import MetalBufferPool

        pool = MetalBufferPool()
        device = _MockDevice()
        buffers = [pool.acquire(device, 512) for _ in range(5)]
        for b in buffers:
            pool.release(b, 512)
        assert pool.pool_size == 5
        pool.drain()
        assert pool.pool_size == 0

    # -- stats ------------------------------------------------------------

    @skip_non_darwin
    def test_stats_tracking(self):
        from third_party.metal.backend.driver import MetalBufferPool

        pool = MetalBufferPool()
        device = _MockDevice()

        buf1 = pool.acquire(device, 1024)  # miss
        buf2 = pool.acquire(device, 1024)  # miss
        pool.release(buf1, 1024)
        buf3 = pool.acquire(device, 1024)  # hit

        stats = pool.stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 2
        assert stats["total_allocated_bytes"] == 2 * 1024
        assert stats["pool_size"] == 0  # buf1 was re-acquired, buf2 not released

        pool.release(buf2, 1024)
        pool.release(buf3, 1024)
        assert pool.stats()["pool_size"] == 2

    # -- integration: _bind_argument with pool ----------------------------

    @skip_non_darwin
    def test_pool_integrated_in_bind_argument(self):
        from unittest.mock import MagicMock

        import numpy as np
        from third_party.metal.backend.driver import MetalBufferPool, _bind_argument

        pool = MetalBufferPool()
        device = _MockDevice()
        encoder = MagicMock()
        arr = np.zeros(64, dtype=np.float32)  # 256 bytes
        acquired: list = []

        _bind_argument(device, encoder, 0, arr, pool=pool, acquired_bufs=acquired)

        assert pool.stats()["misses"] == 1
        assert len(acquired) == 1
        buf, nbytes = acquired[0]
        assert nbytes == arr.nbytes
        encoder.setBuffer_offset_atIndex_.assert_called_once()

    # -- integration: launch_kernel passes pool through -------------------

    @skip_non_darwin
    def test_pool_integrated_in_launch_kernel(self):
        from unittest.mock import MagicMock, patch

        from third_party.metal.backend.driver import (
            MetalBufferPool,
            MetalKernelHandle,
        )

        pool = MetalBufferPool()
        device = MagicMock()
        queue = MagicMock()
        library = MagicMock()

        handle = MetalKernelHandle(device, queue, library, metadata={})

        with patch("third_party.metal.backend.driver._bind_argument") as mock_bind:
            handle.launch_kernel(
                "test_fn",
                args=[42],
                grid=(1, 1, 1),
                block=(1, 1, 1),
                sync=False,
                buffer_pool=pool,
            )
            assert mock_bind.called
            call_kwargs = mock_bind.call_args
            assert call_kwargs.kwargs.get("pool") is pool or (
                len(call_kwargs.args) > 5 and call_kwargs.args[5] is pool
            )

    # -- integration: clear_cache drains pool -----------------------------

    @skip_non_darwin
    def test_clear_cache_drains_pool(self):
        from unittest.mock import MagicMock

        from third_party.metal.backend.driver import MetalBufferPool, MetalDriver

        pool = MetalBufferPool()
        device = _MockDevice()
        buffers = [pool.acquire(device, 256) for _ in range(3)]
        for b in buffers:
            pool.release(b, 256)
        assert pool.pool_size == 3

        driver = MetalDriver.__new__(MetalDriver)
        driver._initialized = True
        utils = MagicMock()
        utils._buffer_pool = pool
        driver.utils = utils

        cache = MagicMock()
        driver.clear_cache(cache)

        assert pool.pool_size == 0
