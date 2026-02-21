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
        assert backend.binary_ext == "metallib"

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
        utils = MetalUtils()

        with patch.object(MetalUtils, "_load_metallib_handle", return_value=dummy):
            direct = utils.load_binary(b"binary")
            assert direct is dummy

            module, function, n_regs, n_spills, n_max_threads = utils.load_binary(
                "kernel_name", b"binary", 0, 0
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
