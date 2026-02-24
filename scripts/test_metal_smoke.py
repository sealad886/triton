#!/usr/bin/env python3
"""Standalone smoke test for the Metal backend compilation pipeline."""

import subprocess
import tempfile
import os
import shutil
import sys


def test_xcrun_compilation():
    """Test the xcrun metal compilation pipeline directly."""
    xcrun = shutil.which("xcrun")
    if xcrun is None:
        print("SKIP: xcrun not found")
        return True

    metal_src = """\
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

    src_path = None
    air_path = None
    metallib_path = None

    try:
        with tempfile.NamedTemporaryFile(suffix=".metal", delete=False, mode="w") as f:
            f.write(metal_src)
            src_path = f.name

        air_path = src_path.replace(".metal", ".air")
        metallib_path = src_path.replace(".metal", ".metallib")

        # .metal -> .air
        result = subprocess.run(
            [xcrun, "metal", "-c", src_path, "-o", air_path],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            print(f"FAIL: metal compile failed: {result.stderr}")
            return False
        print(f"  .air size: {os.path.getsize(air_path)} bytes")

        # .air -> .metallib
        result = subprocess.run(
            [xcrun, "metallib", air_path, "-o", metallib_path],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            print(f"FAIL: metallib link failed: {result.stderr}")
            return False

        with open(metallib_path, "rb") as f:
            binary = f.read()
        print(f"  .metallib size: {len(binary)} bytes")

        if binary[:4] != b"MTLB":
            print(f"FAIL: Expected MTLB magic, got {binary[:4]}")
            return False

        print("PASS: xcrun compilation pipeline")
        return True

    finally:
        for path in [src_path, air_path, metallib_path]:
            if path and os.path.exists(path):
                os.remove(path)


def test_metal_ir_generation():
    """Test Metal IR generation from LLVM IR strings."""
    # Simulate what make_metal_ir does
    import re

    llvm_ir = "define void @my_kernel(ptr %arg0, i32 %arg1) {\nret void\n}"
    names = re.findall(r"define.*void @([a-zA-Z_][a-zA-Z0-9_]*)\(", llvm_ir)
    if not names:
        print("FAIL: No kernel found in LLVM IR")
        return False
    assert names[0] == "my_kernel"
    print("PASS: Metal IR generation (kernel name extraction)")
    return True


def test_metal_pyobjc_runtime():
    """Test PyObjC Metal runtime availability."""
    if sys.platform != "darwin":
        print("SKIP: Not on macOS")
        return True

    try:
        import Metal
        device = Metal.MTLCreateSystemDefaultDevice()
        if device is None:
            print("FAIL: No Metal device found")
            return False
        print(f"  Device: {device.name()}")
        print(f"  Max buffer: {device.maxBufferLength()} bytes")
        print(f"  Max threadgroup memory: {device.maxThreadgroupMemoryLength()} bytes")
        print("PASS: PyObjC Metal runtime")
        return True
    except ImportError as e:
        print(f"SKIP: PyObjC not available: {e}")
        return True


def test_metal_kernel_load():
    """Test loading a metallib and creating a pipeline via URL-based loading."""
    if sys.platform != "darwin":
        print("SKIP: Not on macOS")
        return True

    xcrun = shutil.which("xcrun")
    if xcrun is None:
        print("SKIP: xcrun not found")
        return True

    try:
        import Metal
        import Foundation
    except ImportError:
        print("SKIP: PyObjC not available")
        return True

    metal_src = """\
#include <metal_stdlib>
using namespace metal;

kernel void test_kernel(
    device float* out [[buffer(0)]],
    uint id [[thread_position_in_grid]]
) {
    out[id] = 42.0f;
}
"""

    src_path = None
    air_path = None
    metallib_path = None

    try:
        with tempfile.NamedTemporaryFile(suffix=".metal", delete=False, mode="w") as f:
            f.write(metal_src)
            src_path = f.name

        air_path = src_path.replace(".metal", ".air")
        metallib_path = src_path.replace(".metal", ".metallib")

        subprocess.run([xcrun, "metal", "-c", src_path, "-o", air_path],
                      capture_output=True, check=True)
        subprocess.run([xcrun, "metallib", air_path, "-o", metallib_path],
                      capture_output=True, check=True)

        # Load via URL to avoid NSData/dispatch_data_t segfault through PyObjC
        device = Metal.MTLCreateSystemDefaultDevice()
        url = Foundation.NSURL.fileURLWithPath_(metallib_path)
        result = device.newLibraryWithURL_error_(url, None)

        if isinstance(result, tuple):
            library, error = result
        else:
            library = result
            error = None

        if error is not None:
            print(f"FAIL: Library load error: {error}")
            return False

        fn = library.newFunctionWithName_("test_kernel")
        if fn is None:
            print("FAIL: Could not find test_kernel function")
            return False

        result = device.newComputePipelineStateWithFunction_error_(fn, None)
        if isinstance(result, tuple):
            pipeline, error = result
        else:
            pipeline = result
            error = None

        if error is not None:
            print(f"FAIL: Pipeline creation error: {error}")
            return False

        print(f"  Pipeline created for test_kernel")
        print(f"  Max total threads: {pipeline.maxTotalThreadsPerThreadgroup()}")
        print("PASS: Metal kernel load and pipeline creation")
        return True

    finally:
        for path in [src_path, air_path, metallib_path]:
            if path and os.path.exists(path):
                os.remove(path)


def test_metal_kernel_dispatch():
    """Test actually dispatching a compute kernel and reading back results."""
    if sys.platform != "darwin":
        print("SKIP: Not on macOS")
        return True

    xcrun = shutil.which("xcrun")
    if xcrun is None:
        print("SKIP: xcrun not found")
        return True

    try:
        import Metal
        import Foundation
    except ImportError:
        print("SKIP: PyObjC not available")
        return True

    try:
        import numpy as np
    except ImportError:
        print("SKIP: numpy not available")
        return True

    metal_src = """\
#include <metal_stdlib>
using namespace metal;

kernel void fill_42(
    device float* out [[buffer(0)]],
    uint id [[thread_position_in_grid]]
) {
    out[id] = 42.0f;
}
"""

    src_path = None
    air_path = None
    metallib_path = None

    try:
        with tempfile.NamedTemporaryFile(suffix=".metal", delete=False, mode="w") as f:
            f.write(metal_src)
            src_path = f.name

        air_path = src_path.replace(".metal", ".air")
        metallib_path = src_path.replace(".metal", ".metallib")

        subprocess.run([xcrun, "metal", "-c", src_path, "-o", air_path],
                      capture_output=True, check=True)
        subprocess.run([xcrun, "metallib", air_path, "-o", metallib_path],
                      capture_output=True, check=True)

        device = Metal.MTLCreateSystemDefaultDevice()
        url = Foundation.NSURL.fileURLWithPath_(metallib_path)
        result = device.newLibraryWithURL_error_(url, None)
        if isinstance(result, tuple):
            library, error = result
        else:
            library = result
            error = None
        if error is not None:
            print(f"FAIL: Library load error: {error}")
            return False

        fn = library.newFunctionWithName_("fill_42")
        result = device.newComputePipelineStateWithFunction_error_(fn, None)
        if isinstance(result, tuple):
            pipeline, error = result
        else:
            pipeline = result
            error = None
        if error is not None:
            print(f"FAIL: Pipeline error: {error}")
            return False

        n_elements = 256
        buf_size = n_elements * 4  # float32

        out_buf = device.newBufferWithLength_options_(buf_size, 0)

        queue = device.newCommandQueue()
        cmd_buf = queue.commandBuffer()
        encoder = cmd_buf.computeCommandEncoder()
        encoder.setComputePipelineState_(pipeline)
        encoder.setBuffer_offset_atIndex_(out_buf, 0, 0)

        threadgroup_size = (min(n_elements, 256), 1, 1)
        threadgroups = ((n_elements + threadgroup_size[0] - 1) // threadgroup_size[0], 1, 1)
        encoder.dispatchThreadgroups_threadsPerThreadgroup_(threadgroups, threadgroup_size)
        encoder.endEncoding()
        cmd_buf.commit()
        cmd_buf.waitUntilCompleted()

        # Read back and verify
        mem = out_buf.contents().as_buffer(buf_size)
        results = np.frombuffer(mem, dtype=np.float32)

        expected = np.full(n_elements, 42.0, dtype=np.float32)
        if not np.allclose(results, expected):
            print(f"FAIL: Expected all 42.0, got {results[:8]}...")
            return False

        print(f"  Dispatched fill_42 over {n_elements} elements")
        print(f"  First 8 results: {results[:8]}")
        print("PASS: Metal kernel dispatch with correct output")
        return True

    finally:
        for path in [src_path, air_path, metallib_path]:
            if path and os.path.exists(path):
                os.remove(path)


def _run_harness_script(script_name, mode, extra_args):
    script_path = os.path.join("python", "test", "backend", script_name)
    if not os.path.exists(script_path):
        print(f"SKIP: {script_path} not found")
        return True

    cmd = [
        sys.executable,
        script_path,
        "--mode",
        mode,
        "--iters",
        "16",
        "--shape",
        "2048",
        "--transfer-every",
        "4",
        "--run-root",
        "artifacts/metal-harness-runs",
        "--tag",
        f"smoke-{script_name.replace('.py', '')}-{mode}",
    ]
    cmd.extend(extra_args)
    with tempfile.TemporaryDirectory(prefix="triton-metal-smoke-cache-") as cache_dir:
        env = os.environ.copy()
        env["TRITON_CACHE_DIR"] = cache_dir
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=180,
            env=env,
        )
    if result.returncode != 0:
        print("FAIL: Harness script failed")
        print("  command:", " ".join(cmd))
        print("  stdout:", result.stdout)
        print("  stderr:", result.stderr)
        return False
    print(result.stdout.strip())
    return True


def _torch_mps_available():
    try:
        import torch

        return bool(torch.backends.mps.is_built() and torch.backends.mps.is_available())
    except Exception:
        return False


def test_transfer_harness_cpu():
    """Smoke-test deterministic transfer stress harness in CPU mode."""
    return _run_harness_script("metal_mps_transfer_stress.py", "cpu", [])


def test_transfer_harness_mps():
    """Smoke-test deterministic transfer stress harness in MPS mode."""
    if not _torch_mps_available():
        print("SKIP: torch MPS backend unavailable")
        return True
    return _run_harness_script(
        "metal_mps_transfer_stress.py",
        "mps",
        ["--sync-before-transfer", "--sync-after-transfer"],
    )


def test_project_flow_harness_cpu():
    """Smoke-test project-flow harness in CPU mode."""
    return _run_harness_script("metal_mps_project_flow_stress.py", "cpu", [])


def test_project_flow_harness_mps():
    """Smoke-test project-flow harness in MPS mode."""
    if not _torch_mps_available():
        print("SKIP: torch MPS backend unavailable")
        return True
    return _run_harness_script(
        "metal_mps_project_flow_stress.py",
        "mps",
        ["--sync-before-transfer", "--sync-after-transfer"],
    )


def test_training_loop_harness_cpu():
    """Smoke-test training-loop stress harness in CPU mode."""
    return _run_harness_script("metal_mps_training_loop_stress.py", "cpu", [])


def test_training_loop_harness_mps():
    """Smoke-test training-loop stress harness in MPS mode."""
    if not _torch_mps_available():
        print("SKIP: torch MPS backend unavailable")
        return True
    return _run_harness_script(
        "metal_mps_training_loop_stress.py",
        "mps",
        ["--sync-before-transfer", "--sync-after-transfer"],
    )


def main():
    print("=" * 60)
    print("Metal Backend Smoke Tests")
    print("=" * 60)

    tests = [
        ("xcrun compilation", test_xcrun_compilation),
        ("Metal IR generation", test_metal_ir_generation),
        ("PyObjC Metal runtime", test_metal_pyobjc_runtime),
        ("Metal kernel load", test_metal_kernel_load),
        ("Metal kernel dispatch", test_metal_kernel_dispatch),
        ("Transfer harness (CPU)", test_transfer_harness_cpu),
        ("Transfer harness (MPS)", test_transfer_harness_mps),
        ("Project-flow harness (CPU)", test_project_flow_harness_cpu),
        ("Project-flow harness (MPS)", test_project_flow_harness_mps),
        ("Training-loop harness (CPU)", test_training_loop_harness_cpu),
        ("Training-loop harness (MPS)", test_training_loop_harness_mps),
    ]

    results = []
    for name, test_fn in tests:
        print(f"\n--- {name} ---")
        try:
            passed = test_fn()
            results.append((name, passed))
        except Exception as e:
            print(f"ERROR: {e}")
            results.append((name, False))

    print("\n" + "=" * 60)
    print("Results:")
    all_pass = True
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  {status}: {name}")
        if not passed:
            all_pass = False

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
