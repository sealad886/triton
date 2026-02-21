"""
Metal backend driver for Triton.

Implements DriverBase for Apple Metal. Uses PyObjC on macOS to interact
with the Metal framework for device management, memory allocation, and
kernel dispatch.
"""

import os
import sys
import struct
import threading
import functools
from pathlib import Path

from triton.backends.driver import DriverBase
from triton.backends.compiler import GPUTarget


def _get_metal_module():
    """Lazily import the Metal framework via PyObjC."""
    if sys.platform != "darwin":
        return None
    try:
        import Metal
        return Metal
    except ImportError:
        return None


def _get_foundation_module():
    """Lazily import Foundation via PyObjC."""
    if sys.platform != "darwin":
        return None
    try:
        import Foundation
        return Foundation
    except ImportError:
        return None


class MetalUtils:
    """Utility class for Metal device operations."""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if hasattr(self, "_initialized"):
            return
        self._initialized = True
        self._device = None
        self._command_queue = None
        self._Metal = _get_metal_module()
        self._Foundation = _get_foundation_module()

    @property
    def device(self):
        if self._device is None and self._Metal is not None:
            self._device = self._Metal.MTLCreateSystemDefaultDevice()
        return self._device

    @property
    def command_queue(self):
        if self._command_queue is None and self.device is not None:
            self._command_queue = self.device.newCommandQueue()
        return self._command_queue

    def get_device_properties(self, device_id=0):
        """Get Metal device properties."""
        dev = self.device
        if dev is None:
            return {
                "name": "unknown",
                "max_buffer_length": 0,
                "max_threads_per_threadgroup": 0,
                "max_threadgroup_memory_length": 0,
                "gpu_family": "unknown",
            }

        max_threads = 1
        try:
            tpg = dev.maxThreadsPerThreadgroup()
            max_threads = tpg.width * tpg.height * tpg.depth
        except Exception:
            max_threads = 1024  # Common default for Apple Silicon

        return {
            "name": str(dev.name()),
            "max_buffer_length": int(dev.maxBufferLength()),
            "max_threads_per_threadgroup": max_threads,
            "max_threadgroup_memory_length": int(dev.maxThreadgroupMemoryLength()),
            "gpu_family": _detect_gpu_family(dev),
        }

    def load_binary(self, binary_bytes, metadata=None):
        """
        Load a .metallib binary and return a handle for kernel dispatch.

        Uses newLibraryWithURL:error: because newLibraryWithData:error:
        expects dispatch_data_t (not NSData) and segfaults through PyObjC.

        Args:
            binary_bytes: Compiled .metallib bytes
            metadata: Optional dict with kernel metadata

        Returns:
            MetalKernelHandle
        """
        import tempfile
        metadata = metadata or {}
        dev = self.device
        if dev is None:
            raise RuntimeError("No Metal device available")

        Foundation = self._Foundation
        if Foundation is None:
            raise RuntimeError("Foundation framework not available (install PyObjC)")

        # Write to a temp file and load via URL to avoid the NSData/dispatch_data_t
        # incompatibility that causes segfaults through PyObjC.
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".metallib", delete=False) as f:
                f.write(binary_bytes)
                tmp_path = f.name

            url = Foundation.NSURL.fileURLWithPath_(tmp_path)
            result = dev.newLibraryWithURL_error_(url, None)
            if isinstance(result, tuple):
                library, error = result
                if error is not None:
                    raise RuntimeError(f"Failed to load metallib: {error}")
            else:
                library = result

            if library is None:
                raise RuntimeError("Failed to load metallib: library is None")

        except RuntimeError:
            raise
        except Exception as e:
            raise RuntimeError(f"Failed to load metallib: {e}")
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass

        return MetalKernelHandle(
            device=dev,
            command_queue=self.command_queue,
            library=library,
            metadata=metadata,
            binary_bytes=binary_bytes,
        )

    def launch(self, grid_x, grid_y, grid_z, stream, function,
               launch_cooperative_grid, launch_pdl,
               kernel_metadata, launch_metadata,
               launch_enter_hook, launch_exit_hook,
               global_scratch, profile_scratch,
               arg_annotations, kernel_signature, args):
        """Launch a Metal compute kernel."""
        handle = function
        if not isinstance(handle, MetalKernelHandle):
            raise RuntimeError("Expected MetalKernelHandle for Metal launch")

        kernel_name = kernel_metadata.get("name") if isinstance(kernel_metadata, dict) else None
        if kernel_name is None and hasattr(kernel_metadata, "name"):
            kernel_name = kernel_metadata.name

        block = (256, 1, 1)  # Default threadgroup size
        grid = (grid_x, grid_y, grid_z)

        handle.launch_kernel(
            name=kernel_name,
            args=list(args) if args else [],
            grid=grid,
            block=block,
        )


class MetalKernelHandle:
    """Handle for a loaded Metal library with kernel dispatch capabilities."""

    def __init__(self, device, command_queue, library, metadata=None, binary_bytes=None):
        self.device = device
        self.command_queue = command_queue
        self.library = library
        self.metadata = metadata or {}
        self.binary_bytes = binary_bytes
        self.pipeline_cache = {}
        self._lock = threading.RLock()

    def get_pipeline(self, name):
        """Get or create a compute pipeline for the named kernel."""
        pipeline = self.pipeline_cache.get(name)
        if pipeline is not None:
            return pipeline

        with self._lock:
            pipeline = self.pipeline_cache.get(name)
            if pipeline is not None:
                return pipeline

            fn = self.library.newFunctionWithName_(name)
            if fn is None:
                raise RuntimeError(f"Kernel '{name}' not found in Metal library")

            result = self.device.newComputePipelineStateWithFunction_error_(fn, None)
            if isinstance(result, tuple):
                pipeline, error = result
                if error is not None:
                    raise RuntimeError(f"Failed to create pipeline for '{name}': {error}")
            else:
                pipeline = result

            self.pipeline_cache[name] = pipeline
            return pipeline

    def launch_kernel(self, name, args=None, grid=(1, 1, 1), block=(256, 1, 1)):
        """
        Dispatch a compute kernel on the Metal device.

        Args:
            name: Kernel function name
            args: List of kernel arguments (numpy arrays, bytes, ints, floats)
            grid: (x, y, z) total threads
            block: (x, y, z) threads per threadgroup
        """
        pipeline = self.get_pipeline(name)
        cmd_buf = self.command_queue.commandBuffer()
        encoder = cmd_buf.computeCommandEncoder()
        encoder.setComputePipelineState_(pipeline)

        if args:
            for idx, arg in enumerate(args):
                _bind_argument(self.device, encoder, idx, arg)

        # Compute threadgroups from grid/block
        def ceildiv(a, b):
            return (a + b - 1) // b

        threadgroups = (
            ceildiv(grid[0], block[0]),
            ceildiv(grid[1], block[1]),
            ceildiv(grid[2], block[2]),
        )

        encoder.dispatchThreadgroups_threadsPerThreadgroup_(threadgroups, block)
        encoder.endEncoding()
        cmd_buf.commit()
        cmd_buf.waitUntilCompleted()


def _bind_argument(device, encoder, idx, arg):
    """Bind a single argument to a Metal compute encoder at the given index."""
    try:
        import numpy as np
        has_numpy = True
    except ImportError:
        has_numpy = False

    if has_numpy and isinstance(arg, np.ndarray):
        nbytes = arg.nbytes
        buf = device.newBufferWithBytes_length_options_(arg.tobytes(), nbytes, 0)
        encoder.setBuffer_offset_atIndex_(buf, 0, idx)
    elif isinstance(arg, (bytes, bytearray)):
        encoder.setBytes_length_index_(arg, len(arg), idx)
    elif isinstance(arg, int):
        packed = struct.pack("i", arg)
        encoder.setBytes_length_index_(packed, len(packed), idx)
    elif isinstance(arg, float):
        packed = struct.pack("f", arg)
        encoder.setBytes_length_index_(packed, len(packed), idx)
    else:
        try:
            data = bytes(arg)
            encoder.setBytes_length_index_(data, len(data), idx)
        except Exception:
            raise TypeError(f"Unsupported Metal argument type at index {idx}: {type(arg)}")


def _detect_gpu_family(device):
    """Detect the Apple GPU family from a Metal device."""
    # Try to detect GPU family by checking feature sets
    # Apple Silicon M-series chips map to apple7+
    name = str(device.name()) if device else ""
    name_lower = name.lower()

    if "m4" in name_lower:
        return "apple9"
    elif "m3" in name_lower:
        return "apple9"
    elif "m2" in name_lower:
        return "apple8"
    elif "m1" in name_lower:
        return "apple7"
    elif "a17" in name_lower:
        return "apple9"
    elif "a16" in name_lower:
        return "apple8"
    elif "a15" in name_lower:
        return "apple8"
    elif "a14" in name_lower:
        return "apple7"
    else:
        return "apple8"  # Safe default for modern Apple Silicon


class MetalLauncher:
    """Launcher for Metal compute kernels, mirrors CudaLauncher interface."""

    def __init__(self, src, metadata):
        self.metadata = metadata
        self.src = src

    def __call__(self, gridX, gridY, gridZ, stream, function,
                 kernel_metadata, launch_metadata,
                 launch_enter_hook, launch_exit_hook, *args):
        if launch_enter_hook is not None:
            launch_enter_hook(kernel_metadata, launch_metadata)

        handle = function
        if isinstance(handle, MetalKernelHandle):
            kernel_name = None
            if isinstance(kernel_metadata, dict):
                kernel_name = kernel_metadata.get("name")
            elif hasattr(kernel_metadata, "name"):
                kernel_name = kernel_metadata.name

            handle.launch_kernel(
                name=kernel_name,
                args=list(args) if args else [],
                grid=(gridX, gridY, gridZ),
                block=(256, 1, 1),
            )

        if launch_exit_hook is not None:
            launch_exit_hook(kernel_metadata, launch_metadata)


class MetalDriver(DriverBase):
    """Triton driver implementation for Apple Metal."""

    def __init__(self):
        super().__init__()
        self.utils = MetalUtils()
        self.launcher_cls = MetalLauncher

    @staticmethod
    def is_active():
        if sys.platform != "darwin":
            return False
        metal = _get_metal_module()
        if metal is None:
            return False
        try:
            device = metal.MTLCreateSystemDefaultDevice()
            return device is not None
        except Exception:
            return False

    def get_current_target(self):
        props = self.utils.get_device_properties()
        gpu_family = props.get("gpu_family", "apple8")
        warp_size = 32  # Apple GPU SIMD width
        return GPUTarget("metal", gpu_family, warp_size)

    def get_active_torch_device(self):
        import torch
        return torch.device("mps")

    def get_current_device(self):
        return 0  # Metal typically has one device

    def set_current_device(self, device_id):
        pass  # Metal doesn't support device selection

    def get_current_stream(self, device_id=0):
        return 0  # Metal uses command queues, not streams

    def get_device_capability(self, device_id=0):
        props = self.utils.get_device_properties(device_id)
        gpu_family = props.get("gpu_family", "apple8")
        # Return a tuple similar to CUDA's (major, minor)
        family_map = {
            "apple7": (7, 0),
            "apple8": (8, 0),
            "apple9": (9, 0),
        }
        return family_map.get(gpu_family, (8, 0))

    def map_python_to_cpp_type(self, ty: str) -> str:
        if ty[0] == "*":
            return "MTLBuffer*"
        return {
            "i1": "bool",
            "i8": "int8_t",
            "i16": "int16_t",
            "i32": "int32_t",
            "i64": "int64_t",
            "u1": "uint8_t",
            "u8": "uint8_t",
            "u16": "uint16_t",
            "u32": "uint32_t",
            "u64": "uint64_t",
            "fp16": "half",
            "bf16": "float",
            "fp32": "float",
            "f32": "float",
            "fp64": "double",
        }.get(ty, "uint32_t")

    def get_benchmarker(self):
        from triton.testing import do_bench
        return do_bench
