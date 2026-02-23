"""
Metal backend driver for Triton.

Implements DriverBase for Apple Metal. Uses PyObjC on macOS to interact
with the Metal framework for device management, memory allocation, and
kernel dispatch.
"""

import functools
import logging
import os
import struct
import sys
import threading

from triton.backends.compiler import GPUTarget
from triton.backends.driver import DriverBase

logger = logging.getLogger(__name__)

# ── Argument packing format map ─────────────────────────────────────

_ARG_PACK_FORMAT: dict[str, str] = {
    "i32": "i",
    "i64": "q",
    "u32": "I",
    "u64": "Q",
    "f32": "f",
    "f64": "d",
    "f16": "e",
}


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


def _get_torch_module():
    """Lazily import torch when available."""
    try:
        import torch

        return torch
    except ImportError:
        return None


def _is_metallib_blob(binary):
    if not isinstance(binary, (bytes, bytearray, memoryview)):
        return False
    return bytes(binary[:4]) == b"MTLB"


def _extract_num_warps(metadata):
    if metadata is None:
        return None
    if isinstance(metadata, tuple) and metadata:
        return metadata[0]
    if isinstance(metadata, dict):
        return metadata.get("num_warps")
    if hasattr(metadata, "num_warps"):
        return metadata.num_warps
    return None


def _resolve_kernel_name(kernel_metadata, launcher_metadata, handle):
    kernel_name = None
    for md in (kernel_metadata, launcher_metadata, getattr(handle, "metadata", None)):
        if isinstance(md, dict):
            kernel_name = md.get("name")
        elif hasattr(md, "name"):
            kernel_name = md.name
        if isinstance(kernel_name, str) and kernel_name:
            return kernel_name

    if hasattr(handle, "available_kernel_names"):
        names = handle.available_kernel_names()
        if names:
            return names[0]

    if hasattr(handle, "library"):
        try:
            names = list(handle.library.functionNames())
            if names:
                return str(names[0])
        except Exception:
            pass
    return kernel_name


def _resolve_and_validate_kernel_name(kernel_metadata, launcher_metadata, handle):
    kernel_name = _resolve_kernel_name(kernel_metadata, launcher_metadata, handle)
    if not isinstance(kernel_name, str) or kernel_name == "":
        raise RuntimeError(
            f"Missing/invalid Metal kernel name in launch metadata: {kernel_name!r}"
        )
    return kernel_name


def _scale_grid_for_pyobjc(handle, grid, block):
    """Scale grid to total-thread counts when using the PyObjC dispatch path.

    MetalKernelHandle.launch_kernel computes threadgroups as ceildiv(grid, block),
    so we pass total-thread counts to preserve Triton's grid semantics where grid
    values represent the number of program instances.
    """
    if isinstance(handle, MetalKernelHandle):
        return (
            grid[0] * block[0],
            grid[1] * block[1],
            grid[2] * block[2],
        )
    return grid


def _flatten_signature_value(sig, arg, out):
    if isinstance(sig, tuple):
        if not isinstance(arg, (list, tuple)) or len(sig) != len(arg):
            raise RuntimeError("Kernel argument structure does not match signature")
        for nested_sig, nested_arg in zip(sig, arg):
            _flatten_signature_value(nested_sig, nested_arg, out)
        return
    if sig == "constexpr":
        return
    out.append((sig, arg))


def _flatten_runtime_args(signature_layout, args):
    if len(signature_layout) != len(args):
        return [(None, arg) for arg in args]
    flat = []
    for sig, arg in zip(signature_layout, args):
        _flatten_signature_value(sig, arg, flat)
    return flat


def _normalize_pointer_arg(arg):
    base = getattr(arg, "base", None)
    if base is not None and hasattr(base, "data_ptr") and hasattr(base, "dtype"):
        return base
    return arg


def _normalize_scalar_arg(sig, arg):
    torch = _get_torch_module()
    if torch is None:
        return arg

    dtype_map = {
        "i1": torch.bool,
        "i8": torch.int8,
        "i16": torch.int16,
        "i32": torch.int32,
        "i64": torch.int64,
        "u1": torch.bool,
        "u8": torch.uint8,
        "u16": torch.uint16,
        "u32": torch.uint32,
        "u64": torch.uint64,
        "fp8e4b15": torch.uint8,
        "fp8e5": torch.uint8,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "f32": torch.float32,
        "fp32": torch.float32,
        "fp64": torch.float64,
    }
    dtype = dtype_map.get(sig)
    if dtype is None:
        return arg

    if isinstance(arg, torch.Tensor):
        if arg.ndim == 0 and arg.dtype != dtype:
            return arg.to(dtype=dtype)
        return arg

    if isinstance(arg, (bool, int, float)):
        return torch.tensor(arg, dtype=dtype)

    item_fn = getattr(arg, "item", None)
    if callable(item_fn):
        try:
            return torch.tensor(item_fn(), dtype=dtype)
        except Exception:
            return arg
    return arg


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
        self._torch = _get_torch_module()
        self._command_queues: dict[int, object] = {}
        self._current_stream: int = 0
        self._pending_buffers: dict[int, list] = {}
        self._execution_mode: str | None = None

    @property
    def device(self):
        if self._device is None and self._Metal is not None:
            self._device = self._Metal.MTLCreateSystemDefaultDevice()
        return self._device

    @property
    def command_queue(self):
        return self.get_command_queue(self._current_stream)

    def get_current_stream(self, device_id: int = 0) -> int:
        """Return the active stream id."""
        return self._current_stream

    def set_stream(self, stream_id: int) -> None:
        """Switch to the given stream, creating its command queue lazily."""
        self._current_stream = stream_id
        if stream_id not in self._command_queues and self.device is not None:
            self._command_queues[stream_id] = self.device.newCommandQueue()
            self._pending_buffers[stream_id] = []

    def get_command_queue(self, stream_id: int | None = None) -> object:
        """Return the command queue for *stream_id* (default: current)."""
        if stream_id is None:
            stream_id = self._current_stream
        if stream_id not in self._command_queues:
            self.set_stream(stream_id)
        return self._command_queues[stream_id]

    def synchronize_stream(self, stream_id: int | None = None) -> None:
        """Wait for all pending command buffers on the given stream."""
        if stream_id is None:
            stream_id = self._current_stream
        pending = self._pending_buffers.get(stream_id, [])
        for buf in pending:
            buf.waitUntilCompleted()
        self._pending_buffers[stream_id] = []

    def track_command_buffer(self, stream_id: int, cmd_buf: object) -> None:
        """Track a committed command buffer for later synchronization."""
        self._pending_buffers.setdefault(stream_id, []).append(cmd_buf)

    def resolve_execution_mode(self) -> str:
        """Determine the best available execution path.

        Returns one of 'torch_mps', 'pyobjc', or 'unavailable'.
        """
        if self._execution_mode is not None:
            return self._execution_mode

        prefer_torch = os.environ.get("TRITON_METAL_PREFER_TORCH_MPS", "1") != "0"

        if prefer_torch:
            try:
                import torch

                if hasattr(torch, "mps") and hasattr(torch.mps, "compile_shader"):
                    self._execution_mode = "torch_mps"
                    logger.info("Metal execution mode: torch.mps")
                    return self._execution_mode
            except ImportError:
                pass

        if self._Metal is not None:
            self._execution_mode = "pyobjc"
            logger.info("Metal execution mode: PyObjC")
            return self._execution_mode

        if not prefer_torch:
            try:
                import torch

                if hasattr(torch, "mps") and hasattr(torch.mps, "compile_shader"):
                    self._execution_mode = "torch_mps"
                    logger.info("Metal execution mode: torch.mps (fallback)")
                    return self._execution_mode
            except ImportError:
                pass

        self._execution_mode = "unavailable"
        logger.warning("Metal execution mode: unavailable (no torch.mps or PyObjC)")
        return self._execution_mode

    def get_device_properties(self, device_id=0):
        """Get Metal device properties."""
        dev = self.device
        if dev is None:
            return {
                "name": "unknown",
                "max_shared_mem": 0,
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

        max_threadgroup_memory_length = int(dev.maxThreadgroupMemoryLength())
        return {
            "name": str(dev.name()),
            "max_shared_mem": max_threadgroup_memory_length,
            "max_buffer_length": int(dev.maxBufferLength()),
            "max_threads_per_threadgroup": max_threads,
            "max_threadgroup_memory_length": max_threadgroup_memory_length,
            "gpu_family": _detect_gpu_family(dev),
        }

    def _load_metallib_handle(self, binary_bytes, metadata=None):
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

    def _load_msl_source_handle(self, source, metadata=None):
        metadata = metadata or {}
        mode = self.resolve_execution_mode()
        if mode == "unavailable":
            raise RuntimeError(
                "No Metal execution backend available. "
                "Install PyObjC (pip install pyobjc-framework-Metal) "
                "or use a torch build with torch.mps.compile_shader support."
            )
        torch = self._torch or _get_torch_module()
        if (
            torch is None
            or not hasattr(torch, "mps")
            or not hasattr(torch.mps, "compile_shader")
        ):
            raise RuntimeError(
                "torch.mps.compile_shader is required for Metal runtime launches. "
                f"Current execution mode: {mode}"
            )

        if isinstance(source, str):
            source_text = source
        elif isinstance(source, (bytes, bytearray, memoryview)):
            source_text = bytes(source).decode("utf-8")
        else:
            raise TypeError(
                "Metal source payload must be a UTF-8 string or bytes, "
                f"got: {type(source)}"
            )

        try:
            shader_library = torch.mps.compile_shader(source_text)
        except Exception as e:
            raise RuntimeError(f"Failed to compile Metal shader source: {e}")

        return TorchMetalKernelHandle(
            shader_library=shader_library, metadata=metadata, source_text=source_text
        )

    def load_binary(self, *args):
        """
        Load binary with both direct and Triton runtime-compatible signatures.

        Supported signatures:
        - load_binary(binary_or_source, metadata=None) -> kernel handle
        - load_binary(name, binary_or_source, shared, device_id)
            -> (module, function, n_regs, n_spills, n_max_threads)
        """
        if len(args) == 0:
            raise TypeError("load_binary() missing required arguments")

        # Direct utility usage used by backend tests.
        if len(args) in (1, 2):
            binary_or_source = args[0]
            metadata = args[1] if len(args) == 2 else None
            if _is_metallib_blob(binary_or_source):
                return self._load_metallib_handle(bytes(binary_or_source), metadata)
            return self._load_msl_source_handle(binary_or_source, metadata)

        # Triton CompiledKernel runtime contract.
        if len(args) >= 4:
            name, binary_or_source, _shared, device_id = args[:4]
            metadata = {"name": name}
            if _is_metallib_blob(binary_or_source):
                handle = self._load_metallib_handle(bytes(binary_or_source), metadata)
            else:
                handle = self._load_msl_source_handle(binary_or_source, metadata)
            props = self.get_device_properties(device_id)
            n_max_threads = props.get("max_threads_per_threadgroup", 1024) or 1024
            return handle, handle, 0, 0, n_max_threads

        raise TypeError(
            "load_binary() expected either (binary_or_source, metadata=None) "
            "or (name, binary_or_source, shared, device_id)"
        )

    def unload_module(self, module):
        # Metal libraries are reference-counted objects managed by PyObjC.
        # Clearing Python references is sufficient for teardown semantics.
        return None

    def launch(
        self,
        grid_x,
        grid_y,
        grid_z,
        stream,
        function,
        launch_cooperative_grid,
        launch_pdl,
        kernel_metadata,
        launch_metadata,
        launch_enter_hook,
        launch_exit_hook,
        global_scratch,
        profile_scratch,
        arg_annotations,
        kernel_signature,
        args,
    ):
        """Launch a Metal compute kernel."""
        handle = function
        if not isinstance(handle, (MetalKernelHandle, TorchMetalKernelHandle)):
            raise RuntimeError("Expected Metal kernel handle for Metal launch")

        kernel_name = _resolve_and_validate_kernel_name(kernel_metadata, None, handle)

        num_warps = _extract_num_warps(kernel_metadata) or 4
        block = (max(1, int(num_warps) * 32), 1, 1)
        grid = _scale_grid_for_pyobjc(handle, (grid_x, grid_y, grid_z), block)

        handle.launch_kernel(
            name=kernel_name,
            args=list(args) if args else [],
            grid=grid,
            block=block,
        )


class TorchMetalKernelHandle:
    """Handle backed by torch.mps.compile_shader runtime objects."""

    def __init__(self, shader_library, metadata=None, source_text=None):
        self.library = shader_library
        self.metadata = metadata or {}
        self.source_text = source_text
        self._kernels = {}
        self._lock = threading.RLock()

    def available_kernel_names(self):
        return [name for name in dir(self.library) if not name.startswith("_")]

    def get_kernel(self, name):
        if not isinstance(name, str) or name == "":
            raise RuntimeError(f"Invalid Metal kernel function name: {name!r}")
        kernel = self._kernels.get(name)
        if kernel is not None:
            return kernel
        with self._lock:
            kernel = self._kernels.get(name)
            if kernel is not None:
                return kernel
            fn = getattr(self.library, name, None)
            if fn is None:
                raise RuntimeError(f"Kernel '{name}' not found in shader library")
            self._kernels[name] = fn
            return fn

    def launch_kernel(self, name, args=None, grid=(1, 1, 1), block=(256, 1, 1)):
        kernel = self.get_kernel(name)
        args = list(args) if args else []
        gx, gy, gz = (int(grid[0]), int(grid[1]), int(grid[2]))
        bx, by, bz = (int(block[0]), int(block[1]), int(block[2]))
        threads = (max(1, gx * bx), max(1, gy * by), max(1, gz * bz))
        group_size = (max(1, bx), max(1, by), max(1, bz))
        try:
            kernel(*args, threads=threads, group_size=group_size)
        except Exception as e:
            raise RuntimeError(f"Failed to launch Metal kernel '{name}': {e}")


class MetalKernelHandle:
    """Handle for a loaded Metal library with kernel dispatch capabilities."""

    def __init__(
        self, device, command_queue, library, metadata=None, binary_bytes=None
    ):
        self.device = device
        self.command_queue = command_queue
        self.library = library
        self.metadata = metadata or {}
        self.binary_bytes = binary_bytes
        self.pipeline_cache = {}
        self._lock = threading.RLock()
        self.global_scratch_size: int = self.metadata.get("global_scratch_size", 0)
        self._scratch_buffer = None

    def get_pipeline(self, name):
        """Get or create a compute pipeline for the named kernel."""
        if not isinstance(name, str) or name == "":
            raise RuntimeError(f"Invalid Metal kernel function name: {name!r}")
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
                    raise RuntimeError(
                        f"Failed to create pipeline for '{name}': {error}"
                    )
            else:
                pipeline = result

            self.pipeline_cache[name] = pipeline
            return pipeline

    def _get_scratch_buffer(self):
        """Lazily allocate the global scratch buffer if metadata requests one."""
        if self._scratch_buffer is None and self.global_scratch_size > 0:
            self._scratch_buffer = self.device.newBufferWithLength_options_(
                self.global_scratch_size,
                0,  # MTLResourceStorageModeShared
            )
        return self._scratch_buffer

    def launch_kernel(
        self,
        name: str,
        args=None,
        grid: tuple[int, int, int] = (1, 1, 1),
        block: tuple[int, int, int] = (256, 1, 1),
        arg_types: list[str] | None = None,
        sync: bool = True,
    ):
        """
        Dispatch a compute kernel on the Metal device.

        Args:
            name: Kernel function name
            args: List of kernel arguments (numpy arrays, bytes, ints, floats)
            grid: (x, y, z) total threads
            block: (x, y, z) threads per threadgroup
            arg_types: Optional per-arg type hints (e.g. 'i32', 'i64', 'f16')
            sync: If True (default), wait for completion before returning
        """
        pipeline = self.get_pipeline(name)
        cmd_buf = self.command_queue.commandBuffer()
        encoder = cmd_buf.computeCommandEncoder()
        encoder.setComputePipelineState_(pipeline)

        num_args = 0
        if args:
            for idx, arg in enumerate(args):
                atype = arg_types[idx] if arg_types and idx < len(arg_types) else None
                _bind_argument(self.device, encoder, idx, arg, arg_type=atype)
            num_args = len(args)

        if self.global_scratch_size > 0:
            scratch = self._get_scratch_buffer()
            if scratch is not None:
                encoder.setBuffer_offset_atIndex_(scratch, 0, num_args)

        def ceildiv(a: int, b: int) -> int:
            return (a + b - 1) // b

        threadgroups = (
            ceildiv(grid[0], block[0]),
            ceildiv(grid[1], block[1]),
            ceildiv(grid[2], block[2]),
        )

        encoder.dispatchThreadgroups_threadsPerThreadgroup_(threadgroups, block)
        encoder.endEncoding()
        cmd_buf.commit()

        if sync:
            cmd_buf.waitUntilCompleted()
        else:
            MetalUtils().track_command_buffer(
                MetalUtils().get_current_stream(), cmd_buf
            )


def _bind_argument(device, encoder, idx, arg, arg_type: str | None = None):
    """Bind a single argument to a Metal compute encoder at the given index.

    Args:
        device: MTLDevice instance
        encoder: MTLComputeCommandEncoder
        idx: Argument buffer index
        arg: The argument value
        arg_type: Optional explicit type hint ('i32', 'i64', 'f32', 'f64', 'f16', etc.)
    """
    try:
        import numpy as np

        has_numpy = True
    except ImportError:
        has_numpy = False

    if has_numpy and isinstance(arg, np.ndarray):
        nbytes = arg.nbytes
        buf = device.newBufferWithBytes_length_options_(arg.tobytes(), nbytes, 0)
        encoder.setBuffer_offset_atIndex_(buf, 0, idx)
        return

    if arg_type is not None:
        fmt = _ARG_PACK_FORMAT.get(arg_type)
        if fmt is not None:
            packed = struct.pack(fmt, arg)
            encoder.setBytes_length_index_(packed, len(packed), idx)
            return

    if isinstance(arg, int):
        if arg > 2**31 - 1 or arg < -(2**31):
            packed = struct.pack("q", arg)  # 64-bit signed
        else:
            packed = struct.pack("i", arg)  # 32-bit signed
        encoder.setBytes_length_index_(packed, len(packed), idx)
    elif isinstance(arg, float):
        packed = struct.pack("f", arg)  # 32-bit float default
        encoder.setBytes_length_index_(packed, len(packed), idx)
    elif hasattr(arg, "tobytes"):
        data = arg.tobytes()
        encoder.setBytes_length_index_(data, len(data), idx)
    elif isinstance(arg, (bytes, bytearray)):
        encoder.setBytes_length_index_(arg, len(arg), idx)
    else:
        try:
            data = bytes(arg)
            encoder.setBytes_length_index_(data, len(data), idx)
        except Exception:
            raise TypeError(
                f"Unsupported Metal argument type at index {idx}: {type(arg)}"
            )


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
        self._signature_layout = (
            list(src.signature.values()) if hasattr(src, "signature") else []
        )

    def __call__(
        self,
        gridX,
        gridY,
        gridZ,
        stream,
        function,
        kernel_metadata,
        launch_metadata,
        launch_enter_hook,
        launch_exit_hook,
        *args,
    ):
        if launch_enter_hook is not None:
            launch_enter_hook(kernel_metadata, launch_metadata)

        handle = function
        if not isinstance(handle, (MetalKernelHandle, TorchMetalKernelHandle)):
            raise RuntimeError("Expected Metal kernel handle for Metal launch")

        kernel_name = _resolve_and_validate_kernel_name(
            kernel_metadata, self.metadata, handle
        )

        num_warps = (
            _extract_num_warps(kernel_metadata)
            or _extract_num_warps(self.metadata)
            or _extract_num_warps(getattr(handle, "metadata", None))
            or 4
        )
        block = (max(1, int(num_warps) * 32), 1, 1)

        flat_args = _flatten_runtime_args(self._signature_layout, args)
        runtime_args = []
        for sig, arg in flat_args:
            if isinstance(sig, str) and sig.startswith("*"):
                runtime_args.append(_normalize_pointer_arg(arg))
            else:
                runtime_args.append(_normalize_scalar_arg(sig, arg))

        grid = _scale_grid_for_pyobjc(handle, (gridX, gridY, gridZ), block)

        handle.launch_kernel(
            name=kernel_name,
            args=runtime_args,
            grid=grid,
            block=block,
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
        if metal is not None:
            try:
                device = metal.MTLCreateSystemDefaultDevice()
                if device is not None:
                    return True
            except Exception:
                pass

        torch = _get_torch_module()
        if torch is None:
            return False
        try:
            return bool(torch.backends.mps.is_available())
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
        return self.utils.get_current_stream(device_id)

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
            return "MTLBufferPtr"
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
            "fp16": "uint16_t",
            "bf16": "uint16_t",
            "fp32": "float",
            "f32": "float",
            "fp64": "double",
        }.get(ty, "uint32_t")

    def get_benchmarker(self):
        from triton.testing import do_bench

        return do_bench
