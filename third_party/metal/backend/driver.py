"""
Metal backend driver for Triton.

Implements DriverBase for Apple Metal. Uses PyObjC on macOS to interact
with the Metal framework for device management, memory allocation, and
kernel dispatch.
"""

import ctypes
import logging
import os
import struct
import sys
import threading
import time

from triton.backends.compiler import GPUTarget
from triton.backends.driver import DriverBase

logger = logging.getLogger(__name__)

# ── Lazy module-level imports ────────────────────────────────────────

_numpy_module = None
_numpy_checked = False


def _get_numpy_module():
    """Lazily import numpy once."""
    global _numpy_module, _numpy_checked
    if not _numpy_checked:
        try:
            import numpy

            _numpy_module = numpy
        except ImportError:
            pass
        _numpy_checked = True
    return _numpy_module


def _ceildiv(a: int, b: int) -> int:
    return (a + b - 1) // b


# Register-file sizes per GPU family (16-bit half-word register file in bytes).
# Source: Alyssa Rosenzweig's M1 reverse-engineering; M2-M4 use the same
# register file dimension per GPU core but scale core count instead.
_REGISTER_FILE_BYTES = {
    "apple7": 212992,  # 208 KiB
    "apple8": 212992, "apple9": 212992, "apple10": 212992,  # assumed same until Apple documents otherwise
}


def _estimate_registers_from_occupancy(
    kernel_max_threads: int,
    device_max_threads: int,
    gpu_family: str = "apple8",
) -> int:
    """Estimate 16-bit register count from per-kernel occupancy reduction.

    Apple GPUs limit maxTotalThreadsPerThreadgroup when a shader's register
    pressure exceeds a threshold.  By inverting this relationship we derive
    a rough register-per-thread estimate.

    Returns 0 when occupancy is not limited (register pressure below
    reporting threshold).
    """
    if kernel_max_threads >= device_max_threads:
        return 0
    if kernel_max_threads <= 0:
        return 256
    reg_file = _REGISTER_FILE_BYTES.get(gpu_family, 212992)
    regs_16bit = reg_file // (kernel_max_threads * 2)
    return min(regs_16bit, 256)


def _estimate_max_num_regs(gpu_family: str) -> int:
    """Approximate a CUDA-style per-block register budget for generic heuristics.

    Metal does not expose a direct `max_num_regs` property. We derive a
    compatible 32-bit register count from the per-core register file size so
    generic Triton occupancy heuristics can remain defined on Metal.
    """
    reg_file = _REGISTER_FILE_BYTES.get(gpu_family, _REGISTER_FILE_BYTES["apple8"])
    return max(0, reg_file // 4)


def _estimate_max_threads_per_sm(max_threads_per_threadgroup: int) -> int:
    """Provide a conservative resident-thread bound for schema parity.

    Apple does not publish a direct equivalent of CUDA/HIP's
    `maxThreadsPerMultiProcessor` through the Metal runtime. Using the
    threadgroup limit keeps generic occupancy helpers defined without claiming
    undocumented residency capacity.
    """
    return max(0, int(max_threads_per_threadgroup))


# ── Size-bucketed MTLBuffer pool ────────────────────────────────────


class MetalBufferPool:
    """Size-bucketed MTLBuffer pool for reuse across kernel launches.

    Buffers are allocated in power-of-2 sizes (minimum 256 bytes). When a buffer
    is acquired, the pool provides one of matching bucket size. After GPU work
    completes, buffers are returned to the pool via the synchronization path.
    """

    _MAX_PER_BUCKET = 32
    _MIN_BUCKET = 256

    def __init__(self):
        self._lock = threading.Lock()
        self._free: dict[int, list] = {}
        self.hits = 0
        self.misses = 0
        self.total_allocated_bytes = 0

    @staticmethod
    def _bucket_size(nbytes: int) -> int:
        """Return next power-of-2 >= *nbytes*, minimum 256."""
        size = max(nbytes, MetalBufferPool._MIN_BUCKET)
        # next power-of-2
        size -= 1
        size |= size >> 1
        size |= size >> 2
        size |= size >> 4
        size |= size >> 8
        size |= size >> 16
        size |= size >> 32
        return size + 1

    def acquire(self, device, nbytes: int):
        """Return an MTLBuffer of at least *nbytes* (bucket-aligned)."""
        bucket = self._bucket_size(nbytes)
        with self._lock:
            free_list = self._free.get(bucket)
            if free_list:
                self.hits += 1
                return free_list.pop()
            self.misses += 1
        # Allocate outside the lock
        buf = device.newBufferWithLength_options_(bucket, 0)
        with self._lock:
            self.total_allocated_bytes += bucket
        return buf

    def release(self, buf, nbytes: int) -> None:
        """Return *buf* to the free list for its bucket."""
        bucket = self._bucket_size(nbytes)
        with self._lock:
            free_list = self._free.setdefault(bucket, [])
            if len(free_list) < self._MAX_PER_BUCKET:
                free_list.append(buf)
            # else: discard — cap reached

    def drain(self) -> None:
        """Clear all free lists."""
        with self._lock:
            self._free.clear()

    @property
    def pool_size(self) -> int:
        """Total number of free buffers across all buckets."""
        with self._lock:
            return sum(len(v) for v in self._free.values())

    def stats(self) -> dict:
        """Return pool statistics."""
        with self._lock:
            return {
                "hits": self.hits,
                "misses": self.misses,
                "total_allocated_bytes": self.total_allocated_bytes,
                "pool_size": sum(len(v) for v in self._free.values()),
            }


# ── Argument packing format map ─────────────────────────────────────

_ARG_PACK_FORMAT: dict[str, str] = {
    "i1": "?",  # bool
    "i8": "b",  # signed char
    "u8": "B",  # unsigned char
    "i16": "h",  # signed short
    "u16": "H",  # unsigned short
    "i32": "i", "i64": "q", "u32": "I", "u64": "Q", "f32": "f", "f64": "d", "f16": "e", "bf16":
    "H",  # bfloat16 packed as raw 16-bit unsigned (no struct format)
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


def _is_mtl_buffer_like(arg) -> bool:
    """Detect native/shared Metal buffers passed directly to the PyObjC path."""
    return callable(getattr(arg, "contents", None)) and callable(getattr(arg, "length", None))


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
        raise RuntimeError(f"Missing/invalid Metal kernel name in launch metadata: {kernel_name!r}")
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


# Map signature type string → torch dtype ATTRIBUTE NAME (resolved lazily).
_TORCH_DTYPE_MAP: dict[str, str] = {
    "i1": "bool",
    "i8": "int8",
    "i16": "int16",
    "i32": "int32",
    "i64": "int64",
    "u1": "bool",
    "u8": "uint8",
    "u16": "uint16",
    "u32": "uint32",
    "u64": "uint64",
    "fp8e4b15": "uint8",
    "fp8e5": "uint8",
    "fp16": "float16",
    "bf16": "bfloat16",
    "f32": "float32",
    "fp32": "float32",
    "fp64": "float64",
}


def _normalize_scalar_arg(sig, arg):
    torch = _get_torch_module()
    if torch is None:
        return arg

    dtype_name = _TORCH_DTYPE_MAP.get(sig)
    if dtype_name is None:
        return arg

    dtype = getattr(torch, dtype_name, None)
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


class _MetalTimingEvent:
    """GPU-aware timing event for Metal using gpuStartTime/gpuEndTime.

    Captures the most recently committed MTLCommandBuffer (stored in
    _metal_last_cmd_buf thread-local by launch_kernel) for accurate
    GPU-side timestamps. Falls back to host-side timing when GPU
    timestamps are unavailable.
    """

    def __init__(self, enable_timing=True):
        self.enable_timing = enable_timing
        self._host_timestamp = None
        self._cmd_buf = None

    def record(self):
        torch = _get_torch_module()
        if (torch is not None and hasattr(torch, "mps") and hasattr(torch.mps, "synchronize")):
            torch.mps.synchronize()
        self._host_timestamp = time.perf_counter()
        self._cmd_buf = getattr(_metal_last_cmd_buf, "cmd_buf", None)

    def elapsed_time(self, end_event):
        if self._host_timestamp is None or end_event._host_timestamp is None:
            raise RuntimeError("Event timing requested before record()")
        gpu_ms = _gpu_elapsed_ms(self._cmd_buf, end_event._cmd_buf)
        if gpu_ms is not None:
            return gpu_ms
        return (end_event._host_timestamp - self._host_timestamp) * 1000.0


def _gpu_elapsed_ms(start_buf, end_buf):
    """Return GPU elapsed time in ms, or None if unavailable."""
    if start_buf is None or end_buf is None:
        return None
    try:
        t0 = start_buf.gpuStartTime()
        t1 = end_buf.gpuEndTime()
        if t0 > 0 and t1 > 0 and t1 >= t0:
            return (t1 - t0) * 1000.0
    except Exception:
        pass
    return None


import threading

_metal_last_cmd_buf = threading.local()


class _MetalDeviceInterface:
    """Subset of torch.cuda-like API consumed by triton.testing."""

    Event = _MetalTimingEvent

    @staticmethod
    def synchronize():
        torch = _get_torch_module()
        if (torch is not None and hasattr(torch, "mps") and hasattr(torch.mps, "synchronize")):
            torch.mps.synchronize()

    @staticmethod
    def current_device():
        return 0


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
        self._pending_pool_returns: dict[int, list] = {}
        self._execution_mode: str | None = None
        self._buffer_pool: MetalBufferPool | None = None

    @property
    def buffer_pool(self) -> MetalBufferPool:
        if self._buffer_pool is None:
            self._buffer_pool = MetalBufferPool()
        return self._buffer_pool

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

    def _coerce_stream_id(self, stream_id) -> int:
        if stream_id is None:
            return self._current_stream
        if isinstance(stream_id, bool):
            return int(stream_id)
        if isinstance(stream_id, int):
            if stream_id < 0:
                raise ValueError(f"Metal stream id must be non-negative, got {stream_id}")
            return stream_id
        for attr in ("stream_id", "cuda_stream"):
            if hasattr(stream_id, attr):
                value = int(getattr(stream_id, attr))
                if value < 0:
                    raise ValueError(f"Metal stream id from {attr} must be non-negative, got {value}")
                return value
        try:
            value = int(stream_id)
        except Exception as exc:
            raise TypeError(f"Unsupported stream identifier: {stream_id!r}") from exc
        if value < 0:
            raise ValueError(f"Metal stream id must be non-negative, got {value}")
        return value

    def set_stream(self, stream_id: int) -> None:
        """Switch to the given stream, creating its command queue lazily."""
        stream_id = self._coerce_stream_id(stream_id)
        self._current_stream = stream_id
        if stream_id not in self._command_queues and self.device is not None:
            self._command_queues[stream_id] = self.device.newCommandQueue()
            self._pending_buffers[stream_id] = []

    def activate_stream(self, stream_id) -> tuple[int, int]:
        """Activate a launch stream and return (previous_stream, active_stream)."""
        target_stream = self._coerce_stream_id(stream_id)
        previous_stream = self._current_stream
        if target_stream != previous_stream:
            self.set_stream(target_stream)
        return previous_stream, target_stream

    def restore_stream(self, previous_stream: int) -> None:
        previous_stream = self._coerce_stream_id(previous_stream)
        if previous_stream != self._current_stream:
            self.set_stream(previous_stream)

    def get_command_queue(self, stream_id: int | None = None) -> object:
        """Return the command queue for *stream_id* (default: current)."""
        stream_id = self._coerce_stream_id(stream_id)
        if stream_id not in self._command_queues:
            self.set_stream(stream_id)
        if stream_id not in self._command_queues:
            raise RuntimeError("No Metal command queue available for requested stream")
        return self._command_queues[stream_id]

    def synchronize_stream(self, stream_id: int | None = None) -> None:
        """Wait for all pending command buffers on the given stream."""
        stream_id = self._coerce_stream_id(stream_id)
        pending = self._pending_buffers.get(stream_id, [])
        for buf in pending:
            buf.waitUntilCompleted()
        self._pending_buffers[stream_id] = []

        pool_returns = self._pending_pool_returns.pop(stream_id, [])
        if pool_returns and self._buffer_pool is not None:
            pool = self._buffer_pool
            for acquired_list in pool_returns:
                for mtl_buf, nbytes in acquired_list:
                    pool.release(mtl_buf, nbytes)

        torch = self._torch or _get_torch_module()
        if (torch is not None and hasattr(torch, "mps") and hasattr(torch.mps, "synchronize")):
            torch.mps.synchronize()

    def track_command_buffer(
        self,
        stream_id: int,
        cmd_buf: object,
        acquired_pool_bufs: list | None = None,
    ) -> None:
        """Track a committed command buffer for later synchronization."""
        stream_id = self._coerce_stream_id(stream_id)
        self._pending_buffers.setdefault(stream_id, []).append(cmd_buf)
        if acquired_pool_bufs:
            self._pending_pool_returns.setdefault(stream_id, []).append(acquired_pool_bufs)

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
        """Get Metal device properties.

        Args:
            device_id: Must be 0 (Apple Silicon exposes a single GPU).
        """
        if int(device_id) != 0:
            raise ValueError(f"Metal backend has a single device (0), got device_id={device_id}")
        dev = self.device
        if dev is None:
            return {
                "name": "unknown",
                "arch": "unknown",
                "max_shared_mem": 0,
                "max_num_regs": 0,
                "warpSize": 32,
                "max_threads_per_sm": 0,
                "max_buffer_length": 0,
                "max_threads_per_threadgroup": 0,
                "max_threadgroup_memory_length": 0,
                "gpu_family": "unknown",
                "sm_clock_rate": 0,
                "mem_clock_rate": 0,
                "mem_bus_width": 0,
                "multiprocessor_count": 0,
            }

        max_threads = 1
        try:
            tpg = dev.maxThreadsPerThreadgroup()
            max_threads = tpg.width * tpg.height * tpg.depth
        except Exception:
            max_threads = 1024  # Common default for Apple Silicon

        max_threadgroup_memory_length = int(dev.maxThreadgroupMemoryLength())
        gpu_family = _detect_gpu_family(dev)
        mem_info = _gpu_memory_specs(gpu_family, str(dev.name()))
        return {
            "name": str(dev.name()),
            "arch": gpu_family,
            "max_shared_mem": max_threadgroup_memory_length,
            "max_num_regs": _estimate_max_num_regs(gpu_family),
            "warpSize": 32,
            "max_threads_per_sm": _estimate_max_threads_per_sm(max_threads),
            "max_buffer_length": int(dev.maxBufferLength()),
            "max_threads_per_threadgroup": max_threads,
            "max_threadgroup_memory_length": max_threadgroup_memory_length,
            "gpu_family": gpu_family,
            # Metal does not expose a public shader-core clock. Keep the schema
            # aligned with other backends without fabricating an undocumented
            # value.
            "sm_clock_rate": 0,
            "mem_clock_rate": mem_info["mem_clock_rate"],
            "mem_bus_width": mem_info["mem_bus_width"],
            "multiprocessor_count": mem_info["multiprocessor_count"],
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
            raise RuntimeError("No Metal execution backend available. "
                               "Install PyObjC (pip install pyobjc-framework-Metal) "
                               "or use a torch build with torch.mps.compile_shader support.")

        if isinstance(source, str):
            source_text = source
        elif isinstance(source, (bytes, bytearray, memoryview)):
            source_text = bytes(source).decode("utf-8")
        else:
            raise TypeError("Metal source payload must be a UTF-8 string or bytes, "
                            f"got: {type(source)}")

        torch = self._torch or _get_torch_module()
        has_torch_compile_shader = (torch is not None and hasattr(torch, "mps")
                                    and hasattr(torch.mps, "compile_shader"))

        if has_torch_compile_shader:
            try:
                shader_library = torch.mps.compile_shader(source_text)
            except Exception as e:
                raise RuntimeError(f"Failed to compile Metal shader source: {e}")
            return TorchMetalKernelHandle(
                shader_library=shader_library,
                metadata=metadata,
                source_text=source_text,
            )

        if mode != "pyobjc":
            raise RuntimeError("torch.mps.compile_shader is required for Metal runtime launches "
                               "unless PyObjC fallback mode is active. "
                               f"Current execution mode: {mode}")

        try:
            from third_party.metal.backend.compiler import MetalBackend, MetalOptions

            arch = self.get_device_properties().get("gpu_family", "apple8")
            opts = MetalOptions(arch=arch)
            metallib = MetalBackend.make_metallib(source_text, {}, opts)
        except Exception as e:
            raise RuntimeError("Failed to compile Metal shader source through PyObjC fallback "
                               f"path: {e}")

        fallback_metadata = dict(metadata)
        fallback_metadata.setdefault("source_mode", "pyobjc_metallib_fallback")
        fallback_metadata.setdefault("name", metadata.get("name"))
        return self._load_metallib_handle(metallib, metadata=fallback_metadata)

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
            device_max = props.get("max_threads_per_threadgroup", 1024) or 1024
            gpu_family = props.get("gpu_family", "apple8")

            # Query the per-kernel max threads from the compiled pipeline
            # state — this reflects actual register pressure, unlike the
            # device-level constant.
            try:
                pipeline = handle.get_pipeline(name)
                per_kernel_max = int(pipeline.maxTotalThreadsPerThreadgroup())
            except Exception:
                per_kernel_max = device_max

            n_regs = _estimate_registers_from_occupancy(per_kernel_max, device_max, gpu_family)
            n_spills = 0  # Metal does not expose spill counts
            return handle, handle, n_regs, n_spills, per_kernel_max

        raise TypeError("load_binary() expected either (binary_or_source, metadata=None) "
                        "or (name, binary_or_source, shared, device_id)")

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
        previous_stream, active_stream = self.activate_stream(stream)
        if launch_enter_hook is not None:
            launch_enter_hook(kernel_metadata, launch_metadata)

        try:
            if launch_cooperative_grid:
                raise RuntimeError("Metal backend does not currently support cooperative-grid launches")

            # Accepted to preserve launch contract parity with other backends.
            _ = (
                launch_pdl,
                global_scratch,
                profile_scratch,
                arg_annotations,
                kernel_signature,
            )

            handle = function
            if not isinstance(handle, (MetalKernelHandle, TorchMetalKernelHandle)):
                raise RuntimeError("Expected Metal kernel handle for Metal launch")

            kernel_name = _resolve_and_validate_kernel_name(kernel_metadata, None, handle)

            num_warps = _extract_num_warps(kernel_metadata) or 4
            block = (max(1, int(num_warps) * 32), 1, 1)
            grid = _scale_grid_for_pyobjc(handle, (grid_x, grid_y, grid_z), block)

            launch_kwargs = {
                "name": kernel_name,
                "args": list(args) if args else [],
                "grid": grid,
                "block": block,
                "sync": False,
                "stream_id": active_stream,
                "utils": self,
            }
            if isinstance(handle, MetalKernelHandle):
                launch_kwargs["command_queue"] = self.get_command_queue(active_stream)
                launch_kwargs["buffer_pool"] = self.buffer_pool

            handle.launch_kernel(**launch_kwargs)
        finally:
            if launch_exit_hook is not None:
                launch_exit_hook(kernel_metadata, launch_metadata)
            self.restore_stream(previous_stream)


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

    def launch_kernel(
            self,
            name,
            args=None,
            grid=(1, 1, 1),
            block=(256, 1, 1),
            sync=True,
            stream_id=None,
            utils=None,
    ):
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
        if sync:
            utils = utils or MetalUtils()
            utils.synchronize_stream(stream_id)


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
                    raise RuntimeError(f"Failed to create pipeline for '{name}': {error}")
            else:
                pipeline = result

            self.pipeline_cache[name] = pipeline
            return pipeline

    def _get_scratch_buffer(self):
        """Lazily allocate the global scratch buffer if metadata requests one."""
        if self._scratch_buffer is None and self.global_scratch_size > 0:
            with self._lock:
                if self._scratch_buffer is None:
                    self._scratch_buffer = self.device.newBufferWithLength_options_(self.global_scratch_size,
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
        command_queue=None,
        stream_id: int | None = None,
        utils=None,
        buffer_pool: MetalBufferPool | None = None,
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
            buffer_pool: Optional MetalBufferPool for buffer reuse
        """
        pipeline = self.get_pipeline(name)
        queue = command_queue if command_queue is not None else self.command_queue
        if queue is None:
            raise RuntimeError("No Metal command queue available for kernel launch")
        cmd_buf = queue.commandBuffer()
        encoder = cmd_buf.computeCommandEncoder()
        encoder.setComputePipelineState_(pipeline)

        acquired_bufs: list[tuple] | None = [] if buffer_pool is not None else None
        num_args = 0
        if args:
            for idx, arg in enumerate(args):
                atype = arg_types[idx] if arg_types and idx < len(arg_types) else None
                _bind_argument(
                    self.device,
                    encoder,
                    idx,
                    arg,
                    arg_type=atype,
                    pool=buffer_pool,
                    acquired_bufs=acquired_bufs,
                )
            num_args = len(args)

        if self.global_scratch_size > 0:
            scratch = self._get_scratch_buffer()
            if scratch is not None:
                encoder.setBuffer_offset_atIndex_(scratch, 0, num_args)

        threadgroups = (
            _ceildiv(grid[0], block[0]),
            _ceildiv(grid[1], block[1]),
            _ceildiv(grid[2], block[2]),
        )

        encoder.dispatchThreadgroups_threadsPerThreadgroup_(threadgroups, block)
        encoder.endEncoding()
        cmd_buf.commit()
        _metal_last_cmd_buf.cmd_buf = cmd_buf

        if sync:
            cmd_buf.waitUntilCompleted()
            if buffer_pool is not None and acquired_bufs:
                for mtl_buf, nbytes in acquired_bufs:
                    buffer_pool.release(mtl_buf, nbytes)
        else:
            utils = utils or MetalUtils()
            if stream_id is None:
                stream_id = utils.get_current_stream()
            utils.track_command_buffer(stream_id, cmd_buf, acquired_bufs)


def _bind_argument(
    device,
    encoder,
    idx,
    arg,
    arg_type: str | None = None,
    pool: MetalBufferPool | None = None,
    acquired_bufs: list | None = None,
):
    """Bind a single argument to a Metal compute encoder at the given index.

    Args:
        device: MTLDevice instance
        encoder: MTLComputeCommandEncoder
        idx: Argument buffer index
        arg: The argument value
        arg_type: Optional explicit type hint ('i32', 'i64', 'f32', 'f64', 'f16', etc.)
        pool: Optional MetalBufferPool for buffer reuse
        acquired_bufs: Optional list to collect (buf, nbytes) pairs for later release
    """
    np = _get_numpy_module()
    if np is not None and isinstance(arg, np.ndarray):
        nbytes = arg.nbytes
        if pool is not None:
            buf = pool.acquire(device, nbytes)
            data = arg.tobytes()
            ctypes.memmove(buf.contents(), data, len(data))
            if acquired_bufs is not None:
                acquired_bufs.append((buf, nbytes))
        else:
            buf = device.newBufferWithBytes_length_options_(arg.tobytes(), nbytes, 0)
        encoder.setBuffer_offset_atIndex_(buf, 0, idx)
        return

    if _is_mtl_buffer_like(arg):
        encoder.setBuffer_offset_atIndex_(arg, 0, idx)
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


# Publicly documented Apple Silicon memory specs.
# mem_clock_rate is the base memory clock in kHz (Triton's bandwidth formula
# applies a 2× DDR multiplier: BW = 2 * clock * bus_width / 8).
# LPDDR4X-4266 base clock = 2133 MHz, LPDDR5-6400 = 3200 MHz,
# LPDDR5X-7500 = 3750 MHz.
_APPLE_GPU_SPECS = {
    "m1": {"mem_clock_rate": 2133000, "mem_bus_width": 128, "gpu_cores": 8},
    "m1 pro": {"mem_clock_rate": 3200000, "mem_bus_width": 256, "gpu_cores": 16},
    "m1 max": {"mem_clock_rate": 3200000, "mem_bus_width": 512, "gpu_cores": 32},
    "m1 ultra": {"mem_clock_rate": 3200000, "mem_bus_width": 1024, "gpu_cores": 64},
    "m2": {"mem_clock_rate": 3200000, "mem_bus_width": 128, "gpu_cores": 10},
    "m2 pro": {"mem_clock_rate": 3200000, "mem_bus_width": 256, "gpu_cores": 19},
    "m2 max": {"mem_clock_rate": 3200000, "mem_bus_width": 512, "gpu_cores": 38},
    "m2 ultra": {"mem_clock_rate": 3200000, "mem_bus_width": 1024, "gpu_cores": 76},
    "m3": {"mem_clock_rate": 3200000, "mem_bus_width": 128, "gpu_cores": 10},
    "m3 pro": {"mem_clock_rate": 3200000, "mem_bus_width": 192, "gpu_cores": 18},
    "m3 max": {"mem_clock_rate": 3200000, "mem_bus_width": 512, "gpu_cores": 40},
    "m3 ultra": {"mem_clock_rate": 3200000, "mem_bus_width": 1024, "gpu_cores": 80},
    "m4": {"mem_clock_rate": 3750000, "mem_bus_width": 128, "gpu_cores": 10},
    "m4 pro": {"mem_clock_rate": 3750000, "mem_bus_width": 256, "gpu_cores": 20},
    "m4 max": {"mem_clock_rate": 3750000, "mem_bus_width": 512, "gpu_cores": 40},
}

# Fallback specs by GPU family when exact chip isn't identified.
_FAMILY_DEFAULTS = {
    "apple7": {"mem_clock_rate": 2133000, "mem_bus_width": 128, "gpu_cores": 8},
    "apple8": {"mem_clock_rate": 3200000, "mem_bus_width": 128, "gpu_cores": 10},
    "apple9": {"mem_clock_rate": 3200000, "mem_bus_width": 128, "gpu_cores": 10},
}


def _gpu_memory_specs(gpu_family, device_name=""):
    """Return memory clock rate (kHz), bus width (bits), and GPU core count."""
    name_lower = device_name.lower()
    # Check longer (more specific) chip names first to avoid partial matches.
    for chip_key in sorted(_APPLE_GPU_SPECS, key=len, reverse=True):
        if chip_key in name_lower:
            specs = _APPLE_GPU_SPECS[chip_key]
            return {
                "mem_clock_rate": specs["mem_clock_rate"],
                "mem_bus_width": specs["mem_bus_width"],
                "multiprocessor_count": specs["gpu_cores"],
            }
    defaults = _FAMILY_DEFAULTS.get(gpu_family, _FAMILY_DEFAULTS["apple8"])
    return {
        "mem_clock_rate": defaults["mem_clock_rate"],
        "mem_bus_width": defaults["mem_bus_width"],
        "multiprocessor_count": defaults["gpu_cores"],
    }


class MetalLauncher:
    """Launcher for Metal compute kernels, mirrors CudaLauncher interface."""

    def __init__(self, src, metadata):
        self.metadata = metadata
        self.src = src
        self._utils = MetalUtils()
        self._signature_layout = (list(src.signature.values()) if hasattr(src, "signature") else [])

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
        utils = getattr(self, "_utils", None) or MetalUtils()
        previous_stream, active_stream = utils.activate_stream(stream)

        if launch_enter_hook is not None:
            launch_enter_hook(kernel_metadata, launch_metadata)

        try:
            handle = function
            if not isinstance(handle, (MetalKernelHandle, TorchMetalKernelHandle)):
                raise RuntimeError("Expected Metal kernel handle for Metal launch")

            kernel_name = _resolve_and_validate_kernel_name(kernel_metadata, self.metadata, handle)

            num_warps = (_extract_num_warps(kernel_metadata) or _extract_num_warps(self.metadata)
                         or _extract_num_warps(getattr(handle, "metadata", None)) or 4)
            block = (max(1, int(num_warps) * 32), 1, 1)

            flat_args = _flatten_runtime_args(self._signature_layout, args)
            runtime_args = []
            for sig, arg in flat_args:
                if isinstance(sig, str) and sig.startswith("*"):
                    runtime_args.append(_normalize_pointer_arg(arg))
                else:
                    runtime_args.append(_normalize_scalar_arg(sig, arg))

            grid = _scale_grid_for_pyobjc(handle, (gridX, gridY, gridZ), block)

            launch_kwargs = {
                "name": kernel_name,
                "args": runtime_args,
                "grid": grid,
                "block": block,
                "sync": False,
                "stream_id": active_stream,
                "utils": utils,
            }
            if isinstance(handle, MetalKernelHandle):
                launch_kwargs["command_queue"] = utils.get_command_queue(active_stream)
                launch_kwargs["buffer_pool"] = utils.buffer_pool

            handle.launch_kernel(**launch_kwargs)
        finally:
            if launch_exit_hook is not None:
                launch_exit_hook(kernel_metadata, launch_metadata)
            utils.restore_stream(previous_stream)


class MetalDriver(DriverBase):
    """Triton driver implementation for Apple Metal.

    Apple Silicon GPUs expose exactly one MTLDevice per system — even Ultra
    chips (M1/M2/M3/M4 Ultra) present their dual-die GPU as a single
    unified device.  The driver therefore hardcodes device index 0.

    If Apple ever ships multi-GPU Macs or restores eGPU support on AS,
    this class will need per-device command queues, buffer pools, and
    pipeline caches keyed by device_id.
    """

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

    def get_device_count(self):
        """Return the number of Metal GPU devices available.

        Apple Silicon always exposes exactly 1 MTLDevice, even on Ultra
        chips.  On Intel Macs with eGPUs MTLCopyAllDevices() could
        return more, but this backend targets Apple Silicon only.
        """
        return 1

    def get_current_device(self):
        """Return the active device index (always 0 on Apple Silicon)."""
        return 0

    def set_current_device(self, device_id):
        if int(device_id) != 0:
            raise ValueError(f"Metal backend exposes a single logical device (0), got {device_id}")

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

    _TYPE_MAP = {
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

    def map_python_to_cpp_type(self, ty: str) -> str:
        if "*" in ty:
            return "id<MTLBuffer>"
        try:
            return self._TYPE_MAP[ty]
        except KeyError:
            raise TypeError(f"Unsupported Triton type for Metal AOT codegen: {ty!r}")

    def get_benchmarker(self):
        from triton.testing import do_bench

        return do_bench

    def get_device_interface(self):
        return _MetalDeviceInterface()

    def get_empty_cache_for_benchmark(self):
        torch = _get_torch_module()
        if torch is None or not hasattr(torch, "empty"):
            return None
        try:
            cache_size = 256 * 1024 * 1024
            return torch.empty(int(cache_size // 4), dtype=torch.int, device="mps")
        except Exception:
            return None

    def clear_cache(self, cache):
        if cache is None:
            return
        zero_ = getattr(cache, "zero_", None)
        if callable(zero_):
            zero_()
        pool = getattr(self.utils, "_buffer_pool", None)
        if pool is not None:
            pool.drain()
