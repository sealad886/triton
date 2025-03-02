MPS Backend Documentation
=====================

The Metal Performance Shaders (MPS) backend enables Triton to run on Apple Silicon hardware (M1/M2/M3) using the Metal framework and MPS primitives for optimal performance.

Installation
------------

The MPS backend is automatically enabled when building Triton on macOS with Apple Silicon hardware. It requires:

- macOS 11.0 or later
- Xcode 12.0 or later
- PyTorch with MPS support

Getting Started
-------------

To use the MPS backend, simply specify ``"mps"`` as the device when creating tensors:

.. code-block:: python

    import torch
    import triton

    # Create tensors on MPS device
    device = torch.device("mps")
    x = torch.randn(1000, device=device)
    y = torch.randn(1000, device=device)

    # Define Triton kernel
    @triton.jit
    def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements

        x = tl.load(x_ptr + offsets, mask=mask)
        y = tl.load(y_ptr + offsets, mask=mask)
        output = x + y

        tl.store(output_ptr + offsets, output, mask=mask)

    # Launch kernel
    grid = lambda meta: (triton.cdiv(x.shape[0], meta['BLOCK_SIZE']),)
    add_kernel[grid](x, y, output, x.shape[0], BLOCK_SIZE=128)

Architecture
-----------

The MPS backend consists of several key components:

1. **MPSDriver**: Manages device initialization, memory allocation, and kernel execution
2. **MPSBackend**: Handles the compilation pipeline from Triton IR to Metal shaders
3. **Metal Runtime**: Provides optimized implementations using MPS primitives
4. **Memory Management**: Utilizes unified memory architecture of Apple Silicon

Features
--------

- Full support for Triton's tensor operations
- Optimized matrix multiplication using MPS primitives
- Automatic kernel fusion and optimization
- Efficient memory management with unified memory
- Integration with PyTorch's MPS backend

Performance Optimization
----------------------

The MPS backend includes several optimizations:

1. **Unified Memory**: Takes advantage of Apple Silicon's unified memory architecture
2. **MPS Primitives**: Uses highly optimized Metal Performance Shaders
3. **Kernel Fusion**: Automatically fuses compatible operations
4. **Memory Layout**: Optimizes tensor layouts for Metal's memory model

Configuration Options
------------------

The MPS backend can be configured through the driver:

.. code-block:: python

    driver = triton.runtime.driver.MPSDriver()
    driver.set_kernel_config(
        num_warps=8,               # Number of warps per block
        num_stages=2,              # Pipeline stages
        max_threads=512,           # Max threads per threadgroup
        shared_memory=16384        # Shared memory size (bytes)
    )

Memory Management
---------------

The MPS backend uses Apple's unified memory architecture:

1. **Allocation**: Memory is allocated in unified memory space
2. **Transfers**: Zero-copy for CPU-GPU transfers when possible
3. **Pooling**: Memory pooling for efficient reuse
4. **Synchronization**: Automatic synchronization when needed

Current memory usage can be queried:

.. code-block:: python

    driver = triton.runtime.driver.MPSDriver()
    memory_info = driver.memory_pool()
    print(f"Current allocation: {memory_info['current_allocated']} bytes")
    print(f"Peak allocation: {memory_info['max_allocated']} bytes")

Known Limitations
--------------

1. Some advanced CUDA features may not be available
2. Performance may vary based on specific Apple Silicon chip
3. Limited support for certain atomic operations
4. Dynamic parallelism not supported

Troubleshooting
-------------

Common issues and solutions:

1. **Memory Errors**:
   - Check available memory using ``driver.memory_pool()``
   - Consider reducing batch sizes or model size
   - Enable memory profiling for debugging

2. **Performance Issues**:
   - Verify kernel configurations are optimal
   - Check for unnecessary synchronization points
   - Monitor thermal throttling

3. **Compilation Errors**:
   - Verify Metal shader compatibility
   - Check for unsupported operations
   - Enable verbose logging for debugging

Contributing
-----------

To contribute to the MPS backend:

1. Set up development environment:
   - Install Xcode and Metal tools
   - Build Triton from source
   - Enable MPS backend tests

2. Run tests:
   .. code-block:: bash

       python -m pytest python/test/unit/test_mps_backend.py
       ./build/unittest/Analysis/test_mps_backend

3. Submit pull requests with:
   - Unit tests for new features
   - Documentation updates
   - Performance benchmarks

Future Work
----------

Planned improvements:

1. Additional MPS primitive optimizations
2. Enhanced kernel fusion strategies
3. Better performance profiling tools
4. Extended operation support
5. Improved error handling and debugging
