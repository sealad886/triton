"""
Performance benchmarks for MPS backend comparing against CUDA and CPU implementations.
"""

import torch
import numpy as np
import triton
import triton.language as tl
from triton.testing import get_dram_gbps, get_max_tensorcore_tflops
import time
import pytest

# Skip if MPS is not available
requires_mps = pytest.mark.skipif(
    not torch.backends.mps.is_available(),
    reason="MPS not available"
)

def reference_matmul(a, b):
    """PyTorch reference implementation"""
    return torch.matmul(a, b)

def reference_addmm(a, b, c, alpha, beta):
    """PyTorch reference implementation"""
    return torch.addmm(c, a, b, alpha=alpha, beta=beta)

@requires_mps
@pytest.mark.parametrize("M, N, K", [
    (512, 512, 512),
    (1024, 1024, 1024),
    (2048, 2048, 2048),
    (4096, 4096, 4096)
])
def test_matmul_performance(benchmark, M, N, K):
    """Test matrix multiplication performance"""
    device = torch.device("mps")

    # Create test data
    a = torch.randn((M, K), device=device, dtype=torch.float32)
    b = torch.randn((K, N), device=device, dtype=torch.float32)

    # PyTorch MPS implementation
    torch_output = reference_matmul(a, b)

    # Triton implementation
    triton_kernel = triton.testing.MatmulKernel(M, N, K)
    triton_output = triton_kernel(a, b)

    # Validate results
    torch.testing.assert_close(torch_output, triton_output, rtol=1e-2, atol=1e-3)

    # Benchmark
    def bench_torch():
        return reference_matmul(a, b)

    def bench_triton():
        return triton_kernel(a, b)

    torch_time = benchmark(bench_torch)
    triton_time = benchmark(bench_triton)

    # Calculate performance metrics
    ops = 2 * M * N * K  # Number of FLOPs for matmul
    torch_tflops = ops / (torch_time * 1e12)
    triton_tflops = ops / (triton_time * 1e12)

    print(f"\nMatrix Multiplication {M}x{N}x{K}")
    print(f"PyTorch MPS: {torch_tflops:.2f} TFLOPS")
    print(f"Triton MPS: {triton_tflops:.2f} TFLOPS")
    print(f"Speedup: {torch_time/triton_time:.2f}x")

@requires_mps
@pytest.mark.parametrize("size", [
    1024,
    1024 * 1024,
    10 * 1024 * 1024,
    100 * 1024 * 1024
])
def test_elementwise_performance(benchmark, size):
    """Test element-wise operation performance"""
    device = torch.device("mps")

    # Create test data
    x = torch.randn(size, device=device)
    y = torch.randn(size, device=device)

    # PyTorch reference implementation
    def torch_kernel(x, y):
        return torch.sigmoid(x) * torch.tanh(y)

    # Triton implementation
    @triton.jit
    def triton_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements

        x = tl.load(x_ptr + offsets, mask=mask)
        y = tl.load(y_ptr + offsets, mask=mask)
        output = tl.sigmoid(x) * tl.tanh(y)
        tl.store(output_ptr + offsets, output, mask=mask)

    # Initialize output tensors
    torch_output = torch.empty_like(x)
    triton_output = torch.empty_like(x)

    # Launch triton kernel
    grid = lambda meta: (triton.cdiv(size, meta['BLOCK_SIZE']),)
    triton_kernel[grid](x, y, triton_output, size, BLOCK_SIZE=1024)

    # Validate results
    torch_result = torch_kernel(x, y)
    torch.testing.assert_close(torch_result, triton_output, rtol=1e-2, atol=1e-3)

    # Benchmark
    def bench_torch():
        return torch_kernel(x, y)

    def bench_triton():
        triton_kernel[grid](x, y, triton_output, size, BLOCK_SIZE=1024)
        return triton_output

    torch_time = benchmark(bench_torch)
    triton_time = benchmark(bench_triton)

    # Calculate performance metrics
    bytes_processed = 3 * size * 4  # 2 inputs + 1 output, float32
    torch_gbps = bytes_processed / (torch_time * 1e9)
    triton_gbps = bytes_processed / (triton_time * 1e9)

    print(f"\nElement-wise Operation (size={size})")
    print(f"PyTorch MPS: {torch_gbps:.2f} GB/s")
    print(f"Triton MPS: {triton_gbps:.2f} GB/s")
    print(f"Speedup: {torch_time/triton_time:.2f}x")

@requires_mps
@pytest.mark.parametrize("size", [
    1024,
    1024 * 1024,
    10 * 1024 * 1024
])
def test_memory_bandwidth(benchmark, size):
    """Test memory bandwidth performance"""
    device = torch.device("mps")

    # Create test data
    x = torch.randn(size, device=device)

    # Simple copy kernel
    @triton.jit
    def copy_kernel(src_ptr, dst_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements

        x = tl.load(src_ptr + offsets, mask=mask)
        tl.store(dst_ptr + offsets, x, mask=mask)

    # Initialize output tensor
    output = torch.empty_like(x)

    # Benchmark
    def bench_torch():
        return torch.clone(x)

    def bench_triton():
        grid = lambda meta: (triton.cdiv(size, meta['BLOCK_SIZE']),)
        copy_kernel[grid](x, output, size, BLOCK_SIZE=1024)
        return output

    torch_time = benchmark(bench_torch)
    triton_time = benchmark(bench_triton)

    # Calculate bandwidth
    bytes_processed = 2 * size * 4  # read + write, float32
    torch_gbps = bytes_processed / (torch_time * 1e9)
    triton_gbps = bytes_processed / (triton_time * 1e9)

    print(f"\nMemory Bandwidth Test (size={size})")
    print(f"PyTorch MPS: {torch_gbps:.2f} GB/s")
    print(f"Triton MPS: {triton_gbps:.2f} GB/s")
    print(f"Speedup: {torch_time/triton_time:.2f}x")

if __name__ == "__main__":
    # Run with: python -m pytest python/triton/testing/backends/test_mps_performance.py -v
    pytest.main([__file__, "-v", "--benchmark-only"])
