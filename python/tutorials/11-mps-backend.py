"""
Matrix Multiplication and Element-wise Operations using MPS Backend
----------------------------------------------------------------

This tutorial demonstrates how to use Triton's MPS backend on Apple Silicon
hardware to perform common ML operations efficiently.
"""

import torch
import triton
import triton.language as tl
import numpy as np
import time

# Check MPS availability
assert torch.backends.mps.is_available(), "MPS device not available"
device = torch.device("mps")

# Matrix multiplication kernel
@triton.jit
def matmul_kernel(
    # Pointers to matrices
    a_ptr, b_ptr, c_ptr,
    # Matrix dimensions
    M, N, K,
    # The stride variables represent how much to increase the ptr by when moving by 1
    # element in a particular dimension. E.g. stride_am is how much to increase a_ptr
    # by to get the element one row down (A has M rows)
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr
):
    """Efficient matrix multiplication kernel for MPS backend"""
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # Initialize pointers to A, B and C
    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    # Initialize accumulator
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Iterate to compute a block of the C matrix
    for k in range(0, K, BLOCK_SIZE_K):
        # Load A and B blocks
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        # Compute matrix multiplication
        accumulator += tl.dot(a, b)
        # Increment pointers
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    # Write back the result
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + (offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn)
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)

# Element-wise kernel with automatic vectorization
@triton.jit
def elementwise_kernel(
    x_ptr, y_ptr, output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr
):
    """Efficient element-wise operation kernel for MPS backend"""
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load inputs
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)

    # Compute activation
    output = tl.sigmoid(x) * tl.tanh(y)

    # Store result
    tl.store(output_ptr + offsets, output, mask=mask)

def benchmark_matmul(M, N, K, num_trials=100):
    """Benchmark matrix multiplication performance"""
    print(f"\nBenchmarking MatMul: ({M}, {N}, {K})")

    # Create random matrices
    a = torch.randn((M, K), device=device, dtype=torch.float32)
    b = torch.randn((K, N), device=device, dtype=torch.float32)
    c = torch.empty((M, N), device=device, dtype=torch.float32)

    # Triton implementation
    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),
    )

    # Compile and warmup
    matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_SIZE_M=128, BLOCK_SIZE_N=128, BLOCK_SIZE_K=32,
        GROUP_SIZE_M=8
    )

    # PyTorch implementation
    torch_time = []
    triton_time = []

    for _ in range(num_trials):
        # Triton
        torch.mps.synchronize()
        start = time.perf_counter()
        matmul_kernel[grid](
            a, b, c,
            M, N, K,
            a.stride(0), a.stride(1),
            b.stride(0), b.stride(1),
            c.stride(0), c.stride(1),
            BLOCK_SIZE_M=128, BLOCK_SIZE_N=128, BLOCK_SIZE_K=32,
            GROUP_SIZE_M=8
        )
        torch.mps.synchronize()
        triton_time.append(time.perf_counter() - start)

        # PyTorch
        torch.mps.synchronize()
        start = time.perf_counter()
        torch.matmul(a, b)
        torch.mps.synchronize()
        torch_time.append(time.perf_counter() - start)

    # Print results
    triton_time = np.mean(triton_time[10:])  # Exclude warmup
    torch_time = np.mean(torch_time[10:])    # Exclude warmup
    print(f"Triton: {triton_time*1000:.2f}ms")
    print(f"PyTorch: {torch_time*1000:.2f}ms")
    print(f"Speedup: {torch_time/triton_time:.2f}x")

def benchmark_elementwise(size, num_trials=100):
    """Benchmark element-wise operation performance"""
    print(f"\nBenchmarking Element-wise: ({size})")

    # Create random tensors
    x = torch.randn(size, device=device)
    y = torch.randn(size, device=device)
    output = torch.empty(size, device=device)

    # Triton implementation
    grid = lambda META: (triton.cdiv(size, META['BLOCK_SIZE']),)

    # Compile and warmup
    elementwise_kernel[grid](x, y, output, size, BLOCK_SIZE=1024)

    # PyTorch implementation
    def torch_kernel(x, y):
        return torch.sigmoid(x) * torch.tanh(y)

    torch_time = []
    triton_time = []

    for _ in range(num_trials):
        # Triton
        torch.mps.synchronize()
        start = time.perf_counter()
        elementwise_kernel[grid](x, y, output, size, BLOCK_SIZE=1024)
        torch.mps.synchronize()
        triton_time.append(time.perf_counter() - start)

        # PyTorch
        torch.mps.synchronize()
        start = time.perf_counter()
        torch_kernel(x, y)
        torch.mps.synchronize()
        torch_time.append(time.perf_counter() - start)

    # Print results
    triton_time = np.mean(triton_time[10:])  # Exclude warmup
    torch_time = np.mean(torch_time[10:])    # Exclude warmup
    print(f"Triton: {triton_time*1000:.2f}ms")
    print(f"PyTorch: {torch_time*1000:.2f}ms")
    print(f"Speedup: {torch_time/triton_time:.2f}x")

def main():
    print("Running on device:", device)

    # Test different matrix sizes
    sizes = [
        (512, 512, 512),
        (1024, 1024, 1024),
        (2048, 2048, 2048)
    ]

    for M, N, K in sizes:
        benchmark_matmul(M, N, K)

    # Test different vector sizes
    sizes = [1024, 1024*1024, 10*1024*1024]

    for size in sizes:
        benchmark_elementwise(size)

if __name__ == "__main__":
    main()
