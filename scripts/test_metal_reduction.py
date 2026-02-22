"""Test that dynamic-loop reduction kernels compile through the Metal backend."""

import triton
import triton.language as tl
from triton.backends.compiler import GPUTarget


@triton.jit
def _reduce_sum_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    s = tl.sum(x, axis=0)
    tl.store(out_ptr + pid, s)


@triton.jit
def _reduce_max_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask, other=float("-inf"))
    mx = tl.max(x, axis=0)
    tl.store(out_ptr + pid, mx)


@triton.jit
def _reduce_2d_kernel(x_ptr, out_ptr, m, n, BM: tl.constexpr, BN: tl.constexpr):
    pid = tl.program_id(axis=0)
    offs_m = pid * BM + tl.arange(0, BM)
    offs_n = tl.arange(0, BN)
    ptrs = x_ptr + offs_m[:, None] * n + offs_n[None, :]
    mask = (offs_m[:, None] < m) & (offs_n[None, :] < n)
    x = tl.load(ptrs, mask=mask, other=0.0)
    s = tl.sum(x, axis=1)
    tl.store(out_ptr + offs_m, s, mask=offs_m < m)


def test_reduce_sum():
    src = triton.compiler.ASTSource(
        fn=_reduce_sum_kernel,
        signature={"x_ptr": "*fp32", "out_ptr": "*fp32", "n": "i32"},
        constexprs={"BLOCK": 128},
    )
    kernel = triton.compile(src=src, target=GPUTarget("metal", "apple9", 32))
    assert "llir" in kernel.asm and len(kernel.asm["llir"]) > 0
    assert "metal" in kernel.asm and b"kernel void" in kernel.asm["metal"]
    assert "metallib" in kernel.asm and kernel.asm["metallib"][:4] == b"MTLB"
    print("  [PASS] reduce_sum compiled to metallib")


def test_reduce_max():
    src = triton.compiler.ASTSource(
        fn=_reduce_max_kernel,
        signature={"x_ptr": "*fp32", "out_ptr": "*fp32", "n": "i32"},
        constexprs={"BLOCK": 128},
    )
    kernel = triton.compile(src=src, target=GPUTarget("metal", "apple9", 32))
    assert "llir" in kernel.asm and len(kernel.asm["llir"]) > 0
    assert "metal" in kernel.asm and b"kernel void" in kernel.asm["metal"]
    assert "metallib" in kernel.asm and kernel.asm["metallib"][:4] == b"MTLB"
    print("  [PASS] reduce_max compiled to metallib")


def test_reduce_2d():
    src = triton.compiler.ASTSource(
        fn=_reduce_2d_kernel,
        signature={"x_ptr": "*fp32", "out_ptr": "*fp32", "m": "i32", "n": "i32"},
        constexprs={"BM": 16, "BN": 32},
    )
    kernel = triton.compile(src=src, target=GPUTarget("metal", "apple9", 32))
    assert "llir" in kernel.asm and len(kernel.asm["llir"]) > 0
    assert "metal" in kernel.asm and b"kernel void" in kernel.asm["metal"]
    assert "metallib" in kernel.asm and kernel.asm["metallib"][:4] == b"MTLB"
    print("  [PASS] reduce_2d compiled to metallib")


if __name__ == "__main__":
    print("Testing dynamic-loop reduction kernels on Metal backend...")
    test_reduce_sum()
    test_reduce_max()
    test_reduce_2d()
    print("All reduction tests passed!")
