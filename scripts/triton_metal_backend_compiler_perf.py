#!/usr/bin/env python3
"""
Helper module for performance profiling of Metal backend make_metal_ir.
Generates synthetic LLVM IR that exercises the main code paths.
"""
import sys
import os

_root = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(_root, "python"))
sys.path.insert(0, _root)

from third_party.metal.backend.compiler import MetalBackend, MetalOptions


def generate_sample_llvm_ir(n_instruction_groups: int = 100) -> str:
    """Generate synthetic LLVM IR that exercises all major code paths."""
    lines = []
    lines.append('define void @test_kernel(ptr addrspace(1) %arg0, ptr addrspace(1) %arg1, i32 %arg2) {')
    lines.append("entry:")

    ssa_counter = 0

    def ssa():
        nonlocal ssa_counter
        name = f"%v{ssa_counter}"
        ssa_counter += 1
        return name

    for i in range(n_instruction_groups):
        group = i % 10
        if group == 0:
            # Binary ops (most common)
            v = ssa()
            lines.append(f"  {v} = add nsw i32 %arg2, {i}")
            v2 = ssa()
            lines.append(f"  {v2} = mul i32 {v}, 2")
            v3 = ssa()
            lines.append(f"  {v3} = fadd float 1.0, 2.0")
            v4 = ssa()
            lines.append(f"  {v4} = sub i32 {v2}, {v}")
        elif group == 1:
            # Loads
            v = ssa()
            lines.append(f"  {v} = getelementptr float, ptr addrspace(1) %arg0, i32 {i}")
            v2 = ssa()
            lines.append(f"  {v2} = load float, ptr addrspace(1) {v}")
        elif group == 2:
            # Stores
            v = ssa()
            lines.append(f"  {v} = getelementptr float, ptr addrspace(1) %arg1, i32 {i}")
            lines.append(f"  store float 0.0, ptr addrspace(1) {v}")
        elif group == 3:
            # Casts
            v = ssa()
            lines.append(f"  {v} = sext i32 %arg2 to i64")
            v2 = ssa()
            lines.append(f"  {v2} = sitofp i32 %arg2 to float")
            v3 = ssa()
            lines.append(f"  {v3} = fptrunc double 1.0 to float")
        elif group == 4:
            # ICmp
            v = ssa()
            lines.append(f"  {v} = icmp slt i32 %arg2, {i}")
        elif group == 5:
            # FCmp
            v = ssa()
            lines.append(f"  {v} = getelementptr float, ptr addrspace(1) %arg0, i32 0")
            v2 = ssa()
            lines.append(f"  {v2} = load float, ptr addrspace(1) {v}")
            v3 = ssa()
            lines.append(f"  {v3} = fcmp olt float {v2}, 0.0")
        elif group == 6:
            # Select
            v = ssa()
            lines.append(f"  {v} = icmp sgt i32 %arg2, 0")
            v2 = ssa()
            lines.append(f"  {v2} = select i1 {v}, i32 1, i32 0")
        elif group == 7:
            # Calls (intrinsics)
            v = ssa()
            lines.append(f"  {v} = getelementptr float, ptr addrspace(1) %arg0, i32 {i}")
            v2 = ssa()
            lines.append(f"  {v2} = load float, ptr addrspace(1) {v}")
            v3 = ssa()
            lines.append(f"  {v3} = call float @llvm.fabs.f32(float {v2})")
            v4 = ssa()
            lines.append(f"  {v4} = call float @llvm.sqrt.f32(float {v3})")
        elif group == 8:
            # Fneg + freeze
            v = ssa()
            lines.append(f"  {v} = getelementptr float, ptr addrspace(1) %arg0, i32 {i}")
            v2 = ssa()
            lines.append(f"  {v2} = load float, ptr addrspace(1) {v}")
            v3 = ssa()
            lines.append(f"  {v3} = fneg float {v2}")
            v4 = ssa()
            lines.append(f"  {v4} = freeze float {v3}")
        elif group == 9:
            # Mix: GEP + load + binop + store
            v = ssa()
            lines.append(f"  {v} = getelementptr float, ptr addrspace(1) %arg0, i32 {i}")
            v2 = ssa()
            lines.append(f"  {v2} = load float, ptr addrspace(1) {v}")
            v3 = ssa()
            lines.append(f"  {v3} = fmul float {v2}, 2.0")
            v4 = ssa()
            lines.append(f"  {v4} = getelementptr float, ptr addrspace(1) %arg1, i32 {i}")
            lines.append(f"  store float {v3}, ptr addrspace(1) {v4}")

    lines.append("  ret void")
    lines.append("}")
    return "\n".join(lines)


def run_make_metal_ir(ir_text: str):
    """Run make_metal_ir with the given LLVM IR text."""
    metadata = {"shared": 0}
    opt = MetalOptions(arch="apple8")
    return MetalBackend.make_metal_ir(ir_text, metadata, opt)


if __name__ == "__main__":
    ir = generate_sample_llvm_ir(100)
    print(f"Generated {ir.count(chr(10))} lines of LLVM IR")
    result = run_make_metal_ir(ir)
    print(f"Generated {result.count(chr(10))} lines of MSL")
    print(result[:500])
