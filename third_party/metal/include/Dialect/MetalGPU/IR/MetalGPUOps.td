#ifndef METALGPU_OPS
#define METALGPU_OPS

include "mlir/IR/OpBase.td"
include "mlir/Dialect/LLVMIR/LLVMOpBase.td"
include "MetalGPUDialect.td"
include "MetalGPUAttrDefs.td"

class MetalGPU_Op<string mnemonic, list<Trait> traits = []> :
    LLVM_OpBase<MetalGPU_Dialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// Barrier Operations
//===----------------------------------------------------------------------===//

def MetalGPU_SimdgroupBarrierOp : MetalGPU_Op<"simdgroup_barrier", []> {
  let summary = "Simdgroup-level memory barrier";
  let description = [{
    Issues a simdgroup-level memory barrier with the given memory flags.
    Maps to MSL `simdgroup_barrier(mem_flags)`.

    Memory flags encode which address spaces to fence:
      1 = mem_flags::mem_device
      2 = mem_flags::mem_threadgroup
      4 = mem_flags::mem_texture
    Flags can be OR'd together.
  }];
  let arguments = (ins I32Attr:$memFlags);
  let assemblyFormat = "$memFlags attr-dict";
}

def MetalGPU_ThreadgroupBarrierOp : MetalGPU_Op<"threadgroup_barrier", []> {
  let summary = "Threadgroup-level memory barrier";
  let description = [{
    Issues a threadgroup-level memory barrier with the given memory flags.
    Maps to MSL `threadgroup_barrier(mem_flags)`.

    Memory flags encode which address spaces to fence:
      1 = mem_flags::mem_device
      2 = mem_flags::mem_threadgroup
      4 = mem_flags::mem_texture
    Flags can be OR'd together.
  }];
  let arguments = (ins I32Attr:$memFlags);
  let assemblyFormat = "$memFlags attr-dict";
}

//===----------------------------------------------------------------------===//
// SIMD Shuffle Operations
//===----------------------------------------------------------------------===//

def MetalGPU_SimdShuffleOp : MetalGPU_Op<"simd_shuffle", [Pure]> {
  let summary = "Read value from a specific SIMD lane";
  let description = [{
    Reads a 32-bit value from the specified SIMD lane index.
    Maps to MSL `simd_shuffle(data, simd_lane_id)`.
  }];
  let arguments = (ins I32:$val, I32:$idx);
  let results = (outs I32:$result);
  let assemblyFormat = "$val `,` $idx attr-dict";
}

def MetalGPU_SimdShuffleXorOp : MetalGPU_Op<"simd_shuffle_xor", [Pure]> {
  let summary = "XOR shuffle within a SIMD group";
  let description = [{
    Reads a 32-bit value from the SIMD lane computed by XOR of the current
    lane index with the given mask. Maps to MSL `simd_shuffle_xor(data, mask)`.
  }];
  let arguments = (ins I32:$val, I32:$mask);
  let results = (outs I32:$result);
  let assemblyFormat = "$val `,` $mask attr-dict";
}

def MetalGPU_SimdShuffleUpOp : MetalGPU_Op<"simd_shuffle_up", [Pure]> {
  let summary = "Shift-up shuffle within a SIMD group";
  let description = [{
    Reads a 32-bit value from the SIMD lane at (current_lane - delta).
    Maps to MSL `simd_shuffle_up(data, delta)`.
  }];
  let arguments = (ins I32:$val, I32:$delta);
  let results = (outs I32:$result);
  let assemblyFormat = "$val `,` $delta attr-dict";
}

def MetalGPU_SimdShuffleDownOp : MetalGPU_Op<"simd_shuffle_down", [Pure]> {
  let summary = "Shift-down shuffle within a SIMD group";
  let description = [{
    Reads a 32-bit value from the SIMD lane at (current_lane + delta).
    Maps to MSL `simd_shuffle_down(data, delta)`.
  }];
  let arguments = (ins I32:$val, I32:$delta);
  let results = (outs I32:$result);
  let assemblyFormat = "$val `,` $delta attr-dict";
}

#endif
