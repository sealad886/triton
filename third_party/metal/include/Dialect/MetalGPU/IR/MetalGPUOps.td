#ifndef METALGPU_OPS
#define METALGPU_OPS

include "mlir/IR/OpBase.td"
include "mlir/Dialect/LLVMIR/LLVMOpBase.td"
include "MetalGPUDialect.td"
include "MetalGPUAttrDefs.td"

class MetalGPU_Op<string mnemonic, list<Trait> traits = []> :
    LLVM_OpBase<MetalGPU_Dialect, mnemonic, traits>;

// No operations defined yet — will be added with Metal-specific lowering ops

#endif
