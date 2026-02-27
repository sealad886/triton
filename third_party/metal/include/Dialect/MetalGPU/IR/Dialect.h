#ifndef TRITON_DIALECT_METALGPU_IR_DIALECT_H_
#define TRITON_DIALECT_METALGPU_IR_DIALECT_H_

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dialect.h"

#include "metal/include/Dialect/MetalGPU/IR/Dialect.h.inc"

#define GET_ATTRDEF_CLASSES
#include "metal/include/Dialect/MetalGPU/IR/MetalGPUAttrDefs.h.inc"

#define GET_OP_CLASSES
#include "metal/include/Dialect/MetalGPU/IR/Ops.h.inc"

namespace mlir {
namespace triton {
namespace metalgpu {
} // namespace metalgpu
} // namespace triton
} // namespace mlir

#endif // TRITON_DIALECT_METALGPU_IR_DIALECT_H_
