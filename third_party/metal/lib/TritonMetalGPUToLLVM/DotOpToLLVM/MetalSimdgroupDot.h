#ifndef TRITON_METAL_DOTOPTOLLVM_METALSIMDGROUPDOT_H
#define TRITON_METAL_DOTOPTOLLVM_METALSIMDGROUPDOT_H

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

namespace mlir::triton::Metal {

LogicalResult convertMetalSimdgroupDot(triton::DotOp op,
                                       triton::DotOp::Adaptor adaptor,
                                       const LLVMTypeConverter *typeConverter,
                                       ConversionPatternRewriter &rewriter);

} // namespace mlir::triton::Metal

#endif
