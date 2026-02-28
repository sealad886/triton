#include "DotOpToLLVM/MetalSimdgroupDot.h"
#include "triton/Conversion/TritonGPUToLLVM/PatternTritonGPUOpToLLVM.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Attributes.h"

namespace mlir::triton::Metal {

LogicalResult convertMetalSimdgroupDot(triton::DotOp op,
                                       triton::DotOp::Adaptor adaptor,
                                       const LLVMTypeConverter *typeConverter,
                                       ConversionPatternRewriter &rewriter) {
  // MVP: delegate to FMA implementation.
  // The encoding selection is correct at the MLIR level; actual simdgroup
  // codegen will be added in a follow-up that generates __metal_simdgroup_*
  // LLVM IR intrinsic calls. For now, the FMA path produces correct results.
  return convertFMADot(op, adaptor, typeConverter, rewriter);
}

} // namespace mlir::triton::Metal
