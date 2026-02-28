#ifndef TRITON_METAL_CONVERSION_METALGPUOPS_TO_LLVM_H
#define TRITON_METAL_CONVERSION_METALGPUOPS_TO_LLVM_H

#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir::triton::Metal {

void populateMetalGPUOpsToLLVMPatterns(LLVMTypeConverter &typeConverter,
                                       RewritePatternSet &patterns,
                                       PatternBenefit benefit);

} // namespace mlir::triton::Metal

#endif // TRITON_METAL_CONVERSION_METALGPUOPS_TO_LLVM_H
