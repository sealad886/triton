#ifndef TRITON_METAL_BARRIEROPTOLLVM_H
#define TRITON_METAL_BARRIEROPTOLLVM_H

#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir::triton::Metal {

class TargetInfo;

void populateBarrierOpToLLVMPatterns(LLVMTypeConverter &typeConverter,
                                     RewritePatternSet &patterns,
                                     PatternBenefit benefit,
                                     const TargetInfo &targetInfo);

} // namespace mlir::triton::Metal

#endif // TRITON_METAL_BARRIEROPTOLLVM_H
