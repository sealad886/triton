#ifndef TRITON_METAL_DOTOPTOLLVM_H
#define TRITON_METAL_DOTOPTOLLVM_H

#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir::triton::Metal {

void populateDotOpToLLVMPatterns(LLVMTypeConverter &typeConverter,
                                 RewritePatternSet &patterns,
                                 PatternBenefit benefit);

} // namespace mlir::triton::Metal

#endif // TRITON_METAL_DOTOPTOLLVM_H
