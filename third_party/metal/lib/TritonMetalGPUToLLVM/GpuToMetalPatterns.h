#ifndef TRITON_METAL_LIB_TRITONGPUTOLLVM_GPUTOMETALPATTERNS_H
#define TRITON_METAL_LIB_TRITONGPUTOLLVM_GPUTOMETALPATTERNS_H

#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir::triton::Metal {

void populateGpuToMetalConversionPatterns(LLVMTypeConverter &converter,
                                          RewritePatternSet &patterns,
                                          PatternBenefit benefit);

} // namespace mlir::triton::Metal

#endif // TRITON_METAL_LIB_TRITONGPUTOLLVM_GPUTOMETALPATTERNS_H
