#ifndef TRITON_METAL_SPMDOPTOLLVM_H
#define TRITON_METAL_SPMDOPTOLLVM_H

#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir::triton::Metal {

void populateSPMDOpToLLVMPattern(LLVMTypeConverter &typeConverter,
                                 RewritePatternSet &patterns,
                                 PatternBenefit benefit);

} // namespace mlir::triton::Metal

#endif // TRITON_METAL_SPMDOPTOLLVM_H
