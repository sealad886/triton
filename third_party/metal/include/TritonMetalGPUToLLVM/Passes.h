#ifndef TRITONMETALGPU_CONVERSION_PASSES_H
#define TRITONMETALGPU_CONVERSION_PASSES_H

#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include <memory>

namespace mlir {

class ModuleOp;
template <typename T> class OperationPass;

namespace triton {

#define GEN_PASS_DECL
#include "TritonMetalGPUToLLVM/Passes.h.inc"

#define GEN_PASS_REGISTRATION
#include "TritonMetalGPUToLLVM/Passes.h.inc"

} // namespace triton
} // namespace mlir

#endif
