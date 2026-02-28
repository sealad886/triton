#ifndef TRITONMETALGPU_TRANSFORMS_PASSES_H
#define TRITONMETALGPU_TRANSFORMS_PASSES_H

#include "mlir/Pass/Pass.h"

namespace mlir {

#define GEN_PASS_DECL
#include "TritonMetalGPUTransforms/Passes.h.inc"

} // namespace mlir

namespace mlir {
/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "TritonMetalGPUTransforms/Passes.h.inc"
} // namespace mlir

#endif
