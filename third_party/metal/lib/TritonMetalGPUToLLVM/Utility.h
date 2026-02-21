#ifndef TRITON_THIRD_PARTY_METAL_LIB_TRITONMETALGPUTOLLVM_UTILITY_H_
#define TRITON_THIRD_PARTY_METAL_LIB_TRITONMETALGPUTOLLVM_UTILITY_H_

#include "triton/Conversion/TritonGPUToLLVM/Utility.h"

namespace mlir {
namespace LLVM {
namespace Metal {

Value shuffleXor(Location loc, RewriterBase &rewriter, Value val, int i);
Value shuffleUp(Location loc, RewriterBase &rewriter, Value val, int i);
Value shuffleIdx(Location loc, RewriterBase &rewriter, Value val, int i);
Value shuffleIdx(Location loc, RewriterBase &rewriter, Value val, Value i);
Value llGetPid(Location loc, RewriterBase &rewriter, ModuleOp moduleOp,
               ProgramIDDim axis);

} // namespace Metal
} // namespace LLVM
} // namespace mlir

#endif // TRITON_THIRD_PARTY_METAL_LIB_TRITONMETALGPUTOLLVM_UTILITY_H_
