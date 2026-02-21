#include "Utility.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

namespace mlir {
namespace LLVM {
namespace Metal {

using namespace mlir::triton;

// Metal shuffle implementation using external function calls.
// These generate LLVM IR calls that will be mapped to MSL simd_shuffle
// functions during the LLVM IR -> MSL conversion stage.
static Value shuffleCommon(Location loc, RewriterBase &rewriter, Value val,
                           Value offset, StringRef funcName) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  unsigned bits = val.getType().getIntOrFloatBitWidth();

  // Handle 64-bit values by splitting into two 32-bit shuffles
  if (bits == 64) {
    Type vecTy = vec_ty(f32_ty, 2);
    Value vec = b.bitcast(val, vecTy);
    Value val0 = b.extract_element(f32_ty, vec, b.i32_val(0));
    Value val1 = b.extract_element(f32_ty, vec, b.i32_val(1));
    val0 = shuffleCommon(loc, rewriter, val0, offset, funcName);
    val1 = shuffleCommon(loc, rewriter, val1, offset, funcName);
    vec = b.undef(vecTy);
    vec = b.insert_element(vecTy, vec, val0, b.i32_val(0));
    vec = b.insert_element(vecTy, vec, val1, b.i32_val(1));
    return b.bitcast(vec, val.getType());
  }

  Type type = val.getType();
  // Metal SIMD shuffle operates on 32-bit values
  if (type != i32_ty) {
    val = b.bitcast(val, int_ty(bits));
    if (bits < 32)
      val = b.zext(i32_ty, val);
  }

  // Declare or reuse the external shuffle function.
  // These functions are resolved during MSL code generation.
  auto moduleOp =
      rewriter.getBlock()->getParent()->getParentOfType<ModuleOp>();
  Operation *funcOp = moduleOp.lookupSymbol(funcName);
  LLVM::LLVMFuncOp shuffleFunc;
  if (funcOp) {
    shuffleFunc = cast<LLVM::LLVMFuncOp>(funcOp);
  } else {
    auto funcType = LLVM::LLVMFunctionType::get(i32_ty, {i32_ty, i32_ty});
    RewriterBase::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(moduleOp.getBody());
    shuffleFunc = LLVM::LLVMFuncOp::create(rewriter, loc, funcName, funcType);
    shuffleFunc.setVisibility(SymbolTable::Visibility::Private);
  }

  Value result =
      LLVM::CallOp::create(rewriter, loc, shuffleFunc, ValueRange{val, offset})
          ->getResult(0);

  if (type != i32_ty) {
    if (bits < 32)
      result = b.trunc(int_ty(bits), result);
    result = b.bitcast(result, type);
  }
  return result;
}

Value shuffleXor(Location loc, RewriterBase &rewriter, Value val, int i) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  return shuffleCommon(loc, rewriter, val, b.i32_val(i),
                       "__metal_simd_shuffle_xor");
}

Value shuffleUp(Location loc, RewriterBase &rewriter, Value val, int i) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  return shuffleCommon(loc, rewriter, val, b.i32_val(i),
                       "__metal_simd_shuffle_up");
}

Value shuffleIdx(Location loc, RewriterBase &rewriter, Value val, int i) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  return shuffleIdx(loc, rewriter, val, b.i32_val(i));
}

Value shuffleIdx(Location loc, RewriterBase &rewriter, Value val, Value i) {
  return shuffleCommon(loc, rewriter, val, i, "__metal_simd_shuffle");
}

Value llGetPid(Location loc, RewriterBase &rewriter, ModuleOp moduleOp,
               ProgramIDDim axis) {
  assert(moduleOp);
  // Metal only supports single CTA (no cluster), so block ID = program ID
  Value blockId = ::mlir::gpu::BlockIdOp::create(rewriter, loc,
                                                 mlir::gpu::Dimension(axis));
  return arith::IndexCastOp::create(rewriter, loc, i32_ty, blockId);
}

} // namespace Metal
} // namespace LLVM
} // namespace mlir
