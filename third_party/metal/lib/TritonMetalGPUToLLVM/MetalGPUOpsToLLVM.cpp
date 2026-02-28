#include "MetalGPUOpsToLLVM.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/SymbolTable.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"

#include "Dialect/MetalGPU/IR/Dialect.h"

using namespace mlir;
using namespace mlir::triton;

namespace {

static LLVM::LLVMFuncOp getOrInsertFunction(RewriterBase &rewriter,
                                             ModuleOp module, Location loc,
                                             StringRef name,
                                             LLVM::LLVMFunctionType funcType) {
  if (auto existing = module.lookupSymbol<LLVM::LLVMFuncOp>(name))
    return existing;
  RewriterBase::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToStart(module.getBody());
  auto func = LLVM::LLVMFuncOp::create(rewriter, loc, name, funcType);
  func.setVisibility(SymbolTable::Visibility::Private);
  return func;
}

//===----------------------------------------------------------------------===//
// Barrier Lowering
//===----------------------------------------------------------------------===//

struct SimdgroupBarrierOpConversion
    : public ConvertOpToLLVMPattern<metalgpu::SimdgroupBarrierOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(metalgpu::SimdgroupBarrierOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto module = op->getParentOfType<ModuleOp>();
    auto voidTy = LLVM::LLVMVoidType::get(rewriter.getContext());
    auto i32Ty = rewriter.getI32Type();
    auto funcType = LLVM::LLVMFunctionType::get(voidTy, {i32Ty});
    auto func = getOrInsertFunction(rewriter, module, loc,
                                    "__metal_simdgroup_barrier", funcType);

    auto b = TritonLLVMOpBuilder(loc, rewriter);
    LLVM::CallOp::create(rewriter, loc, func,
                         ValueRange{b.i32_val(op.getMemFlags())});
    rewriter.eraseOp(op);
    return success();
  }
};

struct ThreadgroupBarrierOpConversion
    : public ConvertOpToLLVMPattern<metalgpu::ThreadgroupBarrierOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(metalgpu::ThreadgroupBarrierOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto module = op->getParentOfType<ModuleOp>();
    auto voidTy = LLVM::LLVMVoidType::get(rewriter.getContext());
    auto i32Ty = rewriter.getI32Type();
    auto funcType = LLVM::LLVMFunctionType::get(voidTy, {i32Ty});
    auto func = getOrInsertFunction(rewriter, module, loc,
                                    "__metal_threadgroup_barrier", funcType);

    auto b = TritonLLVMOpBuilder(loc, rewriter);
    LLVM::CallOp::create(rewriter, loc, func,
                         ValueRange{b.i32_val(op.getMemFlags())});
    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Shuffle Lowering
//===----------------------------------------------------------------------===//

struct SimdShuffleOpConversion
    : public ConvertOpToLLVMPattern<metalgpu::SimdShuffleOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(metalgpu::SimdShuffleOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto module = op->getParentOfType<ModuleOp>();
    auto i32Ty = rewriter.getI32Type();
    auto funcType = LLVM::LLVMFunctionType::get(i32Ty, {i32Ty, i32Ty});
    auto func = getOrInsertFunction(rewriter, module, loc,
                                    "__metal_simd_shuffle", funcType);
    auto result = LLVM::CallOp::create(
        rewriter, loc, func, ValueRange{adaptor.getVal(), adaptor.getIdx()});
    rewriter.replaceOp(op, result->getResult(0));
    return success();
  }
};

struct SimdShuffleXorOpConversion
    : public ConvertOpToLLVMPattern<metalgpu::SimdShuffleXorOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(metalgpu::SimdShuffleXorOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto module = op->getParentOfType<ModuleOp>();
    auto i32Ty = rewriter.getI32Type();
    auto funcType = LLVM::LLVMFunctionType::get(i32Ty, {i32Ty, i32Ty});
    auto func = getOrInsertFunction(rewriter, module, loc,
                                    "__metal_simd_shuffle_xor", funcType);
    auto result = LLVM::CallOp::create(
        rewriter, loc, func, ValueRange{adaptor.getVal(), adaptor.getMask()});
    rewriter.replaceOp(op, result->getResult(0));
    return success();
  }
};

struct SimdShuffleUpOpConversion
    : public ConvertOpToLLVMPattern<metalgpu::SimdShuffleUpOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(metalgpu::SimdShuffleUpOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto module = op->getParentOfType<ModuleOp>();
    auto i32Ty = rewriter.getI32Type();
    auto funcType = LLVM::LLVMFunctionType::get(i32Ty, {i32Ty, i32Ty});
    auto func = getOrInsertFunction(rewriter, module, loc,
                                    "__metal_simd_shuffle_up", funcType);
    auto result = LLVM::CallOp::create(
        rewriter, loc, func, ValueRange{adaptor.getVal(), adaptor.getDelta()});
    rewriter.replaceOp(op, result->getResult(0));
    return success();
  }
};

struct SimdShuffleDownOpConversion
    : public ConvertOpToLLVMPattern<metalgpu::SimdShuffleDownOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(metalgpu::SimdShuffleDownOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto module = op->getParentOfType<ModuleOp>();
    auto i32Ty = rewriter.getI32Type();
    auto funcType = LLVM::LLVMFunctionType::get(i32Ty, {i32Ty, i32Ty});
    auto func = getOrInsertFunction(rewriter, module, loc,
                                    "__metal_simd_shuffle_down", funcType);
    auto result = LLVM::CallOp::create(
        rewriter, loc, func, ValueRange{adaptor.getVal(), adaptor.getDelta()});
    rewriter.replaceOp(op, result->getResult(0));
    return success();
  }
};

} // namespace

void mlir::triton::Metal::populateMetalGPUOpsToLLVMPatterns(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    PatternBenefit benefit) {
  patterns.add<SimdgroupBarrierOpConversion, ThreadgroupBarrierOpConversion,
               SimdShuffleOpConversion, SimdShuffleXorOpConversion,
               SimdShuffleUpOpConversion, SimdShuffleDownOpConversion>(
      typeConverter, benefit);
}
