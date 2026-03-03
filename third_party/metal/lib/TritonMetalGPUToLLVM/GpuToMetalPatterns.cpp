#include "GpuToMetalPatterns.h"
#include "LoadStoreOpToLLVM.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"

using namespace mlir;
using namespace mlir::triton;

namespace {

static StringRef getThreadIdFuncName(mlir::gpu::Dimension dim) {
  static constexpr const char *names[] = {
      "__metal_get_thread_position_in_threadgroup_x",
      "__metal_get_thread_position_in_threadgroup_y",
      "__metal_get_thread_position_in_threadgroup_z",
  };
  return names[static_cast<uint32_t>(dim)];
}

struct GPUThreadIdOpToMetal
    : public ConvertOpToLLVMPattern<mlir::gpu::ThreadIdOp> {
  using ConvertOpToLLVMPattern<
      mlir::gpu::ThreadIdOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(mlir::gpu::ThreadIdOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto moduleOp = op->getParentOfType<ModuleOp>();
    StringRef funcName = getThreadIdFuncName(op.getDimension());

    auto i32Ty = rewriter.getI32Type();
    auto fn = Metal::getOrInsertExternFunc(moduleOp, rewriter, funcName,
                                           i32Ty, {});
    Value result =
        LLVM::CallOp::create(rewriter, loc, fn, ValueRange{})->getResult(0);

    Type targetTy = getTypeConverter()->convertType(op.getType());
    if (targetTy != i32Ty)
      result = LLVM::ZExtOp::create(rewriter, loc, targetTy, result);

    rewriter.replaceOp(op, result);
    return success();
  }
};

struct GPUBarrierOpToMetal
    : public ConvertOpToLLVMPattern<mlir::gpu::BarrierOp> {
  using ConvertOpToLLVMPattern<
      mlir::gpu::BarrierOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(mlir::gpu::BarrierOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto moduleOp = op->getParentOfType<ModuleOp>();
    auto voidTy = LLVM::LLVMVoidType::get(rewriter.getContext());
    auto i32Ty = rewriter.getI32Type();

    auto fn = Metal::getOrInsertExternFunc(moduleOp, rewriter,
                                           "__metal_simdgroup_barrier",
                                           voidTy, {i32Ty});

    auto b = TritonLLVMOpBuilder(loc, rewriter);
    LLVM::CallOp::create(rewriter, loc, fn, ValueRange{b.i32_val(1)});
    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

void mlir::triton::Metal::populateGpuToMetalConversionPatterns(
    LLVMTypeConverter &converter, RewritePatternSet &patterns,
    PatternBenefit benefit) {
  patterns.add<GPUThreadIdOpToMetal, GPUBarrierOpToMetal>(converter, benefit);
}
