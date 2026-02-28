#include "BarrierOpToLLVM.h"
#include "TargetInfo.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

using namespace mlir;

namespace {

class MetalBarrierOpConversion
    : public ConvertOpToLLVMPattern<triton::gpu::BarrierOp> {
public:
  MetalBarrierOpConversion(const LLVMTypeConverter &converter,
                           PatternBenefit benefit,
                           const mlir::triton::Metal::TargetInfo &info)
      : ConvertOpToLLVMPattern<triton::gpu::BarrierOp>(converter, benefit),
        targetInfo(info) {}

  using OpAdaptor = typename triton::gpu::BarrierOp::Adaptor;

  LogicalResult
  matchAndRewrite(triton::gpu::BarrierOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    targetInfo.barrier(op.getLoc(), rewriter, op.getAddrSpace());
    rewriter.eraseOp(op);
    return success();
  }

private:
  const mlir::triton::Metal::TargetInfo &targetInfo;
};

} // namespace

void mlir::triton::Metal::populateBarrierOpToLLVMPatterns(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    PatternBenefit benefit, const TargetInfo &targetInfo) {
  patterns.add<MetalBarrierOpConversion>(typeConverter, benefit, targetInfo);
}
