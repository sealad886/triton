#include "FpToFpOpToLLVM.h"
#include "LoadStoreOpToLLVM.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "triton/Conversion/TritonGPUToLLVM/PatternTritonGPUOpToLLVM.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

using namespace mlir;
using namespace mlir::triton;
using mlir::triton::Metal::getOrInsertExternFunc;

namespace {

struct FpToFpOpConversion : public ConvertOpToLLVMPattern<triton::FpToFpOp> {
  using ConvertOpToLLVMPattern<triton::FpToFpOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::FpToFpOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    ModuleOp mod = op->getParentOfType<ModuleOp>();
    if (!mod)
      return failure();

    Type srcElemTy = getElementTypeOrSelf(op.getSrc().getType());
    Type dstElemTy = getElementTypeOrSelf(op.getType());
    Type llvmDstElemTy = getTypeConverter()->convertType(dstElemTy);
    if (!llvmDstElemTy)
      return failure();

    auto isFp8E5 = [](Type ty) {
      return isa<Float8E5M2Type, Float8E5M2FNUZType>(ty);
    };
    bool srcIsFp8E5 = isFp8E5(srcElemTy);
    bool dstIsFp8E5 = isFp8E5(dstElemTy);

    if (!isa<FloatType>(srcElemTy) || !isa<FloatType>(dstElemTy))
      return rewriter.notifyMatchFailure(op, "expected float element types");

    auto emitFloatCast = [&](Value in, Type dstTy) -> FailureOr<Value> {
      if (in.getType() == dstTy)
        return in;
      auto srcFTy = dyn_cast<FloatType>(in.getType());
      auto dstFTy = dyn_cast<FloatType>(dstTy);
      if (!srcFTy || !dstFTy)
        return failure();
      if (srcFTy.getWidth() < dstFTy.getWidth())
        return Value(LLVM::FPExtOp::create(rewriter, loc, dstTy, in));
      if (srcFTy.getWidth() > dstFTy.getWidth())
        return Value(LLVM::FPTruncOp::create(rewriter, loc, dstTy, in));
      return Value(LLVM::BitcastOp::create(rewriter, loc, dstTy, in));
    };

    auto convertOne = [&](Value in) -> FailureOr<Value> {
      if (!srcIsFp8E5 && !dstIsFp8E5)
        return emitFloatCast(in, llvmDstElemTy);

      if (srcIsFp8E5 && !dstIsFp8E5) {
        auto f32Ty = rewriter.getF32Type();
        auto fn = getOrInsertExternFunc(mod, rewriter, "__metal_fp8e5m2_to_fp32",
                                        f32Ty, {in.getType()});
        Value f32 = LLVM::CallOp::create(rewriter, loc, fn, ValueRange{in})
                        ->getResult(0);
        return emitFloatCast(f32, llvmDstElemTy);
      }

      if (!srcIsFp8E5 && dstIsFp8E5) {
        if (op.getRounding().has_value() &&
            op.getRounding().value() != triton::RoundingMode::RTNE) {
          return rewriter.notifyMatchFailure(
              op, "only RTNE rounding is supported for fp8e5m2 lowering");
        }
        auto f32Ty = rewriter.getF32Type();
        auto f32In = emitFloatCast(in, f32Ty);
        if (failed(f32In))
          return failure();
        auto fn = getOrInsertExternFunc(
            mod, rewriter, "__metal_fp32_to_fp8e5m2_rn", rewriter.getI8Type(),
            {f32Ty});
        Value fp8Bits = LLVM::CallOp::create(rewriter, loc, fn,
                                             ValueRange{*f32In})
                            ->getResult(0);
        if (fp8Bits.getType() == llvmDstElemTy)
          return fp8Bits;
        if (auto intDstTy = dyn_cast<IntegerType>(llvmDstElemTy)) {
          auto intSrcTy = dyn_cast<IntegerType>(fp8Bits.getType());
          if (!intSrcTy)
            return failure();
          if (intSrcTy.getWidth() < intDstTy.getWidth())
            return Value(LLVM::ZExtOp::create(rewriter, loc, llvmDstElemTy,
                                              fp8Bits));
          if (intSrcTy.getWidth() > intDstTy.getWidth())
            return Value(LLVM::TruncOp::create(rewriter, loc, llvmDstElemTy,
                                               fp8Bits));
          return Value(LLVM::BitcastOp::create(rewriter, loc, llvmDstElemTy,
                                               fp8Bits));
        }
        return failure();
      }

      if (srcIsFp8E5 && dstIsFp8E5) {
        if (srcElemTy == dstElemTy)
          return in;
      }
      return rewriter.notifyMatchFailure(
          op, "unsupported fp8 conversion kind for Metal lowering");
    };

    SmallVector<Value> srcElems;
    if (isa<RankedTensorType>(op.getType()))
      srcElems = unpackLLElements(loc, adaptor.getSrc(), rewriter);
    else
      srcElems.push_back(adaptor.getSrc());

    SmallVector<Value> dstElems;
    dstElems.reserve(srcElems.size());
    for (Value v : srcElems) {
      auto converted = convertOne(v);
      if (failed(converted))
        return failure();
      dstElems.push_back(*converted);
    }

    if (!isa<RankedTensorType>(op.getType())) {
      rewriter.replaceOp(op, dstElems.front());
      return success();
    }

    Value packed =
        packLLElements(loc, getTypeConverter(), dstElems, rewriter, op.getType());
    rewriter.replaceOp(op, packed);
    return success();
  }
};

} // anonymous namespace

void mlir::triton::Metal::populateFpToFpOpToLLVMPatterns(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    PatternBenefit benefit) {
  patterns.add<FpToFpOpConversion>(typeConverter, benefit);
}
