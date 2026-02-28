#include "TritonMetalGPUTransforms/Passes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Attributes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "tritonmetal-accelerate-matmul"

namespace mlir {

namespace tt = triton;
namespace ttg = triton::gpu;

#define GEN_PASS_DEF_TRITONMETALGPUACCELERATEMATMUL
#include "TritonMetalGPUTransforms/Passes.h.inc"

namespace {

constexpr unsigned kMinShapeForSimdgroup = 16;
constexpr unsigned kSimdgroupM = 8;
constexpr unsigned kSimdgroupN = 8;
constexpr unsigned kSimdgroupK = 8;

static bool isSimdgroupAccumTypeSupported(Type elemType) {
  return elemType.isF32();
}

static bool isSimdgroupOperandTypeSupported(Type elemType,
                                            StringRef gpuFamily) {
  if (elemType.isF32() || elemType.isF16())
    return true;
  if (elemType.isBF16()) {
    // BFloat16 requires Apple GPU family 9+ (M3/M4).
    return gpuFamily.starts_with("apple") &&
           gpuFamily.compare("apple9") >= 0;
  }
  return false;
}

static SmallVector<unsigned> computeWarpsPerCTA(int64_t M, int64_t N,
                                                unsigned numWarps) {
  unsigned tilesM = M / kSimdgroupM;
  unsigned tilesN = N / kSimdgroupN;

  unsigned warpsM = 1, warpsN = 1;
  unsigned remaining = numWarps;
  while (remaining > 1) {
    if (warpsN * 2 <= tilesN && warpsN <= warpsM) {
      warpsN *= 2;
    } else if (warpsM * 2 <= tilesM) {
      warpsM *= 2;
    } else if (warpsN * 2 <= tilesN) {
      warpsN *= 2;
    } else {
      break;
    }
    remaining /= 2;
  }
  return {warpsM, warpsN};
}

class BlockedToMetalSimdgroup : public OpRewritePattern<tt::DotOp> {
  std::string gpuFamily;
  int numWarps;

public:
  BlockedToMetalSimdgroup(MLIRContext *ctx, std::string gpuFamily,
                          int numWarps, PatternBenefit benefit = 1)
      : OpRewritePattern(ctx, benefit), gpuFamily(std::move(gpuFamily)),
        numWarps(numWarps) {}

  LogicalResult matchAndRewrite(triton::DotOp dotOp,
                                PatternRewriter &rewriter) const override {
    auto oldRetType = cast<RankedTensorType>(dotOp.getType());
    auto encoding = oldRetType.getEncoding();

    auto blockedEnc = dyn_cast<ttg::BlockedEncodingAttr>(encoding);
    if (!blockedEnc)
      return failure();

    auto shape = oldRetType.getShape();
    int rank = shape.size();
    bool hasBatch = rank == 3;
    int mIdx = hasBatch ? 1 : 0;
    int nIdx = hasBatch ? 2 : 1;

    int64_t M = shape[mIdx];
    int64_t N = shape[nIdx];

    if (M < kMinShapeForSimdgroup || N < kMinShapeForSimdgroup)
      return failure();
    if (M % kSimdgroupM != 0 || N % kSimdgroupN != 0)
      return failure();

    auto aType = cast<RankedTensorType>(dotOp.getA().getType());
    int64_t K = aType.getShape().back();
    if (K % kSimdgroupK != 0)
      return failure();

    auto elemType = oldRetType.getElementType();
    if (!isSimdgroupAccumTypeSupported(elemType))
      return failure();

    {
      auto aElemType = aType.getElementType();
      auto bTy = cast<RankedTensorType>(dotOp.getB().getType());
      auto bElemType = bTy.getElementType();
      if (!isSimdgroupOperandTypeSupported(aElemType, gpuFamily) ||
          !isSimdgroupOperandTypeSupported(bElemType, gpuFamily))
        return failure();
      if (aElemType != bElemType)
        return failure();
    }

    auto warpsPerCTA = computeWarpsPerCTA(M, N, numWarps);
    SmallVector<unsigned> instrShape = {kSimdgroupM, kSimdgroupN, kSimdgroupK};
    auto ctx = rewriter.getContext();

    // Reuse the CGA layout from the existing blocked encoding (trivial for
    // Metal — always single CTA)
    auto cgaLayout = blockedEnc.getCGALayout();

    SmallVector<unsigned> fullWarpsPerCTA;
    if (hasBatch)
      fullWarpsPerCTA.push_back(1);
    fullWarpsPerCTA.push_back(warpsPerCTA[0]);
    fullWarpsPerCTA.push_back(warpsPerCTA[1]);

    auto simdgroupEnc = ttg::MetalSimdgroupEncodingAttr::get(
        ctx, fullWarpsPerCTA, instrShape, cgaLayout);

    auto newRetType = RankedTensorType::get(shape, elemType, simdgroupEnc);

    auto dotOpEncA = ttg::DotOperandEncodingAttr::get(ctx, 0, simdgroupEnc, 0);
    auto dotOpEncB = ttg::DotOperandEncodingAttr::get(ctx, 1, simdgroupEnc, 0);

    auto aElemType = aType.getElementType();
    auto bType = cast<RankedTensorType>(dotOp.getB().getType());
    auto bElemType = bType.getElementType();

    auto newAType =
        RankedTensorType::get(aType.getShape(), aElemType, dotOpEncA);
    auto newBType =
        RankedTensorType::get(bType.getShape(), bElemType, dotOpEncB);

    auto a = ttg::ConvertLayoutOp::create(rewriter, dotOp.getLoc(), newAType,
                                           dotOp.getA());
    auto b = ttg::ConvertLayoutOp::create(rewriter, dotOp.getLoc(), newBType,
                                           dotOp.getB());

    auto cType = cast<RankedTensorType>(dotOp.getC().getType());
    auto newCType = RankedTensorType::get(cType.getShape(),
                                          cType.getElementType(), simdgroupEnc);
    auto c = ttg::ConvertLayoutOp::create(rewriter, dotOp.getLoc(), newCType,
                                           dotOp.getC());

    auto newDot = tt::DotOp::create(
        rewriter, dotOp.getLoc(), newRetType, a, b, c,
        dotOp.getInputPrecision(), dotOp.getMaxNumImpreciseAcc());

    rewriter.replaceOpWithNewOp<ttg::ConvertLayoutOp>(dotOp, oldRetType,
                                                       newDot);
    return success();
  }
};

struct TritonMetalGPUAccelerateMatmulPass
    : public impl::TritonMetalGPUAccelerateMatmulBase<
          TritonMetalGPUAccelerateMatmulPass> {
  using TritonMetalGPUAccelerateMatmulBase::TritonMetalGPUAccelerateMatmulBase;

  void runOnOperation() override {
    auto module = getOperation();
    MLIRContext *ctx = &getContext();
    IRRewriter rewriter(ctx);

    SmallVector<tt::DotOp> dotOps;
    module.walk([&](tt::DotOp op) { dotOps.push_back(op); });

    for (auto dotOp : dotOps) {
      auto oldRetType = cast<RankedTensorType>(dotOp.getType());
      auto encoding = oldRetType.getEncoding();

      auto blockedEnc = dyn_cast<ttg::BlockedEncodingAttr>(encoding);
      if (!blockedEnc)
        continue;

      auto shape = oldRetType.getShape();
      int rank = shape.size();
      bool hasBatch = rank == 3;
      int mIdx = hasBatch ? 1 : 0;
      int nIdx = hasBatch ? 2 : 1;

      int64_t M = shape[mIdx];
      int64_t N = shape[nIdx];

      if (M < kMinShapeForSimdgroup || N < kMinShapeForSimdgroup)
        continue;
      if (M % kSimdgroupM != 0 || N % kSimdgroupN != 0)
        continue;

      auto aType = cast<RankedTensorType>(dotOp.getA().getType());
      int64_t K = aType.getShape().back();
      if (K % kSimdgroupK != 0)
        continue;

      auto elemType = oldRetType.getElementType();
      if (!isSimdgroupAccumTypeSupported(elemType))
        continue;

      {
        auto aElemType = aType.getElementType();
        auto bTy = cast<RankedTensorType>(dotOp.getB().getType());
        auto bElemType = bTy.getElementType();
        if (!isSimdgroupOperandTypeSupported(aElemType, gpuFamily) ||
            !isSimdgroupOperandTypeSupported(bElemType, gpuFamily))
          continue;
        if (aElemType != bElemType)
          continue;
      }

      auto warpsPerCTA = computeWarpsPerCTA(M, N, numWarps);
      SmallVector<unsigned> instrShape = {kSimdgroupM, kSimdgroupN,
                                          kSimdgroupK};

      auto cgaLayout = blockedEnc.getCGALayout();

      SmallVector<unsigned> fullWarpsPerCTA;
      if (hasBatch)
        fullWarpsPerCTA.push_back(1);
      fullWarpsPerCTA.push_back(warpsPerCTA[0]);
      fullWarpsPerCTA.push_back(warpsPerCTA[1]);

      auto simdgroupEnc = ttg::MetalSimdgroupEncodingAttr::get(
          ctx, fullWarpsPerCTA, instrShape, cgaLayout);

      auto newRetType = RankedTensorType::get(shape, elemType, simdgroupEnc);

      auto dotOpEncA =
          ttg::DotOperandEncodingAttr::get(ctx, 0, simdgroupEnc, 0);
      auto dotOpEncB =
          ttg::DotOperandEncodingAttr::get(ctx, 1, simdgroupEnc, 0);

      auto aElemType = aType.getElementType();
      auto bType = cast<RankedTensorType>(dotOp.getB().getType());
      auto bElemType = bType.getElementType();

      auto newAType =
          RankedTensorType::get(aType.getShape(), aElemType, dotOpEncA);
      auto newBType =
          RankedTensorType::get(bType.getShape(), bElemType, dotOpEncB);

      rewriter.setInsertionPoint(dotOp);
      auto a = ttg::ConvertLayoutOp::create(rewriter, dotOp.getLoc(), newAType,
                                            dotOp.getA());
      auto b = ttg::ConvertLayoutOp::create(rewriter, dotOp.getLoc(), newBType,
                                            dotOp.getB());

      auto cType = cast<RankedTensorType>(dotOp.getC().getType());
      auto newCType = RankedTensorType::get(cType.getShape(),
                                            cType.getElementType(),
                                            simdgroupEnc);
      auto c = ttg::ConvertLayoutOp::create(rewriter, dotOp.getLoc(), newCType,
                                            dotOp.getC());

      auto newDot = tt::DotOp::create(
          rewriter, dotOp.getLoc(), newRetType, a, b, c,
          dotOp.getInputPrecision(), dotOp.getMaxNumImpreciseAcc());

      auto result = ttg::ConvertLayoutOp::create(
          rewriter, dotOp.getLoc(), oldRetType, newDot);
      rewriter.replaceOp(dotOp, result);

      LLVM_DEBUG(llvm::dbgs()
                 << "Accelerated dot op to MetalSimdgroup encoding\n");
    }
  }
};

} // namespace

} // namespace mlir
