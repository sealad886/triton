#include "TritonMetalGPUToLLVM/Passes.h"
#include "DotOpToLLVM.h"
#include "FpToFpOpToLLVM.h"
#include "LoadStoreOpToLLVM.h"
#include "NvidiaArtifactLowering.h"
#include "TargetInfo.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/GPUToNVVM/GPUToNVVMPass.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Conversion/UBToLLVM/UBToLLVM.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Pass/Pass.h"
#include "triton/Conversion/TritonGPUToLLVM/ElementwiseOpToLLVMBase.h"
#include "triton/Conversion/TritonGPUToLLVM/PatternTritonGPUOpToLLVM.h"
#include "triton/Conversion/TritonGPUToLLVM/TypeConverter.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Attributes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

namespace mlir {
namespace triton {
#define GEN_PASS_DEF_CONVERTTRITONMETALGPUTOLLVM
#include "TritonMetalGPUToLLVM/Passes.h.inc"
} // namespace triton
} // namespace mlir

using namespace mlir;
using namespace mlir::triton;

namespace {

class TritonLLVMFunctionConversionTarget : public ConversionTarget {
public:
  explicit TritonLLVMFunctionConversionTarget(MLIRContext &ctx)
      : ConversionTarget(ctx) {
    addLegalDialect<LLVM::LLVMDialect>();
    addLegalDialect<NVVM::NVVMDialect>();
    addLegalOp<mlir::UnrealizedConversionCastOp>();
  }
};

class TritonLLVMConversionTarget : public ConversionTarget {
public:
  explicit TritonLLVMConversionTarget(MLIRContext &ctx)
      : ConversionTarget(ctx) {
    addLegalDialect<LLVM::LLVMDialect>();
    addLegalDialect<NVVM::NVVMDialect>();
    addIllegalDialect<triton::TritonDialect>();
    addDynamicallyLegalDialect<triton::gpu::TritonGPUDialect>(
        [](Operation *op) { return isa<triton::gpu::WarpIdOp>(op); });
    addIllegalDialect<mlir::gpu::GPUDialect>();
    addLegalOp<mlir::UnrealizedConversionCastOp>();
    addDynamicallyLegalOp<triton::gpu::GlobalScratchAllocOp>(
        [](triton::gpu::GlobalScratchAllocOp op) {
          return op.getBackend() != "default";
        });
  }
};

struct ConvertTritonMetalGPUToLLVM
    : public triton::impl::ConvertTritonMetalGPUToLLVMBase<ConvertTritonMetalGPUToLLVM> {
  using ConvertTritonMetalGPUToLLVMBase<
      ConvertTritonMetalGPUToLLVM>::ConvertTritonMetalGPUToLLVMBase;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<LLVM::LLVMDialect, NVVM::NVVMDialect, mlir::gpu::GPUDialect>();
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp mod = getOperation();

    mlir::LowerToLLVMOptions option(context);
    option.overrideIndexBitwidth(32);
    Metal::TargetInfo targetInfo(
        mod->getAttrOfType<StringAttr>("ttg.target")
            .getValue()
            .split(':')
            .second);
    Metal::ensureSharedMemorySymbol(mod, targetInfo.getSharedAddressSpace());
    TritonGPUToLLVMTypeConverter typeConverter(context, option, targetInfo);
    TritonLLVMFunctionConversionTarget funcTarget(*context);
    TritonLLVMConversionTarget convTarget(*context);

    // Lower function signatures first; this matches the staged conversion used
    // by other first-class backends and avoids type-conversion dead-ends.
    {
      RewritePatternSet funcPatterns(context);
      mlir::triton::populateFuncOpConversionPattern(
          typeConverter, funcPatterns, targetInfo, patternBenefitDefault);
      if (failed(applyPartialConversion(mod, funcTarget,
                                        std::move(funcPatterns))))
        return signalPassFailure();
    }

    RewritePatternSet patterns(context);
    // Prefer Triton GPU lowering patterns over generic LLVM conversion
    // patterns so Tensor/TT ops are lowered before scalarization/conversion.
    int benefit = patternBenefitPrioritizeOverLLVMConversions;

    ModuleAxisInfoAnalysis axisInfoAnalysis(mod);

    mlir::triton::populateElementwiseOpToLLVMPatterns(typeConverter, patterns, axisInfoAnalysis, targetInfo, benefit);
    // Generic Triton elementwise lowering does not add float binary/scalar-cast
    // patterns that other first-class backends explicitly register.
#define POPULATE_FLOAT_OP(SRC_OP, DST_OP)                                     \
    patterns.add<mlir::triton::gpu::ElementwiseOpConversion<SRC_OP, DST_OP>>( \
        typeConverter, axisInfoAnalysis, benefit);

    POPULATE_FLOAT_OP(arith::SubFOp, LLVM::FSubOp);
    POPULATE_FLOAT_OP(arith::AddFOp, LLVM::FAddOp);
    POPULATE_FLOAT_OP(arith::MulFOp, LLVM::FMulOp);
    POPULATE_FLOAT_OP(arith::DivFOp, LLVM::FDivOp);
    POPULATE_FLOAT_OP(arith::ExtFOp, LLVM::FPExtOp);
    POPULATE_FLOAT_OP(arith::TruncFOp, LLVM::FPTruncOp);
    POPULATE_FLOAT_OP(arith::FPToSIOp, LLVM::FPToSIOp);
    POPULATE_FLOAT_OP(arith::SIToFPOp, LLVM::SIToFPOp);

#undef POPULATE_FLOAT_OP

    Metal::populateFpToFpOpToLLVMPatterns(typeConverter, patterns, benefit);
    Metal::populateDotOpToLLVMPatterns(typeConverter, patterns, benefit);
    Metal::populateLoadStoreOpToLLVMPatterns(typeConverter, patterns, benefit);
    mlir::triton::populateMemoryOpToLLVMPatterns(typeConverter, targetInfo, patterns, benefit);
    mlir::triton::populateAssertOpToLLVMPattern(typeConverter, patterns, targetInfo, benefit);
    mlir::triton::populateMakeRangeOpToLLVMPattern(typeConverter, targetInfo, patterns, benefit);
    mlir::triton::populateViewOpToLLVMPatterns(typeConverter, patterns, benefit);
    mlir::triton::populateMinMaxFOpToLLVMPattern(typeConverter, patterns, axisInfoAnalysis, false, benefit);
    mlir::triton::populateClampFOpToLLVMPattern(typeConverter, patterns, axisInfoAnalysis, targetInfo, benefit);
    mlir::triton::populateHistogramOpToLLVMPatterns(typeConverter, patterns, targetInfo, benefit);
    mlir::triton::populateReduceOpToLLVMPatterns(typeConverter, patterns, targetInfo, benefit);
    mlir::triton::populateScanOpToLLVMPatterns(typeConverter, patterns, targetInfo, benefit);
    mlir::triton::populateGatherOpToLLVMPatterns(typeConverter, patterns, targetInfo, benefit);
    mlir::triton::populateConvertLayoutOpToLLVMPatterns(typeConverter, targetInfo, patterns, benefit);
    mlir::triton::populateControlFlowOpToLLVMPattern(typeConverter, patterns, targetInfo, benefit);
    mlir::triton::populateSPMDOpToLLVMPattern(typeConverter, patterns, targetInfo, benefit);
    mlir::triton::populatePrintOpToLLVMPattern(typeConverter, patterns, targetInfo, benefit);
    mlir::triton::populateInstrumentationToLLVMPatterns(typeConverter, patterns);

    // Add standard MLIR to LLVM patterns
    mlir::arith::populateCeilFloorDivExpandOpsPatterns(patterns);
    mlir::arith::populateArithToLLVMConversionPatterns(typeConverter, patterns);
    mlir::populateMathToLLVMConversionPatterns(typeConverter, patterns);
    mlir::populateGpuToNVVMConversionPatterns(typeConverter, patterns);
    mlir::cf::populateControlFlowToLLVMConversionPatterns(typeConverter, patterns);
    mlir::ub::populateUBToLLVMConversionPatterns(typeConverter, patterns);

    if (failed(applyPartialConversion(mod, convTarget, std::move(patterns))))
      return signalPassFailure();

    Metal::lowerNvidiaArtifactsToMetal(mod);
  }
};

} // anonymous namespace

namespace mlir {
namespace triton {

std::unique_ptr<OperationPass<ModuleOp>> createConvertTritonMetalGPUToLLVMPass() {
  return std::make_unique<ConvertTritonMetalGPUToLLVM>();
}

} // namespace triton
} // namespace mlir
