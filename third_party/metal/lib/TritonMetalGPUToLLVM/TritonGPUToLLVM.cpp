#include "TritonMetalGPUToLLVM/Passes.h"
#include "TargetInfo.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Conversion/UBToLLVM/UBToLLVM.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Pass/Pass.h"
#include "triton/Conversion/TritonGPUToLLVM/PatternTritonGPUOpToLLVM.h"
#include "triton/Conversion/TritonGPUToLLVM/TypeConverter.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
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
    addLegalOp<mlir::UnrealizedConversionCastOp>();
  }
};

struct ConvertTritonMetalGPUToLLVM
    : public triton::impl::ConvertTritonMetalGPUToLLVMBase<ConvertTritonMetalGPUToLLVM> {
  using ConvertTritonMetalGPUToLLVMBase<
      ConvertTritonMetalGPUToLLVM>::ConvertTritonMetalGPUToLLVMBase;

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
    TritonGPUToLLVMTypeConverter typeConverter(context, option, targetInfo);
    TritonLLVMFunctionConversionTarget target(*context);

    RewritePatternSet patterns(context);
    int benefit = patternBenefitDefault;

    ModuleAxisInfoAnalysis axisInfoAnalysis(mod);

    mlir::triton::populateFuncOpConversionPattern(typeConverter, patterns, targetInfo, benefit);
    mlir::triton::populateElementwiseOpToLLVMPatterns(typeConverter, patterns, axisInfoAnalysis, targetInfo, benefit);
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
    mlir::arith::populateArithToLLVMConversionPatterns(typeConverter, patterns);
    mlir::populateMathToLLVMConversionPatterns(typeConverter, patterns);
    mlir::cf::populateControlFlowToLLVMConversionPatterns(typeConverter, patterns);
    mlir::ub::populateUBToLLVMConversionPatterns(typeConverter, patterns);

    if (failed(applyPartialConversion(mod, target, std::move(patterns))))
      return signalPassFailure();
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
