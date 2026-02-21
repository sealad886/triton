#include "TritonMetalGPUToLLVM/Passes.h"
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
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "triton/Conversion/TritonGPUToLLVM/ElementwiseOpToLLVMBase.h"
#include "triton/Conversion/TritonGPUToLLVM/PatternTritonGPUOpToLLVM.h"
#include "triton/Conversion/TritonGPUToLLVM/TypeConverter.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
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
    addLegalDialect<cf::ControlFlowDialect>();
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

static std::string mangleTypeForSymbol(Type ty) {
  if (auto intTy = dyn_cast<IntegerType>(ty))
    return "i" + std::to_string(intTy.getWidth());
  if (ty.isF16())
    return "f16";
  if (ty.isF32())
    return "f32";
  if (ty.isF64())
    return "f64";
  if (auto ptrTy = dyn_cast<LLVM::LLVMPointerType>(ty))
    return "p" + std::to_string(ptrTy.getAddressSpace());
  if (isa<LLVM::LLVMVoidType>(ty))
    return "void";
  return "ty";
}

static LLVM::LLVMFuncOp getOrInsertExternFunc(ModuleOp mod, OpBuilder &builder,
                                              StringRef baseName, Type retTy,
                                              ArrayRef<Type> argTys,
                                              StringRef suffix = "") {
  std::string fullName =
      suffix.empty() ? baseName.str() : (baseName + "_" + suffix).str();
  if (auto fn = mod.lookupSymbol<LLVM::LLVMFuncOp>(fullName))
    return fn;

  auto fnTy = LLVM::LLVMFunctionType::get(retTy, argTys, /*isVarArg=*/false);
  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(mod.getBody());
  auto fn = LLVM::LLVMFuncOp::create(builder, UnknownLoc::get(mod.getContext()),
                                     fullName, fnTy);
  fn.setVisibility(SymbolTable::Visibility::Private);
  return fn;
}

struct LoadOpConversion : public ConvertOpToLLVMPattern<triton::LoadOp> {
  LoadOpConversion(LLVMTypeConverter &typeConverter, PatternBenefit benefit)
      : ConvertOpToLLVMPattern<triton::LoadOp>(typeConverter, benefit) {}

  LogicalResult
  matchAndRewrite(triton::LoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    ModuleOp mod = op->getParentOfType<ModuleOp>();
    if (!mod)
      return failure();

    SmallVector<Value> ptrElems;
    SmallVector<Value> maskElems;
    SmallVector<Value> otherElems;
    SmallVector<Value> loadedVals;

    bool isTensor = isa<RankedTensorType>(op.getType());
    if (isTensor)
      ptrElems = unpackLLElements(loc, adaptor.getPtr(), rewriter);
    else
      ptrElems.push_back(adaptor.getPtr());

    if (op.getMask()) {
      if (isTensor)
        maskElems = unpackLLElements(loc, adaptor.getMask(), rewriter);
      else
        maskElems.push_back(adaptor.getMask());
    }

    if (op.getOther()) {
      if (isTensor)
        otherElems = unpackLLElements(loc, adaptor.getOther(), rewriter);
      else
        otherElems.push_back(adaptor.getOther());
    }

    Type llvmElemTy = getTypeConverter()->convertType(getElementTypeOrSelf(op.getType()));
    if (!llvmElemTy)
      return failure();

    loadedVals.reserve(ptrElems.size());
    for (auto [idx, ptr] : llvm::enumerate(ptrElems)) {
      Value pred = op.getMask() ? maskElems[idx] : Value();
      Value other = op.getOther()
                        ? otherElems[idx]
                        : Value(LLVM::UndefOp::create(rewriter, loc, llvmElemTy));
      Value loaded;
      if (pred) {
        std::string suffix = mangleTypeForSymbol(llvmElemTy) + "_" +
                             mangleTypeForSymbol(ptr.getType());
        auto fn = getOrInsertExternFunc(
            mod, rewriter, "__metal_predicated_ld_global", llvmElemTy,
            {llvmElemTy, ptr.getType(), pred.getType()}, suffix);
        loaded = LLVM::CallOp::create(rewriter, loc, fn,
                                      ValueRange{other, ptr, pred})
                     ->getResult(0);
      } else {
        loaded = LLVM::LoadOp::create(rewriter, loc, llvmElemTy, ptr);
      }
      loadedVals.push_back(loaded);
    }

    if (!isTensor) {
      rewriter.replaceOp(op, loadedVals.front());
      return success();
    }

    auto resultTy = cast<RankedTensorType>(op.getType());
    Value result = packLLElements(loc, getTypeConverter(), loadedVals, rewriter,
                                  resultTy);
    rewriter.replaceOp(op, result);
    return success();
  }
};

struct StoreOpConversion : public ConvertOpToLLVMPattern<triton::StoreOp> {
  StoreOpConversion(LLVMTypeConverter &typeConverter, PatternBenefit benefit)
      : ConvertOpToLLVMPattern<triton::StoreOp>(typeConverter, benefit) {}

  LogicalResult
  matchAndRewrite(triton::StoreOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    ModuleOp mod = op->getParentOfType<ModuleOp>();
    if (!mod)
      return failure();

    SmallVector<Value> ptrElems;
    SmallVector<Value> valElems;
    SmallVector<Value> maskElems;

    bool isTensor = isa<RankedTensorType>(op.getValue().getType());
    if (isTensor) {
      ptrElems = unpackLLElements(loc, adaptor.getPtr(), rewriter);
      valElems = unpackLLElements(loc, adaptor.getValue(), rewriter);
    } else {
      ptrElems.push_back(adaptor.getPtr());
      valElems.push_back(adaptor.getValue());
    }

    if (op.getMask()) {
      if (isTensor)
        maskElems = unpackLLElements(loc, adaptor.getMask(), rewriter);
      else
        maskElems.push_back(adaptor.getMask());
    }

    Type voidTy = LLVM::LLVMVoidType::get(op.getContext());
    for (auto idx : llvm::seq<size_t>(0, ptrElems.size())) {
      Value pred = op.getMask() ? maskElems[idx] : Value();
      if (pred) {
        std::string suffix = mangleTypeForSymbol(valElems[idx].getType()) +
                             "_" + mangleTypeForSymbol(ptrElems[idx].getType());
        auto fn = getOrInsertExternFunc(
            mod, rewriter, "__metal_predicated_st_global", voidTy,
            {valElems[idx].getType(), ptrElems[idx].getType(), pred.getType()},
            suffix);
        LLVM::CallOp::create(rewriter, loc, fn,
                             ValueRange{valElems[idx], ptrElems[idx], pred});
      } else {
        LLVM::StoreOp::create(rewriter, loc, valElems[idx], ptrElems[idx]);
      }
    }

    rewriter.eraseOp(op);
    return success();
  }
};

static void stripNVVMAttrs(ModuleOp mod) {
  for (auto fn : mod.getOps<LLVM::LLVMFuncOp>()) {
    SmallVector<StringAttr> toErase;
    for (auto attr : fn->getAttrs()) {
      if (attr.getName().strref().starts_with("nvvm."))
        toErase.push_back(attr.getName());
    }
    for (auto attrName : toErase)
      fn->removeAttr(attrName);
  }
}

static std::optional<StringRef> mapNVVMReadSRegToMetalBuiltin(StringRef opName) {
  if (opName == "nvvm.read.ptx.sreg.tid.x")
    return "__metal_get_thread_position_in_threadgroup_x";
  if (opName == "nvvm.read.ptx.sreg.tid.y")
    return "__metal_get_thread_position_in_threadgroup_y";
  if (opName == "nvvm.read.ptx.sreg.tid.z")
    return "__metal_get_thread_position_in_threadgroup_z";
  if (opName == "nvvm.read.ptx.sreg.ctaid.x")
    return "__metal_get_threadgroup_position_in_grid_x";
  if (opName == "nvvm.read.ptx.sreg.ctaid.y")
    return "__metal_get_threadgroup_position_in_grid_y";
  if (opName == "nvvm.read.ptx.sreg.ctaid.z")
    return "__metal_get_threadgroup_position_in_grid_z";
  if (opName == "nvvm.read.ptx.sreg.ntid.x")
    return "__metal_get_threads_per_threadgroup_x";
  if (opName == "nvvm.read.ptx.sreg.ntid.y")
    return "__metal_get_threads_per_threadgroup_y";
  if (opName == "nvvm.read.ptx.sreg.ntid.z")
    return "__metal_get_threads_per_threadgroup_z";
  if (opName == "nvvm.read.ptx.sreg.nctaid.x")
    return "__metal_get_threadgroups_per_grid_x";
  if (opName == "nvvm.read.ptx.sreg.nctaid.y")
    return "__metal_get_threadgroups_per_grid_y";
  if (opName == "nvvm.read.ptx.sreg.nctaid.z")
    return "__metal_get_threadgroups_per_grid_z";
  return std::nullopt;
}

static void rewriteNVVMSRegs(ModuleOp mod) {
  SmallVector<Operation *> nvvmReadOps;
  mod.walk([&](Operation *op) {
    if (mapNVVMReadSRegToMetalBuiltin(op->getName().getStringRef()).has_value())
      nvvmReadOps.push_back(op);
  });

  OpBuilder builder(mod.getContext());
  for (Operation *op : nvvmReadOps) {
    if (!op || op->getNumResults() != 1)
      continue;

    auto builtinName =
        mapNVVMReadSRegToMetalBuiltin(op->getName().getStringRef());
    if (!builtinName)
      continue;

    Type resultTy = op->getResult(0).getType();
    builder.setInsertionPoint(op);
    auto fn = getOrInsertExternFunc(mod, builder, *builtinName, resultTy, {});
    auto call = LLVM::CallOp::create(builder, op->getLoc(), fn, ValueRange{});
    op->replaceAllUsesWith(call->getResults());
    op->erase();
  }
}

static void rewriteWarpId(ModuleOp mod) {
  SmallVector<Operation *> warpIdOps;
  mod.walk([&](Operation *op) {
    if (op->getName().getStringRef() == "ttg.warp_id")
      warpIdOps.push_back(op);
  });

  OpBuilder builder(mod.getContext());
  unsigned threadsPerWarp = triton::gpu::TritonGPUDialect::getThreadsPerWarp(mod);
  Type i32Ty = builder.getI32Type();

  for (Operation *op : warpIdOps) {
    if (!op || op->getNumResults() != 1)
      continue;

    builder.setInsertionPoint(op);
    auto tidFn = getOrInsertExternFunc(
        mod, builder, "__metal_get_thread_position_in_threadgroup_x", i32Ty, {});
    Value tid = LLVM::CallOp::create(builder, op->getLoc(), tidFn, ValueRange{})
                    ->getResult(0);
    Value warpSize = LLVM::ConstantOp::create(
        builder, op->getLoc(), i32Ty, builder.getI32IntegerAttr(threadsPerWarp));
    Value warpId = LLVM::UDivOp::create(builder, op->getLoc(), tid, warpSize);

    Value out = warpId;
    Type resultTy = op->getResult(0).getType();
    if (resultTy != i32Ty) {
      if (auto intTy = dyn_cast<IntegerType>(resultTy)) {
        if (intTy.getWidth() < 32)
          out = LLVM::TruncOp::create(builder, op->getLoc(), resultTy, out);
        else if (intTy.getWidth() > 32)
          out = LLVM::ZExtOp::create(builder, op->getLoc(), resultTy, out);
      } else {
        continue;
      }
    }

    op->getResult(0).replaceAllUsesWith(out);
    op->erase();
  }
}

static void rewriteGlobalInlineAsm(ModuleOp mod) {
  SmallVector<LLVM::InlineAsmOp> inlineAsmOps;
  mod.walk([&](LLVM::InlineAsmOp op) { inlineAsmOps.push_back(op); });

  OpBuilder builder(mod.getContext());
  Type voidTy = LLVM::LLVMVoidType::get(mod.getContext());
  for (LLVM::InlineAsmOp op : inlineAsmOps) {
    StringRef asmString = op.getAsmString();
    bool isGlobalLoad = asmString.contains("ld.global.");
    bool isGlobalStore = asmString.contains("st.global.");
    if (!isGlobalLoad && !isGlobalStore)
      continue;

    auto operands = op.getOperands();
    if (operands.size() < 3)
      continue;

    builder.setInsertionPoint(op);
    if (isGlobalLoad) {
      if (op->getNumResults() != 1)
        continue;
      Type resultTy = op->getResult(0).getType();
      std::string suffix =
          mangleTypeForSymbol(resultTy) + "_" + mangleTypeForSymbol(operands[1].getType());
      auto fn = getOrInsertExternFunc(
          mod, builder, "__metal_predicated_ld_global", resultTy,
          {resultTy, operands[1].getType(), operands[2].getType()}, suffix);
      auto call = LLVM::CallOp::create(
          builder, op.getLoc(), fn,
          ValueRange{operands[0], operands[1], operands[2]});
      op.replaceAllUsesWith(call->getResults());
      op.erase();
      continue;
    }

    if (!op->use_empty())
      continue;
    std::string suffix =
        mangleTypeForSymbol(operands[0].getType()) + "_" +
        mangleTypeForSymbol(operands[1].getType());
    auto fn = getOrInsertExternFunc(
        mod, builder, "__metal_predicated_st_global", voidTy,
        {operands[0].getType(), operands[1].getType(), operands[2].getType()},
        suffix);
    LLVM::CallOp::create(builder, op.getLoc(), fn,
                         ValueRange{operands[0], operands[1], operands[2]});
    op.erase();
  }
}

static void lowerNvidiaArtifactsToMetal(ModuleOp mod) {
  stripNVVMAttrs(mod);
  rewriteNVVMSRegs(mod);
  rewriteWarpId(mod);
  rewriteGlobalInlineAsm(mod);
}

static void ensureSharedMemorySymbol(ModuleOp mod, unsigned addrSpace) {
  if (mod.lookupSymbol("global_smem"))
    return;

  OpBuilder builder(mod.getContext());
  builder.setInsertionPointToStart(mod.getBody());
  auto i8Ty = IntegerType::get(mod.getContext(), 8);
  auto arrayTy = LLVM::LLVMArrayType::get(i8Ty, 0);
  LLVM::GlobalOp::create(builder, UnknownLoc::get(mod.getContext()), arrayTy,
                         /*isConstant=*/false, LLVM::Linkage::External,
                         "global_smem", /*value=*/Attribute(),
                         /*alignment=*/16, addrSpace);
}

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
    ensureSharedMemorySymbol(mod, targetInfo.getSharedAddressSpace());
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

    patterns.add<LoadOpConversion, StoreOpConversion>(typeConverter, benefit);
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

    lowerNvidiaArtifactsToMetal(mod);
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
