#include "LoadStoreOpToLLVM.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/SymbolTable.h"
#include "triton/Conversion/TritonGPUToLLVM/PatternTritonGPUOpToLLVM.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

using namespace mlir;
using namespace mlir::triton;

namespace mlir::triton::Metal {

std::string mangleTypeForSymbol(Type ty) {
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

LLVM::LLVMFuncOp getOrInsertExternFunc(ModuleOp mod, OpBuilder &builder,
                                        StringRef baseName, Type retTy,
                                        ArrayRef<Type> argTys,
                                        StringRef suffix) {
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

} // namespace mlir::triton::Metal

namespace {

using namespace mlir::triton::Metal;

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

} // anonymous namespace

void mlir::triton::Metal::populateLoadStoreOpToLLVMPatterns(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    PatternBenefit benefit) {
  patterns.add<LoadOpConversion, StoreOpConversion>(typeConverter, benefit);
}
