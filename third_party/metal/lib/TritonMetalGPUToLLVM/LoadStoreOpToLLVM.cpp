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
  if (ty.isBF16())
    return "bf16";
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

static Value emitMetalRedundantThreadPred(
  Type ptrType, ModuleOp mod, ConversionPatternRewriter &rewriter,
  Location loc);
static int32_t getRegisterFreeVarMask(Type type);

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

    int addrSpace = triton::getAddressSpace(op.getPtr().getType());
    Value threadPred;
    int32_t regMask = 0;
    if (addrSpace != 3) {
      threadPred =
          emitMetalRedundantThreadPred(op.getPtr().getType(), mod, rewriter, loc);
      regMask = getRegisterFreeVarMask(op.getPtr().getType());
    }

    Type voidTy = LLVM::LLVMVoidType::get(op.getContext());
    for (auto idx : llvm::seq<size_t>(0, ptrElems.size())) {
      if (threadPred && !isCanonicalIndex(idx, regMask))
        continue;

      Value pred = op.getMask() ? maskElems[idx] : Value();
      if (threadPred)
        pred = pred ? LLVM::AndOp::create(rewriter, loc, pred, threadPred)
                    : threadPred;
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

// ── Atomic helpers ──────────────────────────────────────────────────

static LLVM::AtomicBinOp tritonRMWToLLVM(RMWOp op) {
  switch (op) {
  case RMWOp::AND:  return LLVM::AtomicBinOp::_and;
  case RMWOp::OR:   return LLVM::AtomicBinOp::_or;
  case RMWOp::XOR:  return LLVM::AtomicBinOp::_xor;
  case RMWOp::ADD:  return LLVM::AtomicBinOp::add;
  case RMWOp::FADD: return LLVM::AtomicBinOp::fadd;
  case RMWOp::MAX:  return LLVM::AtomicBinOp::max;
  case RMWOp::MIN:  return LLVM::AtomicBinOp::min;
  case RMWOp::UMAX: return LLVM::AtomicBinOp::umax;
  case RMWOp::UMIN: return LLVM::AtomicBinOp::umin;
  case RMWOp::XCHG: return LLVM::AtomicBinOp::xchg;
  }
  llvm_unreachable("unhandled RMWOp");
}

static LLVM::AtomicOrdering tritonSemToLLVM(MemSemantic sem) {
  // Metal Shading Language only supports memory_order_relaxed for device
  // atomics, which corresponds to LLVM monotonic ordering.  Ignore the
  // requested ordering and always emit monotonic.
  (void)sem;
  return LLVM::AtomicOrdering::monotonic;
}

// Compute a predicate that is true only when this thread is the canonical
// representative for its data.  For scalar types every thread is redundant
// except thread 0.  For tensor types `getFreeVariableMasks` identifies which
// lane / warp bits are free (i.e. redundant).
static Value emitMetalRedundantThreadPred(
    Type ptrType, ModuleOp mod, ConversionPatternRewriter &rewriter,
    Location loc) {
  auto freeVarMasks = getFreeVariableMasks(ptrType);
  auto ctx = rewriter.getContext();
  auto kLane = StringAttr::get(ctx, "lane");
  auto kWarp = StringAttr::get(ctx, "warp");

  int32_t laneMask = freeVarMasks.lookup(kLane);
  int32_t warpMask = freeVarMasks.lookup(kWarp);
  if (laneMask == 0 && warpMask == 0)
    return nullptr; // no redundancy

  auto i32Ty = rewriter.getI32Type();
  auto tidFn = getOrInsertExternFunc(
      mod, rewriter, "__metal_get_thread_position_in_threadgroup_x", i32Ty, {});
  Value tid =
      LLVM::CallOp::create(rewriter, loc, tidFn, ValueRange{})->getResult(0);
  Value zero = LLVM::ConstantOp::create(rewriter, loc, i32Ty,
                                         rewriter.getI32IntegerAttr(0));

  Value pred;
  auto andPred = [&](Value newPred) {
    pred = pred ? LLVM::AndOp::create(rewriter, loc, pred, newPred) : newPred;
  };

  if (laneMask != 0) {
    Value laneId = LLVM::AndOp::create(
        rewriter, loc, tid,
        LLVM::ConstantOp::create(rewriter, loc, i32Ty,
                                  rewriter.getI32IntegerAttr(31)));
    Value maskVal = LLVM::ConstantOp::create(rewriter, loc, i32Ty,
                                              rewriter.getI32IntegerAttr(laneMask));
    Value masked = LLVM::AndOp::create(rewriter, loc, laneId, maskVal);
    andPred(LLVM::ICmpOp::create(rewriter, loc, LLVM::ICmpPredicate::eq,
                                  masked, zero));
  }

  if (warpMask != 0) {
    Value c32 = LLVM::ConstantOp::create(rewriter, loc, i32Ty,
                                          rewriter.getI32IntegerAttr(32));
    Value warpId = LLVM::UDivOp::create(rewriter, loc, tid, c32);
    Value maskVal = LLVM::ConstantOp::create(rewriter, loc, i32Ty,
                                              rewriter.getI32IntegerAttr(warpMask));
    Value masked = LLVM::AndOp::create(rewriter, loc, warpId, maskVal);
    andPred(LLVM::ICmpOp::create(rewriter, loc, LLVM::ICmpPredicate::eq,
                                  masked, zero));
  }

  return pred;
}

// Return the register free-variable mask for a type.
static int32_t getRegisterFreeVarMask(Type type) {
  auto freeVarMasks = getFreeVariableMasks(type);
  auto kReg = StringAttr::get(type.getContext(), "register");
  return freeVarMasks.lookup(kReg);
}

// ── AtomicRMWOpConversion ───────────────────────────────────────────

struct AtomicRMWOpConversion
    : public ConvertOpToLLVMPattern<triton::AtomicRMWOp> {
  AtomicRMWOpConversion(LLVMTypeConverter &typeConverter,
                        PatternBenefit benefit)
      : ConvertOpToLLVMPattern<triton::AtomicRMWOp>(typeConverter, benefit) {}

  LogicalResult
  matchAndRewrite(triton::AtomicRMWOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto valueTy = op.getType();
    auto tensorTy = dyn_cast<RankedTensorType>(valueTy);
    Type valueElemTy =
        tensorTy ? getTypeConverter()->convertType(tensorTy.getElementType())
                 : getTypeConverter()->convertType(valueTy);
    if (!valueElemTy)
      return failure();

    auto llvmBinOp = tritonRMWToLLVM(op.getAtomicRmwOp());
    auto ordering = tritonSemToLLVM(op.getSem());

    // ── Scalar (non-tensor) path ──
    // All threads in the threadgroup hold the same scalar value.
    // Only thread 0 must execute the atomic to avoid redundant writes.
    if (!tensorTy) {
      Value ptr = adaptor.getPtr();
      Value val = adaptor.getVal();

      auto mod = op->getParentOfType<ModuleOp>();
      auto i32Ty = rewriter.getI32Type();
      auto tidFn = getOrInsertExternFunc(
          mod, rewriter, "__metal_get_thread_position_in_threadgroup_x",
          i32Ty, {});
      Value tid =
          LLVM::CallOp::create(rewriter, loc, tidFn, ValueRange{})
              ->getResult(0);
      Value zero_i32 = LLVM::ConstantOp::create(rewriter, loc, i32Ty,
                                                 rewriter.getI32IntegerAttr(0));
      Value pred = LLVM::ICmpOp::create(rewriter, loc,
                                         LLVM::ICmpPredicate::eq, tid,
                                         zero_i32);

      if (op.getMask())
        pred = LLVM::AndOp::create(rewriter, loc, pred, adaptor.getMask());

      Value one = LLVM::ConstantOp::create(rewriter, loc, i32Ty,
                                            rewriter.getI32IntegerAttr(1));
      Value slot = LLVM::AllocaOp::create(
          rewriter, loc,
          LLVM::LLVMPointerType::get(rewriter.getContext()),
          valueElemTy, one, /*alignment=*/0);
      Value undef = LLVM::UndefOp::create(rewriter, loc, valueElemTy);
      LLVM::StoreOp::create(rewriter, loc, undef, slot);

      auto [prevBlock, ifBlock, thenBlock] =
          createIfBlock(rewriter, loc, pred);
      (void)prevBlock;
      rewriter.setInsertionPointToStart(ifBlock);
      Value oldVal = LLVM::AtomicRMWOp::create(rewriter, loc, llvmBinOp, ptr,
                                                val, ordering);
      LLVM::StoreOp::create(rewriter, loc, oldVal, slot);
      rewriter.setInsertionPointToStart(thenBlock);
      Value result = LLVM::LoadOp::create(rewriter, loc, valueElemTy, slot);
      rewriter.replaceOp(op, result);
      return success();
    }

    // ── Tensor path: element-wise atomics ──
    // When threads > tensor elements, some threads hold duplicate data.
    // Use free-variable masks to skip redundant threads/registers.
    auto mod = op->getParentOfType<ModuleOp>();
    Value threadPred =
        emitMetalRedundantThreadPred(op.getPtr().getType(), mod, rewriter, loc);
    int32_t regMask = getRegisterFreeVarMask(op.getPtr().getType());

    SmallVector<Value> ptrElems =
        unpackLLElements(loc, adaptor.getPtr(), rewriter);
    SmallVector<Value> valElems =
        unpackLLElements(loc, adaptor.getVal(), rewriter);
    SmallVector<Value> maskElems;
    if (op.getMask())
      maskElems = unpackLLElements(loc, adaptor.getMask(), rewriter);

    SmallVector<Value> resultVals;
    resultVals.reserve(ptrElems.size());

    for (size_t i = 0; i < ptrElems.size(); ++i) {
      // Skip non-canonical register indices (redundant within a thread).
      if (!isCanonicalIndex(i, regMask)) {
        resultVals.push_back(LLVM::UndefOp::create(rewriter, loc, valueElemTy));
        continue;
      }

      Value ptr = ptrElems[i];
      Value val = valElems[i];

      // Build combined predicate: threadPred AND userMask.
      Value pred = threadPred;
      if (!maskElems.empty()) {
        Value userMask = maskElems[i];
        pred = pred ? LLVM::AndOp::create(rewriter, loc, pred, userMask)
                    : userMask;
      }

      if (pred) {
        Value one = LLVM::ConstantOp::create(rewriter, loc,
                                             rewriter.getI32Type(),
                                             rewriter.getI32IntegerAttr(1));
        Value slot = LLVM::AllocaOp::create(rewriter, loc,
                                            LLVM::LLVMPointerType::get(rewriter.getContext()),
                                            valueElemTy, one, /*alignment=*/0);
        Value undef = LLVM::UndefOp::create(rewriter, loc, valueElemTy);
        LLVM::StoreOp::create(rewriter, loc, undef, slot);

        auto [prevBlock, ifBlock, thenBlock] =
            createIfBlock(rewriter, loc, pred);
        (void)prevBlock;
        rewriter.setInsertionPointToStart(ifBlock);
        Value oldVal = LLVM::AtomicRMWOp::create(rewriter, loc, llvmBinOp, ptr,
                                                  val, ordering);
        LLVM::StoreOp::create(rewriter, loc, oldVal, slot);
        rewriter.setInsertionPointToStart(thenBlock);
        Value result = LLVM::LoadOp::create(rewriter, loc, valueElemTy, slot);
        resultVals.push_back(result);
      } else {
        Value result = LLVM::AtomicRMWOp::create(rewriter, loc, llvmBinOp, ptr,
                                                  val, ordering);
        resultVals.push_back(result);
      }
    }

    Value packed = packLLElements(loc, getTypeConverter(), resultVals, rewriter,
                                  tensorTy);
    rewriter.replaceOp(op, packed);
    return success();
  }
};

// ── AtomicCASOpConversion ───────────────────────────────────────────

struct AtomicCASOpConversion
    : public ConvertOpToLLVMPattern<triton::AtomicCASOp> {
  AtomicCASOpConversion(LLVMTypeConverter &typeConverter,
                        PatternBenefit benefit)
      : ConvertOpToLLVMPattern<triton::AtomicCASOp>(typeConverter, benefit) {}

  LogicalResult
  matchAndRewrite(triton::AtomicCASOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto valueTy = op.getType();
    auto tensorTy = dyn_cast<RankedTensorType>(valueTy);
    Type valueElemTy =
        tensorTy ? getTypeConverter()->convertType(tensorTy.getElementType())
                 : getTypeConverter()->convertType(valueTy);
    if (!valueElemTy)
      return failure();

    auto ordering = tritonSemToLLVM(op.getSem());
    auto failOrdering = LLVM::AtomicOrdering::monotonic;

    // ── Scalar (non-tensor) path: thread-0-only guard ──
    if (!tensorTy) {
      Value ptr = adaptor.getPtr();
      Value cmp = adaptor.getCmp();
      Value val = adaptor.getVal();

      auto mod = op->getParentOfType<ModuleOp>();
      auto i32Ty = rewriter.getI32Type();
      auto tidFn = getOrInsertExternFunc(
          mod, rewriter, "__metal_get_thread_position_in_threadgroup_x",
          i32Ty, {});
      Value tid =
          LLVM::CallOp::create(rewriter, loc, tidFn, ValueRange{})
              ->getResult(0);
      Value zero_i32 = LLVM::ConstantOp::create(rewriter, loc, i32Ty,
                                                 rewriter.getI32IntegerAttr(0));
      Value pred = LLVM::ICmpOp::create(rewriter, loc,
                                         LLVM::ICmpPredicate::eq, tid,
                                         zero_i32);

      Value one = LLVM::ConstantOp::create(rewriter, loc, i32Ty,
                                            rewriter.getI32IntegerAttr(1));
      Value slot = LLVM::AllocaOp::create(
          rewriter, loc,
          LLVM::LLVMPointerType::get(rewriter.getContext()),
          valueElemTy, one, /*alignment=*/0);
      Value undef = LLVM::UndefOp::create(rewriter, loc, valueElemTy);
      LLVM::StoreOp::create(rewriter, loc, undef, slot);

      auto [prevBlock, ifBlock, thenBlock] =
          createIfBlock(rewriter, loc, pred);
      (void)prevBlock;
      rewriter.setInsertionPointToStart(ifBlock);
      Value casResult = LLVM::AtomicCmpXchgOp::create(
          rewriter, loc, ptr, cmp, val, ordering, failOrdering);
      Value oldVal = LLVM::ExtractValueOp::create(rewriter, loc, valueElemTy,
                                                  casResult,
                                                  ArrayRef<int64_t>{0});
      LLVM::StoreOp::create(rewriter, loc, oldVal, slot);
      rewriter.setInsertionPointToStart(thenBlock);
      Value result = LLVM::LoadOp::create(rewriter, loc, valueElemTy, slot);
      rewriter.replaceOp(op, result);
      return success();
    }

    // ── Tensor path ──
    auto mod = op->getParentOfType<ModuleOp>();
    Value threadPred =
        emitMetalRedundantThreadPred(op.getPtr().getType(), mod, rewriter, loc);
    int32_t regMask = getRegisterFreeVarMask(op.getPtr().getType());

    SmallVector<Value> ptrElems =
        unpackLLElements(loc, adaptor.getPtr(), rewriter);
    SmallVector<Value> cmpElems =
        unpackLLElements(loc, adaptor.getCmp(), rewriter);
    SmallVector<Value> valElems =
        unpackLLElements(loc, adaptor.getVal(), rewriter);

    SmallVector<Value> resultVals;
    resultVals.reserve(ptrElems.size());

    for (size_t i = 0; i < ptrElems.size(); ++i) {
      if (!isCanonicalIndex(i, regMask)) {
        resultVals.push_back(LLVM::UndefOp::create(rewriter, loc, valueElemTy));
        continue;
      }

      if (threadPred) {
        Value one = LLVM::ConstantOp::create(rewriter, loc,
                                             rewriter.getI32Type(),
                                             rewriter.getI32IntegerAttr(1));
        Value slot = LLVM::AllocaOp::create(rewriter, loc,
                                            LLVM::LLVMPointerType::get(rewriter.getContext()),
                                            valueElemTy, one, /*alignment=*/0);
        Value undef = LLVM::UndefOp::create(rewriter, loc, valueElemTy);
        LLVM::StoreOp::create(rewriter, loc, undef, slot);

        auto [prevBlock, ifBlock, thenBlock] =
            createIfBlock(rewriter, loc, threadPred);
        (void)prevBlock;
        rewriter.setInsertionPointToStart(ifBlock);
        Value casResult = LLVM::AtomicCmpXchgOp::create(
            rewriter, loc, ptrElems[i], cmpElems[i], valElems[i], ordering,
            failOrdering);
        Value oldVal = LLVM::ExtractValueOp::create(rewriter, loc, valueElemTy,
                                                    casResult,
                                                    ArrayRef<int64_t>{0});
        LLVM::StoreOp::create(rewriter, loc, oldVal, slot);
        rewriter.setInsertionPointToStart(thenBlock);
        resultVals.push_back(LLVM::LoadOp::create(rewriter, loc, valueElemTy, slot));
      } else {
        Value casResult = LLVM::AtomicCmpXchgOp::create(
            rewriter, loc, ptrElems[i], cmpElems[i], valElems[i], ordering,
            failOrdering);
        Value oldVal = LLVM::ExtractValueOp::create(rewriter, loc, valueElemTy,
                                                    casResult,
                                                    ArrayRef<int64_t>{0});
        resultVals.push_back(oldVal);
      }
    }

    Value packed = packLLElements(loc, getTypeConverter(), resultVals, rewriter,
                                  tensorTy);
    rewriter.replaceOp(op, packed);
    return success();
  }
};

} // anonymous namespace

void mlir::triton::Metal::populateLoadStoreOpToLLVMPatterns(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    PatternBenefit benefit) {
  patterns.add<LoadOpConversion, StoreOpConversion, AtomicRMWOpConversion,
               AtomicCASOpConversion>(typeConverter, benefit);
}
