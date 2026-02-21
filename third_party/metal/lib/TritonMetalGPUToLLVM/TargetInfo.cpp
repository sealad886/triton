#include "TargetInfo.h"
#include "Utility.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

using namespace mlir;

namespace {

LLVM::LLVMFuncOp getPrintfDeclaration(RewriterBase &rewriter) {
  auto moduleOp =
      rewriter.getBlock()->getParent()->getParentOfType<ModuleOp>();
  StringRef funcName("__metal_printf");
  Operation *funcOp = moduleOp.lookupSymbol(funcName);
  if (funcOp)
    return cast<LLVM::LLVMFuncOp>(*funcOp);

  auto *context = rewriter.getContext();
  auto funcType = LLVM::LLVMFunctionType::get(
      LLVM::LLVMVoidType::get(context),
      {LLVM::LLVMPointerType::get(context)},
      /*isVarArg=*/true);

  RewriterBase::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToStart(moduleOp.getBody());
  return LLVM::LLVMFuncOp::create(rewriter, UnknownLoc::get(context), funcName,
                                  funcType);
}

LLVM::LLVMFuncOp getBallotDeclaration(RewriterBase &rewriter, Type resultTy) {
  auto moduleOp =
      rewriter.getBlock()->getParent()->getParentOfType<ModuleOp>();
  StringRef funcName("__metal_simd_ballot");
  Operation *funcOp = moduleOp.lookupSymbol(funcName);
  if (funcOp)
    return cast<LLVM::LLVMFuncOp>(*funcOp);

  auto *context = rewriter.getContext();
  auto funcType = LLVM::LLVMFunctionType::get(
      resultTy, {IntegerType::get(context, 1)});

  RewriterBase::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToStart(moduleOp.getBody());
  auto func = LLVM::LLVMFuncOp::create(rewriter, UnknownLoc::get(context),
                                       funcName, funcType);
  func.setVisibility(SymbolTable::Visibility::Private);
  return func;
}

LLVM::LLVMFuncOp getBarrierDeclaration(RewriterBase &rewriter,
                                       StringRef funcName) {
  auto moduleOp =
      rewriter.getBlock()->getParent()->getParentOfType<ModuleOp>();
  Operation *funcOp = moduleOp.lookupSymbol(funcName);
  if (funcOp)
    return cast<LLVM::LLVMFuncOp>(*funcOp);

  auto *context = rewriter.getContext();
  auto funcType = LLVM::LLVMFunctionType::get(
      LLVM::LLVMVoidType::get(context), {IntegerType::get(context, 32)});

  RewriterBase::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToStart(moduleOp.getBody());
  auto func = LLVM::LLVMFuncOp::create(rewriter, UnknownLoc::get(context),
                                       funcName, funcType);
  func.setVisibility(SymbolTable::Visibility::Private);
  return func;
}

} // namespace

namespace mlir {
namespace triton {
namespace Metal {

Value TargetInfo::getClusterCTAId(RewriterBase &rewriter,
                                  Location loc) const {
  // Metal does not support clusters; always return 0
  return LLVM::ConstantOp::create(rewriter, loc, rewriter.getI32Type(),
                                  rewriter.getI32IntegerAttr(0));
}

Value TargetInfo::ballot(RewriterBase &rewriter, Location loc, Type type,
                         Value cmp) const {
  auto func = getBallotDeclaration(rewriter, type);
  return LLVM::CallOp::create(rewriter, loc, func, ValueRange{cmp})
      ->getResult(0);
}

void TargetInfo::barrier(Location loc, RewriterBase &rewriter,
                         triton::gpu::AddrSpace targets) const {
  (void)targets;
  auto func = getBarrierDeclaration(rewriter, "__metal_simdgroup_barrier");
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  // mem_flags::mem_none = 0
  LLVM::CallOp::create(rewriter, loc, func, ValueRange{b.i32_val(0)});
}

void TargetInfo::clusterBarrier(Location loc, RewriterBase &rewriter) const {
  // Metal does not support clusters; fall back to regular barrier
  barrier(loc, rewriter, triton::gpu::AddrSpace::Local);
}

void TargetInfo::warpSync(Location loc, RewriterBase &rewriter) const {
  // Metal uses simdgroup_barrier for warp-level sync
  auto func = getBarrierDeclaration(rewriter, "__metal_simdgroup_barrier");
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  // mem_flags::mem_none = 0
  LLVM::CallOp::create(rewriter, loc, func, ValueRange{b.i32_val(0)});
}

void TargetInfo::storeDShared(RewriterBase &rewriter, Location loc, Value ptr,
                              std::optional<Value> ctaId, Value val,
                              Value pred) const {
  if (ctaId.has_value()) {
    llvm::report_fatal_error(
        "Metal does not support cross-CTA shared memory transfers");
  }
  if (pred) {
    Value oldVal = LLVM::LoadOp::create(rewriter, loc, val.getType(), ptr);
    Value merged = LLVM::SelectOp::create(rewriter, loc, pred, val, oldVal);
    LLVM::StoreOp::create(rewriter, loc, merged, ptr);
  } else {
    LLVM::StoreOp::create(rewriter, loc, val, ptr);
  }
}

Value TargetInfo::loadDShared(RewriterBase &rewriter, Location loc, Value ptr,
                              std::optional<Value> ctaId, Type elemTy,
                              Value pred, Operation *localLoadOp) const {
  if (ctaId.has_value()) {
    llvm::report_fatal_error(
        "Metal does not support cross-CTA shared memory transfers");
  }
  Value loaded = LLVM::LoadOp::create(rewriter, loc, elemTy, ptr);
  if (!pred)
    return loaded;

  Value undef = LLVM::UndefOp::create(rewriter, loc, elemTy);
  return LLVM::SelectOp::create(rewriter, loc, pred, loaded, undef);
}

Value TargetInfo::shuffleXor(RewriterBase &rewriter, Location loc, Value val,
                             int i) const {
  return LLVM::Metal::shuffleXor(loc, rewriter, val, i);
}

Value TargetInfo::shuffleUp(RewriterBase &rewriter, Location loc, Value val,
                            int i) const {
  return LLVM::Metal::shuffleUp(loc, rewriter, val, i);
}

Value TargetInfo::shuffleIdx(RewriterBase &rewriter, Location loc, Value val,
                             int i) const {
  return LLVM::Metal::shuffleIdx(loc, rewriter, val, i);
}

Value TargetInfo::shuffleIdx(RewriterBase &rewriter, Location loc, Value val,
                             Value i) const {
  return LLVM::Metal::shuffleIdx(loc, rewriter, val, i);
}

Value TargetInfo::permute(RewriterBase &rewriter, Location loc, Value a,
                          Value b, Value selector) const {
  // Metal doesn't have a direct byte permute instruction like NVIDIA's prmt.
  // Fall back to a software implementation using shifts and masks.
  auto bld = TritonLLVMOpBuilder(loc, rewriter);

  Value a64 = bld.zext(i64_ty, a);
  Value b64 = bld.zext(i64_ty, b);
  Value combined = bld.or_(bld.shl(b64, bld.int_val(64, 32)), a64);

  Value result = bld.i32_val(0);
  for (int byteIdx = 0; byteIdx < 4; byteIdx++) {
    Value byteSelector = bld.and_(
        bld.lshr(selector, bld.i32_val(byteIdx * 4)), bld.i32_val(0x7));
    Value byteSelExt = bld.zext(i64_ty, byteSelector);
    Value shift = bld.mul(byteSelExt, bld.int_val(64, 8));
    Value byte = bld.trunc(
        i32_ty, bld.and_(bld.lshr(combined, shift), bld.int_val(64, 0xFF)));
    result = bld.or_(result, bld.shl(byte, bld.i32_val(byteIdx * 8)));
  }
  return result;
}

Value TargetInfo::programId(RewriterBase &rewriter, Location loc,
                            ModuleOp moduleOp, ProgramIDDim axis) const {
  return LLVM::Metal::llGetPid(loc, rewriter, moduleOp, axis);
}

bool TargetInfo::warpReduce(RewriterBase &rewriter, Location loc,
                            SmallVector<Value> &acc, triton::ReduceOp op,
                            unsigned reduceLaneIdMask) const {
  // Metal doesn't have hardware warp-reduce instructions like NVIDIA's redux.
  // Return false to fall back to the generic shuffle-based reduction.
  return false;
}

std::string TargetInfo::getMulhiFuncName(Type resultElementTy) const {
  if (resultElementTy.isInteger(32))
    return "__metal_mulhi_u32";
  return "__metal_mulhi_u64";
}

void TargetInfo::printf(RewriterBase &rewriter, Value formatStrStart,
                        int formatStrByteCount, ValueRange args,
                        ArrayRef<bool> isSigned) const {
  auto loc = UnknownLoc::get(rewriter.getContext());
  auto func = getPrintfDeclaration(rewriter);
  SmallVector<Value> operands;
  operands.push_back(formatStrStart);
  for (Value arg : args) {
    operands.push_back(arg);
  }
  LLVM::CallOp::create(rewriter, loc, func, operands);
}

void TargetInfo::printf(RewriterBase &rewriter, StringRef msg, ValueRange args,
                        ArrayRef<bool> isSigned) const {
  assert(!msg.empty() && "printf with empty string not supported");
  llvm::SmallString<64> msgNewline(msg);
  msgNewline.push_back('\n');
  msgNewline.push_back('\0');
  Value msgValue =
      LLVM::addStringToModule(UnknownLoc::get(rewriter.getContext()), rewriter,
                              "printfFormat_", msgNewline);
  printf(rewriter, msgValue, msgNewline.size_in_bytes(), args, isSigned);
}

void TargetInfo::assertFail(RewriterBase &rewriter, Location loc,
                            StringRef message, StringRef file, StringRef func,
                            int line) const {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  llvm::SmallString<256> msgBuffer;
  llvm::Twine("device assertion failed: '" + message + "', in " + func +
              " at " + file + ":" + llvm::Twine(line) + "\n\0")
      .toStringRef(msgBuffer);
  Value msgValue =
      LLVM::addStringToModule(loc, rewriter, "assertMessage_", msgBuffer);
  printf(rewriter, msgValue, msgBuffer.size_in_bytes(), /*args=*/ValueRange(),
         /*isSigned=*/{});
  b.barrier(triton::gpu::AddrSpace::Local);
}

int TargetInfo::getSharedAddressSpace() const {
  // Metal threadgroup memory uses address space 3
  return 3;
}

int TargetInfo::getAddressSpace(Attribute addressSpace) const {
  if (isa<triton::gpu::SharedMemorySpaceAttr>(addressSpace)) {
    return 3;
  }
  llvm::report_fatal_error("Only support SharedMemorySpace for Metal backend");
  return 0;
}

bool TargetInfo::supportVectorizedAtomics() const { return false; }

} // namespace Metal
} // namespace triton
} // namespace mlir
