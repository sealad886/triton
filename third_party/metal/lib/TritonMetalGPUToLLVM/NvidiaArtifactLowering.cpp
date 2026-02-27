#include "NvidiaArtifactLowering.h"
#include "LoadStoreOpToLLVM.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/IR/SymbolTable.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

using namespace mlir;
using namespace mlir::triton;
using mlir::triton::Metal::getOrInsertExternFunc;
using mlir::triton::Metal::mangleTypeForSymbol;

namespace {

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

} // anonymous namespace

void mlir::triton::Metal::lowerNvidiaArtifactsToMetal(ModuleOp mod) {
  stripNVVMAttrs(mod);
  rewriteNVVMSRegs(mod);
  rewriteWarpId(mod);
  rewriteGlobalInlineAsm(mod);
}

void mlir::triton::Metal::ensureSharedMemorySymbol(ModuleOp mod, unsigned addrSpace) {
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
