#include "DotOpToLLVM/MetalSimdgroupDot.h"
#include "LoadStoreOpToLLVM.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "triton/Conversion/TritonGPUToLLVM/PatternTritonGPUOpToLLVM.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Attributes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

using namespace mlir;
using namespace mlir::triton;

namespace mlir::triton::Metal {

LogicalResult convertMetalSimdgroupDot(triton::DotOp op,
                                       triton::DotOp::Adaptor adaptor,
                                       const LLVMTypeConverter *typeConverter,
                                       ConversionPatternRewriter &rewriter) {
  Location loc = op.getLoc();
  MLIRContext *ctx = rewriter.getContext();
  auto mod = op->getParentOfType<ModuleOp>();

  auto resultTy = cast<RankedTensorType>(op.getResult().getType());
  auto dEnc =
      dyn_cast<gpu::MetalSimdgroupEncodingAttr>(resultTy.getEncoding());
  if (!dEnc)
    return rewriter.notifyMatchFailure(op, "not MetalSimdgroupEncoding");

  Type elemTy = resultTy.getElementType();
  auto aType = cast<RankedTensorType>(op.getA().getType());
  auto bType = cast<RankedTensorType>(op.getB().getType());
  Type aElemTy = aType.getElementType();
  Type bElemTy = bType.getElementType();

  // Accumulator must be f32.
  if (!elemTy.isF32())
    return convertFMADot(op, adaptor, typeConverter, rewriter);

  // Operands must match and be f32, f16, or bf16.
  if (aElemTy != bElemTy)
    return convertFMADot(op, adaptor, typeConverter, rewriter);
  if (!aElemTy.isF32() && !aElemTy.isF16() && !aElemTy.isBF16())
    return convertFMADot(op, adaptor, typeConverter, rewriter);

  bool isMixedPrecision = !aElemTy.isF32();

  Type llvmResTy = typeConverter->convertType(elemTy);
  Type llvmOpTy =
      isMixedPrecision ? typeConverter->convertType(aElemTy) : llvmResTy;
  if (!llvmResTy || !llvmOpTy)
    return failure();

  auto resultShape = resultTy.getShape();
  int rank = resultShape.size();
  bool hasBatch = (rank == 3);

  unsigned batchSize = hasBatch ? resultShape[0] : 1;
  int mIdx = hasBatch ? 1 : 0;
  int nIdx = hasBatch ? 2 : 1;
  int64_t M = resultShape[mIdx];
  int64_t N = resultShape[nIdx];
  int64_t K = aType.getShape()[hasBatch ? 2 : 1];

  auto instrShape = dEnc.getInstrShape();
  unsigned mDim = instrShape[0]; // 8
  unsigned nDim = instrShape[1]; // 8
  unsigned kDim = instrShape[2]; // 8

  auto warpsPerCTA = dEnc.getWarpsPerCTA();
  unsigned warpsM = hasBatch ? warpsPerCTA[1] : warpsPerCTA[0];
  unsigned warpsN = hasBatch ? warpsPerCTA[2] : warpsPerCTA[1];

  unsigned numRepM = M / (mDim * warpsM);
  unsigned numRepN = N / (nDim * warpsN);
  unsigned numRepK = K / kDim;

  auto aElems = unpackLLElements(loc, adaptor.getA(), rewriter);
  auto bElems = unpackLLElements(loc, adaptor.getB(), rewriter);
  auto cElems = unpackLLElements(loc, adaptor.getC(), rewriter);

  constexpr unsigned regsPerTile = 2;
  unsigned aElemsPerBatch = regsPerTile * numRepK * numRepM;
  unsigned bElemsPerBatch = regsPerTile * numRepK * numRepN;
  unsigned cElemsPerBatch = regsPerTile * numRepN * numRepM;

  assert(aElems.size() == aElemsPerBatch * batchSize &&
         "unexpected A element count");
  assert(bElems.size() == bElemsPerBatch * batchSize &&
         "unexpected B element count");
  assert(cElems.size() == cElemsPerBatch * batchSize &&
         "unexpected C element count");

  auto b = TritonLLVMOpBuilder(loc, rewriter);
  auto i32Ty = rewriter.getI32Type();
  auto i8Ty = rewriter.getI8Type();
  auto ptrTy = LLVM::LLVMPointerType::get(ctx, 3); // threadgroup
  auto voidTy = LLVM::LLVMVoidType::get(ctx);
  auto opMatTy = VectorType::get({8}, llvmOpTy);
  auto resMatTy = VectorType::get({8}, llvmResTy);

  // Scratch memory base.
  auto func = op->getParentOfType<FunctionOpInterface>();
  assert(op->hasAttr("allocation.offset") &&
         "DotOp lacks allocation.offset; was AllocateSharedMemory run?");
  size_t smemOffset =
      cast<IntegerAttr>(op->getAttr("allocation.offset"))
          .getValue()
          .getZExtValue();
  Value smemOffVal = b.i32_val(smemOffset);
  Value smemBase = b.gep(ptrTy, i8Ty,
                         LLVM::getStackPointer(rewriter, func),
                         smemOffVal);

  // Thread / warp / lane IDs.
  auto threadIdFn = getOrInsertExternFunc(
      mod, rewriter, "__metal_get_thread_position_in_threadgroup_x", i32Ty, {});
  Value threadId =
      LLVM::CallOp::create(rewriter, loc, threadIdFn, ValueRange{})
          ->getResult(0);
  Value warpId = b.udiv(threadId, b.i32_val(32));
  Value laneId = b.urem(threadId, b.i32_val(32));

  // Per-warp scratch: two tiles sized for the widest type (result type).
  unsigned resElemBytes = elemTy.getIntOrFloatBitWidth() / 8;
  unsigned tileBytes = 64 * resElemBytes;
  unsigned scratchPerWarp = 2 * tileBytes;

  Value warpByteOff = b.mul(warpId, b.i32_val(scratchPerWarp));
  Value warpScratch = b.gep(ptrTy, i8Ty, smemBase, warpByteOff);
  Value scratchA = warpScratch;
  Value tileBytesVal = b.i32_val(tileBytes);
  Value scratchB = b.gep(ptrTy, i8Ty, warpScratch, tileBytesVal);

  // Per-thread element offsets within an 8×8 tile (row-major, stride = 8).
  Value row = b.and_(laneId, b.i32_val(7));
  Value colPartial = b.and_(b.lshr(laneId, b.i32_val(3)), b.i32_val(3));
  Value col = b.mul(colPartial, b.i32_val(2));
  Value elemOff0 = b.add(b.mul(row, b.i32_val(8)), col);
  Value elemOff1 = b.add(elemOff0, b.i32_val(1));

  // Byte offsets for operand stores (may be f16/bf16 = 2 bytes).
  unsigned opElemBytes = aElemTy.getIntOrFloatBitWidth() / 8;
  Value opByteOff0 = b.mul(elemOff0, b.i32_val(opElemBytes));
  Value opByteOff1 = b.mul(elemOff1, b.i32_val(opElemBytes));
  Value opPtrA0 = b.gep(ptrTy, i8Ty, scratchA, opByteOff0);
  Value opPtrA1 = b.gep(ptrTy, i8Ty, scratchA, opByteOff1);
  Value opPtrB0 = b.gep(ptrTy, i8Ty, scratchB, opByteOff0);
  Value opPtrB1 = b.gep(ptrTy, i8Ty, scratchB, opByteOff1);

  // Byte offsets for result (accumulator) stores (always f32).
  Value resByteOff0 = b.mul(elemOff0, b.i32_val(resElemBytes));
  Value resByteOff1 = b.mul(elemOff1, b.i32_val(resElemBytes));
  Value resPtrA0 = b.gep(ptrTy, i8Ty, scratchA, resByteOff0);
  Value resPtrA1 = b.gep(ptrTy, i8Ty, scratchA, resByteOff1);

  // Declare simdgroup intrinsics.
  std::string opTs = mangleTypeForSymbol(llvmOpTy);
  std::string resTs = mangleTypeForSymbol(llvmResTy);

  auto loadTgOpFn = getOrInsertExternFunc(
      mod, rewriter, "__metal_simdgroup_load_tg_" + opTs, opMatTy,
      {ptrTy, i32Ty});
  auto loadTgResFn = getOrInsertExternFunc(
      mod, rewriter, "__metal_simdgroup_load_tg_" + resTs, resMatTy,
      {ptrTy, i32Ty});
  auto storeTgResFn = getOrInsertExternFunc(
      mod, rewriter, "__metal_simdgroup_store_tg_" + resTs, voidTy,
      {resMatTy, ptrTy, i32Ty});

  std::string mmaSuffix = isMixedPrecision ? resTs + "_" + opTs : resTs;
  auto mmaFn = getOrInsertExternFunc(
      mod, rewriter, "__metal_simdgroup_multiply_accumulate_" + mmaSuffix,
      resMatTy, {opMatTy, opMatTy, resMatTy});

  auto barrierFn = getOrInsertExternFunc(
      mod, rewriter, "__metal_simdgroup_barrier", voidTy, {i32Ty});

  Value stride = b.i32_val(8);
  Value barrierFlags = b.i32_val(1); // mem_threadgroup

  SmallVector<Value> resultElems(cElems.size());

  // Pre-barrier to protect against aliased shared memory.
  b.call(barrierFn, ValueRange{barrierFlags});

  for (unsigned bRep = 0; bRep < batchSize; ++bRep) {
    for (unsigned mRep = 0; mRep < numRepM; ++mRep) {
      for (unsigned nRep = 0; nRep < numRepN; ++nRep) {
        unsigned cBase =
            regsPerTile * (nRep + numRepN * mRep) + bRep * cElemsPerBatch;

        // Load accumulator C via scratch (always f32).
        b.store(cElems[cBase + 0], resPtrA0);
        b.store(cElems[cBase + 1], resPtrA1);
        b.call(barrierFn, ValueRange{barrierFlags});
        Value cMat =
            b.call(loadTgResFn, ValueRange{scratchA, stride})->getResult(0);

        for (unsigned kRep = 0; kRep < numRepK; ++kRep) {
          unsigned aBase =
              regsPerTile * (kRep + numRepK * mRep) + bRep * aElemsPerBatch;
          unsigned bBase =
              regsPerTile * (kRep + numRepK * nRep) + bRep * bElemsPerBatch;

          // Store A and B operands via scratch (may be f16/bf16).
          b.store(aElems[aBase + 0], opPtrA0);
          b.store(aElems[aBase + 1], opPtrA1);
          b.store(bElems[bBase + 0], opPtrB0);
          b.store(bElems[bBase + 1], opPtrB1);
          b.call(barrierFn, ValueRange{barrierFlags});

          Value aMat =
              b.call(loadTgOpFn, ValueRange{scratchA, stride})->getResult(0);
          Value bMat =
              b.call(loadTgOpFn, ValueRange{scratchB, stride})->getResult(0);
          cMat =
              b.call(mmaFn, ValueRange{aMat, bMat, cMat})->getResult(0);
        }

        // Write result and read back per-thread values (always f32).
        b.call(storeTgResFn, ValueRange{cMat, scratchA, stride});
        b.call(barrierFn, ValueRange{barrierFlags});

        resultElems[cBase + 0] = b.load(llvmResTy, resPtrA0);
        resultElems[cBase + 1] = b.load(llvmResTy, resPtrA1);
      }
    }
  }

  Value result =
      packLLElements(loc, typeConverter, resultElems, rewriter, resultTy);
  rewriter.replaceOp(op, result);
  return success();
}

} // namespace mlir::triton::Metal
