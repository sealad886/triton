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

  // Support only f32 x f32 -> f32 for now; fall back to FMA otherwise.
  if (!elemTy.isF32() || !aType.getElementType().isF32() ||
      !bType.getElementType().isF32())
    return convertFMADot(op, adaptor, typeConverter, rewriter);

  Type llvmElemTy = typeConverter->convertType(elemTy);
  if (!llvmElemTy)
    return failure();

  auto resultShape = resultTy.getShape();
  int rank = resultShape.size();
  bool hasBatch = (rank == 3);

  if (hasBatch)
    return convertFMADot(op, adaptor, typeConverter, rewriter);

  int64_t M = resultShape[0];
  int64_t N = resultShape[1];
  int64_t K = aType.getShape().back();

  auto instrShape = dEnc.getInstrShape();
  unsigned mDim = instrShape[0]; // 8
  unsigned nDim = instrShape[1]; // 8
  unsigned kDim = instrShape[2]; // 8

  auto warpsPerCTA = dEnc.getWarpsPerCTA();
  unsigned warpsM = warpsPerCTA[0];
  unsigned warpsN = warpsPerCTA[1];

  unsigned numRepM = M / (mDim * warpsM);
  unsigned numRepN = N / (nDim * warpsN);
  unsigned numRepK = K / kDim;

  auto aElems = unpackLLElements(loc, adaptor.getA(), rewriter);
  auto bElems = unpackLLElements(loc, adaptor.getB(), rewriter);
  auto cElems = unpackLLElements(loc, adaptor.getC(), rewriter);

  assert(aElems.size() == 2 * numRepK * numRepM &&
         "unexpected A element count");
  assert(bElems.size() == 2 * numRepK * numRepN &&
         "unexpected B element count");
  assert(cElems.size() == 2 * numRepN * numRepM &&
         "unexpected C element count");

  auto b = TritonLLVMOpBuilder(loc, rewriter);
  auto i32Ty = rewriter.getI32Type();
  auto i8Ty = rewriter.getI8Type();
  auto ptrTy = LLVM::LLVMPointerType::get(ctx, 3); // threadgroup
  auto voidTy = LLVM::LLVMVoidType::get(ctx);
  auto matTy = VectorType::get({8}, llvmElemTy);

  // Scratch memory base: AllocateSharedMemory sets allocation.offset.
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

  // Per-warp scratch: two 8x8 tiles.
  unsigned elemBytes = elemTy.getIntOrFloatBitWidth() / 8;
  unsigned tileBytes = 64 * elemBytes;
  unsigned scratchPerWarp = 2 * tileBytes;

  Value warpByteOff = b.mul(warpId, b.i32_val(scratchPerWarp));
  Value warpScratch = b.gep(ptrTy, i8Ty, smemBase, warpByteOff);
  Value scratchA = warpScratch;
  Value tileBytesVal = b.i32_val(tileBytes);
  Value scratchB = b.gep(ptrTy, i8Ty, warpScratch, tileBytesVal);

  // Per-thread offsets inside an 8x8 tile (row-major, stride = 8).
  //
  // The linear layout for MetalSimdgroupEncoding operands is:
  //   identity1D(2, register, dimCol) *    // reg bit 0 → col basis 1
  //   identity1D(8, lane, dimRow)     *    // lane bits 0-2 → row bases 1,2,4
  //   identity1D(4, lane, dimCol)          // lane bits 3-4 → col bases 2,4
  //                                        //   (shifted by register's col range)
  //
  // So each thread owns two CONSECUTIVE column values:
  //   row      = lane & 7
  //   col_base = ((lane >> 3) & 3) * 2    (0, 2, 4, or 6)
  //   reg 0 → col_base + 0
  //   reg 1 → col_base + 1
  Value row = b.and_(laneId, b.i32_val(7));
  Value colPartial = b.and_(b.lshr(laneId, b.i32_val(3)), b.i32_val(3));
  Value col = b.mul(colPartial, b.i32_val(2));
  Value elemOff0 = b.add(b.mul(row, b.i32_val(8)), col);
  Value elemOff1 = b.add(elemOff0, b.i32_val(1));

  Value byteOff0 = b.mul(elemOff0, b.i32_val(elemBytes));
  Value byteOff1 = b.mul(elemOff1, b.i32_val(elemBytes));

  Value ptrA0 = b.gep(ptrTy, i8Ty, scratchA, byteOff0);
  Value ptrA1 = b.gep(ptrTy, i8Ty, scratchA, byteOff1);
  Value ptrB0 = b.gep(ptrTy, i8Ty, scratchB, byteOff0);
  Value ptrB1 = b.gep(ptrTy, i8Ty, scratchB, byteOff1);

  // Declare simdgroup intrinsics with type suffix for LLVM IR uniqueness.
  std::string ts = mangleTypeForSymbol(llvmElemTy);

  auto loadTgFn = getOrInsertExternFunc(
      mod, rewriter, "__metal_simdgroup_load_tg_" + ts, matTy,
      {ptrTy, i32Ty});
  auto storeTgFn = getOrInsertExternFunc(
      mod, rewriter, "__metal_simdgroup_store_tg_" + ts, voidTy,
      {matTy, ptrTy, i32Ty});
  auto mmaFn = getOrInsertExternFunc(
      mod, rewriter, "__metal_simdgroup_multiply_accumulate_" + ts, matTy,
      {matTy, matTy, matTy});
  auto barrierFn = getOrInsertExternFunc(
      mod, rewriter, "__metal_simdgroup_barrier", voidTy, {i32Ty});

  Value stride = b.i32_val(8);
  Value barrierFlags = b.i32_val(1); // mem_threadgroup

  SmallVector<Value> resultElems(cElems.size());

  // Ensure prior shared-memory consumers (e.g. ConvertLayoutOps that
  // populate this DotOp's operands) have finished reading before we
  // overwrite the scratch region that may alias their allocations.
  b.call(barrierFn, ValueRange{barrierFlags});

  for (unsigned mRep = 0; mRep < numRepM; ++mRep) {
    for (unsigned nRep = 0; nRep < numRepN; ++nRep) {
      unsigned cBase = 2 * (nRep + numRepN * mRep);

      // Load accumulator C into a simdgroup matrix via scratch.
      b.store(cElems[cBase + 0], ptrA0);
      b.store(cElems[cBase + 1], ptrA1);
      b.call(barrierFn, ValueRange{barrierFlags});
      Value cMat =
          b.call(loadTgFn, ValueRange{scratchA, stride})->getResult(0);

      // K-reduction: load each A and B tile, accumulate.
      for (unsigned kRep = 0; kRep < numRepK; ++kRep) {
        unsigned aBase = 2 * (kRep + numRepK * mRep);
        unsigned bBase = 2 * (kRep + numRepK * nRep);

        b.store(aElems[aBase + 0], ptrA0);
        b.store(aElems[aBase + 1], ptrA1);
        b.store(bElems[bBase + 0], ptrB0);
        b.store(bElems[bBase + 1], ptrB1);

        b.call(barrierFn, ValueRange{barrierFlags});

        Value aMat =
            b.call(loadTgFn, ValueRange{scratchA, stride})->getResult(0);
        Value bMat =
            b.call(loadTgFn, ValueRange{scratchB, stride})->getResult(0);

        cMat = b.call(mmaFn, ValueRange{aMat, bMat, cMat})->getResult(0);
      }

      // Write result matrix to scratch and read back per-thread values.
      b.call(storeTgFn, ValueRange{cMat, scratchA, stride});
      b.call(barrierFn, ValueRange{barrierFlags});

      resultElems[cBase + 0] = b.load(llvmElemTy, ptrA0);
      resultElems[cBase + 1] = b.load(llvmElemTy, ptrA1);
    }
  }

  Value result =
      packLLElements(loc, typeConverter, resultElems, rewriter, resultTy);
  rewriter.replaceOp(op, result);
  return success();
}

} // namespace mlir::triton::Metal
