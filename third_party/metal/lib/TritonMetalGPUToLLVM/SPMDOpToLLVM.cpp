#include "SPMDOpToLLVM.h"
#include "LoadStoreOpToLLVM.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/SymbolTable.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

using namespace mlir;
using namespace mlir::triton;

namespace {

static StringRef getNumProgramsFuncName(ProgramIDDim axis) {
  switch (axis) {
  case ProgramIDDim::X:
    return "__metal_get_threadgroups_per_grid_x";
  case ProgramIDDim::Y:
    return "__metal_get_threadgroups_per_grid_y";
  case ProgramIDDim::Z:
    return "__metal_get_threadgroups_per_grid_z";
  }
  llvm_unreachable("invalid axis");
}

struct GetNumProgramsOpConversion
    : public ConvertOpToLLVMPattern<triton::GetNumProgramsOp> {
  using ConvertOpToLLVMPattern<
      triton::GetNumProgramsOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::GetNumProgramsOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto moduleOp = op->getParentOfType<ModuleOp>();
    StringRef funcName = getNumProgramsFuncName(op.getAxis());

    auto fn = Metal::getOrInsertExternFunc(moduleOp, rewriter, funcName,
                                           i32_ty, {});
    Value result =
        LLVM::CallOp::create(rewriter, loc, fn, ValueRange{})->getResult(0);
    rewriter.replaceOp(op, result);
    return success();
  }
};

} // namespace

void mlir::triton::Metal::populateSPMDOpToLLVMPattern(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    PatternBenefit benefit) {
  patterns.add<GetNumProgramsOpConversion>(typeConverter, benefit);
}
