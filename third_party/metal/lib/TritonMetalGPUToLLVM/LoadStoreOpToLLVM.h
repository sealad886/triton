#ifndef TRITON_METAL_LOADSTOREOPTOLLVM_H
#define TRITON_METAL_LOADSTOREOPTOLLVM_H

#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir::triton::Metal {

void populateLoadStoreOpToLLVMPatterns(LLVMTypeConverter &typeConverter,
                                       RewritePatternSet &patterns,
                                       PatternBenefit benefit);

LLVM::LLVMFuncOp getOrInsertExternFunc(ModuleOp mod, OpBuilder &builder,
                                        StringRef baseName, Type retTy,
                                        ArrayRef<Type> argTys,
                                        StringRef suffix = "");

std::string mangleTypeForSymbol(Type ty);

} // namespace mlir::triton::Metal

#endif // TRITON_METAL_LOADSTOREOPTOLLVM_H
