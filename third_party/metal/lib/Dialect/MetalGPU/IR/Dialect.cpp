#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"

#include "Dialect/MetalGPU/IR/Dialect.h"
#include "Dialect/MetalGPU/IR/Dialect.cpp.inc"

using namespace mlir;
using namespace mlir::triton::metalgpu;

void mlir::triton::metalgpu::MetalGPUDialect::initialize() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "Dialect/MetalGPU/IR/MetalGPUAttrDefs.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "Dialect/MetalGPU/IR/Ops.cpp.inc"
      >();
}

#define GET_OP_CLASSES
#include "Dialect/MetalGPU/IR/Ops.cpp.inc"
