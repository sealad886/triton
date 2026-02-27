#ifndef METALGPU_DIALECT
#define METALGPU_DIALECT

include "mlir/IR/OpBase.td"

def MetalGPU_Dialect : Dialect {
  let name = "mtlg";
  let cppNamespace = "::mlir::triton::metalgpu";
  let summary = "Metal GPU dialect for Apple GPU targets";
  let description = [{
    Dialect for Metal-specific operations and attributes used in the Apple Metal
    backend for Triton GPU kernels targeting Apple Silicon GPUs.
  }];
  let dependentDialects = [ "mlir::LLVM::LLVMDialect" ];
}

#endif
