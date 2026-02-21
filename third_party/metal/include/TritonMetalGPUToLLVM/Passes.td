#ifndef TRITONMETALGPU_CONVERSION_PASSES
#define TRITONMETALGPU_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def ConvertTritonMetalGPUToLLVM : Pass<"convert-triton-metal-gpu-to-llvm", "mlir::ModuleOp"> {
    let summary = "Convert TritonGPU to LLVM for Metal backend";
    let description = [{

    }];

    let dependentDialects = ["mlir::arith::ArithDialect",
                             "mlir::math::MathDialect",
                             "mlir::gpu::GPUDialect",
                             "mlir::scf::SCFDialect",
                             "mlir::LLVM::LLVMDialect",
                             "mlir::triton::TritonDialect",
                             "mlir::triton::gpu::TritonGPUDialect"];
}

#endif // TRITONMETALGPU_CONVERSION_PASSES
