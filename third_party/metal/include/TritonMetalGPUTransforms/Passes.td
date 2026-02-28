#ifndef TRITONMETALGPU_PASSES
#define TRITONMETALGPU_PASSES

include "mlir/Pass/PassBase.td"

def TritonMetalGPUAccelerateMatmul : Pass<"tritonmetalgpu-accelerate-matmul", "mlir::ModuleOp"> {
  let summary = "Select Metal simdgroup encoding for dot operations";
  let description = [{
    Converts tt.dot operations with BlockedEncodingAttr to use
    MetalSimdgroupEncodingAttr when the shapes, dtypes, and GPU family
    qualify for simdgroup_matrix acceleration.
  }];
  let dependentDialects = [
    "mlir::triton::gpu::TritonGPUDialect"
  ];
  let options = [
    Option<"gpuFamily", "gpu-family", "std::string", /*default=*/"\"apple9\"",
           "Target GPU family (e.g., apple7, apple8, apple9)">,
    Option<"numWarps", "num-warps", "int", /*default=*/"4",
           "Number of simdgroups per threadgroup">
  ];
}

#endif
