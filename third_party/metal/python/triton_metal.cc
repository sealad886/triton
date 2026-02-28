#include <pybind11/pybind11.h>
#include "Dialect/MetalGPU/IR/Dialect.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Dialect/GPU/GPUToLLVMIRTranslation.h"
#include "passes.h"
#include "TritonMetalGPUToLLVM/Passes.h"
#include "TritonMetalGPUTransforms/Passes.h"

namespace py = pybind11;

extern "C" PyObject *PyInit_metal_utils(void);

void init_triton_metal_passes_ttgpuir(py::module &&m) {
  m.def("add_to_llvmir", [](mlir::PassManager &pm) {
    pm.addPass(mlir::triton::createConvertTritonMetalGPUToLLVM());
  });
  ADD_PASS_OPTION_WRAPPER_2("add_accelerate_matmul",
                            mlir::createTritonMetalGPUAccelerateMatmul,
                            std::string, int);
}

void init_triton_metal(py::module &&m) {
  py::module metal_utils =
      py::reinterpret_borrow<py::module>(PyInit_metal_utils());
  m.add_object("metal_utils", metal_utils);

  auto passes = m.def_submodule("passes");
  init_triton_metal_passes_ttgpuir(passes.def_submodule("ttgpuir"));

  m.def("load_dialects", [](mlir::MLIRContext &context) {
    mlir::DialectRegistry registry;
    registry.insert<mlir::triton::metalgpu::MetalGPUDialect, mlir::gpu::GPUDialect>();
    mlir::registerGPUDialectTranslation(registry);
    context.appendDialectRegistry(registry);
    context.loadAllAvailableDialects();
  });
}
