add_triton_library(TritonMetalGPUToLLVM
  TargetInfo.cpp
  TritonGPUToLLVM.cpp
  Utility.cpp

  DEPENDS
  TritonMetalGPUConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRGPUDialect
  MLIRGPUTransforms
  TritonAnalysis
  TritonIR
  TritonGPUIR
  TritonGPUTransforms
  TritonGPUToLLVM
)
