add_triton_library(TritonMetalGPUToLLVM
  BarrierOpToLLVM.cpp
  DotOpToLLVM.cpp
  DotOpToLLVM/MetalSimdgroupDot.cpp
  FpToFpOpToLLVM.cpp
  LoadStoreOpToLLVM.cpp
  MetalGPUOpsToLLVM.cpp
  NvidiaArtifactLowering.cpp
  SPMDOpToLLVM.cpp
  TargetInfo.cpp
  TritonGPUToLLVM.cpp
  Utility.cpp

  DEPENDS
  TritonMetalGPUConversionPassIncGen

  LINK_LIBS PUBLIC
  MetalGPUIR
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
