add_triton_library(TritonMetalGPUTransforms
  AccelerateMetalMatmul.cpp

  DEPENDS
  TritonMetalGPUTransformPassIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRTransformUtils
  TritonIR
  TritonGPUIR
  TritonGPUTransforms
)
