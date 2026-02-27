add_triton_library(MetalGPUIR
  Dialect.cpp

  DEPENDS
  MetalGPUTableGen
  MetalGPUAttrDefsIncGen

  LINK_LIBS PUBLIC
  MLIRLLVMDialect
)
