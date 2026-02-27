add_triton_ut(
  NAME TestMetalTargetInfo
  SRCS TargetInfoTest.cpp
  LIBS
    TritonGPUToLLVM
    TritonMetalGPUToLLVM
    MLIRUBToLLVM
)

target_include_directories(TestMetalTargetInfo PRIVATE
  ${CMAKE_SOURCE_DIR}/third_party/metal/lib/TritonMetalGPUToLLVM)

add_triton_ut(
  NAME TestMetalNvidiaArtifactLowering
  SRCS NvidiaArtifactLoweringTest.cpp
  LIBS
    TritonGPUToLLVM
    TritonMetalGPUToLLVM
    MLIRIR
    MLIRParser
    MLIRUBToLLVM
    TritonIR
    TritonGPUIR
)

target_include_directories(TestMetalNvidiaArtifactLowering PRIVATE
  ${CMAKE_SOURCE_DIR}/third_party/metal/lib/TritonMetalGPUToLLVM)
