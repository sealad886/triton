# CMake configuration for Triton
cmake_minimum_required(VERSION 3.18)

# Set proper compilers before project() call
set(CMAKE_C_COMPILER "/opt/homebrew/opt/llvm/bin/clang")
set(CMAKE_CXX_COMPILER "/opt/homebrew/opt/llvm/bin/clang++")

# Set LLVM paths from detected installation
set(LLVM_LIBRARY_DIR "/opt/homebrew/opt/llvm/lib")
set(MLIR_DIR "${LLVM_LIBRARY_DIR}/cmake/mlir")
set(LLVM_DIR "${LLVM_LIBRARY_DIR}/cmake/llvm")

# Debug logging for configuration
message(STATUS "LLVM_LIBRARY_DIR = ${LLVM_LIBRARY_DIR}")
message(STATUS "MLIR_DIR = ${MLIR_DIR}")
message(STATUS "LLVM_DIR = ${LLVM_DIR}")
message(STATUS "C Compiler: ${CMAKE_C_COMPILER}")
message(STATUS "CXX Compiler: ${CMAKE_CXX_COMPILER}")

if(POLICY CMP0116)
# Introduced in cmake 3.20
# https://cmake.org/cmake/help/latest/policy/CMP0116.html
  cmake_policy(SET CMP0116 NEW)
endif()

include(ExternalProject)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_INCLUDE_CURRENT_DIR ON)

project(triton CXX C)
include(CTest)

# Apple platform detection and framework setup
if(APPLE)
  message(STATUS "Configuring for Apple platform")
  message(STATUS "System Processor: ${CMAKE_SYSTEM_PROCESSOR}")
  message(STATUS "OSX Architectures: ${CMAKE_OSX_ARCHITECTURES}")

  execute_process(
    COMMAND xcrun --sdk macosx --show-sdk-path
    OUTPUT_VARIABLE MACOS_SDK_PATH
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )
  message(STATUS "macOS SDK Path: ${MACOS_SDK_PATH}")

  find_library(METAL_FRAMEWORK Metal REQUIRED)
  find_library(FOUNDATION_FRAMEWORK Foundation REQUIRED)
  find_library(MPS_FRAMEWORK MetalPerformanceShaders REQUIRED)

  if(NOT METAL_FRAMEWORK OR NOT FOUNDATION_FRAMEWORK OR NOT MPS_FRAMEWORK)
    message(FATAL_ERROR "Required Apple frameworks not found")
  endif()

  message(STATUS "Metal Framework: ${METAL_FRAMEWORK}")
  message(STATUS "Foundation Framework: ${FOUNDATION_FRAMEWORK}")
  message(STATUS "MPS Framework: ${MPS_FRAMEWORK}")
endif()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# Options
option(TRITON_BUILD_TUTORIALS "Build C++ Triton tutorials" ON)
option(TRITON_BUILD_PYTHON_MODULE "Build Python Triton bindings" OFF)
option(TRITON_BUILD_PROTON "Build the Triton Proton profiler" ON)
option(TRITON_BUILD_UT "Build C++ Triton Unit Tests" ON)
option(TRITON_BUILD_WITH_CCACHE "Build with ccache (if available)" ON)
set(TRITON_CODEGEN_BACKENDS "" CACHE STRING "Enable different codegen backends")

# Configure Apple platform specifics
if(APPLE)
  # Set plugin directory for MPS backend
  set(TRITON_PLUGIN_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/python/triton/backends/apple")
  message(STATUS "Added MPS backend plugin directory: ${TRITON_PLUGIN_DIRS}")

  # Disable incompatible components on Apple platforms
  set(TRITON_BUILD_PROTON OFF CACHE BOOL "Disable proton on Apple platform" FORCE)
  message(STATUS "Disabled proton build on Apple platform")

  # Add MPS-specific compile definitions
  add_definitions(-DTRITON_ENABLE_MPS_BACKEND)

  # Make sure we're using the right compiler on Apple Silicon
  if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(arm|aarch)64")
    if(NOT CMAKE_C_COMPILER MATCHES "clang$")
      set(CMAKE_C_COMPILER "/opt/homebrew/opt/llvm/bin/clang" CACHE STRING "C compiler" FORCE)
    endif()
    if(NOT CMAKE_CXX_COMPILER MATCHES "clang\\+\\+$")
      set(CMAKE_CXX_COMPILER "/opt/homebrew/opt/llvm/bin/clang++" CACHE STRING "C++ compiler" FORCE)
    endif()
  endif()
endif()

if(TRITON_BUILD_WITH_CCACHE)
  find_program(CCACHE_PROGRAM ccache)
  if(CCACHE_PROGRAM)
    set(CMAKE_C_COMPILER_LAUNCHER "${CCACHE_PROGRAM}"
        CACHE STRING "C compiler launcher")
    set(CMAKE_CXX_COMPILER_LAUNCHER "${CCACHE_PROGRAM}"
        CACHE STRING "CXX compiler launcher")
  else()
    message(STATUS "Could not find ccache. Consider installing ccache to speed up compilation.")
  endif()
endif()

set(TRITON_PARALLEL_LINK_JOBS "" CACHE STRING
  "Define the maximum number of concurrent link jobs (Ninja only).")
if (TRITON_PARALLEL_LINK_JOBS)
    set_property(GLOBAL APPEND PROPERTY JOB_POOLS link_job_pool=${TRITON_PARALLEL_LINK_JOBS})
    set(CMAKE_JOB_POOL_LINK link_job_pool)
endif()

# Customized release build type with assertions: TritonRelBuildWithAsserts
if(NOT MSVC)
  set(CMAKE_C_FLAGS_TRITONRELBUILDWITHASSERTS "-O2 -g")
  set(CMAKE_CXX_FLAGS_TRITONRELBUILDWITHASSERTS "-O2 -g")
  set(CMAKE_C_FLAGS_TRITONBUILDWITHO1 "-O1")
  set(CMAKE_CXX_FLAGS_TRITONBUILDWITHO1 "-O1")
else()
  set(CMAKE_C_FLAGS_TRITONRELBUILDWITHASSERTS "/Zi /RTC1 /bigobj /Zc:preprocessor /permissive-")
  set(CMAKE_CXX_FLAGS_TRITONRELBUILDWITHASSERTS "/Zi /RTC1 /bigobj /Zc:preprocessor /permissive-")
  set(CMAKE_EXE_LINKER_FLAGS_TRITONRELBUILDWITHASSERTS "/debug:fastlink /INCREMENTAL")
  set(CMAKE_MODULE_LINKER_FLAGS_TRITONRELBUILDWITHASSERTS "/debug:fastlink /INCREMENTAL")
  set(CMAKE_SHARED_LINKER_FLAGS_TRITONRELBUILDWITHASSERTS "/debug:fastlink /INCREMENTAL")
endif()

# Default build type
if(NOT CMAKE_BUILD_TYPE)
  message(STATUS "Default build type: Release")
  set(CMAKE_BUILD_TYPE "Release")
endif()

if(NOT WIN32)
  find_library(TERMINFO_LIBRARY tinfo)
endif()

# Compiler flags
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
if(NOT MSVC)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D__STDC_FORMAT_MACROS  -fPIC -std=gnu++17")
else()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D__STDC_FORMAT_MACROS")
endif()

# LLVM Configuration
find_package(LLVM REQUIRED CONFIG)
find_package(MLIR REQUIRED CONFIG)

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

include(TableGen)
include(AddLLVM)
include(AddMLIR)

# Utility functions
function(add_triton_object name)
  cmake_parse_arguments(ARG "" "" "DEPENDS;LINK_LIBS" ${ARGN})
  add_library(${name} OBJECT)
  target_sources(${name}
    PRIVATE ${ARG_UNPARSED_ARGUMENTS}
    INTERFACE $<TARGET_OBJECTS:${name}>
  )
  if(ARG_DEPENDS)
    add_dependencies(${name} ${ARG_DEPENDS})
  endif()
  if(ARG_LINK_LIBS)
    target_link_libraries(${name} PUBLIC ${ARG_LINK_LIBS})
  endif()
endfunction()

# Global properties
set_property(GLOBAL PROPERTY TRITON_LIBS "")
function(add_triton_library name)
  set_property(GLOBAL APPEND PROPERTY TRITON_LIBS ${name})
  add_triton_object(${name} ${ARGN})
  llvm_update_compile_flags(${name})
endfunction()

set_property(GLOBAL PROPERTY TRITON_PLUGINS "")
function(add_triton_plugin name)
  set_property(GLOBAL APPEND PROPERTY TRITON_PLUGINS ${name})
  add_triton_object(${name} ${ARGN})
endfunction()

# Include directories
include_directories(.)
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_BINARY_DIR}/include)
include_directories(${PROJECT_SOURCE_DIR}/third_party)
include_directories(${PROJECT_BINARY_DIR}/third_party)

# Add subdirectories
add_subdirectory(include)
add_subdirectory(lib)

if (NOT WIN32 AND NOT APPLE AND NOT BSD)
  link_libraries(stdc++fs)
endif()

# Python module configuration
if(TRITON_BUILD_PYTHON_MODULE)
  message(STATUS "Adding Python module")
  set(PYTHON_SRC_PATH ${CMAKE_CURRENT_SOURCE_DIR}/python/src)
  include_directories(${PYTHON_SRC_PATH})

  find_package(Python3 REQUIRED COMPONENTS Development.Module Interpreter)
  find_package(pybind11 CONFIG REQUIRED HINTS "${Python3_SITELIB}")

  if(DEFINED TRITON_PLUGIN_DIRS)
    foreach(PLUGIN_DIR ${TRITON_PLUGIN_DIRS})
      cmake_path(APPEND PLUGIN_DIR "backend" "name.conf" OUTPUT_VARIABLE PLUGIN_NAME_PATH)
      file(READ ${PLUGIN_NAME_PATH} PLUGIN_NAME)
      string(STRIP ${PLUGIN_NAME} PLUGIN_NAME)
      list(APPEND TRITON_PLUGIN_NAMES ${PLUGIN_NAME})
      cmake_path(APPEND TRITON_BINARY_DIR "third_party" ${PLUGIN_NAME} OUTPUT_VARIABLE PLUGIN_DIR_BUILD_OUTPUT)
      message(STATUS "Building plugin '${PLUGIN_NAME}' from ${PLUGIN_DIR} with output ${PLUGIN_DIR_BUILD_OUTPUT}")
      add_subdirectory(${PLUGIN_DIR} ${PLUGIN_DIR_BUILD_OUTPUT})
    endforeach()
  endif()

  foreach(CODEGEN_BACKEND ${TRITON_CODEGEN_BACKENDS})
    add_subdirectory(third_party/${CODEGEN_BACKEND})
  endforeach()

  if(TRITON_BUILD_PROTON)
    add_subdirectory(third_party/proton)
  endif()
  list(APPEND TRITON_PLUGIN_NAMES "proton")
  add_subdirectory(third_party/proton/dialect)

  # Libraries
  get_property(triton_libs GLOBAL PROPERTY TRITON_LIBS)
  get_property(triton_plugins GLOBAL PROPERTY TRITON_PLUGINS)
  set(TRITON_LIBRARIES
    ${triton_libs}
    ${triton_plugins}

    MLIRAMDGPUDialect
    MLIRNVVMDialect
    MLIRNVVMToLLVMIRTranslation
    MLIRGPUToNVVMTransforms
    MLIRGPUToGPURuntimeTransforms
    MLIRGPUTransforms
    MLIRIR
    MLIRControlFlowToLLVM
    MLIRBytecodeWriter
    MLIRPass
    MLIRTransforms
    MLIRLLVMDialect
    MLIRSupport
    MLIRTargetLLVMIRExport
    MLIRMathToLLVM
    MLIRROCDLToLLVMIRTranslation
    MLIRGPUDialect
    MLIRSCFToControlFlow
    MLIRIndexToLLVM
    MLIRGPUToROCDLTransforms
    MLIRUBToLLVM

    LLVMPasses
    LLVMNVPTXCodeGen
    LLVMAMDGPUCodeGen
    LLVMAMDGPUAsmParser

    Python3::Module
    pybind11::headers
  )

  # Architecture-specific libraries
  if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64" OR
     CMAKE_SYSTEM_PROCESSOR MATCHES "arm64" OR
     CMAKE_OSX_ARCHITECTURES MATCHES "arm64")
    list(APPEND TRITON_LIBRARIES
      LLVMAArch64CodeGen
      LLVMAArch64AsmParser
    )
  elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64" OR CMAKE_SYSTEM_PROCESSOR MATCHES "AMD64")
    list(APPEND TRITON_LIBRARIES
      LLVMX86CodeGen
      LLVMX86AsmParser
    )
  elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "ppc64le")
    list(APPEND TRITON_LIBRARIES
      LLVMPowerPCAsmParser
      LLVMPowerPCCodeGen
    )
  else()
    message(FATAL_ERROR "Unsupported architecture: ${CMAKE_SYSTEM_PROCESSOR}")
  endif()

  # Build triton library
  string(JOIN "," TRITON_BACKENDS_TUPLE ${TRITON_CODEGEN_BACKENDS})
  if(DEFINED TRITON_PLUGIN_NAMES)
    string(JOIN "," TRITON_BACKENDS_TUPLE ${TRITON_BACKENDS_TUPLE} ${TRITON_PLUGIN_NAMES})
  endif()

  message(STATUS "Triton backends tuple: ${TRITON_BACKENDS_TUPLE}")
  set(TRITON_BACKENDS_TUPLE "(${TRITON_BACKENDS_TUPLE})")
  add_compile_definitions(TRITON_BACKENDS_TUPLE=${TRITON_BACKENDS_TUPLE})

  add_library(triton SHARED
    ${PYTHON_SRC_PATH}/main.cc
    ${PYTHON_SRC_PATH}/ir.cc
    ${PYTHON_SRC_PATH}/passes.cc
    ${PYTHON_SRC_PATH}/interpreter.cc
    ${PYTHON_SRC_PATH}/llvm.cc
  )

  target_link_libraries(triton PRIVATE ${TRITON_LIBRARIES})
  if(WIN32)
    target_link_libraries(triton PRIVATE ${CMAKE_DL_LIBS})
    set_target_properties(triton PROPERTIES SUFFIX ".pyd")
    set_target_properties(triton PROPERTIES PREFIX "lib")
  else()
    target_link_libraries(triton PRIVATE z)
  endif()

  target_link_options(triton PRIVATE ${LLVM_LDFLAGS})
endif()

if(UNIX AND NOT APPLE)
  set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,--exclude-libs,ALL")
endif()

if(TRITON_BUILD_PYTHON_MODULE AND NOT WIN32)
  set(CMAKE_SHARED_LIBRARY_SUFFIX ".so")
  if(APPLE)
    set(PYTHON_LDFLAGS "-undefined dynamic_lookup")
  endif()
  target_link_options(triton PRIVATE ${PYTHON_LDFLAGS})
endif()

if(NOT TRITON_BUILD_PYTHON_MODULE)
  foreach(CODEGEN_BACKEND ${TRITON_CODEGEN_BACKENDS})
    add_subdirectory(third_party/${CODEGEN_BACKEND})
  endforeach()
  add_subdirectory(third_party/proton/dialect)
endif()

find_package(Threads REQUIRED)

add_subdirectory(third_party/f2reduce)
add_subdirectory(bin)
add_subdirectory(test)

if(TRITON_BUILD_UT)
  add_subdirectory(unittest)
  add_custom_target(check-triton-unit-tests
    COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure
    DEPENDS TritonUnitTests
    USES_TERMINAL
  )
endif()
