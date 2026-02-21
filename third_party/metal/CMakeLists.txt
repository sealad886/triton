include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include)
add_subdirectory(include)
add_subdirectory(lib)

if(TRITON_BUILD_PYTHON_MODULE AND APPLE)
  # Find the Metal framework
  find_library(METAL_FRAMEWORK Metal)
  find_library(FOUNDATION_FRAMEWORK Foundation)

  if(METAL_FRAMEWORK AND FOUNDATION_FRAMEWORK)
    message(STATUS "Metal framework found: ${METAL_FRAMEWORK}")
    message(STATUS "Foundation framework found: ${FOUNDATION_FRAMEWORK}")

    # Build the metal_utils Python C extension
    add_triton_plugin(TritonMetal ${CMAKE_CURRENT_SOURCE_DIR}/backend/driver.c ${CMAKE_CURRENT_SOURCE_DIR}/python/triton_metal.cc LINK_LIBS TritonMetalGPUToLLVM)
    target_link_libraries(TritonMetal PRIVATE
      ${METAL_FRAMEWORK}
      ${FOUNDATION_FRAMEWORK}
      MLIRGPUToLLVMIRTranslation
      Python3::Module
      pybind11::headers
    )

    # Ensure Objective-C compilation for .c files using Metal headers
    set_source_files_properties(
      ${CMAKE_CURRENT_SOURCE_DIR}/backend/driver.c
      PROPERTIES LANGUAGE OBJC
    )
  else()
    message(WARNING "Metal or Foundation framework not found; Metal backend will be Python-only")
  endif()
endif()


