if(TRITON_BUILD_PYTHON_MODULE AND APPLE)
  # Find the Metal framework
  find_library(METAL_FRAMEWORK Metal)
  find_library(FOUNDATION_FRAMEWORK Foundation)

  if(METAL_FRAMEWORK AND FOUNDATION_FRAMEWORK)
    message(STATUS "Metal framework found: ${METAL_FRAMEWORK}")
    message(STATUS "Foundation framework found: ${FOUNDATION_FRAMEWORK}")

    # Build the metal_utils Python C extension
    add_triton_plugin(TritonMetal ${CMAKE_CURRENT_SOURCE_DIR}/backend/driver.c)
    target_link_libraries(TritonMetal PRIVATE
      ${METAL_FRAMEWORK}
      ${FOUNDATION_FRAMEWORK}
      Python3::Module
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
