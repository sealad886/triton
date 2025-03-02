# CMake configuration for MPS backend tests

# Add the MPS backend test only on macOS
if(APPLE)
    add_triton_unittest(test_mps_backend
        test_mps_backend.cpp
    )

    # Link against MPS backend and required frameworks
    target_link_libraries(test_mps_backend
        PRIVATE
            TritonMPSBackend
            gtest
            gtest_main
            ${FOUNDATION_FRAMEWORK}
            ${METAL_FRAMEWORK}
            ${MPS_FRAMEWORK}
    )

    target_include_directories(test_mps_backend
        PRIVATE
            ${CMAKE_SOURCE_DIR}/python/triton/backends/apple/include
    )

    # Set Objective-C++ specific flags
    set_source_files_properties(
        test_mps_backend.cpp
        PROPERTIES
        COMPILE_FLAGS "-x objective-c++"
    )

    # Enable testing
    gtest_discover_tests(test_mps_backend)
endif()

# Add other existing tests
add_triton_unittest(test-alias
    test-alias.mlir
)

add_triton_unittest(test-allocation
    test-allocation.mlir
)

add_triton_unittest(test-alignment
    test-alignment.mlir
)

add_triton_unittest(test-membar
    test-membar.mlir
)
