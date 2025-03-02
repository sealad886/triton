# Function to add Triton unit tests
function(add_triton_unittest test_name)
  cmake_parse_arguments(ARG "" "" "SOURCES;LIBS" ${ARGN})

  # Create test target
  add_executable(${test_name} ${ARG_SOURCES})

  # Set test properties
  set_target_properties(${test_name} PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
  )

  # Add gtest dependencies
  target_link_libraries(${test_name}
    PRIVATE
      gtest
      gtest_main
      ${ARG_LIBS}
  )

  # Add test to global target
  add_dependencies(TritonUnitTests ${test_name})

  # Register test with ctest
  add_test(NAME ${test_name} COMMAND ${test_name})

  # Set test timeout (5 minutes)
  set_tests_properties(${test_name} PROPERTIES TIMEOUT 300)

  # Set test folder for IDEs
  set_target_properties(${test_name} PROPERTIES FOLDER "Triton/Tests")
endfunction()

# Add a disabled test (for tests that are not yet ready)
function(add_disabled_triton_unittest test_name)
  message(STATUS "Test ${test_name} is disabled")
endfunction()
