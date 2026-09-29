include(FetchContent)
FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.17.0.tar.gz
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    SYSTEM
)
FetchContent_MakeAvailable(googletest)

include(GoogleTest)

# every *_test.cpp under tests/ becomes its own executable, so adding a test only
# requires adding the file. the target name is the file path with / replaced by _,
# e.g. tests/tensor/ops/add_test.cpp -> tensor_ops_add_test
file(GLOB_RECURSE MLFO_TEST_SOURCES CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/*_test.cpp)
foreach(test_source ${MLFO_TEST_SOURCES})
    file(RELATIVE_PATH test_name ${CMAKE_CURRENT_SOURCE_DIR} ${test_source})
    string(REPLACE ".cpp" "" test_name ${test_name})
    string(REPLACE "/" "_" test_name ${test_name})
    add_executable(${test_name} ${test_source})
    target_include_directories(${test_name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_libraries(${test_name} PRIVATE mlfo GTest::gtest_main)
    target_compile_options(${test_name} PRIVATE -Wall -Wextra -Wno-unused-parameter -Wno-unused-variable -Wno-unused-function -Werror)
    gtest_discover_tests(${test_name})
endforeach()
