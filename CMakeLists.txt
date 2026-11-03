cmake_minimum_required(VERSION 3.21)
project(mlfo VERSION 0.4.0 LANGUAGES CXX)

add_library(mlfo INTERFACE)
add_library(mlfo::mlfo ALIAS mlfo)
target_include_directories(mlfo INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_compile_features(mlfo INTERFACE cxx_std_23)

# on mingw, std::print's windows terminal support lives in libstdc++exp
if(MINGW)
    target_link_libraries(mlfo INTERFACE stdc++exp)
endif()

# only build the demo and tests when this is the top-level project,
# not when pulled in via add_subdirectory/FetchContent
option(MLFO_BUILD_EXAMPLES "Build the mlfo demo executable" ${PROJECT_IS_TOP_LEVEL})
option(MLFO_BUILD_TESTS "Build the mlfo tests" ${PROJECT_IS_TOP_LEVEL})

if(MLFO_BUILD_EXAMPLES)
    add_executable(main main.cpp)
    target_link_libraries(main PRIVATE mlfo::mlfo)
    target_compile_options(main PRIVATE -Wall -Wextra -Wno-unused-parameter -Wno-unused-variable -Wno-unused-function -Werror -O2)
endif()

if(MLFO_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
