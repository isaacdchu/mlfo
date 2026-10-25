cmake_minimum_required(VERSION 3.20)
project(mlfo LANGUAGES CXX)

add_library(mlfo INTERFACE)
target_include_directories(mlfo INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/mlfo
)
target_compile_features(mlfo INTERFACE cxx_std_23)

# on mingw, std::print's windows terminal support lives in libstdc++exp
if(MINGW)
    target_link_libraries(mlfo INTERFACE stdc++exp)
endif()

add_executable(main main.cpp)
target_link_libraries(main PRIVATE mlfo)
target_compile_options(main PRIVATE -Wall -Wextra -Wno-unused-parameter -Wno-unused-variable -Wno-unused-function -Werror -O2)

enable_testing()
add_subdirectory(tests)
