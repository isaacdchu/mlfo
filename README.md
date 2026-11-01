# Machine Learning Framework for Othello
Attempt 3 at making a machine learning framework

Header-only, C++23.

## Requirements

- C++23 (uses `<print>`)
- CMake 3.21+
- On MinGW, `stdc++exp` is linked automatically for `std::print`

## Using it in another project

With CMake `FetchContent`:

```cmake
include(FetchContent)
FetchContent_Declare(mlfo
    GIT_REPOSITORY https://github.com/isaacdchu/mlfo.git
    GIT_TAG v0.1.0
)
FetchContent_MakeAvailable(mlfo)

target_link_libraries(your_target PRIVATE mlfo::mlfo)
```

Or as a git submodule:

```cmake
add_subdirectory(external/mlfo)
target_link_libraries(your_target PRIVATE mlfo::mlfo)
```

When included this way, the demo executable and tests are not built.

Then include everything at once:

```cpp
#include <mlfo/mlfo.hpp>
```

or just the parts you need, e.g. `<mlfo/tensor/tensor.hpp>`, `<mlfo/model/model.hpp>`,
`<mlfo/layer/dense_layer.hpp>`.

## Compiling
Configure once:
```bash
cmake -S . -B build
```
- `main.cpp`
```bash
cmake --build build --target main
./build/main
```
- All tests
```bash
cmake --build build --target mlfo_tests
ctest --test-dir build
```

Options (default `ON` when this is the top-level project, `OFF` otherwise):

- `MLFO_BUILD_EXAMPLES` - build the `main` demo
- `MLFO_BUILD_TESTS` - build the googletest suite
