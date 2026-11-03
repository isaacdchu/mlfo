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
    GIT_TAG v0.4.0
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

## Checkpoints

`model.save(path)` / `mlfo::model::load<T>(path)` keep only the architecture and weights.
To pause and resume training, save a checkpoint, which also keeps the parameters' gradients
and the optimizer's hyperparameters and state (e.g. Adam's moments and step count):

```cpp
mlfo::model::save_checkpoint("models/run.ckpt", model, optimizer);

// later: the model and optimizer come back together, already wired up
auto ckpt = mlfo::model::load_checkpoint<float>("models/run.ckpt", batch_size);
ckpt.optimizer->zero_grad();
ckpt.model->backward();
ckpt.optimizer->step();

// in-memory copy that trains independently of the original
auto fork = mlfo::model::clone_checkpoint(model, optimizer);
```

The epoch counter and data loader's shuffle state are part of your training loop, not the
checkpoint.

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
