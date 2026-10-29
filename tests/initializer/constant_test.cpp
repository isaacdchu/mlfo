#include <gtest/gtest.h>

#include <cstddef>
#include <vector>

#include "helpers.hpp"

namespace {

using Tensor = mlfo::tensor::Tensor<float>;
using Shape = std::vector<std::size_t>;

} // namespace

TEST(Constant, FillsEveryValue) {
    Tensor tensor(1, Shape{2, 3});
    mlfo::initializer::Constant<float>(2.5f).initialize(tensor, 2, 3);
    for (float value : tensor.values()) {
        EXPECT_FLOAT_EQ(value, 2.5f);
    }
}

TEST(Zeros, OverwritesExistingValues) {
    Tensor tensor(1, Shape{3});
    tensor.set_values({1, 2, 3});
    mlfo::initializer::Zeros<float>().initialize(tensor, 3, 1);
    for (float value : tensor.values()) {
        EXPECT_FLOAT_EQ(value, 0.0f);
    }
}

TEST(Constant, LeavesGradientsAlone) {
    Tensor tensor(1, Shape{2});
    tensor.set_gradients({4, 5});
    mlfo::initializer::Constant<float>(1.0f).initialize(tensor, 2, 1);
    EXPECT_FLOAT_EQ(tensor.gradients()[0], 4.0f);
    EXPECT_FLOAT_EQ(tensor.gradients()[1], 5.0f);
}
