#include <stdexcept>
#include <vector>

#include <gtest/gtest.h>

#include "helpers.hpp"

namespace {

using mlfo::tensor::Tensor;
using mlfo::tensor::ops::Conv2D;
using mlfo::tensor::ops::DIRECTION;
using mlfo::tests::TensorAccess;

Tensor<float> make(std::size_t batch, std::vector<std::size_t> shape, std::vector<float> values) {
    Tensor<float> t(batch, shape);
    t.set_values(std::move(values));
    return t;
}

void expect_floats(const std::vector<float>& actual, const std::vector<float>& expected) {
    ASSERT_EQ(actual.size(), expected.size());
    for (std::size_t i = 0; i < expected.size(); i++) {
        EXPECT_FLOAT_EQ(actual[i], expected[i]) << "index " << i;
    }
}

TEST(Conv2DForward, SingleChannelNoPadding) {
    // 3x3 input, 2x2 kernel -> 2x2 output
    auto input = make(1, {1, 3, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9});
    auto kernel = make(1, {1, 1, 2, 2}, {1, 0, 0, 1});
    Tensor<float> out(1, {1, 2, 2});
    Conv2D<float>::call(DIRECTION::FORWARD, out, input, kernel, 1, 1, 0, 0);
    expect_floats(out.values(), {6, 8, 12, 14});
}

TEST(Conv2DForward, PaddingAndStride) {
    // 3x3 input padded to 5x5, 3x3 kernel of ones, stride 2 -> 2x2 output
    auto input = make(1, {1, 3, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9});
    auto kernel = make(1, {1, 1, 3, 3}, std::vector<float>(9, 1));
    Tensor<float> out(1, {1, 2, 2});
    Conv2D<float>::call(DIRECTION::FORWARD, out, input, kernel, 2, 2, 1, 1);
    // each output sums the 2x2 corner block of the input
    expect_floats(out.values(), {12, 16, 24, 28});
}

TEST(Conv2DForward, MultiChannelMultiKernel) {
    // 2 channels of 2x2, 2 kernels of 2x2x2 -> 2 outputs of 1x1
    auto input = make(1, {2, 2, 2}, {1, 2, 3, 4, 5, 6, 7, 8});
    auto kernel = make(1, {2, 2, 2, 2}, {
        1, 1, 1, 1, 0, 0, 0, 0,
        0, 0, 0, 0, 1, 0, 0, 1
    });
    Tensor<float> out(1, {2, 1, 1});
    Conv2D<float>::call(DIRECTION::FORWARD, out, input, kernel, 1, 1, 0, 0);
    expect_floats(out.values(), {10, 13});
}

TEST(Conv2DForward, BatchedWithBroadcastKernel) {
    auto input = make(2, {1, 2, 2}, {1, 2, 3, 4, 5, 6, 7, 8});
    auto kernel = make(1, {1, 1, 2, 2}, {1, 1, 1, 1});
    Tensor<float> out(2, {1, 1, 1});
    Conv2D<float>::call(DIRECTION::FORWARD, out, input, kernel, 1, 1, 0, 0);
    expect_floats(out.values(), {10, 26});
}

TEST(Conv2DBackward, MatchesNumericalGradient) {
    // compares against central differences of loss = sum(out * out_gradients)
    const std::size_t stride = 2;
    const std::size_t padding = 1;
    std::vector<float> input_values(2 * 2 * 4 * 5);
    for (std::size_t i = 0; i < input_values.size(); i++) {
        input_values[i] = static_cast<float>((i * 7) % 11) / 5.0f - 1.0f;
    }
    std::vector<float> kernel_values(3 * 2 * 3 * 2);
    for (std::size_t i = 0; i < kernel_values.size(); i++) {
        kernel_values[i] = static_cast<float>((i * 5) % 13) / 6.0f - 1.0f;
    }
    auto input = make(2, {2, 4, 5}, input_values);
    auto kernel = make(1, {3, 2, 3, 2}, kernel_values);
    // p = (4 + 2 - 3) / 2 + 1 = 2, q = (5 + 2 - 2) / 2 + 1 = 3
    Tensor<float> out(2, {3, 2, 3});
    std::vector<float> out_gradients(out.size());
    for (std::size_t i = 0; i < out_gradients.size(); i++) {
        out_gradients[i] = static_cast<float>((i * 3) % 7) / 3.0f - 1.0f;
    }
    TensorAccess<float>::gradients(out) = out_gradients;
    Conv2D<float>::call(DIRECTION::FORWARD, out, input, kernel, stride, stride, padding, padding);
    Conv2D<float>::call(DIRECTION::BACKWARD, out, input, kernel, stride, stride, padding, padding);

    auto loss = [&](const std::vector<float>& in, const std::vector<float>& ker) {
        auto i = make(2, {2, 4, 5}, in);
        auto k = make(1, {3, 2, 3, 2}, ker);
        Tensor<float> o(2, {3, 2, 3});
        Conv2D<float>::call(DIRECTION::FORWARD, o, i, k, stride, stride, padding, padding);
        double total = 0;
        for (std::size_t j = 0; j < o.size(); j++) {
            total += o.values()[j] * out_gradients[j];
        }
        return total;
    };
    const float eps = 1e-2f;
    for (std::size_t i = 0; i < input_values.size(); i++) {
        auto plus = input_values;
        auto minus = input_values;
        plus[i] += eps;
        minus[i] -= eps;
        const double numeric = (loss(plus, kernel_values) - loss(minus, kernel_values)) / (2 * eps);
        EXPECT_NEAR(input.gradients()[i], numeric, 1e-3) << "input index " << i;
    }
    for (std::size_t i = 0; i < kernel_values.size(); i++) {
        auto plus = kernel_values;
        auto minus = kernel_values;
        plus[i] += eps;
        minus[i] -= eps;
        const double numeric = (loss(input_values, plus) - loss(input_values, minus)) / (2 * eps);
        EXPECT_NEAR(kernel.gradients()[i], numeric, 1e-3) << "kernel index " << i;
    }
}

TEST(Conv2DBackward, AccumulatesIntoExistingGradients) {
    auto input = make(1, {1, 2, 2}, {1, 2, 3, 4});
    auto kernel = make(1, {1, 1, 2, 2}, {1, 2, 3, 4});
    Tensor<float> out(1, {1, 1, 1});
    TensorAccess<float>::gradients(out) = {1};
    TensorAccess<float>::gradients(input) = {1, 1, 1, 1};
    Conv2D<float>::call(DIRECTION::BACKWARD, out, input, kernel, 1, 1, 0, 0);
    expect_floats(input.gradients(), {2, 3, 4, 5});
    expect_floats(kernel.gradients(), {1, 2, 3, 4});
}

TEST(Conv2DCall, RejectsChannelMismatch) {
    Tensor<float> input(1, {2, 3, 3});
    Tensor<float> kernel(1, {1, 1, 2, 2});
    Tensor<float> out(1, {1, 2, 2});
    EXPECT_THROW(
        Conv2D<float>::call(DIRECTION::FORWARD, out, input, kernel, 1, 1, 0, 0),
        std::invalid_argument
    );
}

TEST(Conv2DCall, RejectsWrongOutputShape) {
    Tensor<float> input(1, {1, 3, 3});
    Tensor<float> kernel(1, {1, 1, 2, 2});
    Tensor<float> out(1, {1, 3, 3});
    EXPECT_THROW(
        Conv2D<float>::call(DIRECTION::FORWARD, out, input, kernel, 1, 1, 0, 0),
        std::invalid_argument
    );
}

} // namespace
