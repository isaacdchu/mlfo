#include <stdexcept>
#include <vector>

#include <gtest/gtest.h>

#include "helpers.hpp"

namespace {

using mlfo::tensor::Tensor;
using mlfo::tensor::loss::MSE;
using mlfo::tensor::ops::DIRECTION;
using mlfo::tests::TensorAccess;

Tensor<float> make(std::size_t batch, std::vector<std::size_t> shape, std::vector<float> values) {
    Tensor<float> t(batch, shape);
    t.set_values(values);
    return t;
}

void expect_floats(const std::vector<float>& actual, const std::vector<float>& expected) {
    ASSERT_EQ(actual.size(), expected.size());
    for (std::size_t i = 0; i < expected.size(); i++) {
        EXPECT_FLOAT_EQ(actual[i], expected[i]) << "index " << i;
    }
}

} // namespace

TEST(MSEForward, BatchSizeOne) {
    auto a = make(1, {3}, {1, 2, 3});
    auto b = make(1, {3}, {1, 0, 0});
    Tensor<float> out(1, {1});
    MSE<float>::call(DIRECTION::FORWARD, out, a, b);
    // errors = [0, 2, 3]; 0.5 * 13 / 3
    expect_floats(out.values(), {13.0f / 6.0f});
}

TEST(MSEForward, EqualInputsGiveZero) {
    auto a = make(1, {2}, {4, -5});
    auto b = make(1, {2}, {4, -5});
    Tensor<float> out(1, {1});
    MSE<float>::call(DIRECTION::FORWARD, out, a, b);
    expect_floats(out.values(), {0});
}

TEST(MSEForward, OneLossPerBatch) {
    auto a = make(2, {2}, {1, 2, 3, 4});
    auto b = make(2, {2}, {0, 0, 1, 1});
    Tensor<float> out(2, {1});
    MSE<float>::call(DIRECTION::FORWARD, out, a, b);
    // batch 0: 0.5 * (1 + 4) / 2; batch 1: 0.5 * (4 + 9) / 2
    expect_floats(out.values(), {1.25f, 3.25f});
}

TEST(MSEErrors, ShapeMismatchBetweenAAndB) {
    Tensor<float> a(1, {3});
    Tensor<float> b(1, {2});
    Tensor<float> out(1, {1});
    EXPECT_THROW(MSE<float>::call(DIRECTION::FORWARD, out, a, b), std::invalid_argument);
}

TEST(MSEErrors, OutNotSingleValuePerBatch) {
    Tensor<float> a(1, {2});
    Tensor<float> b(1, {2});
    Tensor<float> out(1, {2});
    EXPECT_THROW(MSE<float>::call(DIRECTION::FORWARD, out, a, b), std::invalid_argument);
}

TEST(MSEErrors, BatchSizeMismatch) {
    Tensor<float> a(2, {2});
    Tensor<float> b(1, {2});
    Tensor<float> out(2, {1});
    EXPECT_THROW(MSE<float>::call(DIRECTION::FORWARD, out, a, b), std::invalid_argument);
}

TEST(MSEErrors, BackwardAlsoValidates) {
    Tensor<float> a(1, {3});
    Tensor<float> b(1, {2});
    Tensor<float> out(1, {1});
    EXPECT_THROW(MSE<float>::call(DIRECTION::BACKWARD, out, a, b), std::invalid_argument);
}

TEST(MSEBackward, GradientsFlowToBothInputs) {
    auto a = make(1, {3}, {1, 2, 3});
    auto b = make(1, {3}, {1, 0, 0});
    Tensor<float> out(1, {1});
    TensorAccess<float>::gradients(out) = {1};
    MSE<float>::call(DIRECTION::BACKWARD, out, a, b);
    // dL/da = (a - b) / n = -dL/db
    expect_floats(a.gradients(), {0, 2.0f / 3.0f, 1});
    expect_floats(b.gradients(), {0, -2.0f / 3.0f, -1});
}

TEST(MSEBackward, ScaledByUpstreamGradient) {
    auto a = make(1, {2}, {3, 1});
    auto b = make(1, {2}, {1, 1});
    Tensor<float> out(1, {1});
    TensorAccess<float>::gradients(out) = {3};
    MSE<float>::call(DIRECTION::BACKWARD, out, a, b);
    expect_floats(a.gradients(), {3, 0});
    expect_floats(b.gradients(), {-3, 0});
}

TEST(MSEBackward, GradientsAccumulate) {
    auto a = make(1, {2}, {3, 1});
    auto b = make(1, {2}, {1, 1});
    Tensor<float> out(1, {1});
    TensorAccess<float>::gradients(out) = {1};
    TensorAccess<float>::gradients(a) = {10, 20};
    TensorAccess<float>::gradients(b) = {10, 20};
    MSE<float>::call(DIRECTION::BACKWARD, out, a, b);
    expect_floats(a.gradients(), {11, 20});
    expect_floats(b.gradients(), {9, 20});
}

TEST(MSEBackward, BatchedPerBatchUpstreamGradient) {
    auto a = make(2, {2}, {1, 2, 3, 4});
    auto b = make(2, {2}, {0, 0, 1, 1});
    Tensor<float> out(2, {1});
    TensorAccess<float>::gradients(out) = {1, 2};
    MSE<float>::call(DIRECTION::BACKWARD, out, a, b);
    // batch 0: [1, 2] / 2; batch 1: 2 * [2, 3] / 2
    expect_floats(a.gradients(), {0.5f, 1, 2, 3});
    expect_floats(b.gradients(), {-0.5f, -1, -2, -3});
}
