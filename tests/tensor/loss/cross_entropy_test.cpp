#include <cmath>
#include <stdexcept>
#include <vector>

#include <gtest/gtest.h>

#include "helpers.hpp"

namespace {

using mlfo::tensor::Tensor;
using mlfo::tensor::loss::CrossEntropy;
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

TEST(CrossEntropyForward, UniformLogits) {
    auto a = make(1, {4}, {2, 2, 2, 2});
    auto b = make(1, {4}, {0, 0, 1, 0});
    Tensor<float> out(1, {1});
    CrossEntropy<float>::call(DIRECTION::FORWARD, out, a, b);
    // softmax is 1/4 everywhere, so -log(1/4)
    expect_floats(out.values(), {std::log(4.0f)});
}

TEST(CrossEntropyForward, OneHot) {
    auto a = make(1, {3}, {1, 2, 3});
    auto b = make(1, {3}, {0, 1, 0});
    Tensor<float> out(1, {1});
    CrossEntropy<float>::call(DIRECTION::FORWARD, out, a, b);
    const float lse = std::log(std::exp(1.0f) + std::exp(2.0f) + std::exp(3.0f));
    expect_floats(out.values(), {lse - 2.0f});
}

TEST(CrossEntropyForward, LargeLogitsDoNotOverflow) {
    auto a = make(1, {2}, {1000, 0});
    auto b = make(1, {2}, {0, 1});
    Tensor<float> out(1, {1});
    CrossEntropy<float>::call(DIRECTION::FORWARD, out, a, b);
    // log(exp(1000) + 1) - 0 is 1000 to float precision
    expect_floats(out.values(), {1000});
}

TEST(CrossEntropyForward, OneLossPerBatch) {
    auto a = make(2, {2}, {0, 0, 1, 1});
    auto b = make(2, {2}, {1, 0, 0, 1});
    Tensor<float> out(2, {1});
    CrossEntropy<float>::call(DIRECTION::FORWARD, out, a, b);
    expect_floats(out.values(), {std::log(2.0f), std::log(2.0f)});
}

TEST(CrossEntropyErrors, ShapeMismatchBetweenAAndB) {
    Tensor<float> a(1, {3});
    Tensor<float> b(1, {2});
    Tensor<float> out(1, {1});
    EXPECT_THROW(CrossEntropy<float>::call(DIRECTION::FORWARD, out, a, b), std::invalid_argument);
}

TEST(CrossEntropyErrors, OutNotSingleValuePerBatch) {
    Tensor<float> a(1, {2});
    Tensor<float> b(1, {2});
    Tensor<float> out(1, {2});
    EXPECT_THROW(CrossEntropy<float>::call(DIRECTION::FORWARD, out, a, b), std::invalid_argument);
}

TEST(CrossEntropyErrors, BatchSizeMismatch) {
    Tensor<float> a(2, {2});
    Tensor<float> b(1, {2});
    Tensor<float> out(2, {1});
    EXPECT_THROW(CrossEntropy<float>::call(DIRECTION::FORWARD, out, a, b), std::invalid_argument);
}

TEST(CrossEntropyErrors, BackwardAlsoValidates) {
    Tensor<float> a(1, {3});
    Tensor<float> b(1, {2});
    Tensor<float> out(1, {1});
    EXPECT_THROW(CrossEntropy<float>::call(DIRECTION::BACKWARD, out, a, b), std::invalid_argument);
}

TEST(CrossEntropyBackward, SoftmaxMinusLabels) {
    auto a = make(1, {4}, {2, 2, 2, 2});
    auto b = make(1, {4}, {0, 0, 1, 0});
    Tensor<float> out(1, {1});
    TensorAccess<float>::gradients(out) = {1};
    CrossEntropy<float>::call(DIRECTION::BACKWARD, out, a, b);
    // dL/da = softmax(a) - b, dL/db = -log(softmax(a))
    expect_floats(a.gradients(), {0.25f, 0.25f, -0.75f, 0.25f});
    const float log4 = std::log(4.0f);
    expect_floats(b.gradients(), {log4, log4, log4, log4});
}

TEST(CrossEntropyBackward, ScaledByUpstreamGradient) {
    auto a = make(1, {2}, {0, 0});
    auto b = make(1, {2}, {1, 0});
    Tensor<float> out(1, {1});
    TensorAccess<float>::gradients(out) = {3};
    CrossEntropy<float>::call(DIRECTION::BACKWARD, out, a, b);
    expect_floats(a.gradients(), {-1.5f, 1.5f});
}

TEST(CrossEntropyBackward, GradientsAccumulate) {
    auto a = make(1, {2}, {0, 0});
    auto b = make(1, {2}, {1, 0});
    Tensor<float> out(1, {1});
    TensorAccess<float>::gradients(out) = {1};
    TensorAccess<float>::gradients(a) = {10, 20};
    CrossEntropy<float>::call(DIRECTION::BACKWARD, out, a, b);
    expect_floats(a.gradients(), {9.5f, 20.5f});
}

TEST(CrossEntropyBackward, BatchedPerBatchUpstreamGradient) {
    auto a = make(2, {2}, {0, 0, 0, 0});
    auto b = make(2, {2}, {1, 0, 0, 1});
    Tensor<float> out(2, {1});
    TensorAccess<float>::gradients(out) = {1, 2};
    CrossEntropy<float>::call(DIRECTION::BACKWARD, out, a, b);
    // batch 0: [0.5 - 1, 0.5]; batch 1: 2 * [0.5, 0.5 - 1]
    expect_floats(a.gradients(), {-0.5f, 0.5f, 1, -1});
}
