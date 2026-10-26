#include <cmath>
#include <stdexcept>
#include <vector>

#include <gtest/gtest.h>

#include "helpers.hpp"

namespace {

using mlfo::tensor::Tensor;
using mlfo::tensor::activation::Sigmoid;
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

float sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

float sigmoid_grad(float x) {
    return sigmoid(x) * (1.0f - sigmoid(x));
}

TEST(SigmoidForward, ZeroMapsToHalf) {
    auto a = make(1, {1}, {0});
    Tensor<float> out(1, {1});
    Sigmoid<float>::call(DIRECTION::FORWARD, out, a);
    expect_floats(out.values(), {0.5f});
}

TEST(SigmoidForward, KnownValues) {
    auto a = make(1, {2, 2}, {-2, -1, 1, 2});
    Tensor<float> out(1, {2, 2});
    Sigmoid<float>::call(DIRECTION::FORWARD, out, a);
    expect_floats(out.values(), {0.11920292f, 0.26894142f, 0.73105858f, 0.88079708f});
}

TEST(SigmoidForward, Symmetric) {
    // sigmoid(-x) == 1 - sigmoid(x)
    auto a = make(1, {2}, {-3, 3});
    Tensor<float> out(1, {2});
    Sigmoid<float>::call(DIRECTION::FORWARD, out, a);
    EXPECT_FLOAT_EQ(out.values()[0], 1.0f - out.values()[1]);
}

TEST(SigmoidForward, SaturatesAtExtremes) {
    auto a = make(1, {2}, {-100, 100});
    Tensor<float> out(1, {2});
    Sigmoid<float>::call(DIRECTION::FORWARD, out, a);
    EXPECT_NEAR(out.values()[0], 0.0f, 1e-6f);
    EXPECT_NEAR(out.values()[1], 1.0f, 1e-6f);
    EXPECT_FALSE(std::isnan(out.values()[0]));
    EXPECT_FALSE(std::isnan(out.values()[1]));
}

TEST(SigmoidForward, Batched) {
    auto a = make(2, {3}, {-1, 0, 2, 3, -4, 5});
    Tensor<float> out(2, {3});
    Sigmoid<float>::call(DIRECTION::FORWARD, out, a);
    expect_floats(out.values(), {
        sigmoid(-1), sigmoid(0), sigmoid(2), sigmoid(3), sigmoid(-4), sigmoid(5)
    });
}

TEST(SigmoidErrors, ShapeMismatch) {
    Tensor<float> a(1, {2, 2});
    Tensor<float> out(1, {4});
    EXPECT_THROW(Sigmoid<float>::call(DIRECTION::FORWARD, out, a), std::invalid_argument);
}

TEST(SigmoidErrors, BatchMismatch) {
    Tensor<float> a(2, {2});
    Tensor<float> out(3, {2});
    EXPECT_THROW(Sigmoid<float>::call(DIRECTION::FORWARD, out, a), std::invalid_argument);
}

TEST(SigmoidErrors, BackwardAlsoValidates) {
    Tensor<float> a(2, {2});
    Tensor<float> out(3, {2});
    EXPECT_THROW(Sigmoid<float>::call(DIRECTION::BACKWARD, out, a), std::invalid_argument);
}

TEST(SigmoidBackward, GradientAtZeroIsQuarter) {
    auto a = make(1, {1}, {0});
    Tensor<float> out(1, {1});
    Sigmoid<float>::call(DIRECTION::FORWARD, out, a);
    TensorAccess<float>::gradients(out) = {1};
    Sigmoid<float>::call(DIRECTION::BACKWARD, out, a);
    // derivative at 0 is 0.5 * (1 - 0.5)
    expect_floats(a.gradients(), {0.25f});
}

TEST(SigmoidBackward, ScalesUpstreamByDerivative) {
    auto a = make(1, {4}, {-2, -1, 1, 2});
    Tensor<float> out(1, {4});
    Sigmoid<float>::call(DIRECTION::FORWARD, out, a);
    TensorAccess<float>::gradients(out) = {1, 2, 3, 4};
    Sigmoid<float>::call(DIRECTION::BACKWARD, out, a);
    expect_floats(a.gradients(), {
        1 * sigmoid_grad(-2), 2 * sigmoid_grad(-1), 3 * sigmoid_grad(1), 4 * sigmoid_grad(2)
    });
}

TEST(SigmoidBackward, GradientsAccumulate) {
    auto a = make(1, {3}, {1, -1, 0});
    Tensor<float> out(1, {3});
    Sigmoid<float>::call(DIRECTION::FORWARD, out, a);
    TensorAccess<float>::gradients(out) = {1, 2, 3};
    Sigmoid<float>::call(DIRECTION::BACKWARD, out, a);
    Sigmoid<float>::call(DIRECTION::BACKWARD, out, a);
    expect_floats(a.gradients(), {
        2 * 1 * sigmoid_grad(1), 2 * 2 * sigmoid_grad(-1), 2 * 3 * sigmoid_grad(0)
    });
}

TEST(SigmoidBackward, BatchedPerElementGradients) {
    auto a = make(2, {3}, {-1, 0, 2, 3, -4, 5});
    Tensor<float> out(2, {3});
    ASSERT_EQ(a.gradients().size(), a.size());
    ASSERT_EQ(out.gradients().size(), out.size());
    Sigmoid<float>::call(DIRECTION::FORWARD, out, a);
    TensorAccess<float>::gradients(out) = {1, 2, 3, 4, 5, 6};
    Sigmoid<float>::call(DIRECTION::BACKWARD, out, a);
    expect_floats(a.gradients(), {
        1 * sigmoid_grad(-1), 2 * sigmoid_grad(0), 3 * sigmoid_grad(2),
        4 * sigmoid_grad(3), 5 * sigmoid_grad(-4), 6 * sigmoid_grad(5)
    });
}

} // namespace
