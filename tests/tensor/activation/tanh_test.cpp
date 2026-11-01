#include <cmath>
#include <stdexcept>
#include <vector>

#include <gtest/gtest.h>

#include "helpers.hpp"

namespace {

using mlfo::tensor::Tensor;
using mlfo::tensor::activation::Tanh;
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

float tanh_grad(float x) {
    return 1.0f - std::tanh(x) * std::tanh(x);
}

TEST(TanhForward, ZeroMapsToZero) {
    auto a = make(1, {1}, {0});
    Tensor<float> out(1, {1});
    Tanh<float>::call(DIRECTION::FORWARD, out, a);
    expect_floats(out.values(), {0.0f});
}

TEST(TanhForward, KnownValues) {
    auto a = make(1, {2, 2}, {-2, -1, 1, 2});
    Tensor<float> out(1, {2, 2});
    Tanh<float>::call(DIRECTION::FORWARD, out, a);
    expect_floats(out.values(), {-0.96402758f, -0.76159416f, 0.76159416f, 0.96402758f});
}

TEST(TanhForward, OddFunction) {
    // tanh(-x) == -tanh(x)
    auto a = make(1, {2}, {-3, 3});
    Tensor<float> out(1, {2});
    Tanh<float>::call(DIRECTION::FORWARD, out, a);
    EXPECT_FLOAT_EQ(out.values()[0], -out.values()[1]);
}

TEST(TanhForward, SaturatesAtExtremes) {
    auto a = make(1, {2}, {-100, 100});
    Tensor<float> out(1, {2});
    Tanh<float>::call(DIRECTION::FORWARD, out, a);
    EXPECT_NEAR(out.values()[0], -1.0f, 1e-6f);
    EXPECT_NEAR(out.values()[1], 1.0f, 1e-6f);
    EXPECT_FALSE(std::isnan(out.values()[0]));
    EXPECT_FALSE(std::isnan(out.values()[1]));
}

TEST(TanhForward, Batched) {
    auto a = make(2, {3}, {-1, 0, 2, 3, -4, 5});
    Tensor<float> out(2, {3});
    Tanh<float>::call(DIRECTION::FORWARD, out, a);
    expect_floats(out.values(), {
        std::tanh(-1.0f), 0.0f, std::tanh(2.0f), std::tanh(3.0f), std::tanh(-4.0f), std::tanh(5.0f)
    });
}

TEST(TanhErrors, ShapeMismatch) {
    Tensor<float> a(1, {2, 2});
    Tensor<float> out(1, {4});
    EXPECT_THROW(Tanh<float>::call(DIRECTION::FORWARD, out, a), std::invalid_argument);
}

TEST(TanhErrors, BatchMismatch) {
    Tensor<float> a(2, {2});
    Tensor<float> out(3, {2});
    EXPECT_THROW(Tanh<float>::call(DIRECTION::FORWARD, out, a), std::invalid_argument);
}

TEST(TanhErrors, BackwardAlsoValidates) {
    Tensor<float> a(2, {2});
    Tensor<float> out(3, {2});
    EXPECT_THROW(Tanh<float>::call(DIRECTION::BACKWARD, out, a), std::invalid_argument);
}

TEST(TanhBackward, GradientAtZeroIsOne) {
    auto a = make(1, {1}, {0});
    Tensor<float> out(1, {1});
    Tanh<float>::call(DIRECTION::FORWARD, out, a);
    TensorAccess<float>::gradients(out) = {1};
    Tanh<float>::call(DIRECTION::BACKWARD, out, a);
    // derivative at 0 is 1 - tanh(0)^2
    expect_floats(a.gradients(), {1.0f});
}

TEST(TanhBackward, ScalesUpstreamByDerivative) {
    auto a = make(1, {4}, {-2, -1, 1, 2});
    Tensor<float> out(1, {4});
    Tanh<float>::call(DIRECTION::FORWARD, out, a);
    TensorAccess<float>::gradients(out) = {1, 2, 3, 4};
    Tanh<float>::call(DIRECTION::BACKWARD, out, a);
    expect_floats(a.gradients(), {
        1 * tanh_grad(-2), 2 * tanh_grad(-1), 3 * tanh_grad(1), 4 * tanh_grad(2)
    });
}

TEST(TanhBackward, VanishesWhenSaturated) {
    auto a = make(1, {2}, {-100, 100});
    Tensor<float> out(1, {2});
    Tanh<float>::call(DIRECTION::FORWARD, out, a);
    TensorAccess<float>::gradients(out) = {1, 1};
    Tanh<float>::call(DIRECTION::BACKWARD, out, a);
    EXPECT_NEAR(a.gradients()[0], 0.0f, 1e-6f);
    EXPECT_NEAR(a.gradients()[1], 0.0f, 1e-6f);
}

TEST(TanhBackward, GradientsAccumulate) {
    auto a = make(1, {3}, {1, -1, 0});
    Tensor<float> out(1, {3});
    Tanh<float>::call(DIRECTION::FORWARD, out, a);
    TensorAccess<float>::gradients(out) = {1, 2, 3};
    Tanh<float>::call(DIRECTION::BACKWARD, out, a);
    Tanh<float>::call(DIRECTION::BACKWARD, out, a);
    expect_floats(a.gradients(), {
        2 * 1 * tanh_grad(1), 2 * 2 * tanh_grad(-1), 2 * 3 * tanh_grad(0)
    });
}

TEST(TanhBackward, BatchedPerElementGradients) {
    auto a = make(2, {3}, {-1, 0, 2, 3, -4, 5});
    Tensor<float> out(2, {3});
    ASSERT_EQ(a.gradients().size(), a.size());
    ASSERT_EQ(out.gradients().size(), out.size());
    Tanh<float>::call(DIRECTION::FORWARD, out, a);
    TensorAccess<float>::gradients(out) = {1, 2, 3, 4, 5, 6};
    Tanh<float>::call(DIRECTION::BACKWARD, out, a);
    expect_floats(a.gradients(), {
        1 * tanh_grad(-1), 2 * tanh_grad(0), 3 * tanh_grad(2),
        4 * tanh_grad(3), 5 * tanh_grad(-4), 6 * tanh_grad(5)
    });
}

} // namespace
