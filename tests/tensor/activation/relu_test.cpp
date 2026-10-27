#include <stdexcept>
#include <vector>

#include <gtest/gtest.h>

#include "helpers.hpp"

namespace {

using mlfo::tensor::Tensor;
using mlfo::tensor::activation::ReLU;
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

TEST(ReLUForward, ZeroesNegatives) {
    auto a = make(1, {2, 2}, {-1, 2, -3, 4});
    Tensor<float> out(1, {2, 2});
    ReLU<float>::call(DIRECTION::FORWARD, out, a);
    expect_floats(out.values(), {0, 2, 0, 4});
}

TEST(ReLUForward, ZeroMapsToZero) {
    auto a = make(1, {3}, {0, -0.5f, 0.5f});
    Tensor<float> out(1, {3});
    ReLU<float>::call(DIRECTION::FORWARD, out, a);
    expect_floats(out.values(), {0, 0, 0.5f});
}

TEST(ReLUForward, Batched) {
    auto a = make(2, {3}, {-1, 0, 2, 3, -4, 5});
    Tensor<float> out(2, {3});
    ReLU<float>::call(DIRECTION::FORWARD, out, a);
    expect_floats(out.values(), {0, 0, 2, 3, 0, 5});
}

TEST(ReLUErrors, ShapeMismatch) {
    Tensor<float> a(1, {2, 2});
    Tensor<float> out(1, {4});
    EXPECT_THROW(ReLU<float>::call(DIRECTION::FORWARD, out, a), std::invalid_argument);
}

TEST(ReLUErrors, BatchMismatch) {
    Tensor<float> a(2, {2});
    Tensor<float> out(3, {2});
    EXPECT_THROW(ReLU<float>::call(DIRECTION::FORWARD, out, a), std::invalid_argument);
}

TEST(ReLUErrors, BackwardAlsoValidates) {
    Tensor<float> a(2, {2});
    Tensor<float> out(3, {2});
    EXPECT_THROW(ReLU<float>::call(DIRECTION::BACKWARD, out, a), std::invalid_argument);
}

TEST(ReLUBackward, GradientsPassOnlyWherePositive) {
    auto a = make(1, {4}, {-1, 2, 0, 4});
    Tensor<float> out(1, {4});
    TensorAccess<float>::gradients(out) = {1, 2, 3, 4};
    ReLU<float>::call(DIRECTION::BACKWARD, out, a);
    // relu is not differentiable at 0, the gradient there is taken to be 0
    expect_floats(a.gradients(), {0, 2, 0, 4});
}

TEST(ReLUBackward, GradientsAccumulate) {
    auto a = make(1, {3}, {1, -1, 1});
    Tensor<float> out(1, {3});
    TensorAccess<float>::gradients(out) = {1, 2, 3};
    ReLU<float>::call(DIRECTION::BACKWARD, out, a);
    ReLU<float>::call(DIRECTION::BACKWARD, out, a);
    expect_floats(a.gradients(), {2, 0, 6});
}

TEST(ReLUBackward, BatchedPerElementGradients) {
    auto a = make(2, {3}, {-1, 0, 2, 3, -4, 5});
    Tensor<float> out(2, {3});
    ASSERT_EQ(a.gradients().size(), a.size());
    ASSERT_EQ(out.gradients().size(), out.size());
    TensorAccess<float>::gradients(out) = {1, 2, 3, 4, 5, 6};
    ReLU<float>::call(DIRECTION::BACKWARD, out, a);
    expect_floats(a.gradients(), {0, 0, 3, 4, 0, 6});
}

} // namespace
