#include <stdexcept>
#include <vector>

#include <gtest/gtest.h>

#include "helpers.hpp"

namespace {

using mlfo::tensor::Tensor;
using mlfo::tensor::ops::Add;
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

TEST(AddForward, BatchSizeOne) {
    auto a = make(1, {2, 2}, {1, 2, 3, 4});
    auto b = make(1, {2, 2}, {10, 20, 30, 40});
    Tensor<float> out(1, {2, 2});
    Add<float>::call(DIRECTION::FORWARD, out, a, b);
    expect_floats(out.values(), {11, 22, 33, 44});
}

TEST(AddForward, EqualBatchSizes) {
    auto a = make(3, {2}, {1, 2, 3, 4, 5, 6});
    auto b = make(3, {2}, {10, 20, 30, 40, 50, 60});
    Tensor<float> out(3, {2});
    Add<float>::call(DIRECTION::FORWARD, out, a, b);
    expect_floats(out.values(), {11, 22, 33, 44, 55, 66});
}

TEST(AddForward, BroadcastsB) {
    auto a = make(3, {2}, {1, 2, 3, 4, 5, 6});
    auto b = make(1, {2}, {100, 200});
    Tensor<float> out(3, {2});
    Add<float>::call(DIRECTION::FORWARD, out, a, b);
    expect_floats(out.values(), {101, 202, 103, 204, 105, 206});
}

TEST(AddErrors, ShapeMismatchBetweenAAndB) {
    Tensor<float> a(1, {2, 2});
    Tensor<float> b(1, {4});
    Tensor<float> out(1, {2, 2});
    EXPECT_THROW(Add<float>::call(DIRECTION::FORWARD, out, a, b), std::invalid_argument);
}

TEST(AddErrors, ShapeMismatchWithOut) {
    Tensor<float> a(1, {2, 2});
    Tensor<float> b(1, {2, 2});
    Tensor<float> out(1, {2, 3});
    EXPECT_THROW(Add<float>::call(DIRECTION::FORWARD, out, a, b), std::invalid_argument);
}

TEST(AddErrors, OutBatchDiffersFromA) {
    Tensor<float> a(2, {2});
    Tensor<float> b(2, {2});
    Tensor<float> out(3, {2});
    EXPECT_THROW(Add<float>::call(DIRECTION::FORWARD, out, a, b), std::invalid_argument);
}

TEST(AddErrors, BBatchNeitherEqualNorOne) {
    Tensor<float> a(4, {2});
    Tensor<float> b(2, {2});
    Tensor<float> out(4, {2});
    EXPECT_THROW(Add<float>::call(DIRECTION::FORWARD, out, a, b), std::invalid_argument);
}

TEST(AddErrors, BackwardAlsoValidates) {
    Tensor<float> a(4, {2});
    Tensor<float> b(2, {2});
    Tensor<float> out(4, {2});
    EXPECT_THROW(Add<float>::call(DIRECTION::BACKWARD, out, a, b), std::invalid_argument);
}

TEST(AddBackward, GradientsFlowToBothInputs) {
    Tensor<float> a(1, {2, 2});
    Tensor<float> b(1, {2, 2});
    Tensor<float> out(1, {2, 2});
    TensorAccess<float>::gradients(out) = {1, 2, 3, 4};
    Add<float>::call(DIRECTION::BACKWARD, out, a, b);
    expect_floats(a.gradients(), {1, 2, 3, 4});
    expect_floats(b.gradients(), {1, 2, 3, 4});
}

TEST(AddBackward, GradientsAccumulate) {
    Tensor<float> a(1, {3});
    Tensor<float> b(1, {3});
    Tensor<float> out(1, {3});
    TensorAccess<float>::gradients(out) = {1, 2, 3};
    Add<float>::call(DIRECTION::BACKWARD, out, a, b);
    Add<float>::call(DIRECTION::BACKWARD, out, a, b);
    expect_floats(a.gradients(), {2, 4, 6});
    expect_floats(b.gradients(), {2, 4, 6});
}

TEST(AddBackward, BatchedPerBatchGradients) {
    Tensor<float> a(2, {2});
    Tensor<float> b(2, {2});
    Tensor<float> out(2, {2});
    // gradients are batched, so the op must not run if they are not
    ASSERT_EQ(a.gradients().size(), a.size());
    ASSERT_EQ(b.gradients().size(), b.size());
    ASSERT_EQ(out.gradients().size(), out.size());
    TensorAccess<float>::gradients(out) = {1, 2, 3, 4};
    Add<float>::call(DIRECTION::BACKWARD, out, a, b);
    expect_floats(a.gradients(), {1, 2, 3, 4});
    expect_floats(b.gradients(), {1, 2, 3, 4});
}

TEST(AddBackward, BatchedBroadcastBSumsOverBatch) {
    Tensor<float> a(3, {2});
    Tensor<float> b(1, {2});
    Tensor<float> out(3, {2});
    ASSERT_EQ(a.gradients().size(), a.size());
    ASSERT_EQ(b.gradients().size(), b.size());
    ASSERT_EQ(out.gradients().size(), out.size());
    TensorAccess<float>::gradients(out) = {1, 2, 3, 4, 5, 6};
    Add<float>::call(DIRECTION::BACKWARD, out, a, b);
    expect_floats(a.gradients(), {1, 2, 3, 4, 5, 6});
    // b was broadcast, so its gradient is the sum over the batch
    expect_floats(b.gradients(), {9, 12});
}

} // namespace
