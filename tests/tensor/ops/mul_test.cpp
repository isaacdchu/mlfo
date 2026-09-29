#include <stdexcept>
#include <vector>

#include <gtest/gtest.h>

#include "helpers.hpp"

namespace {

using mlfo::tensor::Tensor;
using mlfo::tensor::ops::DIRECTION;
using mlfo::tensor::ops::Mul;
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

TEST(MulForward, NonSquareMatrices) {
    // [1 2 3; 4 5 6] * [7 8; 9 10; 11 12]
    auto a = make(1, {2, 3}, {1, 2, 3, 4, 5, 6});
    auto b = make(1, {3, 2}, {7, 8, 9, 10, 11, 12});
    Tensor<float> out(1, {2, 2});
    Mul<float>::call(DIRECTION::FORWARD, out, a, b);
    expect_floats(out.values(), {58, 64, 139, 154});
}

TEST(MulForward, RowVectorTimesMatrix) {
    // [1 2 3] * [1 0; 0 1; 2 3]
    auto a = make(1, {1, 3}, {1, 2, 3});
    auto b = make(1, {3, 2}, {1, 0, 0, 1, 2, 3});
    Tensor<float> out(1, {1, 2});
    Mul<float>::call(DIRECTION::FORWARD, out, a, b);
    expect_floats(out.values(), {7, 11});
}

TEST(MulForward, BatchedWithPerBatchB) {
    auto a = make(2, {1, 2}, {1, 2, 3, 4});
    auto b = make(2, {2, 2}, {1, 0, 0, 1, 2, 0, 0, 2});
    Tensor<float> out(2, {1, 2});
    Mul<float>::call(DIRECTION::FORWARD, out, a, b);
    // batch 0: identity; batch 1: scale by 2
    expect_floats(out.values(), {1, 2, 6, 8});
}

TEST(MulForward, BatchedWithBroadcastB) {
    auto a = make(2, {1, 2}, {1, 2, 3, 4});
    auto b = make(1, {2, 2}, {1, 2, 3, 4});
    Tensor<float> out(2, {1, 2});
    Mul<float>::call(DIRECTION::FORWARD, out, a, b);
    // [1 2]*B = [7 10], [3 4]*B = [15 22]
    expect_floats(out.values(), {7, 10, 15, 22});
}

TEST(MulErrors, RankNotTwo) {
    Tensor<float> a3(1, {2, 2, 2});
    Tensor<float> a1(1, {2});
    Tensor<float> m(1, {2, 2});
    Tensor<float> out(1, {2, 2});
    EXPECT_THROW(Mul<float>::call(DIRECTION::FORWARD, out, a3, m), std::invalid_argument);
    EXPECT_THROW(Mul<float>::call(DIRECTION::FORWARD, out, a1, m), std::invalid_argument);
    EXPECT_THROW(Mul<float>::call(DIRECTION::FORWARD, out, m, a1), std::invalid_argument);
    Tensor<float> out1(1, {2});
    EXPECT_THROW(Mul<float>::call(DIRECTION::FORWARD, out1, m, m), std::invalid_argument);
}

TEST(MulErrors, InnerDimensionMismatch) {
    Tensor<float> a(1, {2, 3});
    Tensor<float> b(1, {2, 2});
    Tensor<float> out(1, {2, 2});
    EXPECT_THROW(Mul<float>::call(DIRECTION::FORWARD, out, a, b), std::invalid_argument);
}

TEST(MulErrors, OutShapeMismatch) {
    Tensor<float> a(1, {2, 3});
    Tensor<float> b(1, {3, 4});
    Tensor<float> bad_rows(1, {3, 4});
    Tensor<float> bad_cols(1, {2, 3});
    EXPECT_THROW(Mul<float>::call(DIRECTION::FORWARD, bad_rows, a, b), std::invalid_argument);
    EXPECT_THROW(Mul<float>::call(DIRECTION::FORWARD, bad_cols, a, b), std::invalid_argument);
}

TEST(MulErrors, OutBatchDiffersFromA) {
    Tensor<float> a(2, {2, 2});
    Tensor<float> b(2, {2, 2});
    Tensor<float> out(3, {2, 2});
    EXPECT_THROW(Mul<float>::call(DIRECTION::FORWARD, out, a, b), std::invalid_argument);
}

TEST(MulErrors, BadBBatch) {
    Tensor<float> a(4, {2, 2});
    Tensor<float> b(2, {2, 2});
    Tensor<float> out(4, {2, 2});
    EXPECT_THROW(Mul<float>::call(DIRECTION::FORWARD, out, a, b), std::invalid_argument);
}

TEST(MulBackward, MatchesHandComputedGradients) {
    // A (2,3), B (3,2), dOut (2,2)
    auto a = make(1, {2, 3}, {1, 2, 3, 4, 5, 6});
    auto b = make(1, {3, 2}, {7, 8, 9, 10, 11, 12});
    Tensor<float> out(1, {2, 2});
    TensorAccess<float>::gradients(out) = {1, 2, 3, 4};
    Mul<float>::call(DIRECTION::BACKWARD, out, a, b);
    // dA = dOut * B^T = [1 2; 3 4] * [7 9 11; 8 10 12]
    expect_floats(a.gradients(), {23, 29, 35, 53, 67, 81});
    // dB = A^T * dOut = [1 4; 2 5; 3 6] * [1 2; 3 4]
    expect_floats(b.gradients(), {13, 18, 17, 24, 21, 30});
}

TEST(MulBackward, GradientsAccumulate) {
    auto a = make(1, {1, 2}, {1, 2});
    auto b = make(1, {2, 1}, {3, 4});
    Tensor<float> out(1, {1, 1});
    TensorAccess<float>::gradients(out) = {2};
    Mul<float>::call(DIRECTION::BACKWARD, out, a, b);
    expect_floats(a.gradients(), {6, 8});
    expect_floats(b.gradients(), {2, 4});
    Mul<float>::call(DIRECTION::BACKWARD, out, a, b);
    expect_floats(a.gradients(), {12, 16});
    expect_floats(b.gradients(), {4, 8});
}

TEST(MulBackward, BatchedPerBatchB) {
    auto a = make(2, {1, 2}, {1, 2, 3, 4});
    auto b = make(2, {2, 2}, {1, 0, 0, 1, 2, 0, 0, 2});
    Tensor<float> out(2, {1, 2});
    // gradients are batched, so the op must not run if they are not
    ASSERT_EQ(a.gradients().size(), a.size());
    ASSERT_EQ(b.gradients().size(), b.size());
    ASSERT_EQ(out.gradients().size(), out.size());
    TensorAccess<float>::gradients(out) = {1, 1, 1, 2};
    Mul<float>::call(DIRECTION::BACKWARD, out, a, b);
    // batch 0: dA = [1 1] * I = [1 1], dB = [1; 2] * [1 1]
    // batch 1: dA = [1 2] * 2I = [2 4], dB = [3; 4] * [1 2]
    expect_floats(a.gradients(), {1, 1, 2, 4});
    expect_floats(b.gradients(), {1, 1, 2, 2, 3, 6, 4, 8});
}

TEST(MulBackward, BatchedBroadcastBSumsOverBatch) {
    auto a = make(2, {1, 2}, {1, 2, 3, 4});
    auto b = make(1, {2, 2}, {1, 2, 3, 4});
    Tensor<float> out(2, {1, 2});
    ASSERT_EQ(a.gradients().size(), a.size());
    ASSERT_EQ(b.gradients().size(), b.size());
    ASSERT_EQ(out.gradients().size(), out.size());
    TensorAccess<float>::gradients(out) = {1, 2, 3, 1};
    Mul<float>::call(DIRECTION::BACKWARD, out, a, b);
    // dA per batch = dOut * B^T, with B^T = [1 3; 2 4]
    // batch 0: [1 2] * B^T = [5 11], batch 1: [3 1] * B^T = [5 13]
    expect_floats(a.gradients(), {5, 11, 5, 13});
    // b was broadcast, so dB = [1; 2] * [1 2] + [3; 4] * [3 1]
    expect_floats(b.gradients(), {10, 5, 14, 8});
}

} // namespace
