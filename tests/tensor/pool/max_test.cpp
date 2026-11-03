#include <stdexcept>
#include <vector>

#include <gtest/gtest.h>

#include "helpers.hpp"

namespace {

using mlfo::tensor::Tensor;
using mlfo::tensor::pool::Max;
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

TEST(MaxOutputSize, ExactFit) {
    EXPECT_EQ(Max<float>::output_size(4, 2, 2), 2u);
    EXPECT_EQ(Max<float>::output_size(4, 2, 1), 3u);
}

TEST(MaxOutputSize, PartialLastWindow) {
    // windows start at 0, 2, 4; the last one only covers index 4
    EXPECT_EQ(Max<float>::output_size(5, 2, 2), 3u);
}

TEST(MaxOutputSize, WindowLargerThanInput) {
    EXPECT_EQ(Max<float>::output_size(2, 3, 1), 1u);
}

TEST(MaxOutputSize, NoWindowStartsPastTheEnd) {
    // windows start at 0, 3; a window at 6 would start outside the input
    EXPECT_EQ(Max<float>::output_size(5, 1, 3), 2u);
}

TEST(MaxForward, NonOverlapping) {
    auto a = make(1, {4, 4}, {
        1, 2, 5, 6,
        3, 4, 7, 8,
        9, 1, 2, 3,
        1, 0, 4, 1
    });
    Tensor<float> out(1, {2, 2});
    Max<float>::call(DIRECTION::FORWARD, out, a, 2, 2, 2, 2);
    expect_floats(out.values(), {4, 8, 9, 4});
}

TEST(MaxForward, OverlappingStride1) {
    auto a = make(1, {3, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9});
    Tensor<float> out(1, {2, 2});
    Max<float>::call(DIRECTION::FORWARD, out, a, 2, 2, 1, 1);
    expect_floats(out.values(), {5, 6, 8, 9});
}

TEST(MaxForward, PartialWindowsOnlyUseValidElements) {
    // 3x3 input, 2x2 window, stride 2 -> 2x2 output, the right and bottom windows are clipped
    auto a = make(1, {3, 3}, {-1, -2, -3, -4, -5, -6, -7, -8, -9});
    Tensor<float> out(1, {2, 2});
    Max<float>::call(DIRECTION::FORWARD, out, a, 2, 2, 2, 2);
    expect_floats(out.values(), {-1, -3, -7, -9});
}

TEST(MaxForward, ChannelsAndBatches) {
    // 2 batches of 2x2 with 2 channels, pooled to 1x1
    // channels last, so each pair of values is one position
    auto a = make(2, {2, 2, 2}, {1, 8, 2, 7, 3, 6, 4, 5, -1, 0, -2, 9, -3, 0, -4, 0});
    Tensor<float> out(2, {1, 1, 2});
    Max<float>::call(DIRECTION::FORWARD, out, a, 2, 2, 2, 2);
    expect_floats(out.values(), {4, 8, -1, 9});
}

TEST(MaxBackward, RoutesGradientPerChannel) {
    auto a = make(1, {2, 2, 2}, {1, 8, 2, 7, 3, 6, 4, 5});
    Tensor<float> out(1, {1, 1, 2});
    TensorAccess<float>::gradients(out) = {10, 20};
    Max<float>::call(DIRECTION::BACKWARD, out, a, 2, 2, 2, 2);
    // channel 0's max is 4 at (1, 1), channel 1's max is 8 at (0, 0)
    expect_floats(a.gradients(), {0, 20, 0, 0, 0, 0, 10, 0});
}

TEST(MaxForward, OneByOneWindowWithStrideSubsamples) {
    // a 1x1 window picks every stride-th element
    auto a = make(1, {4, 4}, {
        1, 2, 3, 4,
        5, 6, 7, 8,
        9, 10, 11, 12,
        13, 14, 15, 16
    });
    Tensor<float> out(1, {2, 2});
    Max<float>::call(DIRECTION::FORWARD, out, a, 1, 1, 2, 2);
    expect_floats(out.values(), {1, 3, 9, 11});
}

TEST(MaxForward, OneByOneWindowStride1IsIdentity) {
    auto a = make(1, {2, 3}, {1, -2, 3, -4, 5, -6});
    Tensor<float> out(1, {2, 3});
    Max<float>::call(DIRECTION::FORWARD, out, a, 1, 1, 1, 1);
    expect_floats(out.values(), {1, -2, 3, -4, 5, -6});
}

TEST(MaxBackward, RoutesGradientToMax) {
    auto a = make(1, {2, 4}, {1, 3, 2, 0, 2, 0, 1, 5});
    Tensor<float> out(1, {1, 2});
    TensorAccess<float>::gradients(out) = {10, 20};
    Max<float>::call(DIRECTION::BACKWARD, out, a, 2, 2, 2, 2);
    expect_floats(a.gradients(), {0, 10, 0, 0, 0, 0, 0, 20});
}

TEST(MaxBackward, OneByOneWindowWithStrideRoutesToSampledElements) {
    auto a = make(1, {3, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9});
    Tensor<float> out(1, {2, 2});
    TensorAccess<float>::gradients(out) = {1, 2, 3, 4};
    Max<float>::call(DIRECTION::BACKWARD, out, a, 1, 1, 2, 2);
    expect_floats(a.gradients(), {1, 0, 2, 0, 0, 0, 3, 0, 4});
}

TEST(MaxBackward, OverlappingWindowsAccumulate) {
    // 9 is the max of all 4 windows
    auto a = make(1, {3, 3}, {1, 2, 3, 4, 9, 6, 7, 8, 5});
    Tensor<float> out(1, {2, 2});
    TensorAccess<float>::gradients(out) = {1, 2, 3, 4};
    Max<float>::call(DIRECTION::BACKWARD, out, a, 2, 2, 1, 1);
    expect_floats(a.gradients(), {0, 0, 0, 0, 10, 0, 0, 0, 0});
}

TEST(MaxBackward, TiesGoToFirstElement) {
    auto a = make(1, {2, 2}, {3, 3, 3, 3});
    Tensor<float> out(1, {1, 1});
    TensorAccess<float>::gradients(out) = {1};
    Max<float>::call(DIRECTION::BACKWARD, out, a, 2, 2, 2, 2);
    expect_floats(a.gradients(), {1, 0, 0, 0});
}

TEST(MaxBackward, AccumulatesIntoExistingGradients) {
    auto a = make(1, {2, 2}, {1, 4, 2, 3});
    Tensor<float> out(1, {1, 1});
    TensorAccess<float>::gradients(out) = {5};
    TensorAccess<float>::gradients(a) = {1, 1, 1, 1};
    Max<float>::call(DIRECTION::BACKWARD, out, a, 2, 2, 2, 2);
    expect_floats(a.gradients(), {1, 6, 1, 1});
}

TEST(MaxIndices, ForwardRecordsWhereEachMaxCameFrom) {
    // 2 batches of 3x3 with 2 channels, 2x2 window, stride 2, so the right and bottom are clipped
    std::vector<float> values(2 * 3 * 3 * 2);
    for (std::size_t i = 0; i < values.size(); i++) {
        values[i] = static_cast<float>((i * 7) % 11);
    }
    auto a = make(2, {3, 3, 2}, values);
    Tensor<float> out(2, {2, 2, 2});
    std::vector<std::size_t> indices;
    Max<float>::call(DIRECTION::FORWARD, out, a, indices, 2, 2, 2, 2);
    ASSERT_EQ(indices.size(), out.size());
    for (std::size_t i = 0; i < indices.size(); i++) {
        EXPECT_FLOAT_EQ(a.values()[indices[i]], out.values()[i]) << "index " << i;
    }
}

TEST(MaxIndices, BackwardMatchesRecomputingTheMaxes) {
    std::vector<float> values(2 * 5 * 4 * 3);
    for (std::size_t i = 0; i < values.size(); i++) {
        values[i] = static_cast<float>((i * 5) % 7);  // plenty of ties
    }
    std::vector<float> out_gradients(2 * 2 * 2 * 3);
    for (std::size_t i = 0; i < out_gradients.size(); i++) {
        out_gradients[i] = static_cast<float>(i + 1);
    }
    // 3x3 window with stride 2 overlaps, so some elements receive several gradients
    auto cached = make(2, {5, 4, 3}, values);
    Tensor<float> cached_out(2, {2, 2, 3});
    TensorAccess<float>::gradients(cached_out) = out_gradients;
    std::vector<std::size_t> indices;
    Max<float>::call(DIRECTION::FORWARD, cached_out, cached, indices, 3, 3, 2, 2);
    Max<float>::call(DIRECTION::BACKWARD, cached_out, cached, indices, 3, 3, 2, 2);

    auto recomputed = make(2, {5, 4, 3}, values);
    Tensor<float> recomputed_out(2, {2, 2, 3});
    TensorAccess<float>::gradients(recomputed_out) = out_gradients;
    Max<float>::call(DIRECTION::BACKWARD, recomputed_out, recomputed, 3, 3, 2, 2);
    expect_floats(cached.gradients(), recomputed.gradients());
}

TEST(MaxIndices, TiesGoToFirstElement) {
    auto a = make(1, {2, 2, 1}, {3, 3, 3, 3});
    Tensor<float> out(1, {1, 1, 1});
    TensorAccess<float>::gradients(out) = {1};
    std::vector<std::size_t> indices;
    Max<float>::call(DIRECTION::FORWARD, out, a, indices, 2, 2, 2, 2);
    Max<float>::call(DIRECTION::BACKWARD, out, a, indices, 2, 2, 2, 2);
    expect_floats(a.gradients(), {1, 0, 0, 0});
}

TEST(MaxIndices, BackwardBeforeForwardThrows) {
    Tensor<float> a(1, {2, 2});
    Tensor<float> out(1, {1, 1});
    std::vector<std::size_t> indices;
    EXPECT_THROW(
        Max<float>::call(DIRECTION::BACKWARD, out, a, indices, 2, 2, 2, 2),
        std::logic_error
    );
}

TEST(MaxCall, RejectsWrongOutputShape) {
    Tensor<float> a(1, {4, 4});
    Tensor<float> out(1, {3, 3});
    EXPECT_THROW(Max<float>::call(DIRECTION::FORWARD, out, a, 2, 2, 2, 2), std::invalid_argument);
}

TEST(MaxCall, RejectsChannelMismatch) {
    Tensor<float> a(1, {4, 4, 2});
    Tensor<float> out(1, {2, 2, 3});
    EXPECT_THROW(Max<float>::call(DIRECTION::FORWARD, out, a, 2, 2, 2, 2), std::invalid_argument);
}

TEST(MaxCall, RejectsBadParameters) {
    Tensor<float> a(1, {4, 4});
    Tensor<float> out(1, {2, 2});
    EXPECT_THROW(Max<float>::call(DIRECTION::FORWARD, out, a, 0, 2, 2, 2), std::invalid_argument);
    EXPECT_THROW(Max<float>::call(DIRECTION::FORWARD, out, a, 2, 2, 0, 2), std::invalid_argument);
}

} // namespace
