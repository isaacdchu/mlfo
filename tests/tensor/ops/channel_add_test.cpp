#include <stdexcept>
#include <vector>

#include <gtest/gtest.h>

#include "helpers.hpp"

namespace {

using mlfo::tensor::Tensor;
using mlfo::tensor::ops::ChannelAdd;
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

TEST(ChannelAddForward, BroadcastsOverEachChannel) {
    auto a = make(1, {2, 2, 2}, {1, 2, 3, 4, 5, 6, 7, 8});
    auto b = make(1, {2}, {10, 100});
    Tensor<float> out(1, {2, 2, 2});
    ChannelAdd<float>::call(DIRECTION::FORWARD, out, a, b);
    expect_floats(out.values(), {11, 12, 13, 14, 105, 106, 107, 108});
}

TEST(ChannelAddForward, BroadcastsOverBatch) {
    auto a = make(2, {2, 1}, {1, 2, 3, 4});
    auto b = make(1, {2}, {10, 100});
    Tensor<float> out(2, {2, 1});
    ChannelAdd<float>::call(DIRECTION::FORWARD, out, a, b);
    expect_floats(out.values(), {11, 102, 13, 104});
}

TEST(ChannelAddBackward, SumsGradientsPerChannel) {
    auto a = make(2, {2, 2}, {0, 0, 0, 0, 0, 0, 0, 0});
    auto b = make(1, {2}, {0, 0});
    Tensor<float> out(2, {2, 2});
    TensorAccess<float>::gradients(out) = {1, 2, 3, 4, 5, 6, 7, 8};
    ChannelAdd<float>::call(DIRECTION::BACKWARD, out, a, b);
    expect_floats(a.gradients(), {1, 2, 3, 4, 5, 6, 7, 8});
    // channel 0 gets 1 + 2 + 5 + 6, channel 1 gets 3 + 4 + 7 + 8
    expect_floats(b.gradients(), {14, 22});
}

TEST(ChannelAddErrors, ShapeMismatchWithOut) {
    Tensor<float> a(1, {2, 2});
    Tensor<float> b(1, {2});
    Tensor<float> out(1, {4});
    EXPECT_THROW(ChannelAdd<float>::call(DIRECTION::FORWARD, out, a, b), std::invalid_argument);
}

TEST(ChannelAddErrors, BDoesNotMatchChannels) {
    Tensor<float> a(1, {2, 2});
    Tensor<float> b(1, {3});
    Tensor<float> out(1, {2, 2});
    EXPECT_THROW(ChannelAdd<float>::call(DIRECTION::FORWARD, out, a, b), std::invalid_argument);
}

TEST(ChannelAddErrors, BNotOneDimensional) {
    Tensor<float> a(1, {2, 2});
    Tensor<float> b(1, {2, 1});
    Tensor<float> out(1, {2, 2});
    EXPECT_THROW(ChannelAdd<float>::call(DIRECTION::FORWARD, out, a, b), std::invalid_argument);
}

TEST(ChannelAddErrors, BBatched) {
    Tensor<float> a(2, {2, 2});
    Tensor<float> b(2, {2});
    Tensor<float> out(2, {2, 2});
    EXPECT_THROW(ChannelAdd<float>::call(DIRECTION::FORWARD, out, a, b), std::invalid_argument);
}

} // namespace
