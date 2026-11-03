#include <cstddef>
#include <vector>

#include <gtest/gtest.h>

#include <mlfo/misc/im2col.hpp>

namespace {

using mlfo::misc::Conv2DGeometry;

void expect_floats(const std::vector<float>& actual, const std::vector<float>& expected) {
    ASSERT_EQ(actual.size(), expected.size());
    for (std::size_t i = 0; i < expected.size(); i++) {
        EXPECT_FLOAT_EQ(actual[i], expected[i]) << "index " << i;
    }
}

TEST(Im2col, RowsArePatchesInRSCOrder) {
    // (h, w, c) = (2, 3, 2), 2x2 kernel -> (p, q) = (1, 2)
    const Conv2DGeometry g{2, 3, 2, 2, 2, 1, 2, 1, 1, 0, 0};
    EXPECT_EQ(g.patch_size(), 8u);
    EXPECT_EQ(g.positions(), 2u);
    // value = 10 * position + channel
    const std::vector<float> input = {0, 1, 10, 11, 20, 21, 30, 31, 40, 41, 50, 51};
    std::vector<float> col(g.positions() * g.patch_size(), -1);
    mlfo::misc::im2col(g, input.data(), col.data());
    expect_floats(col, {
        0, 1, 10, 11, 30, 31, 40, 41,
        10, 11, 20, 21, 40, 41, 50, 51
    });
}

TEST(Im2col, PaddingIsZero) {
    // 1x1 input padded by 1, 2x2 kernel -> 2x2 output, each patch holds the input once
    const Conv2DGeometry g{1, 1, 1, 2, 2, 2, 2, 1, 1, 1, 1};
    const std::vector<float> input = {5};
    std::vector<float> col(g.positions() * g.patch_size(), -1);
    mlfo::misc::im2col(g, input.data(), col.data());
    expect_floats(col, {
        0, 0, 0, 5,
        0, 0, 5, 0,
        0, 5, 0, 0,
        5, 0, 0, 0
    });
}

TEST(Im2col, Stride) {
    // 1x4 input, 1x1 kernel, stride 2 -> picks every second value
    const Conv2DGeometry g{1, 4, 1, 1, 1, 1, 2, 1, 2, 0, 0};
    EXPECT_FALSE(g.is_pointwise());
    const std::vector<float> input = {1, 2, 3, 4};
    std::vector<float> col(2);
    mlfo::misc::im2col(g, input.data(), col.data());
    expect_floats(col, {1, 3});
}

TEST(Im2col, KernelRowsClippedOnBothSides) {
    // 1x2 input with 2 channels, padding 2 on each side, 1x5 kernel, stride 2 -> 1x1 output
    // the kernel covers padded columns 0..4 and the input is columns 2 and 3
    const Conv2DGeometry g{1, 2, 2, 1, 5, 1, 1, 1, 2, 0, 2};
    EXPECT_EQ(g.columns_inside(0).first, 2u);
    EXPECT_EQ(g.columns_inside(0).last, 4u);
    const std::vector<float> input = {1, 2, 3, 4};
    std::vector<float> col(g.patch_size(), -1);
    mlfo::misc::im2col(g, input.data(), col.data());
    expect_floats(col, {0, 0, 0, 0, 1, 2, 3, 4, 0, 0});
}

TEST(Im2col, WindowEntirelyInPadding) {
    // 1x1 input, padding 3 on the left, 1x2 kernel, stride 3 -> windows at padded columns 0 and 3
    // the first window is all padding, the second covers the input then padding
    const Conv2DGeometry g{1, 1, 1, 1, 2, 1, 2, 1, 3, 0, 3};
    EXPECT_EQ(g.columns_inside(0).first, g.columns_inside(0).last);
    const std::vector<float> input = {7};
    std::vector<float> col(g.positions() * g.patch_size(), -1);
    mlfo::misc::im2col(g, input.data(), col.data());
    expect_floats(col, {0, 0, 7, 0});
    std::vector<float> back(1, 0);
    mlfo::misc::col2im(g, std::vector<float>{1, 2, 3, 4}.data(), back.data());
    expect_floats(back, {3});
}

TEST(Col2im, SumsOverlappingWindowsAndDropsPadding) {
    // 3x3 input, 2x2 kernel, stride 1 -> 2x2 output, all ones
    // each input element receives one per window that covers it
    const Conv2DGeometry g{3, 3, 1, 2, 2, 2, 2, 1, 1, 0, 0};
    const std::vector<float> col(g.positions() * g.patch_size(), 1);
    std::vector<float> input(9, 0);
    mlfo::misc::col2im(g, col.data(), input.data());
    expect_floats(input, {1, 2, 1, 2, 4, 2, 1, 2, 1});
}

TEST(Col2im, IsTheAdjointOfIm2col) {
    // <im2col(x), y> == <x, col2im(y)> for any x and y
    const Conv2DGeometry g{4, 5, 2, 3, 2, 2, 3, 2, 2, 1, 1};
    std::vector<float> x(g.h * g.w * g.c);
    for (std::size_t i = 0; i < x.size(); i++) {
        x[i] = static_cast<float>(i % 7) - 3;
    }
    std::vector<float> y(g.positions() * g.patch_size());
    for (std::size_t i = 0; i < y.size(); i++) {
        y[i] = static_cast<float>(i % 5) - 2;
    }
    std::vector<float> col(y.size());
    mlfo::misc::im2col(g, x.data(), col.data());
    std::vector<float> back(x.size(), 0);
    mlfo::misc::col2im(g, y.data(), back.data());
    double lhs = 0;
    for (std::size_t i = 0; i < y.size(); i++) {
        lhs += col[i] * y[i];
    }
    double rhs = 0;
    for (std::size_t i = 0; i < x.size(); i++) {
        rhs += x[i] * back[i];
    }
    EXPECT_DOUBLE_EQ(lhs, rhs);
}

TEST(Conv2DGeometry, Pointwise) {
    EXPECT_TRUE((Conv2DGeometry{4, 4, 3, 1, 1, 4, 4, 1, 1, 0, 0}.is_pointwise()));
    EXPECT_FALSE((Conv2DGeometry{4, 4, 3, 1, 1, 6, 6, 1, 1, 1, 1}.is_pointwise()));
    EXPECT_FALSE((Conv2DGeometry{4, 4, 3, 2, 1, 3, 4, 1, 1, 0, 0}.is_pointwise()));
}

} // namespace
