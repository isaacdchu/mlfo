#include <stdexcept>
#include <vector>

#include <gtest/gtest.h>

#include "helpers.hpp"

namespace {

using mlfo::tensor::Tensor;
using mlfo::tensor::ops::Conv2D;
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

void expect_near(const std::vector<float>& actual, const std::vector<float>& expected) {
    ASSERT_EQ(actual.size(), expected.size());
    for (std::size_t i = 0; i < expected.size(); i++) {
        EXPECT_NEAR(actual[i], expected[i], 1e-4) << "index " << i;
    }
}

std::vector<float> pattern(std::size_t size, std::size_t multiplier, std::size_t modulus) {
    std::vector<float> values(size);
    for (std::size_t i = 0; i < size; i++) {
        values[i] = static_cast<float>((i * multiplier) % modulus) / (modulus / 2.0f) - 1.0f;
    }
    return values;
}

struct Config {
    std::size_t batches;
    bool broadcast_kernel;
    std::size_t h, w, c, r, s, k;
    std::size_t stride_h, stride_w, padding_h, padding_w;

    std::size_t p() const { return (h + 2 * padding_h - r) / stride_h + 1; }
    std::size_t q() const { return (w + 2 * padding_w - s) / stride_w + 1; }
};

// direct loop convolution, the definition the gemm version must match
// returns the output, and the input and kernel gradients for the given output gradients
struct Reference {
    std::vector<float> out;
    std::vector<float> input_gradients;
    std::vector<float> kernel_gradients;
};

Reference reference(
    const Config& cfg,
    const std::vector<float>& input,
    const std::vector<float>& kernel,
    const std::vector<float>& out_gradients
) {
    const std::size_t p = cfg.p();
    const std::size_t q = cfg.q();
    const std::size_t input_size = cfg.h * cfg.w * cfg.c;
    const std::size_t kernel_size = cfg.r * cfg.s * cfg.c * cfg.k;
    const std::size_t out_size = p * q * cfg.k;
    Reference result{
        std::vector<float>(cfg.batches * out_size),
        std::vector<float>(input.size()),
        std::vector<float>(kernel.size())
    };
    for (std::size_t n = 0; n < cfg.batches; n++) {
        const std::size_t kernel_offset = cfg.broadcast_kernel ? 0 : n * kernel_size;
        for (std::size_t pi = 0; pi < p; pi++) {
            for (std::size_t qi = 0; qi < q; qi++) {
                for (std::size_t ki = 0; ki < cfg.k; ki++) {
                    const std::size_t out_index = n * out_size + (pi * q + qi) * cfg.k + ki;
                    for (std::size_t ri = 0; ri < cfg.r; ri++) {
                        for (std::size_t si = 0; si < cfg.s; si++) {
                            const std::size_t hi = pi * cfg.stride_h + ri;
                            const std::size_t wi = qi * cfg.stride_w + si;
                            if (
                                hi < cfg.padding_h || hi - cfg.padding_h >= cfg.h ||
                                wi < cfg.padding_w || wi - cfg.padding_w >= cfg.w
                            ) {
                                continue;
                            }
                            for (std::size_t ci = 0; ci < cfg.c; ci++) {
                                const std::size_t input_index = n * input_size + (
                                    ((hi - cfg.padding_h) * cfg.w + (wi - cfg.padding_w)) * cfg.c + ci
                                );
                                const std::size_t kernel_index = kernel_offset + (
                                    ((ri * cfg.s + si) * cfg.c + ci) * cfg.k + ki
                                );
                                result.out[out_index] += input[input_index] * kernel[kernel_index];
                                result.input_gradients[input_index] += (
                                    out_gradients[out_index] * kernel[kernel_index]
                                );
                                result.kernel_gradients[kernel_index] += (
                                    out_gradients[out_index] * input[input_index]
                                );
                            }
                        }
                    }
                }
            }
        }
    }
    return result;
}

void expect_matches_reference(const Config& cfg) {
    const std::size_t kernel_batches = cfg.broadcast_kernel ? 1 : cfg.batches;
    auto input = make(cfg.batches, {cfg.h, cfg.w, cfg.c}, pattern(cfg.batches * cfg.h * cfg.w * cfg.c, 7, 11));
    auto kernel = make(
        kernel_batches, {cfg.r, cfg.s, cfg.c, cfg.k},
        pattern(kernel_batches * cfg.r * cfg.s * cfg.c * cfg.k, 5, 13)
    );
    Tensor<float> out(cfg.batches, {cfg.p(), cfg.q(), cfg.k});
    const auto out_gradients = pattern(out.size(), 3, 7);
    TensorAccess<float>::gradients(out) = out_gradients;
    Conv2D<float>::call(
        DIRECTION::FORWARD, out, input, kernel,
        cfg.stride_h, cfg.stride_w, cfg.padding_h, cfg.padding_w
    );
    Conv2D<float>::call(
        DIRECTION::BACKWARD, out, input, kernel,
        cfg.stride_h, cfg.stride_w, cfg.padding_h, cfg.padding_w
    );
    const auto expected = reference(cfg, input.values(), kernel.values(), out_gradients);
    expect_near(out.values(), expected.out);
    expect_near(input.gradients(), expected.input_gradients);
    expect_near(kernel.gradients(), expected.kernel_gradients);
}

TEST(Conv2DForward, SingleChannelNoPadding) {
    // 3x3 input, 2x2 kernel -> 2x2 output
    auto input = make(1, {3, 3, 1}, {1, 2, 3, 4, 5, 6, 7, 8, 9});
    auto kernel = make(1, {2, 2, 1, 1}, {1, 0, 0, 1});
    Tensor<float> out(1, {2, 2, 1});
    Conv2D<float>::call(DIRECTION::FORWARD, out, input, kernel, 1, 1, 0, 0);
    expect_floats(out.values(), {6, 8, 12, 14});
}

TEST(Conv2DForward, PaddingAndStride) {
    // 3x3 input padded to 5x5, 3x3 kernel of ones, stride 2 -> 2x2 output
    auto input = make(1, {3, 3, 1}, {1, 2, 3, 4, 5, 6, 7, 8, 9});
    auto kernel = make(1, {3, 3, 1, 1}, std::vector<float>(9, 1));
    Tensor<float> out(1, {2, 2, 1});
    Conv2D<float>::call(DIRECTION::FORWARD, out, input, kernel, 2, 2, 1, 1);
    // each output sums the 2x2 corner block of the input
    expect_floats(out.values(), {12, 16, 24, 28});
}

TEST(Conv2DForward, MultiChannelMultiKernel) {
    // 2x2 with 2 channels, 2 kernels of 2x2x2 -> 1x1 with 2 channels
    // channel 0 is 1, 2, 3, 4 and channel 1 is 5, 6, 7, 8, interleaved per position
    auto input = make(1, {2, 2, 2}, {1, 5, 2, 6, 3, 7, 4, 8});
    // (r, s, c, k): kernel 0 sums channel 0, kernel 1 sums the diagonal of channel 1
    auto kernel = make(1, {2, 2, 2, 2}, {
        1, 0, 0, 1,
        1, 0, 0, 0,
        1, 0, 0, 0,
        1, 0, 0, 1
    });
    Tensor<float> out(1, {1, 1, 2});
    Conv2D<float>::call(DIRECTION::FORWARD, out, input, kernel, 1, 1, 0, 0);
    expect_floats(out.values(), {10, 13});
}

TEST(Conv2DForward, OutputChannelsAreInterleaved) {
    // 1x2 input with 1 channel, 1x1 kernel with 3 output channels -> 1x2 output with 3 channels
    auto input = make(1, {1, 2, 1}, {1, 2});
    auto kernel = make(1, {1, 1, 1, 3}, {1, 10, 100});
    Tensor<float> out(1, {1, 2, 3});
    Conv2D<float>::call(DIRECTION::FORWARD, out, input, kernel, 1, 1, 0, 0);
    expect_floats(out.values(), {1, 10, 100, 2, 20, 200});
}

TEST(Conv2DForward, BatchedWithBroadcastKernel) {
    auto input = make(2, {2, 2, 1}, {1, 2, 3, 4, 5, 6, 7, 8});
    auto kernel = make(1, {2, 2, 1, 1}, {1, 1, 1, 1});
    Tensor<float> out(2, {1, 1, 1});
    Conv2D<float>::call(DIRECTION::FORWARD, out, input, kernel, 1, 1, 0, 0);
    expect_floats(out.values(), {10, 26});
}

TEST(Conv2DBackward, MatchesNumericalGradient) {
    // compares against central differences of loss = sum(out * out_gradients)
    const std::size_t stride = 2;
    const std::size_t padding = 1;
    std::vector<float> input_values(2 * 2 * 4 * 5);
    for (std::size_t i = 0; i < input_values.size(); i++) {
        input_values[i] = static_cast<float>((i * 7) % 11) / 5.0f - 1.0f;
    }
    std::vector<float> kernel_values(3 * 2 * 3 * 2);
    for (std::size_t i = 0; i < kernel_values.size(); i++) {
        kernel_values[i] = static_cast<float>((i * 5) % 13) / 6.0f - 1.0f;
    }
    // input (h, w, c) = (4, 5, 2), kernel (r, s, c, k) = (3, 2, 2, 3)
    auto input = make(2, {4, 5, 2}, input_values);
    auto kernel = make(1, {3, 2, 2, 3}, kernel_values);
    // p = (4 + 2 - 3) / 2 + 1 = 2, q = (5 + 2 - 2) / 2 + 1 = 3
    Tensor<float> out(2, {2, 3, 3});
    std::vector<float> out_gradients(out.size());
    for (std::size_t i = 0; i < out_gradients.size(); i++) {
        out_gradients[i] = static_cast<float>((i * 3) % 7) / 3.0f - 1.0f;
    }
    TensorAccess<float>::gradients(out) = out_gradients;
    Conv2D<float>::call(DIRECTION::FORWARD, out, input, kernel, stride, stride, padding, padding);
    Conv2D<float>::call(DIRECTION::BACKWARD, out, input, kernel, stride, stride, padding, padding);

    auto loss = [&](const std::vector<float>& in, const std::vector<float>& ker) {
        auto i = make(2, {4, 5, 2}, in);
        auto k = make(1, {3, 2, 2, 3}, ker);
        Tensor<float> o(2, {2, 3, 3});
        Conv2D<float>::call(DIRECTION::FORWARD, o, i, k, stride, stride, padding, padding);
        double total = 0;
        for (std::size_t j = 0; j < o.size(); j++) {
            total += o.values()[j] * out_gradients[j];
        }
        return total;
    };
    const float eps = 1e-2f;
    for (std::size_t i = 0; i < input_values.size(); i++) {
        auto plus = input_values;
        auto minus = input_values;
        plus[i] += eps;
        minus[i] -= eps;
        const double numeric = (loss(plus, kernel_values) - loss(minus, kernel_values)) / (2 * eps);
        EXPECT_NEAR(input.gradients()[i], numeric, 1e-3) << "input index " << i;
    }
    for (std::size_t i = 0; i < kernel_values.size(); i++) {
        auto plus = kernel_values;
        auto minus = kernel_values;
        plus[i] += eps;
        minus[i] -= eps;
        const double numeric = (loss(input_values, plus) - loss(input_values, minus)) / (2 * eps);
        EXPECT_NEAR(kernel.gradients()[i], numeric, 1e-3) << "kernel index " << i;
    }
}

TEST(Conv2DBackward, AccumulatesIntoExistingGradients) {
    auto input = make(1, {2, 2, 1}, {1, 2, 3, 4});
    auto kernel = make(1, {2, 2, 1, 1}, {1, 2, 3, 4});
    Tensor<float> out(1, {1, 1, 1});
    TensorAccess<float>::gradients(out) = {1};
    TensorAccess<float>::gradients(input) = {1, 1, 1, 1};
    Conv2D<float>::call(DIRECTION::BACKWARD, out, input, kernel, 1, 1, 0, 0);
    expect_floats(input.gradients(), {2, 3, 4, 5});
    expect_floats(kernel.gradients(), {1, 2, 3, 4});
}

TEST(Conv2DReference, BroadcastKernelFoldsBatch) {
    expect_matches_reference({3, true, 6, 5, 3, 3, 2, 4, 1, 1, 0, 0});
}

TEST(Conv2DReference, PerSampleKernels) {
    expect_matches_reference({3, false, 6, 5, 3, 3, 2, 4, 1, 1, 0, 0});
}

TEST(Conv2DReference, StrideAndPadding) {
    expect_matches_reference({2, true, 7, 6, 2, 3, 3, 3, 2, 3, 1, 2});
    expect_matches_reference({2, false, 7, 6, 2, 3, 3, 3, 2, 3, 1, 2});
}

TEST(Conv2DReference, KernelLargerThanInputWithPadding) {
    expect_matches_reference({2, true, 2, 2, 1, 4, 4, 2, 1, 1, 1, 1});
}

TEST(Conv2DReference, PointwiseSkipsIm2col) {
    // a 1x1 kernel with stride 1 and no padding uses the input as the column matrix
    expect_matches_reference({2, true, 4, 3, 5, 1, 1, 3, 1, 1, 0, 0});
    expect_matches_reference({2, false, 4, 3, 5, 1, 1, 3, 1, 1, 0, 0});
}

TEST(Conv2DReference, OneByOneKernelWithStrideOrPadding) {
    // 1x1 kernels that still need im2col
    expect_matches_reference({2, true, 5, 5, 2, 1, 1, 3, 2, 2, 0, 0});
    expect_matches_reference({2, true, 3, 3, 2, 1, 1, 3, 1, 1, 1, 1});
}

TEST(Conv2DForward, OverwritesPreviousOutput) {
    auto input = make(1, {2, 2, 1}, {1, 2, 3, 4});
    auto kernel = make(1, {2, 2, 1, 1}, {1, 1, 1, 1});
    Tensor<float> out(1, {1, 1, 1});
    Conv2D<float>::call(DIRECTION::FORWARD, out, input, kernel, 1, 1, 0, 0);
    Conv2D<float>::call(DIRECTION::FORWARD, out, input, kernel, 1, 1, 0, 0);
    expect_floats(out.values(), {10});
}

TEST(Conv2DBackward, LargerConvolutionDoesNotCorruptSmallerOne) {
    // the scratch space is shared, so a large convolution then a small one must both be correct
    expect_matches_reference({4, true, 9, 9, 3, 3, 3, 4, 1, 1, 1, 1});
    expect_matches_reference({1, true, 3, 3, 1, 2, 2, 1, 1, 1, 0, 0});
}

TEST(Conv2DCall, RejectsChannelMismatch) {
    Tensor<float> input(1, {3, 3, 2});
    Tensor<float> kernel(1, {2, 2, 1, 1});
    Tensor<float> out(1, {2, 2, 1});
    EXPECT_THROW(
        Conv2D<float>::call(DIRECTION::FORWARD, out, input, kernel, 1, 1, 0, 0),
        std::invalid_argument
    );
}

TEST(Conv2DCall, RejectsKernelCountMismatch) {
    Tensor<float> input(1, {3, 3, 1});
    Tensor<float> kernel(1, {2, 2, 1, 2});
    Tensor<float> out(1, {2, 2, 1});
    EXPECT_THROW(
        Conv2D<float>::call(DIRECTION::FORWARD, out, input, kernel, 1, 1, 0, 0),
        std::invalid_argument
    );
}

TEST(Conv2DCall, RejectsWrongOutputShape) {
    Tensor<float> input(1, {3, 3, 1});
    Tensor<float> kernel(1, {2, 2, 1, 1});
    Tensor<float> out(1, {3, 3, 1});
    EXPECT_THROW(
        Conv2D<float>::call(DIRECTION::FORWARD, out, input, kernel, 1, 1, 0, 0),
        std::invalid_argument
    );
}

} // namespace
