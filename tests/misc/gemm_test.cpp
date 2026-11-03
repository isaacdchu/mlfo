#include <cstddef>
#include <vector>

#include <gtest/gtest.h>

#include <mlfo/misc/gemm.hpp>

namespace {

void expect_floats(const std::vector<float>& actual, const std::vector<float>& expected) {
    ASSERT_EQ(actual.size(), expected.size());
    for (std::size_t i = 0; i < expected.size(); i++) {
        EXPECT_FLOAT_EQ(actual[i], expected[i]) << "index " << i;
    }
}

// a (2 x 3), b (3 x 2), a * b = (2 x 2)
const std::vector<float> a = {1, 2, 3, 4, 5, 6};
const std::vector<float> b = {7, 8, 9, 10, 11, 12};
const std::vector<float> a_times_b = {58, 64, 139, 154};

std::vector<float> transpose(const std::vector<float>& m, std::size_t rows, std::size_t cols) {
    std::vector<float> t(m.size());
    for (std::size_t i = 0; i < rows; i++) {
        for (std::size_t j = 0; j < cols; j++) {
            t[j * rows + i] = m[i * cols + j];
        }
    }
    return t;
}

TEST(Gemm, NN) {
    std::vector<float> c(4, 0);
    mlfo::misc::gemm_nn(2, 2, 3, a.data(), b.data(), c.data());
    expect_floats(c, a_times_b);
}

TEST(Gemm, NT) {
    const auto b_t = transpose(b, 3, 2);
    std::vector<float> c(4, 0);
    mlfo::misc::gemm_nt(2, 2, 3, a.data(), b_t.data(), c.data());
    expect_floats(c, a_times_b);
}

TEST(Gemm, TN) {
    const auto a_t = transpose(a, 2, 3);
    std::vector<float> c(4, 0);
    mlfo::misc::gemm_tn(2, 2, 3, a_t.data(), b.data(), c.data());
    expect_floats(c, a_times_b);
}

TEST(Gemm, AllVariantsAccumulate) {
    const auto a_t = transpose(a, 2, 3);
    const auto b_t = transpose(b, 3, 2);
    const std::vector<float> expected = {59, 65, 140, 155};
    std::vector<float> c(4, 1);
    mlfo::misc::gemm_nn(2, 2, 3, a.data(), b.data(), c.data());
    expect_floats(c, expected);
    c.assign(4, 1);
    mlfo::misc::gemm_nt(2, 2, 3, a.data(), b_t.data(), c.data());
    expect_floats(c, expected);
    c.assign(4, 1);
    mlfo::misc::gemm_tn(2, 2, 3, a_t.data(), b.data(), c.data());
    expect_floats(c, expected);
}

// a plain triple loop reference, c (m x n) += A (m x k) * B (k x n) with A and B given row-major
template <typename T>
std::vector<T> reference(
    std::size_t m, std::size_t n, std::size_t k,
    const std::vector<T>& a, const std::vector<T>& b, std::vector<T> c
) {
    for (std::size_t i = 0; i < m; i++) {
        for (std::size_t j = 0; j < n; j++) {
            double sum = 0;
            for (std::size_t p = 0; p < k; p++) {
                sum += static_cast<double>(a[i * k + p]) * b[p * n + j];
            }
            c[i * n + j] += static_cast<T>(sum);
        }
    }
    return c;
}

template <typename T>
std::vector<T> pattern(std::size_t size, std::size_t multiplier, std::size_t modulus) {
    std::vector<T> values(size);
    for (std::size_t i = 0; i < size; i++) {
        values[i] = static_cast<T>((i * multiplier) % modulus) / static_cast<T>(modulus) - T(0.5);
    }
    return values;
}

template <typename T>
std::vector<T> transpose_t(const std::vector<T>& m, std::size_t rows, std::size_t cols) {
    std::vector<T> t(m.size());
    for (std::size_t i = 0; i < rows; i++) {
        for (std::size_t j = 0; j < cols; j++) {
            t[j * rows + i] = m[i * cols + j];
        }
    }
    return t;
}

// checks every variant against the reference for one shape
template <typename T>
void expect_all_variants_match(std::size_t m, std::size_t n, std::size_t k, double tolerance) {
    const auto a = pattern<T>(m * k, 7, 17);
    const auto b = pattern<T>(k * n, 5, 13);
    const auto c0 = pattern<T>(m * n, 3, 11);
    const auto expected = reference(m, n, k, a, b, c0);
    const auto a_t = transpose_t(a, m, k);
    const auto b_t = transpose_t(b, k, n);
    auto c = c0;
    mlfo::misc::gemm_nn(m, n, k, a.data(), b.data(), c.data());
    for (std::size_t i = 0; i < c.size(); i++) {
        ASSERT_NEAR(c[i], expected[i], tolerance) << "nn " << m << "x" << n << "x" << k << " index " << i;
    }
    c = c0;
    mlfo::misc::gemm_nt(m, n, k, a.data(), b_t.data(), c.data());
    for (std::size_t i = 0; i < c.size(); i++) {
        ASSERT_NEAR(c[i], expected[i], tolerance) << "nt " << m << "x" << n << "x" << k << " index " << i;
    }
    c = c0;
    mlfo::misc::gemm_tn(m, n, k, a_t.data(), b.data(), c.data());
    for (std::size_t i = 0; i < c.size(); i++) {
        ASSERT_NEAR(c[i], expected[i], tolerance) << "tn " << m << "x" << n << "x" << k << " index " << i;
    }
}

TEST(Gemm, MatchesReferenceAcrossTileEdges) {
    // full tiles, partial tiles in each dimension, a single row or column, and several k blocks
    const std::size_t shapes[][3] = {
        {4, 8, 1}, {8, 16, 32}, {5, 9, 3}, {3, 7, 2}, {1, 120, 400},
        {37, 6, 25}, {29, 16, 150}, {13, 11, 600}, {150, 16, 517}, {1, 1, 1}
    };
    for (const auto& shape : shapes) {
        expect_all_variants_match<float>(shape[0], shape[1], shape[2], 1e-3);
        expect_all_variants_match<double>(shape[0], shape[1], shape[2], 1e-9);
    }
}

TEST(Gemm, EmptyDimensionsDoNothing) {
    std::vector<float> c = {1, 2};
    const std::vector<float> a = {1, 2};
    mlfo::misc::gemm_nn<float>(1, 2, 0, a.data(), a.data(), c.data());
    expect_floats(c, {1, 2});
    mlfo::misc::gemm_nn<float>(0, 2, 1, a.data(), a.data(), c.data());
    expect_floats(c, {1, 2});
}

TEST(Gemm, NonSquareShapes) {
    // (1 x 3) * (3 x 4)
    const std::vector<float> row = {1, 2, 3};
    const std::vector<float> m = {1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1};
    std::vector<float> c(4, 0);
    mlfo::misc::gemm_nn(1, 4, 3, row.data(), m.data(), c.data());
    expect_floats(c, {1, 2, 3, 6});
}

} // namespace
