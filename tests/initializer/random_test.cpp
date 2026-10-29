#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <random>
#include <vector>

#include "helpers.hpp"

namespace {

using Tensor = mlfo::tensor::Tensor<double>;
using Shape = std::vector<std::size_t>;
using Generator = mlfo::initializer::Generator;
using Normal = std::normal_distribution<double>;
using Uniform = std::uniform_real_distribution<double>;

// enough samples for the sample statistics to land within a few percent of the true ones
constexpr std::size_t SAMPLES = 100000;

double mean(const std::vector<double>& values) {
    double sum = 0;
    for (double value : values) {
        sum += value;
    }
    return sum / static_cast<double>(values.size());
}

double variance(const std::vector<double>& values) {
    const double m = mean(values);
    double sum = 0;
    for (double value : values) {
        sum += (value - m) * (value - m);
    }
    return sum / static_cast<double>(values.size());
}

void expect_within(const std::vector<double>& values, double low, double high) {
    for (double value : values) {
        ASSERT_GE(value, low);
        ASSERT_LE(value, high);
    }
}

// the generated values depend on the standard library's distributions, so the tests check
// statistics rather than exact values
void expect_zero_mean_stddev(const std::vector<double>& values, double stddev) {
    EXPECT_NEAR(mean(values), 0.0, 0.02 * stddev);
    EXPECT_NEAR(variance(values), stddev * stddev, 0.03 * stddev * stddev);
}

// U(-a, a) has variance a^2 / 3. the bound is checked with a small tolerance since it is
// computed as sqrt(3) * stddev rather than directly
void expect_uniform(const std::vector<double>& values, double bound) {
    expect_within(values, -bound * (1 + 1e-12), bound * (1 + 1e-12));
    expect_zero_mean_stddev(values, bound / std::sqrt(3.0));
}

// a user defined distribution: -scale or +scale with equal probability, so its stddev is scale
class Rademacher {
private:
    double scale_;
public:
    using result_type = double;

    Rademacher(double scale) :
    scale_(scale)
    {
        // nothing
    }

    template <class G>
    double operator()(G& generator) {
        return std::bernoulli_distribution(0.5)(generator) ? scale_ : -scale_;
    }
};

} // namespace

// makes the custom distribution usable by xavier and he
template <>
struct mlfo::initializer::ScaledDistribution<Rademacher> {
    static Rademacher make(double stddev) {
        return Rademacher(stddev);
    }
};

TEST(Random, SamplesFromGivenUniform) {
    Generator generator(0);
    Tensor tensor(1, Shape{SAMPLES});
    mlfo::initializer::Random(generator, Uniform(2.0, 5.0)).initialize(tensor, 1, 1);
    expect_within(tensor.values(), 2.0, 5.0);
    EXPECT_NEAR(mean(tensor.values()), 3.5, 0.02);
}

TEST(Random, SamplesFromGivenNormal) {
    Generator generator(0);
    Tensor tensor(1, Shape{SAMPLES});
    mlfo::initializer::Random(generator, Normal(1.0, 0.5)).initialize(tensor, 1, 1);
    EXPECT_NEAR(mean(tensor.values()), 1.0, 0.01);
    EXPECT_NEAR(variance(tensor.values()), 0.25, 0.03 * 0.25);
}

TEST(Random, IgnoresFans) {
    Generator first_generator(7);
    Generator second_generator(7);
    Tensor first(1, Shape{16});
    Tensor second(1, Shape{16});
    mlfo::initializer::Random(first_generator, Normal(0.0, 1.0)).initialize(first, 1, 1);
    mlfo::initializer::Random(second_generator, Normal(0.0, 1.0)).initialize(second, 100, 100);
    EXPECT_EQ(first.values(), second.values());
}

TEST(Random, AcceptsCustomDistribution) {
    Generator generator(0);
    Tensor tensor(1, Shape{1000});
    mlfo::initializer::Random(generator, Rademacher(3.0)).initialize(tensor, 1, 1);
    for (double value : tensor.values()) {
        ASSERT_TRUE(value == 3.0 || value == -3.0);
    }
}

TEST(ScaledDistribution, NormalHasGivenStddev) {
    const Normal distribution = mlfo::initializer::ScaledDistribution<Normal>::make(0.5);
    EXPECT_DOUBLE_EQ(distribution.mean(), 0.0);
    EXPECT_DOUBLE_EQ(distribution.stddev(), 0.5);
}

TEST(ScaledDistribution, UniformHasGivenStddev) {
    const Uniform distribution = mlfo::initializer::ScaledDistribution<Uniform>::make(0.5);
    EXPECT_DOUBLE_EQ(distribution.a(), -std::sqrt(3.0) * 0.5);
    EXPECT_DOUBLE_EQ(distribution.b(), std::sqrt(3.0) * 0.5);
}

TEST(Xavier, UniformBoundUsesFanInAndFanOut) {
    Generator generator(0);
    Tensor tensor(1, Shape{SAMPLES});
    mlfo::initializer::Xavier<Uniform>(generator).initialize(tensor, 100, 200);
    expect_uniform(tensor.values(), std::sqrt(6.0 / 300.0));
}

TEST(Xavier, NormalStddevUsesFanInAndFanOut) {
    Generator generator(0);
    Tensor tensor(1, Shape{SAMPLES});
    mlfo::initializer::Xavier<Normal>(generator).initialize(tensor, 100, 200);
    expect_zero_mean_stddev(tensor.values(), std::sqrt(2.0 / 300.0));
}

TEST(He, UniformBoundUsesFanInOnly) {
    Generator generator(0);
    Tensor tensor(1, Shape{SAMPLES});
    mlfo::initializer::He<Uniform>(generator).initialize(tensor, 100, 200);
    expect_uniform(tensor.values(), std::sqrt(6.0 / 100.0));
}

TEST(He, NormalStddevUsesFanInOnly) {
    Generator generator(0);
    Tensor tensor(1, Shape{SAMPLES});
    mlfo::initializer::He<Normal>(generator).initialize(tensor, 100, 200);
    expect_zero_mean_stddev(tensor.values(), std::sqrt(2.0 / 100.0));
}

TEST(He, AcceptsCustomScaledDistribution) {
    Generator generator(0);
    Tensor tensor(1, Shape{1000});
    mlfo::initializer::He<Rademacher>(generator).initialize(tensor, 8, 1);
    // stddev = sqrt(2 / 8)
    for (double value : tensor.values()) {
        ASSERT_TRUE(value == 0.5 || value == -0.5);
    }
}

TEST(RandomInitializer, AcceptsOtherGenerators) {
    std::minstd_rand generator(0);
    Tensor tensor(1, Shape{1000});
    mlfo::initializer::He<Uniform, std::minstd_rand>(generator).initialize(tensor, 6, 1);
    expect_within(tensor.values(), -1.0 - 1e-12, 1.0 + 1e-12);
}

TEST(RandomInitializer, SameSeedIsReproducible) {
    Generator first_generator(42);
    Generator second_generator(42);
    Tensor first(1, Shape{16});
    Tensor second(1, Shape{16});
    mlfo::initializer::He<Normal>(first_generator).initialize(first, 4, 4);
    mlfo::initializer::He<Normal>(second_generator).initialize(second, 4, 4);
    EXPECT_EQ(first.values(), second.values());
}

TEST(RandomInitializer, SharedGeneratorGivesDifferentValuesPerTensor) {
    // one initializer reused by two layers of the same shape must not repeat itself
    Generator generator(42);
    mlfo::initializer::Xavier<Uniform> initializer(generator);
    Tensor first(1, Shape{16});
    Tensor second(1, Shape{16});
    initializer.initialize(first, 4, 4);
    initializer.initialize(second, 4, 4);
    EXPECT_NE(first.values(), second.values());
}

TEST(RandomInitializer, WorksWithFloat) {
    Generator generator(0);
    mlfo::tensor::Tensor<float> tensor(1, Shape{1000});
    mlfo::initializer::He<std::uniform_real_distribution<float>>(generator).initialize(tensor, 6, 1);
    for (float value : tensor.values()) {
        ASSERT_GE(value, -1.0001f);
        ASSERT_LE(value, 1.0001f);
    }
}
