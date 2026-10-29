#ifndef DISTRIBUTION_HPP
#define DISTRIBUTION_HPP

#include <cmath>
#include <concepts>
#include <random>

#include "../misc/concepts.hpp"

namespace mlfo::initializer {

// the default generator for random initializers
// share one generator between all the initializers of a model. initializers seeded separately
// with the same seed would give layers of the same shape identical values
using Generator = std::mt19937;

// builds a zero mean distribution of type D with the given standard deviation
// variance scaling initializers (xavier, he) pick the standard deviation from the layer's fans,
// and use this to turn it into a distribution of the type the user chose
// to use a custom distribution with them, specialize this for it, e.g.
//
// template <mlfo::misc::Number T>
// struct mlfo::initializer::ScaledDistribution<MyDistribution<T>> {
//     static MyDistribution<T> make(T stddev) { ... }
// };
template <class D>
struct ScaledDistribution;

template <mlfo::misc::Number T>
struct ScaledDistribution<std::normal_distribution<T>> {
    static std::normal_distribution<T> make(T stddev) {
        return std::normal_distribution<T>(T{0}, stddev);
    }
};

template <mlfo::misc::Number T>
struct ScaledDistribution<std::uniform_real_distribution<T>> {
    static std::uniform_real_distribution<T> make(T stddev) {
        // U(-a, a) has variance a^2 / 3
        const T bound = std::sqrt(T{3}) * stddev;
        return std::uniform_real_distribution<T>(-bound, bound);
    }
};

// a distribution that variance scaling initializers can use
// kept next to ScaledDistribution rather than in misc/concepts.hpp since it is defined by it
template <class D, class G>
concept ScalableDistribution =
    mlfo::misc::Distribution<D, G> &&
    requires(typename D::result_type stddev) {
        { ScaledDistribution<D>::make(stddev) } -> std::same_as<D>;
    };

} // namespace mlfo::initializer

#endif // DISTRIBUTION_HPP
