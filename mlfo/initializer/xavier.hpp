#ifndef XAVIER_INITIALIZER_HPP
#define XAVIER_INITIALIZER_HPP

#include <cmath>
#include <cstddef>

#include "distribution.hpp"
#include "random.hpp"

namespace mlfo::initializer {

// glorot & bengio (2010), keeps the variance of activations and gradients roughly constant
// through layers followed by a symmetric activation such as sigmoid or tanh
// stddev = sqrt(2 / (fan_in + fan_out)), e.g.
// Xavier<std::uniform_real_distribution<float>> samples from U(-a, a), a = sqrt(6 / (fan_in + fan_out))
template <class D, class G = Generator>
requires ScalableDistribution<D, G>
class Xavier : public VarianceScaling<D, G> {
protected:
    using T = typename D::result_type;

    T stddev(std::size_t fan_in, std::size_t fan_out) const override {
        return std::sqrt(T{2} / static_cast<T>(fan_in + fan_out));
    }
public:
    using VarianceScaling<D, G>::VarianceScaling;
};

} // namespace mlfo::initializer

#endif // XAVIER_INITIALIZER_HPP
