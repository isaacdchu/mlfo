#ifndef HE_INITIALIZER_HPP
#define HE_INITIALIZER_HPP

#include <cmath>
#include <cstddef>

#include "distribution.hpp"
#include "random.hpp"

namespace mlfo::initializer {

// he et al. (2015), doubles xavier's variance to make up for relu zeroing half of its inputs
// use for layers followed by relu
// stddev = sqrt(2 / fan_in), e.g.
// He<std::uniform_real_distribution<float>> samples from U(-a, a), a = sqrt(6 / fan_in)
template <class D, class G = Generator>
requires ScalableDistribution<D, G>
class He : public VarianceScaling<D, G> {
protected:
    using T = typename D::result_type;

    T stddev(std::size_t fan_in, std::size_t fan_out) const override {
        return std::sqrt(T{2} / static_cast<T>(fan_in));
    }
public:
    using VarianceScaling<D, G>::VarianceScaling;
};

} // namespace mlfo::initializer

#endif // HE_INITIALIZER_HPP
