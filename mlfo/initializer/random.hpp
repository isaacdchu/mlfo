#ifndef RANDOM_INITIALIZER_HPP
#define RANDOM_INITIALIZER_HPP

#include <cstddef>
#include <utility>
#include <vector>

#include "../misc/concepts.hpp"
#include "../tensor/tensor.hpp"
#include "distribution.hpp"
#include "initializer.hpp"

namespace mlfo::initializer {

// fills tensors with samples of distribution type D drawn with generator G
// non-owning, the generator must outlive every build() that uses this initializer
template <class D, class G = Generator>
requires mlfo::misc::Distribution<D, G>
class RandomInitializer : public Initializer<typename D::result_type> {
protected:
    using T = typename D::result_type;

    G& generator_;

    void sample(mlfo::tensor::Tensor<T>& tensor, D& distribution) const {
        std::vector<T>& values = mlfo::tensor::Operation<T>::values(tensor);
        for (T& value : values) {
            value = static_cast<T>(distribution(generator_));
        }
    }
public:
    RandomInitializer(G& generator) :
    generator_(generator)
    {
        // nothing
    }
};

// samples from the given distribution as is, ignoring the fans, e.g.
// Random(generator, std::normal_distribution<float>(0, 0.01f))
template <class D, class G = Generator>
requires mlfo::misc::Distribution<D, G>
class Random : public RandomInitializer<D, G> {
private:
    // sampling advances the state of standard library distributions
    mutable D distribution_;
public:
    Random(G& generator, D distribution) :
    RandomInitializer<D, G>(generator),
    distribution_(std::move(distribution))
    {
        // nothing
    }

    void initialize(
        mlfo::tensor::Tensor<typename D::result_type>& tensor,
        std::size_t fan_in,
        std::size_t fan_out
    ) const override {
        this->sample(tensor, distribution_);
    }
};

// samples from a zero mean distribution of type D, with a standard deviation picked from the fans
// by the derived class. the distribution is built with ScaledDistribution<D>
template <class D, class G = Generator>
requires ScalableDistribution<D, G>
class VarianceScaling : public RandomInitializer<D, G> {
protected:
    using T = typename D::result_type;

    virtual T stddev(std::size_t fan_in, std::size_t fan_out) const = 0;
public:
    using RandomInitializer<D, G>::RandomInitializer;

    void initialize(
        mlfo::tensor::Tensor<T>& tensor,
        std::size_t fan_in,
        std::size_t fan_out
    ) const override final {
        D distribution = ScaledDistribution<D>::make(stddev(fan_in, fan_out));
        this->sample(tensor, distribution);
    }
};

} // namespace mlfo::initializer

#endif // RANDOM_INITIALIZER_HPP
