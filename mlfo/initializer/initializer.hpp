#ifndef INITIALIZER_HPP
#define INITIALIZER_HPP

#include <cstddef>

#include "../misc/concepts.hpp"
#include "../tensor/tensor.hpp"

namespace mlfo::initializer {

// sets the starting values of a layer's parameter tensor. a layer takes its initializers in its
// constructor and calls initialize() on each parameter from build()
// the constructor of an initializer only takes shape agnostic hyperparameters (e.g. a generator),
// so one initializer can be shared by layers of different sizes
template <mlfo::misc::Number T>
class Initializer : public mlfo::tensor::Operation<T> {
public:
    virtual ~Initializer() = default;

    // overwrites every value of tensor
    // fan_in and fan_out are given by the layer, since only the layer knows which dimensions of
    // the parameter's shape are inputs and which are outputs
    virtual void initialize(
        mlfo::tensor::Tensor<T>& tensor,
        std::size_t fan_in,
        std::size_t fan_out
    ) const = 0;
};

} // namespace mlfo::initializer

#endif // INITIALIZER_HPP
