#ifndef CONSTANT_INITIALIZER_HPP
#define CONSTANT_INITIALIZER_HPP

#include <algorithm>
#include <cstddef>
#include <vector>

#include "../misc/concepts.hpp"
#include "../tensor/tensor.hpp"
#include "initializer.hpp"

namespace mlfo::initializer {

// sets every value to the same constant
template <mlfo::misc::Number T>
class Constant : public Initializer<T> {
private:
    T value_;
public:
    Constant(T value) :
    value_(value)
    {
        // nothing
    }

    void initialize(
        mlfo::tensor::Tensor<T>& tensor,
        std::size_t fan_in,
        std::size_t fan_out
    ) const override {
        std::vector<T>& values = mlfo::tensor::Operation<T>::values(tensor);
        std::fill(values.begin(), values.end(), value_);
    }
};

// the default for biases. weights should not use it, since every unit of a layer would then
// compute the same output and receive the same gradient
template <mlfo::misc::Number T>
class Zeros : public Constant<T> {
public:
    Zeros() :
    Constant<T>(T{0})
    {
        // nothing
    }
};

} // namespace mlfo::initializer

#endif // CONSTANT_INITIALIZER_HPP
