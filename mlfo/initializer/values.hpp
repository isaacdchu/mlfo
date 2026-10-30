#ifndef VALUES_INITIALIZER_HPP
#define VALUES_INITIALIZER_HPP

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>

#include "../misc/concepts.hpp"
#include "../tensor/tensor.hpp"
#include "initializer.hpp"

namespace mlfo::initializer {

// copies the given values into the tensor, e.g. parameters read back by Model::deserialize()
// unlike the other initializers it is tied to one shape, so it can only be used by one parameter
template <mlfo::misc::Number T>
class Values : public Initializer<T> {
private:
    std::vector<T> values_;
public:
    Values(std::vector<T> values) :
    values_(std::move(values))
    {
        // nothing
    }

    void initialize(
        mlfo::tensor::Tensor<T>& tensor,
        std::size_t fan_in,
        std::size_t fan_out
    ) const override {
        std::vector<T>& values = mlfo::tensor::Operation<T>::values(tensor);
        if (values.size() != values_.size()) {
            throw std::invalid_argument(
                "[mlfo::initializer::Values::initialize] Tensor size does not match the number of values"
            );
        }
        std::ranges::copy(values_, values.begin());
    }
};

} // namespace mlfo::initializer

#endif // VALUES_INITIALIZER_HPP
