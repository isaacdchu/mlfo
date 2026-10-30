#ifndef RESHAPE_HPP
#define RESHAPE_HPP

#include <concepts>
#include <memory>
#include <vector>

#include "../../misc/concepts.hpp"
#include "../tensor.hpp"

namespace mlfo::tensor::ops {

template <misc::Number T>
struct Reshape : public mlfo::tensor::Operation<T> {
private:
    static inline void reshape_forward(
        Tensor<T>& out,
        const Tensor<T>& a
    ) {
        // values are stored row-major, so reshaping is a straight copy
        auto& out_values = mlfo::tensor::Operation<T>::values(out);
        for (std::size_t i = 0; i < a.size(); i++) {
            out_values[i] = a.values()[i];
        }
    }

    static inline void reshape_backward(
        const Tensor<T>& out,
        Tensor<T>& a
    ) {
        // propagate gradients to a
        // dL/da = dL/dout, element for element
        auto& a_gradients = mlfo::tensor::Operation<T>::gradients(a);
        for (std::size_t i = 0; i < a.size(); i++) {
            a_gradients[i] += out.gradients()[i];
        }
    }

public:
    Reshape() = delete;
    static void call(
        DIRECTION direction,
        Tensor<T>& out,
        Tensor<T>& a
    ) {
        // check that a and out hold the same number of elements per batch
        if (a.unbatched_size() != out.unbatched_size()) {
            throw std::invalid_argument(
                "[tensor::ops::reshape] Input and output tensors must have the same number of elements"
            );
        }
        // no broadcasting for reshape, all tensors must have the same batch size
        if (a.batch_size() != out.batch_size()) {
            throw std::invalid_argument(
                "[tensor::ops::reshape] Batch sizes of output tensor must match input tensor a"
            );
        }
        switch (direction) {
            case DIRECTION::FORWARD:
                reshape_forward(out, a);
                return;
            case DIRECTION::BACKWARD:
                reshape_backward(out, a);
                return;
            default:
                throw std::invalid_argument(
                    "[tensor::ops::reshape] Invalid direction"
                );
        }
    }
};

} // namespace mlfo::tensor::ops

#endif // RESHAPE_HPP
