#ifndef TANH_HPP
#define TANH_HPP

#include <algorithm>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <vector>

#include "../../misc/concepts.hpp"
#include "../tensor.hpp"

namespace mlfo::tensor::activation {

template <misc::Number T>
struct Tanh : public mlfo::tensor::Operation<T> {
private:
    static inline void tanh_forward(
        Tensor<T>& out,
        const Tensor<T>& a
    ) {
        // perform element-wise Tanh: tanh(a)
        auto& out_values = mlfo::tensor::Operation<T>::values(out);
        for (std::size_t batch = 0; batch < a.batch_size(); batch++) {
            const std::size_t a_batch_offset = batch * a.unbatched_size();
            const std::size_t out_batch_offset = batch * out.unbatched_size();
            for (std::size_t i = 0; i < a.unbatched_size(); i++) {
                out_values[out_batch_offset + i] = std::tanh(a.values()[a_batch_offset + i]);
            }
        }
    }

    static inline void tanh_backward(
        const Tensor<T>& out,
        Tensor<T>& a
    ) {
        // propagate gradients to a
        // dL/da = dL/dout * (1 - tanh^2(a))
        auto& a_gradients = mlfo::tensor::Operation<T>::gradients(a);
        for (std::size_t batch = 0; batch < a.batch_size(); batch++) {
            const std::size_t a_batch_offset = batch * a.unbatched_size();
            const std::size_t out_batch_offset = batch * out.unbatched_size();
            for (std::size_t i = 0; i < a.unbatched_size(); i++) {
                const T value = a.values()[a_batch_offset + i];
                const T upstream = out.gradients()[out_batch_offset + i];
                a_gradients[a_batch_offset + i] += upstream * (T{1} - std::tanh(value) * std::tanh(value));
            }
        }
    }

public:
    Tanh() = delete;
    static void call(
        ops::DIRECTION direction,
        Tensor<T>& out,
        Tensor<T>& a
    ) {
        // check that the shapes of a and out match
        if (a.shape() != out.shape()) {
            throw std::invalid_argument(
                "[tensor::activation::tanh] Shapes of input and output tensors must be the same"
            );
        }
        // check that the batch sizes of a and out are compatible
        // no broadcasting for tanh, all tensors must have the same batch size
        if (a.batch_size() != out.batch_size()) {
            throw std::invalid_argument(
                "[tensor::activation::tanh] Batch sizes of output tensor must match input tensor a"
            );
        }
        switch (direction) {
            case mlfo::tensor::ops::DIRECTION::FORWARD:
                tanh_forward(out, a);
                return;
            case mlfo::tensor::ops::DIRECTION::BACKWARD:
                tanh_backward(out, a);
                return;
            default:
                throw std::invalid_argument(
                    "[tensor::activation::tanh] Invalid direction"
                );
        }
    }
};

} // namespace mlfo::tensor::activation

#endif // TANH_HPP
