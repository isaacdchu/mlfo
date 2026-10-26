#ifndef RELU_HPP
#define RELU_HPP

#include <algorithm>
#include <concepts>
#include <memory>
#include <vector>

#include "../../misc/concepts.hpp"
#include "../tensor.hpp"

namespace mlfo::tensor::activation {

template <misc::Number T>
struct ReLU : public mlfo::tensor::Operation<T> {
private:
    static inline void relu_forward(
        Tensor<T>& out,
        const Tensor<T>& a
    ) {
        // perform element-wise ReLU: max(0, a)
        auto& out_values = mlfo::tensor::Operation<T>::values(out);
        for (std::size_t batch = 0; batch < a.batch_size(); batch++) {
            const std::size_t a_batch_offset = batch * a.unbatched_size();
            const std::size_t out_batch_offset = batch * out.unbatched_size();
            for (std::size_t i = 0; i < a.unbatched_size(); i++) {
                out_values[out_batch_offset + i] = std::max(a.values()[a_batch_offset + i], T{0});
            }
        }
    }

    static inline void relu_backward(
        const Tensor<T>& out,
        Tensor<T>& a
    ) {
        // propagate gradients to a
        // dL/da = dL/dout if a > 0 else 0
        auto& a_gradients = mlfo::tensor::Operation<T>::gradients(a);
        for (std::size_t batch = 0; batch < a.batch_size(); batch++) {
            const std::size_t a_batch_offset = batch * a.unbatched_size();
            const std::size_t out_batch_offset = batch * out.unbatched_size();
            for (std::size_t i = 0; i < a.unbatched_size(); i++) {
                const T value = a.values()[a_batch_offset + i];
                const T upstream = out.gradients()[out_batch_offset + i];
                a_gradients[a_batch_offset + i] += (value > T{0} ? upstream : T{0});
            }
        }
    }

public:
    ReLU() = delete;
    static void call(
        ops::DIRECTION direction,
        Tensor<T>& out,
        Tensor<T>& a
    ) {
        // check that the shapes of a and out match
        if (a.shape() != out.shape()) {
            throw std::invalid_argument(
                "[tensor::activation::relu] Shapes of input and output tensors must be the same"
            );
        }
        // check that the batch sizes of a and out are compatible
        // no broadcasting for relu, all tensors must have the same batch size
        if (a.batch_size() != out.batch_size()) {
            throw std::invalid_argument(
                "[tensor::activation::relu] Batch sizes of output tensor must match input tensor a"
            );
        }
        switch (direction) {
            case mlfo::tensor::ops::DIRECTION::FORWARD:
                relu_forward(out, a);
                return;
            case mlfo::tensor::ops::DIRECTION::BACKWARD:
                relu_backward(out, a);
                return;
            default:
                throw std::invalid_argument(
                    "[tensor::activation::relu] Invalid direction"
                );
        }
    }
};

} // namespace mlfo::tensor::activation

#endif // RELU_HPP
