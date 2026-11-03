#ifndef MLFO_CHANNEL_ADD_HPP
#define MLFO_CHANNEL_ADD_HPP

#include <concepts>
#include <memory>
#include <vector>

#include "../../misc/concepts.hpp"
#include "../tensor.hpp"

namespace mlfo::tensor::ops {

// adds a per-channel value to every element of that channel
// channels last: a and out have shape (..., C), b has shape (C) and batch size 1
// e.g. conv2d biases, where each kernel has one bias shared over the output map
template <misc::Number T>
struct ChannelAdd : public mlfo::tensor::Operation<T> {
private:
    static inline void channel_add_forward(
        Tensor<T>& out,
        const Tensor<T>& a,
        const Tensor<T>& b
    ) {
        // perform element-wise addition, broadcasting b over every position
        // the channels are contiguous, so a and out are runs of C values each getting b added
        auto& out_values = mlfo::tensor::Operation<T>::values(out);
        const std::size_t channels = b.unbatched_size();
        const std::size_t positions = a.size() / channels;
        for (std::size_t i = 0; i < positions; i++) {
            const std::size_t offset = i * channels;
            for (std::size_t c = 0; c < channels; c++) {
                out_values[offset + c] = a.values()[offset + c] + b.values()[c];
            }
        }
    }

    static inline void channel_add_backward(
        const Tensor<T>& out,
        Tensor<T>& a,
        Tensor<T>& b
    ) {
        // propagate gradients to a and b
        // dL/da = dL/dout, dL/db[c] = sum of dL/dout over channel c and the batch
        auto& a_gradients = mlfo::tensor::Operation<T>::gradients(a);
        auto& b_gradients = mlfo::tensor::Operation<T>::gradients(b);
        const std::size_t channels = b.unbatched_size();
        const std::size_t positions = a.size() / channels;
        for (std::size_t i = 0; i < positions; i++) {
            const std::size_t offset = i * channels;
            for (std::size_t c = 0; c < channels; c++) {
                a_gradients[offset + c] += out.gradients()[offset + c];
                b_gradients[c] += out.gradients()[offset + c];
            }
        }
    }

public:
    ChannelAdd() = delete;
    static void call(
        DIRECTION direction,
        Tensor<T>& out,
        Tensor<T>& a,
        Tensor<T>& b
    ) {
        // check that the shapes of a, b, and out are compatible
        if (a.shape() != out.shape()) {
            throw std::invalid_argument(
                "[tensor::ops::channel_add] Shapes of output tensor and input tensor a must be the same"
            );
        }
        if (a.shape().empty() || b.shape().size() != 1 || b.shape()[0] != a.shape().back()) {
            throw std::invalid_argument(
                "[tensor::ops::channel_add] Input tensor b must have shape (C) where C is the last dimension of a"
            );
        }
        // out must match a, b is always broadcast over the batch
        if (a.batch_size() != out.batch_size()) {
            throw std::invalid_argument(
                "[tensor::ops::channel_add] Batch sizes of output tensor must match input tensor a"
            );
        }
        if (b.batch_size() != 1) {
            throw std::invalid_argument(
                "[tensor::ops::channel_add] Input tensor b must have batch size 1"
            );
        }
        switch (direction) {
            case DIRECTION::FORWARD:
                channel_add_forward(out, a, b);
                return;
            case DIRECTION::BACKWARD:
                channel_add_backward(out, a, b);
                return;
            default:
                throw std::invalid_argument(
                    "[tensor::ops::channel_add] Invalid direction"
                );
        }
    }
};

} // namespace mlfo::tensor::ops

#endif // MLFO_CHANNEL_ADD_HPP
