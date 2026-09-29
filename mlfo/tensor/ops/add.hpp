#ifndef ADD_HPP
#define ADD_HPP

#include <vector>
#include <memory>
#include <concepts>

#include "../../misc/concepts.hpp"
#include "../tensor.hpp"

namespace mlfo::tensor::ops {

template <misc::Number T>
struct Add : public mlfo::tensor::Operation<T> {
private:
    static inline void add_forward(
        Tensor<T>& out,
        const Tensor<T>& a,
        const Tensor<T>& b
    ) {
        // perform element-wise addition
        // broadcast b to the shape of a if necessary
        auto& out_values = mlfo::tensor::Operation<T>::values(out);
        for (std::size_t batch = 0; batch < a.batch_size(); batch++) {
            const std::size_t a_batch_offset = batch * a.unbatched_size();
            const std::size_t b_batch_offset = (b.batch_size() == 1) ? 0 : batch * b.unbatched_size();
            const std::size_t out_batch_offset = batch * out.unbatched_size();
            for (std::size_t i = 0; i < a.unbatched_size(); i++) {
                out_values[out_batch_offset + i] = (
                    a.values()[a_batch_offset + i] + b.values()[b_batch_offset + i]
                );
            }
        }
    }

    static inline void add_backward(
        const Tensor<T>& out,
        Tensor<T>& a,
        Tensor<T>& b
    ) {
        // propagate gradients to a and b
        // if b is broadcast, its gradients accumulate over the batch
        auto& a_gradients = mlfo::tensor::Operation<T>::gradients(a);
        auto& b_gradients = mlfo::tensor::Operation<T>::gradients(b);
        for (std::size_t batch = 0; batch < a.batch_size(); batch++) {
            const std::size_t a_batch_offset = batch * a.unbatched_size();
            const std::size_t b_batch_offset = (b.batch_size() == 1) ? 0 : batch * b.unbatched_size();
            const std::size_t out_batch_offset = batch * out.unbatched_size();
            for (std::size_t i = 0; i < a.unbatched_size(); i++) {
                a_gradients[a_batch_offset + i] += out.gradients()[out_batch_offset + i];
                b_gradients[b_batch_offset + i] += out.gradients()[out_batch_offset + i];
            }
        }
    }

public:
    Add() = delete;
    static void call(
        DIRECTION direction,
        Tensor<T>& out,
        Tensor<T>& a,
        Tensor<T>& b
    ) {
        // check that the shapes of a, b, and out are compatible
        if (a.shape() != b.shape() || a.shape() != out.shape()) {
            throw std::invalid_argument(
                "[tensor::ops::add] Shapes of input tensors must be the same"
            );
        }
        // check that the batch sizes of a, b, and out are compatible
        // out must match a, b can be broadcasted if its batch size is 1
        if (a.batch_size() != out.batch_size()) {
            throw std::invalid_argument(
                "[tensor::ops::add] Batch sizes of output tensor must match input tensor a"
            );
        }
        if (a.batch_size() != b.batch_size() && b.batch_size() != 1) {
            throw std::invalid_argument(
                "[tensor::ops::add] Batch sizes of input tensors must be the same or b must have batch size 1"
            );
        }
        switch (direction) {
            case DIRECTION::FORWARD:
                add_forward(out, a, b);
                return;
            case DIRECTION::BACKWARD:
                add_backward(out, a, b);
                return;
            default:
                throw std::invalid_argument(
                    "[tensor::ops::add] Invalid direction"
                );
        }
    }
};

} // namespace mlfo::tensor::ops

#endif // ADD_HPP
