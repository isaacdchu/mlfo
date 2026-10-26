#ifndef MSE_HPP
#define MSE_HPP

#include <concepts>
#include <memory>
#include <vector>

#include "../../misc/concepts.hpp"
#include "../tensor.hpp"

namespace mlfo::tensor::loss {

template <misc::Number T>
struct MSE : public mlfo::tensor::Operation<T> {
private:
    static inline void mse_forward(
        Tensor<T>& out,
        const Tensor<T>& a,
        const Tensor<T>& b
    ) {
        // perform element-wise mean squared error: (0.5 * (a - b)^2) / n
        auto& out_values = mlfo::tensor::Operation<T>::values(out);
        for (std::size_t batch = 0; batch < a.batch_size(); batch++) {
            const std::size_t a_batch_offset = batch * a.unbatched_size();
            const std::size_t b_batch_offset = batch * b.unbatched_size();
            const std::size_t out_batch_offset = batch * out.unbatched_size();
            T sum = T{0};
            for (std::size_t i = 0; i < a.unbatched_size(); i++) {
                const T error = (
                    a.values()[a_batch_offset + i] -
                    b.values()[b_batch_offset + i]
                );
                sum += error * error;
            }
            out_values[out_batch_offset] = (
                static_cast<T>(0.5) * sum /
                static_cast<T>(a.unbatched_size())
            );
        }
    }

    static inline void mse_backward(
        const Tensor<T>& out,
        Tensor<T>& a,
        Tensor<T>& b
    ) {
        // propagate gradients to a and b
        // dL/da = (a - b) / n = -dL/db
        auto& a_gradients = mlfo::tensor::Operation<T>::gradients(a);
        auto& b_gradients = mlfo::tensor::Operation<T>::gradients(b);
        for (std::size_t batch = 0; batch < a.batch_size(); batch++) {
            const std::size_t a_batch_offset = batch * a.unbatched_size();
            const std::size_t b_batch_offset = batch * b.unbatched_size();
            const std::size_t out_batch_offset = batch * out.unbatched_size();
            // one loss value per batch, so one upstream gradient per batch
            const T scale = out.gradients()[out_batch_offset] / static_cast<T>(a.unbatched_size());
            for (std::size_t i = 0; i < a.unbatched_size(); i++) {
                const T error = (
                    a.values()[a_batch_offset + i] -
                    b.values()[b_batch_offset + i]
                );
                a_gradients[a_batch_offset + i] += error * scale;
                b_gradients[b_batch_offset + i] -= error * scale;
            }
        }
    }

public:
    MSE() = delete;
    static void call(
        ops::DIRECTION direction,
        Tensor<T>& out,
        Tensor<T>& a,
        Tensor<T>& b
    ) {
        // check that the shapes of a and b match, and that out holds one value per batch
        if (a.shape() != b.shape()) {
            throw std::invalid_argument(
                "[tensor::ops::mse] Shapes of input tensors must be the same"
            );
        }
        if (out.unbatched_size() != 1) {
            throw std::invalid_argument(
                "[tensor::ops::mse] Output tensor must hold a single value per batch"
            );
        }
        // check that the batch sizes of a, b, and out are compatible
        // no broadcasting for mse, all tensors must have the same batch size
        if (a.batch_size() != out.batch_size() || b.batch_size() != out.batch_size()) {
            throw std::invalid_argument(
                "[tensor::ops::mse] Batch sizes of output tensor must match input tensor a and b"
            );
        }
        switch (direction) {
            case mlfo::tensor::ops::DIRECTION::FORWARD:
                mse_forward(out, a, b);
                return;
            case mlfo::tensor::ops::DIRECTION::BACKWARD:
                mse_backward(out, a, b);
                return;
            default:
                throw std::invalid_argument(
                    "[tensor::ops::mse] Invalid direction"
                );
        }
    }
};

} // namespace mlfo::tensor::loss

#endif // MSE_HPP
