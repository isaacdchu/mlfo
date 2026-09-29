#ifndef MUL_HPP
#define MUL_HPP

#include <vector>
#include <memory>
#include <concepts>

#include "../../misc/concepts.hpp"
#include "../tensor.hpp"

namespace mlfo::tensor::ops {

template <misc::Number T>
struct Mul : public mlfo::tensor::Operation<T> {
private:
    static inline void mul_forward(
        Tensor<T>& out,
        const Tensor<T>& a,
        const Tensor<T>& b
    ) {
        // out = a * b, with a (m, n), b (n, p), out (m, p) for each batch
        // b is broadcast if its batch size is 1
        auto& out_values = mlfo::tensor::Operation<T>::values(out);
        const std::size_t m = a.shape()[0];
        const std::size_t n = a.shape()[1];
        const std::size_t p = b.shape()[1];
        for (std::size_t batch = 0; batch < a.batch_size(); batch++) {
            const std::size_t a_batch_offset = batch * a.unbatched_size();
            const std::size_t b_batch_offset = (
                b.batch_size() == 1
            ) ? 0 : batch * b.unbatched_size();
            const std::size_t out_batch_offset = batch * out.unbatched_size();
            for (std::size_t i = 0; i < m; i++) {
                for (std::size_t j = 0; j < p; j++) {
                    T sum = 0;
                    for (std::size_t k = 0; k < n; k++) {
                        sum += (
                            a.values()[a_batch_offset + i * n + k] *
                            b.values()[b_batch_offset + k * p + j]
                        );
                    }
                    out_values[out_batch_offset + i * p + j] = sum;
                }
            }
        }
    }

    static inline void mul_backward(
        const Tensor<T>& out,
        Tensor<T>& a,
        Tensor<T>& b
    ) {
        // a.gradients += out.gradients * b.values^T
        // b.gradients += a.values^T * out.gradients
        // if b is broadcast, its gradients accumulate over the batch
        auto& a_gradients = mlfo::tensor::Operation<T>::gradients(a);
        auto& b_gradients = mlfo::tensor::Operation<T>::gradients(b);
        const std::size_t m = a.shape()[0];
        const std::size_t n = a.shape()[1];
        const std::size_t p = b.shape()[1];
        for (std::size_t batch = 0; batch < a.batch_size(); batch++) {
            const std::size_t a_batch_offset = batch * a.unbatched_size();
            const std::size_t b_batch_offset = (
                b.batch_size() == 1
            ) ? 0 : batch * b.unbatched_size();
            const std::size_t out_batch_offset = batch * out.unbatched_size();
            for (std::size_t i = 0; i < m; i++) {
                for (std::size_t j = 0; j < p; j++) {
                    const T out_gradient = out.gradients()[out_batch_offset + i * p + j];
                    for (std::size_t k = 0; k < n; k++) {
                        a_gradients[a_batch_offset + i * n + k] += (
                            out_gradient * b.values()[b_batch_offset + k * p + j]
                        );
                        b_gradients[b_batch_offset + k * p + j] += (
                            a.values()[a_batch_offset + i * n + k] * out_gradient
                        );
                    }
                }
            }
        }
    }

public:
    Mul() = delete;
    static void call(
        DIRECTION direction,
        Tensor<T>& out,
        Tensor<T>& a,
        Tensor<T>& b
    ) {
        // a is (batch_size, m, n), b is (batch_size, n, p), out is (batch_size, m, p)
        // b can be broadcasted if its batch size is 1
        // check that the shapes of a, b, and out are compatible
        if (a.rank() != 2 || b.rank() != 2 || out.rank() != 2) {
            throw std::invalid_argument(
                "[tensor::ops::mul] Input tensors must be 2D"
            );
        }
        if (a.shape()[1] != b.shape()[0]) {
            throw std::invalid_argument(
                "[tensor::ops::mul] Shapes of input tensors are not compatible for multiplication"
            );
        }
        if (out.shape()[0] != a.shape()[0] || out.shape()[1] != b.shape()[1]) {
            throw std::invalid_argument(
                "[tensor::ops::mul] Shape of output tensor is not compatible with input tensors"
            );
        }
        // check that the batch sizes of a, b, and out are compatible
        // out must match a, b can be broadcasted if its batch size is 1
        if (a.batch_size() != out.batch_size()) {
            throw std::invalid_argument(
                "[tensor::ops::mul] Batch sizes of output tensor must match input tensor a"
            );
        }
        if (a.batch_size() != b.batch_size() && b.batch_size() != 1) {
            throw std::invalid_argument(
                "[tensor::ops::mul] Batch sizes of input tensors must be the same or b must have batch size 1"
            );
        }
        switch (direction) {
            case DIRECTION::FORWARD:
                mul_forward(out, a, b);
                return;
            case DIRECTION::BACKWARD:
                mul_backward(out, a, b);
                return;
            default:
                throw std::invalid_argument(
                    "[tensor::ops::mul] Invalid direction"
                );
        }
    }
};

} // namespace mlfo::tensor::ops

#endif // MUL_HPP
