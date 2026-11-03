#ifndef MLFO_MUL_HPP
#define MLFO_MUL_HPP

#include <algorithm>
#include <concepts>
#include <memory>
#include <vector>

#include "../../misc/concepts.hpp"
#include "../../misc/gemm.hpp"
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
        std::fill(out_values.begin(), out_values.end(), T{0});
        const std::size_t m = a.shape()[0];
        const std::size_t n = a.shape()[1];
        const std::size_t p = b.shape()[1];
        if (b.batch_size() == 1) {
            // the batches of a and out are stacked rows, so a shared b makes the batch one gemm
            // (batch * m, n) * (n, p) = (batch * m, p)
            misc::gemm_nn(
                a.batch_size() * m, p, n,
                a.values().data(), b.values().data(), out_values.data()
            );
            return;
        }
        for (std::size_t batch = 0; batch < a.batch_size(); batch++) {
            misc::gemm_nn(
                m, p, n,
                a.values().data() + batch * a.unbatched_size(),
                b.values().data() + batch * b.unbatched_size(),
                out_values.data() + batch * out.unbatched_size()
            );
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
        if (b.batch_size() == 1) {
            // as in the forward pass, the batch is folded into the rows
            // the gemm for b.gradients then also sums them over the batch
            const std::size_t rows = a.batch_size() * m;
            // (rows, p) * (n, p)^T = (rows, n)
            misc::gemm_nt(
                rows, n, p,
                out.gradients().data(), b.values().data(), a_gradients.data()
            );
            // (rows, n)^T * (rows, p) = (n, p)
            misc::gemm_tn(
                n, p, rows,
                a.values().data(), out.gradients().data(), b_gradients.data()
            );
            return;
        }
        for (std::size_t batch = 0; batch < a.batch_size(); batch++) {
            const T* out_gradients = out.gradients().data() + batch * out.unbatched_size();
            misc::gemm_nt(
                m, n, p,
                out_gradients,
                b.values().data() + batch * b.unbatched_size(),
                a_gradients.data() + batch * a.unbatched_size()
            );
            misc::gemm_tn(
                n, p, m,
                a.values().data() + batch * a.unbatched_size(),
                out_gradients,
                b_gradients.data() + batch * b.unbatched_size()
            );
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

#endif // MLFO_MUL_HPP
