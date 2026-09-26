#ifndef MUL_HPP
#define MUL_HPP

#include <vector>
#include <memory>
#include <concepts>

#include "../../misc/concepts.hpp"
#include "../tensor.hpp"

#define BLOCK_SIZE 32

namespace mlfo::tensor::ops {

template <misc::Number T>
struct Mul : public mlfo::tensor::Operation<T> {
private:
    static inline void mul_forward(
        Tensor<T>& out,
        const Tensor<T>& a,
        const Tensor<T>& b
    ) {
        // perform element-wise multiplication
        // broadcast b to the shape of a if necessary
        for (std::size_t batch = 0; batch < a.batch_size(); batch++) {
            const std::size_t a_batch_offset = batch * a.unbatched_size();
            const std::size_t b_batch_offset = (
                b.batch_size() == 1
            ) ? 0 : batch * b.unbatched_size();
            const std::size_t out_batch_offset = batch * out.unbatched_size();
            // blocked matrix multiplication (32 x 32 blocks)
            for (std::size_t i = 0; i < a.unbatched_size(); i += BLOCK_SIZE) {
                for (std::size_t j = 0; j < a.unbatched_size(); j += BLOCK_SIZE) {
                    for (std::size_t k = 0; k < a.unbatched_size(); k += BLOCK_SIZE) {
                        const std::size_t i_max = std::min(i + BLOCK_SIZE, a.unbatched_size());
                        const std::size_t j_max = std::min(j + BLOCK_SIZE, a.unbatched_size());
                        const std::size_t k_max = std::min(k + BLOCK_SIZE, a.unbatched_size());
                        for (std::size_t ii = i; ii < i_max; ii++) {
                            for (std::size_t jj = j; jj < j_max; jj++) {
                                for (std::size_t kk = k; kk < k_max; kk++) {
                                    mlfo::tensor::Operation<T>::values(out)[out_batch_offset + ii] += (
                                        a.values()[a_batch_offset + ii] *
                                        b.values()[b_batch_offset + kk]
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    static inline void mul_backward(
        const Tensor<T>& out,
        Tensor<T>& a,
        Tensor<T>& b
    ) {
        // propagate gradients to a and b
        for (std::size_t batch = 0; batch < a.batch_size(); batch++) {
            const std::size_t a_batch_offset = batch * a.unbatched_size();
            const std::size_t b_batch_offset = (
                b.batch_size() == 1
            ) ? 0 : batch * b.unbatched_size();
            const std::size_t out_batch_offset = batch * out.unbatched_size();
            // a.gradients = out.gradients * b.values^T
            // b.gradients = a.values^T * out.gradients
            // blocked matrix multiplication (32 x 32 blocks)
            for (std::size_t i = 0; i < a.unbatched_size(); i += BLOCK_SIZE) {
                for (std::size_t j = 0; j < a.unbatched_size(); j += BLOCK_SIZE) {
                    for (std::size_t k = 0; k < a.unbatched_size(); k += BLOCK_SIZE) {
                        const std::size_t i_max = std::min(i + BLOCK_SIZE, a.unbatched_size());
                        const std::size_t j_max = std::min(j + BLOCK_SIZE, a.unbatched_size());
                        const std::size_t k_max = std::min(k + BLOCK_SIZE, a.unbatched_size());
                        for (std::size_t ii = i; ii < i_max; ii++) {
                            for (std::size_t jj = j; jj < j_max; jj++) {
                                for (std::size_t kk = k; kk < k_max; kk++) {
                                    mlfo::tensor::Operation<T>::gradients(a)[a_batch_offset + ii] += (
                                        out.gradients()[out_batch_offset + ii] *
                                        b.values()[b_batch_offset + kk]
                                    );
                                    mlfo::tensor::Operation<T>::gradients(b)[b_batch_offset + kk] += (
                                        a.values()[a_batch_offset + ii] *
                                        out.gradients()[out_batch_offset + ii]
                                    );
                                }
                            }
                        }
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
