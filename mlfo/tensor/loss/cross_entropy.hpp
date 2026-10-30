#ifndef CROSS_ENTROPY_HPP
#define CROSS_ENTROPY_HPP

#include <algorithm>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <vector>

#include "../../misc/concepts.hpp"
#include "../tensor.hpp"

namespace mlfo::tensor::loss {

// softmax cross entropy, a is the logits and b is the target distribution (e.g. one-hot)
// softmax is applied here rather than as its own op, since the fused gradient is simpler
// and log(softmax(a)) can be computed without overflow or log(0)
template <misc::Number T>
struct CrossEntropy : public mlfo::tensor::Operation<T> {
private:
    // log(sum(exp(a))) over one batch, shifted by the max so exp can't overflow
    static inline T log_sum_exp(const Tensor<T>& a, std::size_t a_batch_offset) {
        const auto begin = a.values().begin() + a_batch_offset;
        const T max = *std::max_element(begin, begin + a.unbatched_size());
        T sum = T{0};
        for (std::size_t i = 0; i < a.unbatched_size(); i++) {
            sum += std::exp(a.values()[a_batch_offset + i] - max);
        }
        return max + std::log(sum);
    }

    static inline void cross_entropy_forward(
        Tensor<T>& out,
        const Tensor<T>& a,
        const Tensor<T>& b
    ) {
        // L = -sum(b * log(softmax(a))), where log(softmax(a)) = a - log(sum(exp(a)))
        auto& out_values = mlfo::tensor::Operation<T>::values(out);
        for (std::size_t batch = 0; batch < a.batch_size(); batch++) {
            const std::size_t a_batch_offset = batch * a.unbatched_size();
            const std::size_t b_batch_offset = batch * b.unbatched_size();
            const std::size_t out_batch_offset = batch * out.unbatched_size();
            const T lse = log_sum_exp(a, a_batch_offset);
            T sum = T{0};
            for (std::size_t i = 0; i < a.unbatched_size(); i++) {
                sum += b.values()[b_batch_offset + i] * (a.values()[a_batch_offset + i] - lse);
            }
            out_values[out_batch_offset] = -sum;
        }
    }

    static inline void cross_entropy_backward(
        const Tensor<T>& out,
        Tensor<T>& a,
        Tensor<T>& b
    ) {
        // dL/da = softmax(a) * sum(b) - b, which is softmax(a) - b when b sums to 1
        // dL/db = -log(softmax(a))
        auto& a_gradients = mlfo::tensor::Operation<T>::gradients(a);
        auto& b_gradients = mlfo::tensor::Operation<T>::gradients(b);
        for (std::size_t batch = 0; batch < a.batch_size(); batch++) {
            const std::size_t a_batch_offset = batch * a.unbatched_size();
            const std::size_t b_batch_offset = batch * b.unbatched_size();
            const std::size_t out_batch_offset = batch * out.unbatched_size();
            // one loss value per batch, so one upstream gradient per batch
            const T upstream = out.gradients()[out_batch_offset];
            const T lse = log_sum_exp(a, a_batch_offset);
            T b_sum = T{0};
            for (std::size_t i = 0; i < b.unbatched_size(); i++) {
                b_sum += b.values()[b_batch_offset + i];
            }
            for (std::size_t i = 0; i < a.unbatched_size(); i++) {
                const T log_softmax = a.values()[a_batch_offset + i] - lse;
                const T label = b.values()[b_batch_offset + i];
                a_gradients[a_batch_offset + i] += upstream * (std::exp(log_softmax) * b_sum - label);
                b_gradients[b_batch_offset + i] -= upstream * log_softmax;
            }
        }
    }

public:
    CrossEntropy() = delete;
    static void call(
        ops::DIRECTION direction,
        Tensor<T>& out,
        Tensor<T>& a,
        Tensor<T>& b
    ) {
        // check that the shapes of a and b match, and that out holds one value per batch
        if (a.shape() != b.shape()) {
            throw std::invalid_argument(
                "[tensor::loss::cross_entropy] Shapes of input tensors must be the same"
            );
        }
        if (out.unbatched_size() != 1) {
            throw std::invalid_argument(
                "[tensor::loss::cross_entropy] Output tensor must hold a single value per batch"
            );
        }
        // check that the batch sizes of a, b, and out are compatible
        // no broadcasting for cross entropy, all tensors must have the same batch size
        if (a.batch_size() != out.batch_size() || b.batch_size() != out.batch_size()) {
            throw std::invalid_argument(
                "[tensor::loss::cross_entropy] Batch sizes of output tensor must match input tensor a and b"
            );
        }
        switch (direction) {
            case mlfo::tensor::ops::DIRECTION::FORWARD:
                cross_entropy_forward(out, a, b);
                return;
            case mlfo::tensor::ops::DIRECTION::BACKWARD:
                cross_entropy_backward(out, a, b);
                return;
            default:
                throw std::invalid_argument(
                    "[tensor::loss::cross_entropy] Invalid direction"
                );
        }
    }
};

} // namespace mlfo::tensor::loss

#endif // CROSS_ENTROPY_HPP
