#ifndef MAX_HPP
#define MAX_HPP

#include <algorithm>
#include <concepts>
#include <memory>
#include <vector>

#include "../../misc/concepts.hpp"
#include "../tensor.hpp"

namespace mlfo::tensor::pool {

template <misc::Number T>
struct Max : public mlfo::tensor::Operation<T> {
private:
    // flat index (within one batch) of the max element of the window that produces out[ci, pi, qi]
    // if pooling window goes out of bounds, only consider the valid elements
    // ties go to the first element in row-major order
    static inline std::size_t argmax(
        const Tensor<T>& a,
        std::size_t a_batch_offset,
        std::size_t ci,
        std::size_t pi,
        std::size_t qi,
        std::size_t r,
        std::size_t s,
        std::size_t stride_h,
        std::size_t stride_w
    ) {
        const std::size_t h = a.shape()[a.rank() - 2];
        const std::size_t w = a.shape()[a.rank() - 1];
        const std::size_t h_start = pi * stride_h;
        const std::size_t w_start = qi * stride_w;
        const std::size_t h_end = std::min(h_start + r, h);
        const std::size_t w_end = std::min(w_start + s, w);
        std::size_t best = (ci * h + h_start) * w + w_start;
        for (std::size_t hi = h_start; hi < h_end; hi++) {
            for (std::size_t wi = w_start; wi < w_end; wi++) {
                const std::size_t index = (ci * h + hi) * w + wi;
                if (a.values()[a_batch_offset + index] > a.values()[a_batch_offset + best]) {
                    best = index;
                }
            }
        }
        return best;
    }

    static inline void max_forward(
        Tensor<T>& out,
        const Tensor<T>& a,
        std::size_t r,
        std::size_t s,
        std::size_t stride_h,
        std::size_t stride_w
    ) {
        // out[..., p, q] = max over the window a[..., p * stride_h + [0, r), q * stride_w + [0, s)]
        // every dimension before the last 2 is treated as a channel
        auto& out_values = mlfo::tensor::Operation<T>::values(out);
        const std::size_t p = out.shape()[out.rank() - 2];
        const std::size_t q = out.shape()[out.rank() - 1];
        const std::size_t c = out.unbatched_size() / (p * q);
        for (std::size_t batch = 0; batch < a.batch_size(); batch++) {
            const std::size_t a_batch_offset = batch * a.unbatched_size();
            const std::size_t out_batch_offset = batch * out.unbatched_size();
            for (std::size_t ci = 0; ci < c; ci++) {
                for (std::size_t pi = 0; pi < p; pi++) {
                    for (std::size_t qi = 0; qi < q; qi++) {
                        const std::size_t best = argmax(
                            a, a_batch_offset, ci, pi, qi, r, s, stride_h, stride_w
                        );
                        out_values[out_batch_offset + (ci * p + pi) * q + qi] = (
                            a.values()[a_batch_offset + best]
                        );
                    }
                }
            }
        }
    }

    static inline void max_backward(
        const Tensor<T>& out,
        Tensor<T>& a,
        std::size_t r,
        std::size_t s,
        std::size_t stride_h,
        std::size_t stride_w
    ) {
        // propagate gradients to a
        // dL/da = dL/dout for the max element of each window, 0 for the rest
        // overlapping windows accumulate into the same element
        auto& a_gradients = mlfo::tensor::Operation<T>::gradients(a);
        const std::size_t p = out.shape()[out.rank() - 2];
        const std::size_t q = out.shape()[out.rank() - 1];
        const std::size_t c = out.unbatched_size() / (p * q);
        for (std::size_t batch = 0; batch < a.batch_size(); batch++) {
            const std::size_t a_batch_offset = batch * a.unbatched_size();
            const std::size_t out_batch_offset = batch * out.unbatched_size();
            for (std::size_t ci = 0; ci < c; ci++) {
                for (std::size_t pi = 0; pi < p; pi++) {
                    for (std::size_t qi = 0; qi < q; qi++) {
                        const std::size_t best = argmax(
                            a, a_batch_offset, ci, pi, qi, r, s, stride_h, stride_w
                        );
                        a_gradients[a_batch_offset + best] += (
                            out.gradients()[out_batch_offset + (ci * p + pi) * q + qi]
                        );
                    }
                }
            }
        }
    }

public:
    Max() = delete;

    // number of windows along a dimension of size n with window size k and the given stride
    // the last window may run past the end (ceil mode), but every window starts inside the input
    static std::size_t output_size(std::size_t n, std::size_t k, std::size_t stride) {
        if (n <= k) {
            return 1;
        }
        std::size_t size = (n - k + stride - 1) / stride + 1;
        if ((size - 1) * stride >= n) {
            size--;
        }
        return size;
    }

    static void call(
        ops::DIRECTION direction,
        Tensor<T>& out,
        Tensor<T>& a,
        std::size_t r,
        std::size_t s,
        std::size_t stride_h,
        std::size_t stride_w
    ) {
        // a: (N, ..., H, W)
        // out: (N, ..., P, Q)
        // check that the batch sizes of a and out are compatible
        if (a.batch_size() != out.batch_size()) {
            throw std::invalid_argument(
                "[tensor::pool::max] Batch sizes of output tensor must match input tensor a"
            );
        }
        if (r == 0 || s == 0) {
            throw std::invalid_argument(
                "[tensor::pool::max] Pooling window size must be greater than 0"
            );
        }
        if (stride_h == 0 || stride_w == 0) {
            throw std::invalid_argument(
                "[tensor::pool::max] Pooling stride must be greater than 0"
            );
        }
        if (a.rank() < 2) {
            throw std::invalid_argument(
                "[tensor::pool::max] Input must have at least 2 dimensions"
            );
        }
        if (out.rank() != a.rank()) {
            throw std::invalid_argument(
                "[tensor::pool::max] Ranks of input and output tensors must match"
            );
        }
        if (!std::equal(a.shape().begin(), a.shape().end() - 2, out.shape().begin())) {
            throw std::invalid_argument(
                "[tensor::pool::max] Dimensions before the last 2 must match between input and output"
            );
        }
        if (
            out.shape()[out.rank() - 2] != output_size(a.shape()[a.rank() - 2], r, stride_h) ||
            out.shape()[out.rank() - 1] != output_size(a.shape()[a.rank() - 1], s, stride_w)
        ) {
            throw std::invalid_argument(
                "[tensor::pool::max] Output height and width do not match the pooling parameters"
            );
        }
        switch (direction) {
            case mlfo::tensor::ops::DIRECTION::FORWARD:
                max_forward(out, a, r, s, stride_h, stride_w);
                return;
            case mlfo::tensor::ops::DIRECTION::BACKWARD:
                max_backward(out, a, r, s, stride_h, stride_w);
                return;
            default:
                throw std::invalid_argument(
                    "[tensor::pool::max] Invalid direction"
                );
        }
    }
};

} // namespace mlfo::tensor::pool


#endif // MAX_HPP
