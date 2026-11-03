#ifndef MLFO_MAX_HPP
#define MLFO_MAX_HPP

#include <algorithm>
#include <concepts>
#include <memory>
#include <stdexcept>
#include <vector>

#include "../../misc/concepts.hpp"
#include "../tensor.hpp"

namespace mlfo::tensor::pool {

// channels last max pooling
// a has shape (H, W, ...), out has shape (P, Q, ...)
// every dimension after the first 2 is treated as a channel, so (H, W) is a single channel
//
// the forward pass records which element of a each output came from, so the backward pass only
// has to route each output gradient to that element. the caller owns the indices, so every
// pooling layer keeps its own, and it must run forward before backward
template <misc::Number T>
struct Max : public mlfo::tensor::Operation<T> {
private:
    // writes the max of every window into out and the flat index into a of where it came from
    // into indices, which must have one entry per element of out
    // if pooling window goes out of bounds, only consider the valid elements
    // ties go to the first element in row-major order
    static inline void find_max(
        const Tensor<T>& a,
        T* out_values,
        std::size_t* indices,
        std::size_t p,
        std::size_t q,
        std::size_t r,
        std::size_t s,
        std::size_t stride_h,
        std::size_t stride_w
    ) {
        const std::size_t h = a.shape()[0];
        const std::size_t w = a.shape()[1];
        const std::size_t c = a.unbatched_size() / (h * w);
        const T* a_values = a.values().data();
        // the channels of a position are contiguous, so a window is walked once and every channel's
        // running max is updated together
        for (std::size_t batch = 0; batch < a.batch_size(); batch++) {
            const std::size_t a_batch_offset = batch * a.unbatched_size();
            const std::size_t out_batch_offset = batch * p * q * c;
            for (std::size_t pi = 0; pi < p; pi++) {
                const std::size_t h_start = pi * stride_h;
                const std::size_t h_end = std::min(h_start + r, h);
                for (std::size_t qi = 0; qi < q; qi++) {
                    const std::size_t w_start = qi * stride_w;
                    const std::size_t w_end = std::min(w_start + s, w);
                    T* best = out_values + out_batch_offset + (pi * q + qi) * c;
                    std::size_t* best_index = indices + out_batch_offset + (pi * q + qi) * c;
                    // every window starts inside a, so its first element initializes the max
                    const std::size_t first = a_batch_offset + (h_start * w + w_start) * c;
                    for (std::size_t ci = 0; ci < c; ci++) {
                        best[ci] = a_values[first + ci];
                        best_index[ci] = first + ci;
                    }
                    for (std::size_t hi = h_start; hi < h_end; hi++) {
                        for (std::size_t wi = w_start; wi < w_end; wi++) {
                            const std::size_t position = a_batch_offset + (hi * w + wi) * c;
                            for (std::size_t ci = 0; ci < c; ci++) {
                                // strictly greater, so ties keep the earlier element
                                if (a_values[position + ci] > best[ci]) {
                                    best[ci] = a_values[position + ci];
                                    best_index[ci] = position + ci;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    static inline void max_forward(
        Tensor<T>& out,
        const Tensor<T>& a,
        std::vector<std::size_t>& indices,
        std::size_t r,
        std::size_t s,
        std::size_t stride_h,
        std::size_t stride_w
    ) {
        // out[p, q, ...] = max over the window a[p * stride_h + [0, r), q * stride_w + [0, s), ...]
        auto& out_values = mlfo::tensor::Operation<T>::values(out);
        indices.resize(out.size());
        find_max(
            a, out_values.data(), indices.data(),
            out.shape()[0], out.shape()[1], r, s, stride_h, stride_w
        );
    }

    static inline void max_backward(
        const Tensor<T>& out,
        Tensor<T>& a,
        const std::vector<std::size_t>& indices
    ) {
        // propagate gradients to a
        // dL/da = dL/dout for the max element of each window, 0 for the rest
        // overlapping windows accumulate into the same element
        if (indices.size() != out.size()) {
            throw std::logic_error(
                "[tensor::pool::max] Backward requires the indices from a forward pass"
            );
        }
        auto& a_gradients = mlfo::tensor::Operation<T>::gradients(a);
        const auto& out_gradients = out.gradients();
        for (std::size_t i = 0; i < indices.size(); i++) {
            a_gradients[indices[i]] += out_gradients[i];
        }
    }

    static void validate(
        const Tensor<T>& out,
        const Tensor<T>& a,
        std::size_t r,
        std::size_t s,
        std::size_t stride_h,
        std::size_t stride_w
    ) {
        // a: (N, H, W, ...)
        // out: (N, P, Q, ...)
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
        if (!std::equal(a.shape().begin() + 2, a.shape().end(), out.shape().begin() + 2)) {
            throw std::invalid_argument(
                "[tensor::pool::max] Dimensions after the first 2 must match between input and output"
            );
        }
        if (
            out.shape()[0] != output_size(a.shape()[0], r, stride_h) ||
            out.shape()[1] != output_size(a.shape()[1], s, stride_w)
        ) {
            throw std::invalid_argument(
                "[tensor::pool::max] Output height and width do not match the pooling parameters"
            );
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

    // forward fills indices with where each output came from, backward reads them
    static void call(
        ops::DIRECTION direction,
        Tensor<T>& out,
        Tensor<T>& a,
        std::vector<std::size_t>& indices,
        std::size_t r,
        std::size_t s,
        std::size_t stride_h,
        std::size_t stride_w
    ) {
        validate(out, a, r, s, stride_h, stride_w);
        switch (direction) {
            case mlfo::tensor::ops::DIRECTION::FORWARD:
                max_forward(out, a, indices, r, s, stride_h, stride_w);
                return;
            case mlfo::tensor::ops::DIRECTION::BACKWARD:
                max_backward(out, a, indices);
                return;
            default:
                throw std::invalid_argument(
                    "[tensor::pool::max] Invalid direction"
                );
        }
    }

    // without indices from the caller, backward finds the maxes again from a's current values
    static void call(
        ops::DIRECTION direction,
        Tensor<T>& out,
        Tensor<T>& a,
        std::size_t r,
        std::size_t s,
        std::size_t stride_h,
        std::size_t stride_w
    ) {
        validate(out, a, r, s, stride_h, stride_w);
        std::vector<std::size_t> indices(out.size());
        switch (direction) {
            case mlfo::tensor::ops::DIRECTION::FORWARD:
                max_forward(out, a, indices, r, s, stride_h, stride_w);
                return;
            case mlfo::tensor::ops::DIRECTION::BACKWARD: {
                // the maxes are only needed for their indices, so they go to a scratch copy
                std::vector<T> maxes(out.size());
                find_max(
                    a, maxes.data(), indices.data(),
                    out.shape()[0], out.shape()[1], r, s, stride_h, stride_w
                );
                max_backward(out, a, indices);
                return;
            }
            default:
                throw std::invalid_argument(
                    "[tensor::pool::max] Invalid direction"
                );
        }
    }
};

} // namespace mlfo::tensor::pool


#endif // MLFO_MAX_HPP
