#ifndef MLFO_CONV2D_HPP
#define MLFO_CONV2D_HPP

#include <algorithm>
#include <concepts>
#include <memory>
#include <vector>

#include "../../misc/concepts.hpp"
#include "../../misc/gemm.hpp"
#include "../../misc/im2col.hpp"
#include "../tensor.hpp"

namespace mlfo::tensor::ops {

// channels last 2D convolution, computed as an explicit gemm (im2col -> matmul)
// input (H, W, C), kernel (R, S, C, K), output (P, Q, K)
// the kernel reshapes to (R * S * C, K) for free, and the column matrix of the whole batch is
// (N * P * Q, R * S * C), so with a broadcast kernel the batch is a single gemm whose
// (N * P * Q, K) result is already the output layout
template <misc::Number T>
struct Conv2D : public mlfo::tensor::Operation<T> {
private:
    // column matrix scratch space, reused across calls so they don't allocate
    // it only grows, to the largest column matrix this thread has needed
    static std::vector<T>& scratch(std::size_t size) {
        thread_local std::vector<T> buffer;
        if (buffer.size() < size) {
            buffer.resize(size);
        }
        return buffer;
    }

    static misc::Conv2DGeometry geometry(
        const Tensor<T>& out,
        const Tensor<T>& input,
        const Tensor<T>& kernel,
        std::size_t stride_h,
        std::size_t stride_w,
        std::size_t padding_h,
        std::size_t padding_w
    ) {
        return {
            .h = input.shape()[0],
            .w = input.shape()[1],
            .c = input.shape()[2],
            .r = kernel.shape()[0],
            .s = kernel.shape()[1],
            .p = out.shape()[0],
            .q = out.shape()[1],
            .stride_h = stride_h,
            .stride_w = stride_w,
            .padding_h = padding_h,
            .padding_w = padding_w
        };
    }

    // the column matrix of the whole batch, (N * P * Q, R * S * C)
    // a 1x1 kernel with stride 1 and no padding reads the input as is, so no copy is made
    static const T* lower(const misc::Conv2DGeometry& g, const Tensor<T>& input) {
        if (g.is_pointwise()) {
            return input.values().data();
        }
        const std::size_t col_size = g.positions() * g.patch_size();
        T* col = scratch(input.batch_size() * col_size).data();
        for (std::size_t batch = 0; batch < input.batch_size(); batch++) {
            misc::im2col(
                g,
                input.values().data() + batch * input.unbatched_size(),
                col + batch * col_size
            );
        }
        return col;
    }

    static inline void conv2d_forward(
        Tensor<T>& out,
        const Tensor<T>& input,
        const Tensor<T>& kernel,
        const misc::Conv2DGeometry& g
    ) {
        // out (N * P * Q, K) = col (N * P * Q, R * S * C) * kernel (R * S * C, K)
        auto& out_values = mlfo::tensor::Operation<T>::values(out);
        std::fill(out_values.begin(), out_values.end(), T{0});
        const std::size_t k = kernel.shape()[3];
        const std::size_t positions = g.positions();
        const std::size_t patch = g.patch_size();
        const T* col = lower(g, input);
        if (kernel.batch_size() == 1) {
            // every sample shares the kernel, so the batch folds into the rows of one gemm
            misc::gemm_nn(
                input.batch_size() * positions, k, patch,
                col, kernel.values().data(), out_values.data()
            );
            return;
        }
        for (std::size_t batch = 0; batch < input.batch_size(); batch++) {
            misc::gemm_nn(
                positions, k, patch,
                col + batch * positions * patch,
                kernel.values().data() + batch * kernel.unbatched_size(),
                out_values.data() + batch * out.unbatched_size()
            );
        }
    }

    static inline void conv2d_backward(
        const Tensor<T>& out,
        Tensor<T>& input,
        Tensor<T>& kernel,
        const misc::Conv2DGeometry& g
    ) {
        // kernel.gradients (R * S * C, K) += col^T * out.gradients (N * P * Q, K)
        // col.gradients (N * P * Q, R * S * C) = out.gradients * kernel^T, then col2im adds them
        // back into input.gradients
        // if kernel is broadcast, the gemm over the folded batch sums its gradients over the batch
        auto& input_gradients = mlfo::tensor::Operation<T>::gradients(input);
        auto& kernel_gradients = mlfo::tensor::Operation<T>::gradients(kernel);
        const std::size_t k = kernel.shape()[3];
        const std::size_t positions = g.positions();
        const std::size_t patch = g.patch_size();
        const std::size_t batches = input.batch_size();
        const bool broadcast = kernel.batch_size() == 1;
        const T* out_gradients = out.gradients().data();
        const T* kernel_values = kernel.values().data();
        // the column matrix is recomputed rather than kept from the forward pass, as the scratch
        // space is shared by every convolution
        const T* col = lower(g, input);
        if (broadcast) {
            misc::gemm_tn(
                patch, k, batches * positions,
                col, out_gradients, kernel_gradients.data()
            );
        } else {
            for (std::size_t batch = 0; batch < batches; batch++) {
                misc::gemm_tn(
                    patch, k, positions,
                    col + batch * positions * patch,
                    out_gradients + batch * out.unbatched_size(),
                    kernel_gradients.data() + batch * kernel.unbatched_size()
                );
            }
        }
        if (g.is_pointwise()) {
            // the column matrix is the input, so its gradients go straight into the input's
            if (broadcast) {
                misc::gemm_nt(
                    batches * positions, patch, k,
                    out_gradients, kernel_values, input_gradients.data()
                );
                return;
            }
            for (std::size_t batch = 0; batch < batches; batch++) {
                misc::gemm_nt(
                    positions, patch, k,
                    out_gradients + batch * out.unbatched_size(),
                    kernel_values + batch * kernel.unbatched_size(),
                    input_gradients.data() + batch * input.unbatched_size()
                );
            }
            return;
        }
        // the column matrix is no longer needed, so its scratch space holds its gradients instead
        const std::size_t col_size = positions * patch;
        T* col_gradients = scratch(batches * col_size).data();
        std::fill(col_gradients, col_gradients + batches * col_size, T{0});
        if (broadcast) {
            misc::gemm_nt(
                batches * positions, patch, k,
                out_gradients, kernel_values, col_gradients
            );
        } else {
            for (std::size_t batch = 0; batch < batches; batch++) {
                misc::gemm_nt(
                    positions, patch, k,
                    out_gradients + batch * out.unbatched_size(),
                    kernel_values + batch * kernel.unbatched_size(),
                    col_gradients + batch * col_size
                );
            }
        }
        for (std::size_t batch = 0; batch < batches; batch++) {
            misc::col2im(
                g,
                col_gradients + batch * col_size,
                input_gradients.data() + batch * input.unbatched_size()
            );
        }
    }

public:
    Conv2D() = delete;
    static void call(
        DIRECTION direction,
        Tensor<T>& out,
        Tensor<T>& input,
        Tensor<T>& kernel,
        std::size_t stride_h,
        std::size_t stride_w,
        std::size_t padding_h,
        std::size_t padding_w
    ) {
        // input: (N, H, W, C)
        // kernel: (R, S, C, K), can be broadcasted if its batch size is 1
        // output: (N, P, Q, K)
        // verify input shapes and parameters
        if (input.rank() != 3) {
            throw std::invalid_argument(
                "[tensor::ops::conv2d] Input must be a 3-dimensional tensor"
            );
        }
        if (kernel.rank() != 4) {
            throw std::invalid_argument(
                "[tensor::ops::conv2d] Kernel must be a 4-dimensional tensor"
            );
        }
        if (out.rank() != 3) {
            throw std::invalid_argument(
                "[tensor::ops::conv2d] Output must be a 3-dimensional tensor"
            );
        }
        if (stride_h == 0 || stride_w == 0) {
            throw std::invalid_argument(
                "[tensor::ops::conv2d] Stride must be greater than zero"
            );
        }
        if (input.shape()[2] != kernel.shape()[2]) {
            throw std::invalid_argument(
                "[tensor::ops::conv2d] Input channels must match kernel channels"
            );
        }
        if (out.shape()[2] != kernel.shape()[3]) {
            throw std::invalid_argument(
                "[tensor::ops::conv2d] Output channels must match number of kernels"
            );
        }
        if (
            input.shape()[0] + 2 * padding_h < kernel.shape()[0] ||
            input.shape()[1] + 2 * padding_w < kernel.shape()[1]
        ) {
            throw std::invalid_argument(
                "[tensor::ops::conv2d] Kernel size must be less than or equal to input size + 2 * padding"
            );
        }
        if (
            out.shape()[0] != (input.shape()[0] + 2 * padding_h - kernel.shape()[0]) / stride_h + 1 ||
            out.shape()[1] != (input.shape()[1] + 2 * padding_w - kernel.shape()[1]) / stride_w + 1
        ) {
            throw std::invalid_argument(
                "[tensor::ops::conv2d] Output height and width do not match the convolution parameters"
            );
        }
        if (input.batch_size() != out.batch_size()) {
            throw std::invalid_argument(
                "[tensor::ops::conv2d] Input batch size must match output batch size"
            );
        }
        if (input.batch_size() != kernel.batch_size() && kernel.batch_size() != 1) {
            throw std::invalid_argument(
                "[tensor::ops::conv2d] Kernel batch size must match input batch size or be 1"
            );
        }
        const misc::Conv2DGeometry g = geometry(
            out, input, kernel, stride_h, stride_w, padding_h, padding_w
        );
        switch (direction) {
            case DIRECTION::FORWARD:
                conv2d_forward(out, input, kernel, g);
                return;
            case DIRECTION::BACKWARD:
                conv2d_backward(out, input, kernel, g);
                return;
            default:
                throw std::invalid_argument(
                    "[tensor::ops::conv2d] Invalid direction"
                );
        }
    }
};

} // namespace mlfo::tensor::ops

#endif // MLFO_CONV2D_HPP
