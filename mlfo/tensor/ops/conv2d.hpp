#ifndef CONV2D_HPP
#define CONV2D_HPP

#include <concepts>
#include <memory>
#include <vector>

#include "../../misc/concepts.hpp"
#include "../tensor.hpp"

namespace mlfo::tensor::ops {

template <misc::Number T>
struct Conv2D : public mlfo::tensor::Operation<T> {
private:
    static inline void conv2d_forward(
        Tensor<T>& out,
        const Tensor<T>& input,
        const Tensor<T>& kernel,
        std::size_t stride_h,
        std::size_t stride_w,
        std::size_t padding_h,
        std::size_t padding_w
    ) {
        auto& out_values = mlfo::tensor::Operation<T>::values(out);
        const std::size_t c = input.shape()[0];
        const std::size_t h = input.shape()[1];
        const std::size_t w = input.shape()[2];
        const std::size_t k = kernel.shape()[0];
        const std::size_t r = kernel.shape()[2];
        const std::size_t s = kernel.shape()[3];
        const std::size_t p = out.shape()[1];
        const std::size_t q = out.shape()[2];
        // out[n, k, p, q] = sum over c, r, s of
        // input[n, c, p * stride_h + r - padding_h, q * stride_w + s - padding_w] * kernel[k, c, r, s]
        // positions that fall in the padding contribute zero
        // kernel is broadcast if its batch size is 1
        for (std::size_t batch = 0; batch < input.batch_size(); batch++) {
            const std::size_t input_batch_offset = batch * input.unbatched_size();
            const std::size_t kernel_batch_offset = (
                kernel.batch_size() == 1
            ) ? 0 : batch * kernel.unbatched_size();
            const std::size_t out_batch_offset = batch * out.unbatched_size();
            for (std::size_t ki = 0; ki < k; ki++) {
                for (std::size_t pi = 0; pi < p; pi++) {
                    for (std::size_t qi = 0; qi < q; qi++) {
                        T sum = 0;
                        for (std::size_t ci = 0; ci < c; ci++) {
                            for (std::size_t ri = 0; ri < r; ri++) {
                                // row in the padded input, skip if it is in the padding
                                const std::size_t hi = pi * stride_h + ri;
                                if (hi < padding_h || hi - padding_h >= h) {
                                    continue;
                                }
                                for (std::size_t si = 0; si < s; si++) {
                                    const std::size_t wi = qi * stride_w + si;
                                    if (wi < padding_w || wi - padding_w >= w) {
                                        continue;
                                    }
                                    sum += (
                                        input.values()[
                                            input_batch_offset +
                                            (ci * h + (hi - padding_h)) * w + (wi - padding_w)
                                        ] *
                                        kernel.values()[
                                            kernel_batch_offset + ((ki * c + ci) * r + ri) * s + si
                                        ]
                                    );
                                }
                            }
                        }
                        out_values[out_batch_offset + (ki * p + pi) * q + qi] = sum;
                    }
                }
            }
        }
    }

    static inline void conv2d_backward(
        const Tensor<T>& out,
        Tensor<T>& input,
        Tensor<T>& kernel,
        std::size_t stride_h,
        std::size_t stride_w,
        std::size_t padding_h,
        std::size_t padding_w
    ) {
        auto& input_values = mlfo::tensor::Operation<T>::values(input);
        auto& kernel_values = mlfo::tensor::Operation<T>::values(kernel);
        auto& input_gradients = mlfo::tensor::Operation<T>::gradients(input);
        auto& kernel_gradients = mlfo::tensor::Operation<T>::gradients(kernel);
        const std::size_t c = input.shape()[0];
        const std::size_t h = input.shape()[1];
        const std::size_t w = input.shape()[2];
        const std::size_t k = kernel.shape()[0];
        const std::size_t r = kernel.shape()[2];
        const std::size_t s = kernel.shape()[3];
        const std::size_t p = out.shape()[1];
        const std::size_t q = out.shape()[2];
        // input.gradients[n, c, h, w] += out.gradients[n, k, p, q] * kernel[k, c, r, s]
        // kernel.gradients[k, c, r, s] += out.gradients[n, k, p, q] * input[n, c, h, w]
        // for every (p, q, r, s) with h = p * stride_h + r - padding_h, w = q * stride_w + s - padding_w
        // if kernel is broadcast, its gradients accumulate over the batch
        for (std::size_t batch = 0; batch < input.batch_size(); batch++) {
            const std::size_t input_batch_offset = batch * input.unbatched_size();
            const std::size_t kernel_batch_offset = (
                kernel.batch_size() == 1
            ) ? 0 : batch * kernel.unbatched_size();
            const std::size_t out_batch_offset = batch * out.unbatched_size();
            for (std::size_t ki = 0; ki < k; ki++) {
                for (std::size_t pi = 0; pi < p; pi++) {
                    for (std::size_t qi = 0; qi < q; qi++) {
                        const T out_gradient = out.gradients()[
                            out_batch_offset + (ki * p + pi) * q + qi
                        ];
                        for (std::size_t ci = 0; ci < c; ci++) {
                            for (std::size_t ri = 0; ri < r; ri++) {
                                // row in the padded input, skip if it is in the padding
                                const std::size_t hi = pi * stride_h + ri;
                                if (hi < padding_h || hi - padding_h >= h) {
                                    continue;
                                }
                                for (std::size_t si = 0; si < s; si++) {
                                    const std::size_t wi = qi * stride_w + si;
                                    if (wi < padding_w || wi - padding_w >= w) {
                                        continue;
                                    }
                                    const std::size_t input_index = (
                                        input_batch_offset +
                                        (ci * h + (hi - padding_h)) * w + (wi - padding_w)
                                    );
                                    const std::size_t kernel_index = (
                                        kernel_batch_offset + ((ki * c + ci) * r + ri) * s + si
                                    );
                                    input_gradients[input_index] += (
                                        out_gradient * kernel_values[kernel_index]
                                    );
                                    kernel_gradients[kernel_index] += (
                                        out_gradient * input_values[input_index]
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
        // input: (N, C, H, W)
        // kernel: (K, C, R, S), can be broadcasted if its batch size is 1
        // output: (N, K, P, Q)
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
        if (input.shape()[0] != kernel.shape()[1]) {
            throw std::invalid_argument(
                "[tensor::ops::conv2d] Input channels must match kernel channels"
            );
        }
        if (out.shape()[0] != kernel.shape()[0]) {
            throw std::invalid_argument(
                "[tensor::ops::conv2d] Output channels must match number of kernels"
            );
        }
        if (
            input.shape()[1] + 2 * padding_h < kernel.shape()[2] ||
            input.shape()[2] + 2 * padding_w < kernel.shape()[3]
        ) {
            throw std::invalid_argument(
                "[tensor::ops::conv2d] Kernel size must be less than or equal to input size + 2 * padding"
            );
        }
        if (
            out.shape()[1] != (input.shape()[1] + 2 * padding_h - kernel.shape()[2]) / stride_h + 1 ||
            out.shape()[2] != (input.shape()[2] + 2 * padding_w - kernel.shape()[3]) / stride_w + 1
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
        switch (direction) {
            case DIRECTION::FORWARD:
                conv2d_forward(
                    out,
                    input,
                    kernel,
                    stride_h,
                    stride_w,
                    padding_h,
                    padding_w
                );
                return;
            case DIRECTION::BACKWARD:
                conv2d_backward(
                    out,
                    input,
                    kernel,
                    stride_h,
                    stride_w,
                    padding_h,
                    padding_w
                );
                return;
            default:
                throw std::invalid_argument(
                    "[tensor::ops::conv2d] Invalid direction"
                );
        }
    }
};

} // namespace mlfo::tensor::ops

#endif // CONV2D_HPP
