#ifndef CONV2D_HPP
#define CONV2D_HPP

#include <vector>
#include <memory>
#include <concepts>

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
        // todo
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
        // todo
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
        std::size_t padding_w,
    ) {
        // todo
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
