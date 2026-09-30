#ifndef CONV2D_LAYER_HPP
#define CONV2D_LAYER_HPP

#include <vector>
#include <cstddef>

#include "../misc/concepts.hpp"
#include "../graph/graph.hpp"
#include "../tensor/tensor.hpp"
#include "../tensor/ops.hpp"
#include "layer.hpp"

namespace mlfo::layer {

// for simplicity, only 2D convolution with:
// input tensor of shape = (N, C, H, W)
// kernel tensor of shape = (K, C, R, S)
// output tensor of shape = (N, K, P, Q)
// = (N, K, R, S)
// no dilation or groups is supported
template <mlfo::misc::Number T>
class Conv2DLayer : public mlfo::layer::Layer<T> {
private:
    std::size_t c_;
    std::size_t h_;
    std::size_t w_;
    std::size_t r_;
    std::size_t s_;
    std::size_t k_;
    std::size_t stride_h_;
    std::size_t stride_w_;
    std::size_t padding_h_;
    std::size_t padding_w_;
    std::size_t p_;
    std::size_t q_;
public:
    Conv2DLayer(
        std::size_t c,
        std::size_t h,
        std::size_t w,
        std::size_t r,
        std::size_t s,
        std::size_t k,
        std::size_t stride_h = 1,
        std::size_t stride_w = 1,
        std::size_t padding_h = 0,
        std::size_t padding_w = 0
    ) :
    c_(c),
    h_(h),
    w_(w),
    r_(r),
    s_(s),
    k_(k),
    stride_h_(stride_h),
    stride_w_(stride_w),
    padding_h_(padding_h),
    padding_w_(padding_w)
    {
        // validate parameters
        if (c_ == 0 || h_ == 0 || w_ == 0 || r_ == 0 || s_ == 0 || k_ == 0) {
            throw std::invalid_argument(
                "[mlfo::layer::Conv2DLayer] All dimensions must be greater than zero"
            );
        }
        if (stride_h_ == 0 || stride_w_ == 0) {
            throw std::invalid_argument(
                "[mlfo::layer::Conv2DLayer] Stride must be greater than zero"
            );
        }
        if (h_ + 2 * padding_h_ < r_ || w_ + 2 * padding_w_ < s_) {
            throw std::invalid_argument(
                "[mlfo::layer::Conv2DLayer] Kernel size must be less than or equal to input size + 2 * padding"
            );
        }
        // calculate P and Q
        p_ = (h_ + 2 * padding_h_ - r_) / stride_h_ + 1;
        q_ = (w_ + 2 * padding_w_ - s_) / stride_w_ + 1;
        this->input_shape_ = {c_, h_, w_};
        this->output_shape_ = {k_, p_, q_};
    }

    mlfo::graph::Node<mlfo::tensor::Tensor<T>>& build(
        mlfo::graph::Graph<mlfo::tensor::Tensor<T>>& graph,
        mlfo::graph::Node<mlfo::tensor::Tensor<T>>& input
    ) override {
        auto& weight_node = graph.add_node(
            mlfo::tensor::Tensor<T>(
                1,
                std::vector<std::size_t>{k_, c_, r_, s_}
            )
        );
        auto& bias_node = graph.add_node(
            mlfo::tensor::Tensor<T>(1, std::vector<std::size_t>{k_})
        );
        auto& prebias_node = graph.add_node(
            mlfo::tensor::Tensor<T>(
                1,
                std::vector<std::size_t>{k_, p_, q_}
            )
        );
        auto& output_node = graph.add_node(
            mlfo::tensor::Tensor<T>(
                1,
                std::vector<std::size_t>{k_, p_, q_}
            )
        );
        // conv2d operation: input -> prebias
        prebias_node.add_predecessors(
            {input, weight_node},
            [
                &prebias_node_tensor = prebias_node.data(),
                &input_node_tensor = input.data(),
                &weight_node_tensor = weight_node.data()
            ](mlfo::tensor::ops::DIRECTION direction) -> void {
                mlfo::tensor::ops::Conv2D<T>::call(
                    direction,
                    prebias_node_tensor,
                    input_node_tensor,
                    weight_node_tensor,
                    stride_h_,
                    stride_w_,
                    padding_h_,
                    padding_w_
                );
            }
        );
        // bias operation: prebias -> output
        output_node.add_predecessors(
            {prebias_node, bias_node},
            [
                &output_node_tensor = output_node.data(),
                &prebias_node_tensor = prebias_node.data(),
                &bias_node_tensor = bias_node.data()
            ](mlfo::tensor::ops::DIRECTION direction) -> void {
                mlfo::tensor::ops::Add<T>::call(
                    direction,
                    output_node_tensor,
                    prebias_node_tensor,
                    bias_node_tensor
                );
            }
        );
        return output_node;
    }
};

} // namespace mlfo::layer

#endif // CONV2D_LAYER_HPP
