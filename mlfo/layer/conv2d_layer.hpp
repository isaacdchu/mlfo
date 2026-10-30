#ifndef CONV2D_LAYER_HPP
#define CONV2D_LAYER_HPP

#include <cstddef>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

#include "../graph/graph.hpp"
#include "../initializer/constant.hpp"
#include "../initializer/initializer.hpp"
#include "../misc/concepts.hpp"
#include "../tensor/ops.hpp"
#include "../tensor/tensor.hpp"
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
    std::size_t k_;
    std::size_t r_;
    std::size_t s_;
    std::size_t stride_h_;
    std::size_t stride_w_;
    std::size_t padding_h_;
    std::size_t padding_w_;
    std::shared_ptr<const mlfo::initializer::Initializer<T>> weight_initializer_;
    std::shared_ptr<const mlfo::initializer::Initializer<T>> bias_initializer_;
public:
    // the input shape (C, H, W) is taken from the input node when built
    Conv2DLayer(
        std::size_t k,
        std::size_t r,
        std::size_t s,
        std::size_t stride_h = 1,
        std::size_t stride_w = 1,
        std::size_t padding_h = 0,
        std::size_t padding_w = 0,
        std::shared_ptr<const mlfo::initializer::Initializer<T>> weight_initializer =
            std::make_shared<mlfo::initializer::Zeros<T>>(),
        std::shared_ptr<const mlfo::initializer::Initializer<T>> bias_initializer =
            std::make_shared<mlfo::initializer::Zeros<T>>()
    ) :
    k_(k),
    r_(r),
    s_(s), 
    stride_h_(stride_h),
    stride_w_(stride_w),
    padding_h_(padding_h),
    padding_w_(padding_w),
    weight_initializer_(std::move(weight_initializer)),
    bias_initializer_(std::move(bias_initializer))
    {
        // validate parameters
        if (r_ == 0 || s_ == 0 || k_ == 0) {
            throw std::invalid_argument(
                "[mlfo::layer::Conv2DLayer] All dimensions must be greater than zero"
            );
        }
        if (stride_h_ == 0 || stride_w_ == 0) {
            throw std::invalid_argument(
                "[mlfo::layer::Conv2DLayer] Stride must be greater than zero"
            );
        }
        if (!weight_initializer_ || !bias_initializer_) {
            throw std::invalid_argument(
                "[mlfo::layer::Conv2DLayer] Initializers cannot be null"
            );
        }
    }

    mlfo::graph::Node<mlfo::tensor::Tensor<T>>& build(
        mlfo::graph::Graph<mlfo::tensor::Tensor<T>>& graph,
        mlfo::graph::Node<mlfo::tensor::Tensor<T>>& input_node
    ) override {
        const auto& input_shape = input_node.data().shape();
        if (input_shape.size() != 3) {
            throw std::invalid_argument(
                "[mlfo::layer::Conv2DLayer::build] Input must have shape (C, H, W)"
            );
        }
        const std::size_t c = input_shape[0];
        const std::size_t h = input_shape[1];
        const std::size_t w = input_shape[2];
        if (c == 0 || h == 0 || w == 0) {
            throw std::invalid_argument(
                "[mlfo::layer::Conv2DLayer::build] Input dimensions must be greater than zero"
            );
        }
        if (h + 2 * padding_h_ < r_ || w + 2 * padding_w_ < s_) {
            throw std::invalid_argument(
                "[mlfo::layer::Conv2DLayer::build] Kernel size must be less than or equal to input size + 2 * padding"
            );
        }
        // calculate P and Q
        const std::size_t p = (h + 2 * padding_h_ - r_) / stride_h_ + 1;
        const std::size_t q = (w + 2 * padding_w_ - s_) / stride_w_ + 1;
        this->input_shape_ = input_shape;
        this->output_shape_ = {k_, p, q};
        auto& weight_node = graph.add_node(
            mlfo::tensor::Tensor<T>(
                1,
                std::vector<std::size_t>{k_, c, r_, s_}
            )
        );
        auto& bias_node = graph.add_node(
            mlfo::tensor::Tensor<T>(1, std::vector<std::size_t>{k_})
        );
        // each output value sums one C x R x S window, and each input value is seen by
        // R x S kernel positions of each of the K kernels
        weight_initializer_->initialize(weight_node.data(), c * r_ * s_, k_ * r_ * s_);
        bias_initializer_->initialize(bias_node.data(), c * r_ * s_, k_ * r_ * s_);
        // activations have one batch per sample, the parameters are broadcast over them
        auto& prebias_node = graph.add_node(
            mlfo::tensor::Tensor<T>(
                input_node.data().batch_size(),
                this->output_shape_
            )
        );
        auto& output_node = graph.add_node(
            mlfo::tensor::Tensor<T>(
                input_node.data().batch_size(),
                this->output_shape_
            )
        );
        // conv2d operation: input -> prebias
        prebias_node.add_predecessors(
            {input_node, weight_node},
            [
                &prebias_node_tensor = prebias_node.data(),
                &input_node_tensor = input_node.data(),
                &weight_node_tensor = weight_node.data(),
                this
            ](mlfo::tensor::ops::DIRECTION direction) -> void {
                mlfo::tensor::ops::Conv2D<T>::call(
                    direction,
                    prebias_node_tensor,
                    input_node_tensor,
                    weight_node_tensor,
                    this->stride_h_,
                    this->stride_w_,
                    this->padding_h_,
                    this->padding_w_
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
                mlfo::tensor::ops::ChannelAdd<T>::call(
                    direction,
                    output_node_tensor,
                    prebias_node_tensor,
                    bias_node_tensor
                );
            }
        );
        this->parameters_.clear();
        this->parameters_.push_back(weight_node.data());
        this->parameters_.push_back(bias_node.data());
        return output_node;
    }

    const std::vector<std::reference_wrapper<mlfo::tensor::Tensor<T>>>& parameters() const override {
        return this->parameters_;
    }
};

} // namespace mlfo::layer

#endif // CONV2D_LAYER_HPP
