#ifndef MAX_POOL_HPP
#define MAX_POOL_HPP

#include <cstddef>
#include <optional>
#include <stdexcept>
#include <vector>

#include "../../graph/graph.hpp"
#include "../../graph/node.hpp"
#include "../../misc/concepts.hpp"
#include "../../tensor/pool.hpp"
#include "../../tensor/tensor.hpp"
#include "pool_layer.hpp"

namespace mlfo::layer::pool {

template <mlfo::misc::Number T>
class MaxPool : public PoolLayer<T> {
private:
    // pooling window size (r, s)
    std::size_t r_;
    std::size_t s_;
    // pooling stride, defaults to the window size (non-overlapping windows)
    std::size_t stride_h_;
    std::size_t stride_w_;
public:
    // the input dimensions (h, w) are taken from the last 2 dims of the input node when built
    MaxPool(
        std::size_t r,
        std::size_t s,
        std::optional<std::size_t> stride_h = std::nullopt,
        std::optional<std::size_t> stride_w = std::nullopt
    ) :
    r_(r),
    s_(s),
    stride_h_(stride_h.value_or(r)),
    stride_w_(stride_w.value_or(s))
    {
        if (r == 0 || s == 0) {
            throw std::invalid_argument(
                "[layer::pool::max] Pooling window size must be greater than 0"
            );
        }
        if (stride_h_ == 0 || stride_w_ == 0) {
            throw std::invalid_argument(
                "[layer::pool::max] Pooling stride must be greater than 0"
            );
        }
    }

    virtual ~MaxPool() = default;

    // adds this layer's nodes to graph, taking input as the layer's input node
    // returns the layer's output node, which is the next layer's input node
    virtual mlfo::graph::Node<mlfo::tensor::Tensor<T>>& build(
        mlfo::graph::Graph<mlfo::tensor::Tensor<T>>& graph,
        mlfo::graph::Node<mlfo::tensor::Tensor<T>>& input_node
    ) override {
        // input tensor must have h and w as last 2 dims
        const auto& input_shape = input_node.data().shape();
        if (input_shape.size() < 2) {
            throw std::invalid_argument(
                "[layer::pool::max] Input must have at least 2 dimensions"
            );
        }
        const std::size_t h = input_shape[input_shape.size() - 2];
        const std::size_t w = input_shape[input_shape.size() - 1];
        if (h == 0 || w == 0) {
            throw std::invalid_argument(
                "[layer::pool::max] Input dimensions must be greater than 0"
            );
        }
        this->input_shape_ = input_shape;
        this->output_shape_ = input_shape;
        this->output_shape_[this->output_shape_.size() - 2] = (
            mlfo::tensor::pool::Max<T>::output_size(h, this->r_, this->stride_h_)
        );
        this->output_shape_[this->output_shape_.size() - 1] = (
            mlfo::tensor::pool::Max<T>::output_size(w, this->s_, this->stride_w_)
        );
        auto& output_node = graph.add_node(
            mlfo::tensor::Tensor<T>(
                input_node.data().batch_size(),
                this->output_shape_
            )
        );
        output_node.add_predecessor(
            input_node,
            [
                &output_node_tensor = output_node.data(),
                &input_node_tensor = input_node.data(),
                this
            ](mlfo::tensor::ops::DIRECTION direction) -> void {
                mlfo::tensor::pool::Max<T>::call(
                    direction,
                    output_node_tensor,
                    input_node_tensor,
                    this->r_,
                    this->s_,
                    this->stride_h_,
                    this->stride_w_
                );
            }
        );
        return output_node;
    }
};

} // namespace mlfo::layer::pool

#endif // MAX_POOL_HPP
