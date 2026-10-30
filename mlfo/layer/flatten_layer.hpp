#ifndef FLATTEN_LAYER_HPP
#define FLATTEN_LAYER_HPP

#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>

#include "../graph/graph.hpp"
#include "../graph/node.hpp"
#include "../misc/concepts.hpp"
#include "../tensor/ops/reshape.hpp"
#include "../tensor/tensor.hpp"
#include "layer.hpp"

namespace mlfo::layer {

// flattens an input of any shape into a (1, n) row vector, the shape DenseLayer consumes
template <mlfo::misc::Number T>
class FlattenLayer : public mlfo::layer::Layer<T> {
public:
    // the input shape is taken from the input node when built
    FlattenLayer() = default;

    // the input shape is only needed up front when this is the first layer of a model
    explicit FlattenLayer(std::vector<std::size_t> input_shape) {
        if (input_shape.empty()) {
            throw std::invalid_argument(
                "[mlfo::layer::FlattenLayer] Input shape cannot be empty"
            );
        }
        std::size_t size = 1;
        for (const std::size_t dim : input_shape) {
            size *= dim;
        }
        if (size == 0) {
            throw std::invalid_argument(
                "[mlfo::layer::FlattenLayer] Input dimensions must be greater than 0"
            );
        }
        this->input_shape_ = std::move(input_shape);
        this->output_shape_ = {1, size};
    }

    ~FlattenLayer() override = default;

    mlfo::graph::Node<mlfo::tensor::Tensor<T>>& build(
        mlfo::graph::Graph<mlfo::tensor::Tensor<T>>& graph,
        mlfo::graph::Node<mlfo::tensor::Tensor<T>>& input_node
    ) override {
        const auto& input_shape = input_node.data().shape();
        if (!this->input_shape_.empty() && this->input_shape_ != input_shape) {
            throw std::invalid_argument(
                "[mlfo::layer::FlattenLayer::build] Input node shape does not match the given input shape"
            );
        }
        this->input_shape_ = input_shape;
        // activations are row vectors (1, n) since Mul requires 2D tensors
        this->output_shape_ = {1, input_node.data().unbatched_size()};
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
                &input_node_tensor = input_node.data()
            ](mlfo::tensor::ops::DIRECTION direction) -> void {
                mlfo::tensor::ops::Reshape<T>::call(
                    direction,
                    output_node_tensor,
                    input_node_tensor
                );
            }
        );
        return output_node;
    }

    // flatten layers have no parameters, so this is always empty
    const std::vector<std::reference_wrapper<mlfo::tensor::Tensor<T>>>& parameters() const override {
        return this->parameters_;
    }
};

} // namespace mlfo::layer

#endif // FLATTEN_LAYER_HPP
