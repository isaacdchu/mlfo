#ifndef DENSE_LAYER_HPP
#define DENSE_LAYER_HPP

#include <vector>

#include "../misc/concepts.hpp"
#include "../graph/graph.hpp"
#include "../tensor/tensor.hpp"
#include "../tensor/ops.hpp"
#include "layer.hpp"

namespace mlfo::layer {

template <mlfo::misc::Number T>
class DenseLayer : public mlfo::layer::Layer<T> {
private:
    std::size_t input_size_;
    std::size_t output_size_;
public:
    DenseLayer(std::size_t input_size, std::size_t output_size) :
    input_size_(input_size),
    output_size_(output_size)
    {
        // activations are row vectors (1, n) since Mul requires 2D tensors
        this->input_shape_ = {1, input_size_};
        this->output_shape_ = {1, output_size_};
    }

    mlfo::graph::Node<mlfo::tensor::Tensor<T>>& build(
        mlfo::graph::Graph<mlfo::tensor::Tensor<T>>& graph,
        mlfo::graph::Node<mlfo::tensor::Tensor<T>>& input
    ) override {
        auto& weight_node = graph.add_node(
            mlfo::tensor::Tensor<T>(1, std::vector<std::size_t>{input_size_, output_size_})
        );
        // activations are row vectors (1, n) since Mul requires 2D tensors
        auto& bias_node = graph.add_node(
            mlfo::tensor::Tensor<T>(1, std::vector<std::size_t>{1, output_size_})
        );
        auto& prebias_node = graph.add_node(
            mlfo::tensor::Tensor<T>(1, std::vector<std::size_t>{1, output_size_})
        );
        auto& output_node = graph.add_node(
            mlfo::tensor::Tensor<T>(1, std::vector<std::size_t>{1, output_size_})
        );
        prebias_node.add_predecessors(
            {input, weight_node},
            [
                &prebias_node_tensor = prebias_node.data(),
                &input_node_tensor = input.data(),
                &weight_node_tensor = weight_node.data()
            ](mlfo::tensor::ops::DIRECTION direction) -> void {
                // input * weight
                mlfo::tensor::ops::Mul<T>::call(
                    direction,
                    prebias_node_tensor,
                    input_node_tensor,
                    weight_node_tensor
                );
            }
        );
        output_node.add_predecessors(
            {prebias_node, bias_node},
            [
                &output_node_tensor = output_node.data(),
                &prebias_node_tensor = prebias_node.data(),
                &bias_node_tensor = bias_node.data()
            ](mlfo::tensor::ops::DIRECTION direction) -> void {
                // prebias + bias
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

#endif // DENSE_LAYER_HPP
