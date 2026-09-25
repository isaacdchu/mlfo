#ifndef DENSE_HPP
#define DENSE_HPP

#include <vector>

#include "../misc/concepts.hpp"
#include "../graph/graph.hpp"
#include "../tensor/tensor.hpp"
#include "../tensor/ops.hpp"
#include "layer.hpp"

namespace mlfo::layer {

template <mlfo::misc::Number T>
class DenseLayer : public mlfo::layer::Layer {
private:
    mlfo::graph::Graph<mlfo::tensor::Tensor<T>> graph_;
public:
    DenseLayer(std::size_t input_size, std::size_t output_size) {
        auto input_node = std::make_unique<mlfo::graph::Node<mlfo::tensor::Tensor<T>>>(
            std::make_unique<mlfo::tensor::Tensor<T>>(1, std::vector<std::size_t>{input_size})
        );
        auto weight_node = std::make_unique<mlfo::graph::Node<mlfo::tensor::Tensor<T>>>(
            std::make_unique<mlfo::tensor::Tensor<T>>(
                1,
                std::vector<std::size_t>{input_size, output_size}
            )
        );
        auto bias_node = std::make_unique<mlfo::graph::Node<mlfo::tensor::Tensor<T>>>(
            std::make_unique<mlfo::tensor::Tensor<T>>(1, std::vector<std::size_t>{output_size})
        );
        auto prebias_node = std::make_unique<mlfo::graph::Node<mlfo::tensor::Tensor<T>>>(
            std::make_unique<mlfo::tensor::Tensor<T>>(1, std::vector<std::size_t>{output_size})
        );
        auto output_node = std::make_unique<mlfo::graph::Node<mlfo::tensor::Tensor<T>>>(
            std::make_unique<mlfo::tensor::Tensor<T>>(1, std::vector<std::size_t>{output_size})
        );
        output_node->add_predecessors(
            {*prebias_node, *bias_node},
            [
                &output_node_tensor = output_node->data(),
                &prebias_node_tensor = prebias_node->data(),
                &bias_node_tensor = bias_node->data()
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
        prebias_node->add_predecessors(
            {*input_node, *weight_node},
            [](mlfo::tensor::ops::DIRECTION direction) -> void {
                // todo
            }
        );
        graph_ = mlfo::graph::Graph<mlfo::tensor::Tensor<T>>(
            {
                std::move(input_node),
                std::move(weight_node),
                std::move(bias_node),
                std::move(output_node)
            }
        );
    }

    void forward() override {
        for (auto it = graph_.tail_begin(); it != graph_.tail_end(); ++it) {
            auto& node = *it;
            node.forward();
        }
    }

    void backward() override {
        for (auto it = graph_.head_begin(); it != graph_.head_end(); ++it) {
            auto& node = *it;
            node.backward();
        }
    }
};

} // namespace mlfo::layer

#endif // DENSE_HPP
