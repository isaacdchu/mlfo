#ifndef FLATTEN_LAYER_HPP
#define FLATTEN_LAYER_HPP

#include <cstddef>
#include <iostream>
#include <memory>
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

    ~FlattenLayer() override = default;

    mlfo::graph::Node<mlfo::tensor::Tensor<T>>& build(
        mlfo::graph::Graph<mlfo::tensor::Tensor<T>>& graph,
        mlfo::graph::Node<mlfo::tensor::Tensor<T>>& input_node
    ) override {
        this->input_shape_ = input_node.data().shape();
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

    void serialize(std::ostream& os) const override {
        os << "FlattenLayer\n";
    }

    // nothing follows the type line, as the layer has no configuration
    static std::unique_ptr<FlattenLayer> deserialize(std::istream& is) {
        return std::make_unique<FlattenLayer>();
    }
};

} // namespace mlfo::layer

#endif // FLATTEN_LAYER_HPP
