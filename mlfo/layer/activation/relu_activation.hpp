#ifndef RELU_ACTIVATION_HPP
#define RELU_ACTIVATION_HPP

#include <algorithm>
#include <concepts>
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <vector>

#include "../../graph/graph.hpp"
#include "../../graph/node.hpp"
#include "../../layer/activation/activation_layer.hpp"
#include "../../misc/concepts.hpp"
#include "../../tensor/activation/relu.hpp"
#include "../../tensor/tensor.hpp"

namespace mlfo::layer::activation {

template <mlfo::misc::Number T>
class ReLUActivation : public mlfo::layer::activation::ActivationLayer<T> {
public:
    ReLUActivation() = default;
    ~ReLUActivation() override = default;

    mlfo::graph::Node<mlfo::tensor::Tensor<T>>& build(
        mlfo::graph::Graph<mlfo::tensor::Tensor<T>>& graph,
        mlfo::graph::Node<mlfo::tensor::Tensor<T>>& input_node
    ) override {
        this->input_shape_ = input_node.data().shape();
        this->output_shape_ = input_node.data().shape();
        auto& output_node = graph.add_node(
            mlfo::tensor::Tensor<T>(
                input_node.data().batch_size(),
                input_node.data().shape()
            )
        );
        output_node.add_predecessor(
            input_node,
            [
                &output_node_tensor = output_node.data(),
                &input_node_tensor = input_node.data()
            ](mlfo::tensor::ops::DIRECTION direction) -> void {
                mlfo::tensor::activation::ReLU<T>::call(
                    direction,
                    output_node_tensor,
                    input_node_tensor
                );
            }
        );
        return output_node;
    }

    void serialize(std::ostream& os) const override {
        os << "ReLUActivation\n";
    }

    // nothing follows the type line, as the layer has no configuration
    static std::unique_ptr<ReLUActivation> deserialize(std::istream& is) {
        return std::make_unique<ReLUActivation>();
    }
};

} // namespace mlfo::layer::activation

#endif // RELU_ACTIVATION_HPP
