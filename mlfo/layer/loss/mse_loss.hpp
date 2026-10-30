#ifndef MSE_LOSS_HPP
#define MSE_LOSS_HPP

#include <cstddef>
#include <iostream>
#include <memory>

#include "../../graph/graph.hpp"
#include "../../layer/layer.hpp"
#include "../../misc/concepts.hpp"
#include "../../tensor/loss/mse.hpp"
#include "../../tensor/ops.hpp"
#include "../../tensor/tensor.hpp"
#include "loss_layer.hpp"

namespace mlfo::layer::loss {

template <mlfo::misc::Number T>
class MSELoss : public mlfo::layer::loss::LossLayer<T> {
public:
    // the input shape is taken from the input node when built
    MSELoss() {
        // inputs are from model and from labels
        // output is the mean squared error
        // how to handle multiple inputs?
        // for now, just sum the errors and return a single value
        this->output_shape_ = {1};
    }

    mlfo::graph::Node<mlfo::tensor::Tensor<T>>& build(
        mlfo::graph::Graph<mlfo::tensor::Tensor<T>>& graph,
        mlfo::graph::Node<mlfo::tensor::Tensor<T>>& input_node
    ) override {
        this->input_shape_ = input_node.data().shape();
        // one loss value per batch, the model averages them in backward()
        auto& output_node = graph.add_node(
            mlfo::tensor::Tensor<T>(input_node.data().batch_size(), std::vector<std::size_t>{1})
        );
        auto& label_node = graph.add_node(
            // labels are compared element-wise, so they take the input node's shape
            mlfo::tensor::Tensor<T>(input_node.data().batch_size(), input_node.data().shape())
        );
        this->label_node_ = &label_node;
        output_node.add_predecessors(
            {input_node, label_node},
            [
                &output_node_tensor = output_node.data(),
                &input_node_tensor = input_node.data(),
                &label_node_tensor = label_node.data()
            ](mlfo::tensor::ops::DIRECTION direction) -> void {
                // mean squared error
                mlfo::tensor::loss::MSE<T>::call(
                    direction,
                    output_node_tensor,
                    input_node_tensor,
                    label_node_tensor
                );
            }
        );
        return output_node;
    }

    void serialize(std::ostream& os) const override {
        os << "MSELoss\n";
    }

    // nothing follows the type line, as the layer has no configuration
    static std::unique_ptr<MSELoss> deserialize(std::istream& is) {
        return std::make_unique<MSELoss>();
    }
};

} // namespace mlfo::loss

#endif // MSE_LOSS_HPP
