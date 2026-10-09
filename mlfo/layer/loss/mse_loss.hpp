#ifndef MSE_LOSS_HPP
#define MSE_LOSS_HPP

#include <cstddef>

#include "../../misc/concepts.hpp"
#include "../../graph/graph.hpp"
#include "../../tensor/tensor.hpp"
#include "../../layer/layer.hpp"
#include "loss_layer.hpp"

namespace mlfo::layer::loss {

template <mlfo::misc::Number T>
class MSELoss : public mlfo::layer::loss::LossLayer<T> {
public:
    MSELoss(std::size_t input_size) {
        // inputs are from model and from labels
        this->input_shape_ = {input_size * 2};
        // output is the mean squared error
        // how to handle multiple inputs?
        // for now, just sum the errors and return a single value
        this->output_shape_ = {1};
    }

    mlfo::graph::Node<mlfo::tensor::Tensor<T>>& build(
        mlfo::graph::Graph<mlfo::tensor::Tensor<T>>& graph,
        mlfo::graph::Node<mlfo::tensor::Tensor<T>>& input
    ) override {
        auto& output_node = graph.add_node(
            mlfo::tensor::Tensor<T>(1, std::vector<std::size_t>{1})
        );
        output_node.add_predecessors(
            {input},
            [
                &output_node_tensor = output_node.data(),
                &input_node_tensor = input.data()
            ](mlfo::tensor::ops::DIRECTION direction) -> void {
                // mean squared error
                mlfo::tensor::ops::MSELoss<T>::call(
                    direction,
                    output_node_tensor,
                    input_node_tensor
                );
            }
        );
        return output_node;
    }
};

} // namespace mlfo::loss

#endif // MSE_LOSS_HPP
