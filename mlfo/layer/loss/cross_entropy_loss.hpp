#ifndef CROSS_ENTROPY_LOSS_HPP
#define CROSS_ENTROPY_LOSS_HPP

#include <cstddef>

#include "../../graph/graph.hpp"
#include "../../layer/layer.hpp"
#include "../../misc/concepts.hpp"
#include "../../tensor/loss/cross_entropy.hpp"
#include "../../tensor/ops.hpp"
#include "../../tensor/tensor.hpp"
#include "loss_layer.hpp"

namespace mlfo::layer::loss {

// softmax cross entropy, so the input should be raw logits with no softmax applied
template <mlfo::misc::Number T>
class CrossEntropyLoss : public mlfo::layer::loss::LossLayer<T> {
public:
    // the input shape is taken from the input node when built
    CrossEntropyLoss() {
        // inputs are logits from model and a distribution over classes from labels
        // output is the cross entropy
        this->output_shape_ = {1};
    }

    mlfo::graph::Node<mlfo::tensor::Tensor<T>>& build(
        mlfo::graph::Graph<mlfo::tensor::Tensor<T>>& graph,
        mlfo::graph::Node<mlfo::tensor::Tensor<T>>& input_node
    ) override {
        this->input_shape_ = input_node.data().shape();
        auto& output_node = graph.add_node(
            mlfo::tensor::Tensor<T>(1, std::vector<std::size_t>{1})
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
                // softmax cross entropy
                mlfo::tensor::loss::CrossEntropy<T>::call(
                    direction,
                    output_node_tensor,
                    input_node_tensor,
                    label_node_tensor
                );
            }
        );
        return output_node;
    }
};

} // namespace mlfo::layer::loss

#endif // CROSS_ENTROPY_LOSS_HPP
