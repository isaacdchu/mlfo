#ifndef LOSS_LAYER_HPP
#define LOSS_LAYER_HPP

#include <cstddef>
#include <memory>
#include <stdexcept>
#include <vector>

#include "../../graph/graph.hpp"
#include "../../layer/layer.hpp"
#include "../../misc/concepts.hpp"

namespace mlfo::layer::loss {

template <mlfo::misc::Number T>
class LossLayer : public mlfo::layer::Layer<T> {
protected:
    // non-owning, set by build(); the node is owned by the graph
    mlfo::graph::Node<mlfo::tensor::Tensor<T>>* label_node_ = nullptr;
public:
    virtual ~LossLayer() = default;

    // the node holding the labels the input is compared against
    mlfo::graph::Node<mlfo::tensor::Tensor<T>>& label_node() const {
        if (label_node_ == nullptr) {
            throw std::logic_error(
                "[mlfo::layer::loss::LossLayer::label_node] Layer has not been built"
            );
        }
        return *label_node_;
    }

    // adds this layer's nodes to graph, taking input as the layer's input node
    // returns the layer's output node, which is the next layer's input node
    virtual mlfo::graph::Node<mlfo::tensor::Tensor<T>>& build(
        mlfo::graph::Graph<mlfo::tensor::Tensor<T>>& graph,
        mlfo::graph::Node<mlfo::tensor::Tensor<T>>& input_node
    ) override = 0;

    // loss layers have no parameters, so this is always empty
    const std::vector<std::reference_wrapper<mlfo::tensor::Tensor<T>>>& parameters() const override {
        return this->parameters_;
    }
};

} // namespace mlfo::layer::loss

#endif // LOSS_LAYER_HPP
