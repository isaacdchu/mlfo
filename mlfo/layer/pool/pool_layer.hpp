#ifndef POOL_LAYER_HPP
#define POOL_LAYER_HPP

#include <cstddef>
#include <memory>
#include <stdexcept>
#include <vector>

#include "../../graph/graph.hpp"
#include "../../layer/layer.hpp"
#include "../../misc/concepts.hpp"

namespace mlfo::layer::pool {

template <mlfo::misc::Number T>
class PoolLayer : public mlfo::layer::Layer<T> {
public:
    virtual ~PoolLayer() = default;

    // adds this layer's nodes to graph, taking input as the layer's input node
    // returns the layer's output node, which is the next layer's input node
    virtual mlfo::graph::Node<mlfo::tensor::Tensor<T>>& build(
        mlfo::graph::Graph<mlfo::tensor::Tensor<T>>& graph,
        mlfo::graph::Node<mlfo::tensor::Tensor<T>>& input_node
    ) override = 0;

    // pool layers have no parameters, so this is always empty
    const std::vector<std::reference_wrapper<mlfo::tensor::Tensor<T>>>& parameters() const override {
        return this->parameters_;
    }
};

} // namespace mlfo::layer::pool

#endif // POOL_LAYER_HPP
