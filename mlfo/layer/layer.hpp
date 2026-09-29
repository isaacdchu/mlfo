#ifndef LAYER_HPP
#define LAYER_HPP

#include <cstddef>
#include <vector>

#include "../tensor/tensor.hpp"
#include "../graph/graph.hpp"
#include "../graph/node.hpp"
#include "../misc/concepts.hpp"

namespace mlfo::layer {

// A layer does not own any nodes. It adds its nodes to a graph owned by the model
// and keeps non-owning pointers to the ones it needs (parameters, output).
template <mlfo::misc::Number T>
class Layer {
protected:
    // shapes of the tensors this layer consumes and produces, set by the derived class
    std::vector<std::size_t> input_shape_;
    std::vector<std::size_t> output_shape_;
public:
    virtual ~Layer() = default;

    const std::vector<std::size_t>& input_shape() const {
        return input_shape_;
    }

    const std::vector<std::size_t>& output_shape() const {
        return output_shape_;
    }

    // adds this layer's nodes to graph, taking input as the layer's input node
    // returns the layer's output node, which is the next layer's input node
    virtual mlfo::graph::Node<mlfo::tensor::Tensor<T>>& build(
        mlfo::graph::Graph<mlfo::tensor::Tensor<T>>& graph,
        mlfo::graph::Node<mlfo::tensor::Tensor<T>>& input
    ) = 0;
};

} // namespace mlfo::layer

#endif // LAYER_HPP
