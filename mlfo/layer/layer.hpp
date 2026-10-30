#ifndef LAYER_HPP
#define LAYER_HPP

#include <cstddef>
#include <iostream>
#include <vector>

#include "../graph/graph.hpp"
#include "../graph/node.hpp"
#include "../misc/concepts.hpp"
#include "../tensor/tensor.hpp"

namespace mlfo::layer {

// A layer does not own any nodes. It adds its nodes to a graph owned by the model
// and keeps non-owning pointers to the ones it needs (parameters, output).
template <mlfo::misc::Number T>
class Layer {
protected:
    // unbatched shapes of the tensors this layer consumes and produces, set by the derived class
    std::vector<std::size_t> input_shape_;
    std::vector<std::size_t> output_shape_;
    // the parameters of this layer, set by build() and used by an optimizer
    std::vector<std::reference_wrapper<mlfo::tensor::Tensor<T>>> parameters_;
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
        mlfo::graph::Node<mlfo::tensor::Tensor<T>>& input_node
    ) = 0;

    virtual const std::vector<std::reference_wrapper<mlfo::tensor::Tensor<T>>>& parameters() const = 0;

    // layer type and parameters; initializer not needed
    // format:
    // DenseLayer
    // 2,3
    // 0.1,0.2,0.3,0.4,0.5,0.6
    // 0.1,0.2,0.3
    // each derived class also has a static deserialize(std::istream&) that reads back everything
    // after the type line, which Model::deserialize() reads to pick the class
    virtual void serialize(std::ostream& os) const = 0;
};

} // namespace mlfo::layer

#endif // LAYER_HPP
