#ifndef MODEL_HPP
#define MODEL_HPP

#include <vector>
#include <memory>
#include <functional>
#include <stdexcept>
#include <utility>

#include "../layer/layer.hpp"
#include "../graph/graph.hpp"
#include "../tensor/tensor.hpp"
#include "../misc/concepts.hpp"

namespace mlfo::model {

// owns the layers and the single graph containing every node of the network
template <misc::Number T>
class Model {
protected:
    std::vector<std::unique_ptr<mlfo::layer::Layer<T>>> layers_;
    mlfo::graph::Graph<mlfo::tensor::Tensor<T>> graph_;
    // non-owning, set by build(); the nodes are owned by graph_
    std::vector<mlfo::graph::Node<mlfo::tensor::Tensor<T>>*> inputs_;
    std::vector<mlfo::graph::Node<mlfo::tensor::Tensor<T>>*> outputs_;
    // cached views of the output tensors, set by build()
    std::vector<std::reference_wrapper<const mlfo::tensor::Tensor<T>>> output_tensors_;
public:
    Model(std::vector<std::unique_ptr<mlfo::layer::Layer<T>>>&& layers) :
    layers_(std::move(layers))
    {
        // nothing
    }
    // creates the input node (shaped by the first layer's input shape), chains each layer's output
    // into the next layer's input, then sorts the graph
    void build() {
        if (layers_.empty()) {
            throw std::logic_error("[mlfo::model::Model] Model has no layers");
        }
        inputs_.clear();
        outputs_.clear();
        output_tensors_.clear();
        inputs_.push_back(&graph_.add_node(mlfo::tensor::Tensor<T>(1, layers_.front()->input_shape())));
        mlfo::graph::Node<mlfo::tensor::Tensor<T>>* current = inputs_.front();
        for (auto& layer : layers_) {
            current = &layer->build(graph_, *current);
        }
        outputs_.push_back(current);
        for (const auto* node : outputs_) {
            output_tensors_.push_back(node->data());
        }
        // topologically sort the graph
        graph_.finalize();
    }

    // sets the values of the input tensors, one flattened vector per input tensor
    // the vectors are moved into the tensors, never copied. rvalue only, so an lvalue
    // must be passed with std::move. the next forward() recomputes the whole graph
    void set_inputs(std::vector<std::vector<T>>&& values) {
        if (values.size() != inputs_.size()) {
            throw std::invalid_argument("[mlfo::model::Model] Wrong number of inputs");
        }
        for (std::size_t i = 0; i < inputs_.size(); i++) {
            inputs_[i]->data().set_values(std::move(values[i]));
        }
        graph_.invalidate();
    }

    // the output tensors, valid until the next build()
    const std::vector<std::reference_wrapper<const mlfo::tensor::Tensor<T>>>& outputs() const {
        return output_tensors_;
    }

    void forward() {
        graph_.forward();
    }

    void backward() {
        graph_.backward();
    }
};

} // namespace mlfo::model

#endif // MODEL_HPP
