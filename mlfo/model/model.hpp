#ifndef MODEL_HPP
#define MODEL_HPP

#include <algorithm>
#include <cstddef>
#include <functional>
#include <initializer_list>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "../graph/graph.hpp"
#include "../layer/layer.hpp"
#include "../layer/loss/loss_layer.hpp"
#include "../misc/concepts.hpp"
#include "../tensor/tensor.hpp"

namespace mlfo::model {

// owns the layers and the single graph containing every node of the network
template <misc::Number T>
class Model {
protected:
    // unbatched shape of the model's input, each layer takes its input shape from its predecessor
    std::vector<std::size_t> input_shape_;
    std::vector<std::unique_ptr<mlfo::layer::Layer<T>>> layers_;
    // optional loss layer
    std::unique_ptr<mlfo::layer::loss::LossLayer<T>> loss_;
    mlfo::graph::Graph<mlfo::tensor::Tensor<T>> graph_;
    // non-owning, set by build(); the nodes are owned by graph_
    std::vector<mlfo::graph::Node<mlfo::tensor::Tensor<T>>*> inputs_;
    std::vector<mlfo::graph::Node<mlfo::tensor::Tensor<T>>*> outputs_;
    // cached views of the output tensors, set by build()
    std::vector<std::reference_wrapper<const mlfo::tensor::Tensor<T>>> output_tensors_;
    // model parameters, set by build() and used by an optimizer
    std::vector<std::reference_wrapper<mlfo::tensor::Tensor<T>>> parameters_;
    // every tensor in the graph that is not a parameter, set by build()
    // their gradients are cleared at the start of each backward()
    std::vector<std::reference_wrapper<mlfo::tensor::Tensor<T>>> non_parameters_;
    // flag to indicate whether the model has been built
    // if not, build() must be called before forward() or backward()
    bool built_;
public:
    Model(
        std::vector<std::size_t> input_shape,
        std::vector<std::unique_ptr<mlfo::layer::Layer<T>>>&& layers,
        std::unique_ptr<mlfo::layer::loss::LossLayer<T>>&& loss = nullptr
    ) :
    input_shape_(std::move(input_shape)),
    layers_(std::move(layers)),
    loss_(std::move(loss)),
    built_(false)
    {
        if (input_shape_.empty()) {
            throw std::invalid_argument("[mlfo::model::Model] Input shape cannot be empty");
        }
    }

    void build() {
        // creates the input node (shaped by the model's input shape), chains each layer's output
        // into the next layer's input, then sorts the graph. each layer reads its input shape from
        // the node it is given
        if (built_) {
            // no op since model is already build
            return;
        }
        if (layers_.empty()) {
            throw std::logic_error("[mlfo::model::Model::build] Model has no layers");
        }
        inputs_.clear();
        outputs_.clear();
        output_tensors_.clear();
        inputs_.push_back(
            &graph_.add_node(mlfo::tensor::Tensor<T>(1, input_shape_))
        );
        mlfo::graph::Node<mlfo::tensor::Tensor<T>>* current = inputs_.front();
        for (auto& layer : layers_) {
            current = &layer->build(graph_, *current);
        }
        outputs_.push_back(current);
        for (const auto* node : outputs_) {
            output_tensors_.push_back(node->data());
        }
        // add loss layer if present
        if (loss_) {
            //                +------- model output
            //                |
            // model output --+--+
            //                   |---- loss output
            //        label -----+
            // loss's input is model's output + label, the label node is owned by the loss layer
            current = &loss_->build(graph_, *current);
            outputs_.push_back(current);
            output_tensors_.push_back(current->data());
        }
        // topologically sort the graph
        graph_.finalize();
        // collect parameters from all layers
        parameters_.clear();
        for (const auto& layer : layers_) {
            for (const auto& param_ref : layer->parameters()) {
                parameters_.push_back(param_ref);
            }
        }
        // collect every other tensor, so backward() doesn't search the parameters on each call
        non_parameters_.clear();
        for (const auto& node : graph_.nodes()) {
            auto& tensor = node->data();
            const bool is_parameter = std::ranges::any_of(
                parameters_,
                [&tensor](const auto& param_ref) {
                    return &param_ref.get() == &tensor;
                }
            );
            if (!is_parameter) {
                non_parameters_.push_back(tensor);
            }
        }
        // mark the model as built
        built_ = true;
    }

    void set_inputs(std::initializer_list<std::span<const T>> values) {
        // sets the values of the input tensors, one flattened span per input tensor, e.g.
        // set_inputs({x}). the values are copied into the tensors' existing buffers, so this
        // never allocates. the next forward() recomputes the whole graph
        if (!built_) {
            throw std::logic_error("[mlfo::model::Model::set_inputs] Model has not been built");
        }
        if (values.size() != inputs_.size()) {
            throw std::invalid_argument("[mlfo::model::Model::set_inputs] Wrong number of inputs");
        }
        auto value = values.begin();
        for (std::size_t i = 0; i < inputs_.size(); i++, value++) {
            inputs_[i]->data().set_values(*value);
        }
        // mark every node as needing recomputation
        graph_.invalidate();
    }

    void set_labels(std::initializer_list<std::span<const T>> values) {
        // sets the values of the label tensors, one flattened span per label tensor, e.g.
        // set_labels({y}). the values are copied into the tensors' existing buffers, so this
        // never allocates. the next forward() recomputes the whole graph
        if (!built_) {
            throw std::logic_error("[mlfo::model::Model::set_labels] Model has not been built");
        }
        if (!loss_) {
            throw std::logic_error("[mlfo::model::Model::set_labels] Model has no loss layer");
        }
        if (values.size() != 1) {
            throw std::invalid_argument("[mlfo::model::Model::set_labels] Wrong number of labels");
        }
        loss_->label_node().data().set_values(*values.begin());
        // mark every node as needing recomputation
        graph_.invalidate();
    }

    void invalidate() {
        // marks every node as needing recomputation, e.g. after an optimizer updated the parameters
        if (!built_) {
            throw std::logic_error("[mlfo::model::Model::invalidate] Model has not been built");
        }
        graph_.invalidate();
    }

    const std::vector<std::reference_wrapper<const mlfo::tensor::Tensor<T>>>& outputs() const {
        // the output tensors, valid until the next build()
        if (!built_) {
            throw std::logic_error("[mlfo::model::Model::outputs] Model has not been built");
        }
        return output_tensors_;
    }

    void forward() {
        if (!built_) {
            throw std::logic_error("[mlfo::model::Model::forward] Model has not been built");
        }
        graph_.forward();
    }

    void backward() {
        if (!built_) {
            throw std::logic_error("[mlfo::model::Model::backward] Model has not been built");
        }
        // set loss gradients to 1
        if (!loss_) {
            throw std::logic_error("[mlfo::model::Model::backward] Model has no loss layer");
        }
        // ops accumulate into their inputs' gradients, so clear the gradients left over from
        // the previous backward(). parameter gradients are kept, they are cleared by the
        // optimizer's zero_grad() so that they can be accumulated over several backward() calls
        for (const auto& tensor_ref : non_parameters_) {
            tensor_ref.get().zero_grad();
        }
        // the loss output is the last output node, see build()
        auto& loss_tensor = outputs_.back()->data();
        loss_tensor.fill_gradients(T{1});
        graph_.backward();
    }

    const std::vector<std::reference_wrapper<mlfo::tensor::Tensor<T>>>& parameters() const {
        if (!built_) {
            throw std::logic_error("[mlfo::model::Model::inputs] Model has not been built");
        }
        return this->parameters_;
    }

    std::string to_string() const {
        // todo: print out computational graph and layers
        // don't print out values, only shapes of tensors and layer types
        // use reflection for types?
        return "";
    }
};

} // namespace mlfo::model

#endif // MODEL_HPP
