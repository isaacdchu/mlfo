#ifndef DENSE_LAYER_HPP
#define DENSE_LAYER_HPP

#include <cstddef>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

#include "../graph/graph.hpp"
#include "../initializer/constant.hpp"
#include "../initializer/initializer.hpp"
#include "../initializer/values.hpp"
#include "../misc/concepts.hpp"
#include "../misc/serialization.hpp"
#include "../tensor/ops.hpp"
#include "../tensor/tensor.hpp"
#include "layer.hpp"

namespace mlfo::layer {

template <mlfo::misc::Number T>
class DenseLayer : public mlfo::layer::Layer<T> {
private:
    // set by build() from the input node
    std::size_t input_size_;
    std::size_t output_size_;
    std::shared_ptr<const mlfo::initializer::Initializer<T>> weight_initializer_;
    std::shared_ptr<const mlfo::initializer::Initializer<T>> bias_initializer_;
public:
    // the input size is taken from the input node when built
    DenseLayer(
        std::size_t output_size,
        std::shared_ptr<const mlfo::initializer::Initializer<T>> weight_initializer =
            std::make_shared<mlfo::initializer::Zeros<T>>(),
        std::shared_ptr<const mlfo::initializer::Initializer<T>> bias_initializer =
            std::make_shared<mlfo::initializer::Zeros<T>>()
    ) :
    input_size_(0),
    output_size_(output_size),
    weight_initializer_(std::move(weight_initializer)),
    bias_initializer_(std::move(bias_initializer))
    {
        if (output_size_ == 0) {
            throw std::invalid_argument(
                "[mlfo::layer::DenseLayer] Output size must be greater than zero"
            );
        }
        if (!weight_initializer_ || !bias_initializer_) {
            throw std::invalid_argument(
                "[mlfo::layer::DenseLayer] Initializers cannot be null"
            );
        }
        // activations are row vectors (1, n) since Mul requires 2D tensors
        this->output_shape_ = {1, output_size_};
    }

    mlfo::graph::Node<mlfo::tensor::Tensor<T>>& build(
        mlfo::graph::Graph<mlfo::tensor::Tensor<T>>& graph,
        mlfo::graph::Node<mlfo::tensor::Tensor<T>>& input_node
    ) override {
        const auto& input_shape = input_node.data().shape();
        // activations are row vectors (1, n) since Mul requires 2D tensors
        if (input_shape.size() != 2 || input_shape[0] != 1 || input_shape[1] == 0) {
            throw std::invalid_argument(
                "[mlfo::layer::DenseLayer::build] Input must be a (1, n) row vector, use a FlattenLayer first"
            );
        }
        input_size_ = input_shape[1];
        this->input_shape_ = input_shape;
        auto& weight_node = graph.add_node(
            mlfo::tensor::Tensor<T>(1, std::vector<std::size_t>{input_size_, output_size_})
        );
        // activations are row vectors (1, n) since Mul requires 2D tensors
        auto& bias_node = graph.add_node(
            mlfo::tensor::Tensor<T>(1, std::vector<std::size_t>{1, output_size_})
        );
        // each output unit sums input_size_ inputs, and each input feeds output_size_ units
        weight_initializer_->initialize(weight_node.data(), input_size_, output_size_);
        bias_initializer_->initialize(bias_node.data(), input_size_, output_size_);
        // activations have one batch per sample, the parameters are broadcast over them
        const std::size_t batch_size = input_node.data().batch_size();
        auto& prebias_node = graph.add_node(
            mlfo::tensor::Tensor<T>(batch_size, std::vector<std::size_t>{1, output_size_})
        );
        auto& output_node = graph.add_node(
            mlfo::tensor::Tensor<T>(batch_size, std::vector<std::size_t>{1, output_size_})
        );
        prebias_node.add_predecessors(
            {input_node, weight_node},
            [
                &prebias_node_tensor = prebias_node.data(),
                &input_node_tensor = input_node.data(),
                &weight_node_tensor = weight_node.data()
            ](mlfo::tensor::ops::DIRECTION direction) -> void {
                // input * weight
                mlfo::tensor::ops::Mul<T>::call(
                    direction,
                    prebias_node_tensor,
                    input_node_tensor,
                    weight_node_tensor
                );
            }
        );
        output_node.add_predecessors(
            {prebias_node, bias_node},
            [
                &output_node_tensor = output_node.data(),
                &prebias_node_tensor = prebias_node.data(),
                &bias_node_tensor = bias_node.data()
            ](mlfo::tensor::ops::DIRECTION direction) -> void {
                // prebias + bias
                mlfo::tensor::ops::Add<T>::call(
                    direction,
                    output_node_tensor,
                    prebias_node_tensor,
                    bias_node_tensor
                );
            }
        );
        this->parameters_.clear();
        this->parameters_.push_back(weight_node.data());
        this->parameters_.push_back(bias_node.data());
        return output_node;
    }

    const std::vector<std::reference_wrapper<mlfo::tensor::Tensor<T>>>& parameters() const override {
        // return the weight and bias tensors
        if (this->parameters_.empty()) {
            throw std::logic_error(
                "[mlfo::layer::DenseLayer::parameters] Layer has not been built"
            );
        }
        return this->parameters_;
    }

    // DenseLayer, then input_size,output_size, then the weights and biases
    void serialize(std::ostream& os) const override {
        const auto& params = parameters();
        os << "DenseLayer\n";
        mlfo::misc::write_line(os, std::vector<std::size_t>{input_size_, output_size_});
        mlfo::misc::write_line(os, params[0].get().values());
        mlfo::misc::write_line(os, params[1].get().values());
    }

    // reads everything serialize() writes after the type line
    // the layer is initialized with the read parameters when built
    static std::unique_ptr<DenseLayer> deserialize(std::istream& is) {
        const auto sizes = mlfo::misc::read_values<std::size_t>(is, 2);
        auto weights = mlfo::misc::read_values<T>(is, sizes[0] * sizes[1]);
        auto biases = mlfo::misc::read_values<T>(is, sizes[1]);
        return std::make_unique<DenseLayer>(
            sizes[1],
            std::make_shared<mlfo::initializer::Values<T>>(std::move(weights)),
            std::make_shared<mlfo::initializer::Values<T>>(std::move(biases))
        );
    }
};

} // namespace mlfo::layer

#endif // DENSE_LAYER_HPP
