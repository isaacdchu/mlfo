#ifndef LINEAR_LAYER_HPP
#define LINEAR_LAYER_HPP

#include "layer.hpp"
#include "operations.hpp"

#include <memory>
#include <vector>

class LinearLayer : public Layer {
private:
    std::vector<Tensor*> weights_;
    std::vector<Tensor*> biases_;

public:
    LinearLayer() = delete;

    LinearLayer(
        const std::vector<std::vector<std::size_t>>& input_unbatched_shapes,
        const std::vector<std::vector<std::size_t>>& output_unbatched_shapes,
        const std::vector<Tensor*>& inputs,
        Pool* pool
    ) : Layer(input_unbatched_shapes, output_unbatched_shapes, inputs, pool) {
        // weight shape will be [input x output]
        // bias shape will be [output]
        const std::size_t n = input_unbatched_shapes.size();
        weights_ = std::vector<Tensor*>(n, nullptr);
        biases_ = std::vector<Tensor*>(n, nullptr);
        outputs_.resize(n);
        for (std::size_t i = 0; i < input_unbatched_shapes.size(); i++) {
            const auto& in_shape = input_unbatched_shapes[i];
            const auto& out_shape = output_unbatched_shapes[i];
            if (in_shape.empty() || out_shape.empty()) {
                throw std::runtime_error("[LinearLayer::LinearLayer] Input and output shapes cannot be empty");
            }
        }
        for (std::size_t i = 0; i < input_unbatched_shapes.size(); i++) {
            const auto &in_shape = input_unbatched_shapes[i];
            const auto &out_shape = output_unbatched_shapes[i];
            // weight shape will be [reversed input_unbatched_dims x output_unbatched_dims]
            std::vector<std::size_t> weight_shape(in_shape.rbegin(), in_shape.rend());
            weight_shape.append_range(out_shape);
            weights_[i] = pool_->new_tensor(weight_shape, 0, 0.1f);
            // bias shape will be [output_unbatched_dims]
            biases_[i] = pool_->new_tensor(out_shape, 0, 0.1f);
            // in * weight + bias = out
            // contractions is number of input unbatched dimensions
            const std::size_t contractions = in_shape.size();
            outputs_[i] = Operations::add(
                Operations::matmul(inputs_[i], weights_[i], contractions),
                biases_[i]
            );
        }
        // add all weights and biases as parameters of this layer
        for (std::size_t i = 0; i < weights_.size(); i++) {
            parameters_.push_back(weights_[i]);
            parameters_.push_back(biases_[i]);
        }
    }

    const std::vector<Tensor*>& parameters() override {
        return parameters_;
    }

    static std::unique_ptr<Layer> factory(
        const std::vector<std::vector<std::size_t>>& input_unbatched_shapes,
        const std::vector<std::vector<std::size_t>>& output_unbatched_shapes,
        const std::vector<Tensor*>& inputs,
        Pool* pool
    ) {
        return std::make_unique<LinearLayer>(input_unbatched_shapes, output_unbatched_shapes, inputs, pool);
    }
};

#endif // LINEAR_LAYER_HPP