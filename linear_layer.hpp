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
        std::vector<std::size_t> num_shared_dims = std::vector<std::size_t>(input_unbatched_shapes.size(), 0);
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
            if (in_shape.front() != out_shape.front()) {
                throw std::runtime_error("[LinearLayer::LinearLayer] First dimension of input shape must match first dimension of output shape");
            }
            for (std::size_t j = 0; j < std::min(in_shape.size(), out_shape.size()); j++) {
                if (in_shape[j] != out_shape[j]) {
                    break;
                }
                num_shared_dims[i]++;
            }
            if (num_shared_dims[i] == 0) {
                throw std::runtime_error("[LinearLayer::LinearLayer] Input and output shapes must share at least one dimension");
            }
            // weight shape is [unshared_input_dims..., unshared_output_dims...]
            std::vector<std::size_t> weight_shape;
            weight_shape.insert(weight_shape.end(), in_shape.begin() + num_shared_dims[i], in_shape.end());
            weight_shape.insert(weight_shape.end(), out_shape.begin() + num_shared_dims[i], out_shape.end());
            weights_[i] = pool_->new_tensor(weight_shape, 0, 0.1f);
            // bias shape is the full output unbatched shape for this mapping
            std::vector<std::size_t> bias_shape(out_shape);
            biases_[i] = pool_->new_tensor(bias_shape, 0, 0.1f);
            // in * weight + bias = out
            const std::size_t contractions = in_shape.size() - num_shared_dims[i];
            outputs_[i] = Operations::add(
                Operations::matmul(inputs_[i], weights_[i], contractions),
                biases_[i]
            );
        }
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