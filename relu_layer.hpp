#ifndef RELU_LAYER_HPP
#define RELU_LAYER_HPP

#include "tensor.hpp"
#include "operations.hpp"
#include "layer.hpp"

#include <memory>
#include <vector>

class ReLULayer : public Layer {
public:
    ReLULayer() = delete;
    ReLULayer(
        const std::vector<std::vector<std::size_t>>& input_unbatched_shapes,
        const std::vector<std::vector<std::size_t>>& output_unbatched_shapes,
        const std::vector<Tensor*>& inputs,
        Pool* pool
    ) : Layer(input_unbatched_shapes, output_unbatched_shapes, inputs, pool) {
        // input and output shapes must match
        if (input_unbatched_shapes.size() != output_unbatched_shapes.size()) {
            throw std::runtime_error("[ReLULayer::ReLULayer] Number of input and output tensors must match");
        }
        for (std::size_t i = 0; i < input_unbatched_shapes.size(); i++) {
            if (input_unbatched_shapes[i] != output_unbatched_shapes[i]) {
                throw std::runtime_error("[ReLULayer::ReLULayer] Unbatched shapes of input and output tensors must match");
            }
        }
        for (std::size_t i = 0; i < input_unbatched_shapes.size(); i++) {
            for (std::size_t j = 0; j < input_unbatched_shapes[i].size(); j++) {
                if (input_unbatched_shapes[i][j] == 0 || output_unbatched_shapes[i][j] == 0) {
                    throw std::runtime_error("[ReLULayer::ReLULayer] Input and output shape dimensions cannot be zero");
                }
                if (input_unbatched_shapes[i][j] != output_unbatched_shapes[i][j]) {
                    throw std::runtime_error("[ReLULayer::ReLULayer] Input and output shapes must match");
                }
            }
        }

        parameters_ = {};
        outputs_.resize(inputs.size());
        for (std::size_t i = 0; i < outputs_.size(); i++) {
            outputs_[i] = Operations::relu(inputs_[i]);
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
        return std::make_unique<ReLULayer>(input_unbatched_shapes, output_unbatched_shapes, inputs, pool);
    }
};

#endif // RELU_LAYER_HPP