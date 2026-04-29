#ifndef LINEAR_LAYER_HPP
#define LINEAR_LAYER_HPP

#include "layer.hpp"
#include "operations.hpp"

#include <memory>
#include <vector>

class LinearLayer : public Layer {
private:
    std::vector<std::unique_ptr<Tensor>> weights_;
    std::vector<std::unique_ptr<Tensor>> biases_;
public:
    LinearLayer(
        const std::vector<std::vector<std::size_t>>& input_unbatched_shapes,
        const std::vector<std::vector<std::size_t>>& output_unbatched_shapes,
        const std::vector<Tensor*>& inputs
    ) : Layer(input_unbatched_shapes, output_unbatched_shapes, inputs) {
        // 
    }

    void forward() override {
        for (auto& output : outputs_) {
            output->forward();
        }
    }
};

#endif // LINEAR_LAYER_HPP