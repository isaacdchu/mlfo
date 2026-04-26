#ifndef MODEL_HPP
#define MODEL_HPP

#include "tensor.hpp"
#include "layer.hpp"

#include <vector>
#include <memory>

class Model {
private:
    std::vector<std::unique_ptr<Layer>> layers_;
public:
    Model() = delete;

    Model(std::vector<std::unique_ptr<Layer>>&& layers) {
        layers_ = std::move(layers);
    }

    std::unique_ptr<Tensor> forward(std::unique_ptr<Tensor> input) {
        for (const auto& layer : layers_) {
            input = layer->forward(std::move(input));
        }
        return input;
    }

    std::unique_ptr<Tensor> backward(std::unique_ptr<Tensor> grad_output) {
        for (auto it = layers_.rbegin(); it != layers_.rend(); ++it) {
            grad_output = (*it)->backward(std::move(grad_output));
        }
        return grad_output;
    }
};

#endif // MODEL_HPP