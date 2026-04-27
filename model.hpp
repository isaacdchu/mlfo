#ifndef MODEL_HPP
#define MODEL_HPP

#include "tensor.hpp"
#include "layer.hpp"

#include <vector>
#include <memory>

class Model {
private:
    std::vector<std::unique_ptr<Layer>> layers_;
    std::unique_ptr<Tensor> input_;
    std::unique_ptr<Tensor> output_;
public:
    Model() = delete;

    Model(std::vector<std::unique_ptr<Layer>>&& layers)
    : layers_(std::move(layers)) {
        std::vector<std::size_t> input_shape = {1};
        input_shape.append_range(layers_.front()->input_shape());
        std::vector<std::size_t> output_shape = {1};
        output_shape.append_range(layers_.back()->output_shape());
        input_ = std::make_unique<Tensor>(input_shape, 1, 0.0f);
        output_ = std::make_unique<Tensor>(output_shape, 1, 0.0f);
    }

    void set_input(std::vector<float>&& input_values, std::size_t batch_size) {
        if (input_->size() * batch_size != input_values.size() * input_->batch_size()) {
            throw std::runtime_error("[Model::set_input] Input shape does not match model input shape");
        }
        input_->set_values(std::move(input_values), batch_size);
    }

    const std::vector<std::size_t>& input_shape() const {
        return input_->shape();
    }

    const std::vector<std::size_t>& output_shape() const {
        return output_->shape();
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