#ifndef MODEL_HPP
#define MODEL_HPP

#include "tensor.hpp"
#include "layer.hpp"
#include "loss.hpp"

#include <vector>
#include <memory>

class Model {
private:
    std::vector<std::unique_ptr<Layer>> layers_;
    std::unique_ptr<Loss> loss_;
    std::vector<std::unique_ptr<Tensor>> inputs_;
public:
    Model() = delete;

    Model(std::vector<std::unique_ptr<Layer>>&& layers, std::unique_ptr<Loss> loss)
    : layers_(std::move(layers)), loss_(std::move(loss)) {
        if (layers_.empty()) {
            throw std::runtime_error("[Model::Model] Model must have at least one layer");
        }
        for (std::size_t i = 0; i < layers_.size() - 1; i++) {
            if (layers_[i]->output_unbatched_shapes() != layers_[i + 1]->input_unbatched_shapes()) {
                throw std::runtime_error(
                    "[Model::Model] Output shape of layer " +
                    std::to_string(i) +
                    " does not match input shape of layer " +
                    std::to_string(i + 1)
                );
            }
        }
        for (const auto& shape : layers_.front()->input_unbatched_shapes()) {
            inputs_.emplace_back(std::make_unique<Tensor>(shape, 1));
        }
    }

    void set_inputs(std::vector<std::vector<float>>&& input_values, std::size_t batch_size) {
        for (std::size_t i = 0; i < inputs_.size(); i++) {
            inputs_[i]->set_values(std::move(input_values[i]), batch_size);
        }
    }

    const std::vector<std::vector<std::size_t>>& input_unbatched_shapes() const {
        return layers_.front()->input_unbatched_shapes();
    }

    const std::vector<std::vector<std::size_t>>& output_unbatched_shapes() const {
        return layers_.back()->output_unbatched_shapes();
    }

    void forward() {
        for (const auto& layer : layers_) {
            layer->forward();
        }
        loss_->forward();
    }

    void backward() {
        loss_->backward();
        for (auto it = layers_.rbegin(); it != layers_.rend(); it++) {
            (*it)->backward();
        }
    }
};

#endif // MODEL_HPP