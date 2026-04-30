#ifndef MODEL_HPP
#define MODEL_HPP

#include "tensor.hpp"
#include "pool.hpp"
#include "layer.hpp"
#include "loss.hpp"

#include <vector>
#include <memory>
#include <functional>
#include <stdexcept>

class Model {
private:
    std::vector<std::unique_ptr<Layer>> layers_;
    std::unique_ptr<Loss> loss_;
    std::vector<std::unique_ptr<Tensor>> inputs_;
    std::unique_ptr<Pool> pool_;
public:
    Model() = delete;

    Model (
        const std::vector<
            std::function<
            std::unique_ptr<Layer>(
                const std::vector<std::vector<std::size_t>>& input_unbatched_shapes,
                const std::vector<std::vector<std::size_t>>& output_unbatched_shapes,
                const std::vector<Tensor*>& inputs,
                Pool* pool
            )>
        >& layer_factories,
        const std::vector<std::vector<std::vector<std::size_t>>>& all_input_unbatched_shapes,
        const std::vector<std::vector<std::vector<std::size_t>>>& all_output_unbatched_shapes
    ) {
        if (layer_factories.empty()) {
            throw std::runtime_error("[Model::Model] Model must have at least one layer");
        }
        if (all_input_unbatched_shapes.size() != layer_factories.size() ||
            all_output_unbatched_shapes.size() != layer_factories.size()) {
            throw std::runtime_error("[Model::Model] Size of input and output shape lists must match number of layers");
        }
        pool_ = std::make_unique<Pool>();
        for (const auto& shape : all_input_unbatched_shapes.front()) {
            inputs_.emplace_back(std::make_unique<Tensor>(shape, 1));
        }
        std::vector<Tensor*> input_ptrs;
        for (auto& input : inputs_) {
            input_ptrs.push_back(input.get());
        }
        layers_.emplace_back(layer_factories[0](
            all_input_unbatched_shapes[0],
            all_output_unbatched_shapes[0],
            input_ptrs,
            pool_.get()
        ));
        for (std::size_t i = 1; i < layer_factories.size(); i++) {
            std::vector<Tensor*> prev_output_ptrs;
            for (const auto& output : layers_[i - 1]->outputs()) {
                prev_output_ptrs.push_back(output.get());
            }
            layers_.emplace_back(layer_factories[i](
                all_input_unbatched_shapes[i],
                all_output_unbatched_shapes[i],
                prev_output_ptrs,
                pool_.get()
            ));
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