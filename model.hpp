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
    std::vector<Tensor*> inputs_;
    std::unique_ptr<Pool> pool_;
    std::vector<Tensor*> targets_;
    std::vector<Tensor*> parameters_;
public:
    Model() = delete;

    Model (
        const std::vector<
            std::function<std::unique_ptr<Layer>(
                const std::vector<std::vector<std::size_t>>& input_unbatched_shapes,
                const std::vector<std::vector<std::size_t>>& output_unbatched_shapes,
                const std::vector<Tensor*>& inputs,
                Pool* pool
            )>
        >& layer_factories,
        const std::vector<std::vector<std::vector<std::size_t>>>& all_input_unbatched_shapes,
        const std::vector<std::vector<std::vector<std::size_t>>>& all_output_unbatched_shapes,
        std::function<std::unique_ptr<Loss>(
            const std::vector<std::vector<std::size_t>>& input_unbatched_shapes,
            const std::vector<Tensor*>& inputs,
            const std::vector<Tensor*>& targets,
            Pool* pool
        )> loss_factory
    ) {
        if (layer_factories.empty()) {
            throw std::runtime_error("[Model::Model] Model must have at least one layer");
        }
        if (all_input_unbatched_shapes.size() != layer_factories.size() ||
            all_output_unbatched_shapes.size() != layer_factories.size()) {
            throw std::runtime_error("[Model::Model] Size of input and output shape lists must match number of layers");
        }
        if (!loss_factory) {
            throw std::runtime_error("[Model::Model] Model must have a loss factory");
        }
        pool_ = std::make_unique<Pool>();
        for (const auto& shape : all_input_unbatched_shapes.front()) {
            inputs_.emplace_back(pool_->new_tensor(shape, 1));
        }
        std::vector<Tensor*> input_ptrs;
        for (auto& input : inputs_) {
            input_ptrs.push_back(input);
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
                prev_output_ptrs.push_back(output);
            }
            layers_.emplace_back(
                layer_factories[i](
                    all_input_unbatched_shapes[i],
                    all_output_unbatched_shapes[i],
                    prev_output_ptrs,
                    pool_.get()
                )
            );
        }
        for (const auto& shape : all_output_unbatched_shapes.back()) {
            targets_.emplace_back(pool_->new_tensor(shape, 1));
        }
        loss_ = loss_factory(
            all_output_unbatched_shapes.back(),
            layers_.back()->outputs(),
            targets_,
            pool_.get()
        );

        for (const auto& layer : layers_) {
            parameters_.append_range(layer->parameters());
        }
    }

    void set_inputs(std::vector<std::vector<float>>&& input_values, std::size_t batch_size) {
        batch_size = std::max(batch_size, static_cast<std::size_t>(1));
        if (input_values.size() != inputs_.size()) {
            throw std::runtime_error("[Model::set_inputs] Number of input value sets must match number of model inputs");
        }
        for (std::size_t i = 0; i < inputs_.size(); i++) {
            inputs_[i]->set_values(std::move(input_values[i]), batch_size);
        }
    }

    void set_targets(std::vector<std::vector<float>>&& target_values, std::size_t batch_size) {
        batch_size = std::max(batch_size, static_cast<std::size_t>(1));
        if (target_values.size() != targets_.size()) {
            throw std::runtime_error("[Model::set_targets] Number of target value sets must match number of model targets");
        }
        for (std::size_t i = 0; i < targets_.size(); i++) {
            targets_[i]->set_values(std::move(target_values[i]), batch_size);
        }
    }

    const std::vector<std::vector<std::size_t>>& input_unbatched_shapes() const {
        return layers_.front()->input_unbatched_shapes();
    }

    const std::vector<std::vector<std::size_t>>& output_unbatched_shapes() const {
        return layers_.back()->output_unbatched_shapes();
    }

    const std::vector<Tensor*>& outputs() const {
        return layers_.back()->outputs();
    }

    Tensor* loss_output() const {
        return loss_->output();
    }

    void forward() {
        for (const auto& layer : layers_) {
            layer->forward();
        }
    }

    void loss() {
        loss_->forward();
    }

    void zero_grad() {
        for (Tensor* tensor : pool_->tensors()) {
            tensor->zero_grad();
        }
    }

    void backward() {
        loss_->backward();
        for (auto it = layers_.rbegin(), rend = layers_.rend(); it != rend; it++) {
            (*it)->backward();
        }
    }

    const std::vector<Tensor*>& parameters() const {
        return parameters_;
    }
};

#endif // MODEL_HPP