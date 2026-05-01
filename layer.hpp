#ifndef LAYER_HPP
#define LAYER_HPP

#include "tensor.hpp"
#include "pool.hpp"

#include <memory>
#include <vector>
#include <ranges>

class Layer {
protected:
    const std::vector<std::vector<std::size_t>> input_unbatched_shapes_;
    const std::vector<std::vector<std::size_t>> output_unbatched_shapes_;
    std::vector<Tensor*> inputs_;
    std::vector<Tensor*> outputs_;
    Pool* pool_;
    std::vector<Tensor*> parameters_;

    Layer(
        const std::vector<std::vector<std::size_t>>& input_unbatched_shapes,
        const std::vector<std::vector<std::size_t>>& output_unbatched_shapes,
        const std::vector<Tensor*>& inputs,
        Pool* pool
    ) :
    input_unbatched_shapes_(input_unbatched_shapes),
    output_unbatched_shapes_(output_unbatched_shapes),
    inputs_(inputs),
    pool_(pool) {
        if (inputs.size() != input_unbatched_shapes.size()) {
            throw std::runtime_error("[Layer::Layer] Number of input tensors must match number of input unbatched shapes");
        }
        for (std::size_t i = 0; i < inputs.size(); i++) {
            if (inputs[i]->unbatched_shape() != input_unbatched_shapes[i]) {
                throw std::runtime_error("[Layer::Layer] Unbatched shape of input tensor must match corresponding input unbatched shape");
            }
        }
        if (!pool) {
            throw std::runtime_error("[Layer::Layer] Pool cannot be null");
        }
    }

public:
    Layer() = delete;

    virtual const std::vector<Tensor*>& outputs() const final {
        return outputs_;
    }

    virtual const std::vector<std::vector<std::size_t>>& input_unbatched_shapes() const final {
        return input_unbatched_shapes_;
    }

    virtual const std::vector<std::vector<std::size_t>>& output_unbatched_shapes() const final {
        return output_unbatched_shapes_;
    }

    void forward() {
        for (auto& output : outputs_) {
            output->forward();
        }
    }

    void backward() {
        for (auto& output : outputs_) {
            output->backward();
        }
    }

    virtual const std::vector<Tensor*>& parameters() = 0;
};

#endif // LAYER_HPP