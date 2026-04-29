#ifndef LAYER_HPP
#define LAYER_HPP

#include "tensor.hpp"

#include <memory>
#include <vector>
#include <ranges>

class Layer {
protected:
    const std::vector<std::vector<std::size_t>> input_unbatched_shapes_;
    const std::vector<std::vector<std::size_t>> output_unbatched_shapes_;
    std::vector<Tensor*> inputs_;
    std::vector<std::unique_ptr<Tensor>> outputs_;
public:
    Layer() = delete;

    Layer(
        const std::vector<std::vector<std::size_t>>& input_unbatched_shapes,
        const std::vector<std::vector<std::size_t>>& output_unbatched_shapes,
        const std::vector<Tensor*>& inputs
    ) :
    input_unbatched_shapes_(input_unbatched_shapes),
    output_unbatched_shapes_(output_unbatched_shapes),
    inputs_(inputs) {
        for (const auto& shape : output_unbatched_shapes_) {
            outputs_.emplace_back(std::make_unique<Tensor>(shape, 1));
        }
    }

    virtual const std::vector<std::unique_ptr<Tensor>>& outputs() const final {
        return outputs_;
    }

    virtual const std::vector<std::vector<std::size_t>>& input_unbatched_shapes() const final {
        return input_unbatched_shapes_;
    }

    virtual const std::vector<std::vector<std::size_t>>& output_unbatched_shapes() const final {
        return output_unbatched_shapes_;
    }

    virtual void forward() = 0;
    virtual void backward() = 0;
};

#endif // LAYER_HPP