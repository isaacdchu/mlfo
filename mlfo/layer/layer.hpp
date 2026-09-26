#ifndef LAYER_HPP
#define LAYER_HPP

#include <vector>

#include "../tensor/tensor.hpp"
#include "../graph/graph.hpp"
#include "../misc/concepts.hpp"

namespace mlfo::layer {

template <mlfo::misc::Number T>
class Layer {
protected:
    mlfo::graph::Graph<mlfo::tensor::Tensor<T>> graph_;
public:
    Layer() = default;
    virtual void forward() final {
        for (auto it = graph_.tail_begin(); it != graph_.tail_end(); ++it) {
            auto& node = *it;
            node->forward();
        }
    }

    virtual void backward() final {
        for (auto it = graph_.head_begin(); it != graph_.head_end(); ++it) {
            auto& node = *it;
            node->backward();
        }
    }
};

} // namespace mlfo::layer

#endif // LAYER_HPP
