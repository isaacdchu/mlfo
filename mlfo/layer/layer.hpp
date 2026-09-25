#ifndef LAYER_HPP
#define LAYER_HPP

#include <vector>

#include "../graph/graph.hpp"

namespace mlfo::layer {

class Layer {
public:
    virtual ~Layer() = default;
    virtual void forward() = 0;
    virtual void backward() = 0;
};

} // namespace mlfo::layer

#endif // LAYER_HPP
