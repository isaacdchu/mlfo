#ifndef LAYER_HPP
#define LAYER_HPP

#include "tensor.hpp"

#include <memory>

class Layer {
public:
    virtual ~Layer() = default;
    virtual std::unique_ptr<Tensor> forward(std::unique_ptr<Tensor> input) = 0;
    virtual std::unique_ptr<Tensor> backward(std::unique_ptr<Tensor> grad_output) = 0;
};

#endif // LAYER_HPP