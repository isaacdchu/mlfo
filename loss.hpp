#ifndef LOSS_HPP
#define LOSS_HPP

#include "tensor.hpp"

#include <memory>

class Loss {
public:
    virtual ~Loss() = default;
    virtual std::unique_ptr<Tensor> forward(std::unique_ptr<Tensor> predictions, const Tensor& targets) = 0;
    virtual std::unique_ptr<Tensor> backward(std::unique_ptr<Tensor> predictions, const Tensor& targets) = 0;
};

#endif // LOSS_HPP