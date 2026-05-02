#ifndef OPTIMIZER_HPP
#define OPTIMIZER_HPP

#include "tensor.hpp"

#include <memory>
#include <vector>
#include <stdexcept>

class Optimizer {
protected:
    std::vector<Tensor*> parameters_;

    Optimizer(const std::vector<Tensor*>& parameters) {
        if (parameters.empty()) {
            throw std::runtime_error("[Optimizer::Optimizer] Optimizer must have at least one parameter tensor");
        }
        parameters_ = parameters;
    }

public:
    virtual ~Optimizer() = default;
    virtual void step() = 0;
};

#endif // OPTIMIZER_HPP