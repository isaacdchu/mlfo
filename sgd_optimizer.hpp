#ifndef SGD_OPTIMIZER_HPP
#define SGD_OPTIMIZER_HPP

#include "tensor.hpp"
#include "pool.hpp"
#include "operations.hpp"
#include "optimizer.hpp"

#include <memory>
#include <vector>

class SGDOptimizer : public Optimizer {
private:
    float learning_rate_;

public:
    SGDOptimizer(
        const std::vector<Tensor*>& parameters,
        float learning_rate
    ) : Optimizer(parameters) {
        if (learning_rate <= 0.0f) {
            throw std::runtime_error("[SGDOptimizer::SGDOptimizer] Learning rate must be positive");
        }
        learning_rate_ = learning_rate;
    }

    void step() override {
        for (Tensor* param : parameters_) {
            std::vector<float> new_values = param->values();
            for (std::size_t i = 0; i < new_values.size(); i++) {
                new_values[i] -= learning_rate_ * param->gradients()[i];
            }
            param->set_values(std::move(new_values), param->batch_size());
        }
    }
};

#endif // SGD_OPTIMIZER_HPP