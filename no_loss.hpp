#ifndef NO_LOSS_HPP
#define NO_LOSS_HPP

#include "tensor.hpp"
#include "pool.hpp"
#include "loss.hpp"

#include <memory>
#include <vector>
#include <stdexcept>

class NoLoss : public Loss {
public:
    NoLoss(
        const std::vector<std::vector<std::size_t>> &input_unbatched_shapes,
        const std::vector<Tensor *> &inputs,
        const std::vector<Tensor *> &targets,
        Pool *pool
    ) : Loss(input_unbatched_shapes, inputs, targets, pool) {
        // 
    }

    Tensor* output() const override {
        throw std::runtime_error("[NoLoss::output] NoLoss does not have an output tensor");
        return nullptr;
    }

    void forward() override {
        throw std::runtime_error("[NoLoss::forward] NoLoss does not have a forward pass");
        return;
    }

    void backward() override {
        throw std::runtime_error("[NoLoss::backward] NoLoss does not have a backward pass");
        return;
    }

    static std::unique_ptr<Loss> factory(
        const std::vector<std::vector<std::size_t>> &input_unbatched_shapes,
        const std::vector<Tensor *> &inputs,
        const std::vector<Tensor *> &targets,
        Pool *pool) {
        return std::make_unique<NoLoss>(input_unbatched_shapes, inputs, targets, pool);
    }
};


#endif // NO_LOSS_HPP