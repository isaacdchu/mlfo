#ifndef MSE_LOSS_HPP
#define MSE_LOSS_HPP

#include "tensor.hpp"
#include "pool.hpp"
#include "operations.hpp"
#include "loss.hpp"

#include <memory>
#include <vector>


class MSELoss : public Loss {
public:
    MSELoss(
        const std::vector<std::vector<std::size_t>>& input_unbatched_shapes,
        const std::vector<Tensor*>& inputs,
        const std::vector<Tensor*>& targets,
        Pool* pool
    ) : Loss(input_unbatched_shapes, inputs, targets, pool) {
        /*
        std::vector<Tensor*> losses;
        for (std::size_t i = 0; i < inputs.size(); i++) {
            losses.push_back(Operations::mean_reduce(
                Operations::pow(Operations::sub(targets[i], inputs[i]), 2.0f))
            );
        }
        output_ = Operations::sum(losses);
        */
    }

    static std::unique_ptr<Loss> factory(
        const std::vector<std::vector<std::size_t>>& input_unbatched_shapes,
        const std::vector<Tensor*>& inputs,
        const std::vector<Tensor*>& targets,
        Pool* pool
    ) {
        return std::make_unique<MSELoss>(input_unbatched_shapes, inputs, targets, pool);
    }
};

#endif // MSE_LOSS_HPP