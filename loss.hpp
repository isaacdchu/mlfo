#ifndef LOSS_HPP
#define LOSS_HPP

#include "tensor.hpp"

#include <memory>
#include <vector>

class Loss {
protected:
    const std::vector<std::vector<std::size_t>> input_unbatched_shapes_;
    std::vector<Tensor*> inputs_;
    std::vector<Tensor*> targets_;
    Tensor* output_;
    Pool* pool_;

    Loss() = delete;

    Loss(
        const std::vector<std::vector<std::size_t>>& input_unbatched_shapes,
        const std::vector<Tensor*>& inputs,
        const std::vector<Tensor*>& targets,
        Pool* pool
    ) :
    input_unbatched_shapes_(input_unbatched_shapes),
    inputs_(inputs),
    targets_(targets) {
        if (inputs.size() != targets.size()) {
            throw std::runtime_error("[Loss::Loss] Number of input tensors must match number of target tensors");
        }
        for (std::size_t i = 0; i < inputs.size(); i++) {
            if (inputs[i]->unbatched_shape() != targets[i]->unbatched_shape()) {
                throw std::runtime_error("[Loss::Loss] Unbatched shape of input tensor must match unbatched shape of target tensor");
            }
        }
        if (!pool) {
            throw std::runtime_error("[Loss::Loss] Pool cannot be null");
        }
        pool_ = pool;
    }

public:
    virtual ~Loss() = default;

    virtual const std::vector<std::vector<std::size_t>>& input_unbatched_shapes() const final {
        return input_unbatched_shapes_;
    }

    virtual Tensor* output() const {
        return output_;
    }

    virtual void forward() {
        output_->forward();
    }

    virtual void backward() {
        float scale = 1.0f;
        if (output_->batched()) {
            const std::size_t bs = output_->batch_size();
            if (bs > 0) scale = 1.0f / static_cast<float>(bs);
        }
        output_->set_gradients(std::vector<float>(output_->size(), scale));
        output_->backward();
    }
};

#endif // LOSS_HPP