#ifndef OPERATIONS_HPP
#define OPERATIONS_HPP

#include "tensor.hpp"

#include <vector>
#include <memory>
#include <iostream>

class Operations {
private:
    static bool unbatched_shapes_equal(const std::vector<Tensor*>& tensors) {
        if (tensors.empty()) {
            return true;
        }
        std::vector<std::size_t> ref_shape = tensors.front()->unbatched_shape();
        for (const auto& tensor : tensors) {
            if (tensor->unbatched_shape() != ref_shape) {
                return false;
            }
        }
        return true;
    }
public:
    Operations() = delete;

    static std::unique_ptr<Tensor> add(Tensor* a, Tensor* b) {
        if (!unbatched_shapes_equal({a, b})) {
            throw std::runtime_error("[Operations::add] Unbatched shapes of input tensors do not match");
        }
        if (a->batched() && b->batched() && (a->batch_size() != b->batch_size())) {
            throw std::runtime_error("[Operations::add] Batch sizes of input tensors do not match");
        }
        const std::size_t batch_size = std::max(a->batch_size(), b->batch_size());
        auto output = std::make_unique<Tensor>(a->unbatched_shape(), batch_size, 0.0f);
        output->set_parents({a, b});
        output->forward_ = [](Tensor *output) -> void {
            // update values of output
            Tensor* a = output->parents_[0];
            Tensor* b = output->parents_[1];
            const std::size_t num_batches = std::ranges::max({
                output->batch_size(),
                a->batch_size(),
                b->batch_size(),
                static_cast<std::size_t>(1)
            });
            for (std::size_t batch = 0; batch < num_batches; batch++) {
                const std::size_t output_base = output->batched() ? batch * output->unbatched_size() : 0;
                const std::size_t a_base = a->batched() ? batch * a->unbatched_size() : 0;
                const std::size_t b_base = b->batched() ? batch * b->unbatched_size() : 0;
                for (std::size_t i = 0; i < output->unbatched_size(); i++) {
                    output->values_[output_base + i] = a->values_[a_base + i] + b->values_[b_base + i];
                }
            }
        };
        output->backward_ = [](Tensor* output) -> void {
            // update gradients of parents of output
            Tensor* a = output->parents_[0];
            Tensor* b = output->parents_[1];
            const std::size_t num_batches = std::ranges::max({
                output->batch_size(),
                a->batch_size(),
                b->batch_size(),
                static_cast<std::size_t>(1)
            });
            for (std::size_t batch = 0; batch < num_batches; batch++) {
                const std::size_t output_base = output->batched() ? batch * output->unbatched_size() : 0;
                for (std::size_t i = 0; i < output->unbatched_size(); i++) {
                    a->gradients_[i] += output->gradients_[output_base + i];
                    b->gradients_[i] += output->gradients_[output_base + i];
                }
            }
        };
        return output;
    }
};

#endif // OPERATIONS_HPP