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
        auto output = std::make_unique<Tensor>(a->shape(), batch_size, 0.0f);
        output->set_parents({a, b});
        output->forward_ = [](Tensor *output) {
            // update values of output
            Tensor* a = output->parents_[0];
            Tensor* b = output->parents_[1];
            if (a->batched() == b->batched()) {
                for (std::size_t i = 0; i < output->size(); i++) {
                    output->values_[i] = a->values_[i] + b->values_[i];
                }
            } else if (a->batched()) {
                for (std::size_t i = 0; i < output->size(); i++) {
                    output->values_[i] = a->values_[i] + b->values_[i % b->size()];
                }
            } else {
                for (std::size_t i = 0; i < output->size(); i++) {
                    output->values_[i] = a->values_[i % a->size()] + b->values_[i];
                }
            }
        };
        output->backward_ = [](Tensor* output) {
            // update gradients of parents of output
            Tensor* a = output->parents_[0];
            Tensor* b = output->parents_[1];
            const std::size_t batch_size = std::max(a->batch_size(), b->batch_size());
            if (!a->batched() && !b->batched()) {
                for (std::size_t i = 0; i < output->unbatched_size(); i++) {
                    a->gradients_[i] += output->gradients_[i];
                    b->gradients_[i] += output->gradients_[i];
                }
            } else if (a->batched() && b->batched()) {
                for (std::size_t batch = 0; batch < batch_size; batch++) {
                    std::size_t base = batch * output->unbatched_size();
                    for (std::size_t i = 0; i < output->unbatched_size(); i++) {
                        a->gradients_[base + i] += output->gradients_[base + i];
                        b->gradients_[base + i] += output->gradients_[base + i];
                    }
                }
            } else if (a->batched()) {
                for (std::size_t batch = 0; batch < batch_size; batch++) {
                    std::size_t base = batch * output->unbatched_size();
                    for (std::size_t i = 0; i < output->unbatched_size(); i++) {
                        a->gradients_[base + i] += output->gradients_[base + i];   
                    }
                }
                for (std::size_t i = 0; i < output->unbatched_size(); i++) {
                    b->gradients_[i] += output->gradients_[i];
                }
            } else if (b->batched()) {
                for (std::size_t batch = 0; batch < batch_size; batch++) {
                    std::size_t base = batch * output->unbatched_size();
                    for (std::size_t i = 0; i < output->unbatched_size(); i++) {
                        b->gradients_[base + i] += output->gradients_[base + i];
                    }
                }
                for (std::size_t i = 0; i < output->unbatched_size(); i++) {
                    a->gradients_[i] += output->gradients_[i];
                }
            }
        };
        return output;
    }
};

#endif // OPERATIONS_HPP