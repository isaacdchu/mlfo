#ifndef OPERATIONS_HPP
#define OPERATIONS_HPP

#include "tensor.hpp"
#include "pool.hpp"

#include <vector>
#include <memory>
#include <stdexcept>
#include <ranges>
#include <algorithm>
#include <iostream>
#include <print>
#include <cmath>

static void print_vector(const std::vector<std::size_t>& vec) {
        std::print("[");
        for (size_t i = 0; i < vec.size(); i++) {
            std::print("{}", vec[i]);
            if (i < vec.size() - 1) {
                std::print(", ");
            }
        }
        std::println("]");
        std::flush(std::cout);
    }

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

class Operations {
private:
    static void matmul_helper() {
        // TODO: implement matmul helper function that can be used for both forward and backward pass of matmul
    }

    static bool same_pool(Tensor* a, Tensor* b) {
        if (!a->pool_ || !b->pool_) {
            throw std::runtime_error("[Operations::same_pool] Both tensors must have an associated pool");
        }
        return a->pool_->id() == b->pool_->id();
    }

public:
    Operations() = delete;

    static constexpr float epsilon = 1e-6f;

    static Tensor* add(Tensor* a, Tensor* b) {
        if (!unbatched_shapes_equal({a, b})) {
            throw std::runtime_error("[Operations::add] Unbatched shapes of input tensors do not match");
        }
        if (a->batched() && b->batched() && (a->batch_size() != b->batch_size())) {
            throw std::runtime_error("[Operations::add] Batch sizes of input tensors do not match");
        }
        if (!same_pool(a, b)) {
            throw std::runtime_error("[Operations::add] Input tensors must be from the same pool");
        }
        const std::size_t batch_size = std::max(a->batch_size(), b->batch_size());
        auto output = a->pool_->new_tensor(a->unbatched_shape(), batch_size, 0.0f);
        output->set_parents({a, b});
        output->forward_ = [](Tensor *output) -> void {
            // update values of output
            Tensor* a = output->parents_[0];
            Tensor* b = output->parents_[1];
            const std::size_t num_batches = std::max(output->batch_size(), static_cast<std::size_t>(1));
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
            const std::size_t num_batches = std::max(output->batch_size(), static_cast<std::size_t>(1));
            for (std::size_t batch = 0; batch < num_batches; batch++) {
                const std::size_t output_base = output->batched() ? batch * output->unbatched_size() : 0;
                const std::size_t a_base = a->batched() ? batch * a->unbatched_size() : 0;
                const std::size_t b_base = b->batched() ? batch * b->unbatched_size() : 0;
                for (std::size_t i = 0; i < output->unbatched_size(); i++) {
                    a->gradients_[a_base + i] += output->gradients_[output_base + i];
                    b->gradients_[b_base + i] += output->gradients_[output_base + i];
                }
            }
        };
        return output;
    }

    static Tensor* sub(Tensor* a, Tensor* b) {
        if (!unbatched_shapes_equal({a, b})) {
            throw std::runtime_error("[Operations::sub] Unbatched shapes of input tensors do not match");
        }
        if (a->batched() && b->batched() && (a->batch_size() != b->batch_size())) {
            throw std::runtime_error("[Operations::sub] Batch sizes of input tensors do not match");
        }
        if (!same_pool(a, b)) {
            throw std::runtime_error("[Operations::sub] Input tensors must be from the same pool");
        }
        const std::size_t batch_size = std::max(a->batch_size(), b->batch_size());
        auto output = a->pool_->new_tensor(a->unbatched_shape(), batch_size, 0.0f);
        output->set_parents({a, b});
        output->forward_ = [](Tensor *output) -> void {
            // update values of output
            Tensor* a = output->parents_[0];
            Tensor* b = output->parents_[1];
            const std::size_t num_batches = std::max(output->batch_size(), static_cast<std::size_t>(1));
            for (std::size_t batch = 0; batch < num_batches; batch++) {
                const std::size_t output_base = output->batched() ? batch * output->unbatched_size() : 0;
                const std::size_t a_base = a->batched() ? batch * a->unbatched_size() : 0;
                const std::size_t b_base = b->batched() ? batch * b->unbatched_size() : 0;
                for (std::size_t i = 0; i < output->unbatched_size(); i++) {
                    output->values_[output_base + i] = a->values_[a_base + i] - b->values_[b_base + i];
                }
            }
        };
        output->backward_ = [](Tensor* output) -> void {
            // update gradients of parents of output
            Tensor* a = output->parents_[0];
            Tensor* b = output->parents_[1];
            const std::size_t num_batches = std::max(output->batch_size(), static_cast<std::size_t>(1));
            for (std::size_t batch = 0; batch < num_batches; batch++) {
                const std::size_t output_base = output->batched() ? batch * output->unbatched_size() : 0;
                const std::size_t a_base = a->batched() ? batch * a->unbatched_size() : 0;
                const std::size_t b_base = b->batched() ? batch * b->unbatched_size() : 0;
                for (std::size_t i = 0; i < output->unbatched_size(); i++) {
                    a->gradients_[a_base + i] += output->gradients_[output_base + i];
                    b->gradients_[b_base + i] -= output->gradients_[output_base + i];
                }
            }
        };
        return output;
    }

    static Tensor* pow(Tensor* a, float b) {
        if (-Operations::epsilon <= b && b <= Operations::epsilon) {
            throw std::runtime_error("[Operations::pow] Exponent cannot be zero");
        }
        const std::size_t batch_size = a->batch_size();
        auto output = a->pool_->new_tensor(a->unbatched_shape(), batch_size, 0.0f);
        output->set_parents({a});
        output->forward_ = [b](Tensor *output) -> void {
            // update values of output
            Tensor* a = output->parents_[0];
            const std::size_t num_batches = std::max(output->batch_size(), static_cast<std::size_t>(1));
            for (std::size_t batch = 0; batch < num_batches; batch++) {
                const std::size_t output_base = output->batched() ? batch * output->unbatched_size() : 0;
                const std::size_t a_base = a->batched() ? batch * a->unbatched_size() : 0;
                for (std::size_t i = 0; i < output->unbatched_size(); i++) {
                    output->values_[output_base + i] = std::pow(a->values_[a_base + i], b);
                }
            }
        };
        output->backward_ = [b](Tensor* output) -> void {
            // update gradients of parents of output
            Tensor* a = output->parents_[0];
            const std::size_t num_batches = std::max(output->batch_size(), static_cast<std::size_t>(1));
            for (std::size_t batch = 0; batch < num_batches; batch++) {
                const std::size_t output_base = output->batched() ? batch * output->unbatched_size() : 0;
                const std::size_t a_base = a->batched() ? batch * a->unbatched_size() : 0;
                for (std::size_t i = 0; i < output->unbatched_size(); i++) {
                    a->gradients_[a_base + i] += (
                        output->gradients_[output_base + i] *
                        b * std::pow(a->values_[a_base + i], b - 1)
                    );
                }
            }
        };
        return output;
    }

    static Tensor* sum(const std::vector<Tensor*>& tensors) {
        if (!unbatched_shapes_equal(tensors)) {
            throw std::runtime_error("[Operations::sum] Unbatched shapes of input tensors do not match");
        }
        if (tensors.size() < 2) {
            throw std::runtime_error(
                "[Operations::sum] At least two input tensors required. Use Operations::add for exactly two tensors."
            );
        }
        auto find_max = [](const std::vector<Tensor*>& tensors) -> std::size_t {
            std::size_t max_batch_size = 0;
            for (const auto& tensor : tensors) {
                if (tensor->batch_size() > max_batch_size) {
                    max_batch_size = tensor->batch_size();
                }
            }
            return max_batch_size;
        };
        const std::size_t batch_size = find_max(tensors);
        auto output = tensors[0]->pool_->new_tensor(tensors[0]->unbatched_shape(), batch_size, 0.0f);
        output->set_parents(tensors);
        output->forward_ = [tensors](Tensor *output) -> void {
            const std::size_t num_batches = std::max(output->batch_size(), static_cast<std::size_t>(1));
            // update values of output
            for (std::size_t batch = 0; batch < num_batches; batch++) {
                const std::size_t output_base = output->batched() ? batch * output->unbatched_size() : 0;
                const std::size_t first_base = tensors[0]->batched() ? batch * tensors[0]->unbatched_size() : 0;
                for (std::size_t i = 0; i < output->unbatched_size(); i++) {
                    output->values_[output_base + i] = tensors[0]->values_[first_base + i];
                }
                for (auto it = tensors.begin() + 1, end = tensors.end(); it != end; it++) {
                    const auto& tensor = *it;
                    const std::size_t base = tensor->batched() ? batch * tensor->unbatched_size() : 0;
                    for (std::size_t i = 0; i < output->unbatched_size(); i++) {
                        output->values_[output_base + i] += tensor->values_[base + i];
                    }
                }
            }
        };
        output->backward_ = [tensors](Tensor* output) -> void {
            // update gradients of parents of output
            const std::size_t num_batches = std::max(output->batch_size(), static_cast<std::size_t>(1));
            for (std::size_t batch = 0; batch < num_batches; batch++) {
                const std::size_t output_base = output->batched() ? batch * output->unbatched_size() : 0;
                for (auto& tensor : tensors) {
                    const std::size_t base = tensor->batched() ? batch * tensor->unbatched_size() : 0;
                    for (std::size_t i = 0; i < output->unbatched_size(); i++) {
                        tensor->gradients_[base + i] += output->gradients_[output_base + i];
                    }
                }
            }
        };
        return output;
    }

    static Tensor* mean_reduce(Tensor* a) {
        auto output = a->pool_->new_tensor({1}, 0, 0.0f);
        output->set_parents({a});
        output->forward_ = [](Tensor* output) -> void {
            Tensor* a = output->parents_[0];
            const std::size_t num_batches = std::max(a->batch_size(), static_cast<std::size_t>(1));
            // reset output value before accumulation to avoid repeated sums across forwards
            output->values_[0] = 0.0f;
            for (std::size_t batch = 0; batch < num_batches; batch++) {
                const std::size_t a_base = a->batched() ? batch * a->unbatched_size() : 0;
                float sum = 0.0f;
                for (std::size_t i = 0; i < a->unbatched_size(); i++) {
                    sum += a->values_[a_base + i];
                }
                output->values_[0] += sum / static_cast<float>(a->unbatched_size());
            }
            output->values_[0] /= static_cast<float>(num_batches);
        };
        output->backward_ = [](Tensor* output) -> void {
            Tensor* a = output->parents_[0];
            const std::size_t num_batches = std::max(output->batch_size(), static_cast<std::size_t>(1));
            for (std::size_t batch = 0; batch < num_batches; batch++) {
                const std::size_t a_base = a->batched() ? batch * a->unbatched_size() : 0;
                for (std::size_t i = 0; i < a->unbatched_size(); i++) {
                    a->gradients_[a_base + i] +=
                        output->gradients_[0] /
                        (static_cast<float>(a->unbatched_size()) * static_cast<float>(num_batches));
                }
            }
        };
        return output;
    }

    static Tensor* matmul(Tensor* a, Tensor* b, std::size_t contractions = 1) {
         if (contractions == 0) {
            throw std::runtime_error("[Operations::matmul] Contractions must be greater than zero");
        }
        if (a->batched() && b->batched() && (a->batch_size() != b->batch_size())) {
            throw std::runtime_error("[Operations::matmul] Batch sizes of input tensors do not match");
        }
        if (a->unbatched_shape().size() < contractions || b->unbatched_shape().size() < contractions) {
            throw std::runtime_error("[Operations::matmul] Not enough dimensions to contract");
        }
        for (std::size_t i = 0; i < contractions; i++) {
            if (a->unbatched_shape()[a->unbatched_shape().size() - 1 - i] != b->unbatched_shape()[i]) {
                throw std::runtime_error("[Operations::matmul] Contracted dimensions do not match");
            }
        }
        if (!same_pool(a, b)) {
            throw std::runtime_error("[Operations::matmul] Input tensors must be from the same pool");
        }
        std::vector<std::size_t> output_unbatched_shape;
        output_unbatched_shape.insert(
            output_unbatched_shape.end(),
            a->unbatched_shape().begin(),
            a->unbatched_shape().end() - contractions
        );
        output_unbatched_shape.insert(
            output_unbatched_shape.end(),
            b->unbatched_shape().begin() + contractions,
            b->unbatched_shape().end()
        );
        if (output_unbatched_shape.empty()) {
            output_unbatched_shape.push_back(1);
        }
        const std::size_t batch_size = std::max({a->batch_size(), b->batch_size()});
        Tensor* output = a->pool_->new_tensor(output_unbatched_shape, batch_size, 0.0f);
        output->set_parents({a, b});
        output->forward_ = [contractions](Tensor* output) -> void {
            Tensor* a = output->parents_[0];
            Tensor* b = output->parents_[1];
            const std::size_t num_batches = std::max(output->batch_size(), static_cast<std::size_t>(1));

            // compute flattened matrix dimensions for the contraction
            const auto& a_shape = a->unbatched_shape();
            const auto& b_shape = b->unbatched_shape();
            std::size_t a_rows = 1;
            for (std::size_t i = 0; i < a_shape.size() - contractions; i++) {
                a_rows *= a_shape[i];
            }
            std::size_t shared_k = 1;
            for (std::size_t i = a_shape.size() - contractions; i < a_shape.size(); i++) {
                shared_k *= a_shape[i];
            }
            std::size_t b_cols = 1;
            for (std::size_t i = contractions; i < b_shape.size(); i++) {
                b_cols *= b_shape[i];
            }
            for (std::size_t batch = 0; batch < num_batches; batch++) {
                const std::size_t out_base = output->batched() ? batch * output->unbatched_size() : 0;
                const std::size_t a_base = a->batched() ? batch * a->unbatched_size() : 0;
                const std::size_t b_base = b->batched() ? batch * b->unbatched_size() : 0;
                for (std::size_t r = 0; r < a_rows; r++) {
                    for (std::size_t c = 0; c < b_cols; c++) {
                        float sum = 0.0f;
                        for (std::size_t k = 0; k < shared_k; k++) {
                            const std::size_t a_index = a_base + r * shared_k + k;
                            const std::size_t b_index = b_base + k * b_cols + c;
                            sum += a->values_[a_index] * b->values_[b_index];
                        }
                        const std::size_t out_index = out_base + r * b_cols + c;
                        output->values_[out_index] = sum;
                    }
                }
            }
        };
        output->backward_ = [contractions](Tensor* output) -> void {
            Tensor* a = output->parents_[0];
            Tensor* b = output->parents_[1];
            const std::size_t num_batches = std::max(output->batch_size(), static_cast<std::size_t>(1));
            const auto& a_shape = a->unbatched_shape();
            const auto& b_shape = b->unbatched_shape();
            std::size_t a_rows = 1;
            for (std::size_t i = 0; i < a_shape.size() - contractions; i++) {
                a_rows *= a_shape[i];
            }
            std::size_t shared_k = 1;
            for (std::size_t i = a_shape.size() - contractions; i < a_shape.size(); i++) {
                shared_k *= a_shape[i];
            }
            std::size_t b_cols = 1;
            for (std::size_t i = contractions; i < b_shape.size(); i++) {
                b_cols *= b_shape[i];
            }
            for (std::size_t batch = 0; batch < num_batches; batch++) {
                const std::size_t out_base = output->batched() ? batch * output->unbatched_size() : 0;
                const std::size_t a_base = a->batched() ? batch * a->unbatched_size() : 0;
                const std::size_t b_base = b->batched() ? batch * b->unbatched_size() : 0;

                // ensure parent gradients exist
                if (a->gradients().empty()) a->gradients_ = std::vector<float>(a->size(), 0.0f);
                if (b->gradients().empty()) b->gradients_ = std::vector<float>(b->size(), 0.0f);

                // dA = dC @ B^T
                for (std::size_t r = 0; r < a_rows; r++) {
                    for (std::size_t k = 0; k < shared_k; k++) {
                        float sum = 0.0f;
                        for (std::size_t c = 0; c < b_cols; c++) {
                            const std::size_t out_index = out_base + r * b_cols + c;
                            const std::size_t b_index = b_base + k * b_cols + c;
                            sum += output->gradients_[out_index] * b->values_[b_index];
                        }
                        const std::size_t a_index = a_base + r * shared_k + k;
                        a->gradients_[a_index] += sum;
                    }
                }

                // dB = A^T @ dC
                for (std::size_t k = 0; k < shared_k; k++) {
                    for (std::size_t c = 0; c < b_cols; c++) {
                        float sum = 0.0f;
                        for (std::size_t r = 0; r < a_rows; r++) {
                            const std::size_t a_index = a_base + r * shared_k + k;
                            const std::size_t out_index = out_base + r * b_cols + c;
                            sum += a->values_[a_index] * output->gradients_[out_index];
                        }
                        const std::size_t b_index = b_base + k * b_cols + c;
                        b->gradients_[b_index] += sum;
                    }
                }
            }
            return;
        };
        return output;
    }
};

#endif // OPERATIONS_HPP