#ifndef OPERATIONS_HPP
#define OPERATIONS_HPP

#include "tensor.hpp"

#include <vector>
#include <memory>
#include <iostream>
#include <print>

class Operations {
private:
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

    static std::unique_ptr<Tensor> matmul(Tensor* a, Tensor* b, std::size_t contractions = 1) {
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
        const std::size_t batch_size = std::max({a->batch_size(), b->batch_size()});
        std::unique_ptr<Tensor> output = std::make_unique<Tensor>(output_unbatched_shape, batch_size, 0.0f);
        output->set_parents({a, b});
        output->forward_ = [contractions](Tensor* output) -> void {
            Tensor* a = output->parents_[0];
            Tensor* b = output->parents_[1];
            const std::size_t num_batches = std::max(
                output->batch_size(),
                static_cast<std::size_t>(1)
            );
            const std::size_t output_start_index = output->batched() ? 1 : 0;
            const std::size_t a_start_index = a->batched() ? 1 : 0;
            const std::size_t b_start_index = b->batched() ? 1 : 0;
            for (std::size_t batch = 0; batch < num_batches; batch++) {
                // perform output = a @ b for the current batch
                auto output_indices = std::vector<std::size_t>(output->shape().size(), 0);
                if (output->batched()) {
                    output_indices[0] = batch;
                }
                auto a_indices = std::vector<std::size_t>(a->shape().size(), 0);
                if (a->batched()) {
                    a_indices[0] = batch;
                }
                auto b_indices = std::vector<std::size_t>(b->shape().size(), 0);
                if (b->batched()) {
                    b_indices[0] = batch;
                }
                bool output_done = false;
                while (!output_done) {
                    // calculate corresponding a and b indices
                    for (std::size_t i = a_start_index; i < a_indices.size() - contractions; i++) {
                        a_indices[i] = output_indices[i];
                    }
                    for (std::size_t i = 0; i < b_indices.size() - contractions - b_start_index; i++) {
                        b_indices[b_indices.size() - 1 - i] = output_indices[output_indices.size() - 1 - i];
                    }
                    // calculate output value at the current indices
                    // for i, j, ... in contracted dimensions
                    // output[output_indices] = sum(a[a_indices][i][j][...] * b[...][j][i][b_indices])
                    bool element_done = false;
                    float value = 0.0f;
                    
                    // iterate through all combinations of contracted indices
                    while (true) {
                        // print out a_indices and b_indices
                        value += a->at(a_indices) * b->at(b_indices);
                        // calculate next combination of contracted indices
                        b_indices[b_start_index]++;
                        for (std::size_t dim = b_start_index; dim < b_start_index + contractions; dim++) {
                            if (b_indices[dim] < b->shape()[dim]) {
                                // no need to carry to the next dimension
                                break;
                            }
                            b_indices[dim] = 0;
                            if (dim + 1 >= b_start_index + contractions) {
                                // last dimension is done
                                element_done = true;
                                break;
                            }
                            b_indices[dim + 1]++;
                        }
                        if (element_done) {
                            break;
                        }
                        // a_indices are the reverse of b_indices for the contracted dimensions
                        for (std::size_t dim = 0; dim < contractions; dim++) {
                            a_indices[a_indices.size() - dim - 1] = b_indices[b_start_index + dim];
                        }
                    }
                    output->at(output_indices) = value;
                    // calculate indices for next output value
                    output_indices[output_start_index]++;
                    for (std::size_t dim = output_start_index; dim < output_indices.size(); dim++) {
                        if (output_indices[dim] < output->shape()[dim]) {
                            // no need to carry to the next dimension
                            break;
                        }
                        output_indices[dim] = 0;
                        if (dim + 1 >= output_indices.size()) {
                            // last dimension is done
                            output_done = true;
                            break;
                        }
                        output_indices[dim + 1]++;
                    }
                }
            }
        };
        output->backward_ = [contractions](Tensor* output) -> void {
            Tensor* a = output->parents_[0];
            Tensor* b = output->parents_[1];
            const std::size_t num_batches = std::max(
                output->batch_size(),
                static_cast<std::size_t>(1)
            );
            const std::size_t output_start_index = output->batched() ? 1 : 0;
            const std::size_t a_start_index = a->batched() ? 1 : 0;
            const std::size_t b_start_index = b->batched() ? 1 : 0;
            std::cout << "part a gradients" << std::endl;
            // a_gradients = dL/da = sum over non-contracted dimensions (dL/doutput * B)
            for (std::size_t batch = 0; batch < num_batches; batch++) {
                // perform output = dL/doutput * B for the current batch
                auto output_indices = std::vector<std::size_t>(output->shape().size(), 0);
                if (output->batched()) {
                    output_indices[0] = batch;
                }
                auto a_indices = std::vector<std::size_t>(a->shape().size(), 0);
                if (a->batched()) {
                    a_indices[0] = batch;
                }
                auto b_indices = std::vector<std::size_t>(b->shape().size(), 0);
                if (b->batched()) {
                    b_indices[0] = batch;
                }
                bool a_done = false;
                while (!a_done) {
                    // calculate corresponding output and b indices
                    for (std::size_t i = 0; i < a->shape().size() - a_start_index - contractions; i++) {
                        output_indices[output_start_index + i] = a_indices[a_start_index + i];
                    }
                    for (std::size_t i = 0; i < contractions; i++) {
                        b_indices[b_start_index + i] = a_indices[a_indices.size() - 1 - i];
                    }
                    // calculate dL/da value at the current indices
                    // dL/da[a_indices] = sum(dL/doutput[output_indices] * b[b_indices])
                    bool element_done = false;
                    float gradient = 0.0f;
                    // iterate through all combinations of contracted indices
                    while (true) {
                        // print out output_indices and b_indices
                        // std::cout << "a1" << std::endl;
                        // print_vector(output_indices);
                        // print_vector(b_indices);
                        gradient += output->grad_at(output_indices) * b->at(b_indices);
                        // std::cout << "a2" << std::endl;
                        // calculate next combination of non-contracted indices
                        b_indices[b_start_index + contractions]++;
                        for (std::size_t dim = b_start_index + contractions; dim < b_indices.size(); dim++) {
                            if (b_indices[dim] < b->shape()[dim]) {
                                // no need to carry to the next dimension
                                break;
                            }
                            b_indices[dim] = 0;
                            if (dim + 1 >= b_indices.size()) {
                                // last dimension is done
                                element_done = true;
                                break;
                            }
                            b_indices[dim + 1]++;
                        }
                        if (element_done) {
                            break;
                        }
                        // output_indices are non-contracted a_indices and b_indices
                        for (std::size_t dim = b_start_index + contractions; dim < b->shape().size(); dim++) {
                            output_indices[dim] = b_indices[dim];
                        }
                    }
                    a->grad_at(a_indices) += gradient;
                    // calculate indices for next dL/da value
                    const std::size_t start_index = a->batched() ? 1 : 0;
                    a_indices[start_index]++;
                    for (std::size_t dim = start_index; dim < a_indices.size(); dim++) {
                        if (a_indices[dim] < a->shape()[dim]) {
                            // no need to carry to the next dimension
                            break;
                        }
                        a_indices[dim] = 0;
                        if (dim + 1 >= a_indices.size()) {
                            // last dimension is done
                            a_done = true;
                            break;
                        }
                        a_indices[dim + 1]++;
                    }
                }
            }
            std::cout << "part b gradients" << std::endl;
            // b_gradients = dL/db = sum over batches of sum over non-contracted dimensions (A * dL/doutput)
            for (std::size_t batch = 0; batch < num_batches; batch++) {
                // perform output = dL/doutput * B for the current batch
                auto output_indices = std::vector<std::size_t>(output->shape().size(), 0);
                if (output->batched()) {
                    output_indices[0] = batch;
                }
                auto a_indices = std::vector<std::size_t>(a->shape().size(), 0);
                if (a->batched()) {
                    a_indices[0] = batch;
                }
                auto b_indices = std::vector<std::size_t>(b->shape().size(), 0);
                if (b->batched()) {
                    b_indices[0] = batch;
                }
                bool b_done = false;
                while (!b_done) {
                    // calculate corresponding a and output indices
                    for (std::size_t i = 0; i < contractions; i++) {
                        a_indices[a_indices.size() - 1 - i] = b_indices[b_start_index + i];
                    }
                    for (std::size_t i = 0; i < b_indices.size() - b_start_index - contractions; i++) {
                        output_indices[output_start_index + i] = b_indices[b_start_index + contractions + i];
                    }
                    // calculate dL/db at the current indices
                    // dL/db[b_indices] = sum(a[a_indices] * dL/doutput[output_indices])
                    bool element_done = false;
                    float gradient = 0.0f;
                    // iterate through all combinations of contracted indices
                    while (true) {
                        // print out a_indices and output_indices
                        // std::cout << "b1" << std::endl;
                        // print_vector(a_indices);
                        // print_vector(output_indices);
                        gradient += a->at(a_indices) * output->grad_at(output_indices);
                        // std::cout << "b2" << std::endl;
                        // calculate next combination of non-contracted indices
                        const std::size_t start_index = a->batched() ? 1 : 0;
                        a_indices[start_index]++;
                        for (std::size_t dim = start_index; dim < a_indices.size() - contractions; dim++) {
                            if (a_indices[dim] < a->shape()[dim]) {
                                // no need to carry to the next dimension
                                break;
                            }
                            a_indices[dim] = 0;
                            if (dim + 1 >= a_indices.size() - contractions) {
                                // last dimension is done
                                element_done = true;
                                break;
                            }
                            a_indices[dim + 1]++;
                        }
                        if (element_done) {
                            break;
                        }
                        // output_indices are non-contracted a_indices and b_indices
                        for (std::size_t dim = start_index; dim < a_indices.size() - contractions; dim++) {
                            output_indices[output_start_index + dim] = a_indices[dim];
                        }
                    }
                    b->grad_at(b_indices) += gradient;
                    // calculate indices for next dL/da value
                    // start at 0 since the b is not batched
                    b_indices[b_start_index]++;
                    for (std::size_t dim = b_start_index; dim < b_indices.size(); dim++) {
                        if (b_indices[dim] < b->shape()[dim]) {
                            // no need to carry to the next dimension
                            break;
                        }
                        b_indices[dim] = 0;
                        if (dim + 1 >= b_indices.size()) {
                            // last dimension is done
                            b_done = true;
                            break;
                        }
                        b_indices[dim + 1]++;
                    }
                }
            }
            return;
        };
        return output;
    }
};

#endif // OPERATIONS_HPP