#ifndef TENSOR_HPP
#define TENSOR_HPP

#include <vector>
#include <memory>
#include <stdexcept>
#include <iostream>
#include <functional>
#include <algorithm>
#include <numeric>
#include <string>

class Pool;

class Tensor {
friend class Operations;
friend class Pool;
private:
    Pool* pool_;
    bool batched_;
    std::size_t size_;
    std::size_t unbatched_size_;
    std::vector<std::size_t> shape_;
    std::vector<std::size_t> unbatched_shape_;
    std::vector<std::size_t> strides_;
    std::vector<float> values_;
    std::vector<float> gradients_;
    std::vector<Tensor*> parents_;
    std::vector<Tensor*> children_;
    std::function<void(Tensor*)> forward_;
    std::function<void(Tensor*)> backward_;
    bool forward_dirty_;
    bool backward_dirty_;

public:
    Tensor(const std::vector<std::size_t>& shape, std::size_t batch_size = 0, float fill_value = 0.0f) {
        // if batch_size is specified, it will be the first dimension of the shape
        if (shape.empty()) {
            throw std::runtime_error("[Tensor::Tensor] Shape cannot be empty");
        }
        for (std::size_t dim : shape) {
            if (dim == 0) {
                throw std::runtime_error("[Tensor::Tensor] Shape dimensions cannot be zero");
            }
        }
        if (batch_size > 0) {
            batched_ = true;
            shape_ = {batch_size};
            shape_.append_range(shape);
        } else {
            batched_ = false;
            shape_ = shape;
        }
        unbatched_shape_ = shape;
        strides_ = compute_strides(shape_);
        size_ = strides_[0] * shape_[0];
        unbatched_size_ = size_ / std::max(batch_size, static_cast<std::size_t>(1));
        values_ = std::vector<float>(size_, fill_value);
        shape_.shrink_to_fit();
        strides_.shrink_to_fit();
        forward_dirty_ = true;
        backward_dirty_ = true;
        forward_ = nullptr;
        backward_ = nullptr;
    }

    Tensor() = delete;

    Tensor(const Tensor& other) = default;

    Tensor(Tensor&& other) noexcept = default;

    void forward() {
        if (!forward_ || !forward_dirty_) {
            return;
        }
        // make parents forward first
        for (Tensor* parent : parents_) {
            parent->forward();
        }
        if (forward_dirty_) {
            forward_(this);
            forward_dirty_ = false;
        }
    }

    void backward() {
        if (!backward_ || !backward_dirty_) {
            return;
        }
        if (gradients_.empty()) {
            gradients_ = std::vector<float>(size_, 0.0f);
        }
        for (Tensor* parent : parents_) {
            if (parent->gradients_.empty()) {
                parent->gradients_ = std::vector<float>(parent->size_, 0.0f);
            }
        }
        if (backward_dirty_) {
            backward_(this);
            backward_dirty_ = false;
        }
        // make parents backward
        for (Tensor* parent : parents_) {
            parent->backward();
        }
    }

    void set_parents(const std::vector<Tensor*>& parents) {
        if (!parents_.empty()) {
            for (Tensor* parent : parents_) {
                parent->children_.erase(
                    std::remove(parent->children_.begin(), parent->children_.end(), this),
                    parent->children_.end()
                );
            }
        }
        parents_ = parents;
        for (Tensor* parent : parents) {
            parent->children_.push_back(this);
        }
    }

    void set_values(std::vector<float>&& values, std::size_t batch_size = 0) {
        if (!batched_ && batch_size > 0) {
            // unbatched -> batched
            batched_ = true;
            shape_.insert(shape_.begin(), batch_size);
            strides_ = compute_strides(shape_);
            size_ = strides_[0] * shape_[0];
            unbatched_size_ = size_ / batch_size;
        } else if (batched_ && batch_size == 0) {
            // batched -> unbatched
            batched_ = false;
            shape_.erase(shape_.begin());
            strides_ = compute_strides(shape_);
            size_ = strides_[0] * shape_[0];
            unbatched_size_ = size_;
        }
        if (values.size() != size_) {
            throw std::runtime_error("[Tensor::set_values] Values size does not match tensor size");
        }
        values_ = std::move(values);
        // all successors are forward dirty
        std::vector<Tensor*> stack = children_;
        while (!stack.empty()) {
            Tensor* current = stack.back();
            stack.pop_back();
            current->forward_dirty_ = true;
            stack.append_range(current->children_);
        }
    }

    void set_gradients(std::vector<float>&& gradients) {
        if (gradients.size() != size_) {
            throw std::runtime_error("[Tensor::set_gradients] Gradients size does not match tensor size");
        }
        gradients_ = std::move(gradients);
        // all predecessors are backward dirty
        std::vector<Tensor*> stack = parents_;
        while (!stack.empty()) {
            Tensor* current = stack.back();
            stack.pop_back();
            current->backward_dirty_ = true;
            stack.append_range(current->parents_);
        }
    }

    float at(const std::vector<std::size_t>& indices) const {
        return values_.at(calculate_index(indices));
    }

    float& at(const std::vector<std::size_t>& indices) {
        return values_.at(calculate_index(indices));
    }

    float grad_at(const std::vector<std::size_t>& indices) const {
        return gradients_.at(calculate_index(indices));
    }

    float& grad_at(const std::vector<std::size_t>& indices) {
        return gradients_.at(calculate_index(indices));
    }

    void zero_grad() {
        std::fill(gradients_.begin(), gradients_.end(), 0.0f);
    }

    std::vector<std::size_t> expand_index(std::size_t flat_index) const {
        std::vector<std::size_t> indices(shape_.size(), 0);
        for (std::size_t i = 0; i < shape_.size(); i++) {
            indices[i] = flat_index / strides_[i];
            flat_index %= strides_[i];
        }
        return indices;
    }

    std::size_t flatten_indices(const std::vector<std::size_t>& indices) const {
        return std::inner_product(indices.begin(), indices.end(), strides_.begin(), 0);
    }

    std::size_t size() const {
        return size_;
    }

    std::size_t unbatched_size() const {
        return unbatched_size_;
    }

    std::size_t batch_size() const {
        if (!batched_) {
            return 0;
        }
        return shape_[0];
    }
    
    bool batched() const {
        return batched_;
    }

    std::size_t ndim() const {
        return unbatched_shape_.size();
    }

    const std::vector<std::size_t>& shape() const {
        return shape_;
    }

    const std::vector<std::size_t>& unbatched_shape() const {
        return unbatched_shape_;
    }

    const std::vector<float>& values() const {
        return values_;
    }

    const std::vector<float>& gradients() const {
        return gradients_;
    }

    const std::vector<Tensor*>& parents() const {
        return parents_;
    }

    const std::vector<Tensor*>& children() const {
        return children_;
    }

    std::string to_string() const {
        std::string result = "Tensor(\n\tshape=[";
        for (size_t i = 0; i < shape_.size(); i++) {
            result += std::to_string(shape_[i]);
            if (i < shape_.size() - 1) {
                result += ", ";
            }
        }
        result += "],\n\tbatches=" + std::to_string(batch_size());
        result += ",\n\tvalues=";
        int value_index = 0;
        result += to_string_helper(values_, shape_, 0, value_index);
        if (gradients_.empty()) {
            result += "\n)";
            return result;
        }
        result += ",\n\tgradients=";
        int gradient_index = 0;
        result += to_string_helper(gradients_, shape_, 0, gradient_index);
        result += "\n)";
        return result;
    }

    static std::unique_ptr<Tensor> factory(
        const std::vector<std::size_t>& shape,
        std::size_t batch_size = 0,
        float fill_value = 0.0f
    ) {
        return std::make_unique<Tensor>(shape, batch_size, fill_value);
    }

private:
    static std::vector<std::size_t> compute_strides(const std::vector<std::size_t>& shape) {
        std::vector<std::size_t> strides(shape.size(), 1);
        for (int i = shape.size() - 2; i >= 0; i--) {
            strides[i] = strides[i + 1] * shape[i + 1];
        }
        return strides;
    }

    std::size_t calculate_index(const std::vector<std::size_t>& indices) const {
        return std::inner_product(indices.begin(), indices.end(), strides_.begin(), 0);
    }

    std::string to_string_helper(const std::vector<float>& data, const std::vector<std::size_t>& shape, std::size_t dim, int& index) const {
        if (dim == shape.size() - 1) {
            std::string result = "[";
            for (std::size_t i = 0; i < shape[dim]; i++) {
                result += std::to_string(data[index++]);
                if (i < shape[dim] - 1) {
                    result += ", ";
                }
            }
            result += "]";
            return result;
        }
        std::string result = "[";
        for (std::size_t i = 0; i < shape[dim]; i++) {
            result += to_string_helper(data, shape, dim + 1, index);
            if (i < shape[dim] - 1) {
                result += ", ";
            }
        }
        result += "]";
        return result;
    }
};

#endif // TENSOR_HPP