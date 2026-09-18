#ifndef TENSOR_HPP
#define TENSOR_HPP

#include <array>
#include <algorithm>
#include <stdexcept>
#include <string>
#include <vector>
#include <print>
#include <iostream>

#include "../misc/concepts.hpp"

namespace mlfo::tensor {

template <mlfo::misc::Number T>
class Tensor {
private:
    std::vector<T> values_;
    std::vector<T> gradients_;
    std::size_t batch_size_; // number of batches the tensor contains
    std::vector<std::size_t> shape_; // unbatched shape
    std::vector<std::size_t> strides_; // strides for each non-batch dimension
    std::size_t batch_stride_; // stride for the batch dimension
    std::size_t rank_; // rank of the tensor (number of dimensions). batch is not a dimension
    std::size_t size_; // number of elements in the tensor (including batches)

    std::string to_string_helper(
        const std::vector<T>& data,
        const std::vector<std::size_t>& shape,
        std::size_t dim,
        std::size_t& index
    ) const {
        if (dim + 1 == shape.size()) {
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
public:
    Tensor(std::size_t batch_size, const std::vector<std::size_t>& shape) :
    batch_size_(batch_size),
    shape_(shape),
    strides_(shape.size()),
    batch_stride_(0),
    rank_(shape.size()),
    size_(0)
    {
        if (batch_size_ == 0) {
            throw std::invalid_argument(
                "[mlfo::tensor::Tensor] Batch size must be greater than zero"
            );
        }
        if (shape.empty()) {
            throw std::invalid_argument("[mlfo::tensor::Tensor] Shape cannot be empty");
        }
        size_ = batch_size_;
        for (std::size_t i = 0; i < shape_.size(); i++) {
            if (shape_[i] == 0) {
                throw std::invalid_argument(
                    "[mlfo::tensor::Tensor] Shape dimensions must be greater than zero"
                );
            }
            size_ *= shape_[i];
        }
        values_.resize(size_);
        std::fill(values_.begin(), values_.end(), T{0});

        strides_[shape_.size() - 1] = 1;
        for (std::size_t i = shape_.size() - 1; i > 0; i--) {
            strides_[i - 1] = strides_[i] * shape_[i];
        }
        batch_stride_ = strides_[0] * shape_[0];
    }

    std::size_t flatten_index(std::size_t batch, const std::vector<std::size_t>& indices) const {
        // does not perform bounds checking
        std::size_t index = batch * batch_stride_;
        for (std::size_t i = 0; i < strides_.size(); i++) {
            index += indices[i] * strides_[i];
        }
        return index;
    }

    T operator[](std::size_t batch, const std::vector<std::size_t>& indices) const {
        return values_[flatten_index(batch, indices)];
    }

    T& operator[](std::size_t batch, const std::vector<std::size_t>& indices) {
        return values_[flatten_index(batch, indices)];
    }

    std::size_t rank() const {
        return rank_;
    }

    const std::vector<std::size_t>& shape() const {
        return shape_;
    }

    std::size_t batch_size() const {
        return batch_size_;
    }

    const std::vector<std::size_t>& strides() const {
        return strides_;
    }

    const std::vector<T>& values() const {
        return values_;
    }

    const std::vector<T>& gradients() const {
        return gradients_;
    }

    std::size_t size() const {
        return values_.size();
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
        std::size_t value_index = 0;
        std::vector<std::size_t> shape_with_batch = shape_;
        shape_with_batch.insert(shape_with_batch.begin(), batch_size());
        result += to_string_helper(values_, shape_with_batch, 0, value_index);
        if (gradients_.empty()) {
            result += "\n)";
            return result;
        }
        result += ",\n\tgradients=";
        std::size_t gradient_index = 0;
        std::vector<std::size_t> gradient_shape = shape_;
        if (gradients_.size() == size_) {
            gradient_shape.insert(gradient_shape.begin(), batch_size_);
        }
        result += to_string_helper(gradients_, gradient_shape, 0, gradient_index);
        result += "\n)";
        return result;
    }
};

} // namespace mlfo::tensor

#endif // TENSOR_HPP
