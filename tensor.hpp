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

class Tensor {
private:
    std::size_t size_;
    std::vector<std::size_t> shape_;
    std::vector<std::size_t> strides_;
    std::vector<float> values_;
    std::vector<float> gradients_;
public:
    Tensor() = delete;
    Tensor(const std::vector<std::size_t>& shape)
    : shape_(shape), strides_(shape.size(), 1), gradients_(0) {
        std::size_t stride = 1;
        for (int i = shape_.size() - 1; i >= 0; i--) {
            strides_[i] = stride;
            stride *= shape_[i];
        }
        size_ = stride;
        values_ = std::vector<float>(strides_[0] * shape_[0], 0.0f);
        gradients_ = std::vector<float>(values_.size(), 0.0f);
        shape_.shrink_to_fit();
        strides_.shrink_to_fit();
        values_.shrink_to_fit();
        gradients_.shrink_to_fit();
    }

    float& at(const std::vector<std::size_t>& indices) {
        return values_.at(calculate_index(indices));
    }

    float& at(std::vector<std::size_t>&& indices) {
        return values_.at(calculate_index(indices));
    }

    float& grad_at(const std::vector<std::size_t>& indices) {
        return gradients_.at(calculate_index(indices));
    }

    float& grad_at(std::vector<std::size_t>&& indices) {
        return gradients_.at(calculate_index(indices));
    }

    std::size_t size() const {
        return size_;
    }

    std::size_t dim() const {
        return shape_.size();
    }

    const std::vector<std::size_t>& shape() const {
        return shape_;
    }

    const std::vector<float>& values() const {
        return values_;
    }

    const std::vector<float>& gradients() const {
        return gradients_;
    }

    std::string to_string() const {
        std::string result = "Tensor(\n\tshape=[";
        for (size_t i = 0; i < shape_.size(); i++) {
            result += std::to_string(shape_[i]);
            if (i < shape_.size() - 1) {
                result += ", ";
            }
        }
        result += "],\n\tvalues=";
        int value_index = 0;
        result += to_string_helper(values_, shape_, 0, value_index);
        result += ",\n\tgradients=";
        int gradient_index = 0;
        result += to_string_helper(gradients_, shape_, 0, gradient_index);
        result += "\n)";
        return result;
    }

private:
    inline std::size_t calculate_index(const std::vector<std::size_t>& indices) const {
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