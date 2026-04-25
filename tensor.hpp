#ifndef TENSOR_HPP
#define TENSOR_HPP

#include <vector>
#include <memory>
#include <stdexcept>
#include <iostream>
#include <functional>
#include <initializer_list>
#include <algorithm>
#include <numeric>
#include <string>

template<typename... T>
concept IndexLike = (std::is_integral_v<T> && ...);

class Tensor {
private:
    std::vector<int> shape_;
    std::vector<int> strides_;
    std::vector<float> values_;
    std::vector<float> gradients_;
public:
    template<IndexLike... Dims>
    Tensor(Dims... dims)
    : shape_{static_cast<int>(dims)...}, strides_(shape_.size(), 1), gradients_(0) {
        int stride = 1;
        for (int i = shape_.size() - 1; i >= 0; i--) {
            strides_[i] = stride;
            stride *= shape_[i];
        }
        values_ = std::vector<float>(strides_[0] * shape_[0], 0.0f);
        gradients_ = std::vector<float>(values_.size(), 0.0f);
        shape_.shrink_to_fit();
        strides_.shrink_to_fit();
        values_.shrink_to_fit();
        gradients_.shrink_to_fit();
    }

    template<IndexLike... Indices>
    float& at(Indices... indices) {
        if (sizeof...(Indices) != shape_.size()) {
            throw std::invalid_argument("[Tensor][at] Number of indices must match tensor dimensions");
        }
        std::vector<int> indices_vec{static_cast<int>(indices)...};
        int index = std::inner_product(indices_vec.begin(), indices_vec.end(), strides_.begin(), 0);
        return values_[index];
    }

    template<IndexLike... Indices>
    float& grad_at(Indices... indices) {
        if (sizeof...(Indices) != shape_.size()) {
            throw std::invalid_argument("[Tensor][grad_at] Number of indices must match tensor dimensions");
        }
        std::vector<int> indices_vec{static_cast<int>(indices)...};
        int index = std::inner_product(indices_vec.begin(), indices_vec.end(), strides_.begin(), 0);
        return gradients_[index];
    }

    int dim() const {
        return shape_.size();
    }

    const std::vector<int>& shape() const {
        return shape_;
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
        result += to_string_recursive(values_, shape_, 0, value_index);
        result += ",\n\tgradients=";
        int gradient_index = 0;
        result += to_string_recursive(gradients_, shape_, 0, gradient_index);
        result += "\n)";
        return result;
    }

private:
    std::string to_string_recursive(const std::vector<float>& data, const std::vector<int>& shape, int dim, int& index) const {
        if ((std::size_t)dim == shape.size() - 1) {
            std::string result = "[";
            for (int i = 0; i < shape[dim]; i++) {
                result += std::to_string(data[index++]);
                if (i < shape[dim] - 1) {
                    result += ", ";
                }
            }
            result += "]";
            return result;
        }
        std::string result = "[";
        for (int i = 0; i < shape[dim]; i++) {
            result += to_string_recursive(data, shape, dim + 1, index);
            if (i < shape[dim] - 1) {
                result += ", ";
            }
        }
        result += "]";
        return result;
    }
};

#endif // TENSOR_HPP