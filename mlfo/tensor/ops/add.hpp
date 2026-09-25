#ifndef ADD_HPP
#define ADD_HPP

#include <vector>
#include <memory>
#include <concepts>

#include "../../misc/concepts.hpp"
#include "../tensor.hpp"

namespace mlfo::tensor::ops {

template <misc::Number T>
struct Add : public mlfo::tensor::Operation<T> {
private:
    static inline void add_forward(
        Tensor<T>& out,
        const Tensor<T>& a,
        const Tensor<T>& b
    ) {
        // perform element-wise addition
        for (std::size_t i = 0; i < out.values().size(); i++) {
            mlfo::tensor::Operation<T>::values(out)[i] = a.values()[i] + b.values()[i];
        }
    }

    static inline void add_backward(
        const Tensor<T>& out,
        Tensor<T>& a,
        Tensor<T>& b
    ) {
        // propagate gradients to a and b
        for (std::size_t i = 0; i < out.gradients().size(); i++) {
            mlfo::tensor::Operation<T>::gradients(a)[i] += out.gradients()[i];
            mlfo::tensor::Operation<T>::gradients(b)[i] += out.gradients()[i];
        }
    }

public:
    Add() = delete;
    static void call(
        DIRECTION direction,
        Tensor<T>& out,
        Tensor<T>& a,
        Tensor<T>& b
    ) {
        switch (direction) {
            case DIRECTION::FORWARD:
                add_forward(out, a, b);
                break;
            case DIRECTION::BACKWARD:
                add_backward(out, a, b);
                break;
            default:
                throw std::invalid_argument(
                    "[tensor::ops::add] Invalid direction"
                );
        }
    }
};

} // namespace mlfo::tensor::ops



#endif // ADD_HPP
