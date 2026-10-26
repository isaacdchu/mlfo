#ifndef RELU_ACTIVATION_HPP
#define RELU_ACTIVATION_HPP

#include "../../tensor/activation/relu.hpp"

namespace mlfo::layer::activation {

template <misc::Number T>
class ReLUActivation : public mlfo::layer::Layer<T> {
public:
    ReLUActivation() = default;
    ~ReLUActivation() override = default;

    void forward(Tensor<T>& out, const Tensor<T>& a) override {
        mlfo::tensor::activation::ReLU<T>::call(
            ops::DIRECTION::FORWARD,
            out,
            const_cast<Tensor<T>&>(a)
        );
    }

    void backward(const Tensor<T>& out, Tensor<T>& a) override {
        mlfo::tensor::activation::ReLU<T>::call(
            ops::DIRECTION::BACKWARD,
            const_cast<Tensor<T>&>(out),
            a
        );
    }
};

} // namespace mlfo::layer::activation

#endif // RELU_ACTIVATION_HPP
