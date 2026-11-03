#ifndef MLFO_ADAM_HPP
#define MLFO_ADAM_HPP

#include <cmath>
#include <cstddef>
#include <functional>
#include <stdexcept>
#include <vector>

#include "../misc/concepts.hpp"
#include "../model/model.hpp"
#include "../tensor/ops.hpp"
#include "../tensor/tensor.hpp"
#include "optimizer.hpp"

namespace mlfo::optimizer {

// Adam (Kingma & Ba, 2014)
// m = beta1 * m + (1 - beta1) * g
// v = beta2 * v + (1 - beta2) * g^2
// param -= learning_rate * m_hat / (sqrt(v_hat) + epsilon)
// where m_hat = m / (1 - beta1^t) and v_hat = v / (1 - beta2^t)
template <mlfo::misc::Number T>
class Adam : public Optimizer<T> {
private:
    T learning_rate_;
    T beta1_;
    T beta2_;
    T epsilon_;
    // beta1^t and beta2^t, updated every step
    T beta1_power_ = T{1};
    T beta2_power_ = T{1};
    // first and second moment estimates, one per parameter, same layout as the values
    std::vector<std::vector<T>> m_;
    std::vector<std::vector<T>> v_;

    // allocates the moments on the first step, so the model can be built after the optimizer
    void init_moments() {
        const auto& parameters = this->model_.parameters();
        if (m_.size() == parameters.size()) {
            return;
        }
        m_.clear();
        v_.clear();
        m_.reserve(parameters.size());
        v_.reserve(parameters.size());
        for (const auto& param_ref : parameters) {
            m_.emplace_back(param_ref.get().size(), T{0});
            v_.emplace_back(param_ref.get().size(), T{0});
        }
    }
public:
    Adam(
        mlfo::model::Model<T>& model,
        T learning_rate = T{0.001},
        T beta1 = T{0.9},
        T beta2 = T{0.999},
        T epsilon = T{1e-8}
    ) :
    Optimizer<T>(model),
    learning_rate_(learning_rate),
    beta1_(beta1),
    beta2_(beta2),
    epsilon_(epsilon)
    {
        if (learning_rate_ <= T{0}) {
            throw std::invalid_argument(
                "[mlfo::optimizer::Adam] Learning rate must be greater than zero"
            );
        }
        if (beta1_ < T{0} || beta1_ >= T{1} || beta2_ < T{0} || beta2_ >= T{1}) {
            throw std::invalid_argument(
                "[mlfo::optimizer::Adam] Betas must be in [0, 1)"
            );
        }
        if (epsilon_ <= T{0}) {
            throw std::invalid_argument(
                "[mlfo::optimizer::Adam] Epsilon must be greater than zero"
            );
        }
    }

protected:
    void update() override {
        init_moments();
        beta1_power_ *= beta1_;
        beta2_power_ *= beta2_;
        const T bias_correction1 = T{1} - beta1_power_;
        const T bias_correction2 = T{1} - beta2_power_;
        const auto& parameters = this->model_.parameters();
        for (std::size_t p = 0; p < parameters.size(); p++) {
            mlfo::tensor::Tensor<T>& param = parameters[p].get();
            std::vector<T>& values = mlfo::tensor::Operation<T>::values(param);
            const std::vector<T>& gradients = param.gradients();
            std::vector<T>& m = m_[p];
            std::vector<T>& v = v_[p];
            // values, gradients and moments share the same layout
            for (std::size_t i = 0; i < values.size(); i++) {
                const T g = gradients[i];
                m[i] = beta1_ * m[i] + (T{1} - beta1_) * g;
                v[i] = beta2_ * v[i] + (T{1} - beta2_) * g * g;
                const T m_hat = m[i] / bias_correction1;
                const T v_hat = v[i] / bias_correction2;
                values[i] -= learning_rate_ * m_hat / (std::sqrt(v_hat) + epsilon_);
            }
        }
    }
};

} // namespace mlfo::optimizer

#endif // MLFO_ADAM_HPP
