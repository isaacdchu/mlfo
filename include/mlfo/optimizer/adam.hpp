#ifndef MLFO_ADAM_HPP
#define MLFO_ADAM_HPP

#include <cmath>
#include <cstddef>
#include <functional>
#include <istream>
#include <memory>
#include <ostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "../misc/concepts.hpp"
#include "../misc/serialization.hpp"
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

    // throws unless the moments, if allocated, fit the model's parameters
    void check_moments(const char* where) const {
        if (m_.empty()) {
            return;
        }
        const auto& parameters = this->model_.parameters();
        bool fits = m_.size() == parameters.size() && v_.size() == parameters.size();
        for (std::size_t p = 0; fits && p < parameters.size(); p++) {
            fits = m_[p].size() == parameters[p].get().size() &&
                v_[p].size() == parameters[p].get().size();
        }
        if (!fits) {
            throw std::invalid_argument(
                std::string("[mlfo::optimizer::Adam::") + where +
                "] Moments do not match the model's parameters"
            );
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

    void set_learning_rate(T learning_rate) {
        if (learning_rate <= T{0}) {
            throw std::invalid_argument(
                "[mlfo::optimizer::Adam::set_learning_rate] Learning rate must be greater than zero"
            );
        }
        learning_rate_ = learning_rate;
    }

    // the number of steps taken so far is implied by beta1^t and beta2^t
    // Adam, then learning rate, beta1, beta2 and epsilon, then beta1^t and beta2^t, then the
    // number of parameters with moments (0 before the first step), then for each parameter its
    // first moment followed by its second moment
    void serialize(std::ostream& os) const override {
        os << "Adam\n";
        mlfo::misc::write_line(os, std::vector<T>{learning_rate_, beta1_, beta2_, epsilon_});
        mlfo::misc::write_line(os, std::vector<T>{beta1_power_, beta2_power_});
        mlfo::misc::write_line(os, std::vector<std::size_t>{m_.size()});
        for (std::size_t p = 0; p < m_.size(); p++) {
            mlfo::misc::write_line(os, m_[p]);
            mlfo::misc::write_line(os, v_[p]);
        }
    }

    // reads what serialize() wrote after the type name. model must be built, as the moments'
    // sizes are taken from its parameters
    static std::unique_ptr<Adam<T>> deserialize(std::istream& is, mlfo::model::Model<T>& model) {
        const std::vector<T> hyperparameters = mlfo::misc::read_values<T>(is, 4);
        auto adam = std::make_unique<Adam<T>>(
            model, hyperparameters[0], hyperparameters[1], hyperparameters[2], hyperparameters[3]
        );
        const std::vector<T> powers = mlfo::misc::read_values<T>(is, 2);
        adam->beta1_power_ = powers[0];
        adam->beta2_power_ = powers[1];
        const std::size_t count = mlfo::misc::read_values<std::size_t>(is, 1)[0];
        if (count == 0) {
            return adam;
        }
        const auto& parameters = model.parameters();
        if (count != parameters.size()) {
            throw std::runtime_error(
                "[mlfo::optimizer::Adam::deserialize] Moments do not match the model's parameters"
            );
        }
        adam->m_.reserve(count);
        adam->v_.reserve(count);
        for (const auto& param_ref : parameters) {
            adam->m_.push_back(mlfo::misc::read_values<T>(is, param_ref.get().size()));
            adam->v_.push_back(mlfo::misc::read_values<T>(is, param_ref.get().size()));
        }
        return adam;
    }

    std::unique_ptr<Optimizer<T>> clone(mlfo::model::Model<T>& model) const override {
        auto adam = std::make_unique<Adam<T>>(model, learning_rate_, beta1_, beta2_, epsilon_);
        adam->beta1_power_ = beta1_power_;
        adam->beta2_power_ = beta2_power_;
        adam->m_ = m_;
        adam->v_ = v_;
        adam->check_moments("clone");
        return adam;
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
