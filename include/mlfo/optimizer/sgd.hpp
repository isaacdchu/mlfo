#ifndef MLFO_SGD_HPP
#define MLFO_SGD_HPP

#include <cstddef>
#include <functional>
#include <istream>
#include <memory>
#include <ostream>
#include <vector>

#include "../misc/concepts.hpp"
#include "../misc/serialization.hpp"
#include "../model/model.hpp"
#include "../tensor/ops.hpp"
#include "../tensor/tensor.hpp"
#include "optimizer.hpp"

namespace mlfo::optimizer {

template <mlfo::misc::Number T>
class SGD : public Optimizer<T> {
private:
    T learning_rate_;
public:
    SGD(
        mlfo::model::Model<T>& model,
        T learning_rate
    ) :
    Optimizer<T>(model),
    learning_rate_(learning_rate)
    {
        if (learning_rate_ <= T{0}) {
            throw std::invalid_argument(
                "[mlfo::optimizer::SGD] Learning rate must be greater than zero"
            );
        }
    }

    void set_learning_rate(T learning_rate) {
        if (learning_rate <= T{0}) {
            throw std::invalid_argument(
                "[mlfo::optimizer::SGD::set_learning_rate] Learning rate must be greater than zero"
            );
        }
        learning_rate_ = learning_rate;
    }

    // SGD, then the learning rate. SGD has no other state
    void serialize(std::ostream& os) const override {
        os << "SGD\n";
        mlfo::misc::write_line(os, std::vector<T>{learning_rate_});
    }

    // reads what serialize() wrote after the type name
    static std::unique_ptr<SGD<T>> deserialize(std::istream& is, mlfo::model::Model<T>& model) {
        const T learning_rate = mlfo::misc::read_values<T>(is, 1)[0];
        return std::make_unique<SGD<T>>(model, learning_rate);
    }

    std::unique_ptr<Optimizer<T>> clone(mlfo::model::Model<T>& model) const override {
        return std::make_unique<SGD<T>>(model, learning_rate_);
    }

protected:
    void update() override {
        for (auto& param_ref : this->model_.parameters()) {
            mlfo::tensor::Tensor<T>& param = param_ref.get();
            std::vector<T>& values = mlfo::tensor::Operation<T>::values(param);
            const std::vector<T>& gradients = param.gradients();
            // values and gradients share the same layout
            for (std::size_t i = 0; i < values.size(); i++) {
                values[i] -= learning_rate_ * gradients[i];
            }
        }
    }
};

} // namespace mlfo::optimizer

#endif // MLFO_SGD_HPP
