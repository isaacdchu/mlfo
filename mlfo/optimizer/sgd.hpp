#ifndef SGD_HPP
#define SGD_HPP

#include <cstddef>
#include <functional>
#include <memory>
#include <vector>

#include "../misc/concepts.hpp"
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

#endif // SGD_HPP
