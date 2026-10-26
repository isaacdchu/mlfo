#ifndef OPTIMIZER_HPP
#define OPTIMIZER_HPP

#include <functional>
#include <memory>
#include <utility>
#include <vector>

#include "../misc/concepts.hpp"
#include "../model/model.hpp"
#include "../tensor/ops.hpp"
#include "../tensor/tensor.hpp"

namespace mlfo::optimizer {

// non-owning, the model must outlive the optimizer
template <mlfo::misc::Number T>
class Optimizer : public mlfo::tensor::Operation<T> {
protected:
    mlfo::model::Model<T>& model_;

    // updates the values of the model's parameters, called by step()
    virtual void update() = 0;
public:
    Optimizer(mlfo::model::Model<T>& model) :
    model_(model)
    {
        // nothing
    }

    virtual void zero_grad() final {
        for (auto& param_ref : model_.parameters()) {
            mlfo::tensor::Tensor<T>& param = param_ref.get();
            std::vector<T>& gradients = mlfo::tensor::Operation<T>::gradients(param);
            std::fill(gradients.begin(), gradients.end(), T{0});
        }
    }

    virtual void step() final {
        update();
        // invalidate the model's graph so that the next forward() uses the updated values
        model_.invalidate();
    }

    virtual ~Optimizer() = default;
};

} // namespace mlfo::optimizer

#endif // OPTIMIZER_HPP
