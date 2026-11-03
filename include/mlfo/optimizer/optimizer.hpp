#ifndef MLFO_OPTIMIZER_HPP
#define MLFO_OPTIMIZER_HPP

#include <functional>
#include <memory>
#include <ostream>
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

    // the model this optimizer updates
    const mlfo::model::Model<T>& model() const {
        return model_;
    }

    // writes the optimizer's type name, hyperparameters and any training state (e.g. Adam's
    // moments), so training can resume where it left off. read back by
    // mlfo::optimizer::deserialize() in load.hpp
    virtual void serialize(std::ostream& os) const = 0;

    // a copy with the same hyperparameters and training state that updates model instead,
    // which must have the same architecture as this optimizer's model, e.g. its clone()
    virtual std::unique_ptr<Optimizer<T>> clone(mlfo::model::Model<T>& model) const = 0;

    virtual ~Optimizer() = default;
};

} // namespace mlfo::optimizer

#endif // MLFO_OPTIMIZER_HPP
