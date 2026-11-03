#ifndef MLFO_OPTIMIZER_LOAD_HPP
#define MLFO_OPTIMIZER_LOAD_HPP

#include <istream>
#include <memory>
#include <stdexcept>
#include <string>

#include "../misc/concepts.hpp"
#include "../misc/serialization.hpp"
#include "../model/model.hpp"
#include "adam.hpp"
#include "optimizer.hpp"
#include "sgd.hpp"

// like model/load.hpp, this is the only place that has to know every concrete optimizer type.
// a new optimizer is added to the chain in deserialize() and included above

namespace mlfo::optimizer {

// reads an optimizer written by Optimizer::serialize(), attached to model, which must be built
// and have the same architecture as the model the optimizer was saved with
template <mlfo::misc::Number T>
std::unique_ptr<Optimizer<T>> deserialize(std::istream& is, mlfo::model::Model<T>& model) {
    const std::string type = mlfo::misc::read_line(is);
    if (type == "SGD") {
        return SGD<T>::deserialize(is, model);
    }
    if (type == "Adam") {
        return Adam<T>::deserialize(is, model);
    }
    throw std::runtime_error(
        "[mlfo::optimizer::deserialize] Unknown optimizer type '" + type + "'"
    );
}

} // namespace mlfo::optimizer

#endif // MLFO_OPTIMIZER_LOAD_HPP
