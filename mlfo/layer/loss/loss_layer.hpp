#ifndef LOSS_LAYER_HPP
#define LOSS_LAYER_HPP

#include <vector>
#include <memory>
#include <cstddef>

#include "../../misc/concepts.hpp"
#include "../../layer/layer.hpp"
#include "../../graph/graph.hpp"

namespace mlfo::layer::loss {

template <mlfo::misc::Number T>
class LossLayer : public mlfo::layer::Layer<T> {
public:
};

} // namespace mlfo::layer::loss

#endif // LOSS_LAYER_HPP
