#ifndef OPS_HPP
#define OPS_HPP

#include <vector>

#include "../misc/concepts.hpp"
#include "tensor.hpp"

namespace mlfo::tensor::ops {

enum class DIRECTION {
    FORWARD,
    BACKWARD
};

} // namespace mlfo::tensor::ops

#include "ops/add.hpp"

#endif // OPS_HPP
