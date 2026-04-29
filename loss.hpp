#ifndef LOSS_HPP
#define LOSS_HPP

#include "tensor.hpp"

#include <memory>
#include <vector>

class Loss {
protected:
    const std::vector<std::vector<std::size_t>> input_unbatched_shapes_;
    const std::vector<std::vector<std::size_t>> output_unbatched_shapes_;
    std::vector<Tensor*> inputs_;
    std::unique_ptr<Tensor> output_;
public:
    Loss() = delete;

    Loss(
        const std::vector<std::vector<std::size_t>>& input_unbatched_shapes,
        const std::vector<std::size_t>& output_unbatched_shape,
        const std::vector<Tensor*>& inputs
    ) :
    inputs_(inputs),
    input_unbatched_shapes_(input_unbatched_shapes),
    output_unbatched_shapes_(std::vector<std::vector<std::size_t>>(1, output_unbatched_shape)) {
        output_ = std::make_unique<Tensor>(output_unbatched_shape, 1);
    }

    virtual const std::vector<std::vector<std::size_t>>& input_unbatched_shapes() const final {
        return input_unbatched_shapes_;
    }

    virtual const std::vector<std::vector<std::size_t>>& output_unbatched_shapes() const final {
        return output_unbatched_shapes_;
    }

    virtual void forward() = 0;
    virtual void backward() = 0;
};

#endif // LOSS_HPP