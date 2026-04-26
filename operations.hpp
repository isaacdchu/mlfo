#ifndef OPERATIONS_HPP
#define OPERATIONS_HPP

#include "tensor.hpp"

#include <vector>
#include <memory>

class Operations {
public:
    Operations() = delete;
    
    static std::unique_ptr<Tensor> add(const std::vector<Tensor*>& inputs) {
        auto output = std::make_unique<Tensor>(inputs[0]->shape(), 0.0f);
        output->set_parents(inputs);
        output->forward_ = [&](){
            for (const auto& tensor : output->parents()) {
                for (std::size_t i = 0; i < tensor->size(); i++) {
                    output->values_[i] += tensor->values_[i];
                }
            }
        };
        output->backward_ = [&](){
            // TODO
        };
        return output;
    }
};

#endif // OPERATIONS_HPP