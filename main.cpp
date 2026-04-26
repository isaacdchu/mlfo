#include "tensor.hpp"
#include "operations.hpp"
#include "util.hpp"

#include <vector>
#include <memory>
#include <iostream>

int main() {
    std::vector<std::unique_ptr<Tensor>> inputs;
    inputs.push_back(std::make_unique<Tensor>(std::vector<std::size_t>{2, 3}, 1.0f));
    inputs.push_back(std::make_unique<Tensor>(std::vector<std::size_t>{2, 3}, 2.0f));
    std::unique_ptr<Tensor> c = Operations::add(std::move(inputs));
    std::cout << c->to_string() << std::endl;
    /*
    Model model = Model({
        std::make_unique<DenseLayer>(2, 3),
        std::make_unique<ReLULayer>(),
        std::make_unique<DenseLayer>(3, 2)
    });
    model.set_loss(std::make_unique<MSELoss>());
    model.set_optimizer(std::make_unique<SGDOptimizer>(0.01f));
    for (int epoch = 0; epoch < 100; epoch++) {
        for (const auto& [input_batch, target_batch] : dataset.batches(32)) {
            auto predictions = model.forward(input_batch);
            auto loss = model.loss()->forward(predictions, target_batch);
            auto grad_output = model.loss()->backward(predictions, target_batch);
            model.backward(std::move(grad_output));
            model.optimizer()->step();
        }
    }
    */
    return 0;
}
