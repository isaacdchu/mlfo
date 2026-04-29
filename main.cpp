#include "tensor.hpp"
#include "operations.hpp"

#include <vector>
#include <memory>
#include <iostream>
#include <format>
#include <print>

int main() {
    auto a = Tensor::factory({4, 3, 2}, 0, 1.0f);
    auto b = Tensor::factory({2, 3}, 0, 2.0f);
    auto c = Operations::matmul(a.get(), b.get(), 1);
    c->forward();
    std::println("After forward:");
    std::println("a: {}", a->to_string());
    std::println("b: {}", b->to_string());
    std::println("c: {}", c->to_string());
    c->set_gradients(std::vector<float>(c->size(), 1.0f));
    c->backward();
    std::println("After backward:");
    std::println("a: {}", a->to_string());
    std::println("b: {}", b->to_string());
    std::println("c: {}", c->to_string());
    // std::vector<std::unique_ptr<Tensor>> inputs;
    // inputs.emplace_back(std::make_unique<Tensor>(std::vector<std::size_t>{5, 2, 3}, 0, 1.0f));
    // inputs.emplace_back(std::make_unique<Tensor>(std::vector<std::size_t>{3, 2, 4}, 2, 2.0f));
    // std::unique_ptr<Tensor> c = Operations::matmul(inputs[0].get(), inputs[1].get(), 2);
    // // c is 2, 5, 4
    // std::println("doing forward");
    // c->forward();
    // c->set_gradients(std::vector<float>(c->size(), 1.0f));
    // std::println("doing backward");
    // c->backward();
    // std::println("After forward and backward:");
    // std::println("Input 1: {}", inputs[0]->to_string());
    // std::println("Input 2: {}", inputs[1]->to_string());
    // std::println("Output: {}", c->to_string());
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
