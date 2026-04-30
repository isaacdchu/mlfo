#include "tensor.hpp"
#include "operations.hpp"
#include "pool.hpp"

#include <vector>
#include <memory>
#include <iostream>
#include <format>
#include <print>

int main() {
    Pool pool;
    auto a = pool.new_tensor({2, 2}, 0, 1.0f);
    auto b = pool.new_tensor({2, 2}, 0, 2.0f);
    auto c = pool.new_tensor({2, 2}, 0, 3.0f);
    auto d = Operations::add(
        Operations::add(a, b), // 3s
        Operations::add(b, c)  // 5s
    );
    d->forward();
    std::println("After forward:");
    std::println("a: {}", a->to_string());
    std::println("b: {}", b->to_string());
    std::println("c: {}", c->to_string());
    std::println("d: {}", d->to_string());
    d->set_gradients(std::vector<float>(d->size(), 1.0f));
    d->backward();
    std::println("After backward:");
    std::println("a: {}", a->to_string());
    std::println("b: {}", b->to_string());
    std::println("c: {}", c->to_string());
    std::println("d: {}", d->to_string());
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
