#include "tensor.hpp"
#include "operations.hpp"
#include "pool.hpp"
#include "model.hpp"
#include "layer.hpp"
#include "linear_layer.hpp"
#include "loss.hpp"
#include "mse_loss.hpp"

#include <vector>
#include <memory>
#include <iostream>
#include <format>
#include <print>

int main() {
    std::println("Making model");
    Model model = Model(
        {LinearLayer::factory},
        {{{2, 4}}},
        {{{2, 3}}},
        MSELoss::factory
    );
    std::println("Setting inputs and running forward pass");
    model.set_inputs({{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f}}, 1);
    std::println("Performing forward pass");
    model.forward();
    for (const auto& output : model.outputs()) {
        std::println("Outputs: {}", output->to_string());
    }
    for (auto& output : model.outputs()) {
        output->set_gradients(std::vector<float>(output->size(), 1.0f));
    }
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
