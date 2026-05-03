#include "layer.hpp"
#include "linear_layer.hpp"
#include "loss.hpp"
#include "model.hpp"
#include "mse_loss.hpp"
#include "operations.hpp"
#include "optimizer.hpp"
#include "pool.hpp"
#include "sgd_optimizer.hpp"
#include "tensor.hpp"

#include <format>
#include <iostream>
#include <memory>
#include <print>
#include <vector>

int main() {
    Model model(
        {LinearLayer::factory, LinearLayer::factory},
        {{{2, 4}}, {{3, 3}}},
        {{{3, 3}}, {{2, 3}}},
        MSELoss::factory
    );
    SGDOptimizer optimizer(model.parameters(), 0.03f);
    for (std::size_t epoch = 1; epoch <= 50000; epoch++) {
        model.set_inputs({{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f,
                           2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f,
                           3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f,
                           4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f}},
                         4);
        model.forward();
        model.set_targets({{0.5f, 1.0f, 1.5f, 2.0f, 2.5f, 3.0f,
                            1.0f, 1.5f, 2.0f, 2.5f, 3.0f, 3.5f,
                            1.5f, 2.0f, 2.5f, 3.0f, 3.5f, 4.0f,
                            2.0f, 2.5f, 3.0f, 3.5f, 4.0f, 4.5f}},
                          4);
        model.loss();
        model.backward();
        optimizer.step();
        model.zero_grad();
    }
    std::println("Final outputs after training:");
    model.forward();
    for (const auto &output : model.outputs()) {
        std::println("Outputs: {}", output->to_string());
    }
    // std::println("Printing parameters");
    // const std::vector<std::vector<Tensor*>>& params = model.parameters();
    // for (const auto& layer_params : params) {
    //     for (const auto& param : layer_params) {
    //         std::println("Param: {}", param->to_string());
    //     }
    // }
    // next step: optimizer.step();
    // std::println("Performing another forward pass after backward");
    // model.forward();
    // for (const auto &output : model.outputs()) {
    //     std::println("Outputs: {}", output->to_string());
    // }
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
