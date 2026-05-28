#include "layer.hpp"
#include "linear_layer.hpp"
#include "relu_layer.hpp"
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
        {LinearLayer::factory, ReLULayer::factory, LinearLayer::factory, ReLULayer::factory, LinearLayer::factory},
        {{{8}}, {{9}}, {{9}}, {{6}}, {{6}}},
        {{{9}}, {{9}}, {{6}}, {{6}}, {{6}}},
        MSELoss::factory
    );
    SGDOptimizer optimizer(model.parameters(), 0.03f);
    for (std::size_t epoch = 1; epoch <= 1000; epoch++) {
        model.set_inputs({{
            1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f,
            2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f,
            3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f,
            4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f
        }}, 4);
        model.forward();
        model.set_targets({{
            1.0f, 2.0f, 3.0f, 3.0f, 2.0f, 1.0f,
            8.0f, 7.0f, 6.0f, 1.0f, 2.0f, 5.0f,
            -1.0f, 0.0f, 1.0f, 0.0f, 2.0f, 1.0f,
            4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f
        }}, 4);
        model.loss();
        model.backward();
        optimizer.step();
        model.zero_grad();
    }
    std::println("Final outputs after training:");
    model.forward();
    model.loss();
    model.backward();
    for (const auto &output : model.outputs()) {
        std::println("Outputs: {}", output->to_string());
    }
    std::println("Printing parameters");
    const std::vector<Tensor*>& params = model.parameters();
    for (const auto& param : params) {
        std::println("Param: {}", param->to_string());
    }
    return 0;
}
