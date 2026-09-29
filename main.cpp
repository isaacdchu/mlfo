#include <print>
#include <iostream>

#include <mlfo>

int main() {
    std::vector<std::unique_ptr<mlfo::layer::Layer<float>>> layers;
    layers.push_back(std::make_unique<mlfo::layer::DenseLayer<float>>(3, 2));
    layers.push_back(std::make_unique<mlfo::layer::DenseLayer<float>>(2, 4));
    auto model = mlfo::model::Model<float>(std::move(layers));
    model.build();
    std::vector<std::vector<float>> inputs;
    inputs.push_back({1, 2, 3});
    model.set_inputs(std::move(inputs));
    model.forward();
    for (const auto& output : model.outputs()) {
        std::println("{}", output.get().to_string());
    }
    return 0;
}
