#include <iostream>
#include <print>

#include <mlfo>

int main() {
    std::vector<std::unique_ptr<mlfo::layer::Layer<float>>> layers;
    layers.push_back(std::make_unique<mlfo::layer::DenseLayer<float>>(3, 2));
    layers.push_back(std::make_unique<mlfo::layer::activation::ReLUActivation<float>>());
    layers.push_back(std::make_unique<mlfo::layer::DenseLayer<float>>(2, 4));
    layers.push_back(std::make_unique<mlfo::layer::activation::SigmoidActivation<float>>());
    layers.push_back(std::make_unique<mlfo::layer::DenseLayer<float>>(4, 4));
    auto loss = std::make_unique<mlfo::layer::loss::MSELoss<float>>(4);
    auto model = mlfo::model::Model<float>(std::move(layers), std::move(loss));
    model.build();
    auto optim = mlfo::optimizer::SGD<float>(model, 0.01f);
    const std::vector<float> x = {1, 2, 3};
    const std::vector<float> y = {0, 1, 0, 0};
    model.set_inputs({x});
    model.set_labels({y});
    model.forward();
    model.backward();
    optim.step();
    for (const auto& output : model.outputs()) {
        std::println("{}", output.get().to_string());
    }
    return 0;
}
