#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

#include "helpers.hpp"

namespace {

using Access = mlfo::tests::TensorAccess<float>;
using Layers = std::vector<std::unique_ptr<mlfo::layer::Layer<float>>>;
using Model = mlfo::model::Model<float>;
using Adam = mlfo::optimizer::Adam<float>;

// a single 1 -> 1 dense layer followed by an mse loss
Model make_model() {
    Layers layers;
    layers.push_back(std::make_unique<mlfo::layer::DenseLayer<float>>(1));
    Model model({1, 1}, std::move(layers), std::make_unique<mlfo::layer::loss::MSELoss<float>>());
    model.build();
    return model;
}

float loss(const Model& model) {
    return model.outputs().back().get().values()[0];
}

} // namespace

TEST(Adam, InvalidHyperparametersThrow) {
    Model model = make_model();
    EXPECT_THROW(Adam(model, 0.0f), std::invalid_argument);
    EXPECT_THROW(Adam(model, -0.1f), std::invalid_argument);
    EXPECT_THROW(Adam(model, 0.1f, 1.0f), std::invalid_argument);
    EXPECT_THROW(Adam(model, 0.1f, -0.1f), std::invalid_argument);
    EXPECT_THROW(Adam(model, 0.1f, 0.9f, 1.0f), std::invalid_argument);
    EXPECT_THROW(Adam(model, 0.1f, 0.9f, 0.999f, 0.0f), std::invalid_argument);
}

TEST(Adam, FirstStepMovesByLearningRate) {
    Model model = make_model();
    Adam optim(model, 0.5f);
    for (auto& param_ref : model.parameters()) {
        param_ref.get().set_values({1});
        Access::gradients(param_ref.get()) = {4};
    }
    optim.step();
    for (const auto& param_ref : model.parameters()) {
        // bias correction makes m_hat = g and v_hat = g^2, so the step is lr * sign(g)
        EXPECT_NEAR(param_ref.get().values()[0], 0.5f, 1e-6f);
    }
}

TEST(Adam, MomentsMatchReference) {
    Model model = make_model();
    const float lr = 0.1f;
    const float beta1 = 0.9f;
    const float beta2 = 0.999f;
    const float eps = 1e-8f;
    Adam optim(model, lr, beta1, beta2, eps);
    const std::vector<float> grads = {4.0f, -2.0f, 1.0f};
    float expected = 1.0f;
    float m = 0.0f;
    float v = 0.0f;
    for (auto& param_ref : model.parameters()) {
        param_ref.get().set_values({1});
    }
    for (std::size_t t = 1; t <= grads.size(); t++) {
        const float g = grads[t - 1];
        for (auto& param_ref : model.parameters()) {
            Access::gradients(param_ref.get()) = {g};
        }
        optim.step();
        m = beta1 * m + (1 - beta1) * g;
        v = beta2 * v + (1 - beta2) * g * g;
        const float m_hat = m / (1 - std::pow(beta1, static_cast<float>(t)));
        const float v_hat = v / (1 - std::pow(beta2, static_cast<float>(t)));
        expected -= lr * m_hat / (std::sqrt(v_hat) + eps);
        for (const auto& param_ref : model.parameters()) {
            EXPECT_NEAR(param_ref.get().values()[0], expected, 1e-5f) << "step " << t;
        }
    }
}

TEST(Adam, StepInvalidatesModel) {
    Model model = make_model();
    Adam optim(model, 0.1f);
    const std::vector<float> x = {2};
    const std::vector<float> y = {1};
    model.set_inputs({x});
    model.set_labels({y});
    model.forward();
    // output = 0; loss = 0.5 * 1
    EXPECT_FLOAT_EQ(loss(model), 0.5f);
    model.backward();
    // both gradients are negative, so W = 0.1 and b = 0.1 after the first step
    optim.step();
    // output = 0.1 * 2 + 0.1 = 0.3; loss = 0.5 * 0.49
    model.forward();
    EXPECT_NEAR(loss(model), 0.245f, 1e-6f);
}

TEST(Adam, TrainingReducesLoss) {
    Model model = make_model();
    Adam optim(model, 0.05f);
    const std::vector<float> x = {2};
    const std::vector<float> y = {1};
    model.set_inputs({x});
    model.set_labels({y});
    model.forward();
    const float initial = loss(model);
    for (int i = 0; i < 100; i++) {
        optim.zero_grad();
        model.backward();
        optim.step();
        model.forward();
    }
    EXPECT_LT(loss(model), initial * 0.01f);
}
