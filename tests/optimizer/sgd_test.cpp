#include <gtest/gtest.h>

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
using SGD = mlfo::optimizer::SGD<float>;

// a single 1 -> 1 dense layer followed by an mse loss
Model make_model() {
    Layers layers;
    layers.push_back(std::make_unique<mlfo::layer::DenseLayer<float>>(1, 1));
    Model model(std::move(layers), std::make_unique<mlfo::layer::loss::MSELoss<float>>(1));
    model.build();
    return model;
}

float loss(const Model& model) {
    return model.outputs().back().get().values()[0];
}

} // namespace

TEST(SGD, NonPositiveLearningRateThrows) {
    Model model = make_model();
    EXPECT_THROW(SGD(model, 0.0f), std::invalid_argument);
    EXPECT_THROW(SGD(model, -0.1f), std::invalid_argument);
}

TEST(SGD, StepSubtractsScaledGradients) {
    Model model = make_model();
    SGD optim(model, 0.5f);
    for (auto& param_ref : model.parameters()) {
        param_ref.get().set_values({1});
        Access::gradients(param_ref.get()) = {4};
    }
    optim.step();
    for (const auto& param_ref : model.parameters()) {
        // 1 - 0.5 * 4
        EXPECT_FLOAT_EQ(param_ref.get().values()[0], -1.0f);
    }
}

TEST(SGD, ZeroGradClearsGradients) {
    Model model = make_model();
    SGD optim(model, 0.1f);
    for (auto& param_ref : model.parameters()) {
        Access::gradients(param_ref.get()) = {4};
    }
    optim.zero_grad();
    for (const auto& param_ref : model.parameters()) {
        EXPECT_FLOAT_EQ(param_ref.get().gradients()[0], 0.0f);
    }
}

TEST(SGD, StepInvalidatesModel) {
    Model model = make_model();
    SGD optim(model, 0.1f);
    model.set_inputs({{2}});
    model.set_labels({{1}});
    model.forward();
    // output = 0; loss = 0.5 * 1
    EXPECT_FLOAT_EQ(loss(model), 0.5f);
    model.backward();
    // dL/dout = -1, so dW = -2 and db = -1; W = 0.2 and b = 0.1
    optim.step();
    // forward on the same inputs must recompute: output = 0.5; loss = 0.5 * 0.25
    model.forward();
    EXPECT_FLOAT_EQ(loss(model), 0.125f);
}

TEST(SGD, GradientsAccumulateWithoutZeroGrad) {
    Model model = make_model();
    SGD optim(model, 0.1f);
    model.set_inputs({{2}});
    model.set_labels({{1}});
    for (int i = 0; i < 2; i++) {
        // invalidate so that backward() runs again on the same inputs
        model.invalidate();
        model.forward();
        model.backward();
    }
    // each backward(): dL/dout = -1, so dW = -2 and db = -1
    const auto& parameters = model.parameters();
    ASSERT_EQ(parameters.size(), 2u);
    EXPECT_FLOAT_EQ(parameters[0].get().gradients()[0], -4.0f);
    EXPECT_FLOAT_EQ(parameters[1].get().gradients()[0], -2.0f);
    // zero_grad() resets the accumulation
    optim.zero_grad();
    model.invalidate();
    model.forward();
    model.backward();
    EXPECT_FLOAT_EQ(parameters[0].get().gradients()[0], -2.0f);
    EXPECT_FLOAT_EQ(parameters[1].get().gradients()[0], -1.0f);
}

TEST(SGD, TrainingReducesLoss) {
    Model model = make_model();
    SGD optim(model, 0.1f);
    model.set_inputs({{2}});
    model.set_labels({{1}});
    model.forward();
    float previous = loss(model);
    for (int i = 0; i < 10; i++) {
        optim.zero_grad();
        model.backward();
        optim.step();
        model.forward();
        EXPECT_LT(loss(model), previous) << "iteration " << i;
        previous = loss(model);
    }
}
