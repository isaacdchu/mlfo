#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <functional>
#include <memory>
#include <numeric>
#include <random>
#include <utility>
#include <vector>

#include "helpers.hpp"

// compares the gradients from backward() against central finite differences of the loss,
// for every parameter of an assembled model. runs in double so the numerical estimate is
// accurate enough to catch small errors in an op's backward pass

namespace {

using Layers = std::vector<std::unique_ptr<mlfo::layer::Layer<double>>>;

std::vector<double> random_values(std::mt19937& generator, std::size_t size) {
    std::uniform_real_distribution<double> distribution(-1.0, 1.0);
    std::vector<double> values(size);
    for (double& value : values) {
        value = distribution(generator);
    }
    return values;
}

double loss_at(mlfo::model::Model<double>& model) {
    // the second output is the loss, see Model::build()
    model.invalidate();
    model.forward();
    return model.outputs()[1].get().values()[0];
}

void check_gradients(
    Layers layers,
    const std::vector<std::size_t>& input_shape,
    std::size_t classes,
    std::unique_ptr<mlfo::layer::loss::LossLayer<double>> loss = nullptr
) {
    constexpr double epsilon = 1e-6;
    constexpr double tolerance = 1e-6;
    std::mt19937 generator(7);
    if (loss == nullptr) {
        loss = std::make_unique<mlfo::layer::loss::MSELoss<double>>();
    }
    mlfo::model::Model<double> model(input_shape, std::move(layers), std::move(loss));
    model.build();
    // random parameters, so that no unit is dead and max pool has no ties
    for (const auto& param_ref : model.parameters()) {
        param_ref.get().set_values(random_values(generator, param_ref.get().size()));
    }
    const std::vector<double> x = random_values(
        generator,
        std::accumulate(input_shape.begin(), input_shape.end(), std::size_t{1}, std::multiplies<>{})
    );
    const std::vector<double> y = random_values(generator, classes);
    model.set_inputs({x});
    model.set_labels({y});
    model.forward();
    for (const auto& param_ref : model.parameters()) {
        param_ref.get().zero_grad();
    }
    model.backward();

    for (std::size_t p = 0; p < model.parameters().size(); p++) {
        auto& param = model.parameters()[p].get();
        // copied, since set_values() below needs the whole buffer
        std::vector<double> values = param.values();
        const std::vector<double> analytic = param.gradients();
        for (std::size_t i = 0; i < values.size(); i++) {
            const double original = values[i];
            values[i] = original + epsilon;
            param.set_values(values);
            const double loss_plus = loss_at(model);
            values[i] = original - epsilon;
            param.set_values(values);
            const double loss_minus = loss_at(model);
            values[i] = original;
            param.set_values(values);
            const double numeric = (loss_plus - loss_minus) / (2 * epsilon);
            // relative error, falling back to absolute error near zero
            const double error = (
                std::abs(analytic[i] - numeric) /
                std::max(1.0, std::abs(analytic[i]) + std::abs(numeric))
            );
            EXPECT_LT(error, tolerance)
                << "parameter " << p << " index " << i
                << ": analytic " << analytic[i] << " numeric " << numeric;
        }
    }
}

} // namespace

TEST(GradientCheck, Dense) {
    Layers layers;
    layers.push_back(std::make_unique<mlfo::layer::DenseLayer<double>>(4));
    layers.push_back(std::make_unique<mlfo::layer::activation::SigmoidActivation<double>>());
    layers.push_back(std::make_unique<mlfo::layer::DenseLayer<double>>(3));
    check_gradients(std::move(layers), {1, 5}, 3);
}

TEST(GradientCheck, DenseCrossEntropy) {
    Layers layers;
    layers.push_back(std::make_unique<mlfo::layer::DenseLayer<double>>(4));
    layers.push_back(std::make_unique<mlfo::layer::activation::SigmoidActivation<double>>());
    layers.push_back(std::make_unique<mlfo::layer::DenseLayer<double>>(3));
    check_gradients(
        std::move(layers), {1, 5}, 3,
        std::make_unique<mlfo::layer::loss::CrossEntropyLoss<double>>()
    );
}

TEST(GradientCheck, Conv2D) {
    Layers layers;
    layers.push_back(std::make_unique<mlfo::layer::Conv2DLayer<double>>(3, 3, 3));
    layers.push_back(std::make_unique<mlfo::layer::FlattenLayer<double>>());
    layers.push_back(std::make_unique<mlfo::layer::DenseLayer<double>>(2));
    check_gradients(std::move(layers), {2, 5, 5}, 2);
}

TEST(GradientCheck, Conv2DPaddingAndStride) {
    Layers layers;
    // (5 + 2 * 1 - 3) / 2 + 1 = 3
    layers.push_back(std::make_unique<mlfo::layer::Conv2DLayer<double>>(2, 3, 3, 2, 2, 1, 1));
    layers.push_back(std::make_unique<mlfo::layer::FlattenLayer<double>>());
    layers.push_back(std::make_unique<mlfo::layer::DenseLayer<double>>(2));
    check_gradients(std::move(layers), {2, 5, 5}, 2);
}

TEST(GradientCheck, Conv2DMaxPool) {
    Layers layers;
    layers.push_back(std::make_unique<mlfo::layer::Conv2DLayer<double>>(2, 3, 3, 1, 1, 1, 1));
    layers.push_back(std::make_unique<mlfo::layer::pool::MaxPool<double>>(2, 2));
    layers.push_back(std::make_unique<mlfo::layer::FlattenLayer<double>>());
    layers.push_back(std::make_unique<mlfo::layer::DenseLayer<double>>(2));
    check_gradients(std::move(layers), {1, 6, 6}, 2);
}

TEST(GradientCheck, LeNetLike) {
    // the same layer sequence as main.cpp, scaled down
    Layers layers;
    layers.push_back(std::make_unique<mlfo::layer::Conv2DLayer<double>>(2, 3, 3, 1, 1, 1, 1));
    layers.push_back(std::make_unique<mlfo::layer::activation::SigmoidActivation<double>>());
    layers.push_back(std::make_unique<mlfo::layer::pool::MaxPool<double>>(2, 2));
    layers.push_back(std::make_unique<mlfo::layer::Conv2DLayer<double>>(3, 3, 3));
    layers.push_back(std::make_unique<mlfo::layer::activation::SigmoidActivation<double>>());
    layers.push_back(std::make_unique<mlfo::layer::pool::MaxPool<double>>(2, 2));
    layers.push_back(std::make_unique<mlfo::layer::FlattenLayer<double>>());
    layers.push_back(std::make_unique<mlfo::layer::DenseLayer<double>>(4));
    layers.push_back(std::make_unique<mlfo::layer::activation::SigmoidActivation<double>>());
    layers.push_back(std::make_unique<mlfo::layer::DenseLayer<double>>(3));
    check_gradients(std::move(layers), {1, 8, 8}, 3);
}
