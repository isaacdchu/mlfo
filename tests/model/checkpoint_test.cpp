#include <gtest/gtest.h>

#include <cstddef>
#include <filesystem>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "helpers.hpp"

namespace {

using Model = mlfo::model::Model<float>;
using Optimizer = mlfo::optimizer::Optimizer<float>;
using Adam = mlfo::optimizer::Adam<float>;
using SGD = mlfo::optimizer::SGD<float>;
using Checkpoint = mlfo::model::Checkpoint<float>;
using Layers = std::vector<std::unique_ptr<mlfo::layer::Layer<float>>>;

// 3 -> 4 -> tanh -> 2 with an mse loss, with distinct non-zero parameters
std::unique_ptr<Model> make_model(std::size_t batch_size = 2) {
    Layers layers;
    layers.push_back(std::make_unique<mlfo::layer::DenseLayer<float>>(4));
    layers.push_back(std::make_unique<mlfo::layer::activation::TanhActivation<float>>());
    layers.push_back(std::make_unique<mlfo::layer::DenseLayer<float>>(2));
    auto model = std::make_unique<Model>(
        std::vector<std::size_t>{1, 3},
        std::move(layers),
        std::make_unique<mlfo::layer::loss::MSELoss<float>>(),
        batch_size
    );
    model->name("checkpointed");
    model->build();
    float value = 0.1f;
    for (const auto& param : model->parameters()) {
        std::vector<float> values(param.get().size());
        for (float& v : values) {
            v = value;
            value = -value * 1.13f;
            if (value > 1.0f || value < -1.0f) {
                value *= 0.1f;
            }
        }
        param.get().set_values(values);
    }
    return model;
}

const std::vector<float> x = {1, -2, 0.5f, 0.25f, 3, -1};
const std::vector<float> y = {0.5f, -0.5f, 1, 0};

void train(Model& model, Optimizer& optimizer, int steps) {
    model.set_inputs({x});
    model.set_labels({y});
    for (int i = 0; i < steps; i++) {
        optimizer.zero_grad();
        model.backward();
        optimizer.step();
    }
}

std::vector<std::vector<float>> parameter_values(const Model& model) {
    std::vector<std::vector<float>> values;
    for (const auto& param : model.parameters()) {
        values.push_back(param.get().values());
    }
    return values;
}

std::vector<std::vector<float>> parameter_gradients(const Model& model) {
    std::vector<std::vector<float>> gradients;
    for (const auto& param : model.parameters()) {
        gradients.push_back(param.get().gradients());
    }
    return gradients;
}

std::string serialized(const Model& model, const Optimizer& optimizer) {
    std::ostringstream os;
    mlfo::model::serialize_checkpoint(os, model, optimizer);
    return os.str();
}

Checkpoint round_trip(const Model& model, const Optimizer& optimizer, std::size_t batch_size = 2) {
    std::istringstream is(serialized(model, optimizer));
    return mlfo::model::deserialize_checkpoint<float>(is, batch_size);
}

} // namespace

TEST(Checkpoint, AdamResumesExactlyAfterRoundTrip) {
    // train 10 steps straight through
    auto reference = make_model();
    Adam reference_optim(*reference, 0.05f);
    train(*reference, reference_optim, 10);

    // train 4 steps, save, load, then train the other 6
    auto paused = make_model();
    Adam paused_optim(*paused, 0.05f);
    train(*paused, paused_optim, 4);
    Checkpoint resumed = round_trip(*paused, paused_optim);
    train(*resumed.model, *resumed.optimizer, 6);

    EXPECT_EQ(parameter_values(*resumed.model), parameter_values(*reference));
}

TEST(Checkpoint, SGDResumesExactlyAfterRoundTrip) {
    auto reference = make_model();
    SGD reference_optim(*reference, 0.1f);
    train(*reference, reference_optim, 8);

    auto paused = make_model();
    SGD paused_optim(*paused, 0.1f);
    train(*paused, paused_optim, 3);
    Checkpoint resumed = round_trip(*paused, paused_optim);
    train(*resumed.model, *resumed.optimizer, 5);

    EXPECT_EQ(parameter_values(*resumed.model), parameter_values(*reference));
}

TEST(Checkpoint, RoundTripIsStable) {
    auto model = make_model();
    Adam optim(*model, 0.05f);
    train(*model, optim, 3);
    const std::string text = serialized(*model, optim);
    Checkpoint loaded = round_trip(*model, optim);
    EXPECT_EQ(loaded.model->name(), "checkpointed");
    EXPECT_EQ(serialized(*loaded.model, *loaded.optimizer), text);
}

TEST(Checkpoint, AdamBeforeFirstStepRoundTrips) {
    // the moments are not allocated until the first step
    auto reference = make_model();
    Adam reference_optim(*reference, 0.05f);
    train(*reference, reference_optim, 3);

    auto fresh = make_model();
    Adam fresh_optim(*fresh, 0.05f);
    Checkpoint resumed = round_trip(*fresh, fresh_optim);
    train(*resumed.model, *resumed.optimizer, 3);

    EXPECT_EQ(parameter_values(*resumed.model), parameter_values(*reference));
}

TEST(Checkpoint, KeepsAccumulatedGradients) {
    auto model = make_model();
    SGD optim(*model, 0.1f);
    model->set_inputs({x});
    model->set_labels({y});
    optim.zero_grad();
    model->backward();
    Checkpoint loaded = round_trip(*model, optim);
    EXPECT_EQ(parameter_gradients(*loaded.model), parameter_gradients(*model));
}

TEST(Checkpoint, UsesGivenBatchSize) {
    auto model = make_model();
    SGD optim(*model, 0.1f);
    Checkpoint loaded = round_trip(*model, optim, 7);
    EXPECT_EQ(loaded.model->max_batch_size(), 7u);
}

TEST(Checkpoint, MovedCheckpointKeepsOptimizerAttached) {
    auto model = make_model();
    Adam optim(*model, 0.05f);
    Checkpoint loaded = round_trip(*model, optim);
    Checkpoint moved = std::move(loaded);
    EXPECT_EQ(&moved.optimizer->model(), moved.model.get());
}

TEST(Checkpoint, SaveAndLoadFile) {
    auto model = make_model();
    Adam optim(*model, 0.05f);
    train(*model, optim, 2);
    const auto path = std::filesystem::temp_directory_path() / "mlfo_checkpoint_test.txt";
    mlfo::model::save_checkpoint(path, *model, optim);
    Checkpoint loaded = mlfo::model::load_checkpoint<float>(path, 2);
    std::filesystem::remove(path);
    EXPECT_EQ(serialized(*loaded.model, *loaded.optimizer), serialized(*model, optim));
}

TEST(Checkpoint, LoadMissingFileThrows) {
    const auto path = std::filesystem::temp_directory_path() / "mlfo_checkpoint_missing.txt";
    std::filesystem::remove(path);
    EXPECT_THROW(mlfo::model::load_checkpoint<float>(path), std::runtime_error);
}

TEST(Checkpoint, RejectsOptimizerOfAnotherModel) {
    auto model = make_model();
    auto other = make_model();
    SGD optim(*other, 0.1f);
    std::ostringstream os;
    EXPECT_THROW(mlfo::model::serialize_checkpoint(os, *model, optim), std::invalid_argument);
    EXPECT_THROW(mlfo::model::clone_checkpoint(*model, optim), std::invalid_argument);
}

TEST(Checkpoint, RejectsMalformedInput) {
    auto model = make_model();
    Adam optim(*model, 0.05f);
    train(*model, optim, 1);
    const std::string text = serialized(*model, optim);
    const auto load = [](const std::string& input) {
        std::istringstream is(input);
        return mlfo::model::deserialize_checkpoint<float>(is);
    };
    // a plain model file is not a checkpoint
    std::ostringstream plain;
    model->serialize(plain);
    EXPECT_THROW(load(plain.str()), std::runtime_error);
    // truncated
    EXPECT_THROW(load(text.substr(0, text.size() / 2)), std::runtime_error);
    // unknown optimizer
    std::string unknown = text;
    unknown.replace(unknown.find("Adam\n"), 4, "Nope");
    EXPECT_THROW(load(unknown), std::runtime_error);
}

TEST(Checkpoint, CloneContinuesExactlyAndIndependently) {
    auto reference = make_model();
    Adam reference_optim(*reference, 0.05f);
    train(*reference, reference_optim, 10);

    auto original = make_model();
    Adam original_optim(*original, 0.05f);
    train(*original, original_optim, 4);
    const auto before = parameter_values(*original);
    const std::string before_text = serialized(*original, original_optim);

    Checkpoint copy = mlfo::model::clone_checkpoint(*original, original_optim);
    EXPECT_EQ(serialized(*copy.model, *copy.optimizer), before_text);
    EXPECT_EQ(copy.model->max_batch_size(), original->max_batch_size());
    train(*copy.model, *copy.optimizer, 6);

    EXPECT_EQ(parameter_values(*copy.model), parameter_values(*reference));
    // training the copy left the original and its optimizer untouched
    EXPECT_EQ(parameter_values(*original), before);
    EXPECT_EQ(serialized(*original, original_optim), before_text);
}

TEST(Checkpoint, CloneUsesGivenBatchSize) {
    auto model = make_model();
    SGD optim(*model, 0.1f);
    Checkpoint copy = mlfo::model::clone_checkpoint(*model, optim, 5);
    EXPECT_EQ(copy.model->max_batch_size(), 5u);
}

TEST(Checkpoint, AdamCloneRejectsDifferentArchitecture) {
    auto model = make_model();
    Adam optim(*model, 0.05f);
    train(*model, optim, 1);
    Layers layers;
    layers.push_back(std::make_unique<mlfo::layer::DenseLayer<float>>(3));
    Model other({1, 3}, std::move(layers), std::make_unique<mlfo::layer::loss::MSELoss<float>>());
    other.build();
    EXPECT_THROW(optim.clone(other), std::invalid_argument);
}
