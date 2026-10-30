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

using Tensor = mlfo::tensor::Tensor<float>;
using Node = mlfo::graph::Node<Tensor>;
using Shape = std::vector<std::size_t>;
using Layers = std::vector<std::unique_ptr<mlfo::layer::Layer<float>>>;

// dense layers with the given output sizes, the input sizes are inferred when built
Layers make_layers(const std::vector<std::size_t>& sizes) {
    Layers layers;
    for (const std::size_t size : sizes) {
        layers.push_back(std::make_unique<mlfo::layer::DenseLayer<float>>(size));
    }
    return layers;
}

// exposes the parameter nodes of a built two layer model
class TestModel : public mlfo::model::Model<float> {
public:
    using mlfo::model::Model<float>::Model;

    // sets the parameters of the last dense layer, and of the one before it
    // by walking the wiring back from the output node
    void set_parameters(
        const std::vector<float>& w1, const std::vector<float>& b1,
        const std::vector<float>& w2, const std::vector<float>& b2
    ) {
        Node& output = *outputs_.front();
        Node& prebias2 = output.predecessors()[0].get();
        Node& hidden = prebias2.predecessors()[0].get();
        hidden.predecessors()[1].get().data().set_values(b1);
        hidden.predecessors()[0].get().predecessors()[1].get().data().set_values(w1);
        output.predecessors()[1].get().data().set_values(b2);
        prebias2.predecessors()[1].get().data().set_values(w2);
    }
};

} // namespace

TEST(Model, BuildWithoutLayersThrows) {
    mlfo::model::Model<float> model({1, 3}, Layers{});
    EXPECT_THROW(model.build(), std::logic_error);
}

TEST(Model, EmptyInputShapeThrows) {
    EXPECT_THROW(mlfo::model::Model<float>({}, make_layers({2})), std::invalid_argument);
}

TEST(Model, BuildInfersLayerInputShapes) {
    Layers layers;
    layers.push_back(std::make_unique<mlfo::layer::Conv2DLayer<float>>(4, 3, 3));
    layers.push_back(std::make_unique<mlfo::layer::pool::MaxPool<float>>(2, 2));
    layers.push_back(std::make_unique<mlfo::layer::FlattenLayer<float>>());
    layers.push_back(std::make_unique<mlfo::layer::DenseLayer<float>>(5));
    std::vector<mlfo::layer::Layer<float>*> views;
    for (const auto& layer : layers) {
        views.push_back(layer.get());
    }
    mlfo::model::Model<float> model({2, 8, 8}, std::move(layers));
    model.build();
    // (2, 8, 8) -> conv (4, 6, 6) -> pool (4, 3, 3) -> flatten (1, 36) -> dense (1, 5)
    EXPECT_EQ(views[0]->input_shape(), (Shape{2, 8, 8}));
    EXPECT_EQ(views[1]->input_shape(), (Shape{4, 6, 6}));
    EXPECT_EQ(views[2]->input_shape(), (Shape{4, 3, 3}));
    EXPECT_EQ(views[3]->input_shape(), (Shape{1, 36}));
    EXPECT_EQ(model.outputs()[0].get().shape(), (Shape{1, 5}));
}

TEST(Model, BuildRejectsIncompatibleLayers) {
    // dense needs a (1, n) input, so a (C, H, W) input must be flattened first
    mlfo::model::Model<float> model({1, 4, 4}, make_layers({2}));
    EXPECT_THROW(model.build(), std::invalid_argument);
}

TEST(Model, SetInputsWrongCountThrows) {
    mlfo::model::Model<float> model({1, 3}, make_layers({2}));
    model.build();
    EXPECT_THROW(model.set_inputs({}), std::invalid_argument);
    const std::vector<float> x = {1, 2, 3};
    EXPECT_THROW(model.set_inputs({x, x}), std::invalid_argument);
}

TEST(Model, SetInputsWrongSizeThrows) {
    mlfo::model::Model<float> model({1, 3}, make_layers({2}));
    model.build();
    const std::vector<float> inputs = {1, 2};
    EXPECT_THROW(model.set_inputs({inputs}), std::invalid_argument);
}

TEST(Model, OutputShapeMatchesLastLayer) {
    Layers layers = make_layers({2, 4});
    const Shape expected = layers.back()->output_shape();
    mlfo::model::Model<float> model({1, 3}, std::move(layers));
    model.build();
    ASSERT_EQ(model.outputs().size(), 1u);
    EXPECT_EQ(model.outputs()[0].get().shape(), expected);
}

TEST(Model, ForwardWithDefaultParametersIsZero) {
    mlfo::model::Model<float> model({1, 3}, make_layers({2, 4}));
    model.build();
    const std::vector<float> inputs = {1, 2, 3};
    model.set_inputs({inputs});
    model.forward();
    const auto& values = model.outputs()[0].get().values();
    ASSERT_EQ(values.size(), 4u);
    for (float value : values) {
        EXPECT_FLOAT_EQ(value, 0.0f);
    }
}

TEST(Model, ForwardMatchesHandComputation) {
    TestModel model({1, 2}, make_layers({2, 1}));
    model.build();
    model.set_parameters({1, 0, 1, 1}, {0, 1}, {2, 3}, {1});
    const std::vector<float> inputs = {1, 2};
    model.set_inputs({inputs});
    model.forward();
    // hidden = [3, 3]; output = 2 * 3 + 3 * 3 + 1
    ASSERT_EQ(model.outputs()[0].get().values().size(), 1u);
    EXPECT_FLOAT_EQ(model.outputs()[0].get().values()[0], 16.0f);
}

TEST(Model, SetInputsRecomputesForward) {
    TestModel model({1, 2}, make_layers({2, 1}));
    model.build();
    model.set_parameters({1, 0, 1, 1}, {0, 1}, {2, 3}, {1});
    const std::vector<float> first = {1, 2};
    model.set_inputs({first});
    model.forward();
    EXPECT_FLOAT_EQ(model.outputs()[0].get().values()[0], 16.0f);
    // x = [0, 1]: hidden = [1, 1] + [0, 1] = [1, 2]; output = 2 + 6 + 1
    const std::vector<float> second = {0, 1};
    model.set_inputs({second});
    model.forward();
    EXPECT_FLOAT_EQ(model.outputs()[0].get().values()[0], 9.0f);
}

TEST(Model, ZeroBatchSizeThrows) {
    EXPECT_THROW(
        mlfo::model::Model<float>({1, 2}, make_layers({1}), nullptr, 0),
        std::invalid_argument
    );
}

TEST(Model, BatchedForwardMatchesSingleSamples) {
    TestModel model({1, 2}, make_layers({2, 1}), nullptr, 2);
    model.build();
    model.set_parameters({1, 0, 1, 1}, {0, 1}, {2, 3}, {1});
    // the two samples from ForwardMatchesHandComputation and SetInputsRecomputesForward
    const std::vector<float> inputs = {1, 2, 0, 1};
    model.set_inputs({inputs});
    model.forward();
    EXPECT_EQ(model.batch_size(), 2u);
    EXPECT_EQ(model.outputs()[0].get().values(), (std::vector<float>{16, 9}));
}

TEST(Model, PartialBatchShrinksAndGrows) {
    TestModel model({1, 2}, make_layers({2, 1}), nullptr, 3);
    model.build();
    model.set_parameters({1, 0, 1, 1}, {0, 1}, {2, 3}, {1});
    const auto& output = model.outputs()[0].get();
    const std::vector<float> full = {1, 2, 0, 1, 1, 2};
    model.set_inputs({full});
    model.forward();
    EXPECT_EQ(output.values(), (std::vector<float>{16, 9, 16}));
    // e.g. the last batch of an epoch
    const std::vector<float> partial = {0, 1};
    model.set_inputs({partial});
    model.forward();
    EXPECT_EQ(model.batch_size(), 1u);
    EXPECT_EQ(output.values(), (std::vector<float>{9}));
    model.set_inputs({full});
    model.forward();
    EXPECT_EQ(output.values(), (std::vector<float>{16, 9, 16}));
}

TEST(Model, SetInputsRejectsBadBatches) {
    mlfo::model::Model<float> model({1, 2}, make_layers({1}), nullptr, 2);
    model.build();
    // more samples than the model was built for
    const std::vector<float> too_many = {1, 2, 3, 4, 5, 6};
    EXPECT_THROW(model.set_inputs({too_many}), std::invalid_argument);
    // not a whole number of samples
    const std::vector<float> partial_sample = {1, 2, 3};
    EXPECT_THROW(model.set_inputs({partial_sample}), std::invalid_argument);
    const std::vector<float> empty;
    EXPECT_THROW(model.set_inputs({empty}), std::invalid_argument);
}

TEST(Model, SetLabelsMustMatchBatch) {
    mlfo::model::Model<float> model(
        {1, 2}, make_layers({1}), std::make_unique<mlfo::layer::loss::MSELoss<float>>(), 2
    );
    model.build();
    const std::vector<float> inputs = {1, 2};
    model.set_inputs({inputs});
    const std::vector<float> labels = {1, 2};
    EXPECT_THROW(model.set_labels({labels}), std::invalid_argument);
}

TEST(Model, BatchedBackwardAveragesGradients) {
    // the gradient of a batch is the mean of its samples' gradients
    const std::vector<std::vector<float>> xs = {{1, 2}, {0, 1}};
    const std::vector<std::vector<float>> ys = {{3}, {-2}};
    auto make_model = [](std::size_t batch_size) {
        auto model = std::make_unique<TestModel>(
            Shape{1, 2}, make_layers({2, 1}),
            std::make_unique<mlfo::layer::loss::MSELoss<float>>(), batch_size
        );
        model->build();
        model->set_parameters({1, 0, 1, 1}, {0, 1}, {2, 3}, {1});
        for (const auto& param_ref : model->parameters()) {
            param_ref.get().zero_grad();
        }
        return model;
    };

    // one sample at a time, accumulating the parameter gradients
    auto single = make_model(1);
    for (std::size_t i = 0; i < xs.size(); i++) {
        single->set_inputs({xs[i]});
        single->set_labels({ys[i]});
        single->forward();
        single->backward();
    }

    auto batched = make_model(2);
    const std::vector<float> x = {1, 2, 0, 1};
    const std::vector<float> y = {3, -2};
    batched->set_inputs({x});
    batched->set_labels({y});
    batched->forward();
    batched->backward();

    ASSERT_EQ(single->parameters().size(), batched->parameters().size());
    for (std::size_t p = 0; p < single->parameters().size(); p++) {
        const auto& summed = single->parameters()[p].get().gradients();
        const auto& averaged = batched->parameters()[p].get().gradients();
        ASSERT_EQ(summed.size(), averaged.size());
        for (std::size_t i = 0; i < summed.size(); i++) {
            EXPECT_FLOAT_EQ(averaged[i], summed[i] / 2) << "parameter " << p << " index " << i;
        }
    }
}

TEST(Model, SerializeBeforeBuildThrows) {
    mlfo::model::Model<float> model({1, 3}, make_layers({2}));
    std::ostringstream os;
    EXPECT_THROW(model.serialize(os), std::logic_error);
}

TEST(Model, SerializeWritesNameShapeLayersAndLoss) {
    Layers layers;
    layers.push_back(std::make_unique<mlfo::layer::Conv2DLayer<float>>(1, 2, 2));
    layers.push_back(std::make_unique<mlfo::layer::pool::MaxPool<float>>(1, 1));
    layers.push_back(std::make_unique<mlfo::layer::activation::ReLUActivation<float>>());
    layers.push_back(std::make_unique<mlfo::layer::FlattenLayer<float>>());
    layers.push_back(std::make_unique<mlfo::layer::DenseLayer<float>>(2));
    layers.push_back(std::make_unique<mlfo::layer::activation::SigmoidActivation<float>>());
    mlfo::model::Model<float> model(
        {1, 2, 2},
        std::move(layers),
        std::make_unique<mlfo::layer::loss::CrossEntropyLoss<float>>()
    );
    model.name("net");
    model.build();
    // conv weight, conv bias, dense weight, dense bias
    const auto& params = model.parameters();
    params[0].get().set_values({0.5f, -1, 2, 0.25f});
    params[1].get().set_values({3});
    params[2].get().set_values({1, 2});
    params[3].get().set_values({-0.5f, 0.1f});

    std::ostringstream os;
    model.serialize(os);
    EXPECT_EQ(
        os.str(),
        "net\n"
        "1,2,2\n"
        "Conv2DLayer\n"
        "1,1,2,2,1,1,0,0\n"
        "0.5,-1,2,0.25\n"
        "3\n"
        "MaxPool\n"
        "1,1,1,1\n"
        "ReLUActivation\n"
        "FlattenLayer\n"
        "DenseLayer\n"
        "1,2\n"
        "1,2\n"
        // written with enough digits to read back the exact float
        "-0.5,0.100000001\n"
        "SigmoidActivation\n"
        "CrossEntropyLoss\n"
    );
}

TEST(Model, SerializeRoundTripsFloats) {
    mlfo::model::Model<float> model({1, 1}, make_layers({1}));
    model.build();
    const float value = 1.0f / 3.0f;
    model.parameters()[0].get().set_values({value});
    std::ostringstream os;
    model.serialize(os);
    // name, shape, layer type, sizes, then the weight
    std::istringstream is(os.str());
    std::string line;
    for (int i = 0; i < 5; i++) {
        std::getline(is, line);
    }
    EXPECT_EQ(std::stof(line), value);
}

namespace {

// a model using every layer type, with distinct parameter values so a round trip can be checked
std::unique_ptr<mlfo::model::Model<float>> make_serializable_model() {
    Layers layers;
    layers.push_back(std::make_unique<mlfo::layer::Conv2DLayer<float>>(2, 2, 2, 1, 1, 1, 1));
    layers.push_back(std::make_unique<mlfo::layer::pool::MaxPool<float>>(2, 2, 1, 1));
    layers.push_back(std::make_unique<mlfo::layer::activation::ReLUActivation<float>>());
    layers.push_back(std::make_unique<mlfo::layer::FlattenLayer<float>>());
    layers.push_back(std::make_unique<mlfo::layer::DenseLayer<float>>(3));
    layers.push_back(std::make_unique<mlfo::layer::activation::SigmoidActivation<float>>());
    auto model = std::make_unique<mlfo::model::Model<float>>(
        Shape{1, 3, 3},
        std::move(layers),
        std::make_unique<mlfo::layer::loss::MSELoss<float>>()
    );
    model->name("round trip");
    model->build();
    float value = 0.1f;
    for (const auto& param : model->parameters()) {
        std::vector<float> values(param.get().size());
        for (float& v : values) {
            v = value;
            value = -value * 1.37f;
        }
        param.get().set_values(values);
    }
    return model;
}

std::string serialized(const mlfo::model::Model<float>& model) {
    std::ostringstream os;
    model.serialize(os);
    return os.str();
}

mlfo::model::Model<float> deserialized(const std::string& text) {
    std::istringstream is(text);
    return mlfo::model::Model<float>::deserialize(is);
}

} // namespace

TEST(Model, DeserializeRoundTrips) {
    const auto original = make_serializable_model();
    const std::string text = serialized(*original);
    auto loaded = deserialized(text);
    EXPECT_EQ(loaded.name(), "round trip");
    EXPECT_EQ(serialized(loaded), text);

    // the loaded model computes the same outputs, including the loss
    const std::vector<float> x = {1, -2, 3, 0.5f, 0, -1, 2, 4, -3};
    const std::vector<float> y = {0, 1, 0.5f};
    original->set_inputs({x});
    original->set_labels({y});
    original->forward();
    loaded.set_inputs({x});
    loaded.set_labels({y});
    loaded.forward();
    ASSERT_EQ(loaded.outputs().size(), original->outputs().size());
    for (std::size_t i = 0; i < loaded.outputs().size(); i++) {
        EXPECT_EQ(loaded.outputs()[i].get().values(), original->outputs()[i].get().values());
    }
}

TEST(Model, DeserializeUsesGivenBatchSize) {
    const auto original = make_serializable_model();
    std::istringstream is(serialized(*original));
    const auto loaded = mlfo::model::Model<float>::deserialize(is, 4);
    EXPECT_EQ(loaded.max_batch_size(), 4u);
}

TEST(Model, DeserializeAcceptsWindowsLineEndings) {
    const auto original = make_serializable_model();
    std::string text;
    for (const char c : serialized(*original)) {
        if (c == '\n') {
            text += '\r';
        }
        text += c;
    }
    EXPECT_EQ(serialized(deserialized(text)), serialized(*original));
}

TEST(Model, DeserializeRejectsMalformedInput) {
    // unknown layer type
    EXPECT_THROW(deserialized("m\n1,2\nFooLayer\n"), std::runtime_error);
    // truncated parameters
    EXPECT_THROW(deserialized("m\n1,2\nDenseLayer\n2,1\n0.5,0.5\n"), std::runtime_error);
    // wrong number of weights for the sizes
    EXPECT_THROW(deserialized("m\n1,2\nDenseLayer\n2,1\n0.5\n0\n"), std::runtime_error);
    // non numeric value
    EXPECT_THROW(deserialized("m\n1,2\nDenseLayer\n2,1\n0.5,abc\n0\n"), std::runtime_error);
    // layer after the loss
    EXPECT_THROW(deserialized("m\n1,1\nMSELoss\nFlattenLayer\n"), std::runtime_error);
    // missing input shape
    EXPECT_THROW(deserialized("m\n"), std::runtime_error);
    // saved sizes do not match the input shape, caught when the parameters are restored
    EXPECT_THROW(deserialized("m\n1,3\nDenseLayer\n2,1\n0.5,0.5\n0\n"), std::invalid_argument);
}

TEST(Model, SaveAndLoadFile) {
    const auto original = make_serializable_model();
    const auto path = std::filesystem::temp_directory_path() / "mlfo_model_test_save_load.txt";
    original->save(path);
    const auto loaded = mlfo::model::Model<float>::load(path);
    std::filesystem::remove(path);
    EXPECT_EQ(serialized(loaded), serialized(*original));
}

TEST(Model, LoadMissingFileThrows) {
    const auto path = std::filesystem::temp_directory_path() / "mlfo_model_test_missing.txt";
    std::filesystem::remove(path);
    EXPECT_THROW(mlfo::model::Model<float>::load(path), std::runtime_error);
}
