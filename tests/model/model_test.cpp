#include <gtest/gtest.h>

#include <cstddef>
#include <memory>
#include <stdexcept>
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
