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
