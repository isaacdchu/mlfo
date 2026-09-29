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

Layers make_layers(const std::vector<std::pair<std::size_t, std::size_t>>& sizes) {
    Layers layers;
    for (const auto& [in, out] : sizes) {
        layers.push_back(std::make_unique<mlfo::layer::DenseLayer<float>>(in, out));
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
        std::vector<float> w1, std::vector<float> b1,
        std::vector<float> w2, std::vector<float> b2
    ) {
        Node& output = *outputs_.front();
        Node& prebias2 = output.predecessors()[0].get();
        Node& hidden = prebias2.predecessors()[0].get();
        hidden.predecessors()[1].get().data().set_values(std::move(b1));
        hidden.predecessors()[0].get().predecessors()[1].get().data().set_values(std::move(w1));
        output.predecessors()[1].get().data().set_values(std::move(b2));
        prebias2.predecessors()[1].get().data().set_values(std::move(w2));
    }
};

} // namespace

TEST(Model, BuildWithoutLayersThrows) {
    mlfo::model::Model<float> model(Layers{});
    EXPECT_THROW(model.build(), std::logic_error);
}

TEST(Model, SetInputsWrongCountThrows) {
    mlfo::model::Model<float> model(make_layers({{3, 2}}));
    model.build();
    std::vector<std::vector<float>> none;
    EXPECT_THROW(model.set_inputs(std::move(none)), std::invalid_argument);
    std::vector<std::vector<float>> two = {{1, 2, 3}, {1, 2, 3}};
    EXPECT_THROW(model.set_inputs(std::move(two)), std::invalid_argument);
}

TEST(Model, SetInputsWrongSizeThrows) {
    mlfo::model::Model<float> model(make_layers({{3, 2}}));
    model.build();
    std::vector<std::vector<float>> inputs = {{1, 2}};
    EXPECT_THROW(model.set_inputs(std::move(inputs)), std::invalid_argument);
}

TEST(Model, OutputShapeMatchesLastLayer) {
    Layers layers = make_layers({{3, 2}, {2, 4}});
    const Shape expected = layers.back()->output_shape();
    mlfo::model::Model<float> model(std::move(layers));
    model.build();
    ASSERT_EQ(model.outputs().size(), 1u);
    EXPECT_EQ(model.outputs()[0].get().shape(), expected);
}

TEST(Model, ForwardWithDefaultParametersIsZero) {
    mlfo::model::Model<float> model(make_layers({{3, 2}, {2, 4}}));
    model.build();
    std::vector<std::vector<float>> inputs = {{1, 2, 3}};
    model.set_inputs(std::move(inputs));
    model.forward();
    const auto& values = model.outputs()[0].get().values();
    ASSERT_EQ(values.size(), 4u);
    for (float value : values) {
        EXPECT_FLOAT_EQ(value, 0.0f);
    }
}

TEST(Model, ForwardMatchesHandComputation) {
    TestModel model(make_layers({{2, 2}, {2, 1}}));
    model.build();
    model.set_parameters({1, 0, 1, 1}, {0, 1}, {2, 3}, {1});
    std::vector<std::vector<float>> inputs = {{1, 2}};
    model.set_inputs(std::move(inputs));
    model.forward();
    // hidden = [3, 3]; output = 2 * 3 + 3 * 3 + 1
    ASSERT_EQ(model.outputs()[0].get().values().size(), 1u);
    EXPECT_FLOAT_EQ(model.outputs()[0].get().values()[0], 16.0f);
}

TEST(Model, SetInputsRecomputesForward) {
    TestModel model(make_layers({{2, 2}, {2, 1}}));
    model.build();
    model.set_parameters({1, 0, 1, 1}, {0, 1}, {2, 3}, {1});
    std::vector<std::vector<float>> first = {{1, 2}};
    model.set_inputs(std::move(first));
    model.forward();
    EXPECT_FLOAT_EQ(model.outputs()[0].get().values()[0], 16.0f);
    // x = [0, 1]: hidden = [1, 1] + [0, 1] = [1, 2]; output = 2 + 6 + 1
    std::vector<std::vector<float>> second = {{0, 1}};
    model.set_inputs(std::move(second));
    model.forward();
    EXPECT_FLOAT_EQ(model.outputs()[0].get().values()[0], 9.0f);
}
