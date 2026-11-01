#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <sstream>
#include <vector>

#include "helpers.hpp"

namespace {

using Tensor = mlfo::tensor::Tensor<float>;
using Graph = mlfo::graph::Graph<Tensor>;
using Node = mlfo::graph::Node<Tensor>;
using Access = mlfo::tests::TensorAccess<float>;
using Shape = std::vector<std::size_t>;
using TanhActivation = mlfo::layer::activation::TanhActivation<float>;

void expect_floats(const std::vector<float>& actual, const std::vector<float>& expected) {
    ASSERT_EQ(actual.size(), expected.size());
    for (std::size_t i = 0; i < expected.size(); i++) {
        EXPECT_FLOAT_EQ(actual[i], expected[i]) << "index " << i;
    }
}

float tanh_grad(float x) {
    return 1.0f - std::tanh(x) * std::tanh(x);
}

} // namespace

TEST(TanhActivation, HasNoParameters) {
    TanhActivation layer;
    EXPECT_TRUE(layer.parameters().empty());
}

TEST(TanhActivation, BuildAddsOneNode) {
    Graph graph;
    Node& input = graph.add_node(Tensor(1, Shape{2, 3}));
    TanhActivation layer;
    Node& output = layer.build(graph, input);
    EXPECT_EQ(graph.size(), 2u);
    ASSERT_EQ(output.predecessors().size(), 1u);
    EXPECT_EQ(&output.predecessors()[0].get(), &input);
}

TEST(TanhActivation, BuildMatchesInputShape) {
    Graph graph;
    Node& input = graph.add_node(Tensor(4, Shape{2, 3}));
    TanhActivation layer;
    Node& output = layer.build(graph, input);
    EXPECT_EQ(output.data().shape(), (Shape{2, 3}));
    EXPECT_EQ(output.data().batch_size(), 4u);
}

TEST(TanhActivation, Forward) {
    Graph graph;
    Node& input = graph.add_node(Tensor(1, Shape{1, 4}));
    TanhActivation layer;
    Node& output = layer.build(graph, input);
    input.data().set_values({-1, 2, 0, -3.5f});
    graph.finalize();
    graph.forward();
    expect_floats(output.data().values(), {
        std::tanh(-1.0f), std::tanh(2.0f), 0.0f, std::tanh(-3.5f)
    });
}

TEST(TanhActivation, ForwardBatched) {
    Graph graph;
    Node& input = graph.add_node(Tensor(2, Shape{1, 3}));
    TanhActivation layer;
    Node& output = layer.build(graph, input);
    input.data().set_values({-1, 0, 2, 3, -4, 5});
    graph.finalize();
    graph.forward();
    expect_floats(output.data().values(), {
        std::tanh(-1.0f), 0.0f, std::tanh(2.0f), std::tanh(3.0f), std::tanh(-4.0f), std::tanh(5.0f)
    });
}

TEST(TanhActivation, Backward) {
    Graph graph;
    Node& input = graph.add_node(Tensor(1, Shape{1, 4}));
    TanhActivation layer;
    Node& output = layer.build(graph, input);
    input.data().set_values({-1, 2, 0, 3});
    graph.finalize();
    graph.forward();
    Access::gradients(output.data()) = {1, 2, 3, 4};
    graph.backward();
    // dIn = dOut * (1 - tanh(in)^2)
    expect_floats(input.data().gradients(), {
        1 * tanh_grad(-1), 2 * tanh_grad(2), 3 * 1.0f, 4 * tanh_grad(3)
    });
}

TEST(TanhActivation, ComposesWithDense) {
    Graph graph;
    Node& input = graph.add_node(Tensor(1, Shape{1, 2}));
    mlfo::layer::DenseLayer<float> dense(2);
    TanhActivation tanh_layer;
    Node& hidden = dense.build(graph, input);
    Node& output = tanh_layer.build(graph, hidden);
    // dense output -> prebias, bias; prebias -> input, weight
    Node& prebias = hidden.predecessors()[0].get();
    Node& bias = hidden.predecessors()[1].get();
    Node& weight = prebias.predecessors()[1].get();
    input.data().set_values({1, 2});
    weight.data().set_values({1, -1, 1, -1});
    bias.data().set_values({0, 1});
    graph.finalize();
    graph.forward();
    // hidden = [1 + 2, -1 - 2] + [0, 1] = [3, -2]; tanh -> [tanh(3), tanh(-2)]
    expect_floats(hidden.data().values(), {3, -2});
    expect_floats(output.data().values(), {std::tanh(3.0f), std::tanh(-2.0f)});
}

TEST(TanhActivation, SerializeWritesTypeOnly) {
    TanhActivation layer;
    std::ostringstream os;
    layer.serialize(os);
    EXPECT_EQ(os.str(), "TanhActivation\n");
    std::istringstream is("");
    EXPECT_NE(TanhActivation::deserialize(is), nullptr);
}
