#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <vector>

#include "helpers.hpp"

namespace {

using Tensor = mlfo::tensor::Tensor<float>;
using Graph = mlfo::graph::Graph<Tensor>;
using Node = mlfo::graph::Node<Tensor>;
using Access = mlfo::tests::TensorAccess<float>;
using Shape = std::vector<std::size_t>;
using SigmoidActivation = mlfo::layer::activation::SigmoidActivation<float>;

void expect_floats(const std::vector<float>& actual, const std::vector<float>& expected) {
    ASSERT_EQ(actual.size(), expected.size());
    for (std::size_t i = 0; i < expected.size(); i++) {
        EXPECT_FLOAT_EQ(actual[i], expected[i]) << "index " << i;
    }
}

float sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

float sigmoid_grad(float x) {
    return sigmoid(x) * (1.0f - sigmoid(x));
}

} // namespace

TEST(SigmoidActivation, HasNoParameters) {
    SigmoidActivation layer;
    EXPECT_TRUE(layer.parameters().empty());
}

TEST(SigmoidActivation, BuildAddsOneNode) {
    Graph graph;
    Node& input = graph.add_node(Tensor(1, Shape{2, 3}));
    SigmoidActivation layer;
    Node& output = layer.build(graph, input);
    EXPECT_EQ(graph.size(), 2u);
    ASSERT_EQ(output.predecessors().size(), 1u);
    EXPECT_EQ(&output.predecessors()[0].get(), &input);
}

TEST(SigmoidActivation, BuildMatchesInputShape) {
    Graph graph;
    Node& input = graph.add_node(Tensor(4, Shape{2, 3}));
    SigmoidActivation layer;
    Node& output = layer.build(graph, input);
    EXPECT_EQ(output.data().shape(), (Shape{2, 3}));
    EXPECT_EQ(output.data().batch_size(), 4u);
}

TEST(SigmoidActivation, Forward) {
    Graph graph;
    Node& input = graph.add_node(Tensor(1, Shape{1, 4}));
    SigmoidActivation layer;
    Node& output = layer.build(graph, input);
    input.data().set_values({-1, 2, 0, -3.5f});
    graph.finalize();
    graph.forward();
    expect_floats(output.data().values(), {sigmoid(-1), sigmoid(2), 0.5f, sigmoid(-3.5f)});
}

TEST(SigmoidActivation, ForwardBatched) {
    Graph graph;
    Node& input = graph.add_node(Tensor(2, Shape{1, 3}));
    SigmoidActivation layer;
    Node& output = layer.build(graph, input);
    input.data().set_values({-1, 0, 2, 3, -4, 5});
    graph.finalize();
    graph.forward();
    expect_floats(output.data().values(), {
        sigmoid(-1), sigmoid(0), sigmoid(2), sigmoid(3), sigmoid(-4), sigmoid(5)
    });
}

TEST(SigmoidActivation, Backward) {
    Graph graph;
    Node& input = graph.add_node(Tensor(1, Shape{1, 4}));
    SigmoidActivation layer;
    Node& output = layer.build(graph, input);
    input.data().set_values({-1, 2, 0, 3});
    graph.finalize();
    graph.forward();
    Access::gradients(output.data()) = {1, 2, 3, 4};
    graph.backward();
    // dIn = dOut * sigmoid(in) * (1 - sigmoid(in))
    expect_floats(input.data().gradients(), {
        1 * sigmoid_grad(-1), 2 * sigmoid_grad(2), 3 * 0.25f, 4 * sigmoid_grad(3)
    });
}

TEST(SigmoidActivation, ComposesWithDense) {
    Graph graph;
    Node& input = graph.add_node(Tensor(1, Shape{1, 2}));
    mlfo::layer::DenseLayer<float> dense(2);
    SigmoidActivation sigmoid_layer;
    Node& hidden = dense.build(graph, input);
    Node& output = sigmoid_layer.build(graph, hidden);
    // dense output -> prebias, bias; prebias -> input, weight
    Node& prebias = hidden.predecessors()[0].get();
    Node& bias = hidden.predecessors()[1].get();
    Node& weight = prebias.predecessors()[1].get();
    input.data().set_values({1, 2});
    weight.data().set_values({1, -1, 1, -1});
    bias.data().set_values({0, 1});
    graph.finalize();
    graph.forward();
    // hidden = [1 + 2, -1 - 2] + [0, 1] = [3, -2]; sigmoid -> [s(3), s(-2)]
    expect_floats(hidden.data().values(), {3, -2});
    expect_floats(output.data().values(), {sigmoid(3), sigmoid(-2)});
}
