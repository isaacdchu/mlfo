#include <gtest/gtest.h>

#include <cstddef>
#include <vector>

#include "helpers.hpp"

namespace {

using Tensor = mlfo::tensor::Tensor<float>;
using Graph = mlfo::graph::Graph<Tensor>;
using Node = mlfo::graph::Node<Tensor>;
using Access = mlfo::tests::TensorAccess<float>;
using Shape = std::vector<std::size_t>;
using ReLUActivation = mlfo::layer::activation::ReLUActivation<float>;

void expect_floats(const std::vector<float>& actual, const std::vector<float>& expected) {
    ASSERT_EQ(actual.size(), expected.size());
    for (std::size_t i = 0; i < expected.size(); i++) {
        EXPECT_FLOAT_EQ(actual[i], expected[i]) << "index " << i;
    }
}

} // namespace

TEST(ReLUActivation, HasNoParameters) {
    ReLUActivation layer;
    EXPECT_TRUE(layer.parameters().empty());
}

TEST(ReLUActivation, BuildAddsOneNode) {
    Graph graph;
    Node& input = graph.add_node(Tensor(1, Shape{2, 3}));
    ReLUActivation layer;
    Node& output = layer.build(graph, input);
    EXPECT_EQ(graph.size(), 2u);
    ASSERT_EQ(output.predecessors().size(), 1u);
    EXPECT_EQ(&output.predecessors()[0].get(), &input);
}

TEST(ReLUActivation, BuildMatchesInputShape) {
    Graph graph;
    Node& input = graph.add_node(Tensor(4, Shape{2, 3}));
    ReLUActivation layer;
    Node& output = layer.build(graph, input);
    EXPECT_EQ(output.data().shape(), (Shape{2, 3}));
    EXPECT_EQ(output.data().batch_size(), 4u);
}

TEST(ReLUActivation, ForwardZeroesNegatives) {
    Graph graph;
    Node& input = graph.add_node(Tensor(1, Shape{1, 4}));
    ReLUActivation layer;
    Node& output = layer.build(graph, input);
    input.data().set_values({-1, 2, 0, -3.5f});
    graph.finalize();
    graph.forward();
    expect_floats(output.data().values(), {0, 2, 0, 0});
}

TEST(ReLUActivation, ForwardBatched) {
    Graph graph;
    Node& input = graph.add_node(Tensor(2, Shape{1, 3}));
    ReLUActivation layer;
    Node& output = layer.build(graph, input);
    input.data().set_values({-1, 0, 2, 3, -4, 5});
    graph.finalize();
    graph.forward();
    expect_floats(output.data().values(), {0, 0, 2, 3, 0, 5});
}

TEST(ReLUActivation, BackwardMasksNonPositiveInputs) {
    Graph graph;
    Node& input = graph.add_node(Tensor(1, Shape{1, 4}));
    ReLUActivation layer;
    Node& output = layer.build(graph, input);
    input.data().set_values({-1, 2, 0, 3});
    graph.finalize();
    graph.forward();
    Access::gradients(output.data()) = {1, 2, 3, 4};
    graph.backward();
    // dIn == dOut where input > 0, else 0
    expect_floats(input.data().gradients(), {0, 2, 0, 4});
}

TEST(ReLUActivation, ComposesWithDense) {
    Graph graph;
    Node& input = graph.add_node(Tensor(1, Shape{1, 2}));
    mlfo::layer::DenseLayer<float> dense(2);
    ReLUActivation relu;
    Node& hidden = dense.build(graph, input);
    Node& output = relu.build(graph, hidden);
    // dense output -> prebias, bias; prebias -> input, weight
    Node& prebias = hidden.predecessors()[0].get();
    Node& bias = hidden.predecessors()[1].get();
    Node& weight = prebias.predecessors()[1].get();
    input.data().set_values({1, 2});
    weight.data().set_values({1, -1, 1, -1});
    bias.data().set_values({0, 1});
    graph.finalize();
    graph.forward();
    // hidden = [1 + 2, -1 - 2] + [0, 1] = [3, -2]; relu -> [3, 0]
    expect_floats(hidden.data().values(), {3, -2});
    expect_floats(output.data().values(), {3, 0});
}
