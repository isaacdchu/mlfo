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

// the nodes of a built dense layer, reached through the wiring
struct DenseNodes {
    Node* output;
    Node* prebias;
    Node* bias;
    Node* weight;
};

DenseNodes find_nodes(Node& output, Node& input) {
    DenseNodes nodes;
    nodes.output = &output;
    nodes.prebias = &output.predecessors()[0].get();
    nodes.bias = &output.predecessors()[1].get();
    nodes.weight = &nodes.prebias->predecessors()[1].get();
    EXPECT_EQ(&nodes.prebias->predecessors()[0].get(), &input);
    return nodes;
}

} // namespace

TEST(DenseLayer, Shapes) {
    mlfo::layer::DenseLayer<float> layer(3, 2);
    EXPECT_EQ(layer.input_shape(), (Shape{1, 3}));
    EXPECT_EQ(layer.output_shape(), (Shape{1, 2}));
}

TEST(DenseLayer, BuildAddsFourNodes) {
    Graph graph;
    Node& input = graph.add_node(Tensor(1, Shape{1, 3}));
    mlfo::layer::DenseLayer<float> layer(3, 2);
    Node& output = layer.build(graph, input);
    EXPECT_EQ(graph.size(), 5u);
    EXPECT_EQ(output.data().shape(), (Shape{1, 2}));
    DenseNodes nodes = find_nodes(output, input);
    EXPECT_EQ(nodes.weight->data().shape(), (Shape{3, 2}));
    EXPECT_EQ(nodes.bias->data().shape(), (Shape{1, 2}));
    EXPECT_EQ(nodes.prebias->data().shape(), (Shape{1, 2}));
}

TEST(DenseLayer, ForwardComputesXWPlusB) {
    Graph graph;
    Node& input = graph.add_node(Tensor(1, Shape{1, 2}));
    mlfo::layer::DenseLayer<float> layer(2, 3);
    Node& output = layer.build(graph, input);
    DenseNodes nodes = find_nodes(output, input);
    input.data().set_values({1, 2});
    nodes.weight->data().set_values({1, 2, 3, 4, 5, 6});
    nodes.bias->data().set_values({0.5f, -1, 2});
    graph.finalize();
    graph.forward();
    ASSERT_EQ(output.data().values().size(), 3u);
    EXPECT_FLOAT_EQ(output.data().values()[0], 9.5f);
    EXPECT_FLOAT_EQ(output.data().values()[1], 11.0f);
    EXPECT_FLOAT_EQ(output.data().values()[2], 17.0f);
}

TEST(DenseLayer, DefaultParametersGiveZeroOutput) {
    Graph graph;
    Node& input = graph.add_node(Tensor(1, Shape{1, 2}));
    mlfo::layer::DenseLayer<float> layer(2, 3);
    Node& output = layer.build(graph, input);
    input.data().set_values({4, -7});
    graph.finalize();
    graph.forward();
    for (float value : output.data().values()) {
        EXPECT_FLOAT_EQ(value, 0.0f);
    }
}

TEST(DenseLayer, BackwardBatchSizeOne) {
    Graph graph;
    Node& input = graph.add_node(Tensor(1, Shape{1, 2}));
    mlfo::layer::DenseLayer<float> layer(2, 3);
    Node& output = layer.build(graph, input);
    DenseNodes nodes = find_nodes(output, input);
    input.data().set_values({1, 2});
    nodes.weight->data().set_values({1, 2, 3, 4, 5, 6});
    nodes.bias->data().set_values({0.5f, -1, 2});
    graph.finalize();
    graph.forward();
    Access::gradients(output.data()) = {1, 2, 3};
    graph.backward();

    // bias grad == dOut
    const std::vector<float> bias_expected = {1, 2, 3};
    ASSERT_EQ(nodes.bias->data().gradients().size(), 3u);
    for (std::size_t i = 0; i < 3; i++) {
        EXPECT_FLOAT_EQ(nodes.bias->data().gradients()[i], bias_expected[i]);
    }
    // weight grad == x^T * dOut
    const std::vector<float> weight_expected = {1, 2, 3, 2, 4, 6};
    ASSERT_EQ(nodes.weight->data().gradients().size(), 6u);
    for (std::size_t i = 0; i < 6; i++) {
        EXPECT_FLOAT_EQ(nodes.weight->data().gradients()[i], weight_expected[i]);
    }
    // input grad == dOut * W^T
    const std::vector<float> input_expected = {14, 32};
    ASSERT_EQ(input.data().gradients().size(), 2u);
    for (std::size_t i = 0; i < 2; i++) {
        EXPECT_FLOAT_EQ(input.data().gradients()[i], input_expected[i]);
    }
}

TEST(DenseLayer, StackedLayersCompose) {
    Graph graph;
    Node& input = graph.add_node(Tensor(1, Shape{1, 2}));
    mlfo::layer::DenseLayer<float> first(2, 2);
    mlfo::layer::DenseLayer<float> second(2, 1);
    Node& hidden = first.build(graph, input);
    Node& output = second.build(graph, hidden);
    EXPECT_EQ(graph.size(), 9u);
    DenseNodes first_nodes = find_nodes(hidden, input);
    DenseNodes second_nodes = find_nodes(output, hidden);
    input.data().set_values({1, 2});
    first_nodes.weight->data().set_values({1, 0, 1, 1});
    first_nodes.bias->data().set_values({0, 1});
    second_nodes.weight->data().set_values({2, 3});
    second_nodes.bias->data().set_values({1});
    graph.finalize();
    graph.forward();
    // hidden = [1 + 2, 0 + 2] + [0, 1] = [3, 3]; output = 6 + 9 + 1
    EXPECT_FLOAT_EQ(hidden.data().values()[0], 3.0f);
    EXPECT_FLOAT_EQ(hidden.data().values()[1], 3.0f);
    ASSERT_EQ(output.data().values().size(), 1u);
    EXPECT_FLOAT_EQ(output.data().values()[0], 16.0f);
}
