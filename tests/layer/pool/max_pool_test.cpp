#include <gtest/gtest.h>

#include <cstddef>
#include <stdexcept>
#include <vector>

#include "helpers.hpp"

namespace {

using Tensor = mlfo::tensor::Tensor<float>;
using Graph = mlfo::graph::Graph<Tensor>;
using Node = mlfo::graph::Node<Tensor>;
using Access = mlfo::tests::TensorAccess<float>;
using Shape = std::vector<std::size_t>;
using MaxPool = mlfo::layer::pool::MaxPool<float>;

void expect_floats(const std::vector<float>& actual, const std::vector<float>& expected) {
    ASSERT_EQ(actual.size(), expected.size());
    for (std::size_t i = 0; i < expected.size(); i++) {
        EXPECT_FLOAT_EQ(actual[i], expected[i]) << "index " << i;
    }
}

} // namespace

TEST(MaxPool, HasNoParameters) {
    MaxPool layer(4, 4, 2, 2, 2, 2);
    EXPECT_TRUE(layer.parameters().empty());
}

TEST(MaxPool, RejectsInvalidConstruction) {
    EXPECT_THROW(MaxPool(0, 4, 2, 2), std::invalid_argument);
    EXPECT_THROW(MaxPool(4, 4, 0, 2), std::invalid_argument);
    EXPECT_THROW(MaxPool(4, 4, 2, 2, 0, 1), std::invalid_argument);
}

TEST(MaxPool, BuildComputesOutputShape) {
    Graph graph;
    Node& input = graph.add_node(Tensor(3, Shape{2, 5, 4}));
    MaxPool layer(5, 4, 2, 2, 2, 2);
    Node& output = layer.build(graph, input);
    EXPECT_EQ(graph.size(), 2u);
    EXPECT_EQ(output.data().shape(), (Shape{2, 3, 2}));
    EXPECT_EQ(output.data().batch_size(), 3u);
    EXPECT_EQ(layer.input_shape(), (Shape{2, 5, 4}));
    EXPECT_EQ(layer.output_shape(), (Shape{2, 3, 2}));
}

TEST(MaxPool, StrideDefaultsToWindowSize) {
    Graph graph;
    Node& input = graph.add_node(Tensor(1, Shape{6, 6}));
    MaxPool layer(6, 6, 2, 3);
    Node& output = layer.build(graph, input);
    EXPECT_EQ(output.data().shape(), (Shape{3, 2}));
}

TEST(MaxPool, BuildRejectsMismatchedInput) {
    Graph graph;
    Node& input = graph.add_node(Tensor(1, Shape{1, 4, 4}));
    MaxPool layer(5, 4, 2, 2);
    EXPECT_THROW(layer.build(graph, input), std::invalid_argument);
}

TEST(MaxPool, ForwardAndBackward) {
    Graph graph;
    Node& input = graph.add_node(Tensor(1, Shape{1, 2, 4}));
    MaxPool layer(2, 4, 2, 2, 2, 2);
    Node& output = layer.build(graph, input);
    input.data().set_values({1, 3, 2, 0, 2, 0, 1, 5});
    graph.finalize();
    graph.forward();
    expect_floats(output.data().values(), {3, 5});
    Access::gradients(output.data()) = {10, 20};
    graph.backward();
    expect_floats(input.data().gradients(), {0, 10, 0, 0, 0, 0, 0, 20});
}

TEST(MaxPool, OneByOneWindowWithStride) {
    Graph graph;
    Node& input = graph.add_node(Tensor(1, Shape{3, 4}));
    MaxPool layer(3, 4, 1, 1, 2, 2);
    Node& output = layer.build(graph, input);
    EXPECT_EQ(output.data().shape(), (Shape{2, 2}));
    input.data().set_values({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
    graph.finalize();
    graph.forward();
    expect_floats(output.data().values(), {1, 3, 9, 11});
    Access::gradients(output.data()) = {1, 2, 3, 4};
    graph.backward();
    expect_floats(input.data().gradients(), {1, 0, 2, 0, 0, 0, 0, 0, 3, 0, 4, 0});
}
