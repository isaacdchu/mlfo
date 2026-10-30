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
using FlattenLayer = mlfo::layer::FlattenLayer<float>;

void expect_floats(const std::vector<float>& actual, const std::vector<float>& expected) {
    ASSERT_EQ(actual.size(), expected.size());
    for (std::size_t i = 0; i < expected.size(); i++) {
        EXPECT_FLOAT_EQ(actual[i], expected[i]) << "index " << i;
    }
}

} // namespace

TEST(FlattenLayer, HasNoParameters) {
    FlattenLayer layer;
    EXPECT_TRUE(layer.parameters().empty());
}

TEST(FlattenLayer, ConstructorSetsShapes) {
    FlattenLayer layer(Shape{3, 4, 5});
    EXPECT_EQ(layer.input_shape(), (Shape{3, 4, 5}));
    EXPECT_EQ(layer.output_shape(), (Shape{1, 60}));
}

TEST(FlattenLayer, ConstructorRejectsBadShapes) {
    EXPECT_THROW(FlattenLayer(Shape{}), std::invalid_argument);
    EXPECT_THROW(FlattenLayer(Shape{2, 0}), std::invalid_argument);
}

TEST(FlattenLayer, BuildAddsOneNode) {
    Graph graph;
    Node& input = graph.add_node(Tensor(1, Shape{2, 3}));
    FlattenLayer layer;
    Node& output = layer.build(graph, input);
    EXPECT_EQ(graph.size(), 2u);
    ASSERT_EQ(output.predecessors().size(), 1u);
    EXPECT_EQ(&output.predecessors()[0].get(), &input);
}

TEST(FlattenLayer, BuildFlattensAnyRank) {
    for (const Shape& shape : {Shape{6}, Shape{2, 3}, Shape{2, 3, 4}, Shape{2, 1, 3, 2}}) {
        Graph graph;
        Node& input = graph.add_node(Tensor(3, shape));
        FlattenLayer layer;
        Node& output = layer.build(graph, input);
        const std::size_t size = input.data().unbatched_size();
        EXPECT_EQ(output.data().shape(), (Shape{1, size}));
        EXPECT_EQ(output.data().batch_size(), 3u);
        EXPECT_EQ(layer.input_shape(), shape);
        EXPECT_EQ(layer.output_shape(), (Shape{1, size}));
    }
}

TEST(FlattenLayer, BuildRejectsMismatchedInputShape) {
    Graph graph;
    Node& input = graph.add_node(Tensor(1, Shape{2, 3}));
    FlattenLayer layer(Shape{3, 2});
    EXPECT_THROW(layer.build(graph, input), std::invalid_argument);
}

TEST(FlattenLayer, ForwardKeepsRowMajorOrder) {
    Graph graph;
    Node& input = graph.add_node(Tensor(2, Shape{2, 1, 2}));
    FlattenLayer layer;
    Node& output = layer.build(graph, input);
    input.data().set_values({1, 2, 3, 4, 5, 6, 7, 8});
    graph.finalize();
    graph.forward();
    expect_floats(output.data().values(), {1, 2, 3, 4, 5, 6, 7, 8});
}

TEST(FlattenLayer, BackwardPassesGradientsThrough) {
    Graph graph;
    Node& input = graph.add_node(Tensor(1, Shape{2, 2}));
    FlattenLayer layer;
    Node& output = layer.build(graph, input);
    input.data().set_values({1, 2, 3, 4});
    graph.finalize();
    graph.forward();
    Access::gradients(output.data()) = {0.5f, -1, 2, 3};
    graph.backward();
    expect_floats(input.data().gradients(), {0.5f, -1, 2, 3});
}

TEST(FlattenLayer, ComposesWithDense) {
    Graph graph;
    Node& input = graph.add_node(Tensor(1, Shape{2, 1, 2}));
    FlattenLayer flatten;
    mlfo::layer::DenseLayer<float> dense(4, 1);
    Node& flat = flatten.build(graph, input);
    Node& output = dense.build(graph, flat);
    // dense output -> prebias, bias; prebias -> input, weight
    Node& prebias = output.predecessors()[0].get();
    Node& weight = prebias.predecessors()[1].get();
    input.data().set_values({1, 2, 3, 4});
    weight.data().set_values({1, 1, 1, 1});
    graph.finalize();
    graph.forward();
    // 1 + 2 + 3 + 4, bias is zero
    expect_floats(output.data().values(), {10});
    Access::gradients(output.data()) = {1};
    graph.backward();
    // dIn = weight
    expect_floats(input.data().gradients(), {1, 1, 1, 1});
}
