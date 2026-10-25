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

} // namespace

TEST(MSELoss, Shapes) {
    mlfo::layer::loss::MSELoss<float> layer(4);
    EXPECT_EQ(layer.input_shape(), (Shape{4}));
    EXPECT_EQ(layer.output_shape(), (Shape{1}));
}

TEST(MSELoss, HasNoParameters) {
    Graph graph;
    Node& input = graph.add_node(Tensor(1, Shape{1, 3}));
    mlfo::layer::loss::MSELoss<float> layer(3);
    layer.build(graph, input);
    EXPECT_TRUE(layer.parameters().empty());
}

TEST(MSELoss, LabelNodeBeforeBuildThrows) {
    mlfo::layer::loss::MSELoss<float> layer(3);
    EXPECT_THROW(layer.label_node(), std::logic_error);
}

TEST(MSELoss, BuildAddsOutputAndLabelNodes) {
    Graph graph;
    Node& input = graph.add_node(Tensor(1, Shape{1, 3}));
    mlfo::layer::loss::MSELoss<float> layer(3);
    Node& output = layer.build(graph, input);
    EXPECT_EQ(graph.size(), 3u);
    EXPECT_EQ(output.data().shape(), (Shape{1}));
    // labels take the input node's shape
    EXPECT_EQ(layer.label_node().data().shape(), input.data().shape());
    ASSERT_EQ(output.predecessors().size(), 2u);
    EXPECT_EQ(&output.predecessors()[0].get(), &input);
    EXPECT_EQ(&output.predecessors()[1].get(), &layer.label_node());
}

TEST(MSELoss, ForwardComputesLoss) {
    Graph graph;
    Node& input = graph.add_node(Tensor(1, Shape{1, 3}));
    mlfo::layer::loss::MSELoss<float> layer(3);
    Node& output = layer.build(graph, input);
    input.data().set_values({1, 2, 3});
    layer.label_node().data().set_values({1, 0, 0});
    graph.finalize();
    graph.forward();
    ASSERT_EQ(output.data().values().size(), 1u);
    EXPECT_FLOAT_EQ(output.data().values()[0], 13.0f / 6.0f);
}

TEST(MSELoss, BackwardPropagatesToInput) {
    Graph graph;
    Node& input = graph.add_node(Tensor(1, Shape{1, 3}));
    mlfo::layer::loss::MSELoss<float> layer(3);
    Node& output = layer.build(graph, input);
    input.data().set_values({1, 2, 3});
    layer.label_node().data().set_values({1, 0, 0});
    graph.finalize();
    graph.forward();
    Access::gradients(output.data()) = {1};
    graph.backward();
    const std::vector<float> expected = {0, 2.0f / 3.0f, 1};
    ASSERT_EQ(input.data().gradients().size(), 3u);
    for (std::size_t i = 0; i < 3; i++) {
        EXPECT_FLOAT_EQ(input.data().gradients()[i], expected[i]);
    }
}
