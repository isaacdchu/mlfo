#include <gtest/gtest.h>

#include <cstddef>
#include <memory>
#include <stdexcept>
#include <vector>

#include "helpers.hpp"

namespace {

using Tensor = mlfo::tensor::Tensor<float>;
using Graph = mlfo::graph::Graph<Tensor>;
using Node = mlfo::graph::Node<Tensor>;
using Shape = std::vector<std::size_t>;
using Recorder = mlfo::tests::RecordingInitializer<float>;

} // namespace

TEST(Conv2DLayer, BuildInitializesParametersWithFans) {
    Graph graph;
    Node& input = graph.add_node(Tensor(1, Shape{5, 5, 3}));
    auto weight_init = std::make_shared<Recorder>(0.5f);
    auto bias_init = std::make_shared<Recorder>(0.0f);
    // k = 4, r = s = 2, c = 3 and h = w = 5 come from the input node
    mlfo::layer::Conv2DLayer<float> layer(4, 2, 2, 1, 1, 0, 0, weight_init, bias_init);
    layer.build(graph, input);
    ASSERT_EQ(weight_init->calls.size(), 1u);
    // (r, s, c, k)
    EXPECT_EQ(weight_init->calls[0].shape, (Shape{2, 2, 3, 4}));
    // fan_in = c * r * s, fan_out = k * r * s
    EXPECT_EQ(weight_init->calls[0].fan_in, 12u);
    EXPECT_EQ(weight_init->calls[0].fan_out, 16u);
    ASSERT_EQ(bias_init->calls.size(), 1u);
    EXPECT_EQ(bias_init->calls[0].shape, (Shape{4}));
    const auto& parameters = layer.parameters();
    ASSERT_EQ(parameters.size(), 2u);
    for (float value : parameters[0].get().values()) {
        EXPECT_FLOAT_EQ(value, 0.5f);
    }
}

TEST(Conv2DLayer, BuildTakesInputShapeFromInputNode) {
    Graph graph;
    Node& input = graph.add_node(Tensor(1, Shape{7, 6, 2}));
    // (7 + 2 * 1 - 3) / 2 + 1 = 4, (6 + 2 * 0 - 2) / 1 + 1 = 5
    mlfo::layer::Conv2DLayer<float> layer(3, 3, 2, 2, 1, 1, 0);
    Node& output = layer.build(graph, input);
    EXPECT_EQ(layer.input_shape(), (Shape{7, 6, 2}));
    EXPECT_EQ(layer.output_shape(), (Shape{4, 5, 3}));
    EXPECT_EQ(output.data().shape(), (Shape{4, 5, 3}));
    EXPECT_EQ(layer.parameters()[0].get().shape(), (Shape{3, 2, 2, 3}));
}

TEST(Conv2DLayer, BuildRejectsBadInput) {
    // not (H, W, C), and kernel larger than the padded input
    for (const Shape& shape : {Shape{5, 5}, Shape{5, 5, 1, 1}, Shape{2, 5, 1}}) {
        Graph graph;
        Node& input = graph.add_node(Tensor(1, shape));
        mlfo::layer::Conv2DLayer<float> layer(1, 3, 3);
        EXPECT_THROW(layer.build(graph, input), std::invalid_argument);
    }
}

TEST(Conv2DLayer, InvalidConstructionThrows) {
    EXPECT_THROW(mlfo::layer::Conv2DLayer<float>(0, 2, 2), std::invalid_argument);
    EXPECT_THROW(mlfo::layer::Conv2DLayer<float>(1, 2, 2, 0, 1), std::invalid_argument);
    EXPECT_THROW(
        mlfo::layer::Conv2DLayer<float>(1, 2, 2, 1, 1, 0, 0, nullptr),
        std::invalid_argument
    );
}
