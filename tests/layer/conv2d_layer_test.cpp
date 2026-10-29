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
    Node& input = graph.add_node(Tensor(1, Shape{3, 5, 5}));
    auto weight_init = std::make_shared<Recorder>(0.5f);
    auto bias_init = std::make_shared<Recorder>(0.0f);
    // c = 3, h = w = 5, r = s = 2, k = 4
    mlfo::layer::Conv2DLayer<float> layer(3, 5, 5, 2, 2, 4, 1, 1, 0, 0, weight_init, bias_init);
    layer.build(graph, input);
    ASSERT_EQ(weight_init->calls.size(), 1u);
    EXPECT_EQ(weight_init->calls[0].shape, (Shape{4, 3, 2, 2}));
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

TEST(Conv2DLayer, NullInitializerThrows) {
    EXPECT_THROW(
        mlfo::layer::Conv2DLayer<float>(1, 3, 3, 2, 2, 1, 1, 1, 0, 0, nullptr),
        std::invalid_argument
    );
}
