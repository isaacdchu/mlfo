#include <gtest/gtest.h>

#include <algorithm>
#include <functional>
#include <stdexcept>
#include <vector>

#include "helpers.hpp"

using mlfo::graph::Graph;
using mlfo::graph::Node;
using mlfo::tensor::ops::DIRECTION;

namespace {

using Op = std::function<void(DIRECTION)>;

Op noop() {
    return [](DIRECTION) {};
}

// records the node's data value each time the operation runs
Op logger(std::vector<int>& order, int id) {
    return [&order, id](DIRECTION) { order.push_back(id); };
}

int index_of(const Graph<int>& graph, const Node<int>& node) {
    const auto& nodes = graph.nodes();
    for (std::size_t i = 0; i < nodes.size(); ++i) {
        if (nodes[i].get() == &node) {
            return static_cast<int>(i);
        }
    }
    return -1;
}

void expect_topological(const Graph<int>& graph) {
    for (const auto& node : graph.nodes()) {
        for (Node<int>& predecessor : node->predecessors()) {
            EXPECT_LT(index_of(graph, predecessor), index_of(graph, *node));
        }
    }
}

} // namespace

TEST(GraphTest, AddNodeHoldsDataAndSizeCounts) {
    Graph<int> graph;
    EXPECT_EQ(graph.size(), 0u);
    Node<int>& a = graph.add_node(5);
    EXPECT_EQ(a.data(), 5);
    EXPECT_EQ(graph.size(), 1u);
    Node<int>& b = graph.add_node(9);
    EXPECT_EQ(b.data(), 9);
    EXPECT_EQ(graph.size(), 2u);
    EXPECT_NE(&a, &b);
}

TEST(GraphTest, ReferencesStayValidAfterGrowthAndFinalize) {
    Graph<int> graph;
    std::vector<Node<int>*> refs;
    Node<int>& first = graph.add_node(0);
    refs.push_back(&first);
    for (int i = 1; i < 100; ++i) {
        refs.push_back(&graph.add_node(i));
    }
    for (int i = 0; i < 100; ++i) {
        EXPECT_EQ(refs[i]->data(), i);
    }
    // wire in reverse so finalize has to reorder
    for (int i = 0; i < 99; ++i) {
        refs[i]->add_predecessor(*refs[i + 1], noop());
    }
    graph.finalize();
    for (int i = 0; i < 100; ++i) {
        EXPECT_EQ(refs[i]->data(), i);
        EXPECT_GE(index_of(graph, *refs[i]), 0);
    }
    EXPECT_EQ(graph.size(), 100u);
}

TEST(GraphTest, FinalizeOrdersChainTopologically) {
    Graph<int> graph;
    // added successors first: c depends on b depends on a
    Node<int>& c = graph.add_node(3);
    Node<int>& b = graph.add_node(2);
    Node<int>& a = graph.add_node(1);
    c.add_predecessor(b, noop());
    b.add_predecessor(a, noop());
    graph.finalize();
    ASSERT_EQ(graph.size(), 3u);
    EXPECT_EQ(index_of(graph, a), 0);
    EXPECT_EQ(index_of(graph, b), 1);
    EXPECT_EQ(index_of(graph, c), 2);
}

TEST(GraphTest, FinalizeOrdersDiamondTopologically) {
    Graph<int> graph;
    Node<int>& d = graph.add_node(4);
    Node<int>& b = graph.add_node(2);
    Node<int>& c = graph.add_node(3);
    Node<int>& a = graph.add_node(1);
    b.add_predecessor(a, noop());
    c.add_predecessor(a, noop());
    d.add_predecessors({b, c}, noop());
    graph.finalize();
    ASSERT_EQ(graph.size(), 4u);
    expect_topological(graph);
    EXPECT_EQ(index_of(graph, a), 0);
    EXPECT_EQ(index_of(graph, d), 3);
}

TEST(GraphTest, FinalizeThrowsOnCycle) {
    Graph<int> graph;
    Node<int>& a = graph.add_node(1);
    Node<int>& b = graph.add_node(2);
    Node<int>& c = graph.add_node(3);
    b.add_predecessor(a, noop());
    c.add_predecessor(b, noop());
    a.add_predecessor(c, noop());
    EXPECT_THROW(graph.finalize(), std::runtime_error);
}

TEST(GraphTest, ForwardRunsInTopologicalOrderBackwardInReverse) {
    Graph<int> graph;
    std::vector<int> order;
    Node<int>& c = graph.add_node(3);
    Node<int>& b = graph.add_node(2);
    Node<int>& a = graph.add_node(1);
    c.add_predecessor(b, logger(order, 3));
    b.add_predecessor(a, logger(order, 2));
    // a has no predecessors, so its operation is a no-op and is not recorded
    graph.finalize();

    graph.forward();
    EXPECT_EQ(order, (std::vector<int>{2, 3}));

    order.clear();
    graph.backward();
    EXPECT_EQ(order, (std::vector<int>{3, 2}));
}

TEST(GraphTest, SecondForwardDoesNotRerunUntilInvalidated) {
    Graph<int> graph;
    std::vector<int> order;
    Node<int>& a = graph.add_node(1);
    Node<int>& b = graph.add_node(2);
    b.add_predecessor(a, logger(order, 2));
    graph.finalize();

    graph.forward();
    ASSERT_EQ(order.size(), 1u);
    graph.forward();
    EXPECT_EQ(order.size(), 1u);

    graph.invalidate();
    graph.forward();
    EXPECT_EQ(order.size(), 2u);
}
