#include <gtest/gtest.h>

#include <functional>
#include <vector>

#include "helpers.hpp"

using mlfo::graph::Node;
using mlfo::tensor::ops::DIRECTION;

namespace {

using Op = std::function<void(DIRECTION)>;

Op recorder(std::vector<DIRECTION>& calls) {
    return [&calls](DIRECTION direction) { calls.push_back(direction); };
}

} // namespace

TEST(NodeTest, DataConstAccess) {
    const Node<int> node(7);
    EXPECT_EQ(node.data(), 7);
}

TEST(NodeTest, DataMutation) {
    Node<int> node(7);
    node.data() = 42;
    EXPECT_EQ(node.data(), 42);
    const Node<int>& const_node = node;
    EXPECT_EQ(const_node.data(), 42);
}

TEST(NodeTest, NoPredecessorsForwardBackwardAreNoOps) {
    Node<int> node(1);
    EXPECT_NO_THROW(node.forward());
    EXPECT_NO_THROW(node.backward());
    EXPECT_EQ(node.data(), 1);
    EXPECT_TRUE(node.predecessors().empty());
    EXPECT_TRUE(node.successors().empty());
}

TEST(NodeTest, AddPredecessorWiresBothDirections) {
    Node<int> a(1);
    Node<int> b(2);
    b.add_predecessor(a, [](DIRECTION) {});
    ASSERT_EQ(b.predecessors().size(), 1u);
    EXPECT_EQ(&b.predecessors()[0].get(), &a);
    ASSERT_EQ(a.successors().size(), 1u);
    EXPECT_EQ(&a.successors()[0].get(), &b);
    EXPECT_TRUE(a.predecessors().empty());
    EXPECT_TRUE(b.successors().empty());
}

TEST(NodeTest, AddPredecessorsWiresMultiple) {
    Node<int> a(1);
    Node<int> b(2);
    Node<int> c(3);
    c.add_predecessors({a, b}, [](DIRECTION) {});
    ASSERT_EQ(c.predecessors().size(), 2u);
    EXPECT_EQ(&c.predecessors()[0].get(), &a);
    EXPECT_EQ(&c.predecessors()[1].get(), &b);
    ASSERT_EQ(a.successors().size(), 1u);
    EXPECT_EQ(&a.successors()[0].get(), &c);
    ASSERT_EQ(b.successors().size(), 1u);
    EXPECT_EQ(&b.successors()[0].get(), &c);
}

TEST(NodeTest, ForwardRunsOperationOnceUntilInvalidated) {
    Node<int> a(1);
    Node<int> b(2);
    std::vector<DIRECTION> calls;
    b.add_predecessor(a, recorder(calls));
    b.forward();
    b.forward();
    ASSERT_EQ(calls.size(), 1u);
    EXPECT_EQ(calls[0], DIRECTION::FORWARD);
    b.invalidate();
    b.forward();
    ASSERT_EQ(calls.size(), 2u);
    EXPECT_EQ(calls[1], DIRECTION::FORWARD);
}

TEST(NodeTest, BackwardRunsOperationOnceUntilInvalidated) {
    Node<int> a(1);
    Node<int> b(2);
    std::vector<DIRECTION> calls;
    b.add_predecessor(a, recorder(calls));
    b.backward();
    b.backward();
    ASSERT_EQ(calls.size(), 1u);
    EXPECT_EQ(calls[0], DIRECTION::BACKWARD);
    b.invalidate();
    b.backward();
    ASSERT_EQ(calls.size(), 2u);
    EXPECT_EQ(calls[1], DIRECTION::BACKWARD);
}

TEST(NodeTest, ForwardAndBackwardDirtyFlagsAreIndependent) {
    Node<int> a(1);
    Node<int> b(2);
    std::vector<DIRECTION> calls;
    b.add_predecessor(a, recorder(calls));
    b.forward();
    b.backward();
    ASSERT_EQ(calls.size(), 2u);
    EXPECT_EQ(calls[0], DIRECTION::FORWARD);
    EXPECT_EQ(calls[1], DIRECTION::BACKWARD);
    b.forward();
    b.backward();
    EXPECT_EQ(calls.size(), 2u);
}

TEST(NodeTest, InvalidateReenablesBoth) {
    Node<int> a(1);
    Node<int> b(2);
    std::vector<DIRECTION> calls;
    b.add_predecessor(a, recorder(calls));
    b.forward();
    b.backward();
    b.invalidate();
    b.forward();
    b.backward();
    ASSERT_EQ(calls.size(), 4u);
    EXPECT_EQ(calls[2], DIRECTION::FORWARD);
    EXPECT_EQ(calls[3], DIRECTION::BACKWARD);
}

TEST(NodeTest, LaterAddPredecessorReplacesOperation) {
    Node<int> a(1);
    Node<int> b(2);
    Node<int> c(3);
    int first_calls = 0;
    int second_calls = 0;
    c.add_predecessor(a, [&first_calls](DIRECTION) { ++first_calls; });
    c.add_predecessor(b, [&second_calls](DIRECTION) { ++second_calls; });
    c.forward();
    EXPECT_EQ(first_calls, 0);
    EXPECT_EQ(second_calls, 1);
    EXPECT_EQ(c.predecessors().size(), 2u);
}

TEST(NodeTest, LaterAddPredecessorsReplacesOperation) {
    Node<int> a(1);
    Node<int> b(2);
    Node<int> c(3);
    int first_calls = 0;
    int second_calls = 0;
    c.add_predecessor(a, [&first_calls](DIRECTION) { ++first_calls; });
    c.add_predecessors({b}, [&second_calls](DIRECTION) { ++second_calls; });
    c.backward();
    EXPECT_EQ(first_calls, 0);
    EXPECT_EQ(second_calls, 1);
}
