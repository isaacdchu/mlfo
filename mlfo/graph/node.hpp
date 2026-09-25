#ifndef NODE_HPP
#define NODE_HPP

#include <functional>
#include <memory>
#include <vector>
#include <utility>

#include "../tensor/ops.hpp"

namespace mlfo::graph {

template <class T>
class Node {
private:
    std::unique_ptr<T> data_;
    std::vector<std::reference_wrapper<Node<T>>> successors_;
    std::vector<std::reference_wrapper<Node<T>>> predecessors_;
    // function that performs forward and backward pass
    std::function<void(mlfo::tensor::ops::DIRECTION)> operation_;
    bool forward_dirty_;
    bool backward_dirty_;
public:
    Node(std::unique_ptr<T> data) :
    data_(std::move(data)),
    operation_(
        [](mlfo::tensor::ops::DIRECTION direction) -> void {
            // nothing
        }
    ),
    forward_dirty_(true),
    backward_dirty_(true)
    {
        // nothing
    }

    const T& data() const {
        return *data_;
    }

    T& data() {
        return *data_;
    }

    void add_predecessor(
        Node<T>& predecessor,
        std::function<void(mlfo::tensor::ops::DIRECTION)> operation
    ) {
        predecessors_.push_back(predecessor);
        predecessor.successors_.push_back(*this);
        operation_ = operation;
    }

    void add_predecessors(
        const std::vector<std::reference_wrapper<Node<T>>>& predecessors,
        std::function<void(mlfo::tensor::ops::DIRECTION)> operation
    ) {
        for (auto& predecessor : predecessors) {
            predecessors_.push_back(predecessor);
            predecessor.successors_.push_back(*this);
        }
        operation_ = operation;
    }

    const std::vector<std::reference_wrapper<Node<T>>>& successors() {
        return successors_;
    }

    const std::vector<std::reference_wrapper<Node<T>>>& predecessors() {
        return predecessors_;
    }

    void forward() {
        if (!forward_dirty_) {
            return;
        }
        operation_(mlfo::tensor::ops::DIRECTION::FORWARD);
        forward_dirty_ = false;
    }

    void backward() {
        if (!backward_dirty_) {
            return;
        }
        operation_(mlfo::tensor::ops::DIRECTION::BACKWARD);
        backward_dirty_ = false;
    }
};

} // namespace mlfo::graph

#endif // NODE_HPP
