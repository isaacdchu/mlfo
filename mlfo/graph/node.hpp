#ifndef NODE_HPP
#define NODE_HPP

#include <memory>
#include <vector>
#include <utility>

namespace mlfo::graph {

template <class T>
class Node {
private:
    std::unique_ptr<T> data_;
    std::vector<Node*> successors_;
    std::vector<Node*> predecessors_;
public:
    template <class... Args>
    Node(Args... args) :
    data_(std::make_unique<T>(std::forward<Args>(args)...)) {
        //
    }

    constexpr const T& data() const {
        return *data_;
    }

    void add_successor(Node* successor) {
        successors_.push_back(successor);
        successor->predecessors_.push_back(this);
    }

    void add_successors(const std::vector<Node*>& successors) {
        for (Node* successor : successors) {
            add_successor(successor);
        }
    }

    void add_predecessor(Node* predecessor) {
        predecessors_.push_back(predecessor);
        predecessor->successors_.push_back(this);
    }

    void add_predecessors(const std::vector<Node*>& predecessors) {
        for (Node* predecessor : predecessors) {
            add_predecessor(predecessor);
        }
    }

    const std::vector<Node*>& successors() const {
        return successors_;
    }

    const std::vector<Node*>& predecessors() const {
        return predecessors_;
    }
};

} // namespace mlfo::graph

#endif // NODE_HPP
