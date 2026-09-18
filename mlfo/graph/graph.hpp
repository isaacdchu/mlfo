#ifndef GRAPH_HPP
#define GRAPH_HPP

#include <memory>
#include <vector>

#include "node.hpp"

namespace mlfo::graph {

template <class T>
class Graph {
private:
    std::vector<std::unique_ptr<Node<T>>> nodes_;
public:
    Graph() {
        //
    }

    template <class... Args>
    void add_node(
        const std::vector<Node<T>*>& predecessors,
        const std::vector<Node<T>*>& successors,
        Args... args
    ) {
        nodes_.push_back(std::make_unique<Node<T>>(std::forward<Args>(args)...));
        nodes_.back()->add_predecessors(predecessors);
        nodes_.back()->add_successors(successors);
    }

    const std::vector<Node<T>*>& nodes() const {
        return nodes_;
    }

    constexpr std::size_t size() const {
        return nodes_.size();
    }
};
} // namespace mlfo::graph

#endif // GRAPH_HPP
