#ifndef GRAPH_HPP
#define GRAPH_HPP

#include <ranges>
#include <memory>
#include <vector>
#include <utility>
#include <stdexcept>
#include <unordered_map>

#include "node.hpp"

namespace mlfo::graph {

template <class T>
class Graph {
private:
    // owns every node in the network
    // after finalize(), nodes are in topological order (predecessors before successors)
    // nodes are heap allocated so references to them and their data stay valid as the graph grows
    std::vector<std::unique_ptr<Node<T>>> nodes_;
public:
    Graph() = default;

    // creates a node owned by this graph
    // the returned reference is valid for the lifetime of the graph
    Node<T>& add_node(T data) {
        nodes_.push_back(std::make_unique<Node<T>>(std::move(data)));
        return *nodes_.back();
    }

    // topologically sorts the nodes (Kahn's algorithm)
    // to be called after all nodes are wired up
    void finalize() {
        std::unordered_map<Node<T>*, std::size_t> in_degree;
        std::vector<Node<T>*> ready;
        for (const auto& node : nodes_) {
            in_degree[node.get()] = node->predecessors().size();
            if (node->predecessors().empty()) {
                ready.push_back(node.get());
            }
        }
        std::vector<Node<T>*> sorted;
        for (std::size_t i = 0; i < ready.size(); ++i) {
            sorted.push_back(ready[i]);
            for (Node<T>& successor : ready[i]->successors()) {
                if (--in_degree[&successor] == 0) {
                    ready.push_back(&successor);
                }
            }
        }
        if (sorted.size() != nodes_.size()) {
            throw std::runtime_error("[mlfo::graph::Graph] Graph has at least one cycle");
        }
        std::unordered_map<Node<T>*, std::unique_ptr<Node<T>>> owned;
        for (auto& node : nodes_) {
            owned[node.get()] = std::move(node);
        }
        nodes_.clear();
        for (auto* node : sorted) {
            nodes_.push_back(std::move(owned[node]));
        }
    }
    
    // marks every node as needing recomputation, e.g. after an input changed
    void invalidate() {
        for (auto& node : nodes_) {
            node->invalidate();
        }
    }

    void forward() {
        for (auto& node : nodes_) {
            node->forward();
        }
    }

    void backward() {
        for (auto& node : nodes_ | std::views::reverse) {
            node->backward();
        }
    }

    const std::vector<std::unique_ptr<Node<T>>>& nodes() const {
        return nodes_;
    }

    std::size_t size() const {
        return nodes_.size();
    }
};

} // namespace mlfo::graph

#endif // GRAPH_HPP
