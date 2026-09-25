#ifndef GRAPH_HPP
#define GRAPH_HPP

#include <algorithm>
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
    // topological sort of nodes in the (directed acyclic) graph
    std::vector<std::unique_ptr<Node<T>>> nodes_;
    // stores the iterator after the last head node
    typename std::vector<std::unique_ptr<Node<T>>>::const_iterator head_end_;
    // stores the iterator of the first tail node
    typename std::vector<std::unique_ptr<Node<T>>>::const_iterator tail_begin_;

    void topological_sort() {
        // Kahn's algorithm for topological sorting, arranged so that all head nodes
        // (no predecessors) come first and all tail nodes (no successors) come last
        std::vector<Node<T>*> sorted_nodes;
        std::vector<Node<T>*> tails;
        std::vector<Node<T>*> no_predecessors;
        std::unordered_map<Node<T>*, std::size_t> in_degree;
        for (const auto& node : nodes_) {
            in_degree[node.get()] = node->predecessors().size();
            if (node->predecessors().empty()) {
                sorted_nodes.push_back(node.get());
            } else if (node->successors().empty()) {
                tails.push_back(node.get());
            }
        }
        const std::size_t num_heads = sorted_nodes.size();
        // Process heads, then the middle nodes; tails are held back until the end
        for (std::size_t i = 0; i < num_heads; ++i) {
            for (Node<T>& successor : sorted_nodes[i]->successors()) {
                if (--in_degree[&successor] == 0 && !successor.successors().empty()) {
                    no_predecessors.push_back(&successor);
                }
            }
        }
        while (!no_predecessors.empty()) {
            auto current = no_predecessors.back();
            no_predecessors.pop_back();
            sorted_nodes.push_back(current);
            for (Node<T>& successor : current->successors()) {
                if (--in_degree[&successor] == 0 && !successor.successors().empty()) {
                    no_predecessors.push_back(&successor);
                }
            }
        }
        if (sorted_nodes.size() + tails.size() != nodes_.size()) {
            throw std::runtime_error("[mlfo::graph::Graph] Graph has at least one cycle");
        }
        sorted_nodes.insert(sorted_nodes.end(), tails.begin(), tails.end());
        // Reorder nodes_ according to sorted_nodes
        std::vector<std::unique_ptr<Node<T>>> sorted_nodes_unique;
        for (auto* node : sorted_nodes) {
            auto it = std::find_if(
                nodes_.begin(),
                nodes_.end(),
                [node](const std::unique_ptr<Node<T>>& n) -> bool {
                    return n.get() == node;
                }
            );
            if (it != nodes_.end()) {
                sorted_nodes_unique.push_back(std::move(*it));
            }
        }
        nodes_ = std::move(sorted_nodes_unique);
        // Update head_end_ and tail_begin_ iterators
        head_end_ = nodes_.begin() + num_heads;
        tail_begin_ = nodes_.end() - tails.size();
    }
public:
    Graph() = default;

    Graph(std::vector<std::unique_ptr<Node<T>>> nodes) {
        // Assumes that nodes are able to form a directed acyclic graph (DAG)
        for (auto& node : nodes) {
            if (node->predecessors().empty() && node->successors().empty()) {
                throw std::invalid_argument(
                    "[mlfo::graph::Graph] Node must have at least one predecessor or successor"
                );
            }
            nodes_.push_back(std::move(node));
        }
        topological_sort();
    }

    void set_nodes(std::vector<std::unique_ptr<Node<T>>> nodes) {
        nodes_.clear();
        for (auto& node : nodes) {
            if (node->predecessors().empty() && node->successors().empty()) {
                throw std::invalid_argument(
                    "[mlfo::graph::Graph] Node must have at least one predecessor or successor"
                );
            }
            nodes_.push_back(std::move(node));
        }
        topological_sort();
    }

    typename std::vector<std::unique_ptr<Node<T>>>::const_iterator begin() const {
        return nodes_.begin();
    }

    typename std::vector<std::unique_ptr<Node<T>>>::const_iterator end() const {
        return nodes_.end();
    }

    typename std::vector<std::unique_ptr<Node<T>>>::const_iterator head_begin() const {
        return nodes_.begin();
    }

    typename std::vector<std::unique_ptr<Node<T>>>::const_iterator head_end() const {
        return head_end_;
    }

    typename std::vector<std::unique_ptr<Node<T>>>::const_iterator tail_begin() const {
        return tail_begin_;
    }

    typename std::vector<std::unique_ptr<Node<T>>>::const_iterator tail_end() const {
        return nodes_.end();
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
