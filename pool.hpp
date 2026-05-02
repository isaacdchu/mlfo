#ifndef POOL_HPP
#define POOL_HPP

#include "tensor.hpp"

#include <memory>
#include <vector>
#include <stdexcept>
#include <ranges>
#include <limits>

class Pool {
private:
    const std::size_t id_;
    std::vector<std::unique_ptr<Tensor>> tensors_;
    static std::vector<std::size_t> freed_old_ids_;
    static std::size_t next_new_id_;
    static std::size_t next_id() {
        if (!freed_old_ids_.empty()) {
            std::size_t id = freed_old_ids_.back();
            freed_old_ids_.pop_back();
            return id;
        }
        // Generate a new ID if no reusable IDs are available
        if (next_new_id_ == std::numeric_limits<std::size_t>::max()) {
            throw std::runtime_error("[Pool::next_id] Exceeded maximum number of pools");
        }
        return next_new_id_++;
    }

public:
    Pool() : id_(next_id()) {
        // 
    }

    std::size_t id() const {
        return id_;
    }

    Tensor* new_tensor(
        const std::vector<std::size_t>& shape,
        std::size_t batch_size = 0,
        float fill_value = 0.0f
    ) {
        std::unique_ptr<Tensor> tensor = std::make_unique<Tensor>(shape, batch_size, fill_value);
        tensor->pool_ = this;
        Tensor* tensor_ptr = tensor.get();
        tensors_.push_back(std::move(tensor));
        return tensor_ptr;
    }

    auto tensors() const {
        return tensors_ | std::views::transform(
            [](const std::unique_ptr<Tensor>& tensor) -> Tensor* {
                return tensor.get();
            }
        );
    }
};

inline std::size_t Pool::next_new_id_ = 0;
inline std::vector<std::size_t> Pool::freed_old_ids_;

#endif // POOL_HPP