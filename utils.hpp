#ifndef UTILS_HPP
#define UTILS_HPP

#include <vector>
#include <print>
#include <concepts>
#include <limits>
#include <functional>
#include <iostream>

namespace utils {
    template<typename T, typename U> requires std::is_integral_v<U>
    U max(const std::vector<T>& vec, std::function<U(const T&)> to_value, U default_value = 0) {
        if (vec.empty()) {
            return default_value;
        }
        U max_value = to_value(vec.front());
        for (const auto& value : vec) {
            if (to_value(value) > max_value) {
                max_value = to_value(value);
            }
        }
        return max_value;
    }
}

#endif // UTILS_HPP