#ifndef CONCEPTS_HPP
#define CONCEPTS_HPP

#include <concepts>
#include <type_traits>

namespace mlfo::misc {

template <typename T>
concept Number = std::is_arithmetic_v<T>;

}

#endif // CONCEPTS_HPP
