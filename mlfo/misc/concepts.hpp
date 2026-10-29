#ifndef CONCEPTS_HPP
#define CONCEPTS_HPP

#include <concepts>
#include <random>
#include <type_traits>

namespace mlfo::misc {

template <typename T>
concept Number = std::is_floating_point_v<T>;

// a distribution that can be sampled with generator G, e.g. any standard library distribution
// result_type is the type of the values it produces
template <class D, class G>
concept Distribution =
    std::uniform_random_bit_generator<G> &&
    Number<typename D::result_type> &&
    requires(D distribution, G& generator) {
        { distribution(generator) } -> std::convertible_to<typename D::result_type>;
    };

} // namespace mlfo::misc

#endif // CONCEPTS_HPP
