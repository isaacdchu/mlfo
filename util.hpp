#ifndef UTIL_HPP
#define UTIL_HPP

#include <print>

namespace util {
    template<class... Args>
    void print(std::format_string<Args...> fmt, Args&&... args ) {
        std::print(std::cout, fmt, std::forward<Args>(args)...);
    }

    template<class... Args>
    void println(std::format_string<Args...> fmt, Args&&... args ) {
        std::println(std::cout, fmt, std::forward<Args>(args)...);
    }
}

#endif // UTIL_HPP