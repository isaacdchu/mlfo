#ifndef TESTS_COMMON_HPP
#define TESTS_COMMON_HPP

#include <iostream>
#include <cmath>
#include <string>

static inline void assert_true(bool cond, const std::string& msg) {
    if (!cond) {
        std::cerr << "FAIL: " << msg << std::endl;
        std::exit(1);
    }
    std::cout << "PASS: " << msg << std::endl;
}

static inline bool nearf(float a, float b, float eps=1e-4f) {
    return std::fabs(a - b) <= eps;
}

#endif // TESTS_COMMON_HPP