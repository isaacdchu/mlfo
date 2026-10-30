#include <gtest/gtest.h>

#include <cstddef>
#include <stdexcept>
#include <vector>

#include "helpers.hpp"

TEST(ValuesInitializer, CopiesValues) {
    mlfo::tensor::Tensor<float> tensor(1, std::vector<std::size_t>{2, 2});
    mlfo::initializer::Values<float>({1, 2, 3, 4}).initialize(tensor, 2, 2);
    EXPECT_EQ(tensor.values(), (std::vector<float>{1, 2, 3, 4}));
}

TEST(ValuesInitializer, WrongSizeThrows) {
    mlfo::tensor::Tensor<float> tensor(1, std::vector<std::size_t>{2, 2});
    EXPECT_THROW(
        mlfo::initializer::Values<float>({1, 2, 3}).initialize(tensor, 2, 2),
        std::invalid_argument
    );
}
