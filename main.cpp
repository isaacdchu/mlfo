#include <print>
#include <iostream>

#include <mlfo>

int main() {
    auto t_1 = mlfo::tensor::Tensor<float>(1, {2, 3});
    auto t_2 = mlfo::tensor::Tensor<float>(1, {2, 3});
    auto t_3 = mlfo::tensor::Tensor<float>(1, {2, 3});
    mlfo::tensor::ops::Add<float>::call(
        mlfo::tensor::ops::DIRECTION::FORWARD,
        t_3,
        t_1,
        t_2
    );
    std::cout << t_3.to_string() << std::endl;
    return 0;
}
