#include <print>

#include <mlfo>

int main() {
    auto tensor = mlfo::tensor::Tensor<float>(1, {2, 3});
    std::println("{}", tensor.to_string());
    return 0;
}
