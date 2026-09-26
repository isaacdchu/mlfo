#include <print>
#include <iostream>

#include <mlfo>

int main() {
    mlfo::layer::DenseLayer<float> dense(3, 2);
    dense.forward();
    return 0;
}
