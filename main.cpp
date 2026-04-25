#include "tensor.hpp"
#include "util.hpp"

int main() {
    Tensor t = Tensor({2, 3, 4});
    t.at({0, 0, 3}) = 42.0f;
    util::println("{}", t.to_string());
    return 0;
}
