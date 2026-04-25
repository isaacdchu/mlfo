#include "tensor.hpp"
#include "util.hpp"

int main() {
    Tensor t(2, 3, 4);
    util::println("{}", t.to_string());
    return 0;
}
