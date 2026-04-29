#include <iostream>

extern void run_tensor_tests();
extern void run_operations_tests();

int main() {
    std::cout << "Running all tests..." << std::endl;
    run_tensor_tests();
    run_operations_tests();
    std::cout << "All tests passed." << std::endl;
    return 0;
}
