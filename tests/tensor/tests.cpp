#include "../../tensor.hpp"
#include "../../operations.hpp"
#include "../common.hpp"

void run_tensor_tests() {
    std::cout << "Running tensor tests..." << std::endl;
    // test_tensor_basic
    {
        auto t = Tensor::factory({2, 3}, 0, 1.5f);
        assert_true(t->size() == 6, "Tensor size is correct");
        for (std::size_t i = 0; i < t->size(); ++i) {
            assert_true(nearf(t->values()[i], 1.5f), "Tensor initial values all set to fill value");
        }
        assert_true(nearf(t->at({0,0}), 1.5f), "at({0,0}) returns correct value");
        assert_true(nearf(t->at({1,2}), 1.5f), "at({1,2}) returns correct value");
    }

    // test_tensor_gradients_and_zero
    {
        auto t = Tensor::factory({2,2}, 0, 0.0f);
        std::vector<float> grads(t->size(), 3.14f);
        t->set_gradients(std::move(grads));
        assert_true(t->gradients().size() == t->size(), "Gradients vector set and sized correctly");
        t->zero_grad();
        for (float g : t->gradients()) {
            assert_true(nearf(g, 0.0f), "zero_grad zeros all gradients");
        }
    }
}
