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

    // test_set_values_and_batching
    {
        auto t = Tensor::factory({2,2}, 0, 1.0f);
        // convert to batched with batch_size=3
        std::vector<float> batched_vals(3 * t->size());
        for (std::size_t i = 0; i < batched_vals.size(); ++i) batched_vals[i] = static_cast<float>(i);
        t->set_values(std::move(batched_vals), 3);
        assert_true(t->batched(), "set_values converted to batched");
        assert_true(t->batch_size() == 3, "batch size set correctly");
        assert_true(t->size() == 12, "size updated correctly for batched tensor");
        assert_true(nearf(t->at({0,0,0}), 0.0f), "first batched element correct");

        // convert back to unbatched values
        std::vector<float> back_vals = {42.0f, 43.0f, 44.0f, 45.0f};
        t->set_values(std::move(back_vals), 0);
        assert_true(!t->batched(), "set_values converted back to unbatched");
        assert_true(t->size() == 4, "size restored after unbatching");
        assert_true(nearf(t->at({1,1}), 45.0f), "values restored after unbatching");
    }

    // test_expand_and_flatten_roundtrip
    {
        auto t = Tensor::factory({2,3}, 0, 0.0f);
        for (std::size_t i = 0; i < t->size(); ++i) {
            auto indices = t->expand_index(i);
            auto flat = t->flatten_indices(indices);
            assert_true(flat == i, "expand_index and flatten_indices roundtrip");
        }
    }

    // test_set_parents_and_children_links
    {
        auto a = Tensor::factory({2}, 0, 1.0f);
        auto b = Tensor::factory({2}, 0, 2.0f);
        auto c = Tensor::factory({2}, 0, 0.0f);
        c->set_parents({a.get(), b.get()});
        assert_true(c->parents().size() == 2, "parents size set correctly");
        assert_true(a->children().size() == 1 && a->children()[0] == c.get(), "parent a has child c");
        assert_true(b->children().size() == 1 && b->children()[0] == c.get(), "parent b has child c");
    }

    // test_set_gradients_and_grad_at
    {
        auto t = Tensor::factory({2,2}, 0, 0.0f);
        std::vector<float> grads = {1.0f, 2.0f, 3.0f, 4.0f};
        t->set_gradients(std::move(grads));
        assert_true(nearf(t->grad_at({0,0}), 1.0f), "grad_at returns set gradient");
        assert_true(nearf(t->grad_at({1,1}), 4.0f), "grad_at returns set gradient");
    }

    // test_forward_backward_noop
    {
        auto t = Tensor::factory({1}, 0, 0.0f);
        // forward/backward should be no-ops when callbacks not set
        t->forward();
        t->backward();
        assert_true(true, "forward/backward no-op when not configured");
    }

    // test_to_string_nonempty
    {
        auto t = Tensor::factory({2,2}, 0, 0.5f);
        auto s = t->to_string();
        assert_true(!s.empty(), "to_string produces non-empty output");
    }
}
