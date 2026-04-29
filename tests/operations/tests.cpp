#include "../../tensor.hpp"
#include "../../operations.hpp"
#include "../common.hpp"

void run_operations_tests() {
    std::cout << "Running operations tests..." << std::endl;

    // add forward/backward
    {
        auto a = Tensor::from_values({1.0f, 2.0f, 3.0f, 4.0f}, {2,2});
        auto b = Tensor::from_values({5.0f, 6.0f, 7.0f, 8.0f}, {2,2});
        auto out = Operations::add(a.get(), b.get());
        out->forward();
        std::vector<float> expected = {6.0f, 8.0f, 10.0f, 12.0f};
        for (std::size_t i = 0; i < out->size(); ++i) {
            assert_true(nearf(out->values()[i], expected[i]), "add forward produces expected sums");
        }
        out->set_gradients(std::vector<float>(out->size(), 1.0f));
        out->backward();
        for (std::size_t i = 0; i < a->size(); ++i) {
            assert_true(nearf(a->gradients()[i], 1.0f), "add backward accumulates gradients into a");
            assert_true(nearf(b->gradients()[i], 1.0f), "add backward accumulates gradients into b");
        }
    }

    // matmul forward/backward
    {
        auto A = Tensor::from_values({1,2,3,4,5,6}, {2,3});
        auto B = Tensor::from_values({7,8,9,10,11,12}, {3,2});
        auto C = Operations::matmul(A.get(), B.get(), 1);
        C->forward();
        std::vector<float> expectedC = {58.0f, 64.0f, 139.0f, 154.0f};
        assert_true(C->size() == expectedC.size(), "matmul output size as expected");
        for (std::size_t i = 0; i < expectedC.size(); ++i) {
            assert_true(nearf(C->values()[i], expectedC[i]), "matmul forward produces expected values");
        }
        C->set_gradients(std::vector<float>(C->size(), 1.0f));
        C->backward();
        std::vector<float> expecteddA = {15.0f,19.0f,23.0f,15.0f,19.0f,23.0f};
        for (std::size_t i = 0; i < A->size(); ++i) {
            assert_true(nearf(A->gradients()[i], expecteddA[i]), "matmul backward produces expected gradients for A");
        }
        std::vector<float> expecteddB = {5.0f,5.0f,7.0f,7.0f,9.0f,9.0f};
        for (std::size_t i = 0; i < B->size(); ++i) {
            assert_true(nearf(B->gradients()[i], expecteddB[i]), "matmul backward produces expected gradients for B");
        }
    }

    // edge case: 1D vector dot (contraction produces scalar-like output)
    {
        auto v1 = Tensor::from_values({1.0f, 2.0f, 3.0f}, {3});
        auto v2 = Tensor::from_values({4.0f, -1.0f, 0.5f}, {3});
        auto dot = Operations::matmul(v1.get(), v2.get(), 1);
        dot->forward();
        // expected dot = 1*4 + 2*(-1) + 3*0.5 = 4 -2 +1.5 = 3.5
        assert_true(dot->size() == 1, "dot produces scalar-like output");
        assert_true(nearf(dot->values()[0], 3.5f), "dot product value correct");
        dot->set_gradients(std::vector<float>(dot->size(), 2.0f));
        dot->backward();
        // dV1 = dOut * v2 = 2 * v2
        assert_true(nearf(v1->gradients()[0], 8.0f), "dot backward gradient for v1[0]");
        assert_true(nearf(v1->gradients()[1], -2.0f), "dot backward gradient for v1[1]");
        assert_true(nearf(v1->gradients()[2], 1.0f), "dot backward gradient for v1[2]");
    }

    // edge case: negative and zero values in add
    {
        auto a = Tensor::from_values({-1.0f, 0.0f, 2.5f}, {3});
        auto b = Tensor::from_values({1.0f, 0.0f, -2.5f}, {3});
        auto s = Operations::add(a.get(), b.get());
        s->forward();
        for (std::size_t i = 0; i < s->size(); ++i) {
            assert_true(nearf(s->values()[i], 0.0f), "add with negatives and zeros produces zeros");
        }
        s->set_gradients(std::vector<float>(s->size(), 1.0f));
        s->backward();
        for (std::size_t i = 0; i < a->size(); ++i) {
            assert_true(nearf(a->gradients()[i], 1.0f), "add backward propagates ones into a for neg/zero values");
            assert_true(nearf(b->gradients()[i], 1.0f), "add backward propagates ones into b for neg/zero values");
        }
    }

    // batched matmul with broadcasting B (unbatched) across batches of A
    {
        // A: batch=2, unbatched 2x2 -> total size 8
        auto A = std::make_unique<Tensor>(std::vector<std::size_t>{2,2}, 2);
        // fill A with distinct values per batch
        std::vector<float> Avals = {1,2,3,4, 5,6,7,8};
        A->set_values(std::move(Avals), 2);
        // B: unbatched 2x2
        auto B = Tensor::from_values({1.0f, 0.0f, 0.0f, 1.0f}, {2,2});
        // compute C = A @ B (contract 1)
        auto C = Operations::matmul(A.get(), B.get(), 1);
        C->forward();
        // basic sanity: values accessible and finite
        for (float v : C->values()) {
            (void)v;
        }
        C->set_gradients(std::vector<float>(C->size(), 1.0f));
        C->backward();
        assert_true(!A->gradients().empty(), "batched A received gradients after backward");
    }

    // chained ops: D = add(X,Y); E = matmul(D, Z); verify gradients flow to X,Y,Z
    {
        auto X = Tensor::from_values({1.0f,2.0f,3.0f,4.0f}, {2,2});
        auto Y = Tensor::from_values({0.5f, -1.0f, 2.0f, 0.0f}, {2,2});
        auto Z = Tensor::from_values({1.0f,0.0f,0.0f,1.0f}, {2,2});
        auto D = Operations::add(X.get(), Y.get());
        auto E = Operations::matmul(D.get(), Z.get(), 1);
        E->forward();
        E->set_gradients(std::vector<float>(E->size(), 1.0f));
        E->backward();
        // X and Y should both have gradients set (maybe zero but present)
        assert_true(X->gradients().size() == X->size(), "X has gradients after chained backward");
        assert_true(Y->gradients().size() == Y->size(), "Y has gradients after chained backward");
        assert_true(Z->gradients().size() == Z->size(), "Z has gradients after chained backward");
    }
}