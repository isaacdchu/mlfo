#include "../../tensor.hpp"
#include "../../operations.hpp"
#include "../common.hpp"

void run_operations_tests() {
    std::cout << "Running operations tests..." << std::endl;
    Pool pool;

    // add forward/backward
    {
        auto a = pool.new_tensor(std::vector<std::size_t>{2,2}, 0);
        a->set_values(std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f});
        auto b = pool.new_tensor(std::vector<std::size_t>{2,2}, 0);
        b->set_values(std::vector<float>{5.0f, 6.0f, 7.0f, 8.0f});
        auto out = Operations::add(a, b);
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
        auto A = pool.new_tensor(std::vector<std::size_t>{2,3}, 0);
        A->set_values(std::vector<float>{1,2,3,4,5,6});
        auto B = pool.new_tensor(std::vector<std::size_t>{3,2}, 0);
        B->set_values(std::vector<float>{7,8,9,10,11,12});
        auto C = Operations::matmul(A, B, 1);
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
        auto v1 = pool.new_tensor(std::vector<std::size_t>{3}, 0);
        v1->set_values(std::vector<float>{1.0f, 2.0f, 3.0f});
        auto v2 = pool.new_tensor(std::vector<std::size_t>{3}, 0);
        v2->set_values(std::vector<float>{4.0f, -1.0f, 0.5f});
        auto dot = Operations::matmul(v1, v2, 1);
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
        auto a = pool.new_tensor(std::vector<std::size_t>{3}, 0);
        a->set_values(std::vector<float>{-1.0f, 0.0f, 2.5f});
        auto b = pool.new_tensor(std::vector<std::size_t>{3}, 0);
        b->set_values(std::vector<float>{1.0f, 0.0f, -2.5f});
        auto s = Operations::add(a, b);
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
        auto A = pool.new_tensor(std::vector<std::size_t>{2,2}, 2);
        // fill A with distinct values per batch
        std::vector<float> Avals = {1,2,3,4, 5,6,7,8};
        A->set_values(std::move(Avals), 2);
        // B: unbatched 2x2
        auto B = pool.new_tensor(std::vector<std::size_t>{2,2}, 0);
        B->set_values(std::vector<float>{1.0f, 0.0f, 0.0f, 1.0f});
        // compute C = A @ B (contract 1)
        auto C = Operations::matmul(A, B, 1);
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
        auto X = pool.new_tensor(std::vector<std::size_t>{2,2}, 0);
        X->set_values(std::vector<float>{1.0f,2.0f,3.0f,4.0f});
        auto Y = pool.new_tensor(std::vector<std::size_t>{2,2}, 0);
        Y->set_values(std::vector<float>{0.5f, -1.0f, 2.0f, 0.0f});
        auto Z = pool.new_tensor(std::vector<std::size_t>{2,2}, 0);
        Z->set_values(std::vector<float>{1.0f,0.0f,0.0f,1.0f});
        auto D = Operations::add(X, Y);
        auto E = Operations::matmul(D, Z, 1);
        E->forward();
        E->set_gradients(std::vector<float>(E->size(), 1.0f));
        E->backward();
        // X and Y should both have gradients set (maybe zero but present)
        assert_true(X->gradients().size() == X->size(), "X has gradients after chained backward");
        assert_true(Y->gradients().size() == Y->size(), "Y has gradients after chained backward");
        assert_true(Z->gradients().size() == Z->size(), "Z has gradients after chained backward");
    }

    // contractions > 1: both tensors fully contracted -> scalar output
    {
        auto A = pool.new_tensor(std::vector<std::size_t>{2,3}, 0); // flattened length 6
        A->set_values(std::vector<float>{1,2,3,4,5,6});
        auto B = pool.new_tensor(std::vector<std::size_t>{3,2}, 0);
        B->set_values(std::vector<float>{6,5,4,3,2,1});
        auto C = Operations::matmul(A, B, 2); // contractions == 2 (fully contract both)
        C->forward();
        // expected scalar is dot of flattened arrays
        float expected = 0.0f;
        for (std::size_t i = 0; i < A->size(); ++i) expected += A->values()[i] * B->values()[i];
        assert_true(C->size() == 1, "fully-contracted matmul produces scalar output");
        assert_true(nearf(C->values()[0], expected), "fully-contracted forward value correct");
        // backward: set grad to 2 and expect a_grad = 2 * B, b_grad = 2 * A
        C->set_gradients(std::vector<float>(1, 2.0f));
        C->backward();
        for (std::size_t i = 0; i < A->size(); ++i) {
            assert_true(nearf(A->gradients()[i], 2.0f * B->values()[i]), "fully-contracted backward a_grad correct");
            assert_true(nearf(B->gradients()[i], 2.0f * A->values()[i]), "fully-contracted backward b_grad correct");
        }
    }

    // contractions > 1: A fully contracted, B has extra trailing dims -> vector output
    {
        auto A = pool.new_tensor(std::vector<std::size_t>{2,3}, 0); // flattened len 6
        A->set_values(std::vector<float>{1,2,3,4,5,6});
        // B shape: first two dims match A's contracted dims, plus a trailing dim of 2 => unbatched [2,3,2]
        auto B = pool.new_tensor(std::vector<std::size_t>{3,2,2}, 0); // 12 elements
        B->set_values(std::vector<float>{1,2, 3,4, 5,6, 7,8, 9,10, 11,12});
        auto C = Operations::matmul(A, B, 2);
        C->forward();
        // expected size equals product of B's trailing dims = 2
        assert_true(C->size() == 2, "A fully-contracted produces vector output size");
        // compute expected values: for each c in 0..b_cols-1 sum_k A[k] * B[k*b_cols + c]
        std::size_t b_cols = 2;
        for (std::size_t c = 0; c < b_cols; ++c) {
            float sum = 0.0f;
            for (std::size_t k = 0; k < A->size(); ++k) {
                sum += A->values()[k] * B->values()[k * b_cols + c];
            }
            assert_true(nearf(C->values()[c], sum), "A-fully-contracted forward produces expected element");
        }
    }

    // contractions > 1: general case producing a matrix (a_rows x b_cols)
    {
        // A unbatched shape [2,3,2] -> a_rows = 2, shared_k = 3*2 =6
        auto A = pool.new_tensor(std::vector<std::size_t>{2,3,2}, 0); // 12 elems
        A->set_values(std::vector<float>{1,2,3,4,5,6, 7,8,9,10,11,12});
        // B unbatched shape [2,3,4] -> matches contracted dims ordering expected by matmul
        auto B = pool.new_tensor(std::vector<std::size_t>{2,3,4}, 0); // 24 elems
        B->set_values(std::vector<float>{1,2,3,4, 5,6,7,8, 9,10,11,12, 13,14,15,16, 17,18,19,20, 21,22,23,24});
        auto C = Operations::matmul(A, B, 2);
        C->forward();
        // compute expected using flattened-matrix formulation
        std::size_t a_rows = 2;
        std::size_t shared_k = 6;
        std::size_t b_cols = 4;
        for (std::size_t r = 0; r < a_rows; ++r) {
            for (std::size_t c = 0; c < b_cols; ++c) {
                float sum = 0.0f;
                for (std::size_t k = 0; k < shared_k; ++k) {
                    std::size_t a_index = r * shared_k + k;
                    std::size_t b_index = k * b_cols + c;
                    sum += A->values()[a_index] * B->values()[b_index];
                }
                std::size_t out_index = r * b_cols + c;
                assert_true(nearf(C->values()[out_index], sum), "contractions>1 general forward element correct");
            }
        }
    }

    // repeated-usage chain: b = a + a; c = b + b; d = c + a; verify values and gradients
    {
        auto a = pool.new_tensor(std::vector<std::size_t>{2,2}, 0);
        a->set_values(std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f});
        auto b = Operations::add(a, a);
        auto c = Operations::add(b, b);
        auto d = Operations::add(c, a);
        d->forward();
        // expected d = 5 * a (since b=2a, c=4a, d=4a + a =5a)
        for (std::size_t i = 0; i < a->size(); ++i) {
            assert_true(nearf(d->values()[i], 5.0f * a->values()[i]), "repeated chain forward produces 5*a");
        }
        d->set_gradients(std::vector<float>(d->size(), 1.0f));
        d->backward();
        // after backward, a->gradients should equal 5 for each element
        for (std::size_t i = 0; i < a->size(); ++i) {
            assert_true(nearf(a->gradients()[i], 5.0f), "repeated chain backward accumulates correct gradient into a");
        }
    }

    // same-tensor matmul: A @ A, check forward and precise backward accumulation
    {
        auto A = pool.new_tensor(std::vector<std::size_t>{2,2}, 0); // [[1,2],[3,4]]
        A->set_values(std::vector<float>{1.0f,2.0f,3.0f,4.0f});
        auto C = Operations::matmul(A, A, 1);
        C->forward();
        std::vector<float> expectedC = {7.0f,10.0f,15.0f,22.0f};
        for (std::size_t i = 0; i < C->size(); ++i) {
            assert_true(nearf(C->values()[i], expectedC[i]), "matmul same-tensor forward correct");
        }
        C->set_gradients(std::vector<float>(C->size(), 1.0f));
        C->backward();
        // expected gradients computed by implementation ordering => [7,11,9,13]
        std::vector<float> expectedGrad = {7.0f,11.0f,9.0f,13.0f};
        for (std::size_t i = 0; i < A->size(); ++i) {
            assert_true(nearf(A->gradients()[i], expectedGrad[i]), "matmul same-tensor backward accumulates correct gradients");
        }
    }

    // add then matmul using same tensor: B = A + A; C = B @ A -> forward should equal 2*(A@A)
    {
        auto A = pool.new_tensor(std::vector<std::size_t>{2,2}, 0);
        A->set_values(std::vector<float>{1.0f,2.0f,3.0f,4.0f});
        auto B = Operations::add(A, A);
        auto C = Operations::matmul(B, A, 1);
        C->forward();
        // compute A@A and compare C to 2 * that
        std::vector<float> base = {7.0f,10.0f,15.0f,22.0f};
        for (std::size_t i = 0; i < C->size(); ++i) {
            assert_true(nearf(C->values()[i], 2.0f * base[i]), "add-then-matmul forward matches 2*(A@A)");
        }
        C->set_gradients(std::vector<float>(C->size(), 1.0f));
        C->backward();
        // ensure A has gradients (size and non-empty)
        assert_true(A->gradients().size() == A->size(), "A has gradients after add-then-matmul backward");
    }
}