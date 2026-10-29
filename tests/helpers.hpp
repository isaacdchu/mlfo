#ifndef TESTS_HELPERS_HPP
#define TESTS_HELPERS_HPP

#include <cstddef>
#include <vector>

#include <mlfo>

namespace mlfo::tests {

// gives tests write access to a tensor's gradients, which is otherwise only
// available to operations
template <mlfo::misc::Number T>
struct TensorAccess : public mlfo::tensor::Operation<T> {
    static std::vector<T>& gradients(mlfo::tensor::Tensor<T>& tensor) {
        return mlfo::tensor::Operation<T>::gradients(tensor);
    }
};

// records the fans it was called with and fills the tensor with value, so tests can check
// which initializer a layer used for each parameter and what fans it passed
template <mlfo::misc::Number T>
class RecordingInitializer : public mlfo::initializer::Constant<T> {
public:
    struct Call {
        std::vector<std::size_t> shape;
        std::size_t fan_in;
        std::size_t fan_out;
    };

    // mutable since initialize() is const, as initializers do not change when used
    mutable std::vector<Call> calls;

    using mlfo::initializer::Constant<T>::Constant;

    void initialize(
        mlfo::tensor::Tensor<T>& tensor,
        std::size_t fan_in,
        std::size_t fan_out
    ) const override {
        calls.push_back({tensor.shape(), fan_in, fan_out});
        mlfo::initializer::Constant<T>::initialize(tensor, fan_in, fan_out);
    }
};

} // namespace mlfo::tests

#endif // TESTS_HELPERS_HPP
