#ifndef TESTS_HELPERS_HPP
#define TESTS_HELPERS_HPP

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

} // namespace mlfo::tests

#endif // TESTS_HELPERS_HPP
