#include <gtest/gtest.h>

#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

#include "helpers.hpp"

using mlfo::tensor::Tensor;
using mlfo::tests::TensorAccess;

using Shape = std::vector<std::size_t>;

TEST(TensorConstructor, ZeroBatchSizeThrows) {
    EXPECT_THROW(Tensor<float>(0, Shape{2, 3}), std::invalid_argument);
}

TEST(TensorConstructor, EmptyShapeThrows) {
    EXPECT_THROW(Tensor<float>(1, Shape{}), std::invalid_argument);
}

TEST(TensorConstructor, ZeroDimensionThrows) {
    EXPECT_THROW(Tensor<float>(1, Shape{0}), std::invalid_argument);
    EXPECT_THROW(Tensor<float>(1, Shape{0, 3}), std::invalid_argument);
    EXPECT_THROW(Tensor<float>(1, Shape{2, 0, 4}), std::invalid_argument);
    EXPECT_THROW(Tensor<float>(1, Shape{2, 3, 0}), std::invalid_argument);
}

TEST(TensorConstructor, ValidArgumentsDoNotThrow) {
    EXPECT_NO_THROW(Tensor<float>(1, Shape{1}));
    EXPECT_NO_THROW(Tensor<float>(4, Shape{2, 3}));
}

TEST(TensorProperties, RankShapeAndBatchSize) {
    Tensor<float> tensor(5, Shape{2, 3, 4});
    EXPECT_EQ(tensor.rank(), 3u);
    EXPECT_EQ(tensor.shape(), (Shape{2, 3, 4}));
    EXPECT_EQ(tensor.batch_size(), 5u);
}

TEST(TensorProperties, RankDoesNotCountBatch) {
    Tensor<float> tensor(7, Shape{4});
    EXPECT_EQ(tensor.rank(), 1u);
}

TEST(TensorProperties, SizeIncludesBatches) {
    Tensor<float> tensor(5, Shape{2, 3, 4});
    EXPECT_EQ(tensor.size(), 5u * 2u * 3u * 4u);
    EXPECT_EQ(tensor.unbatched_size(), 2u * 3u * 4u);
}

TEST(TensorProperties, SizeWithSingleBatch) {
    Tensor<float> tensor(1, Shape{3, 2});
    EXPECT_EQ(tensor.size(), 6u);
    EXPECT_EQ(tensor.unbatched_size(), 6u);
}

TEST(TensorStrides, RowMajor3D) {
    Tensor<float> tensor(1, Shape{2, 3, 4});
    EXPECT_EQ(tensor.strides(), (Shape{12, 4, 1}));
}

TEST(TensorStrides, RowMajor1D) {
    Tensor<float> tensor(1, Shape{6});
    EXPECT_EQ(tensor.strides(), (Shape{1}));
}

TEST(TensorStrides, IndependentOfBatchSize) {
    Tensor<float> tensor(3, Shape{2, 5});
    EXPECT_EQ(tensor.strides(), (Shape{5, 1}));
}

TEST(TensorFlattenIndex, FirstBatch) {
    Tensor<float> tensor(2, Shape{2, 3, 4});
    EXPECT_EQ(tensor.flatten_index(0, {0, 0, 0}), 0u);
    EXPECT_EQ(tensor.flatten_index(0, {0, 0, 3}), 3u);
    EXPECT_EQ(tensor.flatten_index(0, {0, 2, 0}), 8u);
    EXPECT_EQ(tensor.flatten_index(0, {1, 0, 0}), 12u);
    EXPECT_EQ(tensor.flatten_index(0, {1, 2, 3}), 23u);
}

TEST(TensorFlattenIndex, IncludesBatchOffset) {
    Tensor<float> tensor(3, Shape{2, 3, 4});
    EXPECT_EQ(tensor.flatten_index(1, {0, 0, 0}), 24u);
    EXPECT_EQ(tensor.flatten_index(2, {0, 0, 0}), 48u);
    EXPECT_EQ(tensor.flatten_index(2, {1, 2, 3}), 48u + 23u);
}

TEST(TensorFlattenIndex, LastElementIsSizeMinusOne) {
    Tensor<float> tensor(3, Shape{2, 3, 4});
    EXPECT_EQ(tensor.flatten_index(2, {1, 2, 3}), tensor.size() - 1);
}

TEST(TensorStorage, ValuesAreZeroInitialized) {
    Tensor<float> tensor(3, Shape{2, 2});
    ASSERT_EQ(tensor.values().size(), 3u * 2u * 2u);
    for (float value : tensor.values()) {
        EXPECT_EQ(value, 0.0f);
    }
}

TEST(TensorStorage, GradientsAreZeroInitializedWithBatches) {
    Tensor<float> tensor(3, Shape{2, 2});
    ASSERT_EQ(tensor.gradients().size(), tensor.size());
    EXPECT_EQ(tensor.gradients().size(), 12u);
    for (float gradient : tensor.gradients()) {
        EXPECT_EQ(gradient, 0.0f);
    }
}

TEST(TensorStorage, GradientsAreWritableThroughHelper) {
    Tensor<float> tensor(1, Shape{3});
    TensorAccess<float>::gradients(tensor)[1] = 2.5f;
    EXPECT_EQ(tensor.gradients(), (std::vector<float>{0.0f, 2.5f, 0.0f}));
}

TEST(TensorSetValues, ReplacesValues) {
    Tensor<float> tensor(2, Shape{2});
    tensor.set_values({1.0f, 2.0f, 3.0f, 4.0f});
    EXPECT_EQ(tensor.values(), (std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f}));
}

TEST(TensorSetValues, ValuesMatchFlattenIndex) {
    Tensor<float> tensor(2, Shape{2, 2});
    tensor.set_values({0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f});
    EXPECT_EQ(tensor.values()[tensor.flatten_index(1, {1, 0})], 6.0f);
}

TEST(TensorSetValues, TooFewValuesThrowsAndKeepsValues) {
    Tensor<float> tensor(1, Shape{3});
    tensor.set_values({1.0f, 2.0f, 3.0f});
    EXPECT_THROW(tensor.set_values({9.0f, 9.0f}), std::invalid_argument);
    EXPECT_EQ(tensor.values(), (std::vector<float>{1.0f, 2.0f, 3.0f}));
}

TEST(TensorSetValues, TooManyValuesThrowsAndKeepsValues) {
    Tensor<float> tensor(1, Shape{3});
    tensor.set_values({1.0f, 2.0f, 3.0f});
    EXPECT_THROW(tensor.set_values({9.0f, 9.0f, 9.0f, 9.0f}), std::invalid_argument);
    EXPECT_EQ(tensor.values(), (std::vector<float>{1.0f, 2.0f, 3.0f}));
}

TEST(TensorSetValues, UnbatchedSizeOnlyThrowsForBatchedTensor) {
    Tensor<float> tensor(2, Shape{3});
    EXPECT_THROW(tensor.set_values({1.0f, 2.0f, 3.0f}), std::invalid_argument);
    EXPECT_EQ(tensor.values(), (std::vector<float>(6, 0.0f)));
}

TEST(TensorToString, ContainsShapeBatchesAndValues) {
    Tensor<float> tensor(2, Shape{2, 1});
    tensor.set_values({1.0f, 2.0f, 3.0f, 4.0f});
    const std::string text = tensor.to_string();
    EXPECT_NE(text.find("shape=[2, 1]"), std::string::npos);
    EXPECT_NE(text.find("batches=2"), std::string::npos);
    EXPECT_NE(text.find("values="), std::string::npos);
    EXPECT_NE(text.find(std::to_string(1.0f)), std::string::npos);
    EXPECT_NE(text.find(std::to_string(2.0f)), std::string::npos);
    EXPECT_NE(text.find(std::to_string(3.0f)), std::string::npos);
    EXPECT_NE(text.find(std::to_string(4.0f)), std::string::npos);
}

TEST(TensorTemplating, DoubleTensor) {
    Tensor<double> tensor(2, Shape{3});
    EXPECT_EQ(tensor.size(), 6u);
    EXPECT_EQ(tensor.gradients().size(), 6u);
    tensor.set_values({1.5, 2.5, 3.5, 4.5, 5.5, 6.5});
    EXPECT_EQ(tensor.values()[tensor.flatten_index(1, {2})], 6.5);
    EXPECT_THROW(tensor.set_values({1.0}), std::invalid_argument);
}

TEST(TensorTemplating, IntTensor) {
    Tensor<int> tensor(1, Shape{2, 2});
    EXPECT_EQ(tensor.values(), (std::vector<int>(4, 0)));
    tensor.set_values({1, 2, 3, 4});
    EXPECT_EQ(tensor.values(), (std::vector<int>{1, 2, 3, 4}));
    EXPECT_NE(tensor.to_string().find("shape=[2, 2]"), std::string::npos);
}
