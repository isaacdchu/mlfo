#include <gtest/gtest.h>

#include <cstddef>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "dataset/dataset_helpers.hpp"
#include "helpers.hpp"

namespace {

using Dataset = mlfo::dataset::Dataset<float>;
using DatasetTest = mlfo::tests::FileTest;
using mlfo::tests::parse;

} // namespace

TEST_F(DatasetTest, MissingFileThrows) {
    EXPECT_THROW(Dataset("mlfo_missing_file.txt", 1, parse), std::runtime_error);
}

TEST_F(DatasetTest, ZeroChunkSizeThrows) {
    write("1 | 2\n");
    EXPECT_THROW(Dataset(path_, 0, parse), std::invalid_argument);
}

TEST_F(DatasetTest, NullTransformThrows) {
    write("1 | 2\n");
    EXPECT_THROW(Dataset(path_, 1, nullptr), std::invalid_argument);
}

TEST_F(DatasetTest, SizeSkipsEmptyLines) {
    write("1 | 2\n\n3 | 4\r\n\r\n5 | 6");
    Dataset dataset(path_, 2, parse);
    EXPECT_EQ(dataset.size(), 3u);
}

TEST_F(DatasetTest, GetAppliesTransform) {
    write("1 2 | 3\n4 5 | 6\n");
    Dataset dataset(path_, 1, parse);
    const auto& [x, y] = dataset[1];
    EXPECT_EQ(x, (std::vector<float>{4, 5}));
    EXPECT_EQ(y, (std::vector<float>{6}));
}

TEST_F(DatasetTest, GetIntoSample) {
    write("1 2 | 3\n4 5 | 6\n");
    Dataset dataset(path_, 1, parse);
    Dataset::Sample sample;
    dataset.get(0, sample);
    EXPECT_EQ(sample.first, (std::vector<float>{1, 2}));
    EXPECT_EQ(sample.second, (std::vector<float>{3}));
}

TEST_F(DatasetTest, GetReusesSample) {
    write("1 2 | 3\n4 5 | 6\n");
    Dataset dataset(path_, 1, parse);
    const auto& first = dataset[0];
    const float* inputs = first.first.data();
    const float* labels = first.second.data();
    const auto& second = dataset[1];
    EXPECT_EQ(&first, &second);
    // same sizes, so the transform's clear() and push_back() keep the buffers
    EXPECT_EQ(second.first.data(), inputs);
    EXPECT_EQ(second.second.data(), labels);
}

TEST_F(DatasetTest, GetOutOfRangeThrows) {
    write("1 | 2\n");
    Dataset dataset(path_, 1, parse);
    EXPECT_THROW(dataset.get(1), std::out_of_range);
}

TEST_F(DatasetTest, StripsWindowsLineEndings) {
    std::string seen;
    write("1 | 2\r\n");
    Dataset dataset(path_, 1, [&seen](const std::string& line, Dataset::Sample& sample) {
        seen = line;
        parse(line, sample);
    });
    dataset.get(0);
    EXPECT_EQ(seen, "1 | 2");
}

TEST_F(DatasetTest, RandomAccessAcrossChunks) {
    // 7 samples with chunks of 3, accessed out of order so every chunk is loaded more than once
    std::string contents = mlfo::tests::numbered_lines(7);
    contents.insert(contents.find("4 |"), "\n");
    write(contents);
    Dataset dataset(path_, 3, parse);
    ASSERT_EQ(dataset.size(), 7u);
    for (std::size_t i : {6u, 0u, 4u, 2u, 5u, 1u, 3u, 6u}) {
        const auto& [x, y] = dataset.get(i);
        EXPECT_EQ(x, (std::vector<float>{static_cast<float>(i)}));
        EXPECT_EQ(y, (std::vector<float>{static_cast<float>(i * 10)}));
    }
}

TEST_F(DatasetTest, IteratesInOrder) {
    write("1 | 2\n3 | 4\n5 | 6\n");
    Dataset dataset(path_, 2, parse);
    std::vector<float> inputs;
    std::vector<float> labels;
    for (const auto& [x, y] : dataset) {
        inputs.push_back(x[0]);
        labels.push_back(y[0]);
    }
    EXPECT_EQ(inputs, (std::vector<float>{1, 3, 5}));
    EXPECT_EQ(labels, (std::vector<float>{2, 4, 6}));
}

TEST_F(DatasetTest, FeedsModel) {
    write("1 2 3 | 0 1 0 0\n");
    Dataset dataset(path_, 1, parse);
    std::vector<std::unique_ptr<mlfo::layer::Layer<float>>> layers;
    layers.push_back(std::make_unique<mlfo::layer::DenseLayer<float>>(3, 4));
    mlfo::model::Model<float> model(
        std::move(layers), std::make_unique<mlfo::layer::loss::MSELoss<float>>(4)
    );
    model.build();
    for (const auto& [x, y] : dataset) {
        model.set_inputs({x});
        model.set_labels({y});
        model.forward();
    }
    // zero parameters give zero outputs, so the loss is 0.5 * mean(y^2)
    EXPECT_FLOAT_EQ(model.outputs().back().get().values()[0], 0.125f);
}
