#include <gtest/gtest.h>

#include <cstddef>
#include <set>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "dataset/dataset_helpers.hpp"
#include "helpers.hpp"

namespace {

using Dataset = mlfo::dataset::Dataset<float>;
using DataLoader = mlfo::dataset::DataLoader<float>;
using DataLoaderTest = mlfo::tests::FileTest;
using mlfo::tests::parse;

// collects the inputs of every sample in one epoch
std::vector<float> epoch(DataLoader& loader) {
    std::vector<float> inputs;
    for (const auto& [x, y, count] : loader) {
        EXPECT_EQ(x.size(), count);
        EXPECT_EQ(y.size(), count);
        for (std::size_t i = 0; i < count; i++) {
            EXPECT_EQ(y[i], x[i] * 10);
            inputs.push_back(x[i]);
        }
    }
    return inputs;
}

// the number of samples in each batch of one epoch
std::vector<std::size_t> counts(DataLoader& loader) {
    std::vector<std::size_t> result;
    for (const auto& batch : loader) {
        result.push_back(batch.count);
    }
    return result;
}

std::vector<float> range(std::size_t count) {
    std::vector<float> values;
    for (std::size_t i = 0; i < count; i++) {
        values.push_back(static_cast<float>(i));
    }
    return values;
}

} // namespace

TEST_F(DataLoaderTest, ZeroBatchSizeThrows) {
    write("1 | 10\n");
    Dataset dataset(path_, 2, parse);
    EXPECT_THROW(DataLoader(dataset, 0), std::invalid_argument);
}

TEST_F(DataLoaderTest, EmptyDataset) {
    write("\n\n");
    Dataset dataset(path_, 2, parse);
    DataLoader loader(dataset, 3);
    EXPECT_EQ(loader.size(), 0u);
    EXPECT_TRUE(epoch(loader).empty());
}

TEST_F(DataLoaderTest, IteratesInOrder) {
    // sizes smaller than, equal to, a multiple of, and not a multiple of the batch size
    for (std::size_t count : {1u, 3u, 6u, 7u}) {
        for (std::size_t batch_size : {1u, 3u}) {
            write(mlfo::tests::numbered_lines(count));
            // a chunk size that doesn't line up with the batches
            Dataset dataset(path_, 2, parse);
            DataLoader loader(dataset, batch_size);
            EXPECT_EQ(loader.size(), (count + batch_size - 1) / batch_size);
            EXPECT_EQ(epoch(loader), range(count)) << "count " << count << " batch " << batch_size;
        }
    }
}

TEST_F(DataLoaderTest, LastBatchHoldsRemainder) {
    write(mlfo::tests::numbered_lines(7));
    Dataset dataset(path_, 4, parse);
    DataLoader loader(dataset, 3);
    EXPECT_EQ(counts(loader), (std::vector<std::size_t>{3, 3, 1}));
}

TEST_F(DataLoaderTest, BatchLargerThanDataset) {
    write(mlfo::tests::numbered_lines(2));
    Dataset dataset(path_, 4, parse);
    DataLoader loader(dataset, 5);
    EXPECT_EQ(counts(loader), (std::vector<std::size_t>{2}));
    EXPECT_EQ(epoch(loader), range(2));
}

TEST_F(DataLoaderTest, SeveralEpochs) {
    write(mlfo::tests::numbered_lines(10));
    Dataset dataset(path_, 4, parse);
    DataLoader loader(dataset, 3);
    for (int i = 0; i < 3; i++) {
        EXPECT_EQ(epoch(loader), range(10));
    }
}

TEST_F(DataLoaderTest, ReusesBuffersAcrossEpochs) {
    write(mlfo::tests::numbered_lines(8));
    Dataset dataset(path_, 2, parse);
    DataLoader loader(dataset, 2);
    // every batch is full, so after the first epoch fills both buffers the same vector storage
    // is used for every later batch
    std::set<const float*> first;
    for (const auto& [x, y, count] : loader) {
        first.insert(x.data());
    }
    std::set<const float*> second;
    for (const auto& [x, y, count] : loader) {
        second.insert(x.data());
    }
    EXPECT_EQ(first.size(), 2u);
    EXPECT_EQ(first, second);
}

TEST_F(DataLoaderTest, BreakThenRestart) {
    write(mlfo::tests::numbered_lines(20));
    Dataset dataset(path_, 3, parse);
    DataLoader loader(dataset, 2);
    std::size_t seen = 0;
    for (const auto& batch : loader) {
        if (++seen == 4) {
            break;
        }
    }
    // the worker is blocked on a full buffer, begin() must stop it and start over
    EXPECT_EQ(epoch(loader), range(20));
}

TEST_F(DataLoaderTest, DestroyedMidEpoch) {
    write(mlfo::tests::numbered_lines(20));
    Dataset dataset(path_, 3, parse);
    {
        DataLoader loader(dataset, 2);
        auto it = loader.begin();
        ++it;
        // the destructor must stop the worker instead of waiting for the epoch to finish
    }
    SUCCEED();
}

TEST_F(DataLoaderTest, TransformExceptionIsRethrown) {
    write(mlfo::tests::numbered_lines(10));
    Dataset dataset(path_, 3, [](const std::string& line, Dataset::Sample& sample) {
        parse(line, sample);
        if (sample.first[0] == 7) {
            throw std::runtime_error("bad line");
        }
    });
    DataLoader loader(dataset, 3);
    std::vector<float> inputs;
    EXPECT_THROW(
        {
            for (const auto& [x, y, count] : loader) {
                inputs.insert(inputs.end(), x.begin(), x.end());
            }
        },
        std::runtime_error
    );
    // the batches before the failing one are still delivered
    EXPECT_EQ(inputs, range(6));
}

TEST_F(DataLoaderTest, MismatchedSampleSizesThrow) {
    // the second sample has two inputs, so the batch can't be split back into samples
    write("1 | 10\n2 3 | 20\n");
    Dataset dataset(path_, 2, parse);
    DataLoader loader(dataset, 2);
    EXPECT_THROW(epoch(loader), std::runtime_error);
}

TEST_F(DataLoaderTest, TransformRunsOnWorkerThread) {
    write(mlfo::tests::numbered_lines(5));
    std::set<std::thread::id> threads;
    Dataset dataset(path_, 2, [&threads](const std::string& line, Dataset::Sample& sample) {
        // only the worker calls the transform, so this doesn't race
        threads.insert(std::this_thread::get_id());
        parse(line, sample);
    });
    DataLoader loader(dataset, 2);
    EXPECT_EQ(epoch(loader), range(5));
    ASSERT_EQ(threads.size(), 1u);
    EXPECT_NE(*threads.begin(), std::this_thread::get_id());
}
