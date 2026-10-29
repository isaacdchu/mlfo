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
    for (const auto& [x, y] : loader) {
        EXPECT_EQ(y[0], x[0] * 10);
        inputs.push_back(x[0]);
    }
    return inputs;
}

std::vector<float> range(std::size_t count) {
    std::vector<float> values;
    for (std::size_t i = 0; i < count; i++) {
        values.push_back(static_cast<float>(i));
    }
    return values;
}

} // namespace

TEST_F(DataLoaderTest, EmptyDataset) {
    write("\n\n");
    Dataset dataset(path_, 2, parse);
    DataLoader loader(dataset);
    EXPECT_TRUE(epoch(loader).empty());
}

TEST_F(DataLoaderTest, IteratesInOrder) {
    // sizes smaller than, equal to, a multiple of, and not a multiple of the chunk size
    for (std::size_t count : {1u, 2u, 3u, 6u, 7u}) {
        write(mlfo::tests::numbered_lines(count));
        Dataset dataset(path_, 3, parse);
        DataLoader loader(dataset);
        EXPECT_EQ(loader.size(), count);
        EXPECT_EQ(epoch(loader), range(count)) << "count " << count;
    }
}

TEST_F(DataLoaderTest, SeveralEpochs) {
    write(mlfo::tests::numbered_lines(10));
    Dataset dataset(path_, 4, parse);
    DataLoader loader(dataset);
    for (int i = 0; i < 3; i++) {
        EXPECT_EQ(epoch(loader), range(10));
    }
}

TEST_F(DataLoaderTest, ReusesSamplesAcrossEpochs) {
    write(mlfo::tests::numbered_lines(8));
    Dataset dataset(path_, 2, parse);
    DataLoader loader(dataset);
    // every sample has the same size, so after the first epoch fills both buffers the same
    // vector storage is used for every later sample
    std::set<const float*> first;
    for (const auto& [x, y] : loader) {
        first.insert(x.data());
    }
    std::set<const float*> second;
    for (const auto& [x, y] : loader) {
        second.insert(x.data());
    }
    EXPECT_EQ(first.size(), 4u);
    EXPECT_EQ(first, second);
}

TEST_F(DataLoaderTest, BreakThenRestart) {
    write(mlfo::tests::numbered_lines(20));
    Dataset dataset(path_, 3, parse);
    DataLoader loader(dataset);
    std::size_t seen = 0;
    for (const auto& sample : loader) {
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
        DataLoader loader(dataset);
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
    DataLoader loader(dataset);
    std::vector<float> inputs;
    EXPECT_THROW(
        {
            for (const auto& [x, y] : loader) {
                inputs.push_back(x[0]);
            }
        },
        std::runtime_error
    );
    // the chunks before the failing one are still delivered
    EXPECT_EQ(inputs, range(6));
}

TEST_F(DataLoaderTest, TransformRunsOnWorkerThread) {
    write(mlfo::tests::numbered_lines(5));
    std::set<std::thread::id> threads;
    Dataset dataset(path_, 2, [&threads](const std::string& line, Dataset::Sample& sample) {
        // only the worker calls the transform, so this doesn't race
        threads.insert(std::this_thread::get_id());
        parse(line, sample);
    });
    DataLoader loader(dataset);
    EXPECT_EQ(epoch(loader), range(5));
    ASSERT_EQ(threads.size(), 1u);
    EXPECT_NE(*threads.begin(), std::this_thread::get_id());
}
