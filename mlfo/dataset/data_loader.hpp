#ifndef DATA_LOADER_HPP
#define DATA_LOADER_HPP

#include <algorithm>
#include <array>
#include <condition_variable>
#include <cstddef>
#include <exception>
#include <iterator>
#include <mutex>
#include <stop_token>
#include <thread>
#include <vector>

#include "../misc/concepts.hpp"
#include "dataset.hpp"

namespace mlfo::dataset {

// iterates over a dataset while a worker thread loads and transforms the next chunk, e.g.
//     auto loader = DataLoader<float>(dataset);
//     for (const auto& [x, y] : loader) {
//         model.set_inputs({x});
//         model.set_labels({y});
//         ...
//     }
// the chunks are double buffered: the worker fills one buffer of dataset.chunk_size() samples
// while the other is being read, so the threads only synchronize once per chunk
// the samples in each buffer are reused between chunks and epochs, so nothing is allocated once
// both buffers have been filled
// the dataset must outlive the loader, and must not be used elsewhere while the loader iterates
// since the worker reads from it. the transform runs on the worker thread
template <mlfo::misc::Number T>
class DataLoader {
public:
    using Sample = typename Dataset<T>::Sample;

    class Iterator {
    private:
        DataLoader* loader_;
        std::size_t index_;
    public:
        using iterator_category = std::input_iterator_tag;
        using value_type = Sample;
        using difference_type = std::ptrdiff_t;

        Iterator() :
        loader_(nullptr),
        index_(0)
        {
            // nothing
        }

        Iterator(DataLoader& loader, std::size_t index) :
        loader_(&loader),
        index_(index)
        {
            // nothing
        }

        // valid until the iterator is incremented
        const Sample& operator*() const {
            return loader_->sample(index_);
        }

        Iterator& operator++() {
            index_++;
            loader_->advance(index_);
            return *this;
        }

        void operator++(int) {
            ++*this;
        }

        bool operator==(const Iterator& other) const {
            return loader_ == other.loader_ && index_ == other.index_;
        }
    };
private:
    struct Buffer {
        std::vector<Sample> samples;
        // set by the worker once samples holds a chunk, cleared by the reader once it is done
        bool full;
    };

    Dataset<T>& dataset_;
    std::size_t chunk_size_;
    std::array<Buffer, 2> buffers_;
    // guards buffers_[i].full and error_
    std::mutex mutex_;
    std::condition_variable_any condition_;
    // set by the worker if the dataset throws, rethrown by the reader
    std::exception_ptr error_;
    // declared last so it is stopped and joined before the members it uses are destroyed
    std::jthread worker_;

    Buffer& buffer(std::size_t chunk) {
        // chunk i is loaded into buffer i % 2
        return buffers_[chunk % 2];
    }

    void load(std::stop_token stop) {
        // runs on the worker thread, loading every chunk in order into alternating buffers
        try {
            const std::size_t chunks = (dataset_.size() + chunk_size_ - 1) / chunk_size_;
            for (std::size_t chunk = 0; chunk < chunks; chunk++) {
                Buffer& current = buffer(chunk);
                {
                    // wait until the reader is done with the chunk that was in this buffer
                    std::unique_lock lock(mutex_);
                    if (
                        !condition_.wait(
                            lock,
                            stop,
                            [&current]() -> bool {
                                return !current.full;
                            }
                        )
                    ) {
                        return;
                    }
                }
                // the reader doesn't touch a buffer that isn't full, so it is filled without the lock
                const std::size_t start = chunk * chunk_size_;
                const std::size_t count = std::min(chunk_size_, dataset_.size() - start);
                for (std::size_t i = 0; i < count; i++) {
                    if (stop.stop_requested()) {
                        return;
                    }
                    dataset_.get(start + i, current.samples[i]);
                }
                {
                    std::lock_guard lock(mutex_);
                    current.full = true;
                }
                condition_.notify_all();
            }
        } catch (...) {
            {
                std::lock_guard lock(mutex_);
                error_ = std::current_exception();
            }
            condition_.notify_all();
        }
    }

    void acquire(std::size_t chunk) {
        // waits until the worker has loaded chunk, rethrowing the worker's exception if it failed
        std::unique_lock lock(mutex_);
        Buffer& current = buffer(chunk);
        condition_.wait(
            lock,
            [this, &current]() -> bool {
                return current.full || error_;
            }
        );
        if (!current.full) {
            std::rethrow_exception(error_);
        }
    }

    void release(std::size_t chunk) {
        // hands the buffer holding chunk back to the worker
        {
            std::lock_guard lock(mutex_);
            buffer(chunk).full = false;
        }
        condition_.notify_all();
    }

    void stop() {
        if (worker_.joinable()) {
            worker_.request_stop();
            worker_.join();
        }
    }

    void start() {
        // stops the previous epoch, if any, and starts loading from the first chunk
        stop();
        for (Buffer& current : buffers_) {
            current.full = false;
        }
        error_ = nullptr;
        if (dataset_.size() == 0) {
            return;
        }
        worker_ = std::jthread([this](std::stop_token stop) {
            load(stop);
        });
        acquire(0);
    }

    const Sample& sample(std::size_t index) {
        return buffer(index / chunk_size_).samples[index % chunk_size_];
    }

    void advance(std::size_t index) {
        // called after the iterator moves to index. when it crosses into a new chunk, the previous
        // chunk's buffer is handed back to the worker and the new chunk is waited for
        if (index % chunk_size_ != 0) {
            return;
        }
        const std::size_t chunk = index / chunk_size_;
        release(chunk - 1);
        if (index < dataset_.size()) {
            acquire(chunk);
        }
    }
public:
    explicit DataLoader(Dataset<T>& dataset) :
    dataset_(dataset),
    chunk_size_(dataset.chunk_size())
    {
        // the buffers match the dataset's chunks, so the dataset reads each chunk from the file once
        for (Buffer& current : buffers_) {
            current.samples.resize(chunk_size_);
            current.full = false;
        }
    }

    // the worker holds a pointer to this loader, so it cannot be copied or moved
    DataLoader(const DataLoader&) = delete;
    DataLoader& operator=(const DataLoader&) = delete;

    ~DataLoader() {
        stop();
    }

    std::size_t size() const {
        return dataset_.size();
    }

    // starts a new epoch, restarting the worker if the previous epoch was not finished
    Iterator begin() {
        start();
        return Iterator(*this, 0);
    }

    Iterator end() {
        return Iterator(*this, dataset_.size());
    }
};

} // namespace mlfo::dataset

#endif // DATA_LOADER_HPP
