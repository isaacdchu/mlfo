#ifndef DATA_LOADER_HPP
#define DATA_LOADER_HPP

#include <algorithm>
#include <array>
#include <condition_variable>
#include <cstddef>
#include <exception>
#include <iterator>
#include <mutex>
#include <stdexcept>
#include <stop_token>
#include <thread>
#include <vector>

#include "../misc/concepts.hpp"
#include "dataset.hpp"

namespace mlfo::dataset {

// iterates over a dataset in batches while a worker thread loads and transforms the next batch, e.g.
//     auto loader = DataLoader<float>(dataset, batch_size);
//     auto model = Model<float>(input_shape, std::move(layers), std::move(loss), batch_size);
//     for (const auto& [x, y, count] : loader) {
//         model.set_inputs({x});
//         model.set_labels({y});
//         ...
//     }
// every batch holds batch_size samples except the last, which holds the remainder when the
// dataset size is not a multiple of batch_size. the model accepts any batch up to the size it was
// built with, so the loader and the model should be given the same batch size
// the batches are double buffered: the worker fills one buffer while the other is being read, so
// the threads only synchronize once per batch
// the buffers are reused between batches and epochs, so nothing is allocated once both buffers
// have been filled with a full batch
// the dataset must outlive the loader, and must not be used elsewhere while the loader iterates
// since the worker reads from it. the transform runs on the worker thread
template <mlfo::misc::Number T>
class DataLoader {
public:
    using Sample = typename Dataset<T>::Sample;

    // count samples, flattened and concatenated in order, which is the layout Model::set_inputs()
    // and Model::set_labels() take
    struct Batch {
        std::vector<T> inputs;
        std::vector<T> labels;
        std::size_t count;
    };

    class Iterator {
    private:
        DataLoader* loader_;
        std::size_t index_;
    public:
        using iterator_category = std::input_iterator_tag;
        using value_type = Batch;
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
        const Batch& operator*() const {
            return loader_->batch(index_);
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
        Batch batch;
        // set by the worker once batch holds a batch, cleared by the reader once it is done
        bool full;
    };

    Dataset<T>& dataset_;
    std::size_t batch_size_;
    std::array<Buffer, 2> buffers_;
    // only used by the worker, each sample is transformed into it then appended to a batch
    Sample sample_;
    // guards buffers_[i].full and error_
    std::mutex mutex_;
    std::condition_variable_any condition_;
    // set by the worker if the dataset throws, rethrown by the reader
    std::exception_ptr error_;
    // declared last so it is stopped and joined before the members it uses are destroyed
    std::jthread worker_;

    Buffer& buffer(std::size_t index) {
        // batch i is loaded into buffer i % 2
        return buffers_[index % 2];
    }

    void fill(Batch& batch, std::size_t start, std::size_t count, std::stop_token stop) {
        // transforms count samples from start and appends them to batch. returns early if stopped
        // clear() keeps the vectors' capacity, so this doesn't allocate once they are large enough
        batch.inputs.clear();
        batch.labels.clear();
        batch.count = count;
        std::size_t input_size = 0;
        std::size_t label_size = 0;
        for (std::size_t i = 0; i < count; i++) {
            if (stop.stop_requested()) {
                return;
            }
            dataset_.get(start + i, sample_);
            if (i == 0) {
                input_size = sample_.first.size();
                label_size = sample_.second.size();
            } else if (sample_.first.size() != input_size || sample_.second.size() != label_size) {
                // the batch is flattened, so a sample's size is what separates it from the next
                throw std::runtime_error(
                    "[mlfo::dataset::DataLoader] Samples in a batch must all have the same size"
                );
            }
            batch.inputs.insert(batch.inputs.end(), sample_.first.begin(), sample_.first.end());
            batch.labels.insert(batch.labels.end(), sample_.second.begin(), sample_.second.end());
        }
    }

    void load(std::stop_token stop) {
        // runs on the worker thread, loading every batch in order into alternating buffers
        try {
            const std::size_t batches = size();
            for (std::size_t index = 0; index < batches; index++) {
                Buffer& current = buffer(index);
                {
                    // wait until the reader is done with the batch that was in this buffer
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
                // the last batch holds whatever is left when the dataset doesn't divide evenly
                const std::size_t start = index * batch_size_;
                const std::size_t count = std::min(batch_size_, dataset_.size() - start);
                fill(current.batch, start, count, stop);
                if (stop.stop_requested()) {
                    return;
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

    void acquire(std::size_t index) {
        // waits until the worker has loaded batch index, rethrowing the worker's exception if it failed
        std::unique_lock lock(mutex_);
        Buffer& current = buffer(index);
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

    void release(std::size_t index) {
        // hands the buffer holding batch index back to the worker
        {
            std::lock_guard lock(mutex_);
            buffer(index).full = false;
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
        // stops the previous epoch, if any, and starts loading from the first batch
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

    const Batch& batch(std::size_t index) {
        return buffer(index).batch;
    }

    void advance(std::size_t index) {
        // called after the iterator moves to batch index. the previous batch's buffer is handed
        // back to the worker and the new batch is waited for
        release(index - 1);
        if (index < size()) {
            acquire(index);
        }
    }
public:
    // the samples are read in order, so the dataset reads each of its chunks from the file once
    DataLoader(Dataset<T>& dataset, std::size_t batch_size) :
    dataset_(dataset),
    batch_size_(batch_size)
    {
        if (batch_size_ == 0) {
            throw std::invalid_argument(
                "[mlfo::dataset::DataLoader] Batch size must be greater than zero"
            );
        }
        for (Buffer& current : buffers_) {
            current.batch.count = 0;
            current.full = false;
        }
    }

    // the worker holds a pointer to this loader, so it cannot be copied or moved
    DataLoader(const DataLoader&) = delete;
    DataLoader& operator=(const DataLoader&) = delete;

    ~DataLoader() {
        stop();
    }

    // the number of batches in an epoch, including a smaller last batch
    std::size_t size() const {
        return (dataset_.size() + batch_size_ - 1) / batch_size_;
    }

    std::size_t batch_size() const {
        return batch_size_;
    }

    // starts a new epoch, restarting the worker if the previous epoch was not finished
    Iterator begin() {
        start();
        return Iterator(*this, 0);
    }

    Iterator end() {
        return Iterator(*this, size());
    }
};

} // namespace mlfo::dataset

#endif // DATA_LOADER_HPP
