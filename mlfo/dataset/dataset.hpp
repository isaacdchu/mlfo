#ifndef DATASET_HPP
#define DATASET_HPP

#include <algorithm>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iterator>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "../misc/concepts.hpp"

namespace mlfo::dataset {

// a dataset backed by a file with one sample per line, e.g.
//     auto dataset = Dataset<float>("data/train.csv", 1024,
//         [](const std::string& line, Dataset<float>::Sample& sample) {
//             sample.first.clear();
//             sample.second.clear();
//             ... parse line, push_back into sample.first and sample.second ...
//         }
//     );
//     for (const auto& [x, y] : dataset) {
//         model.set_inputs({x});
//         model.set_labels({y});
//         ...
//     }
// only chunk_size lines are kept in memory at a time, the transform is called on every access
// so it may produce different samples each time, e.g. for random augmentation
// the transform fills an existing sample instead of returning a new one, so the sample's vectors
// keep their capacity and nothing is allocated once they are large enough
template <mlfo::misc::Number T>
class Dataset {
public:
    // first is the input data, second is the label
    using Sample = std::pair<std::vector<T>, std::vector<T>>;
    // fills the sample from one line of the file. the sample holds whatever the previous call
    // left in it, so the transform must clear or overwrite both vectors
    using Transform = std::function<void(const std::string&, Sample&)>;

    class Iterator {
    private:
        Dataset* dataset_;
        std::size_t index_;
    public:
        using iterator_category = std::input_iterator_tag;
        using value_type = Sample;
        using difference_type = std::ptrdiff_t;

        Iterator() :
        dataset_(nullptr),
        index_(0)
        {
            // nothing
        }

        Iterator(Dataset& dataset, std::size_t index) :
        dataset_(&dataset),
        index_(index)
        {
            // nothing
        }

        // valid until the next access to the dataset
        const Sample& operator*() const {
            return dataset_->get(index_);
        }

        Iterator& operator++() {
            index_++;
            return *this;
        }

        void operator++(int) {
            index_++;
        }

        bool operator==(const Iterator& other) const {
            return dataset_ == other.dataset_ && index_ == other.index_;
        }
    };
private:
    std::filesystem::path path_;
    std::size_t chunk_size_;
    Transform transform_;
    // byte offset of the start of each sample's line, set by the constructor
    std::vector<std::streamoff> offsets_;
    // raw lines of the loaded chunk, which starts at sample chunk_start_
    std::vector<std::string> chunk_;
    std::size_t chunk_start_;
    // reused by get(index), so it doesn't allocate on every access
    Sample sample_;

    static void strip_carriage_return(std::string& line) {
        // the file is read in binary mode, so windows line endings leave a trailing \r
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
    }

    std::ifstream open() const {
        std::ifstream file(path_, std::ios::binary);
        if (!file) {
            throw std::runtime_error(
                "[mlfo::dataset::Dataset] Could not open file " + path_.string()
            );
        }
        return file;
    }

    void load_chunk(std::size_t index) {
        // loads the chunk containing sample index
        chunk_start_ = index - index % chunk_size_;
        const std::size_t count = std::min(chunk_size_, offsets_.size() - chunk_start_);
        chunk_.clear();
        std::ifstream file = open();
        std::string line;
        for (std::size_t i = 0; i < count; i++) {
            // seek to each sample since empty lines between samples are skipped
            file.seekg(offsets_[chunk_start_ + i]);
            if (!std::getline(file, line)) {
                throw std::runtime_error(
                    "[mlfo::dataset::Dataset::load_chunk] File changed after it was indexed"
                );
            }
            strip_carriage_return(line);
            chunk_.push_back(std::move(line));
        }
    }
public:
    Dataset(
        std::filesystem::path path,
        std::size_t chunk_size,
        Transform transform
    ) :
    path_(std::move(path)),
    chunk_size_(chunk_size),
    transform_(std::move(transform)),
    chunk_start_(0)
    {
        if (chunk_size_ == 0) {
            throw std::invalid_argument(
                "[mlfo::dataset::Dataset] Chunk size must be greater than zero"
            );
        }
        if (!transform_) {
            throw std::invalid_argument("[mlfo::dataset::Dataset] Transform cannot be null");
        }
        // index the start of every non-empty line, so any sample can be read without
        // keeping the whole file in memory
        std::ifstream file = open();
        std::string line;
        std::streamoff offset = 0;
        while (std::getline(file, line)) {
            const std::streamoff next = offset + static_cast<std::streamoff>(line.size()) + 1;
            strip_carriage_return(line);
            if (!line.empty()) {
                offsets_.push_back(offset);
            }
            offset = next;
        }
    }

    std::size_t size() const {
        return offsets_.size();
    }

    std::size_t chunk_size() const {
        return chunk_size_;
    }

    void get(std::size_t index, Sample& sample) {
        // transforms the sample at index into sample, loading its chunk from the file if needed
        if (index >= offsets_.size()) {
            throw std::out_of_range("[mlfo::dataset::Dataset::get] Index out of range");
        }
        if (chunk_.empty() || index < chunk_start_ || index >= chunk_start_ + chunk_.size()) {
            load_chunk(index);
        }
        transform_(chunk_[index - chunk_start_], sample);
    }

    const Sample& get(std::size_t index) {
        // the returned sample is owned by the dataset and is overwritten by the next access
        get(index, sample_);
        return sample_;
    }

    const Sample& operator[](std::size_t index) {
        return get(index);
    }

    Iterator begin() {
        return Iterator(*this, 0);
    }

    Iterator end() {
        return Iterator(*this, offsets_.size());
    }
};

} // namespace mlfo::dataset

#endif // DATASET_HPP
