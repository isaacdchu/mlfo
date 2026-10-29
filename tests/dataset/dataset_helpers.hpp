#ifndef TESTS_DATASET_HELPERS_HPP
#define TESTS_DATASET_HELPERS_HPP

#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>

#include <mlfo>

namespace mlfo::tests {

// writes contents to a file named after the running test, removed when the fixture is destroyed
class FileTest : public ::testing::Test {
protected:
    std::filesystem::path path_;

    void write(const std::string& contents) {
        const auto* info = ::testing::UnitTest::GetInstance()->current_test_info();
        path_ = std::filesystem::temp_directory_path() /
            (std::string("mlfo_") + info->test_suite_name() + "_" + info->name() + ".txt");
        std::ofstream file(path_, std::ios::binary);
        file << contents;
    }

    void TearDown() override {
        if (!path_.empty()) {
            std::filesystem::remove(path_);
        }
    }
};

// parses "x0 x1 ... | y0 y1 ..." into sample
inline void parse(const std::string& line, mlfo::dataset::Dataset<float>::Sample& sample) {
    sample.first.clear();
    sample.second.clear();
    std::istringstream stream(line);
    std::string token;
    bool label = false;
    while (stream >> token) {
        if (token == "|") {
            label = true;
            continue;
        }
        (label ? sample.second : sample.first).push_back(std::stof(token));
    }
}

// one sample per line, "i | 10 * i", for i in [0, count)
inline std::string numbered_lines(std::size_t count) {
    std::string contents;
    for (std::size_t i = 0; i < count; i++) {
        contents += std::to_string(i) + " | " + std::to_string(i * 10) + "\n";
    }
    return contents;
}

} // namespace mlfo::tests

#endif // TESTS_DATASET_HELPERS_HPP
