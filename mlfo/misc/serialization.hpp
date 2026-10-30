#ifndef SERIALIZATION_HPP
#define SERIALIZATION_HPP

#include <charconv>
#include <cstddef>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

namespace mlfo::misc {

// models are serialized as text, one item per line, with lists of numbers comma separated
// see Model::serialize() and Layer::serialize() for the full format

// writes values comma separated on one line
// floating point values are written with enough digits to be read back exactly
template <typename U>
void write_line(std::ostream& os, const std::vector<U>& values) {
    const auto precision = os.precision(std::numeric_limits<U>::max_digits10);
    for (std::size_t i = 0; i < values.size(); i++) {
        if (i > 0) {
            os << ",";
        }
        os << values[i];
    }
    os << "\n";
    os.precision(precision);
}

// reads one line, dropping the \r of a file written with windows line endings
inline std::string read_line(std::istream& is) {
    std::string line;
    if (!std::getline(is, line)) {
        throw std::runtime_error("[mlfo::misc::read_line] Unexpected end of input");
    }
    if (!line.empty() && line.back() == '\r') {
        line.pop_back();
    }
    return line;
}

// reads one line of comma separated values, which must hold exactly count values
template <typename U>
std::vector<U> read_values(std::istream& is, std::size_t count) {
    const std::string line = read_line(is);
    std::vector<U> values;
    values.reserve(count);
    std::string_view rest = line;
    while (!rest.empty()) {
        const std::size_t comma = rest.find(',');
        const std::string_view token = rest.substr(0, comma);
        U value{};
        const auto [end, error] = std::from_chars(token.data(), token.data() + token.size(), value);
        if (error != std::errc{} || end != token.data() + token.size()) {
            throw std::runtime_error(
                "[mlfo::misc::read_values] Invalid value '" + std::string(token) + "'"
            );
        }
        values.push_back(value);
        if (comma == std::string_view::npos) {
            break;
        }
        rest.remove_prefix(comma + 1);
        if (rest.empty()) {
            throw std::runtime_error("[mlfo::misc::read_values] Trailing comma");
        }
    }
    if (values.size() != count) {
        throw std::runtime_error(
            "[mlfo::misc::read_values] Expected " + std::to_string(count) + " values, got " +
            std::to_string(values.size())
        );
    }
    return values;
}

} // namespace mlfo::misc

#endif // SERIALIZATION_HPP
