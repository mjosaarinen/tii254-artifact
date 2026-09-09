// Stream CADO's coefficient-major binary Pi and extract the selected right
// relations used by the TII-254 reconstruction fallback.
//
// Input Pi has (left+right)^2 bits per polynomial coefficient, row-major with
// little-endian u64 packing.  Columns are selected by exactly the rule in the
// pinned fast verifier: increasing shifted degree, nonzero top block, and
// numerator degree strictly below top degree.  For a selected column of top
// degree d, the emitted reconstruction coefficients are
//
//     F_j = P_(d-j),  j=0,...,d.
//
// The output is coefficient-major.  Each F_j is a right-by-right binary
// matrix, row-major in little-endian u64 words.  Columns with smaller degree
// are zero-padded at later j; padding never reindexes a relation.

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <unistd.h>
#include <utility>
#include <vector>

namespace {

using u64 = std::uint64_t;

void write_all(int descriptor, const void *data, std::size_t size) {
    const auto *bytes = static_cast<const unsigned char *>(data);
    while (size != 0) {
        const ssize_t written = ::write(descriptor, bytes, size);
        if (written < 0) {
            if (errno == EINTR) {
                continue;
            }
            throw std::runtime_error(
                std::string("write relation payload: ") + std::strerror(errno));
        }
        if (written == 0) {
            throw std::runtime_error("short write of relation payload");
        }
        bytes += written;
        size -= static_cast<std::size_t>(written);
    }
}

unsigned int packed_rank(
    std::vector<std::vector<u64>> rows, unsigned int columns) {
    unsigned int rank = 0;
    for (int column = static_cast<int>(columns) - 1;
         column >= 0 && rank < rows.size(); --column) {
        const u64 mask = UINT64_C(1) << (static_cast<unsigned int>(column) % 64U);
        const unsigned int word = static_cast<unsigned int>(column) / 64U;
        unsigned int pivot = rank;
        while (pivot < rows.size() && !(rows[pivot][word] & mask)) {
            ++pivot;
        }
        if (pivot == rows.size()) {
            continue;
        }
        std::swap(rows[rank], rows[pivot]);
        for (unsigned int row = 0; row < rows.size(); ++row) {
            if (row == rank || !(rows[row][word] & mask)) {
                continue;
            }
            for (unsigned int index = 0; index < rows[row].size(); ++index) {
                rows[row][index] ^= rows[rank][index];
            }
        }
        ++rank;
    }
    return rank;
}

std::string histogram_json(std::vector<int> values) {
    std::sort(values.begin(), values.end());
    std::string output = "{";
    bool first = true;
    for (std::size_t begin = 0; begin < values.size();) {
        std::size_t end = begin + 1;
        while (end < values.size() && values[end] == values[begin]) {
            ++end;
        }
        if (!first) {
            output += ',';
        }
        first = false;
        output += '\"' + std::to_string(values[begin]) + "\":"
            + std::to_string(end - begin);
        begin = end;
    }
    output += '}';
    return output;
}

std::string integer_array_json(std::vector<unsigned int> const &values) {
    std::string output = "[";
    for (std::size_t index = 0; index < values.size(); ++index) {
        if (index != 0) {
            output += ',';
        }
        output += std::to_string(values[index]);
    }
    output += ']';
    return output;
}

u64 fnv_update(u64 state, const void *data, std::size_t size) {
    const auto *bytes = static_cast<const unsigned char *>(data);
    for (std::size_t index = 0; index < size; ++index) {
        state ^= bytes[index];
        state *= UINT64_C(1099511628211);
    }
    return state;
}

void extract(
    std::filesystem::path const &pi_path,
    unsigned int left,
    unsigned int right,
    std::size_t order,
    std::filesystem::path const &output_path) {
    if (left == 0 || right == 0 || left % 64U || right % 64U) {
        throw std::runtime_error("positive left/right widths must be multiples of 64");
    }
    const unsigned int dimension = left + right;
    if (dimension % 64U) {
        throw std::runtime_error("total basis dimension is not u64-aligned");
    }
    const std::size_t row_words = dimension / 64U;
    const std::size_t coefficient_words =
        static_cast<std::size_t>(dimension) * row_words;
    const std::uintmax_t coefficient_bytes = coefficient_words * sizeof(u64);
    const std::uintmax_t pi_bytes = std::filesystem::file_size(pi_path);
    if (coefficient_bytes == 0 || pi_bytes % coefficient_bytes) {
        throw std::runtime_error("Pi payload has a partial coefficient");
    }
    const std::size_t coefficient_count = pi_bytes / coefficient_bytes;
    if (coefficient_count == 0 || coefficient_count > order + 2) {
        throw std::runtime_error("Pi coefficient count is outside the direct-basis bound");
    }

    std::ifstream input(pi_path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("cannot open Pi payload");
    }
    std::vector<u64> coefficient(coefficient_words);
    std::vector<int> top_degree(dimension, -1);
    std::vector<int> numerator_degree(dimension, -1);
    std::vector<int> shifted_degree(dimension, -1);
    std::vector<u64> top_presence(row_words);
    std::vector<u64> numerator_presence(row_words);
    for (std::size_t degree = 0; degree < coefficient_count; ++degree) {
        input.read(
            reinterpret_cast<char *>(coefficient.data()),
            static_cast<std::streamsize>(coefficient_bytes));
        if (!input) {
            throw std::runtime_error("short Pi read during degree scan");
        }
        std::fill(top_presence.begin(), top_presence.end(), 0);
        std::fill(numerator_presence.begin(), numerator_presence.end(), 0);
        for (unsigned int row = 0; row < right; ++row) {
            for (std::size_t word = 0; word < row_words; ++word) {
                top_presence[word] |= coefficient[row * row_words + word];
            }
        }
        for (unsigned int row = right; row < dimension; ++row) {
            for (std::size_t word = 0; word < row_words; ++word) {
                numerator_presence[word] |= coefficient[row * row_words + word];
            }
        }
        for (std::size_t word = 0; word < row_words; ++word) {
            u64 top = top_presence[word];
            while (top) {
                const unsigned int bit = static_cast<unsigned int>(__builtin_ctzll(top));
                const unsigned int column = static_cast<unsigned int>(word * 64U + bit);
                top_degree[column] = static_cast<int>(degree);
                shifted_degree[column] = static_cast<int>(degree);
                top &= top - 1;
            }
            u64 numerator = numerator_presence[word];
            while (numerator) {
                const unsigned int bit =
                    static_cast<unsigned int>(__builtin_ctzll(numerator));
                const unsigned int column = static_cast<unsigned int>(word * 64U + bit);
                numerator_degree[column] = static_cast<int>(degree);
                shifted_degree[column] = std::max(
                    shifted_degree[column], static_cast<int>(degree + 1));
                numerator &= numerator - 1;
            }
        }
    }
    if (input.peek() != std::char_traits<char>::eof()) {
        throw std::runtime_error("trailing Pi data after degree scan");
    }

    std::vector<unsigned int> ordered(dimension);
    std::iota(ordered.begin(), ordered.end(), 0U);
    std::sort(ordered.begin(), ordered.end(), [&](unsigned int first, unsigned int second) {
        return std::pair(shifted_degree[first], first)
            < std::pair(shifted_degree[second], second);
    });
    std::vector<unsigned int> selected;
    for (unsigned int column : ordered) {
        if (top_degree[column] >= 0
            && numerator_degree[column] < top_degree[column]) {
            if (shifted_degree[column] != top_degree[column]) {
                throw std::runtime_error("selected relation has nonzero boundary exponent");
            }
            selected.push_back(column);
            if (selected.size() == right) {
                break;
            }
        }
    }
    if (selected.size() != right) {
        throw std::runtime_error("complete selected right-relation set is absent");
    }
    std::vector<int> relation_degrees;
    relation_degrees.reserve(right);
    int maximum_degree = -1;
    for (unsigned int column : selected) {
        relation_degrees.push_back(top_degree[column]);
        maximum_degree = std::max(maximum_degree, top_degree[column]);
    }
    if (maximum_degree < 0 || static_cast<std::size_t>(maximum_degree) >= order) {
        throw std::runtime_error("selected relation degree is outside the sequence order");
    }

    const std::size_t relation_words = (selected.size() + 63U) / 64U;
    const std::size_t matrix_words = static_cast<std::size_t>(right) * relation_words;
    if (static_cast<std::size_t>(maximum_degree + 1)
        > std::numeric_limits<std::size_t>::max() / matrix_words) {
        throw std::runtime_error("relation output size overflows");
    }
    std::vector<u64> relations(
        static_cast<std::size_t>(maximum_degree + 1) * matrix_words, 0);
    std::vector<std::vector<u64>> constant_rows(
        right, std::vector<u64>(relation_words));

    input.clear();
    input.seekg(0);
    for (std::size_t degree = 0; degree < coefficient_count; ++degree) {
        input.read(
            reinterpret_cast<char *>(coefficient.data()),
            static_cast<std::streamsize>(coefficient_bytes));
        if (!input) {
            throw std::runtime_error("short Pi read during relation extraction");
        }
        for (unsigned int output = 0; output < selected.size(); ++output) {
            const int relation_degree = relation_degrees[output];
            if (degree > static_cast<std::size_t>(relation_degree)) {
                continue;
            }
            const std::size_t offset = static_cast<std::size_t>(relation_degree) - degree;
            const unsigned int basis_column = selected[output];
            const std::size_t source_word = basis_column / 64U;
            const u64 source_mask = UINT64_C(1) << (basis_column % 64U);
            const std::size_t output_word = output / 64U;
            const u64 output_mask = UINT64_C(1) << (output % 64U);
            for (unsigned int row = 0; row < right; ++row) {
                if (coefficient[row * row_words + source_word] & source_mask) {
                    relations[offset * matrix_words + row * relation_words + output_word]
                        |= output_mask;
                    if (degree == 0) {
                        constant_rows[row][output_word] |= output_mask;
                    }
                }
            }
        }
    }
    if (input.peek() != std::char_traits<char>::eof()) {
        throw std::runtime_error("trailing Pi data after relation extraction");
    }

    std::vector<std::vector<u64>> leading_rows(
        right, std::vector<u64>(relation_words));
    for (unsigned int row = 0; row < right; ++row) {
        std::copy_n(
            relations.data() + row * relation_words,
            relation_words,
            leading_rows[row].data());
    }
    const unsigned int leading_rank = packed_rank(leading_rows, right);
    const unsigned int constant_rank = packed_rank(constant_rows, right);

    const int descriptor = ::open(
        output_path.c_str(), O_WRONLY | O_CREAT | O_EXCL, 0444);
    if (descriptor < 0) {
        throw std::runtime_error(
            std::string("create relation payload: ") + std::strerror(errno));
    }
    bool closed = false;
    try {
        write_all(descriptor, relations.data(), relations.size() * sizeof(u64));
        if (::fsync(descriptor) != 0) {
            throw std::runtime_error("fsync relation payload failed");
        }
        if (::close(descriptor) != 0) {
            throw std::runtime_error("close relation payload failed");
        }
        closed = true;
    } catch (...) {
        if (!closed) {
            ::close(descriptor);
        }
        throw;
    }

    const u64 fnv = fnv_update(
        UINT64_C(1469598103934665603),
        relations.data(),
        relations.size() * sizeof(u64));
    std::cout
        << "{\n"
        << "  \"schema\": \"mceliecex-cado-gf2-direct-relation-extraction-native-v1\",\n"
        << "  \"left_width\": " << left << ",\n"
        << "  \"right_width\": " << right << ",\n"
        << "  \"order\": " << order << ",\n"
        << "  \"pi_coefficient_count\": " << coefficient_count << ",\n"
        << "  \"selected_relation_count\": " << selected.size() << ",\n"
        << "  \"selected_relation_columns\": " << integer_array_json(selected) << ",\n"
        << "  \"selected_relation_degree_profile\": "
        << histogram_json(relation_degrees) << ",\n"
        << "  \"maximum_relation_degree\": " << maximum_degree << ",\n"
        << "  \"boundary_exponents_all_zero\": true,\n"
        << "  \"selected_relation_leading_rank\": " << leading_rank << ",\n"
        << "  \"selected_relation_constant_rank\": " << constant_rank << ",\n"
        << "  \"layout\": \"coefficient_major_right_by_right_u64_le\",\n"
        << "  \"coefficient_matrix_count\": " << maximum_degree + 1 << ",\n"
        << "  \"coefficient_matrix_words\": " << matrix_words << ",\n"
        << "  \"output_bytes\": " << relations.size() * sizeof(u64) << ",\n"
        << "  \"output_fnv1a64_le\": \"" << std::hex << std::setw(16)
        << std::setfill('0') << fnv << std::dec << std::setfill(' ') << "\",\n"
        << "  \"terminal\": \"cado_gf2_direct_relation_extraction_native_pass\"\n"
        << "}\n";
}

unsigned long parse_unsigned(char const *text, char const *name) {
    std::size_t consumed = 0;
    const std::string value(text);
    const unsigned long result = std::stoul(value, &consumed, 10);
    if (consumed != value.size()) {
        throw std::runtime_error(std::string("invalid ") + name);
    }
    return result;
}

}  // namespace

int main(int argc, char **argv) {
    try {
        if (argc != 6) {
            std::cerr << "usage: " << argv[0]
                      << " PI LEFT_WIDTH RIGHT_WIDTH ORDER NEW_RELATIONS.bin\n";
            return 2;
        }
        extract(
            argv[1],
            static_cast<unsigned int>(parse_unsigned(argv[2], "left width")),
            static_cast<unsigned int>(parse_unsigned(argv[3], "right width")),
            parse_unsigned(argv[4], "order"),
            argv[5]);
        return 0;
    } catch (std::exception const &error) {
        std::cerr << "direct GF(2) relation extractor refused: "
                  << error.what() << '\n';
        return 1;
    }
}
