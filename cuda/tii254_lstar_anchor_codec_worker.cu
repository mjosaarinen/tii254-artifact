// Exact CPU/CUDA forward and transpose for the TII-254 L_star anchor codec.
//
// Source layout:
//   retained relations | non-anchor complement axes | free common-image axes.
//
// The last part is mapped into image(F) intersect ker(S) by the bound dense
// 1224 x 101913 correction K, then through the sparse common map F.  This is
// a standalone differential and timing gate; no Krylov iteration is enabled.

#ifndef MCELIECEX_HOLDOUT_CUDA_ALREADY_EMBEDDED
#define main mceliecex_holdout_cuda_embedded_main
#include "holdout_cuda_worker.cu"
#undef main
#endif

namespace mceliecex_tii254_lstar_anchor {

constexpr const char *REQUEST_SCHEMA =
    "mceliecex-tii254-d6-lstar-anchor-cuda-request-v1";
constexpr const char *RESULT_SCHEMA =
    "mceliecex-tii254-d6-lstar-anchor-cuda-result-v1";
constexpr const char *SOURCE_LAYOUT =
    "retained_relations_then_nonanchor_lstar_complement_axes_then_free_common_image_kernel_coordinates";
constexpr std::array<unsigned char, 8> COMMON_MAGIC{{
    'M', 'C', 'X', 'E', '4', 'S', '1', '\0'}};
constexpr std::array<unsigned char, 8> CORRECTION_MAGIC{{
    'M', 'C', 'X', 'L', 'S', 'K', '1', '\0'}};
constexpr std::array<unsigned char, 8> RETAINED_MAGIC{{
    'M', 'C', 'X', 'L', 'S', 'R', '1', '\0'}};

struct Request {
    std::string codec_identity;
    u64 source_rows = 0;
    u64 t64_rows = 0;
    u64 literal_rows = 0;
    u64 retained_rows = 0;
    u64 nonanchor_rows = 0;
    u64 anchor_rows = 0;
    u64 common_rows = 0;
    u64 removed_rows = 0;
    u64 free_rows = 0;
    u64 anchor_literal_offset = 0;
    int right_bits = 0;
    int right_words = 0;
    std::string source_layout;
};

template <class Value>
void read_named(std::istream &stream, const char *expected, Value &value) {
    std::string name;
    if (!(stream >> name) || name != expected || !(stream >> value)) {
        throw std::runtime_error(std::string("expected request key ") + expected);
    }
}

Request read_request(const std::filesystem::path &path) {
    std::ifstream stream(path);
    if (!stream) throw std::runtime_error("cannot open L_star anchor request");
    std::string schema;
    std::getline(stream, schema);
    if (schema != REQUEST_SCHEMA) throw std::runtime_error("request schema differs");
    Request request;
    read_named(stream, "codec_identity", request.codec_identity);
    read_named(stream, "source_rows", request.source_rows);
    read_named(stream, "t64_rows", request.t64_rows);
    read_named(stream, "literal_rows", request.literal_rows);
    read_named(stream, "retained_rows", request.retained_rows);
    read_named(stream, "nonanchor_rows", request.nonanchor_rows);
    read_named(stream, "anchor_rows", request.anchor_rows);
    read_named(stream, "common_rows", request.common_rows);
    read_named(stream, "removed_rows", request.removed_rows);
    read_named(stream, "free_rows", request.free_rows);
    read_named(stream, "right_bits", request.right_bits);
    read_named(stream, "anchor_literal_offset", request.anchor_literal_offset);
    read_named(stream, "source_layout", request.source_layout);
    std::string trailing;
    if (stream >> trailing) throw std::runtime_error("request has trailing tokens");
    request.right_words = (request.right_bits + 63) / 64;
    if (request.codec_identity.size() != 64
        || request.source_layout != SOURCE_LAYOUT
        || request.source_rows
            != request.retained_rows + request.nonanchor_rows + request.free_rows
        || request.t64_rows != 15'654'333
        || request.literal_rows != 20'271'300
        || request.retained_rows != 64
        || request.nonanchor_rows != 15'425'193
        || request.anchor_rows != 230'300
        || request.common_rows != 103'137
        || request.removed_rows != 1'224
        || request.free_rows != 101'913
        || request.anchor_literal_offset + request.anchor_rows > request.literal_rows
        || request.right_bits <= 0 || request.right_bits > 512) {
        throw std::runtime_error("request dimensions differ");
    }
    return request;
}

std::vector<unsigned char> read_all(const std::filesystem::path &path) {
    std::ifstream stream(path, std::ios::binary | std::ios::ate);
    if (!stream) throw std::runtime_error("cannot open codec payload");
    const std::streamoff end = stream.tellg();
    if (end < 0 || static_cast<unsigned long long>(end)
            > std::numeric_limits<std::size_t>::max()) {
        throw std::runtime_error("payload size escaped host size_t");
    }
    std::vector<unsigned char> output(static_cast<std::size_t>(end));
    stream.seekg(0);
    if (!output.empty()) stream.read(
        reinterpret_cast<char *>(output.data()),
        static_cast<std::streamsize>(output.size()));
    if (!stream || stream.peek() != std::char_traits<char>::eof()) {
        throw std::runtime_error("payload read differs");
    }
    return output;
}

std::uint32_t u32_at(const std::vector<unsigned char> &payload, std::size_t at) {
    if (at + 4 > payload.size()) throw std::runtime_error("u32 was truncated");
    return static_cast<std::uint32_t>(payload[at])
        | (static_cast<std::uint32_t>(payload[at + 1]) << 8)
        | (static_cast<std::uint32_t>(payload[at + 2]) << 16)
        | (static_cast<std::uint32_t>(payload[at + 3]) << 24);
}

u64 u64_at(const std::vector<unsigned char> &payload, std::size_t at) {
    if (at + 8 > payload.size()) throw std::runtime_error("u64 was truncated");
    u64 value = 0;
    for (int index = 0; index < 8; ++index) {
        value |= static_cast<u64>(payload[at + index]) << (8 * index);
    }
    return value;
}

std::vector<std::uint32_t> read_u32s(
    const std::filesystem::path &path, u64 expected) {
    const auto payload = read_all(path);
    if (expected > std::numeric_limits<std::size_t>::max() / 4
        || payload.size() != static_cast<std::size_t>(expected) * 4) {
        throw std::runtime_error("u32 payload size differs");
    }
    std::vector<std::uint32_t> answer(static_cast<std::size_t>(expected));
    for (std::size_t index = 0; index < answer.size(); ++index) {
        answer[index] = u32_at(payload, 4 * index);
    }
    return answer;
}

struct SparseColumns {
    u64 columns = 0;
    u64 rows = 0;
    std::vector<std::uint32_t> offsets;
    std::vector<std::uint32_t> row_indices;
};

struct RowCsr {
    std::vector<std::uint32_t> offsets;
    std::vector<std::uint32_t> columns;
};

SparseColumns read_common(
    const std::filesystem::path &path, const Request &request) {
    const auto payload = read_all(path);
    if (payload.size() < 48
        || !std::equal(COMMON_MAGIC.begin(), COMMON_MAGIC.end(), payload.begin())) {
        throw std::runtime_error("common sparse magic differs");
    }
    const u64 target = u64_at(payload, 8);
    const u64 original = u64_at(payload, 16);
    const u64 reduced = u64_at(payload, 24);
    const u64 kernel = u64_at(payload, 32);
    const u64 incidences = u64_at(payload, 40);
    if (target != request.anchor_rows || original != 106'971
        || reduced != request.common_rows || kernel != 3'834
        || incidences != 28'283'842
        || incidences > std::numeric_limits<std::uint32_t>::max()) {
        throw std::runtime_error("common sparse dimensions differ");
    }
    std::size_t at = 48;
    std::uint32_t previous = 0;
    for (u64 index = 0; index < reduced; ++index) {
        const std::uint32_t value = u32_at(payload, at);
        at += 4;
        if (value >= original || (index && value <= previous)) {
            throw std::runtime_error("common retained indices differ");
        }
        previous = value;
    }
    SparseColumns answer;
    answer.columns = reduced;
    answer.rows = target;
    answer.offsets.reserve(static_cast<std::size_t>(reduced + 1));
    answer.row_indices.reserve(static_cast<std::size_t>(incidences));
    answer.offsets.push_back(0);
    for (u64 column = 0; column < reduced; ++column) {
        const std::uint32_t count = u32_at(payload, at);
        at += 4;
        std::uint32_t prior = 0;
        for (std::uint32_t index = 0; index < count; ++index) {
            const std::uint32_t row = u32_at(payload, at);
            at += 4;
            if (row >= target || (index && row <= prior)) {
                throw std::runtime_error("common sparse column differs");
            }
            answer.row_indices.push_back(row);
            prior = row;
        }
        answer.offsets.push_back(
            static_cast<std::uint32_t>(answer.row_indices.size()));
    }
    if (at != payload.size() || answer.row_indices.size() != incidences) {
        throw std::runtime_error("common sparse accounting differs");
    }
    return answer;
}

SparseColumns read_retained(
    const std::filesystem::path &path, const Request &request) {
    const auto payload = read_all(path);
    if (payload.size() < 32
        || !std::equal(RETAINED_MAGIC.begin(), RETAINED_MAGIC.end(), payload.begin())) {
        throw std::runtime_error("retained relation magic differs");
    }
    SparseColumns answer;
    answer.columns = u64_at(payload, 8);
    const u64 incidences = u64_at(payload, 16);
    answer.rows = u64_at(payload, 24);
    if (answer.columns != request.retained_rows
        || answer.rows != request.literal_rows
        || incidences > std::numeric_limits<std::uint32_t>::max()) {
        throw std::runtime_error("retained relation dimensions differ");
    }
    const std::size_t offset_count = static_cast<std::size_t>(answer.columns + 1);
    const std::size_t rows_at = 32 + 8 * offset_count;
    if (rows_at > payload.size() || payload.size() != rows_at + 4 * incidences) {
        throw std::runtime_error("retained relation size differs");
    }
    answer.offsets.resize(offset_count);
    for (std::size_t index = 0; index < offset_count; ++index) {
        const u64 value = u64_at(payload, 32 + 8 * index);
        if (value > std::numeric_limits<std::uint32_t>::max()) {
            throw std::runtime_error("retained offset exceeds u32");
        }
        answer.offsets[index] = static_cast<std::uint32_t>(value);
    }
    if (answer.offsets.front() != 0 || answer.offsets.back() != incidences) {
        throw std::runtime_error("retained offsets differ");
    }
    answer.row_indices.resize(static_cast<std::size_t>(incidences));
    for (std::size_t index = 0; index < answer.row_indices.size(); ++index) {
        answer.row_indices[index] = u32_at(payload, rows_at + 4 * index);
        if (answer.row_indices[index] >= answer.rows) {
            throw std::runtime_error("retained row escaped literal dimension");
        }
    }
    return answer;
}

RowCsr transpose_columns(const SparseColumns &input) {
    RowCsr answer;
    answer.offsets.assign(static_cast<std::size_t>(input.rows + 1), 0);
    for (std::uint32_t row : input.row_indices) {
        ++answer.offsets[static_cast<std::size_t>(row + 1)];
    }
    for (std::size_t row = 1; row < answer.offsets.size(); ++row) {
        answer.offsets[row] += answer.offsets[row - 1];
    }
    answer.columns.resize(input.row_indices.size());
    std::vector<std::uint32_t> cursor = answer.offsets;
    for (u64 column = 0; column < input.columns; ++column) {
        for (std::uint32_t at = input.offsets[static_cast<std::size_t>(column)];
             at < input.offsets[static_cast<std::size_t>(column + 1)]; ++at) {
            const std::uint32_t row = input.row_indices[at];
            answer.columns[cursor[row]++] = static_cast<std::uint32_t>(column);
        }
    }
    return answer;
}

struct Correction {
    u64 rows = 0;
    u64 columns = 0;
    u64 words = 0;
    u64 transpose_words = 0;
    std::vector<u64> row_words;
    std::vector<u64> transpose_row_words;
};

Correction read_correction(
    const std::filesystem::path &path, const Request &request) {
    const auto payload = read_all(path);
    if (payload.size() < 32
        || !std::equal(
            CORRECTION_MAGIC.begin(), CORRECTION_MAGIC.end(), payload.begin())) {
        throw std::runtime_error("correction magic differs");
    }
    Correction answer;
    answer.rows = u64_at(payload, 8);
    answer.columns = u64_at(payload, 16);
    answer.words = u64_at(payload, 24);
    answer.transpose_words = (answer.rows + 63) / 64;
    if (answer.rows != request.removed_rows
        || answer.columns != request.free_rows
        || answer.words != (answer.columns + 63) / 64
        || payload.size() != 32 + 8 * answer.rows * answer.words) {
        throw std::runtime_error("correction dimensions differ");
    }
    answer.row_words.resize(static_cast<std::size_t>(answer.rows * answer.words));
    for (std::size_t index = 0; index < answer.row_words.size(); ++index) {
        answer.row_words[index] = u64_at(payload, 32 + 8 * index);
    }
    if (answer.columns % 64) {
        for (u64 row = 0; row < answer.rows; ++row) {
            const u64 tail = answer.row_words[static_cast<std::size_t>(
                row * answer.words + answer.words - 1)];
            if (tail >> (answer.columns % 64)) {
                throw std::runtime_error("correction tail padding differs");
            }
        }
    }
    answer.transpose_row_words.assign(
        static_cast<std::size_t>(answer.columns * answer.transpose_words), 0);
    for (u64 row = 0; row < answer.rows; ++row) {
        for (u64 word = 0; word < answer.words; ++word) {
            u64 value = answer.row_words[static_cast<std::size_t>(row * answer.words + word)];
            while (value) {
                const int bit = __builtin_ctzll(value);
                const u64 column = 64 * word + static_cast<u64>(bit);
                if (column < answer.columns) {
                    answer.transpose_row_words[
                        static_cast<std::size_t>(column * answer.transpose_words + row / 64)]
                        |= UINT64_C(1) << (row % 64);
                }
                value &= value - 1;
            }
        }
    }
    return answer;
}

struct Codec {
    std::vector<std::uint32_t> nonanchor_literal_rows;
    std::vector<std::uint32_t> pivot_sources;
    std::vector<std::uint32_t> free_sources;
    SparseColumns common;
    RowCsr common_rows;
    SparseColumns retained;
    Correction correction;
};

Codec load_codec(
    const Request &request,
    const std::filesystem::path &t64_selector_path,
    const std::filesystem::path &nonanchor_path,
    const std::filesystem::path &common_path,
    const std::filesystem::path &correction_path,
    const std::filesystem::path &pivot_path,
    const std::filesystem::path &free_path,
    const std::filesystem::path &retained_path) {
    const auto selector = read_u32s(
        t64_selector_path, request.t64_rows - request.retained_rows);
    const auto nonanchor = read_u32s(nonanchor_path, request.nonanchor_rows);
    Codec answer;
    answer.pivot_sources = read_u32s(pivot_path, request.removed_rows);
    answer.free_sources = read_u32s(free_path, request.free_rows);
    std::vector<unsigned char> used_common(static_cast<std::size_t>(request.common_rows), 0);
    for (std::uint32_t source : answer.pivot_sources) {
        if (source >= request.common_rows || used_common[source]++) {
            throw std::runtime_error("pivot common-source map differs");
        }
    }
    for (std::uint32_t source : answer.free_sources) {
        if (source >= request.common_rows || used_common[source]++) {
            throw std::runtime_error("free common-source map differs");
        }
    }
    if (std::find(used_common.begin(), used_common.end(), 0) != used_common.end()) {
        throw std::runtime_error("common-source partition is incomplete");
    }
    answer.nonanchor_literal_rows.reserve(nonanchor.size());
    std::uint32_t previous = 0;
    for (std::size_t index = 0; index < nonanchor.size(); ++index) {
        const std::uint32_t source = nonanchor[index];
        if (source < request.retained_rows || source >= request.t64_rows
            || (index && source <= previous)) {
            throw std::runtime_error("nonanchor T64 positions differ");
        }
        const std::uint32_t literal = selector[source - request.retained_rows];
        if (literal >= request.literal_rows
            || (literal >= request.anchor_literal_offset
                && literal < request.anchor_literal_offset + request.anchor_rows)) {
            throw std::runtime_error("nonanchor selector entered anchor top");
        }
        answer.nonanchor_literal_rows.push_back(literal);
        previous = source;
    }
    answer.common = read_common(common_path, request);
    answer.common_rows = transpose_columns(answer.common);
    answer.retained = read_retained(retained_path, request);
    answer.correction = read_correction(correction_path, request);
    return answer;
}

template <class Value>
Value *copy_device(const std::vector<Value> &values) {
    Value *device = nullptr;
    if (!values.empty()) {
        CUDA_CHECK(cudaMalloc(
            reinterpret_cast<void **>(&device), values.size() * sizeof(Value)));
        CUDA_CHECK(cudaMemcpy(
            device, values.data(), values.size() * sizeof(Value),
            cudaMemcpyHostToDevice));
    }
    return device;
}

__global__ void nonanchor_forward_kernel(
    const u64 *source, const std::uint32_t *literal_rows,
    u64 retained, u64 rows, int words, u64 *literal) {
    const u64 items = rows * static_cast<u64>(words);
    const u64 stride = static_cast<u64>(blockDim.x) * gridDim.x;
    for (u64 item = static_cast<u64>(blockIdx.x) * blockDim.x + threadIdx.x;
         item < items; item += stride) {
        const u64 row = item / words;
        const int word = static_cast<int>(item % words);
        literal[static_cast<u64>(literal_rows[row]) * words + word]
            = source[(retained + row) * words + word];
    }
}

__global__ void fill_free_kernel(
    const u64 *source, const std::uint32_t *free_sources,
    u64 source_offset, u64 rows, int words, u64 *common) {
    const u64 items = rows * static_cast<u64>(words);
    const u64 stride = static_cast<u64>(blockDim.x) * gridDim.x;
    for (u64 item = static_cast<u64>(blockIdx.x) * blockDim.x + threadIdx.x;
         item < items; item += stride) {
        const u64 row = item / words;
        const int word = static_cast<int>(item % words);
        common[static_cast<u64>(free_sources[row]) * words + word]
            = source[(source_offset + row) * words + word];
    }
}

__global__ void correction_forward_kernel(
    const u64 *source, const u64 *correction,
    const std::uint32_t *pivot_sources,
    u64 source_offset, u64 removed, u64 free_rows,
    u64 correction_words, int words, u64 *common) {
    const u64 items = removed * static_cast<u64>(words);
    const u64 stride = static_cast<u64>(blockDim.x) * gridDim.x;
    for (u64 item = static_cast<u64>(blockIdx.x) * blockDim.x + threadIdx.x;
         item < items; item += stride) {
        const u64 row = item / words;
        const int lane_word = static_cast<int>(item % words);
        u64 answer = 0;
        for (u64 packed = 0; packed < correction_words; ++packed) {
            u64 selected = correction[row * correction_words + packed];
            while (selected) {
                const int bit = __ffsll(static_cast<long long>(selected)) - 1;
                const u64 free = 64 * packed + static_cast<u64>(bit);
                if (free < free_rows) {
                    answer ^= source[(source_offset + free) * words + lane_word];
                }
                selected &= selected - 1;
            }
        }
        common[static_cast<u64>(pivot_sources[row]) * words + lane_word] = answer;
    }
}

__global__ void common_forward_kernel(
    const u64 *common, const std::uint32_t *offsets,
    const std::uint32_t *columns, u64 rows, u64 literal_offset,
    int words, u64 *literal) {
    const u64 items = rows * static_cast<u64>(words);
    const u64 stride = static_cast<u64>(blockDim.x) * gridDim.x;
    for (u64 item = static_cast<u64>(blockIdx.x) * blockDim.x + threadIdx.x;
         item < items; item += stride) {
        const u64 row = item / words;
        const int word = static_cast<int>(item % words);
        u64 value = 0;
        for (std::uint32_t at = offsets[row]; at < offsets[row + 1]; ++at) {
            value ^= common[static_cast<u64>(columns[at]) * words + word];
        }
        literal[(literal_offset + row) * words + word] = value;
    }
}

__global__ void retained_forward_kernel(
    const u64 *source, const std::uint32_t *offsets,
    const std::uint32_t *positions, u64 rows, int words, u64 *literal) {
    const u64 items = rows * static_cast<u64>(words);
    const u64 stride = static_cast<u64>(blockDim.x) * gridDim.x;
    for (u64 item = static_cast<u64>(blockIdx.x) * blockDim.x + threadIdx.x;
         item < items; item += stride) {
        const u64 row = item / words;
        const int word = static_cast<int>(item % words);
        const u64 value = source[row * words + word];
        for (std::uint32_t at = offsets[row]; at < offsets[row + 1]; ++at) {
            atomicXor(
                reinterpret_cast<unsigned long long *>(
                    &literal[static_cast<u64>(positions[at]) * words + word]),
                static_cast<unsigned long long>(value));
        }
    }
}

__global__ void nonanchor_transpose_kernel(
    const u64 *literal, const std::uint32_t *literal_rows,
    u64 retained, u64 rows, int words, u64 *source) {
    const u64 items = rows * static_cast<u64>(words);
    const u64 stride = static_cast<u64>(blockDim.x) * gridDim.x;
    for (u64 item = static_cast<u64>(blockIdx.x) * blockDim.x + threadIdx.x;
         item < items; item += stride) {
        const u64 row = item / words;
        const int word = static_cast<int>(item % words);
        source[(retained + row) * words + word]
            = literal[static_cast<u64>(literal_rows[row]) * words + word];
    }
}

__global__ void common_transpose_kernel(
    const u64 *literal, const std::uint32_t *offsets,
    const std::uint32_t *rows, u64 common_rows, u64 literal_offset,
    int words, u64 *common) {
    const u64 items = common_rows * static_cast<u64>(words);
    const u64 stride = static_cast<u64>(blockDim.x) * gridDim.x;
    for (u64 item = static_cast<u64>(blockIdx.x) * blockDim.x + threadIdx.x;
         item < items; item += stride) {
        const u64 column = item / words;
        const int word = static_cast<int>(item % words);
        u64 value = 0;
        for (std::uint32_t at = offsets[column]; at < offsets[column + 1]; ++at) {
            value ^= literal[(literal_offset + rows[at]) * words + word];
        }
        common[column * words + word] = value;
    }
}

__global__ void correction_transpose_kernel(
    const u64 *common, const u64 *correction_transpose,
    const std::uint32_t *pivot_sources, const std::uint32_t *free_sources,
    u64 retained, u64 nonanchor, u64 free_rows, u64 transpose_words,
    int words, u64 *source) {
    const u64 items = free_rows * static_cast<u64>(words);
    const u64 stride = static_cast<u64>(blockDim.x) * gridDim.x;
    for (u64 item = static_cast<u64>(blockIdx.x) * blockDim.x + threadIdx.x;
         item < items; item += stride) {
        const u64 row = item / words;
        const int lane_word = static_cast<int>(item % words);
        u64 value = common[static_cast<u64>(free_sources[row]) * words + lane_word];
        for (u64 packed = 0; packed < transpose_words; ++packed) {
            u64 selected = correction_transpose[row * transpose_words + packed];
            while (selected) {
                const int bit = __ffsll(static_cast<long long>(selected)) - 1;
                const u64 pivot = 64 * packed + static_cast<u64>(bit);
                value ^= common[static_cast<u64>(pivot_sources[pivot]) * words + lane_word];
                selected &= selected - 1;
            }
        }
        source[(retained + nonanchor + row) * words + lane_word] = value;
    }
}

__global__ void retained_transpose_kernel(
    const u64 *literal, const std::uint32_t *offsets,
    const std::uint32_t *positions, u64 rows, int words, u64 *source) {
    const u64 items = rows * static_cast<u64>(words);
    const u64 stride = static_cast<u64>(blockDim.x) * gridDim.x;
    for (u64 item = static_cast<u64>(blockIdx.x) * blockDim.x + threadIdx.x;
         item < items; item += stride) {
        const u64 row = item / words;
        const int word = static_cast<int>(item % words);
        u64 value = 0;
        for (std::uint32_t at = offsets[row]; at < offsets[row + 1]; ++at) {
            value ^= literal[static_cast<u64>(positions[at]) * words + word];
        }
        source[row * words + word] = value;
    }
}

struct DeviceCodec {
    std::uint32_t *nonanchor = nullptr;
    std::uint32_t *pivot_sources = nullptr;
    std::uint32_t *free_sources = nullptr;
    std::uint32_t *common_row_offsets = nullptr;
    std::uint32_t *common_row_columns = nullptr;
    std::uint32_t *common_offsets = nullptr;
    std::uint32_t *common_rows = nullptr;
    std::uint32_t *retained_offsets = nullptr;
    std::uint32_t *retained_positions = nullptr;
    u64 *correction = nullptr;
    u64 *correction_transpose = nullptr;

    void release() {
        cudaFree(correction_transpose); cudaFree(correction);
        cudaFree(retained_positions); cudaFree(retained_offsets);
        cudaFree(common_rows); cudaFree(common_offsets);
        cudaFree(common_row_columns); cudaFree(common_row_offsets);
        cudaFree(free_sources); cudaFree(pivot_sources); cudaFree(nonanchor);
        *this = DeviceCodec{};
    }
};

DeviceCodec upload(const Codec &codec) {
    DeviceCodec answer;
    answer.nonanchor = copy_device(codec.nonanchor_literal_rows);
    answer.pivot_sources = copy_device(codec.pivot_sources);
    answer.free_sources = copy_device(codec.free_sources);
    answer.common_row_offsets = copy_device(codec.common_rows.offsets);
    answer.common_row_columns = copy_device(codec.common_rows.columns);
    answer.common_offsets = copy_device(codec.common.offsets);
    answer.common_rows = copy_device(codec.common.row_indices);
    answer.retained_offsets = copy_device(codec.retained.offsets);
    answer.retained_positions = copy_device(codec.retained.row_indices);
    answer.correction = copy_device(codec.correction.row_words);
    answer.correction_transpose = copy_device(codec.correction.transpose_row_words);
    return answer;
}

struct Run {
    std::vector<u64> output;
    double seconds = 0;
};

Run gpu_forward(
    const Request &request, const Codec &codec, const std::vector<u64> &source) {
    u64 *device_source = nullptr, *device_literal = nullptr, *device_common = nullptr;
    DeviceCodec device;
    cudaEvent_t before{}, after{};
    try {
        device = upload(codec);
        device_source = copy_device(source);
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&device_literal),
            static_cast<std::size_t>(request.literal_rows) * request.right_words * sizeof(u64)));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&device_common),
            static_cast<std::size_t>(request.common_rows) * request.right_words * sizeof(u64)));
        CUDA_CHECK(cudaMemset(device_literal, 0,
            static_cast<std::size_t>(request.literal_rows) * request.right_words * sizeof(u64)));
        CUDA_CHECK(cudaMemset(device_common, 0,
            static_cast<std::size_t>(request.common_rows) * request.right_words * sizeof(u64)));
        CUDA_CHECK(cudaEventCreate(&before)); CUDA_CHECK(cudaEventCreate(&after));
        CUDA_CHECK(cudaEventRecord(before));
        nonanchor_forward_kernel<<<launch_blocks(request.nonanchor_rows * request.right_words), THREADS>>>(
            device_source, device.nonanchor, request.retained_rows,
            request.nonanchor_rows, request.right_words, device_literal);
        fill_free_kernel<<<launch_blocks(request.free_rows * request.right_words), THREADS>>>(
            device_source, device.free_sources,
            request.retained_rows + request.nonanchor_rows,
            request.free_rows, request.right_words, device_common);
        correction_forward_kernel<<<launch_blocks(request.removed_rows * request.right_words), THREADS>>>(
            device_source, device.correction, device.pivot_sources,
            request.retained_rows + request.nonanchor_rows,
            request.removed_rows, request.free_rows, codec.correction.words,
            request.right_words, device_common);
        common_forward_kernel<<<launch_blocks(request.anchor_rows * request.right_words), THREADS>>>(
            device_common, device.common_row_offsets, device.common_row_columns,
            request.anchor_rows, request.anchor_literal_offset,
            request.right_words, device_literal);
        retained_forward_kernel<<<launch_blocks(request.retained_rows * request.right_words), THREADS>>>(
            device_source, device.retained_offsets, device.retained_positions,
            request.retained_rows, request.right_words, device_literal);
        CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaEventRecord(after));
        CUDA_CHECK(cudaEventSynchronize(after));
        float milliseconds = 0; CUDA_CHECK(cudaEventElapsedTime(&milliseconds, before, after));
        Run answer;
        answer.output.resize(static_cast<std::size_t>(request.literal_rows) * request.right_words);
        CUDA_CHECK(cudaMemcpy(answer.output.data(), device_literal,
            answer.output.size() * sizeof(u64), cudaMemcpyDeviceToHost));
        answer.seconds = milliseconds / 1000.0;
        cudaEventDestroy(after); cudaEventDestroy(before);
        cudaFree(device_common); cudaFree(device_literal); cudaFree(device_source); device.release();
        return answer;
    } catch (...) {
        cudaEventDestroy(after); cudaEventDestroy(before);
        cudaFree(device_common); cudaFree(device_literal); cudaFree(device_source); device.release();
        throw;
    }
}

Run gpu_transpose(
    const Request &request, const Codec &codec, const std::vector<u64> &literal) {
    u64 *device_literal = nullptr, *device_source = nullptr, *device_common = nullptr;
    DeviceCodec device;
    cudaEvent_t before{}, after{};
    try {
        device = upload(codec);
        device_literal = copy_device(literal);
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&device_source),
            static_cast<std::size_t>(request.source_rows) * request.right_words * sizeof(u64)));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&device_common),
            static_cast<std::size_t>(request.common_rows) * request.right_words * sizeof(u64)));
        CUDA_CHECK(cudaMemset(device_source, 0,
            static_cast<std::size_t>(request.source_rows) * request.right_words * sizeof(u64)));
        CUDA_CHECK(cudaEventCreate(&before)); CUDA_CHECK(cudaEventCreate(&after));
        CUDA_CHECK(cudaEventRecord(before));
        nonanchor_transpose_kernel<<<launch_blocks(request.nonanchor_rows * request.right_words), THREADS>>>(
            device_literal, device.nonanchor, request.retained_rows,
            request.nonanchor_rows, request.right_words, device_source);
        common_transpose_kernel<<<launch_blocks(request.common_rows * request.right_words), THREADS>>>(
            device_literal, device.common_offsets, device.common_rows,
            request.common_rows, request.anchor_literal_offset,
            request.right_words, device_common);
        correction_transpose_kernel<<<launch_blocks(request.free_rows * request.right_words), THREADS>>>(
            device_common, device.correction_transpose,
            device.pivot_sources, device.free_sources,
            request.retained_rows, request.nonanchor_rows, request.free_rows,
            codec.correction.transpose_words, request.right_words, device_source);
        retained_transpose_kernel<<<launch_blocks(request.retained_rows * request.right_words), THREADS>>>(
            device_literal, device.retained_offsets, device.retained_positions,
            request.retained_rows, request.right_words, device_source);
        CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaEventRecord(after));
        CUDA_CHECK(cudaEventSynchronize(after));
        float milliseconds = 0; CUDA_CHECK(cudaEventElapsedTime(&milliseconds, before, after));
        Run answer;
        answer.output.resize(static_cast<std::size_t>(request.source_rows) * request.right_words);
        CUDA_CHECK(cudaMemcpy(answer.output.data(), device_source,
            answer.output.size() * sizeof(u64), cudaMemcpyDeviceToHost));
        answer.seconds = milliseconds / 1000.0;
        cudaEventDestroy(after); cudaEventDestroy(before);
        cudaFree(device_common); cudaFree(device_source); cudaFree(device_literal); device.release();
        return answer;
    } catch (...) {
        cudaEventDestroy(after); cudaEventDestroy(before);
        cudaFree(device_common); cudaFree(device_source); cudaFree(device_literal); device.release();
        throw;
    }
}

Run cpu_forward(
    const Request &request, const Codec &codec, const std::vector<u64> &source) {
    const auto started = std::chrono::steady_clock::now();
    std::vector<u64> literal(
        static_cast<std::size_t>(request.literal_rows) * request.right_words, 0);
    std::vector<u64> common(
        static_cast<std::size_t>(request.common_rows) * request.right_words, 0);
    for (u64 row = 0; row < request.nonanchor_rows; ++row) {
        for (int word = 0; word < request.right_words; ++word) {
            literal[static_cast<u64>(codec.nonanchor_literal_rows[row]) * request.right_words + word]
                = source[(request.retained_rows + row) * request.right_words + word];
        }
    }
    const u64 source_offset = request.retained_rows + request.nonanchor_rows;
    for (u64 row = 0; row < request.free_rows; ++row) {
        for (int word = 0; word < request.right_words; ++word) {
            common[static_cast<u64>(codec.free_sources[row]) * request.right_words + word]
                = source[(source_offset + row) * request.right_words + word];
        }
    }
    for (u64 row = 0; row < request.removed_rows; ++row) {
        for (int lane = 0; lane < request.right_words; ++lane) {
            u64 value = 0;
            for (u64 packed = 0; packed < codec.correction.words; ++packed) {
                u64 selected = codec.correction.row_words[row * codec.correction.words + packed];
                while (selected) {
                    const int bit = __builtin_ctzll(selected);
                    const u64 free = 64 * packed + bit;
                    if (free < request.free_rows) {
                        value ^= source[(source_offset + free) * request.right_words + lane];
                    }
                    selected &= selected - 1;
                }
            }
            common[static_cast<u64>(codec.pivot_sources[row]) * request.right_words + lane] = value;
        }
    }
    for (u64 row = 0; row < request.anchor_rows; ++row) {
        for (int word = 0; word < request.right_words; ++word) {
            u64 value = 0;
            for (std::uint32_t at = codec.common_rows.offsets[row];
                 at < codec.common_rows.offsets[row + 1]; ++at) {
                value ^= common[static_cast<u64>(codec.common_rows.columns[at])
                    * request.right_words + word];
            }
            literal[(request.anchor_literal_offset + row) * request.right_words + word] = value;
        }
    }
    for (u64 row = 0; row < request.retained_rows; ++row) {
        for (int word = 0; word < request.right_words; ++word) {
            const u64 value = source[row * request.right_words + word];
            for (std::uint32_t at = codec.retained.offsets[row];
                 at < codec.retained.offsets[row + 1]; ++at) {
                literal[static_cast<u64>(codec.retained.row_indices[at])
                    * request.right_words + word] ^= value;
            }
        }
    }
    return {std::move(literal), std::chrono::duration<double>(
        std::chrono::steady_clock::now() - started).count()};
}

Run cpu_transpose(
    const Request &request, const Codec &codec, const std::vector<u64> &literal) {
    const auto started = std::chrono::steady_clock::now();
    std::vector<u64> source(
        static_cast<std::size_t>(request.source_rows) * request.right_words, 0);
    std::vector<u64> common(
        static_cast<std::size_t>(request.common_rows) * request.right_words, 0);
    for (u64 row = 0; row < request.nonanchor_rows; ++row) {
        for (int word = 0; word < request.right_words; ++word) {
            source[(request.retained_rows + row) * request.right_words + word]
                = literal[static_cast<u64>(codec.nonanchor_literal_rows[row])
                    * request.right_words + word];
        }
    }
    for (u64 column = 0; column < request.common_rows; ++column) {
        for (int word = 0; word < request.right_words; ++word) {
            u64 value = 0;
            for (std::uint32_t at = codec.common.offsets[column];
                 at < codec.common.offsets[column + 1]; ++at) {
                value ^= literal[(request.anchor_literal_offset
                    + codec.common.row_indices[at]) * request.right_words + word];
            }
            common[column * request.right_words + word] = value;
        }
    }
    for (u64 row = 0; row < request.free_rows; ++row) {
        for (int word = 0; word < request.right_words; ++word) {
            u64 value = common[static_cast<u64>(codec.free_sources[row])
                * request.right_words + word];
            for (u64 packed = 0; packed < codec.correction.transpose_words; ++packed) {
                u64 selected = codec.correction.transpose_row_words[
                    row * codec.correction.transpose_words + packed];
                while (selected) {
                    const int bit = __builtin_ctzll(selected);
                    const u64 pivot = 64 * packed + bit;
                    value ^= common[static_cast<u64>(codec.pivot_sources[pivot])
                        * request.right_words + word];
                    selected &= selected - 1;
                }
            }
            source[(request.retained_rows + request.nonanchor_rows + row)
                * request.right_words + word] = value;
        }
    }
    for (u64 row = 0; row < request.retained_rows; ++row) {
        for (int word = 0; word < request.right_words; ++word) {
            u64 value = 0;
            for (std::uint32_t at = codec.retained.offsets[row];
                 at < codec.retained.offsets[row + 1]; ++at) {
                value ^= literal[static_cast<u64>(codec.retained.row_indices[at])
                    * request.right_words + word];
            }
            source[row * request.right_words + word] = value;
        }
    }
    return {std::move(source), std::chrono::duration<double>(
        std::chrono::steady_clock::now() - started).count()};
}

std::vector<u64> lane_dot(
    const std::vector<u64> &left, const std::vector<u64> &right,
    u64 rows, int words) {
    std::vector<u64> answer(static_cast<std::size_t>(words), 0);
    for (u64 row = 0; row < rows; ++row) {
        for (int word = 0; word < words; ++word) {
            const std::size_t at = static_cast<std::size_t>(row) * words + word;
            answer[word] ^= left[at] & right[at];
        }
    }
    return answer;
}

void write_result(
    const std::filesystem::path &path, const Request &request,
    const Codec &codec, const char *mode,
    const Run &forward, const Run &transpose,
    bool comparison_performed, bool cpu_cuda_equal, const std::string &device) {
    std::ostringstream output;
    output << "{\n"
           << "  \"schema\": \"" << RESULT_SCHEMA << "\",\n"
           << "  \"terminal\": \"tii254_d6_lstar_anchor_" << mode << "_pass\",\n"
           << "  \"claim_boundary\": \"codec forward/transpose differential and timing only; no Krylov, relation, locator, polynomial, or key\",\n"
           << "  \"codec_identity\": \"" << request.codec_identity << "\",\n"
           << "  \"mode\": \"" << mode << "\",\n"
           << "  \"source_rows\": " << request.source_rows << ",\n"
           << "  \"literal_rows\": " << request.literal_rows << ",\n"
           << "  \"right_bits\": " << request.right_bits << ",\n"
           << "  \"common_incidences\": " << codec.common.row_indices.size() << ",\n"
           << "  \"forward_seconds\": " << std::setprecision(9) << forward.seconds << ",\n"
           << "  \"transpose_seconds\": " << std::setprecision(9) << transpose.seconds << ",\n"
           << "  \"forward_fnv1a64_le\": \"" << hex_u64(checksum_words(forward.output)) << "\",\n"
           << "  \"transpose_fnv1a64_le\": \"" << hex_u64(checksum_words(transpose.output)) << "\",\n"
           << "  \"adjoint_identity\": true,\n"
           << "  \"cuda_comparison_performed\": "
           << (comparison_performed ? "true" : "false") << ",\n"
           << "  \"cpu_cuda_equal\": ";
    if (comparison_performed) {
        output << (cpu_cuda_equal ? "true" : "false");
    } else {
        output << "null";
    }
    output << ",\n"
           << "  \"device\": \"" << device << "\"\n"
           << "}\n";
    write_new(path, output.str());
}

void run(
    const std::string &mode,
    const std::filesystem::path &request_path,
    const std::filesystem::path &t64_selector_path,
    const std::filesystem::path &nonanchor_path,
    const std::filesystem::path &common_path,
    const std::filesystem::path &correction_path,
    const std::filesystem::path &pivot_path,
    const std::filesystem::path &free_path,
    const std::filesystem::path &retained_path,
    const std::filesystem::path &result_path) {
    const Request request = read_request(request_path);
    const Codec codec = load_codec(
        request, t64_selector_path, nonanchor_path, common_path,
        correction_path, pivot_path, free_path, retained_path);
    const auto source = random_words(
        static_cast<std::size_t>(request.source_rows) * request.right_words,
        UINT64_C(0xc4d6a254e9173b01));
    const auto literal_dual = random_words(
        static_cast<std::size_t>(request.literal_rows) * request.right_words,
        UINT64_C(0x72e4f93b1a65c820));
    Run forward, transpose;
    const bool comparison_performed = mode == "differential";
    bool equal = false;
    std::string device = "CPU reference";
    if (mode == "cpu") {
        forward = cpu_forward(request, codec, source);
        transpose = cpu_transpose(request, codec, literal_dual);
    } else if (mode == "cuda") {
        cudaDeviceProp properties{}; device_name(properties); device = properties.name;
        forward = gpu_forward(request, codec, source);
        transpose = gpu_transpose(request, codec, literal_dual);
    } else if (mode == "differential") {
        cudaDeviceProp properties{}; device_name(properties); device = properties.name;
        const Run cpu_f = cpu_forward(request, codec, source);
        const Run cpu_t = cpu_transpose(request, codec, literal_dual);
        forward = gpu_forward(request, codec, source);
        transpose = gpu_transpose(request, codec, literal_dual);
        equal = forward.output == cpu_f.output && transpose.output == cpu_t.output;
        if (!equal) throw std::runtime_error("CPU/CUDA target differential differs");
    } else {
        throw std::runtime_error("unknown benchmark mode");
    }
    if (lane_dot(forward.output, literal_dual, request.literal_rows, request.right_words)
        != lane_dot(source, transpose.output, request.source_rows, request.right_words)) {
        throw std::runtime_error("L_star anchor adjoint identity differs");
    }
    write_result(
        result_path, request, codec, mode.c_str(), forward, transpose,
        comparison_performed, equal, device);
}

}  // namespace mceliecex_tii254_lstar_anchor

int main(int argc, char **argv) {
    try {
        if (argc == 11) {
            mceliecex_tii254_lstar_anchor::run(
                argv[1], argv[2], argv[3], argv[4], argv[5], argv[6],
                argv[7], argv[8], argv[9], argv[10]);
            return 0;
        }
        std::cerr << "usage: " << argv[0]
                  << " (cpu|cuda|differential) REQUEST T64_SELECTOR NONANCHOR"
                  << " COMMON_SPARSE CORRECTION PIVOT_SOURCES FREE_SOURCES"
                  << " RETAINED_RELATIONS RESULT.json\n";
        return 2;
    } catch (const std::exception &error) {
        std::cerr << "TII-254 L_star anchor codec refused: " << error.what() << '\n';
        return 1;
    }
}
