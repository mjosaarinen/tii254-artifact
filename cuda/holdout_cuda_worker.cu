// Portable CUDA baseline for the binary Holdout operator.
//
// For one public point p with support P, a row indexed by a j-subset T and a
// degree-d column indexed by S has value
//
//     M_p[T,S] = [T subset S subset T union P].
//
// The worker applies B = sum_p M_p^T M_p point by point.  A CUDA thread owns a
// complete row in M_p and a complete column in M_p^T, so the implementation has
// no atomics.  Two implementations are retained deliberately: a direct
// combinadic traversal used as a correctness oracle, and a production-shaped
// slice traversal which factors rows and columns by
//
//     A = T intersect complement(P) = S intersect complement(P).
//
// The slice path precomputes its global column map once and then performs only
// small support-local subset operations in the Krylov hot loop.

#include <cuda_runtime.h>

#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#ifndef MCELIECEX_CUDA_TARGET_ARCH
#define MCELIECEX_CUDA_TARGET_ARCH 0
#endif

namespace {

constexpr int LEGACY_MAX_K = 63;
constexpr int MAX_K = 68;
constexpr int MAX_D = 9;
constexpr int MAX_WORDS = 16;
constexpr int THREADS = 256;
constexpr int MAX_BLOCKS = 65535;
constexpr std::uint64_t PROJECTION_SEED_XOR = 0x9e3779b97f4a7c15ULL;
constexpr const char *REQUEST_SCHEMA = "mceliecex-holdout-cuda-request-v1";
constexpr const char *EXTENDED_REQUEST_SCHEMA =
    "mceliecex-holdout-cuda-two-word-request-v1";
constexpr const char *RESPONSE_SCHEMA = "mceliecex-holdout-cuda-response-v2";
constexpr const char *LEFT_RESPONSE_SCHEMA =
    "mceliecex-holdout-cuda-left-response-v1";
constexpr const char *STATE_SCHEMA = "mceliecex-holdout-cuda-state-v1";
constexpr const char *SEQUENCE_SCHEMA = "mceliecex-holdout-cuda-sequence-v1";
constexpr const char *INJECTIVE_STATE_SCHEMA =
    "mceliecex-holdout-cuda-injective-state-v1";
constexpr const char *INJECTIVE_SEQUENCE_SCHEMA =
    "mceliecex-holdout-cuda-injective-sequence-v1";
constexpr const char *WITNESS_REQUEST_SCHEMA =
    "mceliecex-tii254-cuda-literal-witness-request-v1";
constexpr const char *WITNESS_DOMAIN_SCHEMA =
    "mceliecex-tii254-cuda-literal-domain-v1";
constexpr const char *WITNESS_RESPONSE_SCHEMA =
    "mceliecex-tii254-cuda-literal-witness-response-v1";
constexpr int WITNESS_TARGETS = 3;

using Clock = std::chrono::steady_clock;
using u64 = std::uint64_t;

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t cuda_status_ = (call);                                       \
        if (cuda_status_ != cudaSuccess) {                                       \
            std::ostringstream cuda_message_;                                   \
            cuda_message_ << #call << ": " << cudaGetErrorString(cuda_status_); \
            throw std::runtime_error(cuda_message_.str());                      \
        }                                                                       \
    } while (false)

__constant__ u64 DEVICE_CHOOSE[(MAX_K + 1) * (MAX_D + 1)];

struct Point {
    u64 support = 0;
    u64 support_high = 0;
    std::vector<int> levels;
    std::vector<u64> offsets;
    u64 rows = 0;
    u64 nonzeros = 0;
};

struct Request {
    std::string schema;
    int k = 0;
    int degree = 0;
    int words = 0;
    int repetitions = 0;
    int emit_output = 0;
    u64 seed = 0;
    u64 expected_columns = 0;
    u64 expected_rows = 0;
    u64 expected_nonzeros = 0;
    std::string public_sha256;
    std::vector<Point> points;
};

struct WitnessPointData {
    int multiplicity = 0;
    std::vector<unsigned char> jets;
    std::vector<unsigned char> weights;
};

struct WitnessRequest {
    Request layout;
    int emit_coefficients = 0;
    int verify_target = 0;
    u64 target_support = 0;
    u64 target_support_high = 0;
    std::vector<unsigned char> target_jets;
    std::vector<WitnessPointData> point_data;
};

struct SliceClass {
    int complement_size = 0;
    int support_column_size = 0;
    u64 slice_count = 0;
    u64 columns_per_slice = 0;
    u64 rows_per_slice = 0;
    u64 mapping_offset = 0;
    u64 range_offset = 0;
    std::array<int, 4> row_sizes{};
    std::array<u64, 4> row_offsets{};
    int row_size_count = 0;
    std::size_t pattern_index = 0;
};

struct LocalPatternPlan {
    int support_weight = 0;
    int support_column_size = 0;
    std::array<int, 4> row_sizes{};
    int row_size_count = 0;
    u64 columns_offset = 0;
    u64 starts_offset = 0;
    u64 row_count = 0;
    u64 incidence_count = 0;
};

struct SlicePointPlan {
    std::vector<SliceClass> classes;
};

struct SlicePlan {
    std::vector<SlicePointPlan> points;
    std::vector<LocalPatternPlan> patterns;
    std::vector<std::uint32_t> pattern_columns;
    std::vector<std::uint32_t> pattern_starts;
    u64 mapping_entries = 0;
    u64 max_class_mapping_entries = 0;
    u64 max_range_rows = 0;
    u64 total_range_rows = 0;
};

struct HostChoose {
    std::array<u64, (MAX_K + 1) * (MAX_D + 1)> values{};

    HostChoose() {
        for (int n = 0; n <= MAX_K; ++n) {
            at(n, 0) = 1;
            for (int r = 1; r <= std::min(n, MAX_D); ++r) {
                at(n, r) = r == n ? 1 : get(n - 1, r - 1) + get(n - 1, r);
            }
        }
    }

    u64 get(int n, int r) const {
        if (n < 0 || n > MAX_K || r < 0 || r > MAX_D || r > n) {
            return 0;
        }
        return values[static_cast<std::size_t>(n) * (MAX_D + 1) + r];
    }

    u64 &at(int n, int r) {
        return values[static_cast<std::size_t>(n) * (MAX_D + 1) + r];
    }
};

const HostChoose HOST_CHOOSE;

__device__ __forceinline__ u64 device_choose(int n, int r) {
    if (n < 0 || n > MAX_K || r < 0 || r > MAX_D || r > n) {
        return 0;
    }
    return DEVICE_CHOOSE[n * (MAX_D + 1) + r];
}

__host__ __device__ __forceinline__ int low_bit(u64 value) {
#ifdef __CUDA_ARCH__
    return __ffsll(static_cast<long long>(value)) - 1;
#else
    return __builtin_ctzll(value);
#endif
}

int point_weight(const Point &point) {
    return __builtin_popcountll(point.support)
        + __builtin_popcountll(point.support_high);
}

bool point_contains(const Point &point, int coordinate) {
    if (coordinate < 64) {
        return (point.support & (u64{1} << coordinate)) != 0;
    }
    return (point.support_high & (u64{1} << (coordinate - 64))) != 0;
}

void require_one_word_ground_set(
    const Request &request, const char *backend) {
    bool high_support_present = false;
    for (const Point &point : request.points) {
        high_support_present = high_support_present || point.support_high != 0;
    }
    if (request.k > LEGACY_MAX_K || high_support_present) {
        throw std::runtime_error(
            std::string(backend)
            + " is restricted to the one-word k<=63 ground-set contract");
    }
}

__device__ __forceinline__ u64 colex_rank_device(u64 mask) {
    u64 rank = 0;
    int index = 1;
    while (mask != 0) {
        const int bit = low_bit(mask);
        mask &= mask - 1;
        rank += device_choose(bit, index);
        ++index;
    }
    return rank;
}

__device__ __forceinline__ u64 colex_unrank_device(
    u64 rank, int size, int k) {
    u64 mask = 0;
    int upper = k;
    for (int index = size; index >= 1; --index) {
        int bit = upper - 1;
        while (bit >= index && device_choose(bit, index) > rank) {
            --bit;
        }
        mask |= u64{1} << bit;
        rank -= device_choose(bit, index);
        upper = bit;
    }
    return mask;
}

__device__ __forceinline__ void colex_unrank_positions_device(
    u64 rank, int size, int k, unsigned char *positions) {
    int upper = k;
    for (int index = size; index >= 1; --index) {
        int bit = upper - 1;
        while (bit >= index && device_choose(bit, index) > rank) {
            --bit;
        }
        positions[index - 1] = static_cast<unsigned char>(bit);
        rank -= device_choose(bit, index);
        upper = bit;
    }
}

__device__ __forceinline__ bool two_word_contains_device(
    u64 low, u64 high, int coordinate) {
    if (coordinate < 64) {
        return (low & (u64{1} << coordinate)) != 0;
    }
    return (high & (u64{1} << (coordinate - 64))) != 0;
}

__device__ __forceinline__ unsigned char gf256_multiply_device(
    unsigned char left, unsigned char right) {
    unsigned char result = 0;
    for (int bit = 0; bit < 8; ++bit) {
        if ((right & 1U) != 0) {
            result ^= left;
        }
        const bool carry = (left & 0x80U) != 0;
        left = static_cast<unsigned char>(left << 1);
        if (carry) {
            left ^= 0x1dU;
        }
        right = static_cast<unsigned char>(right >> 1);
    }
    return result;
}

u64 colex_rank_host(u64 mask) {
    u64 rank = 0;
    int index = 1;
    while (mask != 0) {
        const int bit = low_bit(mask);
        mask &= mask - 1;
        rank += HOST_CHOOSE.get(bit, index);
        ++index;
    }
    return rank;
}

u64 colex_unrank_host(u64 rank, int size, int k) {
    u64 mask = 0;
    int upper = k;
    for (int index = size; index >= 1; --index) {
        int bit = upper - 1;
        while (bit >= index && HOST_CHOOSE.get(bit, index) > rank) {
            --bit;
        }
        mask |= u64{1} << bit;
        rank -= HOST_CHOOSE.get(bit, index);
        upper = bit;
    }
    return mask;
}

__device__ __forceinline__ bool next_indices(
    int *indices, int size, int count) {
    if (size == 0) {
        return false;
    }
    int cursor = size - 1;
    while (cursor >= 0 && indices[cursor] == cursor + count - size) {
        --cursor;
    }
    if (cursor < 0) {
        return false;
    }
    ++indices[cursor];
    for (int index = cursor + 1; index < size; ++index) {
        indices[index] = indices[index - 1] + 1;
    }
    return true;
}

__device__ __forceinline__ int mask_positions(u64 mask, unsigned char *positions) {
    int count = 0;
    while (mask != 0) {
        positions[count++] = static_cast<unsigned char>(low_bit(mask));
        mask &= mask - 1;
    }
    return count;
}

__device__ __forceinline__ u64 colex_rank_merged_local_masks_device(
    u64 left_mask,
    const unsigned char *__restrict__ left_positions,
    u64 right_mask,
    const unsigned char *__restrict__ right_positions) {
    unsigned char selected[MAX_D];
    int selected_count = 0;
    while (left_mask != 0) {
        const int local_bit = low_bit(left_mask);
        left_mask &= left_mask - 1;
        selected[selected_count++] = left_positions[local_bit];
    }
    while (right_mask != 0) {
        const int local_bit = low_bit(right_mask);
        right_mask &= right_mask - 1;
        selected[selected_count++] = right_positions[local_bit];
    }
    for (int index = 1; index < selected_count; ++index) {
        const unsigned char value = selected[index];
        int cursor = index;
        while (cursor > 0 && selected[cursor - 1] > value) {
            selected[cursor] = selected[cursor - 1];
            --cursor;
        }
        selected[cursor] = value;
    }
    u64 rank = 0;
    for (int index = 0; index < selected_count; ++index) {
        rank += device_choose(selected[index], index + 1);
    }
    return rank;
}

u64 colex_rank_merged_local_masks_host(
    u64 left_mask,
    const unsigned char *left_positions,
    u64 right_mask,
    const unsigned char *right_positions) {
    std::array<unsigned char, MAX_D> selected{};
    int selected_count = 0;
    while (left_mask != 0) {
        const int local_bit = low_bit(left_mask);
        left_mask &= left_mask - 1;
        selected[selected_count++] = left_positions[local_bit];
    }
    while (right_mask != 0) {
        const int local_bit = low_bit(right_mask);
        right_mask &= right_mask - 1;
        selected[selected_count++] = right_positions[local_bit];
    }
    std::sort(selected.begin(), selected.begin() + selected_count);
    u64 rank = 0;
    for (int index = 0; index < selected_count; ++index) {
        rank += HOST_CHOOSE.get(selected[index], index + 1);
    }
    return rank;
}

__global__ void fill_slice_column_map_kernel(
    std::uint32_t *__restrict__ mapping,
    u64 mapping_count,
    u64 slices,
    u64 columns_per_slice,
    int complement_size,
    int support_column_size,
    int complement_weight,
    int support_weight,
    const unsigned char *__restrict__ complement_positions,
    const unsigned char *__restrict__ support_positions) {
    const u64 stride = static_cast<u64>(blockDim.x) * gridDim.x;
    for (u64 entry = static_cast<u64>(blockIdx.x) * blockDim.x + threadIdx.x;
         entry < mapping_count;
         entry += stride) {
        const u64 slice = entry / columns_per_slice;
        const u64 local_column = entry - slice * columns_per_slice;
        const u64 local_a = colex_unrank_device(
            slice, complement_size, complement_weight);
        const u64 local_b = colex_unrank_device(
            local_column, support_column_size, support_weight);
        mapping[entry] = static_cast<std::uint32_t>(
            colex_rank_merged_local_masks_device(
                local_a,
                complement_positions,
                local_b,
                support_positions));
    }
}

__global__ void generate_literal_witness_coefficients_kernel(
    u64 *__restrict__ coefficients,
    u64 row_count,
    u64 row_offset,
    int k,
    int level,
    int multiplicity,
    std::uint16_t derivative_order_mask,
    u64 support_low,
    u64 support_high,
    const unsigned char *__restrict__ jets,
    const unsigned char *__restrict__ weights) {
    const u64 stride = static_cast<u64>(blockDim.x) * gridDim.x;
    for (u64 row = static_cast<u64>(blockIdx.x) * blockDim.x + threadIdx.x;
         row < row_count;
         row += stride) {
        unsigned char derivative[MAX_D]{};
        colex_unrank_positions_device(row, level, k, derivative);
        unsigned char outside[MAX_D]{};
        unsigned char inside[MAX_D]{};
        int outside_count = 0;
        int inside_count = 0;
        for (int index = 0; index < level; ++index) {
            const unsigned char coordinate = derivative[index];
            if (two_word_contains_device(
                    support_low, support_high, coordinate)) {
                inside[inside_count++] = coordinate;
            } else {
                outside[outside_count++] = coordinate;
            }
        }

        unsigned char accumulated[WITNESS_TARGETS]{};
        for (int derivative_order = 0;
             derivative_order < multiplicity;
             ++derivative_order) {
            if ((derivative_order_mask
                    & (std::uint16_t{1} << derivative_order)) == 0) {
                continue;
            }
            const int selected_inside_count =
                derivative_order - outside_count;
            if (selected_inside_count < 0
                || selected_inside_count > inside_count) {
                continue;
            }
            int indices[MAX_D]{};
            for (int index = 0;
                 index < selected_inside_count;
                 ++index) {
                indices[index] = index;
            }
            bool more = true;
            while (more) {
                unsigned char selected[MAX_D]{};
                int selected_count = 0;
                for (int index = 0; index < outside_count; ++index) {
                    selected[selected_count++] = outside[index];
                }
                for (int index = 0;
                     index < selected_inside_count;
                     ++index) {
                    selected[selected_count++] = inside[indices[index]];
                }

                unsigned char polynomial[MAX_D + 1]{};
                polynomial[0] = 1;
                int polynomial_degree = 0;
                for (int selected_index = 0;
                     selected_index < selected_count;
                     ++selected_index) {
                    unsigned char product[MAX_D + 1]{};
                    const int coordinate = selected[selected_index];
                    for (int current = 0;
                         current <= polynomial_degree;
                         ++current) {
                        for (int jet_order = 1;
                             current + jet_order < multiplicity;
                             ++jet_order) {
                            product[current + jet_order] ^=
                                gf256_multiply_device(
                                    polynomial[current],
                                    jets[(jet_order - 1) * k + coordinate]);
                        }
                    }
                    for (int order = 0; order < multiplicity; ++order) {
                        polynomial[order] = product[order];
                    }
                    const int next_degree =
                        polynomial_degree + multiplicity - 1;
                    polynomial_degree = next_degree < multiplicity
                        ? next_degree
                        : multiplicity - 1;
                }
                for (int target = 0; target < WITNESS_TARGETS; ++target) {
                    for (int order = derivative_order;
                         order < multiplicity;
                         ++order) {
                        accumulated[target] ^=
                            gf256_multiply_device(
                                weights[target * multiplicity + order],
                                polynomial[order]);
                    }
                }
                more = next_indices(
                    indices, selected_inside_count, inside_count);
            }
        }
        coefficients[row_offset + row] =
            static_cast<u64>(accumulated[0])
            | (static_cast<u64>(accumulated[1]) << 8)
            | (static_cast<u64>(accumulated[2]) << 16);
    }
}

__global__ void apply_mt_literal_slice_kernel(
    const u64 *__restrict__ coefficients,
    u64 *__restrict__ output,
    u64 mapping_count,
    u64 columns_per_slice,
    int complement_size,
    int support_column_size,
    int complement_weight,
    int support_weight,
    const unsigned char *__restrict__ complement_positions,
    const unsigned char *__restrict__ support_positions,
    int row_size_count,
    int size0,
    int size1,
    int size2,
    int size3,
    u64 offset0,
    u64 offset1,
    u64 offset2,
    u64 offset3) {
    const int sizes[4] = {size0, size1, size2, size3};
    const u64 offsets[4] = {offset0, offset1, offset2, offset3};
    const u64 stride = static_cast<u64>(blockDim.x) * gridDim.x;
    for (u64 entry = static_cast<u64>(blockIdx.x) * blockDim.x + threadIdx.x;
         entry < mapping_count;
         entry += stride) {
        const u64 slice = entry / columns_per_slice;
        const u64 local_column_rank = entry - slice * columns_per_slice;
        const u64 local_complement = colex_unrank_device(
            slice, complement_size, complement_weight);
        const u64 local_column = colex_unrank_device(
            local_column_rank, support_column_size, support_weight);
        const u64 global_column = colex_rank_merged_local_masks_device(
            local_complement,
            complement_positions,
            local_column,
            support_positions);
        unsigned char positions[MAX_D]{};
        const int position_count = mask_positions(local_column, positions);
        u64 accumulated = 0;
        for (int size_index = 0; size_index < row_size_count; ++size_index) {
            const int row_size = sizes[size_index];
            int indices[MAX_D]{};
            for (int index = 0; index < row_size; ++index) {
                indices[index] = index;
            }
            bool more = row_size <= position_count;
            while (more) {
                u64 local_derivative = 0;
                for (int index = 0; index < row_size; ++index) {
                    local_derivative |= u64{1} << positions[indices[index]];
                }
                const u64 global_derivative =
                    colex_rank_merged_local_masks_device(
                        local_complement,
                        complement_positions,
                        local_derivative,
                        support_positions);
                accumulated ^= coefficients[
                    offsets[size_index] + global_derivative];
                more = next_indices(indices, row_size, position_count);
            }
        }
        output[global_column] ^= accumulated;
    }
}

__global__ void verify_literal_witness_target_kernel(
    const u64 *__restrict__ observed,
    u64 column_count,
    int k,
    int degree,
    u64 target_support,
    u64 target_support_high,
    const unsigned char *__restrict__ target_jets,
    u64 *__restrict__ mismatches) {
    const u64 stride = static_cast<u64>(blockDim.x) * gridDim.x;
    for (u64 column = static_cast<u64>(blockIdx.x) * blockDim.x + threadIdx.x;
         column < column_count;
         column += stride) {
        unsigned char monomial[MAX_D]{};
        colex_unrank_positions_device(column, degree, k, monomial);
        unsigned char polynomial[3] = {1, 0, 0};
        for (int index = 0; index < degree; ++index) {
            const int coordinate = monomial[index];
            const unsigned char point_value =
                two_word_contains_device(
                    target_support, target_support_high, coordinate)
                ? 1
                : 0;
            const unsigned char first = target_jets[coordinate];
            const unsigned char second = target_jets[k + coordinate];
            const unsigned char next0 = gf256_multiply_device(
                polynomial[0], point_value);
            const unsigned char next1 =
                gf256_multiply_device(polynomial[1], point_value)
                ^ gf256_multiply_device(polynomial[0], first);
            const unsigned char next2 =
                gf256_multiply_device(polynomial[2], point_value)
                ^ gf256_multiply_device(polynomial[1], first)
                ^ gf256_multiply_device(polynomial[0], second);
            polynomial[0] = next0;
            polynomial[1] = next1;
            polynomial[2] = next2;
        }
        const u64 expected = static_cast<u64>(polynomial[0])
            | (static_cast<u64>(polynomial[1]) << 8)
            | (static_cast<u64>(polynomial[2]) << 16);
        if (observed[column] != expected) {
            atomicAdd(
                reinterpret_cast<unsigned long long *>(mismatches), 1ULL);
        }
    }
}

__global__ void column_degree_histogram_kernel(
    u64 columns,
    int k,
    int degree,
    const u64 *__restrict__ supports,
    const std::uint16_t *__restrict__ level_masks,
    int point_count,
    u64 *__restrict__ histogram) {
    const u64 stride = static_cast<u64>(blockDim.x) * gridDim.x;
    for (u64 column = static_cast<u64>(blockIdx.x) * blockDim.x + threadIdx.x;
         column < columns;
         column += stride) {
        const u64 monomial = colex_unrank_device(column, degree, k);
        int column_degree = 0;
        for (int point = 0; point < point_count; ++point) {
            const int intersection = __popcll(monomial & supports[point]);
            const int outside = degree - intersection;
            const std::uint16_t levels = level_masks[point];
            for (int level = 0; level <= degree; ++level) {
                if ((levels & (std::uint16_t{1} << level)) != 0) {
                    column_degree += static_cast<int>(
                        device_choose(intersection, level - outside));
                }
            }
        }
        atomicAdd(
            reinterpret_cast<unsigned long long *>(histogram + column_degree),
            1ULL);
    }
}

__global__ void apply_m_slice_pattern_kernel(
    const u64 *__restrict__ gathered,
    u64 *__restrict__ range,
    const std::uint32_t *__restrict__ pattern_columns,
    const std::uint32_t *__restrict__ pattern_starts,
    u64 total_rows,
    u64 rows_per_slice,
    u64 columns_per_slice,
    int words) {
    const u64 total_items = total_rows * words;
    const u64 stride = static_cast<u64>(blockDim.x) * gridDim.x;
    for (u64 item = static_cast<u64>(blockIdx.x) * blockDim.x + threadIdx.x;
         item < total_items;
         item += stride) {
        const u64 row = item / words;
        const int word = static_cast<int>(item - row * words);
        const u64 slice = row / rows_per_slice;
        const u64 local_row = row - slice * rows_per_slice;
        const std::uint32_t begin = pattern_starts[local_row];
        const std::uint32_t end = pattern_starts[local_row + 1];
        u64 accum = 0;
        for (std::uint32_t position = begin; position < end; ++position) {
            const u64 local_column = slice * columns_per_slice
                + pattern_columns[position];
            accum ^= gathered[local_column * words + word];
        }
        range[row * words + word] = accum;
    }
}

__global__ void gather_slice_input_kernel(
    const u64 *__restrict__ input,
    u64 *__restrict__ gathered,
    const std::uint32_t *__restrict__ mapping,
    u64 mapping_count,
    int words) {
    const u64 total_items = mapping_count * words;
    const u64 stride = static_cast<u64>(blockDim.x) * gridDim.x;
    for (u64 item = static_cast<u64>(blockIdx.x) * blockDim.x + threadIdx.x;
         item < total_items;
         item += stride) {
        const u64 entry = item / words;
        const int word = static_cast<int>(item - entry * words);
        gathered[item] = input[static_cast<u64>(mapping[entry]) * words + word];
    }
}

__global__ void project_partial_kernel(
    const u64 *__restrict__ z,
    const u64 *__restrict__ v,
    u64 *__restrict__ partials,
    u64 column_count,
    int words,
    int chunks) {
    const int lane = threadIdx.x & 31;
    const u64 warps_per_block = blockDim.x / 32;
    const u64 warp = static_cast<u64>(blockIdx.x) * warps_per_block
        + threadIdx.x / 32;
    const u64 warp_count = static_cast<u64>(words) * words * chunks;
    if (warp >= warp_count) {
        return;
    }
    const int chunk = static_cast<int>(warp % chunks);
    const u64 tile = warp / chunks;
    const int column_word = static_cast<int>(tile % words);
    const int row_word = static_cast<int>(tile / words);
    const u64 begin = column_count * chunk / chunks;
    const u64 end = column_count * (chunk + 1) / chunks;
    u64 low_accum = 0;
    u64 high_accum = 0;
    for (u64 column = begin; column < end; ++column) {
        unsigned long long zword = 0;
        unsigned long long vword = 0;
        if (lane == 0) {
            zword = static_cast<unsigned long long>(
                z[column * words + row_word]);
            vword = static_cast<unsigned long long>(
                v[column * words + column_word]);
        }
        zword = __shfl_sync(0xffffffffU, zword, 0);
        vword = __shfl_sync(0xffffffffU, vword, 0);
        if ((zword >> lane) & 1ULL) {
            low_accum ^= static_cast<u64>(vword);
        }
        if ((zword >> (lane + 32)) & 1ULL) {
            high_accum ^= static_cast<u64>(vword);
        }
    }
    const u64 base = (tile * chunks + chunk) * 64;
    partials[base + lane] = low_accum;
    partials[base + lane + 32] = high_accum;
}

__global__ void project_reduce_kernel(
    const u64 *__restrict__ partials,
    u64 *__restrict__ output,
    int words,
    int chunks) {
    const u64 output_count = static_cast<u64>(64) * words * words;
    const u64 stride = static_cast<u64>(blockDim.x) * gridDim.x;
    for (u64 item = static_cast<u64>(blockIdx.x) * blockDim.x + threadIdx.x;
         item < output_count;
         item += stride) {
        const int row_within_word = static_cast<int>(item % 64);
        const u64 tile = item / 64;
        u64 accum = 0;
        const u64 base = tile * chunks * 64 + row_within_word;
        for (int chunk = 0; chunk < chunks; ++chunk) {
            accum ^= partials[base + static_cast<u64>(chunk) * 64];
        }
        const int column_word = static_cast<int>(tile % words);
        const int row_word = static_cast<int>(tile / words);
        const int output_row = row_word * 64 + row_within_word;
        output[static_cast<u64>(output_row) * words + column_word] = accum;
    }
}

template<int WORD_CAP>
__global__ void apply_mt_slice_kernel(
    const u64 *__restrict__ range,
    u64 *__restrict__ output,
    const std::uint32_t *__restrict__ mapping,
    u64 mapping_count,
    u64 columns_per_slice,
    u64 rows_per_slice,
    int support_weight,
    int support_column_size,
    int words,
    int row_size_count,
    int size0,
    int size1,
    int size2,
    int size3,
    u64 offset0,
    u64 offset1,
    u64 offset2,
    u64 offset3) {
    const int sizes[4] = {size0, size1, size2, size3};
    const u64 offsets[4] = {offset0, offset1, offset2, offset3};
    const u64 stride = static_cast<u64>(blockDim.x) * gridDim.x;
    for (u64 entry = static_cast<u64>(blockIdx.x) * blockDim.x + threadIdx.x;
         entry < mapping_count;
         entry += stride) {
        const u64 slice = entry / columns_per_slice;
        const u64 local_column_rank = entry - slice * columns_per_slice;
        const u64 local_column = colex_unrank_device(
            local_column_rank, support_column_size, support_weight);
        unsigned char positions[MAX_D];
        const int position_count = mask_positions(local_column, positions);
        u64 accum[WORD_CAP]{};
        for (int size_index = 0; size_index < row_size_count; ++size_index) {
            const int row_size = sizes[size_index];
            int indices[MAX_D]{};
            for (int index = 0; index < row_size; ++index) {
                indices[index] = index;
            }
            bool more = row_size <= position_count;
            while (more) {
                u64 local_derivative = 0;
                for (int index = 0; index < row_size; ++index) {
                    local_derivative |= u64{1} << positions[indices[index]];
                }
                const u64 local_row = offsets[size_index]
                    + colex_rank_device(local_derivative);
                const u64 *source = range
                    + (slice * rows_per_slice + local_row) * words;
#pragma unroll
                for (int word = 0; word < WORD_CAP; ++word) {
                    if (word < words) {
                        accum[word] ^= source[word];
                    }
                }
                more = next_indices(indices, row_size, position_count);
            }
        }
        u64 *destination = output + static_cast<u64>(mapping[entry]) * words;
#pragma unroll
        for (int word = 0; word < WORD_CAP; ++word) {
            if (word < words) {
                destination[word] ^= accum[word];
            }
        }
    }
}

template<int WORD_CAP>
__global__ void apply_m_slice_recompute_kernel(
    const u64 *__restrict__ input,
    u64 *__restrict__ range,
    u64 total_rows,
    u64 rows_per_slice,
    int complement_size,
    int support_column_size,
    int complement_weight,
    int support_weight,
    const unsigned char *__restrict__ complement_positions,
    const unsigned char *__restrict__ support_positions,
    int words,
    int row_size_count,
    int size0,
    int size1,
    int size2,
    int size3,
    u64 offset0,
    u64 offset1,
    u64 offset2,
    u64 offset3) {
    const int sizes[4] = {size0, size1, size2, size3};
    const u64 offsets[4] = {offset0, offset1, offset2, offset3};
    const u64 stride = static_cast<u64>(blockDim.x) * gridDim.x;
    for (u64 row = static_cast<u64>(blockIdx.x) * blockDim.x + threadIdx.x;
         row < total_rows;
         row += stride) {
        const u64 slice = row / rows_per_slice;
        const u64 local_row = row - slice * rows_per_slice;
        int row_size_index = -1;
        for (int index = 0; index < row_size_count; ++index) {
            const u64 count = device_choose(support_weight, sizes[index]);
            if (local_row >= offsets[index]
                && local_row - offsets[index] < count) {
                row_size_index = index;
                break;
            }
        }
        if (row_size_index < 0) {
            continue;
        }
        const int row_size = sizes[row_size_index];
        const u64 local_derivative = colex_unrank_device(
            local_row - offsets[row_size_index],
            row_size,
            support_weight);
        const u64 local_complement = colex_unrank_device(
            slice, complement_size, complement_weight);
        const u64 support_mask = support_weight == 64
            ? ~u64{0}
            : ((u64{1} << support_weight) - 1);
        const u64 available = support_mask & ~local_derivative;
        const int extension_size = support_column_size - row_size;
        unsigned char available_positions[MAX_K]{};
        const int available_count = mask_positions(
            available, available_positions);
        u64 accum[WORD_CAP]{};
        if (extension_size <= available_count) {
            int indices[MAX_D]{};
            for (int index = 0; index < extension_size; ++index) {
                indices[index] = index;
            }
            bool more = true;
            while (more) {
                u64 extension = 0;
                for (int index = 0; index < extension_size; ++index) {
                    extension |= u64{1}
                        << available_positions[indices[index]];
                }
                const u64 global_column =
                    colex_rank_merged_local_masks_device(
                        local_complement,
                        complement_positions,
                        local_derivative | extension,
                        support_positions);
                const u64 *source = input + global_column * words;
#pragma unroll
                for (int word = 0; word < WORD_CAP; ++word) {
                    if (word < words) {
                        accum[word] ^= source[word];
                    }
                }
                more = next_indices(
                    indices, extension_size, available_count);
            }
        }
        u64 *destination = range + row * words;
#pragma unroll
        for (int word = 0; word < WORD_CAP; ++word) {
            if (word < words) {
                destination[word] = accum[word];
            }
        }
    }
}

template<int WORD_CAP>
__global__ void apply_mt_slice_recompute_kernel(
    const u64 *__restrict__ range,
    u64 *__restrict__ output,
    u64 mapping_count,
    u64 columns_per_slice,
    u64 rows_per_slice,
    int complement_size,
    int support_column_size,
    int complement_weight,
    int support_weight,
    const unsigned char *__restrict__ complement_positions,
    const unsigned char *__restrict__ support_positions,
    int words,
    int row_size_count,
    int size0,
    int size1,
    int size2,
    int size3,
    u64 offset0,
    u64 offset1,
    u64 offset2,
    u64 offset3) {
    const int sizes[4] = {size0, size1, size2, size3};
    const u64 offsets[4] = {offset0, offset1, offset2, offset3};
    const u64 stride = static_cast<u64>(blockDim.x) * gridDim.x;
    for (u64 entry = static_cast<u64>(blockIdx.x) * blockDim.x + threadIdx.x;
         entry < mapping_count;
         entry += stride) {
        const u64 slice = entry / columns_per_slice;
        const u64 local_column_rank = entry - slice * columns_per_slice;
        const u64 local_complement = colex_unrank_device(
            slice, complement_size, complement_weight);
        const u64 local_column = colex_unrank_device(
            local_column_rank, support_column_size, support_weight);
        const u64 global_column = colex_rank_merged_local_masks_device(
            local_complement,
            complement_positions,
            local_column,
            support_positions);
        unsigned char positions[MAX_D]{};
        const int position_count = mask_positions(local_column, positions);
        u64 accum[WORD_CAP]{};
        for (int size_index = 0; size_index < row_size_count; ++size_index) {
            const int row_size = sizes[size_index];
            int indices[MAX_D]{};
            for (int index = 0; index < row_size; ++index) {
                indices[index] = index;
            }
            bool more = row_size <= position_count;
            while (more) {
                u64 local_derivative = 0;
                for (int index = 0; index < row_size; ++index) {
                    local_derivative |= u64{1} << positions[indices[index]];
                }
                const u64 local_row = offsets[size_index]
                    + colex_rank_device(local_derivative);
                const u64 *source = range
                    + (slice * rows_per_slice + local_row) * words;
#pragma unroll
                for (int word = 0; word < WORD_CAP; ++word) {
                    if (word < words) {
                        accum[word] ^= source[word];
                    }
                }
                more = next_indices(indices, row_size, position_count);
            }
        }
        u64 *destination = output + global_column * words;
#pragma unroll
        for (int word = 0; word < WORD_CAP; ++word) {
            if (word < words) {
                destination[word] ^= accum[word];
            }
        }
    }
}

template<int WORD_CAP>
__global__ void apply_m_level_kernel(
    const u64 *__restrict__ input,
    u64 *__restrict__ range,
    u64 row_count,
    u64 row_offset,
    u64 support,
    int k,
    int degree,
    int level,
    int words) {
    const u64 stride = static_cast<u64>(blockDim.x) * gridDim.x;
    for (u64 row = static_cast<u64>(blockIdx.x) * blockDim.x + threadIdx.x;
         row < row_count;
         row += stride) {
        const u64 derivative = colex_unrank_device(row, level, k);
        const u64 available = support & ~derivative;
        const int outside_size = degree - level;
        unsigned char positions[MAX_K];
        const int position_count = mask_positions(available, positions);
        u64 accum[WORD_CAP]{};

        if (outside_size <= position_count) {
            int indices[MAX_D]{};
            for (int index = 0; index < outside_size; ++index) {
                indices[index] = index;
            }
            bool more = true;
            while (more) {
                u64 outside = 0;
                for (int index = 0; index < outside_size; ++index) {
                    outside |= u64{1} << positions[indices[index]];
                }
                const u64 column = colex_rank_device(derivative | outside);
                const u64 *source = input + column * words;
#pragma unroll
                for (int word = 0; word < WORD_CAP; ++word) {
                    if (word < words) {
                        accum[word] ^= source[word];
                    }
                }
                more = next_indices(indices, outside_size, position_count);
            }
        }

        u64 *destination = range + (row_offset + row) * words;
#pragma unroll
        for (int word = 0; word < WORD_CAP; ++word) {
            if (word < words) {
                destination[word] = accum[word];
            }
        }
    }
}

template<int WORD_CAP>
__global__ void apply_mt_point_kernel(
    const u64 *__restrict__ range,
    u64 *__restrict__ output,
    u64 column_count,
    u64 support,
    int k,
    int degree,
    int words,
    int level_count,
    int level0,
    int level1,
    int level2,
    int level3,
    u64 offset0,
    u64 offset1,
    u64 offset2,
    u64 offset3) {
    const int levels[4] = {level0, level1, level2, level3};
    const u64 offsets[4] = {offset0, offset1, offset2, offset3};
    const u64 stride = static_cast<u64>(blockDim.x) * gridDim.x;
    for (u64 column = static_cast<u64>(blockIdx.x) * blockDim.x + threadIdx.x;
         column < column_count;
         column += stride) {
        const u64 monomial = colex_unrank_device(column, degree, k);
        u64 accum[WORD_CAP]{};
        for (int level_index = 0; level_index < level_count; ++level_index) {
            const int level = levels[level_index];
            const int outside_size = degree - level;
            const u64 available = monomial & support;
            unsigned char positions[MAX_D];
            const int position_count = mask_positions(available, positions);
            if (outside_size > position_count) {
                continue;
            }
            int indices[MAX_D]{};
            for (int index = 0; index < outside_size; ++index) {
                indices[index] = index;
            }
            bool more = true;
            while (more) {
                u64 outside = 0;
                for (int index = 0; index < outside_size; ++index) {
                    outside |= u64{1} << positions[indices[index]];
                }
                const u64 derivative = monomial ^ outside;
                const u64 row = offsets[level_index] + colex_rank_device(derivative);
                const u64 *source = range + row * words;
#pragma unroll
                for (int word = 0; word < WORD_CAP; ++word) {
                    if (word < words) {
                        accum[word] ^= source[word];
                    }
                }
                more = next_indices(indices, outside_size, position_count);
            }
        }
        u64 *destination = output + column * words;
#pragma unroll
        for (int word = 0; word < WORD_CAP; ++word) {
            if (word < words) {
                destination[word] ^= accum[word];
            }
        }
    }
}

int launch_blocks(u64 items) {
    const u64 needed = (items + THREADS - 1) / THREADS;
    return static_cast<int>(std::min<u64>(needed, MAX_BLOCKS));
}

u64 next_random(u64 &state) {
    state ^= state << 13;
    state ^= state >> 7;
    state ^= state << 17;
    return state;
}

std::vector<u64> random_words(std::size_t count, u64 seed) {
    std::vector<u64> output(count);
    u64 state = seed | 1;
    for (u64 &value : output) {
        value = next_random(state);
    }
    return output;
}

template <class Callback>
void each_subset_host(u64 available, int size, Callback callback) {
    std::array<unsigned char, MAX_K> positions{};
    int count = 0;
    while (available != 0) {
        positions[count++] = static_cast<unsigned char>(low_bit(available));
        available &= available - 1;
    }
    if (size > count) {
        return;
    }
    std::array<int, MAX_D> indices{};
    for (int index = 0; index < size; ++index) {
        indices[index] = index;
    }
    bool more = true;
    while (more) {
        u64 subset = 0;
        for (int index = 0; index < size; ++index) {
            subset |= u64{1} << positions[indices[index]];
        }
        callback(subset);
        if (size == 0) {
            more = false;
            continue;
        }
        int cursor = size - 1;
        while (cursor >= 0 && indices[cursor] == cursor + count - size) {
            --cursor;
        }
        if (cursor < 0) {
            more = false;
        } else {
            ++indices[cursor];
            for (int index = cursor + 1; index < size; ++index) {
                indices[index] = indices[index - 1] + 1;
            }
        }
    }
}

void cpu_apply_m_point(
    const Request &request,
    const Point &point,
    const std::vector<u64> &input,
    std::vector<u64> &range) {
    std::fill(range.begin(), range.end(), 0);
    for (std::size_t level_index = 0; level_index < point.levels.size(); ++level_index) {
        const int level = point.levels[level_index];
        const u64 count = HOST_CHOOSE.get(request.k, level);
        for (u64 row = 0; row < count; ++row) {
            const u64 derivative = colex_unrank_host(row, level, request.k);
            each_subset_host(point.support & ~derivative, request.degree - level, [&](u64 outside) {
                const u64 column = colex_rank_host(derivative | outside);
                for (int word = 0; word < request.words; ++word) {
                    range[(point.offsets[level_index] + row) * request.words + word] ^=
                        input[column * request.words + word];
                }
            });
        }
    }
}

void cpu_apply_mt_point(
    const Request &request,
    const Point &point,
    const std::vector<u64> &range,
    std::vector<u64> &output) {
    const u64 columns = HOST_CHOOSE.get(request.k, request.degree);
    for (u64 column = 0; column < columns; ++column) {
        const u64 monomial = colex_unrank_host(column, request.degree, request.k);
        for (std::size_t level_index = 0; level_index < point.levels.size(); ++level_index) {
            const int level = point.levels[level_index];
            each_subset_host(monomial & point.support, request.degree - level, [&](u64 outside) {
                const u64 derivative = monomial ^ outside;
                const u64 row = point.offsets[level_index] + colex_rank_host(derivative);
                for (int word = 0; word < request.words; ++word) {
                    output[column * request.words + word] ^=
                        range[row * request.words + word];
                }
            });
        }
    }
}

std::vector<u64> cpu_apply_b(const Request &request, const std::vector<u64> &input) {
    require_one_word_ground_set(request, "CPU reference backend");
    const u64 columns = HOST_CHOOSE.get(request.k, request.degree);
    std::vector<u64> output(static_cast<std::size_t>(columns) * request.words, 0);
    for (const Point &point : request.points) {
        std::vector<u64> range(static_cast<std::size_t>(point.rows) * request.words, 0);
        cpu_apply_m_point(request, point, input, range);
        cpu_apply_mt_point(request, point, range, output);
    }
    return output;
}

std::vector<u64> cpu_project(
    const Request &request,
    const std::vector<u64> &z,
    const std::vector<u64> &v) {
    const int block_bits = request.words * 64;
    std::vector<u64> output(
        static_cast<std::size_t>(block_bits) * request.words, 0);
    for (u64 column = 0; column < request.expected_columns; ++column) {
        const u64 *z_row = z.data() + column * request.words;
        const u64 *v_row = v.data() + column * request.words;
        for (int row_word = 0; row_word < request.words; ++row_word) {
            u64 mask = z_row[row_word];
            while (mask != 0) {
                const int output_row = row_word * 64 + low_bit(mask);
                mask &= mask - 1;
                for (int column_word = 0;
                     column_word < request.words;
                     ++column_word) {
                    output[static_cast<std::size_t>(output_row) * request.words
                        + column_word] ^= v_row[column_word];
                }
            }
        }
    }
    return output;
}

u64 point_nonzeros(const Request &request, const Point &point) {
    const int weight = point_weight(point);
    u64 total = 0;
    for (const int level : point.levels) {
        for (int intersection = 0; intersection <= level; ++intersection) {
            total += HOST_CHOOSE.get(weight, intersection)
                * HOST_CHOOSE.get(request.k - weight, level - intersection)
                * HOST_CHOOSE.get(weight - intersection, request.degree - level);
        }
    }
    return total;
}

void finish_layout(Request &request) {
    if (request.k <= 0 || request.k > MAX_K) {
        throw std::runtime_error("k is outside 1..68");
    }
    if (request.schema == REQUEST_SCHEMA && request.k > LEGACY_MAX_K) {
        throw std::runtime_error(
            "legacy request schema is restricted to k<=63");
    }
    if (request.degree <= 0 || request.degree > MAX_D || request.degree > request.k) {
        throw std::runtime_error("degree is outside the CUDA combinadic contract");
    }
    if (request.words <= 0 || request.words > MAX_WORDS) {
        throw std::runtime_error("words is outside 1..16");
    }
    if (request.repetitions <= 0) {
        throw std::runtime_error("repetitions must be positive");
    }
    if (request.emit_output != 0 && request.emit_output != 1) {
        throw std::runtime_error("emit_output must be zero or one");
    }
    if (request.points.empty()) {
        throw std::runtime_error("point list is empty");
    }
    const u64 valid_low_mask = request.k >= 64
        ? std::numeric_limits<u64>::max()
        : ((u64{1} << request.k) - 1);
    const int high_bits = std::max(0, request.k - 64);
    const u64 valid_high_mask = high_bits == 0
        ? 0
        : ((u64{1} << high_bits) - 1);
    u64 rows = 0;
    u64 nonzeros = 0;
    for (Point &point : request.points) {
        if ((point.support & ~valid_low_mask) != 0
            || (point.support_high & ~valid_high_mask) != 0) {
            throw std::runtime_error("point support leaves the ground set");
        }
        if (point.levels.empty() || point.levels.size() > 4) {
            throw std::runtime_error("this baseline supports one through four levels per point");
        }
        if (!std::is_sorted(point.levels.begin(), point.levels.end())
            || std::adjacent_find(point.levels.begin(), point.levels.end()) != point.levels.end()) {
            throw std::runtime_error("levels must be strictly increasing");
        }
        point.offsets.clear();
        point.rows = 0;
        for (const int level : point.levels) {
            if (level < 0 || level > request.degree) {
                throw std::runtime_error("level is outside 0..degree");
            }
            point.offsets.push_back(point.rows);
            point.rows += HOST_CHOOSE.get(request.k, level);
        }
        point.nonzeros = point_nonzeros(request, point);
        rows += point.rows;
        nonzeros += point.nonzeros;
    }
    const u64 columns = HOST_CHOOSE.get(request.k, request.degree);
    if (request.expected_columns != columns
        || request.expected_rows != rows
        || request.expected_nonzeros != nonzeros) {
        std::ostringstream message;
        message << "declared layout differs: columns " << columns << "/"
                << request.expected_columns << ", rows " << rows << "/"
                << request.expected_rows << ", nonzeros " << nonzeros << "/"
                << request.expected_nonzeros;
        throw std::runtime_error(message.str());
    }
    if (request.emit_output && columns * static_cast<u64>(request.words) > 65536) {
        throw std::runtime_error("emit_output is restricted to at most 65536 u64 words");
    }
}

u64 checked_add_u64(u64 left, u64 right, const char *label) {
    if (right > std::numeric_limits<u64>::max() - left) {
        throw std::runtime_error(std::string("u64 overflow in ") + label);
    }
    return left + right;
}

u64 checked_multiply_u64(u64 left, u64 right, const char *label) {
    if (left != 0 && right > std::numeric_limits<u64>::max() / left) {
        throw std::runtime_error(std::string("u64 overflow in ") + label);
    }
    return left * right;
}

bool same_local_pattern(
    const LocalPatternPlan &pattern,
    int support_weight,
    const SliceClass &slice_class) {
    if (pattern.support_weight != support_weight
        || pattern.support_column_size != slice_class.support_column_size
        || pattern.row_size_count != slice_class.row_size_count) {
        return false;
    }
    for (int index = 0; index < pattern.row_size_count; ++index) {
        if (pattern.row_sizes[index] != slice_class.row_sizes[index]) {
            return false;
        }
    }
    return true;
}

std::size_t register_local_pattern(
    SlicePlan &plan,
    int support_weight,
    const SliceClass &slice_class) {
    for (std::size_t index = 0; index < plan.patterns.size(); ++index) {
        if (same_local_pattern(
                plan.patterns[index], support_weight, slice_class)) {
            return index;
        }
    }
    LocalPatternPlan pattern;
    pattern.support_weight = support_weight;
    pattern.support_column_size = slice_class.support_column_size;
    pattern.row_sizes = slice_class.row_sizes;
    pattern.row_size_count = slice_class.row_size_count;
    pattern.row_count = slice_class.rows_per_slice;
    for (int index = 0; index < pattern.row_size_count; ++index) {
        const int row_size = pattern.row_sizes[index];
        pattern.incidence_count = checked_add_u64(
            pattern.incidence_count,
            checked_multiply_u64(
                HOST_CHOOSE.get(support_weight, row_size),
                HOST_CHOOSE.get(
                    support_weight - row_size,
                    pattern.support_column_size - row_size),
                "local pattern incidence count"),
            "local pattern incidence count");
    }
    if (pattern.incidence_count > std::numeric_limits<std::uint32_t>::max()) {
        throw std::runtime_error("one local incidence pattern exceeds u32 offsets");
    }
    plan.patterns.push_back(pattern);
    return plan.patterns.size() - 1;
}

void materialize_local_patterns(SlicePlan &plan) {
    u64 total_columns = 0;
    u64 total_starts = 0;
    for (const LocalPatternPlan &pattern : plan.patterns) {
        total_columns = checked_add_u64(
            total_columns, pattern.incidence_count, "all local pattern columns");
        total_starts = checked_add_u64(
            total_starts, pattern.row_count + 1, "all local pattern starts");
    }
    if (total_columns > std::numeric_limits<std::size_t>::max()
        || total_starts > std::numeric_limits<std::size_t>::max()) {
        throw std::runtime_error("local pattern tables exceed host size_t");
    }
    plan.pattern_columns.reserve(static_cast<std::size_t>(total_columns));
    plan.pattern_starts.reserve(static_cast<std::size_t>(total_starts));
    for (LocalPatternPlan &pattern : plan.patterns) {
        pattern.columns_offset = plan.pattern_columns.size();
        pattern.starts_offset = plan.pattern_starts.size();
        plan.pattern_starts.push_back(0);
        const u64 local_mask = pattern.support_weight == 64
            ? ~u64{0}
            : ((u64{1} << pattern.support_weight) - 1);
        for (int size_index = 0;
             size_index < pattern.row_size_count;
             ++size_index) {
            const int row_size = pattern.row_sizes[size_index];
            const u64 row_count = HOST_CHOOSE.get(
                pattern.support_weight, row_size);
            for (u64 row = 0; row < row_count; ++row) {
                const u64 derivative = colex_unrank_host(
                    row, row_size, pattern.support_weight);
                each_subset_host(
                    local_mask & ~derivative,
                    pattern.support_column_size - row_size,
                    [&](u64 outside) {
                        plan.pattern_columns.push_back(
                            static_cast<std::uint32_t>(
                                colex_rank_host(derivative | outside)));
                    });
                const u64 local_count = plan.pattern_columns.size()
                    - pattern.columns_offset;
                if (local_count > std::numeric_limits<std::uint32_t>::max()) {
                    throw std::runtime_error("local pattern offset exceeds u32");
                }
                plan.pattern_starts.push_back(
                    static_cast<std::uint32_t>(local_count));
            }
        }
        if (plan.pattern_columns.size() - pattern.columns_offset
                != pattern.incidence_count
            || plan.pattern_starts.size() - pattern.starts_offset
                != pattern.row_count + 1) {
            throw std::runtime_error("materialized local pattern count differs");
        }
    }
}

SlicePlan build_slice_plan(
    const Request &request, bool materialize_patterns = true) {
    if (request.expected_columns > std::numeric_limits<std::uint32_t>::max()) {
        throw std::runtime_error("slice column map requires fewer than 2^32 columns");
    }
    SlicePlan plan;
    plan.points.resize(request.points.size());
    for (std::size_t point_index = 0; point_index < request.points.size(); ++point_index) {
        const Point &point = request.points[point_index];
        const int support_weight = point_weight(point);
        const int complement_weight = request.k - support_weight;
        if (support_weight > LEGACY_MAX_K
            || complement_weight > LEGACY_MAX_K) {
            throw std::runtime_error(
                "slice local ground set exceeds the 63-bit local-mask contract");
        }
        SlicePointPlan &point_plan = plan.points[point_index];
        for (int complement_size = 0;
             complement_size <= request.degree;
             ++complement_size) {
            const int support_column_size = request.degree - complement_size;
            if (complement_size > complement_weight
                || support_column_size > support_weight) {
                continue;
            }
            SliceClass slice_class;
            slice_class.complement_size = complement_size;
            slice_class.support_column_size = support_column_size;
            slice_class.slice_count = HOST_CHOOSE.get(
                complement_weight, complement_size);
            slice_class.columns_per_slice = HOST_CHOOSE.get(
                support_weight, support_column_size);
            for (const int level : point.levels) {
                if (level < complement_size) {
                    continue;
                }
                const int row_size = level - complement_size;
                if (row_size > support_column_size) {
                    continue;
                }
                const int row_index = slice_class.row_size_count++;
                slice_class.row_sizes[row_index] = row_size;
                slice_class.row_offsets[row_index] = slice_class.rows_per_slice;
                slice_class.rows_per_slice = checked_add_u64(
                    slice_class.rows_per_slice,
                    HOST_CHOOSE.get(support_weight, row_size),
                    "slice rows per slice");
            }
            if (slice_class.row_size_count == 0) {
                continue;
            }
            slice_class.mapping_offset = plan.mapping_entries;
            const u64 mapping_count = checked_multiply_u64(
                slice_class.slice_count,
                slice_class.columns_per_slice,
                "slice mapping entries");
            plan.mapping_entries = checked_add_u64(
                plan.mapping_entries, mapping_count, "total slice mapping entries");
            plan.max_class_mapping_entries = std::max(
                plan.max_class_mapping_entries, mapping_count);
            const u64 total_rows = checked_multiply_u64(
                slice_class.slice_count,
                slice_class.rows_per_slice,
                "slice range rows");
            slice_class.range_offset = plan.total_range_rows;
            plan.total_range_rows = checked_add_u64(
                plan.total_range_rows,
                total_rows,
                "total slice range rows");
            plan.max_range_rows = std::max(plan.max_range_rows, total_rows);
            if (materialize_patterns) {
                slice_class.pattern_index = register_local_pattern(
                    plan, support_weight, slice_class);
            }
            point_plan.classes.push_back(slice_class);
        }
    }
    if (plan.mapping_entries == 0 || plan.max_range_rows == 0) {
        throw std::runtime_error("slice plan has no active incidence classes");
    }
    if (plan.total_range_rows != request.expected_rows) {
        throw std::runtime_error(
            "slice range ordering is not a permutation of literal rows");
    }
    if (materialize_patterns) {
        materialize_local_patterns(plan);
    }
    return plan;
}

Request read_request(const std::filesystem::path &path) {
    std::ifstream stream(path);
    if (!stream) {
        throw std::runtime_error("cannot open request: " + path.string());
    }
    std::string schema;
    std::getline(stream, schema);
    if (schema != REQUEST_SCHEMA && schema != EXTENDED_REQUEST_SCHEMA) {
        throw std::runtime_error("request schema differs");
    }
    Request request;
    request.schema = schema;
    std::string key;
    std::size_t point_count = 0;
    auto require_key = [&](const char *expected) {
        if (!(stream >> key) || key != expected) {
            throw std::runtime_error(std::string("expected request key ") + expected);
        }
    };
    require_key("public_sha256"); stream >> request.public_sha256;
    require_key("k"); stream >> request.k;
    require_key("degree"); stream >> request.degree;
    require_key("words"); stream >> request.words;
    require_key("repetitions"); stream >> request.repetitions;
    require_key("seed"); stream >> request.seed;
    require_key("emit_output"); stream >> request.emit_output;
    require_key("expected_columns"); stream >> request.expected_columns;
    require_key("expected_rows"); stream >> request.expected_rows;
    require_key("expected_nonzeros"); stream >> request.expected_nonzeros;
    require_key("point_count"); stream >> point_count;
    request.points.reserve(point_count);
    for (std::size_t index = 0; index < point_count; ++index) {
        require_key("point");
        Point point;
        std::size_t level_count = 0;
        stream >> point.support;
        if (schema == EXTENDED_REQUEST_SCHEMA) {
            stream >> point.support_high;
        }
        stream >> level_count;
        point.levels.resize(level_count);
        for (int &level : point.levels) {
            stream >> level;
        }
        if (!stream) {
            throw std::runtime_error("truncated point record");
        }
        request.points.push_back(std::move(point));
    }
    std::string trailing;
    if (stream >> trailing) {
        throw std::runtime_error("unexpected trailing request data");
    }
    finish_layout(request);
    return request;
}

int hex_nibble(char value) {
    if (value >= '0' && value <= '9') {
        return value - '0';
    }
    if (value >= 'a' && value <= 'f') {
        return value - 'a' + 10;
    }
    if (value >= 'A' && value <= 'F') {
        return value - 'A' + 10;
    }
    return -1;
}

std::vector<unsigned char> decode_hex_bytes(
    const std::string &text, std::size_t expected_bytes, const char *label) {
    if (text.size() != expected_bytes * 2) {
        throw std::runtime_error(
            std::string(label) + " hex payload has the wrong length");
    }
    std::vector<unsigned char> output(expected_bytes);
    for (std::size_t index = 0; index < expected_bytes; ++index) {
        const int high = hex_nibble(text[2 * index]);
        const int low = hex_nibble(text[2 * index + 1]);
        if (high < 0 || low < 0) {
            throw std::runtime_error(
                std::string(label) + " hex payload is not hexadecimal");
        }
        output[index] = static_cast<unsigned char>((high << 4) | low);
    }
    return output;
}

int visible_lift_level_host(
    int degree, int derivative_order, const std::vector<int> &levels) {
    for (const int level : levels) {
        if (level >= derivative_order
            && (HOST_CHOOSE.get(
                    degree - derivative_order,
                    level - derivative_order) & 1U) != 0) {
            return level;
        }
    }
    return -1;
}

WitnessRequest read_witness_request(const std::filesystem::path &path) {
    std::ifstream stream(path);
    if (!stream) {
        throw std::runtime_error(
            "cannot open literal-witness request: " + path.string());
    }
    std::string schema;
    std::getline(stream, schema);
    if (schema != WITNESS_REQUEST_SCHEMA) {
        throw std::runtime_error("literal-witness request schema differs");
    }
    WitnessRequest request;
    request.layout.schema = schema;
    request.layout.words = 1;
    request.layout.repetitions = 1;
    request.layout.emit_output = 0;
    std::string key;
    std::size_t point_count = 0;
    int target_count = 0;
    auto require_key = [&](const char *expected) {
        if (!(stream >> key) || key != expected) {
            throw std::runtime_error(
                std::string("expected literal-witness request key ")
                + expected);
        }
    };
    require_key("public_sha256");
    stream >> request.layout.public_sha256;
    require_key("k");
    stream >> request.layout.k;
    require_key("degree");
    stream >> request.layout.degree;
    require_key("expected_columns");
    stream >> request.layout.expected_columns;
    require_key("expected_rows");
    stream >> request.layout.expected_rows;
    require_key("expected_nonzeros");
    stream >> request.layout.expected_nonzeros;
    require_key("target_count");
    stream >> target_count;
    require_key("emit_coefficients");
    stream >> request.emit_coefficients;
    require_key("verify_target");
    stream >> request.verify_target;
    if (request.verify_target != 0 && request.verify_target != 1) {
        throw std::runtime_error("verify_target must be zero or one");
    }
    if (request.verify_target) {
        require_key("target_point");
        stream >> request.target_support >> request.target_support_high;
        require_key("target_jets_hex");
        std::string target_jets_hex;
        stream >> target_jets_hex;
        request.target_jets = decode_hex_bytes(
            target_jets_hex,
            static_cast<std::size_t>(2) * request.layout.k,
            "target jets");
    }
    require_key("point_count");
    stream >> point_count;
    if (target_count != WITNESS_TARGETS) {
        throw std::runtime_error(
            "literal-witness request must contain exactly three targets");
    }
    if (request.emit_coefficients != 0 && request.emit_coefficients != 1) {
        throw std::runtime_error(
            "emit_coefficients must be zero or one");
    }
    request.layout.points.reserve(point_count);
    request.point_data.reserve(point_count);
    for (std::size_t point_index = 0;
         point_index < point_count;
         ++point_index) {
        require_key("point");
        Point point;
        std::size_t level_count = 0;
        stream >> point.support >> point.support_high >> level_count;
        point.levels.resize(level_count);
        for (int &level : point.levels) {
            stream >> level;
        }
        require_key("multiplicity");
        WitnessPointData data;
        stream >> data.multiplicity;
        if (data.multiplicity < 2 || data.multiplicity > MAX_D) {
            throw std::runtime_error(
                "literal-witness multiplicity is outside 2..9");
        }
        require_key("jets_hex");
        std::string jets_hex;
        stream >> jets_hex;
        data.jets = decode_hex_bytes(
            jets_hex,
            static_cast<std::size_t>(data.multiplicity - 1)
                * request.layout.k,
            "jets");
        require_key("weights_hex");
        std::string weights_hex;
        stream >> weights_hex;
        data.weights = decode_hex_bytes(
            weights_hex,
            static_cast<std::size_t>(WITNESS_TARGETS)
                * data.multiplicity,
            "weights");
        if (!stream) {
            throw std::runtime_error(
                "truncated literal-witness point record");
        }
        request.layout.points.push_back(std::move(point));
        request.point_data.push_back(std::move(data));
    }
    std::string trailing;
    if (stream >> trailing) {
        throw std::runtime_error(
            "unexpected trailing literal-witness request data");
    }
    finish_layout(request.layout);
    const Point target_point{
        request.target_support,
        request.target_support_high,
        {},
        {},
        0,
        0,
    };
    if (request.verify_target) {
        const u64 valid_low_mask = request.layout.k >= 64
            ? std::numeric_limits<u64>::max()
            : ((u64{1} << request.layout.k) - 1);
        const int high_bits = std::max(0, request.layout.k - 64);
        const u64 valid_high_mask = high_bits == 0
            ? 0
            : ((u64{1} << high_bits) - 1);
        if ((target_point.support & ~valid_low_mask) != 0
            || (target_point.support_high & ~valid_high_mask) != 0) {
            throw std::runtime_error(
                "literal-witness target point leaves the ground set");
        }
    }
    if (request.emit_coefficients
        && request.layout.expected_rows > 65536) {
        throw std::runtime_error(
            "emitted literal coefficients are restricted to 65536 rows");
    }
    for (std::size_t point_index = 0;
         point_index < request.layout.points.size();
         ++point_index) {
        const Point &point = request.layout.points[point_index];
        const int multiplicity = request.point_data[point_index].multiplicity;
        for (int derivative_order = 0;
             derivative_order < multiplicity;
             ++derivative_order) {
            if (visible_lift_level_host(
                    request.layout.degree,
                    derivative_order,
                    point.levels) < 0) {
                throw std::runtime_error(
                    "literal-witness derivative order has no odd visible lift");
            }
        }
    }
    return request;
}

struct DeviceBuffers {
    u64 *input = nullptr;
    u64 *output = nullptr;
    u64 *range = nullptr;
    u64 *gathered = nullptr;
    std::uint32_t *mapping = nullptr;
    std::uint32_t *pattern_columns = nullptr;
    std::uint32_t *pattern_starts = nullptr;
    unsigned char *positions = nullptr;

    ~DeviceBuffers() {
        cudaFree(positions);
        cudaFree(pattern_starts);
        cudaFree(pattern_columns);
        cudaFree(mapping);
        cudaFree(gathered);
        cudaFree(range);
        cudaFree(output);
        cudaFree(input);
    }
};

template<int WORD_CAP>
void launch_m_point_cap(
    const Request &request,
    const Point &point,
    const DeviceBuffers &buffers) {
    for (std::size_t index = 0; index < point.levels.size(); ++index) {
        const int level = point.levels[index];
        const u64 rows = HOST_CHOOSE.get(request.k, level);
        apply_m_level_kernel<WORD_CAP><<<launch_blocks(rows), THREADS>>>(
            buffers.input,
            buffers.range,
            rows,
            point.offsets[index],
            point.support,
            request.k,
            request.degree,
            level,
            request.words);
        CUDA_CHECK(cudaGetLastError());
    }
}

void launch_m_point(
    const Request &request,
    const Point &point,
    const DeviceBuffers &buffers) {
    if (request.words <= 8) {
        launch_m_point_cap<8>(request, point, buffers);
    } else {
        launch_m_point_cap<16>(request, point, buffers);
    }
}

template<int WORD_CAP>
void launch_mt_point_cap(
    const Request &request,
    const Point &point,
    const DeviceBuffers &buffers) {
    std::array<int, 4> levels{};
    std::array<u64, 4> offsets{};
    std::copy(point.levels.begin(), point.levels.end(), levels.begin());
    std::copy(point.offsets.begin(), point.offsets.end(), offsets.begin());
    apply_mt_point_kernel<WORD_CAP><<<
        launch_blocks(request.expected_columns), THREADS>>>(
        buffers.range,
        buffers.output,
        request.expected_columns,
        point.support,
        request.k,
        request.degree,
        request.words,
        static_cast<int>(point.levels.size()),
        levels[0], levels[1], levels[2], levels[3],
        offsets[0], offsets[1], offsets[2], offsets[3]);
    CUDA_CHECK(cudaGetLastError());
}

void launch_mt_point(
    const Request &request,
    const Point &point,
    const DeviceBuffers &buffers) {
    if (request.words <= 8) {
        launch_mt_point_cap<8>(request, point, buffers);
    } else {
        launch_mt_point_cap<16>(request, point, buffers);
    }
}

struct RunResult {
    std::vector<u64> output;
    std::vector<u64> transpose_output;
    std::string backend;
    double host_to_device_seconds = 0;
    double device_to_host_seconds = 0;
    double preprocessing_seconds = 0;
    double mx_seconds = 0;
    double mt_seconds = 0;
    std::size_t free_device_bytes_before = 0;
    std::size_t total_device_bytes = 0;
    std::size_t allocated_device_bytes = 0;
    u64 mapping_entries = 0;
};

RunResult gpu_apply_b_direct(const Request &request, const std::vector<u64> &input) {
    require_one_word_ground_set(request, "direct CUDA backend");
    RunResult result;
    result.backend = "direct-combinadic";
    std::size_t free_bytes = 0;
    std::size_t total_bytes = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
    result.free_device_bytes_before = free_bytes;
    result.total_device_bytes = total_bytes;

    const std::size_t panel_words = static_cast<std::size_t>(request.expected_columns)
        * request.words;
    u64 max_point_rows = 0;
    for (const Point &point : request.points) {
        max_point_rows = std::max(max_point_rows, point.rows);
    }
    const std::size_t range_words = static_cast<std::size_t>(max_point_rows)
        * request.words;
    const std::size_t panel_bytes = panel_words * sizeof(u64);
    const std::size_t range_bytes = range_words * sizeof(u64);
    result.allocated_device_bytes = 2 * panel_bytes + range_bytes;
    constexpr std::size_t reserve = std::size_t{256} << 20;
    if (result.allocated_device_bytes > free_bytes
        || free_bytes - result.allocated_device_bytes < reserve) {
        throw std::runtime_error("CUDA allocation would leave less than 256 MiB free");
    }

    DeviceBuffers buffers;
    CUDA_CHECK(cudaMalloc(&buffers.input, panel_bytes));
    CUDA_CHECK(cudaMalloc(&buffers.output, panel_bytes));
    CUDA_CHECK(cudaMalloc(&buffers.range, range_bytes));

    auto started = Clock::now();
    CUDA_CHECK(cudaMemcpy(buffers.input, input.data(), panel_bytes, cudaMemcpyHostToDevice));
    result.host_to_device_seconds = std::chrono::duration<double>(Clock::now() - started).count();

    cudaEvent_t before_m = nullptr;
    cudaEvent_t after_m = nullptr;
    cudaEvent_t after_mt = nullptr;
    CUDA_CHECK(cudaEventCreate(&before_m));
    CUDA_CHECK(cudaEventCreate(&after_m));
    CUDA_CHECK(cudaEventCreate(&after_mt));
    struct EventCleanup {
        cudaEvent_t &a;
        cudaEvent_t &b;
        cudaEvent_t &c;
        ~EventCleanup() { cudaEventDestroy(c); cudaEventDestroy(b); cudaEventDestroy(a); }
    } cleanup{before_m, after_m, after_mt};

    for (int repetition = 0; repetition < request.repetitions; ++repetition) {
        CUDA_CHECK(cudaMemset(buffers.output, 0, panel_bytes));
        for (const Point &point : request.points) {
            CUDA_CHECK(cudaEventRecord(before_m));
            launch_m_point(request, point, buffers);
            CUDA_CHECK(cudaEventRecord(after_m));
            launch_mt_point(request, point, buffers);
            CUDA_CHECK(cudaEventRecord(after_mt));
            CUDA_CHECK(cudaEventSynchronize(after_mt));
            float mx_ms = 0;
            float mt_ms = 0;
            CUDA_CHECK(cudaEventElapsedTime(&mx_ms, before_m, after_m));
            CUDA_CHECK(cudaEventElapsedTime(&mt_ms, after_m, after_mt));
            result.mx_seconds += mx_ms / 1000.0;
            result.mt_seconds += mt_ms / 1000.0;
        }
    }

    result.output.resize(panel_words);
    started = Clock::now();
    CUDA_CHECK(cudaMemcpy(result.output.data(), buffers.output, panel_bytes, cudaMemcpyDeviceToHost));
    result.device_to_host_seconds = std::chrono::duration<double>(Clock::now() - started).count();
    return result;
}

std::vector<unsigned char> point_position_tables(const Request &request) {
    std::vector<unsigned char> positions(
        request.points.size() * 2 * MAX_K, 0);
    for (std::size_t point_index = 0; point_index < request.points.size(); ++point_index) {
        const Point &point = request.points[point_index];
        unsigned char *support_output = positions.data()
            + point_index * 2 * MAX_K;
        unsigned char *complement_output = support_output + MAX_K;
        int support_index = 0;
        int complement_index = 0;
        for (int coordinate = 0; coordinate < request.k; ++coordinate) {
            if (point_contains(point, coordinate)) {
                support_output[support_index++] =
                    static_cast<unsigned char>(coordinate);
            } else {
                complement_output[complement_index++] =
                    static_cast<unsigned char>(coordinate);
            }
        }
    }
    return positions;
}

std::vector<u64> cpu_apply_left_slice_order(
    const Request &request,
    const std::vector<u64> &input,
    std::vector<u64> *transpose_output = nullptr) {
    const SlicePlan plan = build_slice_plan(request, false);
    const std::size_t range_words = static_cast<std::size_t>(
        plan.total_range_rows) * request.words;
    if (input.size() != range_words) {
        throw std::runtime_error("CPU left input has the wrong packed length");
    }
    const std::size_t domain_words = static_cast<std::size_t>(
        request.expected_columns) * request.words;
    std::vector<u64> domain(domain_words, 0);
    std::vector<u64> output(range_words, 0);
    const std::vector<unsigned char> positions =
        point_position_tables(request);

    auto visit_incidence = [&](auto callback) {
        for (std::size_t point_index = 0;
             point_index < request.points.size();
             ++point_index) {
            const Point &point = request.points[point_index];
            const int support_weight = point_weight(point);
            const unsigned char *support_positions = positions.data()
                + point_index * 2 * MAX_K;
            const unsigned char *complement_positions =
                support_positions + MAX_K;
            const u64 support_mask = support_weight == 64
                ? ~u64{0}
                : ((u64{1} << support_weight) - 1);
            for (const SliceClass &slice_class :
                 plan.points[point_index].classes) {
                for (u64 slice = 0;
                     slice < slice_class.slice_count;
                     ++slice) {
                    const u64 local_complement = colex_unrank_host(
                        slice,
                        slice_class.complement_size,
                        request.k - support_weight);
                    for (int size_index = 0;
                         size_index < slice_class.row_size_count;
                         ++size_index) {
                        const int row_size =
                            slice_class.row_sizes[size_index];
                        const u64 local_row_count = HOST_CHOOSE.get(
                            support_weight, row_size);
                        for (u64 local_row = 0;
                             local_row < local_row_count;
                             ++local_row) {
                            const u64 local_derivative = colex_unrank_host(
                                local_row, row_size, support_weight);
                            const u64 range_row = slice_class.range_offset
                                + slice * slice_class.rows_per_slice
                                + slice_class.row_offsets[size_index]
                                + local_row;
                            each_subset_host(
                                support_mask & ~local_derivative,
                                slice_class.support_column_size - row_size,
                                [&](u64 extension) {
                                    const u64 column =
                                        colex_rank_merged_local_masks_host(
                                            local_complement,
                                            complement_positions,
                                            local_derivative | extension,
                                            support_positions);
                                    callback(range_row, column);
                                });
                        }
                    }
                }
            }
        }
    };

    visit_incidence([&](u64 row, u64 column) {
        for (int word = 0; word < request.words; ++word) {
            domain[column * request.words + word] ^=
                input[row * request.words + word];
        }
    });
    visit_incidence([&](u64 row, u64 column) {
        for (int word = 0; word < request.words; ++word) {
            output[row * request.words + word] ^=
                domain[column * request.words + word];
        }
    });
    if (transpose_output != nullptr) {
        *transpose_output = domain;
    }
    return output;
}

void launch_fill_slice_mapping(
    const Request &request,
    const SlicePlan &plan,
    const DeviceBuffers &buffers) {
    for (std::size_t point_index = 0; point_index < request.points.size(); ++point_index) {
        const int support_weight = point_weight(request.points[point_index]);
        const int complement_weight = request.k - support_weight;
        const unsigned char *support_positions = buffers.positions
            + point_index * 2 * MAX_K;
        const unsigned char *complement_positions = support_positions + MAX_K;
        for (const SliceClass &slice_class : plan.points[point_index].classes) {
            const u64 mapping_count = slice_class.slice_count
                * slice_class.columns_per_slice;
            fill_slice_column_map_kernel<<<launch_blocks(mapping_count), THREADS>>>(
                buffers.mapping + slice_class.mapping_offset,
                mapping_count,
                slice_class.slice_count,
                slice_class.columns_per_slice,
                slice_class.complement_size,
                slice_class.support_column_size,
                complement_weight,
                support_weight,
                complement_positions,
                support_positions);
            CUDA_CHECK(cudaGetLastError());
        }
    }
}

void launch_m_slice_class(
    const Request &request,
    const Point &point,
    const SlicePlan &plan,
    const SliceClass &slice_class,
    const DeviceBuffers &buffers) {
    (void)point;
    const u64 total_rows = slice_class.slice_count * slice_class.rows_per_slice;
    const LocalPatternPlan &pattern = plan.patterns[slice_class.pattern_index];
    apply_m_slice_pattern_kernel<<<
        launch_blocks(total_rows * request.words), THREADS>>>(
        buffers.gathered,
        buffers.range,
        buffers.pattern_columns + pattern.columns_offset,
        buffers.pattern_starts + pattern.starts_offset,
        total_rows,
        slice_class.rows_per_slice,
        slice_class.columns_per_slice,
        request.words);
    CUDA_CHECK(cudaGetLastError());
}

void launch_gather_slice_class(
    const Request &request,
    const SliceClass &slice_class,
    const DeviceBuffers &buffers) {
    const u64 mapping_count = slice_class.slice_count
        * slice_class.columns_per_slice;
    gather_slice_input_kernel<<<
        launch_blocks(mapping_count * request.words), THREADS>>>(
        buffers.input,
        buffers.gathered,
        buffers.mapping + slice_class.mapping_offset,
        mapping_count,
        request.words);
    CUDA_CHECK(cudaGetLastError());
}

template<int WORD_CAP>
void launch_mt_slice_class_cap(
    const Request &request,
    const Point &point,
    const SliceClass &slice_class,
    const DeviceBuffers &buffers) {
    const int support_weight = point_weight(point);
    const u64 mapping_count = slice_class.slice_count
        * slice_class.columns_per_slice;
    apply_mt_slice_kernel<WORD_CAP><<<launch_blocks(mapping_count), THREADS>>>(
        buffers.range,
        buffers.output,
        buffers.mapping + slice_class.mapping_offset,
        mapping_count,
        slice_class.columns_per_slice,
        slice_class.rows_per_slice,
        support_weight,
        slice_class.support_column_size,
        request.words,
        slice_class.row_size_count,
        slice_class.row_sizes[0],
        slice_class.row_sizes[1],
        slice_class.row_sizes[2],
        slice_class.row_sizes[3],
        slice_class.row_offsets[0],
        slice_class.row_offsets[1],
        slice_class.row_offsets[2],
        slice_class.row_offsets[3]);
    CUDA_CHECK(cudaGetLastError());
}

void launch_mt_slice_class(
    const Request &request,
    const Point &point,
    const SliceClass &slice_class,
    const DeviceBuffers &buffers) {
    if (request.words <= 8) {
        launch_mt_slice_class_cap<8>(request, point, slice_class, buffers);
    } else {
        launch_mt_slice_class_cap<16>(request, point, slice_class, buffers);
    }
}

template<int WORD_CAP>
void launch_m_slice_recompute_class_cap(
    const Request &request,
    const Point &point,
    const SliceClass &slice_class,
    const u64 *input,
    u64 *range,
    const unsigned char *positions,
    std::size_t point_index) {
    const int support_weight = point_weight(point);
    const int complement_weight = request.k - support_weight;
    const unsigned char *support_positions = positions
        + point_index * 2 * MAX_K;
    const unsigned char *complement_positions =
        support_positions + MAX_K;
    const u64 total_rows = slice_class.slice_count
        * slice_class.rows_per_slice;
    apply_m_slice_recompute_kernel<WORD_CAP><<<
        launch_blocks(total_rows), THREADS>>>(
        input,
        range,
        total_rows,
        slice_class.rows_per_slice,
        slice_class.complement_size,
        slice_class.support_column_size,
        complement_weight,
        support_weight,
        complement_positions,
        support_positions,
        request.words,
        slice_class.row_size_count,
        slice_class.row_sizes[0],
        slice_class.row_sizes[1],
        slice_class.row_sizes[2],
        slice_class.row_sizes[3],
        slice_class.row_offsets[0],
        slice_class.row_offsets[1],
        slice_class.row_offsets[2],
        slice_class.row_offsets[3]);
    CUDA_CHECK(cudaGetLastError());
}

void launch_m_slice_recompute_class(
    const Request &request,
    const Point &point,
    const SliceClass &slice_class,
    const DeviceBuffers &buffers,
    std::size_t point_index) {
    if (request.words <= 8) {
        launch_m_slice_recompute_class_cap<8>(
            request,
            point,
            slice_class,
            buffers.input,
            buffers.range,
            buffers.positions,
            point_index);
    } else {
        launch_m_slice_recompute_class_cap<16>(
            request,
            point,
            slice_class,
            buffers.input,
            buffers.range,
            buffers.positions,
            point_index);
    }
}

void launch_m_slice_recompute_class_raw(
    const Request &request,
    const Point &point,
    const SliceClass &slice_class,
    const u64 *input,
    u64 *range,
    const unsigned char *positions,
    std::size_t point_index) {
    if (request.words <= 8) {
        launch_m_slice_recompute_class_cap<8>(
            request,
            point,
            slice_class,
            input,
            range,
            positions,
            point_index);
    } else {
        launch_m_slice_recompute_class_cap<16>(
            request,
            point,
            slice_class,
            input,
            range,
            positions,
            point_index);
    }
}

template<int WORD_CAP>
void launch_mt_slice_recompute_class_cap(
    const Request &request,
    const Point &point,
    const SliceClass &slice_class,
    const u64 *range,
    u64 *output,
    const unsigned char *positions,
    std::size_t point_index) {
    const int support_weight = point_weight(point);
    const int complement_weight = request.k - support_weight;
    const unsigned char *support_positions = positions
        + point_index * 2 * MAX_K;
    const unsigned char *complement_positions =
        support_positions + MAX_K;
    const u64 mapping_count = slice_class.slice_count
        * slice_class.columns_per_slice;
    apply_mt_slice_recompute_kernel<WORD_CAP><<<
        launch_blocks(mapping_count), THREADS>>>(
        range,
        output,
        mapping_count,
        slice_class.columns_per_slice,
        slice_class.rows_per_slice,
        slice_class.complement_size,
        slice_class.support_column_size,
        complement_weight,
        support_weight,
        complement_positions,
        support_positions,
        request.words,
        slice_class.row_size_count,
        slice_class.row_sizes[0],
        slice_class.row_sizes[1],
        slice_class.row_sizes[2],
        slice_class.row_sizes[3],
        slice_class.row_offsets[0],
        slice_class.row_offsets[1],
        slice_class.row_offsets[2],
        slice_class.row_offsets[3]);
    CUDA_CHECK(cudaGetLastError());
}

void launch_mt_slice_recompute_class(
    const Request &request,
    const Point &point,
    const SliceClass &slice_class,
    const DeviceBuffers &buffers,
    std::size_t point_index) {
    if (request.words <= 8) {
        launch_mt_slice_recompute_class_cap<8>(
            request,
            point,
            slice_class,
            buffers.range,
            buffers.output,
            buffers.positions,
            point_index);
    } else {
        launch_mt_slice_recompute_class_cap<16>(
            request,
            point,
            slice_class,
            buffers.range,
            buffers.output,
            buffers.positions,
            point_index);
    }
}

void launch_mt_slice_recompute_class_raw(
    const Request &request,
    const Point &point,
    const SliceClass &slice_class,
    const u64 *range,
    u64 *output,
    const unsigned char *positions,
    std::size_t point_index) {
    if (request.words <= 8) {
        launch_mt_slice_recompute_class_cap<8>(
            request,
            point,
            slice_class,
            range,
            output,
            positions,
            point_index);
    } else {
        launch_mt_slice_recompute_class_cap<16>(
            request,
            point,
            slice_class,
            range,
            output,
            positions,
            point_index);
    }
}

void launch_b_slice_device(
    const Request &request,
    const SlicePlan &plan,
    const DeviceBuffers &buffers,
    std::size_t panel_bytes) {
    CUDA_CHECK(cudaMemset(buffers.output, 0, panel_bytes));
    for (std::size_t point_index = 0;
         point_index < request.points.size();
         ++point_index) {
        const Point &point = request.points[point_index];
        for (const SliceClass &slice_class : plan.points[point_index].classes) {
            launch_gather_slice_class(request, slice_class, buffers);
            launch_m_slice_class(request, point, plan, slice_class, buffers);
            launch_mt_slice_class(request, point, slice_class, buffers);
        }
    }
}

void launch_b_slice_recompute_device(
    const Request &request,
    const SlicePlan &plan,
    const DeviceBuffers &buffers,
    std::size_t panel_bytes) {
    CUDA_CHECK(cudaMemset(buffers.output, 0, panel_bytes));
    for (std::size_t point_index = 0;
         point_index < request.points.size();
         ++point_index) {
        const Point &point = request.points[point_index];
        for (const SliceClass &slice_class :
             plan.points[point_index].classes) {
            launch_m_slice_recompute_class(
                request,
                point,
                slice_class,
                buffers,
                point_index);
            launch_mt_slice_recompute_class(
                request,
                point,
                slice_class,
                buffers,
                point_index);
        }
    }
}

void launch_injective_slice_recompute_device(
    const Request &request,
    const SlicePlan &plan,
    const DeviceBuffers &buffers,
    std::size_t panel_bytes,
    std::size_t domain_bytes) {
    // The coefficient state uses the slice-row permutation.  Apply
    // E=M^T once, then inject the form vector into the prefix of the same
    // coefficient space.  The remaining coefficient coordinates are zero.
    CUDA_CHECK(cudaMemset(buffers.range, 0, domain_bytes));
    for (std::size_t point_index = 0;
         point_index < request.points.size();
         ++point_index) {
        const Point &point = request.points[point_index];
        for (const SliceClass &slice_class :
             plan.points[point_index].classes) {
            launch_mt_slice_recompute_class_raw(
                request,
                point,
                slice_class,
                buffers.input + slice_class.range_offset * request.words,
                buffers.range,
                buffers.positions,
                point_index);
        }
    }
    CUDA_CHECK(cudaMemset(buffers.output, 0, panel_bytes));
    CUDA_CHECK(cudaMemcpy(
        buffers.output,
        buffers.range,
        domain_bytes,
        cudaMemcpyDeviceToDevice));
}

RunResult gpu_apply_b_slice(const Request &request, const std::vector<u64> &input) {
    RunResult result;
    result.backend = "support-complement-slices";
    auto host_plan_started = Clock::now();
    const SlicePlan plan = build_slice_plan(request);
    const double host_plan_seconds = std::chrono::duration<double>(
        Clock::now() - host_plan_started).count();
    result.mapping_entries = plan.mapping_entries;

    std::size_t free_bytes = 0;
    std::size_t total_bytes = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
    result.free_device_bytes_before = free_bytes;
    result.total_device_bytes = total_bytes;

    if (plan.mapping_entries > std::numeric_limits<std::size_t>::max()
            / sizeof(std::uint32_t)
        || plan.max_range_rows > std::numeric_limits<std::size_t>::max()
            / (static_cast<std::size_t>(request.words) * sizeof(u64))) {
        throw std::runtime_error("slice device allocation exceeds host size_t");
    }
    const std::size_t panel_words = static_cast<std::size_t>(
        request.expected_columns) * request.words;
    const std::size_t panel_bytes = panel_words * sizeof(u64);
    const std::size_t range_bytes = static_cast<std::size_t>(plan.max_range_rows)
        * request.words * sizeof(u64);
    const std::size_t mapping_bytes = static_cast<std::size_t>(plan.mapping_entries)
        * sizeof(std::uint32_t);
    const std::size_t gathered_bytes = static_cast<std::size_t>(
        plan.max_class_mapping_entries) * request.words * sizeof(u64);
    const std::size_t pattern_column_bytes = plan.pattern_columns.size()
        * sizeof(std::uint32_t);
    const std::size_t pattern_start_bytes = plan.pattern_starts.size()
        * sizeof(std::uint32_t);
    const std::size_t position_bytes = request.points.size() * 2 * MAX_K;
    result.allocated_device_bytes = 2 * panel_bytes + range_bytes
        + gathered_bytes + mapping_bytes
        + pattern_column_bytes + pattern_start_bytes
        + position_bytes;
    constexpr std::size_t reserve = std::size_t{256} << 20;
    if (result.allocated_device_bytes > free_bytes
        || free_bytes - result.allocated_device_bytes < reserve) {
        throw std::runtime_error(
            "slice CUDA allocation would leave less than 256 MiB free");
    }

    DeviceBuffers buffers;
    CUDA_CHECK(cudaMalloc(&buffers.input, panel_bytes));
    CUDA_CHECK(cudaMalloc(&buffers.output, panel_bytes));
    CUDA_CHECK(cudaMalloc(&buffers.range, range_bytes));
    CUDA_CHECK(cudaMalloc(&buffers.gathered, gathered_bytes));
    CUDA_CHECK(cudaMalloc(&buffers.mapping, mapping_bytes));
    CUDA_CHECK(cudaMalloc(&buffers.pattern_columns, pattern_column_bytes));
    CUDA_CHECK(cudaMalloc(&buffers.pattern_starts, pattern_start_bytes));
    CUDA_CHECK(cudaMalloc(&buffers.positions, position_bytes));

    const std::vector<unsigned char> positions = point_position_tables(request);
    auto started = Clock::now();
    CUDA_CHECK(cudaMemcpy(
        buffers.input, input.data(), panel_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(
        buffers.positions,
        positions.data(),
        position_bytes,
        cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(
        buffers.pattern_columns,
        plan.pattern_columns.data(),
        pattern_column_bytes,
        cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(
        buffers.pattern_starts,
        plan.pattern_starts.data(),
        pattern_start_bytes,
        cudaMemcpyHostToDevice));
    result.host_to_device_seconds = std::chrono::duration<double>(
        Clock::now() - started).count();

    cudaEvent_t before = nullptr;
    cudaEvent_t middle = nullptr;
    cudaEvent_t after = nullptr;
    CUDA_CHECK(cudaEventCreate(&before));
    CUDA_CHECK(cudaEventCreate(&middle));
    CUDA_CHECK(cudaEventCreate(&after));
    struct EventCleanup {
        cudaEvent_t &a;
        cudaEvent_t &b;
        cudaEvent_t &c;
        ~EventCleanup() { cudaEventDestroy(c); cudaEventDestroy(b); cudaEventDestroy(a); }
    } cleanup{before, middle, after};

    CUDA_CHECK(cudaEventRecord(before));
    launch_fill_slice_mapping(request, plan, buffers);
    CUDA_CHECK(cudaEventRecord(after));
    CUDA_CHECK(cudaEventSynchronize(after));
    float preprocessing_ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&preprocessing_ms, before, after));
    result.preprocessing_seconds = host_plan_seconds + preprocessing_ms / 1000.0;

    for (int repetition = 0; repetition < request.repetitions; ++repetition) {
        CUDA_CHECK(cudaMemset(buffers.output, 0, panel_bytes));
        for (std::size_t point_index = 0;
             point_index < request.points.size();
             ++point_index) {
            const Point &point = request.points[point_index];
            for (const SliceClass &slice_class : plan.points[point_index].classes) {
                CUDA_CHECK(cudaEventRecord(before));
                launch_gather_slice_class(request, slice_class, buffers);
                launch_m_slice_class(request, point, plan, slice_class, buffers);
                CUDA_CHECK(cudaEventRecord(middle));
                launch_mt_slice_class(request, point, slice_class, buffers);
                CUDA_CHECK(cudaEventRecord(after));
                CUDA_CHECK(cudaEventSynchronize(after));
                float mx_ms = 0;
                float mt_ms = 0;
                CUDA_CHECK(cudaEventElapsedTime(&mx_ms, before, middle));
                CUDA_CHECK(cudaEventElapsedTime(&mt_ms, middle, after));
                result.mx_seconds += mx_ms / 1000.0;
                result.mt_seconds += mt_ms / 1000.0;
            }
        }
    }

    result.output.resize(panel_words);
    started = Clock::now();
    CUDA_CHECK(cudaMemcpy(
        result.output.data(), buffers.output, panel_bytes, cudaMemcpyDeviceToHost));
    result.device_to_host_seconds = std::chrono::duration<double>(
        Clock::now() - started).count();
    return result;
}

RunResult gpu_apply_b_slice_recompute(
    const Request &request, const std::vector<u64> &input) {
    RunResult result;
    result.backend = "support-complement-slices-recomputed-map";
    const auto host_plan_started = Clock::now();
    const SlicePlan plan = build_slice_plan(request, false);
    result.preprocessing_seconds = std::chrono::duration<double>(
        Clock::now() - host_plan_started).count();
    result.mapping_entries = plan.mapping_entries;

    CUDA_CHECK(cudaMemGetInfo(
        &result.free_device_bytes_before,
        &result.total_device_bytes));
    if (plan.max_range_rows > std::numeric_limits<std::size_t>::max()
            / (static_cast<std::size_t>(request.words) * sizeof(u64))
        || request.expected_columns > std::numeric_limits<std::size_t>::max()
            / (static_cast<std::size_t>(request.words) * sizeof(u64))) {
        throw std::runtime_error(
            "recomputed slice device allocation exceeds host size_t");
    }
    const std::size_t panel_words = static_cast<std::size_t>(
        request.expected_columns) * request.words;
    const std::size_t panel_bytes = panel_words * sizeof(u64);
    const std::size_t range_bytes = static_cast<std::size_t>(
        plan.max_range_rows) * request.words * sizeof(u64);
    const std::size_t position_bytes = request.points.size() * 2 * MAX_K;
    result.allocated_device_bytes = 2 * panel_bytes + range_bytes
        + position_bytes;
    constexpr std::size_t reserve = std::size_t{256} << 20;
    if (result.allocated_device_bytes > result.free_device_bytes_before
        || result.free_device_bytes_before - result.allocated_device_bytes
            < reserve) {
        throw std::runtime_error(
            "recomputed slice CUDA allocation would leave less than 256 MiB free");
    }

    DeviceBuffers buffers;
    CUDA_CHECK(cudaMalloc(&buffers.input, panel_bytes));
    CUDA_CHECK(cudaMalloc(&buffers.output, panel_bytes));
    CUDA_CHECK(cudaMalloc(&buffers.range, range_bytes));
    CUDA_CHECK(cudaMalloc(&buffers.positions, position_bytes));
    const std::vector<unsigned char> positions =
        point_position_tables(request);
    auto started = Clock::now();
    CUDA_CHECK(cudaMemcpy(
        buffers.input,
        input.data(),
        panel_bytes,
        cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(
        buffers.positions,
        positions.data(),
        position_bytes,
        cudaMemcpyHostToDevice));
    result.host_to_device_seconds = std::chrono::duration<double>(
        Clock::now() - started).count();

    cudaEvent_t before = nullptr;
    cudaEvent_t middle = nullptr;
    cudaEvent_t after = nullptr;
    CUDA_CHECK(cudaEventCreate(&before));
    CUDA_CHECK(cudaEventCreate(&middle));
    CUDA_CHECK(cudaEventCreate(&after));
    struct EventCleanup {
        cudaEvent_t &a;
        cudaEvent_t &b;
        cudaEvent_t &c;
        ~EventCleanup() {
            cudaEventDestroy(c);
            cudaEventDestroy(b);
            cudaEventDestroy(a);
        }
    } cleanup{before, middle, after};

    for (int repetition = 0;
         repetition < request.repetitions;
         ++repetition) {
        CUDA_CHECK(cudaMemset(buffers.output, 0, panel_bytes));
        for (std::size_t point_index = 0;
             point_index < request.points.size();
             ++point_index) {
            const Point &point = request.points[point_index];
            for (const SliceClass &slice_class :
                 plan.points[point_index].classes) {
                CUDA_CHECK(cudaEventRecord(before));
                launch_m_slice_recompute_class(
                    request,
                    point,
                    slice_class,
                    buffers,
                    point_index);
                CUDA_CHECK(cudaEventRecord(middle));
                launch_mt_slice_recompute_class(
                    request,
                    point,
                    slice_class,
                    buffers,
                    point_index);
                CUDA_CHECK(cudaEventRecord(after));
                CUDA_CHECK(cudaEventSynchronize(after));
                float mx_ms = 0;
                float mt_ms = 0;
                CUDA_CHECK(cudaEventElapsedTime(
                    &mx_ms, before, middle));
                CUDA_CHECK(cudaEventElapsedTime(
                    &mt_ms, middle, after));
                result.mx_seconds += mx_ms / 1000.0;
                result.mt_seconds += mt_ms / 1000.0;
            }
        }
    }

    result.output.resize(panel_words);
    started = Clock::now();
    CUDA_CHECK(cudaMemcpy(
        result.output.data(),
        buffers.output,
        panel_bytes,
        cudaMemcpyDeviceToHost));
    result.device_to_host_seconds = std::chrono::duration<double>(
        Clock::now() - started).count();
    return result;
}

RunResult gpu_apply_left_slice_recompute(
    const Request &request, const std::vector<u64> &input) {
    RunResult result;
    result.backend = "left-support-complement-slices-recomputed-map";
    const auto host_plan_started = Clock::now();
    const SlicePlan plan = build_slice_plan(request, false);
    result.preprocessing_seconds = std::chrono::duration<double>(
        Clock::now() - host_plan_started).count();
    result.mapping_entries = plan.mapping_entries;

    CUDA_CHECK(cudaMemGetInfo(
        &result.free_device_bytes_before,
        &result.total_device_bytes));
    if (plan.total_range_rows > std::numeric_limits<std::size_t>::max()
            / (static_cast<std::size_t>(request.words) * sizeof(u64))
        || request.expected_columns > std::numeric_limits<std::size_t>::max()
            / (static_cast<std::size_t>(request.words) * sizeof(u64))) {
        throw std::runtime_error(
            "left recomputed-slice device allocation exceeds host size_t");
    }
    const std::size_t range_words = static_cast<std::size_t>(
        plan.total_range_rows) * request.words;
    const std::size_t domain_words = static_cast<std::size_t>(
        request.expected_columns) * request.words;
    if (input.size() != range_words) {
        throw std::runtime_error("left CUDA input has the wrong packed length");
    }
    const std::size_t range_bytes = range_words * sizeof(u64);
    const std::size_t domain_bytes = domain_words * sizeof(u64);
    const std::size_t position_bytes = request.points.size() * 2 * MAX_K;
    result.allocated_device_bytes = 2 * range_bytes + domain_bytes
        + position_bytes;
    constexpr std::size_t reserve = std::size_t{256} << 20;
    if (result.allocated_device_bytes > result.free_device_bytes_before
        || result.free_device_bytes_before - result.allocated_device_bytes
            < reserve) {
        throw std::runtime_error(
            "left recomputed-slice CUDA allocation would leave less than 256 MiB free");
    }

    u64 *range_input = nullptr;
    u64 *range_output = nullptr;
    u64 *domain = nullptr;
    unsigned char *device_positions = nullptr;
    struct LeftCleanup {
        u64 *&input;
        u64 *&output;
        u64 *&domain;
        unsigned char *&positions;
        ~LeftCleanup() {
            cudaFree(positions);
            cudaFree(domain);
            cudaFree(output);
            cudaFree(input);
        }
    } cleanup{range_input, range_output, domain, device_positions};
    CUDA_CHECK(cudaMalloc(&range_input, range_bytes));
    CUDA_CHECK(cudaMalloc(&range_output, range_bytes));
    CUDA_CHECK(cudaMalloc(&domain, domain_bytes));
    CUDA_CHECK(cudaMalloc(&device_positions, position_bytes));
    const std::vector<unsigned char> positions =
        point_position_tables(request);
    auto started = Clock::now();
    CUDA_CHECK(cudaMemcpy(
        range_input,
        input.data(),
        range_bytes,
        cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(
        device_positions,
        positions.data(),
        position_bytes,
        cudaMemcpyHostToDevice));
    result.host_to_device_seconds = std::chrono::duration<double>(
        Clock::now() - started).count();

    cudaEvent_t before = nullptr;
    cudaEvent_t middle = nullptr;
    cudaEvent_t after = nullptr;
    CUDA_CHECK(cudaEventCreate(&before));
    CUDA_CHECK(cudaEventCreate(&middle));
    CUDA_CHECK(cudaEventCreate(&after));
    struct EventCleanup {
        cudaEvent_t &before;
        cudaEvent_t &middle;
        cudaEvent_t &after;
        ~EventCleanup() {
            cudaEventDestroy(after);
            cudaEventDestroy(middle);
            cudaEventDestroy(before);
        }
    } event_cleanup{before, middle, after};

    for (int repetition = 0;
         repetition < request.repetitions;
         ++repetition) {
        CUDA_CHECK(cudaMemset(domain, 0, domain_bytes));
        CUDA_CHECK(cudaEventRecord(before));
        for (std::size_t point_index = 0;
             point_index < request.points.size();
             ++point_index) {
            const Point &point = request.points[point_index];
            for (const SliceClass &slice_class :
                 plan.points[point_index].classes) {
                launch_mt_slice_recompute_class_raw(
                    request,
                    point,
                    slice_class,
                    range_input
                        + slice_class.range_offset * request.words,
                    domain,
                    device_positions,
                    point_index);
            }
        }
        CUDA_CHECK(cudaEventRecord(middle));
        for (std::size_t point_index = 0;
             point_index < request.points.size();
             ++point_index) {
            const Point &point = request.points[point_index];
            for (const SliceClass &slice_class :
                 plan.points[point_index].classes) {
                launch_m_slice_recompute_class_raw(
                    request,
                    point,
                    slice_class,
                    domain,
                    range_output
                        + slice_class.range_offset * request.words,
                    device_positions,
                    point_index);
            }
        }
        CUDA_CHECK(cudaEventRecord(after));
        CUDA_CHECK(cudaEventSynchronize(after));
        float mt_ms = 0;
        float mx_ms = 0;
        CUDA_CHECK(cudaEventElapsedTime(&mt_ms, before, middle));
        CUDA_CHECK(cudaEventElapsedTime(&mx_ms, middle, after));
        result.mt_seconds += mt_ms / 1000.0;
        result.mx_seconds += mx_ms / 1000.0;
    }

    result.output.resize(range_words);
    result.transpose_output.resize(domain_words);
    started = Clock::now();
    CUDA_CHECK(cudaMemcpy(
        result.output.data(),
        range_output,
        range_bytes,
        cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(
        result.transpose_output.data(),
        domain,
        domain_bytes,
        cudaMemcpyDeviceToHost));
    result.device_to_host_seconds = std::chrono::duration<double>(
        Clock::now() - started).count();
    return result;
}

struct WitnessRunResult {
    std::vector<u64> output;
    std::vector<u64> emitted_coefficients;
    double host_to_device_seconds = 0;
    double device_to_host_seconds = 0;
    double coefficient_kernel_seconds = 0;
    double transpose_kernel_seconds = 0;
    double target_replay_kernel_seconds = 0;
    std::size_t free_device_bytes_before = 0;
    std::size_t total_device_bytes = 0;
    std::size_t allocated_device_bytes = 0;
    u64 logical_mapping_entries = 0;
    u64 target_replay_mismatches = 0;
};

std::uint16_t derivative_order_mask_for_level(
    const Request &request,
    const Point &point,
    int multiplicity,
    int level) {
    std::uint16_t mask = 0;
    for (int derivative_order = 0;
         derivative_order < multiplicity;
         ++derivative_order) {
        if (visible_lift_level_host(
                request.degree,
                derivative_order,
                point.levels) == level) {
            mask |= std::uint16_t{1} << derivative_order;
        }
    }
    return mask;
}

void launch_literal_transpose_class(
    const Request &request,
    const Point &point,
    const SliceClass &slice_class,
    std::size_t point_index,
    const u64 *device_coefficients,
    u64 *device_output,
    const unsigned char *device_positions) {
    std::array<u64, 4> global_offsets{};
    for (int row_index = 0;
         row_index < slice_class.row_size_count;
         ++row_index) {
        const int level = slice_class.complement_size
            + slice_class.row_sizes[row_index];
        const auto found = std::find(
            point.levels.begin(), point.levels.end(), level);
        if (found == point.levels.end()) {
            throw std::runtime_error(
                "slice literal row level is absent from the point layout");
        }
        global_offsets[row_index] = point.offsets[
            static_cast<std::size_t>(found - point.levels.begin())];
    }
    const int support_weight = point_weight(point);
    const int complement_weight = request.k - support_weight;
    const unsigned char *support_positions = device_positions
        + point_index * 2 * MAX_K;
    const unsigned char *complement_positions =
        support_positions + MAX_K;
    const u64 mapping_count = slice_class.slice_count
        * slice_class.columns_per_slice;
    apply_mt_literal_slice_kernel<<<
        launch_blocks(mapping_count), THREADS>>>(
        device_coefficients,
        device_output,
        mapping_count,
        slice_class.columns_per_slice,
        slice_class.complement_size,
        slice_class.support_column_size,
        complement_weight,
        support_weight,
        complement_positions,
        support_positions,
        slice_class.row_size_count,
        slice_class.row_sizes[0],
        slice_class.row_sizes[1],
        slice_class.row_sizes[2],
        slice_class.row_sizes[3],
        global_offsets[0],
        global_offsets[1],
        global_offsets[2],
        global_offsets[3]);
    CUDA_CHECK(cudaGetLastError());
}

WitnessRunResult gpu_literal_witness_transpose(
    const WitnessRequest &witness_request) {
    const Request &request = witness_request.layout;
    // Literal transpose ranks global columns on demand and never consumes the
    // cached local-incidence tables used by the Krylov operator.  Building
    // those tables here is both wasted work and invalid for target points
    // whose local incidence count exceeds the legacy u32 offset contract.
    const SlicePlan plan = build_slice_plan(request, false);
    const std::vector<unsigned char> positions =
        point_position_tables(request);
    WitnessRunResult result;
    result.logical_mapping_entries = plan.mapping_entries;
    CUDA_CHECK(cudaMemGetInfo(
        &result.free_device_bytes_before,
        &result.total_device_bytes));

    u64 max_point_rows = 0;
    std::size_t max_jet_bytes = 0;
    std::size_t max_weight_bytes = 0;
    for (std::size_t point_index = 0;
         point_index < request.points.size();
         ++point_index) {
        max_point_rows = std::max(
            max_point_rows, request.points[point_index].rows);
        max_jet_bytes = std::max(
            max_jet_bytes,
            witness_request.point_data[point_index].jets.size());
        max_weight_bytes = std::max(
            max_weight_bytes,
            witness_request.point_data[point_index].weights.size());
    }
    if (witness_request.verify_target) {
        max_jet_bytes = std::max(
            max_jet_bytes, witness_request.target_jets.size());
    }
    if (request.expected_columns
            > std::numeric_limits<std::size_t>::max() / sizeof(u64)
        || max_point_rows
            > std::numeric_limits<std::size_t>::max() / sizeof(u64)) {
        throw std::runtime_error(
            "literal-witness device allocation exceeds host size_t");
    }
    const std::size_t output_bytes =
        static_cast<std::size_t>(request.expected_columns) * sizeof(u64);
    const std::size_t coefficient_bytes =
        static_cast<std::size_t>(max_point_rows) * sizeof(u64);
    const std::size_t position_bytes = positions.size();
    result.allocated_device_bytes = output_bytes + coefficient_bytes
        + position_bytes + max_jet_bytes + max_weight_bytes
        + (witness_request.verify_target ? sizeof(u64) : 0);
    constexpr std::size_t reserve = std::size_t{256} << 20;
    if (result.allocated_device_bytes > result.free_device_bytes_before
        || result.free_device_bytes_before - result.allocated_device_bytes
            < reserve) {
        throw std::runtime_error(
            "literal-witness CUDA allocation would leave less than 256 MiB free");
    }

    u64 *device_output = nullptr;
    u64 *device_coefficients = nullptr;
    unsigned char *device_positions = nullptr;
    unsigned char *device_jets = nullptr;
    unsigned char *device_weights = nullptr;
    u64 *device_target_mismatches = nullptr;
    struct WitnessCleanup {
        u64 *&output;
        u64 *&coefficients;
        unsigned char *&positions;
        unsigned char *&jets;
        unsigned char *&weights;
        u64 *&target_mismatches;
        ~WitnessCleanup() {
            cudaFree(target_mismatches);
            cudaFree(weights);
            cudaFree(jets);
            cudaFree(positions);
            cudaFree(coefficients);
            cudaFree(output);
        }
    } cleanup{
        device_output,
        device_coefficients,
        device_positions,
        device_jets,
        device_weights,
        device_target_mismatches,
    };
    CUDA_CHECK(cudaMalloc(&device_output, output_bytes));
    CUDA_CHECK(cudaMalloc(&device_coefficients, coefficient_bytes));
    CUDA_CHECK(cudaMalloc(&device_positions, position_bytes));
    CUDA_CHECK(cudaMalloc(&device_jets, max_jet_bytes));
    CUDA_CHECK(cudaMalloc(&device_weights, max_weight_bytes));
    if (witness_request.verify_target) {
        CUDA_CHECK(cudaMalloc(&device_target_mismatches, sizeof(u64)));
        CUDA_CHECK(cudaMemset(device_target_mismatches, 0, sizeof(u64)));
    }
    CUDA_CHECK(cudaMemset(device_output, 0, output_bytes));
    auto host_copy_started = Clock::now();
    CUDA_CHECK(cudaMemcpy(
        device_positions,
        positions.data(),
        position_bytes,
        cudaMemcpyHostToDevice));
    result.host_to_device_seconds += std::chrono::duration<double>(
        Clock::now() - host_copy_started).count();

    cudaEvent_t before = nullptr;
    cudaEvent_t after = nullptr;
    CUDA_CHECK(cudaEventCreate(&before));
    CUDA_CHECK(cudaEventCreate(&after));
    struct EventCleanup {
        cudaEvent_t &before;
        cudaEvent_t &after;
        ~EventCleanup() {
            cudaEventDestroy(after);
            cudaEventDestroy(before);
        }
    } event_cleanup{before, after};

    if (witness_request.emit_coefficients) {
        result.emitted_coefficients.reserve(
            static_cast<std::size_t>(request.expected_rows));
    }
    for (std::size_t point_index = 0;
         point_index < request.points.size();
         ++point_index) {
        const Point &point = request.points[point_index];
        const WitnessPointData &data =
            witness_request.point_data[point_index];
        host_copy_started = Clock::now();
        CUDA_CHECK(cudaMemcpy(
            device_jets,
            data.jets.data(),
            data.jets.size(),
            cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(
            device_weights,
            data.weights.data(),
            data.weights.size(),
            cudaMemcpyHostToDevice));
        result.host_to_device_seconds += std::chrono::duration<double>(
            Clock::now() - host_copy_started).count();

        CUDA_CHECK(cudaEventRecord(before));
        for (std::size_t level_index = 0;
             level_index < point.levels.size();
             ++level_index) {
            const int level = point.levels[level_index];
            const u64 row_count = HOST_CHOOSE.get(request.k, level);
            const std::uint16_t derivative_order_mask =
                derivative_order_mask_for_level(
                    request, point, data.multiplicity, level);
            generate_literal_witness_coefficients_kernel<<<
                launch_blocks(row_count), THREADS>>>(
                device_coefficients,
                row_count,
                point.offsets[level_index],
                request.k,
                level,
                data.multiplicity,
                derivative_order_mask,
                point.support,
                point.support_high,
                device_jets,
                device_weights);
            CUDA_CHECK(cudaGetLastError());
        }
        CUDA_CHECK(cudaEventRecord(after));
        CUDA_CHECK(cudaEventSynchronize(after));
        float coefficient_ms = 0;
        CUDA_CHECK(cudaEventElapsedTime(
            &coefficient_ms, before, after));
        result.coefficient_kernel_seconds += coefficient_ms / 1000.0;

        if (witness_request.emit_coefficients) {
            const std::size_t old_size =
                result.emitted_coefficients.size();
            result.emitted_coefficients.resize(
                old_size + static_cast<std::size_t>(point.rows));
            const auto copy_started = Clock::now();
            CUDA_CHECK(cudaMemcpy(
                result.emitted_coefficients.data() + old_size,
                device_coefficients,
                static_cast<std::size_t>(point.rows) * sizeof(u64),
                cudaMemcpyDeviceToHost));
            result.device_to_host_seconds += std::chrono::duration<double>(
                Clock::now() - copy_started).count();
        }

        CUDA_CHECK(cudaEventRecord(before));
        for (const SliceClass &slice_class : plan.points[point_index].classes) {
            launch_literal_transpose_class(
                request,
                point,
                slice_class,
                point_index,
                device_coefficients,
                device_output,
                device_positions);
        }
        CUDA_CHECK(cudaEventRecord(after));
        CUDA_CHECK(cudaEventSynchronize(after));
        float transpose_ms = 0;
        CUDA_CHECK(cudaEventElapsedTime(
            &transpose_ms, before, after));
        result.transpose_kernel_seconds += transpose_ms / 1000.0;
    }

    if (witness_request.verify_target) {
        host_copy_started = Clock::now();
        CUDA_CHECK(cudaMemcpy(
            device_jets,
            witness_request.target_jets.data(),
            witness_request.target_jets.size(),
            cudaMemcpyHostToDevice));
        result.host_to_device_seconds += std::chrono::duration<double>(
            Clock::now() - host_copy_started).count();
        CUDA_CHECK(cudaEventRecord(before));
        verify_literal_witness_target_kernel<<<
            launch_blocks(request.expected_columns), THREADS>>>(
            device_output,
            request.expected_columns,
            request.k,
            request.degree,
            witness_request.target_support,
            witness_request.target_support_high,
            device_jets,
            device_target_mismatches);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventRecord(after));
        CUDA_CHECK(cudaEventSynchronize(after));
        float replay_ms = 0;
        CUDA_CHECK(cudaEventElapsedTime(&replay_ms, before, after));
        result.target_replay_kernel_seconds = replay_ms / 1000.0;
        CUDA_CHECK(cudaMemcpy(
            &result.target_replay_mismatches,
            device_target_mismatches,
            sizeof(u64),
            cudaMemcpyDeviceToHost));
        if (result.target_replay_mismatches != 0) {
            throw std::runtime_error(
                "literal-witness transpose differs from the direct target functional");
        }
    }

    result.output.resize(static_cast<std::size_t>(request.expected_columns));
    const auto output_copy_started = Clock::now();
    CUDA_CHECK(cudaMemcpy(
        result.output.data(),
        device_output,
        output_bytes,
        cudaMemcpyDeviceToHost));
    result.device_to_host_seconds += std::chrono::duration<double>(
        Clock::now() - output_copy_started).count();
    return result;
}

struct ProjectionRunResult {
    std::vector<u64> output;
    double host_to_device_seconds = 0;
    double device_to_host_seconds = 0;
    double kernel_seconds = 0;
    std::size_t free_device_bytes_before = 0;
    std::size_t total_device_bytes = 0;
    std::size_t allocated_device_bytes = 0;
    int chunks = 0;
};

struct ColumnDegreeCensusResult {
    std::vector<u64> histogram;
    double host_to_device_seconds = 0;
    double kernel_seconds = 0;
    double device_to_host_seconds = 0;
    std::size_t free_device_bytes_before = 0;
    std::size_t total_device_bytes = 0;
    std::size_t allocated_device_bytes = 0;
};

std::size_t column_degree_bound(const Request &request) {
    std::size_t bound = 0;
    for (const Point &point : request.points) {
        std::size_t point_maximum = 0;
        for (int intersection = 0; intersection <= request.degree; ++intersection) {
            const int outside = request.degree - intersection;
            std::size_t candidate = 0;
            for (const int level : point.levels) {
                candidate += HOST_CHOOSE.get(intersection, level - outside);
            }
            point_maximum = std::max(point_maximum, candidate);
        }
        bound += point_maximum;
    }
    return bound;
}

ColumnDegreeCensusResult gpu_column_degree_census(const Request &request) {
    require_one_word_ground_set(request, "column-degree census");
    ColumnDegreeCensusResult result;
    std::size_t free_bytes = 0;
    std::size_t total_bytes = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
    result.free_device_bytes_before = free_bytes;
    result.total_device_bytes = total_bytes;

    const std::size_t point_count = request.points.size();
    const std::size_t bins = column_degree_bound(request) + 1;
    if (bins > std::numeric_limits<std::size_t>::max() / sizeof(u64)) {
        throw std::runtime_error("column-degree histogram exceeds host size_t");
    }
    const std::size_t support_bytes = point_count * sizeof(u64);
    const std::size_t level_bytes = point_count * sizeof(std::uint16_t);
    const std::size_t histogram_bytes = bins * sizeof(u64);
    result.allocated_device_bytes = support_bytes + level_bytes + histogram_bytes;

    std::vector<u64> supports;
    std::vector<std::uint16_t> level_masks;
    supports.reserve(point_count);
    level_masks.reserve(point_count);
    for (const Point &point : request.points) {
        std::uint16_t mask = 0;
        for (const int level : point.levels) {
            if (level < 0 || level >= 16) {
                throw std::runtime_error(
                    "column-degree census level does not fit its wire mask");
            }
            mask |= std::uint16_t{1} << level;
        }
        supports.push_back(point.support);
        level_masks.push_back(mask);
    }

    u64 *device_supports = nullptr;
    std::uint16_t *device_level_masks = nullptr;
    u64 *device_histogram = nullptr;
    struct CensusCleanup {
        u64 *&supports;
        std::uint16_t *&levels;
        u64 *&histogram;
        ~CensusCleanup() {
            cudaFree(histogram);
            cudaFree(levels);
            cudaFree(supports);
        }
    } cleanup{device_supports, device_level_masks, device_histogram};
    CUDA_CHECK(cudaMalloc(&device_supports, support_bytes));
    CUDA_CHECK(cudaMalloc(&device_level_masks, level_bytes));
    CUDA_CHECK(cudaMalloc(&device_histogram, histogram_bytes));
    CUDA_CHECK(cudaMemset(device_histogram, 0, histogram_bytes));

    auto started = Clock::now();
    CUDA_CHECK(cudaMemcpy(
        device_supports,
        supports.data(),
        support_bytes,
        cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(
        device_level_masks,
        level_masks.data(),
        level_bytes,
        cudaMemcpyHostToDevice));
    result.host_to_device_seconds = std::chrono::duration<double>(
        Clock::now() - started).count();

    cudaEvent_t before = nullptr;
    cudaEvent_t after = nullptr;
    CUDA_CHECK(cudaEventCreate(&before));
    CUDA_CHECK(cudaEventCreate(&after));
    struct EventCleanup {
        cudaEvent_t &before;
        cudaEvent_t &after;
        ~EventCleanup() { cudaEventDestroy(after); cudaEventDestroy(before); }
    } event_cleanup{before, after};
    CUDA_CHECK(cudaEventRecord(before));
    column_degree_histogram_kernel<<<
        launch_blocks(request.expected_columns), THREADS>>>(
        request.expected_columns,
        request.k,
        request.degree,
        device_supports,
        device_level_masks,
        static_cast<int>(point_count),
        device_histogram);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(after));
    CUDA_CHECK(cudaEventSynchronize(after));
    float kernel_ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&kernel_ms, before, after));
    result.kernel_seconds = kernel_ms / 1000.0;

    result.histogram.resize(bins);
    started = Clock::now();
    CUDA_CHECK(cudaMemcpy(
        result.histogram.data(),
        device_histogram,
        histogram_bytes,
        cudaMemcpyDeviceToHost));
    result.device_to_host_seconds = std::chrono::duration<double>(
        Clock::now() - started).count();

    u64 columns = 0;
    u64 incidences = 0;
    for (std::size_t degree = 0; degree < result.histogram.size(); ++degree) {
        columns += result.histogram[degree];
        incidences += result.histogram[degree] * degree;
    }
    if (columns != request.expected_columns
        || incidences != request.expected_nonzeros) {
        std::ostringstream message;
        message << "column-degree census integrity differs: columns "
                << columns << "/" << request.expected_columns
                << ", incidences " << incidences << "/"
                << request.expected_nonzeros;
        throw std::runtime_error(message.str());
    }
    return result;
}

int projection_chunk_count(u64 columns) {
    constexpr u64 TARGET_COLUMNS_PER_CHUNK = 150000;
    u64 required = (columns + TARGET_COLUMNS_PER_CHUNK - 1)
        / TARGET_COLUMNS_PER_CHUNK;
    int chunks = 1;
    while (static_cast<u64>(chunks) < required && chunks < 1024) {
        chunks *= 2;
    }
    return chunks;
}

void launch_projection_device(
    const u64 *device_z,
    const u64 *device_v,
    u64 *device_partials,
    u64 *device_output,
    u64 columns,
    int words,
    int chunks) {
    const u64 warp_count = static_cast<u64>(words) * words * chunks;
    constexpr int projection_threads = 256;
    const int partial_blocks = static_cast<int>(
        (warp_count + projection_threads / 32 - 1)
        / (projection_threads / 32));
    project_partial_kernel<<<partial_blocks, projection_threads>>>(
        device_z,
        device_v,
        device_partials,
        columns,
        words,
        chunks);
    CUDA_CHECK(cudaGetLastError());
    const u64 output_words = static_cast<u64>(64) * words * words;
    project_reduce_kernel<<<launch_blocks(output_words), THREADS>>>(
        device_partials, device_output, words, chunks);
    CUDA_CHECK(cudaGetLastError());
}

ProjectionRunResult gpu_project(
    const Request &request,
    const std::vector<u64> &z,
    const std::vector<u64> &v) {
    ProjectionRunResult result;
    std::size_t free_bytes = 0;
    std::size_t total_bytes = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
    result.free_device_bytes_before = free_bytes;
    result.total_device_bytes = total_bytes;
    result.chunks = projection_chunk_count(request.expected_columns);

    const std::size_t panel_words = static_cast<std::size_t>(
        request.expected_columns) * request.words;
    const std::size_t panel_bytes = panel_words * sizeof(u64);
    const std::size_t output_words = static_cast<std::size_t>(64)
        * request.words * request.words;
    const std::size_t output_bytes = output_words * sizeof(u64);
    const std::size_t partial_words = output_words * result.chunks;
    const std::size_t partial_bytes = partial_words * sizeof(u64);
    result.allocated_device_bytes = 2 * panel_bytes + output_bytes + partial_bytes;
    constexpr std::size_t reserve = std::size_t{256} << 20;
    if (result.allocated_device_bytes > free_bytes
        || free_bytes - result.allocated_device_bytes < reserve) {
        throw std::runtime_error(
            "projection CUDA allocation would leave less than 256 MiB free");
    }

    u64 *device_z = nullptr;
    u64 *device_v = nullptr;
    u64 *device_partials = nullptr;
    u64 *device_output = nullptr;
    struct ProjectionCleanup {
        u64 *&z;
        u64 *&v;
        u64 *&partials;
        u64 *&output;
        ~ProjectionCleanup() {
            cudaFree(output);
            cudaFree(partials);
            cudaFree(v);
            cudaFree(z);
        }
    } cleanup{device_z, device_v, device_partials, device_output};
    CUDA_CHECK(cudaMalloc(&device_z, panel_bytes));
    CUDA_CHECK(cudaMalloc(&device_v, panel_bytes));
    CUDA_CHECK(cudaMalloc(&device_partials, partial_bytes));
    CUDA_CHECK(cudaMalloc(&device_output, output_bytes));

    auto started = Clock::now();
    CUDA_CHECK(cudaMemcpy(
        device_z, z.data(), panel_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(
        device_v, v.data(), panel_bytes, cudaMemcpyHostToDevice));
    result.host_to_device_seconds = std::chrono::duration<double>(
        Clock::now() - started).count();

    cudaEvent_t before = nullptr;
    cudaEvent_t after = nullptr;
    CUDA_CHECK(cudaEventCreate(&before));
    CUDA_CHECK(cudaEventCreate(&after));
    struct EventCleanup {
        cudaEvent_t &before;
        cudaEvent_t &after;
        ~EventCleanup() { cudaEventDestroy(after); cudaEventDestroy(before); }
    } event_cleanup{before, after};
    CUDA_CHECK(cudaEventRecord(before));
    launch_projection_device(
        device_z,
        device_v,
        device_partials,
        device_output,
        request.expected_columns,
        request.words,
        result.chunks);
    CUDA_CHECK(cudaEventRecord(after));
    CUDA_CHECK(cudaEventSynchronize(after));
    float kernel_ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&kernel_ms, before, after));
    result.kernel_seconds = kernel_ms / 1000.0;

    result.output.resize(output_words);
    started = Clock::now();
    CUDA_CHECK(cudaMemcpy(
        result.output.data(),
        device_output,
        output_bytes,
        cudaMemcpyDeviceToHost));
    result.device_to_host_seconds = std::chrono::duration<double>(
        Clock::now() - started).count();
    return result;
}

u64 checksum_words(const std::vector<u64> &values) {
    u64 hash = 1469598103934665603ULL;
    for (const u64 value : values) {
        for (int byte = 0; byte < 8; ++byte) {
            hash ^= (value >> (8 * byte)) & 0xff;
            hash *= 1099511628211ULL;
        }
    }
    return hash;
}

std::string hex_u64(u64 value) {
    std::ostringstream stream;
    stream << std::hex << std::setw(16) << std::setfill('0') << value;
    return stream.str();
}

std::string json_escape(const std::string &value) {
    std::ostringstream output;
    for (const unsigned char character : value) {
        switch (character) {
            case '\\': output << "\\\\"; break;
            case '"': output << "\\\""; break;
            case '\n': output << "\\n"; break;
            case '\r': output << "\\r"; break;
            case '\t': output << "\\t"; break;
            default:
                if (character < 0x20) {
                    output << "\\u" << std::hex << std::setw(4) << std::setfill('0')
                           << static_cast<int>(character) << std::dec;
                } else {
                    output << character;
                }
        }
    }
    return output.str();
}

std::string device_name(cudaDeviceProp &properties) {
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaGetDeviceProperties(&properties, device));
    return properties.name;
}

std::string response_json(
    const Request &request,
    const RunResult &run,
    const cudaDeviceProp &properties) {
    const double repetitions = request.repetitions;
    const double kernel_seconds = run.mx_seconds + run.mt_seconds;
    std::ostringstream output;
    output << std::setprecision(17);
    output << "{\n"
           << "  \"schema\": \"" << RESPONSE_SCHEMA << "\",\n"
           << "  \"terminal\": \"holdout_cuda_b_backend_calibration_complete\",\n"
           << "  \"claim_boundary\": \"CUDA M^T M backend timing and deterministic output only; no recurrence, kernel, Holdout supplier, Stage 5, or key recovery\",\n"
           << "  \"backend\": \"" << json_escape(run.backend) << "\",\n"
           << "  \"request_schema\": \"" << json_escape(request.schema) << "\",\n"
           << "  \"public_sha256\": \"" << json_escape(request.public_sha256) << "\",\n"
           << "  \"gpu_name\": \"" << json_escape(properties.name) << "\",\n"
           << "  \"compute_capability\": \"" << properties.major << "." << properties.minor << "\",\n"
           << "  \"compiled_sm\": " << MCELIECEX_CUDA_TARGET_ARCH << ",\n"
           << "  \"k\": " << request.k << ",\n"
           << "  \"degree\": " << request.degree << ",\n"
           << "  \"words\": " << request.words << ",\n"
           << "  \"block_bits\": " << 64 * request.words << ",\n"
           << "  \"seed\": " << request.seed << ",\n"
           << "  \"repetitions\": " << request.repetitions << ",\n"
           << "  \"point_count\": " << request.points.size() << ",\n"
           << "  \"columns\": " << request.expected_columns << ",\n"
           << "  \"rows\": " << request.expected_rows << ",\n"
           << "  \"structural_nonzeros\": " << request.expected_nonzeros << ",\n"
           << "  \"free_device_bytes_before\": " << run.free_device_bytes_before << ",\n"
           << "  \"total_device_bytes\": " << run.total_device_bytes << ",\n"
           << "  \"allocated_device_bytes\": " << run.allocated_device_bytes << ",\n"
           << "  \"slice_mapping_entries\": " << run.mapping_entries << ",\n"
           << "  \"host_to_device_seconds\": " << run.host_to_device_seconds << ",\n"
           << "  \"device_to_host_seconds\": " << run.device_to_host_seconds << ",\n"
           << "  \"preprocessing_seconds\": " << run.preprocessing_seconds << ",\n"
           << "  \"mx_seconds_total\": " << run.mx_seconds << ",\n"
           << "  \"mt_seconds_total\": " << run.mt_seconds << ",\n"
           << "  \"kernel_seconds_total\": " << kernel_seconds << ",\n"
           << "  \"kernel_seconds_per_repetition\": " << kernel_seconds / repetitions << ",\n"
           << "  \"output_fnv1a64_le\": \"" << hex_u64(checksum_words(run.output)) << "\"";
    if (request.emit_output) {
        output << ",\n  \"output_words_hex\": [";
        for (std::size_t index = 0; index < run.output.size(); ++index) {
            if (index != 0) {
                output << ",";
            }
            output << "\"" << hex_u64(run.output[index]) << "\"";
        }
        output << "]";
    }
    output << "\n}\n";
    return output.str();
}

std::string left_response_json(
    const Request &request,
    const RunResult &run,
    const cudaDeviceProp &properties) {
    const double repetitions = request.repetitions;
    const double kernel_seconds = run.mt_seconds + run.mx_seconds;
    std::ostringstream output;
    output << std::setprecision(17);
    output << "{\n"
           << "  \"schema\": \"" << LEFT_RESPONSE_SCHEMA << "\",\n"
           << "  \"terminal\": \"holdout_cuda_left_mm_transpose_bounded_complete\",\n"
           << "  \"claim_boundary\": \"CUDA M M^T in the documented slice-row permutation, with emitted M^T input for bounded differential replay only; no recurrence, Kloc quotient, relation yield, locator recovery, or key recovery\",\n"
           << "  \"backend\": \"" << json_escape(run.backend) << "\",\n"
           << "  \"request_schema\": \"" << json_escape(request.schema) << "\",\n"
           << "  \"public_sha256\": \"" << json_escape(request.public_sha256) << "\",\n"
           << "  \"gpu_name\": \"" << json_escape(properties.name) << "\",\n"
           << "  \"compute_capability\": \"" << properties.major << "." << properties.minor << "\",\n"
           << "  \"compiled_sm\": " << MCELIECEX_CUDA_TARGET_ARCH << ",\n"
           << "  \"orientation\": \"M_M_transpose_on_literal_range\",\n"
           << "  \"range_row_order\": \"point_then_complement_size_then_complement_colex_then_level_then_support_derivative_colex\",\n"
           << "  \"k\": " << request.k << ",\n"
           << "  \"degree\": " << request.degree << ",\n"
           << "  \"words\": " << request.words << ",\n"
           << "  \"block_bits\": " << 64 * request.words << ",\n"
           << "  \"seed\": " << request.seed << ",\n"
           << "  \"repetitions\": " << request.repetitions << ",\n"
           << "  \"point_count\": " << request.points.size() << ",\n"
           << "  \"columns\": " << request.expected_columns << ",\n"
           << "  \"rows\": " << request.expected_rows << ",\n"
           << "  \"structural_nonzeros\": " << request.expected_nonzeros << ",\n"
           << "  \"free_device_bytes_before\": " << run.free_device_bytes_before << ",\n"
           << "  \"total_device_bytes\": " << run.total_device_bytes << ",\n"
           << "  \"allocated_device_bytes\": " << run.allocated_device_bytes << ",\n"
           << "  \"slice_mapping_entries\": " << run.mapping_entries << ",\n"
           << "  \"host_to_device_seconds\": " << run.host_to_device_seconds << ",\n"
           << "  \"device_to_host_seconds\": " << run.device_to_host_seconds << ",\n"
           << "  \"preprocessing_seconds\": " << run.preprocessing_seconds << ",\n"
           << "  \"mt_seconds_total\": " << run.mt_seconds << ",\n"
           << "  \"mx_seconds_total\": " << run.mx_seconds << ",\n"
           << "  \"kernel_seconds_total\": " << kernel_seconds << ",\n"
           << "  \"kernel_seconds_per_repetition\": "
           << kernel_seconds / repetitions << ",\n"
           << "  \"transpose_output_fnv1a64_le\": \""
           << hex_u64(checksum_words(run.transpose_output)) << "\",\n"
           << "  \"output_fnv1a64_le\": \""
           << hex_u64(checksum_words(run.output)) << "\"";
    if (request.emit_output) {
        output << ",\n  \"transpose_output_words_hex\": [";
        for (std::size_t index = 0;
             index < run.transpose_output.size();
             ++index) {
            if (index != 0) {
                output << ",";
            }
            output << "\"" << hex_u64(run.transpose_output[index]) << "\"";
        }
        output << "],\n  \"output_words_hex\": [";
        for (std::size_t index = 0; index < run.output.size(); ++index) {
            if (index != 0) {
                output << ",";
            }
            output << "\"" << hex_u64(run.output[index]) << "\"";
        }
        output << "]";
    }
    output << "\n}\n";
    return output.str();
}

std::string projection_response_json(
    const Request &request,
    const ProjectionRunResult &run,
    const cudaDeviceProp &properties) {
    std::ostringstream output;
    output << std::setprecision(17);
    output << "{\n"
           << "  \"schema\": \"mceliecex-holdout-cuda-projection-response-v1\",\n"
           << "  \"terminal\": \"holdout_cuda_full_projection_calibration_complete\",\n"
           << "  \"claim_boundary\": \"full CUDA Z^T V projection timing and deterministic output only; no recurrence, kernel, Holdout supplier, Stage 5, or key recovery\",\n"
           << "  \"request_schema\": \"" << json_escape(request.schema) << "\",\n"
           << "  \"public_sha256\": \"" << json_escape(request.public_sha256) << "\",\n"
           << "  \"gpu_name\": \"" << json_escape(properties.name) << "\",\n"
           << "  \"compute_capability\": \"" << properties.major << "." << properties.minor << "\",\n"
           << "  \"compiled_sm\": " << MCELIECEX_CUDA_TARGET_ARCH << ",\n"
           << "  \"columns\": " << request.expected_columns << ",\n"
           << "  \"words\": " << request.words << ",\n"
           << "  \"block_bits\": " << 64 * request.words << ",\n"
           << "  \"seed_v\": " << request.seed << ",\n"
           << "  \"seed_z\": " << (request.seed ^ PROJECTION_SEED_XOR) << ",\n"
           << "  \"projection_chunks\": " << run.chunks << ",\n"
           << "  \"free_device_bytes_before\": " << run.free_device_bytes_before << ",\n"
           << "  \"total_device_bytes\": " << run.total_device_bytes << ",\n"
           << "  \"allocated_device_bytes\": " << run.allocated_device_bytes << ",\n"
           << "  \"host_to_device_seconds\": " << run.host_to_device_seconds << ",\n"
           << "  \"device_to_host_seconds\": " << run.device_to_host_seconds << ",\n"
           << "  \"kernel_seconds\": " << run.kernel_seconds << ",\n"
           << "  \"output_fnv1a64_le\": \""
           << hex_u64(checksum_words(run.output)) << "\"";
    if (request.emit_output) {
        output << ",\n  \"output_words_hex\": [";
        for (std::size_t index = 0; index < run.output.size(); ++index) {
            if (index != 0) {
                output << ",";
            }
            output << "\"" << hex_u64(run.output[index]) << "\"";
        }
        output << "]";
    }
    output << "\n}\n";
    return output.str();
}

u64 histogram_count_through(const std::vector<u64> &histogram, std::size_t limit) {
    u64 total = 0;
    for (std::size_t degree = 0;
         degree < histogram.size() && degree <= limit;
         ++degree) {
        total += histogram[degree];
    }
    return total;
}

std::size_t histogram_quantile(
    const std::vector<u64> &histogram,
    u64 numerator,
    u64 denominator) {
    u64 total = 0;
    for (const u64 count : histogram) {
        total += count;
    }
    if (total == 0 || denominator == 0 || numerator > denominator) {
        throw std::runtime_error("invalid column-degree quantile request");
    }
    const u64 rank = static_cast<u64>(
        (static_cast<unsigned __int128>(total - 1) * numerator) / denominator);
    u64 cumulative = 0;
    for (std::size_t degree = 0; degree < histogram.size(); ++degree) {
        cumulative += histogram[degree];
        if (cumulative > rank) {
            return degree;
        }
    }
    throw std::runtime_error("column-degree quantile exceeds histogram");
}

std::string column_degree_census_json(
    const Request &request,
    const ColumnDegreeCensusResult &result,
    const cudaDeviceProp &properties) {
    std::size_t minimum = 0;
    while (minimum < result.histogram.size() && result.histogram[minimum] == 0) {
        ++minimum;
    }
    std::size_t maximum = result.histogram.size();
    while (maximum > 0 && result.histogram[maximum - 1] == 0) {
        --maximum;
    }
    if (minimum == result.histogram.size() || maximum == 0) {
        throw std::runtime_error("column-degree census is empty");
    }
    --maximum;

    std::ostringstream output;
    output << std::setprecision(17)
           << "{\n"
           << "  \"schema\": \"mceliecex-holdout-cuda-column-degree-census-v1\",\n"
           << "  \"terminal\": \"holdout_cuda_column_degree_census_complete\",\n"
           << "  \"claim_boundary\": \"exact initial column-degree census only; no elimination, rank, kernel, Holdout supplier, Stage 5, or key recovery\",\n"
           << "  \"request_schema\": \"" << json_escape(request.schema) << "\",\n"
           << "  \"public_sha256\": \"" << json_escape(request.public_sha256) << "\",\n"
           << "  \"gpu_name\": \"" << json_escape(properties.name) << "\",\n"
           << "  \"compute_capability\": \"" << properties.major << "." << properties.minor << "\",\n"
           << "  \"compiled_sm\": " << MCELIECEX_CUDA_TARGET_ARCH << ",\n"
           << "  \"columns\": " << request.expected_columns << ",\n"
           << "  \"rows\": " << request.expected_rows << ",\n"
           << "  \"structural_nonzeros\": " << request.expected_nonzeros << ",\n"
           << "  \"average_degree\": "
           << static_cast<double>(request.expected_nonzeros)
                / static_cast<double>(request.expected_columns) << ",\n"
           << "  \"minimum_degree\": " << minimum << ",\n"
           << "  \"median_degree\": "
           << histogram_quantile(result.histogram, 1, 2) << ",\n"
           << "  \"p90_degree\": "
           << histogram_quantile(result.histogram, 9, 10) << ",\n"
           << "  \"p99_degree\": "
           << histogram_quantile(result.histogram, 99, 100) << ",\n"
           << "  \"p999_degree\": "
           << histogram_quantile(result.histogram, 999, 1000) << ",\n"
           << "  \"maximum_degree\": " << maximum << ",\n"
           << "  \"degree_at_most_1\": "
           << histogram_count_through(result.histogram, 1) << ",\n"
           << "  \"degree_at_most_2\": "
           << histogram_count_through(result.histogram, 2) << ",\n"
           << "  \"degree_at_most_4\": "
           << histogram_count_through(result.histogram, 4) << ",\n"
           << "  \"degree_at_most_8\": "
           << histogram_count_through(result.histogram, 8) << ",\n"
           << "  \"degree_at_most_16\": "
           << histogram_count_through(result.histogram, 16) << ",\n"
           << "  \"degree_at_most_32\": "
           << histogram_count_through(result.histogram, 32) << ",\n"
           << "  \"degree_bound\": " << result.histogram.size() - 1 << ",\n"
           << "  \"free_device_bytes_before\": "
           << result.free_device_bytes_before << ",\n"
           << "  \"total_device_bytes\": " << result.total_device_bytes << ",\n"
           << "  \"allocated_device_bytes\": "
           << result.allocated_device_bytes << ",\n"
           << "  \"host_to_device_seconds\": "
           << result.host_to_device_seconds << ",\n"
           << "  \"kernel_seconds\": " << result.kernel_seconds << ",\n"
           << "  \"device_to_host_seconds\": "
           << result.device_to_host_seconds << ",\n"
           << "  \"nonzero_histogram\": [";
    bool first = true;
    for (std::size_t degree = 0; degree < result.histogram.size(); ++degree) {
        if (result.histogram[degree] == 0) {
            continue;
        }
        if (!first) {
            output << ",";
        }
        first = false;
        output << "[" << degree << "," << result.histogram[degree] << "]";
    }
    output << "]\n}\n";
    return output.str();
}

std::string literal_witness_response_json(
    const WitnessRequest &request,
    const WitnessRunResult &result,
    const cudaDeviceProp &properties,
    std::uintmax_t domain_bytes) {
    const Request &layout = request.layout;
    std::ostringstream output;
    output << std::setprecision(17)
           << "{\n"
           << "  \"schema\": \"" << WITNESS_RESPONSE_SCHEMA << "\",\n"
           << "  \"terminal\": \""
           << (request.verify_target
                   ? "tii254_cuda_literal_witness_target_replay_pass"
                   : "tii254_cuda_literal_witness_transpose_bounded_complete")
           << "\",\n"
           << "  \"claim_boundary\": \"candidate-conditioned branch-zero literal coefficient generation and exact transpose only; no public Holdout quotient, HOVER selectivity, kernel, finisher, or key recovery\",\n"
           << "  \"request_schema\": \"" << WITNESS_REQUEST_SCHEMA << "\",\n"
           << "  \"public_sha256\": \""
           << json_escape(layout.public_sha256) << "\",\n"
           << "  \"gpu_name\": \"" << json_escape(properties.name) << "\",\n"
           << "  \"compute_capability\": \"" << properties.major << "."
           << properties.minor << "\",\n"
           << "  \"compiled_sm\": " << MCELIECEX_CUDA_TARGET_ARCH << ",\n"
           << "  \"k\": " << layout.k << ",\n"
           << "  \"degree\": " << layout.degree << ",\n"
           << "  \"target_count\": " << WITNESS_TARGETS << ",\n"
           << "  \"point_count\": " << layout.points.size() << ",\n"
           << "  \"literal_rows\": " << layout.expected_rows << ",\n"
           << "  \"ambient_columns\": " << layout.expected_columns << ",\n"
           << "  \"logical_slice_mapping_entries\": "
           << result.logical_mapping_entries << ",\n"
           << "  \"coefficient_packing\": \"three_GF256_polynomial_basis_bytes_in_low_24_bits\",\n"
           << "  \"field_modulus\": \"x^8+x^4+x^3+x^2+1\",\n"
           << "  \"domain_file\": \"domain.bin\",\n"
           << "  \"domain_bytes\": " << domain_bytes << ",\n"
           << "  \"domain_fnv1a64_le\": \""
           << hex_u64(checksum_words(result.output)) << "\",\n"
           << "  \"free_device_bytes_before\": "
           << result.free_device_bytes_before << ",\n"
           << "  \"total_device_bytes\": "
           << result.total_device_bytes << ",\n"
           << "  \"allocated_device_bytes\": "
           << result.allocated_device_bytes << ",\n"
           << "  \"host_to_device_seconds\": "
           << result.host_to_device_seconds << ",\n"
           << "  \"device_to_host_seconds\": "
           << result.device_to_host_seconds << ",\n"
           << "  \"coefficient_kernel_seconds\": "
           << result.coefficient_kernel_seconds << ",\n"
           << "  \"transpose_kernel_seconds\": "
           << result.transpose_kernel_seconds << ",\n"
           << "  \"target_replay_performed\": "
           << (request.verify_target ? "true" : "false") << ",\n"
           << "  \"target_replay_mismatches\": "
           << result.target_replay_mismatches << ",\n"
           << "  \"target_replay_kernel_seconds\": "
           << result.target_replay_kernel_seconds << ",\n"
           << "  \"target_replay_pass\": "
           << (request.verify_target
                   && result.target_replay_mismatches == 0 ? "true" : "false");
    if (request.emit_coefficients) {
        output << ",\n  \"coefficient_fnv1a64_le\": \""
               << hex_u64(checksum_words(result.emitted_coefficients))
               << "\",\n"
               << "  \"coefficient_words_hex\": [";
        for (std::size_t index = 0;
             index < result.emitted_coefficients.size();
             ++index) {
            if (index != 0) {
                output << ",";
            }
            output << "\"" << hex_u64(result.emitted_coefficients[index])
                   << "\"";
        }
        output << "]";
    }
    output << "\n}\n";
    return output.str();
}

void write_new(const std::filesystem::path &path, const std::string &contents) {
    if (std::filesystem::exists(path)) {
        throw std::runtime_error("refusing to overwrite response: " + path.string());
    }
    const std::filesystem::path temporary = path.string() + ".tmp";
    if (std::filesystem::exists(temporary)) {
        throw std::runtime_error("response temporary already exists: " + temporary.string());
    }
    {
        std::ofstream stream(temporary, std::ios::binary | std::ios::trunc);
        if (!stream) {
            throw std::runtime_error("cannot create response temporary");
        }
        stream.write(contents.data(), static_cast<std::streamsize>(contents.size()));
        stream.flush();
        if (!stream) {
            throw std::runtime_error("cannot write response temporary");
        }
    }
    std::error_code error;
    std::filesystem::rename(temporary, path, error);
    if (error) {
        std::filesystem::remove(temporary);
        throw std::runtime_error("cannot publish response: " + error.message());
    }
}

u64 update_checksum_words(u64 hash, const std::vector<u64> &values) {
    for (const u64 value : values) {
        for (int byte = 0; byte < 8; ++byte) {
            hash ^= (value >> (8 * byte)) & 0xff;
            hash *= 1099511628211ULL;
        }
    }
    return hash;
}

void publish_temporary_new(
    const std::filesystem::path &temporary,
    const std::filesystem::path &final_path) {
    if (std::filesystem::exists(final_path)) {
        throw std::runtime_error(
            "refusing to overwrite published artifact: " + final_path.string());
    }
    std::error_code error;
    std::filesystem::rename(temporary, final_path, error);
    if (error) {
        throw std::runtime_error(
            "cannot publish artifact " + final_path.string() + ": "
            + error.message());
    }
}

void fsync_path(const std::filesystem::path &path, bool directory) {
    const int flags = directory ? (O_RDONLY | O_DIRECTORY) : O_RDONLY;
    const int descriptor = ::open(path.c_str(), flags);
    if (descriptor < 0) {
        throw std::runtime_error(
            "cannot open artifact for fsync: " + path.string() + ": "
            + std::strerror(errno));
    }
    if (::fsync(descriptor) != 0) {
        const std::string message = std::strerror(errno);
        ::close(descriptor);
        throw std::runtime_error(
            "cannot fsync artifact: " + path.string() + ": " + message);
    }
    if (::close(descriptor) != 0) {
        throw std::runtime_error(
            "cannot close synced artifact: " + path.string());
    }
}

void write_literal_domain(
    const std::filesystem::path &path,
    const WitnessRequest &request,
    const std::vector<u64> &output) {
    const std::uint16_t endian_probe = 1;
    if (*reinterpret_cast<const unsigned char *>(&endian_probe) != 1) {
        throw std::runtime_error(
            "literal-witness domain v1 requires a little-endian host");
    }
    std::ofstream stream(path, std::ios::binary | std::ios::trunc);
    if (!stream) {
        throw std::runtime_error("cannot create literal-witness domain");
    }
    stream << WITNESS_DOMAIN_SCHEMA << "\n"
           << "public_sha256 " << request.layout.public_sha256 << "\n"
           << "k " << request.layout.k << "\n"
           << "degree " << request.layout.degree << "\n"
           << "columns " << request.layout.expected_columns << "\n"
           << "target_count " << WITNESS_TARGETS << "\n"
           << "packing three_GF256_polynomial_basis_bytes_in_low_24_bits\n"
           << "data_le\n";
    stream.write(
        reinterpret_cast<const char *>(output.data()),
        static_cast<std::streamsize>(output.size() * sizeof(u64)));
    stream.flush();
    if (!stream) {
        throw std::runtime_error("cannot write literal-witness domain");
    }
}

struct LoadedKrylovState {
    std::vector<u64> panel;
    u64 completed_steps = 0;
};

LoadedKrylovState read_or_create_krylov_state(
    const Request &request,
    const std::string &state_path,
    bool injective_padding) {
    const std::uint16_t endian_probe = 1;
    if (*reinterpret_cast<const unsigned char *>(&endian_probe) != 1) {
        throw std::runtime_error(
            "Krylov state v1 requires a little-endian host");
    }
    const u64 state_coordinates = injective_padding
        ? request.expected_rows
        : request.expected_columns;
    const char *state_schema = injective_padding
        ? INJECTIVE_STATE_SCHEMA
        : STATE_SCHEMA;
    const std::size_t panel_words = static_cast<std::size_t>(
        state_coordinates) * request.words;
    if (state_path == "-") {
        return LoadedKrylovState{
            random_words(panel_words, request.seed),
            0,
        };
    }
    std::ifstream stream(state_path, std::ios::binary);
    if (!stream) {
        throw std::runtime_error("cannot open Krylov state: " + state_path);
    }
    std::string schema;
    std::getline(stream, schema);
    if (schema != state_schema) {
        throw std::runtime_error("Krylov state schema differs");
    }
    std::string key;
    std::string public_sha256;
    int k = 0;
    int degree = 0;
    int words = 0;
    u64 columns = 0;
    u64 seed = 0;
    u64 completed_steps = 0;
    u64 declared_panel_words = 0;
    auto require_key = [&](const char *expected) {
        if (!(stream >> key) || key != expected) {
            throw std::runtime_error(
                std::string("expected Krylov state key ") + expected);
        }
    };
    require_key("public_sha256"); stream >> public_sha256;
    require_key("k"); stream >> k;
    require_key("degree"); stream >> degree;
    require_key("words"); stream >> words;
    require_key("columns"); stream >> columns;
    require_key("seed"); stream >> seed;
    require_key("completed_steps"); stream >> completed_steps;
    require_key("panel_words"); stream >> declared_panel_words;
    require_key("data_le");
    char newline = 0;
    stream.get(newline);
    if (newline != '\n') {
        throw std::runtime_error("Krylov state binary boundary differs");
    }
    if (public_sha256 != request.public_sha256
        || k != request.k
        || degree != request.degree
        || words != request.words
        || columns != state_coordinates
        || seed != request.seed
        || declared_panel_words != panel_words) {
        throw std::runtime_error("Krylov state metadata differs from request");
    }
    LoadedKrylovState state;
    state.completed_steps = completed_steps;
    state.panel.resize(panel_words);
    stream.read(
        reinterpret_cast<char *>(state.panel.data()),
        static_cast<std::streamsize>(panel_words * sizeof(u64)));
    if (stream.gcount()
        != static_cast<std::streamsize>(panel_words * sizeof(u64))) {
        throw std::runtime_error("Krylov state panel is truncated");
    }
    char trailing = 0;
    if (stream.get(trailing)) {
        throw std::runtime_error("Krylov state has trailing data");
    }
    return state;
}

void write_krylov_state_temporary(
    const std::filesystem::path &path,
    const Request &request,
    u64 completed_steps,
    const std::vector<u64> &panel,
    bool injective_padding) {
    std::ofstream stream(path, std::ios::binary | std::ios::trunc);
    if (!stream) {
        throw std::runtime_error("cannot create Krylov state temporary");
    }
    const u64 state_coordinates = injective_padding
        ? request.expected_rows
        : request.expected_columns;
    stream << (injective_padding ? INJECTIVE_STATE_SCHEMA : STATE_SCHEMA) << "\n"
           << "public_sha256 " << request.public_sha256 << "\n"
           << "k " << request.k << "\n"
           << "degree " << request.degree << "\n"
           << "words " << request.words << "\n"
           << "columns " << state_coordinates << "\n"
           << "seed " << request.seed << "\n"
           << "completed_steps " << completed_steps << "\n"
           << "panel_words " << panel.size() << "\n"
           << "data_le\n";
    stream.write(
        reinterpret_cast<const char *>(panel.data()),
        static_cast<std::streamsize>(panel.size() * sizeof(u64)));
    stream.flush();
    if (!stream) {
        throw std::runtime_error("cannot write Krylov state temporary");
    }
}

void write_sequence_header(
    std::ostream &stream,
    const Request &request,
    u64 start_step,
    u64 term_count,
    bool injective_padding) {
    const u64 term_words = static_cast<u64>(64)
        * request.words * request.words;
    const u64 state_coordinates = injective_padding
        ? request.expected_rows
        : request.expected_columns;
    stream << (injective_padding ? INJECTIVE_SEQUENCE_SCHEMA : SEQUENCE_SCHEMA)
           << "\n"
           << "public_sha256 " << request.public_sha256 << "\n"
           << "k " << request.k << "\n"
           << "degree " << request.degree << "\n"
           << "words " << request.words << "\n"
           << "columns " << state_coordinates << "\n"
           << "seed_y " << request.seed << "\n"
           << "seed_z " << (request.seed ^ PROJECTION_SEED_XOR) << "\n"
           << "start_step " << start_step << "\n"
           << "term_count " << term_count << "\n"
           << "term_words " << term_words << "\n"
           << "origin "
           << (injective_padding
                   ? "Z^T_(J_M^T)^(i+1)_Y"
                   : "Z^T_B^(i+1)_Y")
           << "\n"
           << "data_le\n";
    if (!stream) {
        throw std::runtime_error("cannot write Krylov sequence header");
    }
}

void run_krylov_segment(
    const std::filesystem::path &request_path,
    const std::string &input_state_path,
    u64 step_count,
    const std::filesystem::path &output_directory,
    bool recompute_mapping,
    bool injective_padding) {
    if (step_count == 0) {
        throw std::runtime_error("Krylov segment step count must be positive");
    }
    if (std::filesystem::exists(output_directory)) {
        throw std::runtime_error(
            "refusing existing Krylov segment directory: "
            + output_directory.string());
    }
    const auto wall_started = Clock::now();
    Request request = read_request(request_path);
    if (injective_padding && !recompute_mapping) {
        throw std::runtime_error(
            "injective padding currently requires recomputed transpose slices");
    }
    CUDA_CHECK(cudaMemcpyToSymbol(
        DEVICE_CHOOSE, HOST_CHOOSE.values.data(), sizeof(HOST_CHOOSE.values)));
    LoadedKrylovState state = read_or_create_krylov_state(
        request, input_state_path, injective_padding);
    if (step_count > std::numeric_limits<u64>::max() - state.completed_steps) {
        throw std::runtime_error("Krylov completed-step counter would overflow");
    }
    const u64 start_step = state.completed_steps;
    const u64 end_step = start_step + step_count;

    auto preprocessing_started = Clock::now();
    const SlicePlan plan = build_slice_plan(request, !recompute_mapping);
    if (plan.total_range_rows != request.expected_rows) {
        throw std::runtime_error(
            "slice-row permutation does not cover every coefficient coordinate");
    }
    const std::vector<unsigned char> positions = point_position_tables(request);
    const u64 state_coordinates = injective_padding
        ? request.expected_rows
        : request.expected_columns;
    const int projection_chunks = projection_chunk_count(state_coordinates);
    const std::size_t panel_words = static_cast<std::size_t>(
        state_coordinates) * request.words;
    const std::size_t panel_bytes = panel_words * sizeof(u64);
    const std::size_t range_bytes = static_cast<std::size_t>(
        injective_padding ? request.expected_columns : plan.max_range_rows)
        * request.words * sizeof(u64);
    const std::size_t gathered_bytes = recompute_mapping
        ? 0
        : static_cast<std::size_t>(plan.max_class_mapping_entries)
            * request.words * sizeof(u64);
    const std::size_t mapping_bytes = recompute_mapping
        ? 0
        : static_cast<std::size_t>(plan.mapping_entries)
            * sizeof(std::uint32_t);
    const std::size_t pattern_column_bytes = recompute_mapping
        ? 0
        : plan.pattern_columns.size() * sizeof(std::uint32_t);
    const std::size_t pattern_start_bytes = recompute_mapping
        ? 0
        : plan.pattern_starts.size() * sizeof(std::uint32_t);
    const std::size_t position_bytes = request.points.size() * 2 * MAX_K;
    const std::size_t term_words = static_cast<std::size_t>(64)
        * request.words * request.words;
    const std::size_t term_bytes = term_words * sizeof(u64);
    const std::size_t partial_bytes = term_bytes * projection_chunks;
    const std::size_t allocated_bytes = 3 * panel_bytes + range_bytes
        + gathered_bytes + mapping_bytes + pattern_column_bytes
        + pattern_start_bytes + position_bytes + partial_bytes + term_bytes;
    std::size_t free_bytes = 0;
    std::size_t total_bytes = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
    constexpr std::size_t reserve = std::size_t{256} << 20;
    if (allocated_bytes > free_bytes || free_bytes - allocated_bytes < reserve) {
        throw std::runtime_error(
            "Krylov CUDA allocation would leave less than 256 MiB free");
    }

    DeviceBuffers buffers;
    CUDA_CHECK(cudaMalloc(&buffers.input, panel_bytes));
    CUDA_CHECK(cudaMalloc(&buffers.output, panel_bytes));
    CUDA_CHECK(cudaMalloc(&buffers.range, range_bytes));
    if (!recompute_mapping) {
        CUDA_CHECK(cudaMalloc(&buffers.gathered, gathered_bytes));
        CUDA_CHECK(cudaMalloc(&buffers.mapping, mapping_bytes));
        CUDA_CHECK(cudaMalloc(
            &buffers.pattern_columns, pattern_column_bytes));
        CUDA_CHECK(cudaMalloc(
            &buffers.pattern_starts, pattern_start_bytes));
    }
    CUDA_CHECK(cudaMalloc(&buffers.positions, position_bytes));
    u64 *device_z = nullptr;
    u64 *device_partials = nullptr;
    u64 *device_term = nullptr;
    struct ExtraCleanup {
        u64 *&z;
        u64 *&partials;
        u64 *&term;
        ~ExtraCleanup() {
            cudaFree(term);
            cudaFree(partials);
            cudaFree(z);
        }
    } extra_cleanup{device_z, device_partials, device_term};
    CUDA_CHECK(cudaMalloc(&device_z, panel_bytes));
    CUDA_CHECK(cudaMalloc(&device_partials, partial_bytes));
    CUDA_CHECK(cudaMalloc(&device_term, term_bytes));

    std::vector<u64> z = random_words(
        panel_words, request.seed ^ PROJECTION_SEED_XOR);
    CUDA_CHECK(cudaMemcpy(
        buffers.input,
        state.panel.data(),
        panel_bytes,
        cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(
        device_z, z.data(), panel_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(
        buffers.positions,
        positions.data(),
        position_bytes,
        cudaMemcpyHostToDevice));
    if (!recompute_mapping) {
        CUDA_CHECK(cudaMemcpy(
            buffers.pattern_columns,
            plan.pattern_columns.data(),
            pattern_column_bytes,
            cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(
            buffers.pattern_starts,
            plan.pattern_starts.data(),
            pattern_start_bytes,
            cudaMemcpyHostToDevice));
        launch_fill_slice_mapping(request, plan, buffers);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    z.clear();
    z.shrink_to_fit();
    const double preprocessing_seconds = std::chrono::duration<double>(
        Clock::now() - preprocessing_started).count();

    if (!std::filesystem::create_directory(output_directory)) {
        throw std::runtime_error(
            "cannot create Krylov segment directory: "
            + output_directory.string());
    }
    const std::filesystem::path sequence_temporary =
        output_directory / "sequence.bin.tmp";
    const std::filesystem::path sequence_path =
        output_directory / "sequence.bin";
    const std::filesystem::path state_temporary =
        output_directory / "state.bin.tmp";
    const std::filesystem::path state_path =
        output_directory / "state.bin";
    std::ofstream sequence(
        sequence_temporary, std::ios::binary | std::ios::trunc);
    if (!sequence) {
        throw std::runtime_error("cannot create Krylov sequence temporary");
    }
    write_sequence_header(
        sequence, request, start_step, step_count, injective_padding);

    cudaEvent_t before_b = nullptr;
    cudaEvent_t after_b = nullptr;
    cudaEvent_t after_projection = nullptr;
    CUDA_CHECK(cudaEventCreate(&before_b));
    CUDA_CHECK(cudaEventCreate(&after_b));
    CUDA_CHECK(cudaEventCreate(&after_projection));
    struct EventCleanup {
        cudaEvent_t &a;
        cudaEvent_t &b;
        cudaEvent_t &c;
        ~EventCleanup() { cudaEventDestroy(c); cudaEventDestroy(b); cudaEventDestroy(a); }
    } event_cleanup{before_b, after_b, after_projection};

    std::vector<u64> term(term_words);
    u64 sequence_checksum = 1469598103934665603ULL;
    double operator_seconds = 0;
    double projection_seconds = 0;
    for (u64 local_step = 0; local_step < step_count; ++local_step) {
        CUDA_CHECK(cudaEventRecord(before_b));
        if (injective_padding) {
            const std::size_t domain_bytes = static_cast<std::size_t>(
                request.expected_columns) * request.words * sizeof(u64);
            launch_injective_slice_recompute_device(
                request,
                plan,
                buffers,
                panel_bytes,
                domain_bytes);
        } else if (recompute_mapping) {
            launch_b_slice_recompute_device(
                request, plan, buffers, panel_bytes);
        } else {
            launch_b_slice_device(request, plan, buffers, panel_bytes);
        }
        CUDA_CHECK(cudaEventRecord(after_b));
        launch_projection_device(
            device_z,
            buffers.output,
            device_partials,
            device_term,
            state_coordinates,
            request.words,
            projection_chunks);
        CUDA_CHECK(cudaEventRecord(after_projection));
        CUDA_CHECK(cudaEventSynchronize(after_projection));
        float operator_ms = 0;
        float projection_ms = 0;
        CUDA_CHECK(cudaEventElapsedTime(
            &operator_ms, before_b, after_b));
        CUDA_CHECK(cudaEventElapsedTime(
            &projection_ms, after_b, after_projection));
        operator_seconds += operator_ms / 1000.0;
        projection_seconds += projection_ms / 1000.0;
        CUDA_CHECK(cudaMemcpy(
            term.data(), device_term, term_bytes, cudaMemcpyDeviceToHost));
        sequence.write(
            reinterpret_cast<const char *>(term.data()),
            static_cast<std::streamsize>(term_bytes));
        if (!sequence) {
            throw std::runtime_error("cannot append Krylov projected term");
        }
        sequence_checksum = update_checksum_words(sequence_checksum, term);
        std::swap(buffers.input, buffers.output);
        const double elapsed = std::chrono::duration<double>(
            Clock::now() - wall_started).count();
        std::cout << "{\"schema\":\"mceliecex-holdout-cuda-heartbeat-v1\","
                  << "\"stage\":\"krylov\","
                  << "\"completed_steps\":" << (start_step + local_step + 1) << ","
                  << "\"segment_end_step\":" << end_step << ","
                  << "\"elapsed_seconds\":" << std::setprecision(17)
                  << elapsed << "}\n" << std::flush;
    }
    sequence.flush();
    if (!sequence) {
        throw std::runtime_error("cannot flush Krylov sequence temporary");
    }
    sequence.close();

    state.panel.resize(panel_words);
    CUDA_CHECK(cudaMemcpy(
        state.panel.data(),
        buffers.input,
        panel_bytes,
        cudaMemcpyDeviceToHost));
    state.completed_steps = end_step;
    write_krylov_state_temporary(
        state_temporary,
        request,
        end_step,
        state.panel,
        injective_padding);
    fsync_path(sequence_temporary, false);
    fsync_path(state_temporary, false);
    publish_temporary_new(sequence_temporary, sequence_path);
    publish_temporary_new(state_temporary, state_path);
    fsync_path(output_directory, true);

    const u64 state_checksum = checksum_words(state.panel);
    const double wall_seconds = std::chrono::duration<double>(
        Clock::now() - wall_started).count();
    std::ostringstream manifest;
    manifest << std::setprecision(17)
             << "{\n"
             << "  \"schema\": \""
             << (injective_padding
                     ? "mceliecex-holdout-cuda-injective-krylov-segment-result-v1"
                     : "mceliecex-holdout-cuda-krylov-segment-result-v1")
             << "\",\n"
             << "  \"terminal\": \""
             << (injective_padding
                     ? "holdout_cuda_injective_krylov_segment_complete"
                     : "holdout_cuda_krylov_segment_complete")
             << "\",\n"
             << "  \"claim_boundary\": \""
             << (injective_padding
                     ? "restartable CUDA B=J*M^T terms only; coefficient state uses the bound slice-row permutation and prefix injection; no recurrence, reconstructed relation, quotient yield, or key recovery"
                     : "restartable CUDA Krylov terms only; no recurrence, kernel, Holdout supplier, Stage 5, or key recovery")
             << "\",\n"
             << "  \"public_sha256\": \"" << json_escape(request.public_sha256) << "\",\n"
             << "  \"origin\": \""
             << (injective_padding
                     ? "Z^T (J M^T)^(i+1) Y"
                     : "Z^T B^(i+1) Y")
             << "\",\n"
             << "  \"seed_y\": " << request.seed << ",\n"
             << "  \"seed_z\": "
             << (request.seed ^ PROJECTION_SEED_XOR) << ",\n"
             << "  \"operator_backend\": \""
             << (injective_padding
                     ? "injective-prefix-recomputed-transpose-slices"
                     : (recompute_mapping
                            ? "support-complement-slices-recomputed-map"
                            : "support-complement-slices"))
             << "\",\n"
             << "  \"start_step\": " << start_step << ",\n"
             << "  \"term_count\": " << step_count << ",\n"
             << "  \"end_step\": " << end_step << ",\n"
             << "  \"words\": " << request.words << ",\n"
             << "  \"block_bits\": " << request.words * 64 << ",\n"
             << "  \"projection_chunks\": " << projection_chunks << ",\n"
             << "  \"allocated_device_bytes\": " << allocated_bytes << ",\n"
             << "  \"free_device_bytes_before\": " << free_bytes << ",\n"
             << "  \"total_device_bytes\": " << total_bytes << ",\n"
             << "  \"preprocessing_seconds\": " << preprocessing_seconds << ",\n"
             << "  \"operator_seconds_total\": " << operator_seconds << ",\n"
             << "  \"projection_seconds_total\": " << projection_seconds << ",\n"
             << "  \"wall_seconds\": " << wall_seconds << ",\n"
             << "  \"sequence_fnv1a64_le\": \""
             << hex_u64(sequence_checksum) << "\",\n"
             << "  \"state_fnv1a64_le\": \""
             << hex_u64(state_checksum) << "\",\n"
             << "  \"sequence_file\": \"sequence.bin\",\n"
             << "  \"sequence_bytes\": "
             << std::filesystem::file_size(sequence_path) << ",\n"
             << "  \"state_file\": \"state.bin\",\n"
             << "  \"state_bytes\": "
             << std::filesystem::file_size(state_path) << "\n"
             << "}\n";
    const std::filesystem::path result_path = output_directory / "result.json";
    write_new(result_path, manifest.str());
    fsync_path(result_path, false);
    fsync_path(output_directory, true);
}

u64 self_test_k68_degree7_slice_mapping() {
    Request request;
    request.schema = EXTENDED_REQUEST_SCHEMA;
    request.k = 68;
    request.degree = 7;
    request.words = 1;
    request.repetitions = 1;
    request.emit_output = 0;
    request.seed = 0x4b36384445475237ULL;
    request.public_sha256 = "self-test-k68-degree7-mapping";
    Point point;
    for (const int coordinate : {0, 12, 31, 63}) {
        point.support |= u64{1} << coordinate;
    }
    for (const int coordinate : {64, 66, 67}) {
        point.support_high |= u64{1} << (coordinate - 64);
    }
    point.levels = {1};
    request.points.push_back(point);
    request.expected_columns = HOST_CHOOSE.get(request.k, request.degree);
    request.expected_rows = HOST_CHOOSE.get(request.k, 1);
    request.expected_nonzeros = point_nonzeros(request, request.points.front());
    finish_layout(request);

    const SlicePlan plan = build_slice_plan(request);
    const std::vector<unsigned char> positions = point_position_tables(request);
    DeviceBuffers buffers;
    CUDA_CHECK(cudaMalloc(
        &buffers.mapping,
        static_cast<std::size_t>(plan.mapping_entries)
            * sizeof(std::uint32_t)));
    CUDA_CHECK(cudaMalloc(&buffers.positions, positions.size()));
    CUDA_CHECK(cudaMemcpy(
        buffers.positions,
        positions.data(),
        positions.size(),
        cudaMemcpyHostToDevice));
    launch_fill_slice_mapping(request, plan, buffers);
    std::vector<std::uint32_t> observed(
        static_cast<std::size_t>(plan.mapping_entries));
    CUDA_CHECK(cudaMemcpy(
        observed.data(),
        buffers.mapping,
        observed.size() * sizeof(std::uint32_t),
        cudaMemcpyDeviceToHost));

    const int support_weight = point_weight(request.points.front());
    const int complement_weight = request.k - support_weight;
    const unsigned char *support_positions = positions.data();
    const unsigned char *complement_positions =
        support_positions + MAX_K;
    u64 compared = 0;
    for (const SliceClass &slice_class : plan.points.front().classes) {
        const u64 mapping_count = slice_class.slice_count
            * slice_class.columns_per_slice;
        for (u64 entry = 0; entry < mapping_count; ++entry) {
            const u64 slice = entry / slice_class.columns_per_slice;
            const u64 local_column =
                entry - slice * slice_class.columns_per_slice;
            const u64 local_a = colex_unrank_host(
                slice,
                slice_class.complement_size,
                complement_weight);
            const u64 local_b = colex_unrank_host(
                local_column,
                slice_class.support_column_size,
                support_weight);
            const u64 expected = colex_rank_merged_local_masks_host(
                local_a,
                complement_positions,
                local_b,
                support_positions);
            if (observed[slice_class.mapping_offset + entry] != expected) {
                std::ostringstream message;
                message << "k=68 degree-seven slice mapping mismatch at entry "
                        << slice_class.mapping_offset + entry;
                throw std::runtime_error(message.str());
            }
            ++compared;
        }
    }
    if (compared != 428) {
        throw std::runtime_error(
            "k=68 degree-seven slice mapping fixture cardinality drifted");
    }
    return compared;
}

Request self_test_request(int words) {
    Request request;
    request.schema = REQUEST_SCHEMA;
    request.k = 9;
    request.degree = 4;
    request.words = words;
    request.repetitions = 1;
    request.emit_output = 1;
    request.seed = 0x4355444154455354ULL;
    request.public_sha256 = "self-test";
    request.points = {
        Point{0b111101011ULL, 0, {0, 1, 3}, {}, 0, 0},
        Point{0b010111101ULL, 0, {1, 2, 4}, {}, 0, 0},
        Point{0b101011010ULL, 0, {0, 2, 3}, {}, 0, 0},
    };
    request.expected_columns = HOST_CHOOSE.get(request.k, request.degree);
    request.expected_rows = 0;
    request.expected_nonzeros = 0;
    for (Point &point : request.points) {
        point.offsets.clear();
        point.rows = 0;
        for (const int level : point.levels) {
            point.offsets.push_back(point.rows);
            point.rows += HOST_CHOOSE.get(request.k, level);
        }
        point.nonzeros = point_nonzeros(request, point);
        request.expected_rows += point.rows;
        request.expected_nonzeros += point.nonzeros;
    }
    finish_layout(request);
    return request;
}

void run_self_test() {
    CUDA_CHECK(cudaMemcpyToSymbol(
        DEVICE_CHOOSE, HOST_CHOOSE.values.data(), sizeof(HOST_CHOOSE.values)));
    cudaDeviceProp properties{};
    device_name(properties);
    int direct_compared = 0;
    int slice_compared = 0;
    int recomputed_slice_compared = 0;
    int left_recomputed_slice_compared = 0;
    int projection_compared = 0;
    const u64 extended_mapping_compared =
        self_test_k68_degree7_slice_mapping();
    for (const int words : {1, 3, 8, 16}) {
        Request request = self_test_request(words);
        const std::vector<u64> input = random_words(
            static_cast<std::size_t>(request.expected_columns) * words,
            request.seed ^ static_cast<u64>(words));
        const std::vector<u64> expected = cpu_apply_b(request, input);
        const RunResult direct = gpu_apply_b_direct(request, input);
        const RunResult sliced = gpu_apply_b_slice(request, input);
        const RunResult recomputed = gpu_apply_b_slice_recompute(
            request, input);
        if (direct.output != expected) {
            std::size_t index = 0;
            while (index < expected.size() && expected[index] == direct.output[index]) {
                ++index;
            }
            std::ostringstream message;
            message << "CPU/direct-CUDA B mismatch at words=" << words
                    << " index=" << index;
            throw std::runtime_error(message.str());
        }
        ++direct_compared;
        if (sliced.output != expected) {
            std::size_t index = 0;
            while (index < expected.size() && expected[index] == sliced.output[index]) {
                ++index;
            }
            std::ostringstream message;
            message << "CPU/slice-CUDA B mismatch at words=" << words
                    << " index=" << index;
            throw std::runtime_error(message.str());
        }
        ++slice_compared;
        if (recomputed.output != expected) {
            std::size_t index = 0;
            while (index < expected.size()
                && expected[index] == recomputed.output[index]) {
                ++index;
            }
            std::ostringstream message;
            message << "CPU/recomputed-slice-CUDA B mismatch at words="
                    << words << " index=" << index;
            throw std::runtime_error(message.str());
        }
        ++recomputed_slice_compared;
        const std::vector<u64> left_input = random_words(
            static_cast<std::size_t>(request.expected_rows) * words,
            (request.seed ^ static_cast<u64>(words))
                ^ 0x4c4546544d4d5455ULL);
        std::vector<u64> expected_transpose;
        const std::vector<u64> expected_left =
            cpu_apply_left_slice_order(
                request, left_input, &expected_transpose);
        const RunResult observed_left =
            gpu_apply_left_slice_recompute(request, left_input);
        if (observed_left.transpose_output != expected_transpose) {
            std::size_t index = 0;
            while (index < expected_transpose.size()
                && expected_transpose[index]
                    == observed_left.transpose_output[index]) {
                ++index;
            }
            std::ostringstream message;
            message << "CPU/recomputed-slice CUDA M^T mismatch at words="
                    << words << " index=" << index;
            throw std::runtime_error(message.str());
        }
        if (observed_left.output != expected_left) {
            std::size_t index = 0;
            while (index < expected_left.size()
                && expected_left[index] == observed_left.output[index]) {
                ++index;
            }
            std::ostringstream message;
            message << "CPU/recomputed-slice CUDA M M^T mismatch at words="
                    << words << " index=" << index;
            throw std::runtime_error(message.str());
        }
        ++left_recomputed_slice_compared;
        const std::vector<u64> z = random_words(
            static_cast<std::size_t>(request.expected_columns) * words,
            (request.seed ^ static_cast<u64>(words)) ^ PROJECTION_SEED_XOR);
        const std::vector<u64> expected_projection = cpu_project(
            request, z, input);
        const ProjectionRunResult observed_projection = gpu_project(
            request, z, input);
        if (observed_projection.output != expected_projection) {
            std::size_t index = 0;
            while (index < expected_projection.size()
                && expected_projection[index]
                    == observed_projection.output[index]) {
                ++index;
            }
            std::ostringstream message;
            message << "CPU/CUDA projection mismatch at words=" << words
                    << " index=" << index;
            throw std::runtime_error(message.str());
        }
        ++projection_compared;
    }
    std::cout << "{\n"
              << "  \"schema\": \"mceliecex-holdout-cuda-self-test-v4\",\n"
              << "  \"gpu_name\": \"" << json_escape(properties.name) << "\",\n"
              << "  \"compute_capability\": \"" << properties.major << "." << properties.minor << "\",\n"
              << "  \"compiled_sm\": " << MCELIECEX_CUDA_TARGET_ARCH << ",\n"
              << "  \"widths_words\": [1,3,8,16],\n"
              << "  \"cpu_direct_cuda_exact_equal_cases\": " << direct_compared << ",\n"
              << "  \"cpu_slice_cuda_exact_equal_cases\": " << slice_compared << ",\n"
              << "  \"cpu_recomputed_slice_cuda_exact_equal_cases\": "
              << recomputed_slice_compared << ",\n"
              << "  \"cpu_recomputed_slice_cuda_left_exact_equal_cases\": "
              << left_recomputed_slice_compared << ",\n"
              << "  \"cpu_cuda_full_projection_exact_equal_cases\": "
              << projection_compared << ",\n"
              << "  \"two_word_k68_degree7_slice_mapping_entries\": "
              << extended_mapping_compared << ",\n"
              << "  \"pass\": true\n"
              << "}\n";
}

void run_literal_witness_request(
    const std::filesystem::path &request_path,
    const std::filesystem::path &output_directory) {
    if (std::filesystem::exists(output_directory)) {
        throw std::runtime_error(
            "refusing existing literal-witness output directory: "
            + output_directory.string());
    }
    const std::filesystem::path temporary =
        output_directory.string() + ".tmp";
    if (std::filesystem::exists(temporary)) {
        throw std::runtime_error(
            "literal-witness temporary directory already exists: "
            + temporary.string());
    }
    WitnessRequest request = read_witness_request(request_path);
    CUDA_CHECK(cudaMemcpyToSymbol(
        DEVICE_CHOOSE, HOST_CHOOSE.values.data(), sizeof(HOST_CHOOSE.values)));
    cudaDeviceProp properties{};
    device_name(properties);
    WitnessRunResult result = gpu_literal_witness_transpose(request);
    for (const u64 value : result.output) {
        if ((value >> 24) != 0) {
            throw std::runtime_error(
                "literal-witness domain escaped its three GF(256) lanes");
        }
    }
    if (request.emit_coefficients) {
        if (result.emitted_coefficients.size()
                != static_cast<std::size_t>(request.layout.expected_rows)) {
            throw std::runtime_error(
                "emitted literal coefficient count differs from the layout");
        }
        for (const u64 value : result.emitted_coefficients) {
            if ((value >> 24) != 0) {
                throw std::runtime_error(
                    "literal coefficient escaped its three GF(256) lanes");
            }
        }
    }

    if (!std::filesystem::create_directory(temporary)) {
        throw std::runtime_error(
            "cannot create literal-witness temporary directory");
    }
    const std::filesystem::path domain_path = temporary / "domain.bin";
    write_literal_domain(domain_path, request, result.output);
    fsync_path(domain_path, false);
    const std::uintmax_t domain_bytes =
        std::filesystem::file_size(domain_path);
    const std::filesystem::path result_path = temporary / "result.json";
    write_new(
        result_path,
        literal_witness_response_json(
            request, result, properties, domain_bytes));
    fsync_path(result_path, false);
    fsync_path(temporary, true);
    publish_temporary_new(temporary, output_directory);
    std::filesystem::path parent = output_directory.parent_path();
    if (parent.empty()) {
        parent = ".";
    }
    fsync_path(parent, true);
}

void run_request(
    const std::filesystem::path &request_path,
    const std::filesystem::path &response_path,
    const std::string &backend) {
    Request request = read_request(request_path);
    CUDA_CHECK(cudaMemcpyToSymbol(
        DEVICE_CHOOSE, HOST_CHOOSE.values.data(), sizeof(HOST_CHOOSE.values)));
    cudaDeviceProp properties{};
    device_name(properties);
    if (backend == "run-slice-recompute") {
        const SlicePlan plan = build_slice_plan(request, false);
        if (plan.max_range_rows > std::numeric_limits<std::size_t>::max()
                / (static_cast<std::size_t>(request.words) * sizeof(u64))
            || request.expected_columns > std::numeric_limits<std::size_t>::max()
                / (static_cast<std::size_t>(request.words) * sizeof(u64))) {
            throw std::runtime_error(
                "recomputed slice preflight exceeds host size_t");
        }
        const std::size_t panel_bytes = static_cast<std::size_t>(
            request.expected_columns) * request.words * sizeof(u64);
        const std::size_t range_bytes = static_cast<std::size_t>(
            plan.max_range_rows) * request.words * sizeof(u64);
        const std::size_t position_bytes =
            request.points.size() * 2 * MAX_K;
        const std::size_t required = 2 * panel_bytes + range_bytes
            + position_bytes;
        std::size_t free_bytes = 0;
        std::size_t total_bytes = 0;
        CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
        constexpr std::size_t reserve = std::size_t{256} << 20;
        if (required > free_bytes
            || free_bytes - required < reserve) {
            std::ostringstream message;
            message << "recomputed slice preflight requires " << required
                    << " device bytes plus a " << reserve
                    << "-byte reserve, but only " << free_bytes
                    << " bytes are free";
            throw std::runtime_error(message.str());
        }
    }
    const std::vector<u64> input = random_words(
        static_cast<std::size_t>(request.expected_columns) * request.words,
        request.seed);
    const RunResult run = backend == "run-slice"
        ? gpu_apply_b_slice(request, input)
        : (backend == "run-slice-recompute"
               ? gpu_apply_b_slice_recompute(request, input)
               : gpu_apply_b_direct(request, input));
    write_new(response_path, response_json(request, run, properties));
}

void run_left_request(
    const std::filesystem::path &request_path,
    const std::filesystem::path &response_path) {
    Request request = read_request(request_path);
    CUDA_CHECK(cudaMemcpyToSymbol(
        DEVICE_CHOOSE, HOST_CHOOSE.values.data(), sizeof(HOST_CHOOSE.values)));
    cudaDeviceProp properties{};
    device_name(properties);
    const SlicePlan plan = build_slice_plan(request, false);
    if (plan.total_range_rows > std::numeric_limits<std::size_t>::max()
            / (static_cast<std::size_t>(request.words) * sizeof(u64))
        || request.expected_columns > std::numeric_limits<std::size_t>::max()
            / (static_cast<std::size_t>(request.words) * sizeof(u64))) {
        throw std::runtime_error("left request preflight exceeds host size_t");
    }
    const std::size_t range_words = static_cast<std::size_t>(
        plan.total_range_rows) * request.words;
    const std::size_t range_bytes = range_words * sizeof(u64);
    const std::size_t domain_bytes = static_cast<std::size_t>(
        request.expected_columns) * request.words * sizeof(u64);
    const std::size_t position_bytes = request.points.size() * 2 * MAX_K;
    const std::size_t required = 2 * range_bytes + domain_bytes
        + position_bytes;
    std::size_t free_bytes = 0;
    std::size_t total_bytes = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
    (void)total_bytes;
    constexpr std::size_t reserve = std::size_t{256} << 20;
    if (required > free_bytes || free_bytes - required < reserve) {
        std::ostringstream message;
        message << "left recomputed-slice preflight requires " << required
                << " device bytes plus a " << reserve
                << "-byte reserve, but only " << free_bytes
                << " bytes are free";
        throw std::runtime_error(message.str());
    }
    if (request.emit_output && range_words > 65536) {
        throw std::runtime_error(
            "emitted left output is restricted to 65536 u64 words");
    }
    const std::vector<u64> input = random_words(range_words, request.seed);
    const RunResult run = gpu_apply_left_slice_recompute(request, input);
    write_new(
        response_path,
        left_response_json(request, run, properties));
}

void run_projection_request(
    const std::filesystem::path &request_path,
    const std::filesystem::path &response_path) {
    Request request = read_request(request_path);
    cudaDeviceProp properties{};
    device_name(properties);
    const std::size_t panel_words = static_cast<std::size_t>(
        request.expected_columns) * request.words;
    const std::vector<u64> v = random_words(panel_words, request.seed);
    const std::vector<u64> z = random_words(
        panel_words, request.seed ^ PROJECTION_SEED_XOR);
    const ProjectionRunResult run = gpu_project(request, z, v);
    write_new(
        response_path,
        projection_response_json(request, run, properties));
}

void run_column_degree_census_request(
    const std::filesystem::path &request_path,
    const std::filesystem::path &response_path) {
    Request request = read_request(request_path);
    CUDA_CHECK(cudaMemcpyToSymbol(
        DEVICE_CHOOSE, HOST_CHOOSE.values.data(), sizeof(HOST_CHOOSE.values)));
    cudaDeviceProp properties{};
    device_name(properties);
    const ColumnDegreeCensusResult result = gpu_column_degree_census(request);
    write_new(
        response_path,
        column_degree_census_json(request, result, properties));
}

}  // namespace

int main(int argc, char **argv) {
    try {
        if (argc == 2 && std::string(argv[1]) == "self-test") {
            run_self_test();
            return 0;
        }
        if (argc == 4
            && (std::string(argv[1]) == "run"
                || std::string(argv[1]) == "run-slice"
                || std::string(argv[1]) == "run-slice-recompute")) {
            run_request(argv[2], argv[3], argv[1]);
            return 0;
        }
        if (argc == 4
            && std::string(argv[1]) == "run-left-slice-recompute") {
            run_left_request(argv[2], argv[3]);
            return 0;
        }
        if (argc == 4 && std::string(argv[1]) == "project") {
            run_projection_request(argv[2], argv[3]);
            return 0;
        }
        if (argc == 4 && std::string(argv[1]) == "column-degree-census") {
            run_column_degree_census_request(argv[2], argv[3]);
            return 0;
        }
        if (argc == 4 && std::string(argv[1]) == "literal-witness") {
            run_literal_witness_request(argv[2], argv[3]);
            return 0;
        }
        if (argc == 6
            && (std::string(argv[1]) == "krylov-segment"
                || std::string(argv[1])
                    == "krylov-segment-recompute"
                || std::string(argv[1])
                    == "krylov-injective-segment-recompute")) {
            std::size_t parsed = 0;
            const std::string step_text = argv[4];
            const u64 steps = std::stoull(step_text, &parsed, 10);
            if (parsed != step_text.size()) {
                throw std::runtime_error("Krylov step count is not an integer");
            }
            run_krylov_segment(
                argv[2],
                argv[3],
                steps,
                argv[5],
                std::string(argv[1]) != "krylov-segment",
                std::string(argv[1])
                    == "krylov-injective-segment-recompute");
            return 0;
        }
        std::cerr << "usage: " << argv[0]
                  << " self-test | (run|run-slice|run-slice-recompute|run-left-slice-recompute|project|column-degree-census)"
                  << " REQUEST.txt RESPONSE.json"
                  << " | literal-witness REQUEST.txt NEW_OUTPUT_DIR"
                  << " | (krylov-segment|krylov-segment-recompute|krylov-injective-segment-recompute)"
                  << " REQUEST.txt (STATE.bin|-) STEPS OUTPUT_DIR\n";
        return 2;
    } catch (const std::exception &error) {
        std::cerr << "holdout CUDA worker refused: " << error.what() << "\n";
        return 1;
    }
}
