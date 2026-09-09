// Sharded right-generator reconstruction for the corrected TII-254 operator
//
//     A = R E J : F2^15,527,170 -> F2^15,527,170.
//
// Each shard starts from an authenticated production Krylov checkpoint and
// emits only its interval contribution to
//
//     Q = sum_j A^j Y F_j.
//
// A separate combine invocation XORs an exact interval partition and replays
// A Q.  Both outputs remain unpromoted until a separately implemented CPU
// path also verifies E(JQ)=0.

#define MCELIECEX_TII254_LSTAR_REJ_KRYLOV_EMBEDDED 1
#include "tii254_lstar_rej_krylov_worker.cu"
#undef MCELIECEX_TII254_LSTAR_REJ_KRYLOV_EMBEDDED

#include <sys/stat.h>

namespace mceliecex_tii254_lstar_rej_reconstruction {

namespace base = mceliecex_tii254_lstar_rej;
namespace anchor = mceliecex_tii254_lstar_anchor;
namespace krylov = mceliecex_tii254_lstar_rej_krylov;
#ifdef MCELIECEX_TII254_LSTAR_REJ_ANCHOR5
namespace anchor_request_policy = mceliecex_tii254_lstar_anchor_v2;
#endif

constexpr const char *REQUEST_SCHEMA =
    "mceliecex-tii254-d6-lstar-rej-reconstruction-request-v2";
constexpr const char *SLICE_SCHEMA =
    "mceliecex-tii254-d6-lstar-rej-relation-slice-v1";
constexpr const char *CONTRIBUTION_SCHEMA =
    "mceliecex-tii254-d6-lstar-rej-reconstruction-contribution-v1";
[[maybe_unused]] constexpr const char *SHARD_SCHEMA =
    "mceliecex-tii254-d6-lstar-rej-reconstruction-shard-result-v1";
[[maybe_unused]] constexpr const char *COMBINE_SCHEMA =
    "mceliecex-tii254-d6-lstar-rej-reconstruction-combine-result-v1";
#ifdef MCELIECEX_TII254_LSTAR_REJ_ANCHOR5
constexpr u64 REQUIRED_F0_RANK = 0;
constexpr const char *EXPECTED_PM_TERMINAL =
    "tii254_d6_lstar_anchor5_rej_pmbasis_complete";
#else
constexpr u64 REQUIRED_F0_RANK = 8;
constexpr const char *EXPECTED_PM_TERMINAL_PASS =
    "tii254_d6_lstar_rej_pmbasis_relations_F0_gate_pass";
constexpr const char *EXPECTED_PM_TERMINAL_REFUSAL =
    "tii254_d6_lstar_rej_pmbasis_relations_F0_gate_refusal";
#endif
constexpr u64 MATRIX_WORDS = 512 * 8;
constexpr std::size_t PANEL_WORDS =
    static_cast<std::size_t>(base::SOURCE_ROWS) * krylov::WORDS;
constexpr std::size_t PANEL_BYTES = PANEL_WORDS * sizeof(u64);
constexpr std::size_t DEVICE_RESERVE = std::size_t{256} << 20;

template <class Value>
void read_field(std::istream &stream, const char *expected, Value &value) {
    std::string key;
    if (!(stream >> key) || key != expected || !(stream >> value)) {
        throw std::runtime_error(
            std::string("expected Lstar reconstruction key ") + expected);
    }
}

struct ReconstructionRequest {
    std::string binding_sha256;
    std::string production_package_identity;
    std::string pmbasis_validation_sha256;
    std::string relation_extraction_sha256;
    std::string historical_pm_terminal;
    u64 emitted_relation_f0_rank = 0;
    u64 first_anchor_required_f0_rank = 0;
    std::string relations_sha256;
    std::string relations_fnv1a64_le;
    u64 coefficient_matrix_count = 0;
};

ReconstructionRequest read_reconstruction_request(
    const std::filesystem::path &path) {
    std::ifstream stream(path);
    if (!stream) throw std::runtime_error("cannot open Lstar reconstruction request");
    std::string schema;
    std::getline(stream, schema);
    if (schema != REQUEST_SCHEMA) {
        throw std::runtime_error("Lstar reconstruction request schema differs");
    }
    ReconstructionRequest request;
    std::string request_identity;
    std::string operator_name;
    std::string origin;
    std::string reconstruction;
    std::string layout;
    std::string acceptance;
    u64 source_rows = 0;
    int right_bits = 0;
    u64 matrix_words = 0;
    read_field(stream, "binding_sha256", request.binding_sha256);
    read_field(stream, "request_identity", request_identity);
    read_field(
        stream,
        "production_package_identity",
        request.production_package_identity);
    read_field(
        stream, "pmbasis_validation_sha256", request.pmbasis_validation_sha256);
    read_field(
        stream, "relation_extraction_sha256", request.relation_extraction_sha256);
    read_field(
        stream, "historical_pm_terminal", request.historical_pm_terminal);
    read_field(
        stream, "emitted_relation_F0_rank", request.emitted_relation_f0_rank);
    read_field(
        stream, "first_anchor_required_F0_rank",
        request.first_anchor_required_f0_rank);
    read_field(stream, "relations_sha256", request.relations_sha256);
    read_field(
        stream, "relations_fnv1a64_le", request.relations_fnv1a64_le);
    read_field(
        stream, "coefficient_matrix_count", request.coefficient_matrix_count);
    read_field(stream, "coefficient_matrix_words", matrix_words);
    read_field(stream, "source_rows", source_rows);
    read_field(stream, "right_bits", right_bits);
    read_field(stream, "operator", operator_name);
    read_field(stream, "sequence_origin", origin);
    read_field(stream, "reconstruction", reconstruction);
    read_field(stream, "relation_layout", layout);
    read_field(stream, "positive_acceptance", acceptance);
    std::string trailing;
    if (stream >> trailing) {
        throw std::runtime_error("Lstar reconstruction request has trailing tokens");
    }
    if (request.binding_sha256.size() != 64
        || request_identity != krylov::REQUEST_IDENTITY
        || request.production_package_identity
            != "b69c0e17fc66bcb2414d09f6e4251d5587df3ed52483b182a77139699af9e698"
        || request.pmbasis_validation_sha256.size() != 64
        || request.relation_extraction_sha256.size() != 64
        || request.emitted_relation_f0_rank < REQUIRED_F0_RANK
        || request.emitted_relation_f0_rank > krylov::WIDTH
        || request.first_anchor_required_f0_rank != REQUIRED_F0_RANK
#ifdef MCELIECEX_TII254_LSTAR_REJ_ANCHOR5
        || request.historical_pm_terminal != EXPECTED_PM_TERMINAL
#else
        || request.historical_pm_terminal
            != (request.emitted_relation_f0_rank >= 16
                    ? EXPECTED_PM_TERMINAL_PASS
                    : EXPECTED_PM_TERMINAL_REFUSAL)
#endif
        || request.relations_sha256.size() != 64
        || request.relations_fnv1a64_le.size() != 16
        || !request.coefficient_matrix_count
        || request.coefficient_matrix_count > krylov::TOTAL_STEPS
        || matrix_words != MATRIX_WORDS
        || source_rows != base::SOURCE_ROWS
        || right_bits != krylov::WIDTH
        || operator_name != "A=R_E_J"
        || origin != "Y_then_A_power"
        || reconstruction != "Q=sum_j_A^j_Y_F_j"
        || layout != "coefficient_major_512_by_512_u64_le"
        || acceptance != "A_Q_zero_and_independent_literal_E_J_Q_zero") {
        throw std::runtime_error("Lstar reconstruction request binding differs");
    }
    return request;
}

struct RelationSlice {
    u64 start = 0;
    u64 end = 0;
    std::string payload_sha256;
    std::string payload_fnv1a64_le;
    std::string input_state_sha256;
    std::string input_state_inventory_sha256;
};

RelationSlice read_slice_manifest(
    const std::filesystem::path &path,
    const ReconstructionRequest &request) {
    std::ifstream stream(path);
    if (!stream) throw std::runtime_error("cannot open Lstar relation slice manifest");
    std::string schema;
    std::getline(stream, schema);
    if (schema != SLICE_SCHEMA) {
        throw std::runtime_error("Lstar relation slice schema differs");
    }
    RelationSlice slice;
    std::string request_binding;
    std::string full_relations_sha256;
    u64 matrix_words = 0;
    u64 payload_words = 0;
    read_field(stream, "request_binding_sha256", request_binding);
    read_field(stream, "full_relations_sha256", full_relations_sha256);
    read_field(stream, "start_coefficient", slice.start);
    read_field(stream, "end_coefficient", slice.end);
    read_field(stream, "coefficient_matrix_words", matrix_words);
    read_field(stream, "payload_words", payload_words);
    read_field(stream, "payload_sha256", slice.payload_sha256);
    read_field(stream, "payload_fnv1a64_le", slice.payload_fnv1a64_le);
    read_field(stream, "input_state_sha256", slice.input_state_sha256);
    read_field(
        stream,
        "input_state_inventory_sha256",
        slice.input_state_inventory_sha256);
    std::string trailing;
    if (stream >> trailing) {
        throw std::runtime_error("Lstar relation slice has trailing tokens");
    }
    if (request_binding != request.binding_sha256
        || full_relations_sha256 != request.relations_sha256
        || slice.start >= slice.end
        || slice.end > request.coefficient_matrix_count
        || matrix_words != MATRIX_WORDS
        || payload_words != (slice.end - slice.start) * MATRIX_WORDS
        || slice.payload_sha256.size() != 64
        || slice.payload_fnv1a64_le.size() != 16
        || (slice.start == 0
                ? (slice.input_state_sha256 != "-"
                    || slice.input_state_inventory_sha256 != "-")
                : (slice.input_state_sha256.size() != 64
                    || slice.input_state_inventory_sha256.size() != 64))) {
        throw std::runtime_error("Lstar relation slice binding differs");
    }
    return slice;
}

std::vector<u64> read_words_exact(
    const std::filesystem::path &path, std::size_t words) {
    if (words > std::numeric_limits<std::size_t>::max() / sizeof(u64)
        || std::filesystem::file_size(path) != words * sizeof(u64)) {
        throw std::runtime_error("Lstar packed payload size differs");
    }
    std::vector<u64> values(words);
    std::ifstream stream(path, std::ios::binary);
    stream.read(
        reinterpret_cast<char *>(values.data()),
        static_cast<std::streamsize>(values.size() * sizeof(u64)));
    if (!stream || stream.peek() != std::char_traits<char>::eof()) {
        throw std::runtime_error("Lstar packed payload read differs");
    }
    return values;
}

void write_words(const std::filesystem::path &path, const std::vector<u64> &words) {
    std::ofstream stream(path, std::ios::binary | std::ios::out | std::ios::trunc);
    if (!stream) throw std::runtime_error("cannot create Lstar packed temporary");
    stream.write(
        reinterpret_cast<const char *>(words.data()),
        static_cast<std::streamsize>(words.size() * sizeof(u64)));
    stream.flush();
    if (!stream) throw std::runtime_error("cannot write Lstar packed temporary");
}

std::vector<u64> relation_nibble_table(const u64 *coefficient) {
    constexpr std::size_t table_words =
        krylov::WORDS * 16U * 16U * krylov::WORDS;
    std::vector<u64> table(table_words);
    for (int input_word = 0; input_word < krylov::WORDS; ++input_word) {
        for (int nibble = 0; nibble < 16; ++nibble) {
            for (int value = 0; value < 16; ++value) {
                u64 *destination = table.data()
                    + (((static_cast<std::size_t>(input_word) * 16U + nibble)
                        * 16U + value) * krylov::WORDS);
                for (int bit = 0; bit < 4; ++bit) {
                    if (!(value & (1 << bit))) continue;
                    const int row = input_word * 64 + nibble * 4 + bit;
                    for (int word = 0; word < krylov::WORDS; ++word) {
                        destination[word] ^=
                            coefficient[static_cast<std::size_t>(row)
                                * krylov::WORDS + word];
                    }
                }
            }
        }
    }
    return table;
}

__global__ void accumulate_relation_kernel(
    const u64 *current,
    u64 rows,
    const u64 *table,
    u64 *candidates) {
    const u64 stride = static_cast<u64>(blockDim.x) * gridDim.x;
    for (u64 row = static_cast<u64>(blockIdx.x) * blockDim.x + threadIdx.x;
         row < rows; row += stride) {
        u64 output[krylov::WORDS]{};
        for (int input_word = 0; input_word < krylov::WORDS; ++input_word) {
            const u64 input = current[row * krylov::WORDS + input_word];
            for (int nibble = 0; nibble < 16; ++nibble) {
                const int value = static_cast<int>((input >> (4 * nibble)) & 15);
                const u64 *source = table
                    + (((static_cast<u64>(input_word) * 16 + nibble) * 16
                        + static_cast<u64>(value)) * krylov::WORDS);
#pragma unroll
                for (int word = 0; word < krylov::WORDS; ++word) {
                    output[word] ^= source[word];
                }
            }
        }
#pragma unroll
        for (int word = 0; word < krylov::WORDS; ++word) {
            candidates[row * krylov::WORDS + word] ^= output[word];
        }
    }
}

struct HostOperator {
    Request literal;
    anchor::Request anchor_request;
    anchor::Codec codec;
    std::vector<std::uint32_t> permutation;
    std::vector<unsigned char> diagonal;
    SlicePlan plan;
    std::vector<unsigned char> positions;
};

HostOperator load_operator(
    const std::filesystem::path &operator_path,
    const std::filesystem::path &anchor_root,
    const std::filesystem::path &permutation_path,
    const std::filesystem::path &diagonal_path) {
    HostOperator host;
    host.literal = ::read_request(operator_path);
    if (host.literal.public_sha256 != krylov::PROFILE_IDENTITY
        || host.literal.expected_columns != base::FORM_ROWS
        || host.literal.expected_rows != base::LITERAL_ROWS
        || host.literal.expected_nonzeros != 6'223'124'835ULL
        || host.literal.words != 1) {
        throw std::runtime_error("reconstruction literal operator differs");
    }
    host.literal.words = krylov::WORDS;
    const auto input = anchor_root / "input";
    host.anchor_request =
#ifdef MCELIECEX_TII254_LSTAR_REJ_ANCHOR5
        anchor_request_policy::read_request(
#else
        anchor::read_request(
#endif
            input / "request-r512.txt");
    if (host.anchor_request.source_rows != base::SOURCE_ROWS
        || host.anchor_request.literal_rows != base::LITERAL_ROWS
        || host.anchor_request.right_words != krylov::WORDS
        || host.anchor_request.codec_identity != krylov::CODEC_IDENTITY) {
        throw std::runtime_error("reconstruction anchor codec differs");
    }
    host.codec =
#ifdef MCELIECEX_TII254_LSTAR_REJ_ANCHOR5
        anchor_request_policy::load_codec(host.anchor_request, input);
#else
        anchor::load_codec(
        host.anchor_request,
        input / "complement-selector.u32le",
        input / "nonanchor-t64-positions.u32le",
        input / "common-e4-reduced.sparse",
        input / "correction.matrix",
        input / "pivot-common-sources.u32le",
        input / "free-common-sources.u32le",
        input / "retained-relations.bin");
#endif
    host.permutation = anchor::read_u32s(permutation_path, base::LITERAL_ROWS);
    std::vector<unsigned char> seen(
        static_cast<std::size_t>(base::LITERAL_ROWS), 0);
    for (const std::uint32_t row : host.permutation) {
        if (row >= base::LITERAL_ROWS || seen[row]++) {
            throw std::runtime_error("reconstruction literal permutation differs");
        }
    }
    host.diagonal = base::read_diagonal(diagonal_path);
    host.plan = build_slice_plan(host.literal, false);
    host.positions = point_position_tables(host.literal);
    CUDA_CHECK(cudaMemcpyToSymbol(
        DEVICE_CHOOSE, HOST_CHOOSE.values.data(), sizeof(HOST_CHOOSE.values)));
    return host;
}

struct DeviceOperator {
    HostOperator &host;
    anchor::DeviceCodec codec;
    std::uint32_t *permutation = nullptr;
    unsigned char *positions = nullptr;
    u64 *literal_original = nullptr;
    u64 *literal_slice = nullptr;
    u64 *form = nullptr;
    u64 *common = nullptr;
    base::DeviceToeplitz toeplitz;

    DeviceOperator(HostOperator &host_value, int batch)
        : host(host_value), codec(anchor::upload(host.codec)),
          permutation(anchor::copy_device(host.permutation)),
          positions(anchor::copy_device(host.positions)),
          toeplitz(host.diagonal, krylov::WIDTH, batch) {
        CUDA_CHECK(cudaMalloc(
            &literal_original,
            static_cast<std::size_t>(base::LITERAL_ROWS)
                * krylov::WORDS * sizeof(u64)));
        CUDA_CHECK(cudaMalloc(
            &literal_slice,
            static_cast<std::size_t>(base::LITERAL_ROWS)
                * krylov::WORDS * sizeof(u64)));
        CUDA_CHECK(cudaMalloc(
            &form,
            static_cast<std::size_t>(base::FORM_ROWS)
                * krylov::WORDS * sizeof(u64)));
        CUDA_CHECK(cudaMalloc(
            &common,
            static_cast<std::size_t>(host.anchor_request.common_rows)
                * krylov::WORDS * sizeof(u64)));
    }

    ~DeviceOperator() {
        cudaFree(common);
        cudaFree(form);
        cudaFree(literal_slice);
        cudaFree(literal_original);
        cudaFree(positions);
        cudaFree(permutation);
        codec.release();
    }

    base::ToeplitzTiming apply(const u64 *input, u64 *output) {
        base::launch_j_forward(
            host.anchor_request, host.codec, codec,
            input, literal_original, common);
        base::scatter_original_to_slice_kernel<<<
            launch_blocks(
                static_cast<u64>(base::LITERAL_ROWS) * krylov::WORDS),
            THREADS>>>(
                literal_original, permutation, base::LITERAL_ROWS,
                krylov::WORDS, literal_slice);
        CUDA_CHECK(cudaGetLastError());
        base::launch_e_forward(
            host.literal, host.plan, positions, literal_slice, form);
        return toeplitz.apply(false, form, output);
    }
};

unsigned int panel_rank(const std::vector<u64> &panel) {
    std::array<std::array<u64, krylov::WORDS>, krylov::WIDTH> pivots{};
    std::array<unsigned char, krylov::WIDTH> present{};
    unsigned int rank = 0;
    for (u64 row = 0; row < base::SOURCE_ROWS; ++row) {
        std::array<u64, krylov::WORDS> value{};
        std::copy_n(
            panel.data() + static_cast<std::size_t>(row) * krylov::WORDS,
            krylov::WORDS,
            value.begin());
        for (int bit = krylov::WIDTH - 1; bit >= 0; --bit) {
            if (!(value[bit / 64] & (UINT64_C(1) << (bit % 64)))) continue;
            if (present[bit]) {
                for (int word = 0; word < krylov::WORDS; ++word) {
                    value[word] ^= pivots[bit][word];
                }
            } else {
                pivots[bit] = value;
                present[bit] = 1;
                ++rank;
                break;
            }
        }
        if (rank == krylov::WIDTH) break;
    }
    return rank;
}

struct ResidualStats {
    u64 nonzero_words = 0;
    u64 hamming_weight = 0;
    std::string fnv1a64_le;
};

ResidualStats residual_stats(const std::vector<u64> &values) {
    ResidualStats answer;
    for (const u64 value : values) {
        answer.nonzero_words += value != 0;
        answer.hamming_weight += static_cast<u64>(__builtin_popcountll(value));
    }
    answer.fnv1a64_le = hex_u64(checksum_words(values));
    return answer;
}

void run_accumulation_self_test() {
    constexpr u64 rows = 37;
    std::vector<u64> current(static_cast<std::size_t>(rows) * krylov::WORDS);
    std::vector<u64> coefficient(MATRIX_WORDS);
    std::vector<u64> initial(static_cast<std::size_t>(rows) * krylov::WORDS);
    for (std::size_t index = 0; index < current.size(); ++index) {
        current[index] = base::splitmix_word(
            UINT64_C(0x6a09e667f3bcc909), static_cast<u64>(index));
        initial[index] = base::splitmix_word(
            UINT64_C(0xbb67ae8584caa73b), static_cast<u64>(index));
    }
    for (std::size_t index = 0; index < coefficient.size(); ++index) {
        coefficient[index] = base::splitmix_word(
            UINT64_C(0x3c6ef372fe94f82b), static_cast<u64>(index));
    }
    std::vector<u64> expected = initial;
    for (u64 row = 0; row < rows; ++row) {
        for (int input_bit = 0; input_bit < krylov::WIDTH; ++input_bit) {
            if (!(current[row * krylov::WORDS + input_bit / 64]
                  & (UINT64_C(1) << (input_bit % 64)))) {
                continue;
            }
            for (int output_word = 0; output_word < krylov::WORDS;
                 ++output_word) {
                expected[row * krylov::WORDS + output_word] ^=
                    coefficient[static_cast<std::size_t>(input_bit)
                        * krylov::WORDS + output_word];
            }
        }
    }
    const std::vector<u64> table = relation_nibble_table(coefficient.data());
    u64 *device_current = nullptr;
    u64 *device_table = nullptr;
    u64 *device_output = nullptr;
    CUDA_CHECK(cudaMalloc(&device_current, current.size() * sizeof(u64)));
    CUDA_CHECK(cudaMalloc(&device_table, table.size() * sizeof(u64)));
    CUDA_CHECK(cudaMalloc(&device_output, initial.size() * sizeof(u64)));
    CUDA_CHECK(cudaMemcpy(
        device_current, current.data(), current.size() * sizeof(u64),
        cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(
        device_table, table.data(), table.size() * sizeof(u64),
        cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(
        device_output, initial.data(), initial.size() * sizeof(u64),
        cudaMemcpyHostToDevice));
    accumulate_relation_kernel<<<launch_blocks(rows), THREADS>>>(
        device_current, rows, device_table, device_output);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<u64> observed(initial.size());
    CUDA_CHECK(cudaMemcpy(
        observed.data(), device_output, observed.size() * sizeof(u64),
        cudaMemcpyDeviceToHost));
    cudaFree(device_output);
    cudaFree(device_table);
    cudaFree(device_current);
    if (observed != expected) {
        throw std::runtime_error(
            "Lstar reconstruction accumulator self-test differs");
    }
    std::cout
        << "{\"schema\":\"mceliecex-tii254-d6-lstar-rej-reconstruction-self-test-v1\","
        << "\"terminal\":\"tii254_d6_lstar_rej_reconstruction_accumulator_self_test_pass\","
        << "\"rows\":" << rows << ','
        << "\"right_bits\":" << krylov::WIDTH << ','
        << "\"output_fnv1a64_le\":\"" << hex_u64(checksum_words(observed))
        << "\"}\n";
}

void write_json_new(
    const std::filesystem::path &temporary,
    const std::filesystem::path &target,
    const std::string &payload) {
    {
        std::ofstream stream(temporary, std::ios::out | std::ios::trunc);
        stream << payload;
        stream.flush();
        if (!stream) throw std::runtime_error("cannot write reconstruction JSON");
    }
    fsync_path(temporary, false);
    publish_temporary_new(temporary, target);
}

void write_contribution_manifest(
    const std::filesystem::path &path,
    const ReconstructionRequest &request,
    const RelationSlice &slice,
    const std::string &contribution_fnv) {
    std::ofstream stream(path, std::ios::out | std::ios::trunc);
    if (!stream) throw std::runtime_error("cannot write contribution manifest");
    stream << CONTRIBUTION_SCHEMA << '\n'
           << "request_binding_sha256 " << request.binding_sha256 << '\n'
           << "relation_slice_sha256 " << slice.payload_sha256 << '\n'
           << "input_state_sha256 " << slice.input_state_sha256 << '\n'
           << "start_coefficient " << slice.start << '\n'
           << "end_coefficient " << slice.end << '\n'
           << "source_rows " << base::SOURCE_ROWS << '\n'
           << "right_bits " << krylov::WIDTH << '\n'
           << "contribution_words " << PANEL_WORDS << '\n'
           << "contribution_fnv1a64_le " << contribution_fnv << '\n';
    stream.flush();
    if (!stream) throw std::runtime_error("cannot flush contribution manifest");
}

struct ContributionManifest {
    std::filesystem::path root;
    std::string relation_slice_sha256;
    std::string input_state_sha256;
    u64 start = 0;
    u64 end = 0;
    std::string fnv1a64_le;
};

ContributionManifest read_contribution_manifest(
    const std::filesystem::path &root,
    const ReconstructionRequest &request) {
    std::ifstream stream(root / "contribution.txt");
    if (!stream) throw std::runtime_error("cannot open contribution manifest");
    std::string schema;
    std::getline(stream, schema);
    if (schema != CONTRIBUTION_SCHEMA) {
        throw std::runtime_error("contribution manifest schema differs");
    }
    ContributionManifest answer;
    answer.root = root;
    std::string request_binding;
    u64 source_rows = 0;
    int right_bits = 0;
    u64 contribution_words = 0;
    read_field(stream, "request_binding_sha256", request_binding);
    read_field(
        stream, "relation_slice_sha256", answer.relation_slice_sha256);
    read_field(stream, "input_state_sha256", answer.input_state_sha256);
    read_field(stream, "start_coefficient", answer.start);
    read_field(stream, "end_coefficient", answer.end);
    read_field(stream, "source_rows", source_rows);
    read_field(stream, "right_bits", right_bits);
    read_field(stream, "contribution_words", contribution_words);
    read_field(stream, "contribution_fnv1a64_le", answer.fnv1a64_le);
    std::string trailing;
    if (stream >> trailing) {
        throw std::runtime_error("contribution manifest has trailing tokens");
    }
    if (request_binding != request.binding_sha256
        || answer.relation_slice_sha256.size() != 64
        || answer.input_state_sha256.size() != (answer.start ? 64U : 1U)
        || (!answer.start && answer.input_state_sha256 != "-")
        || answer.start >= answer.end
        || answer.end > request.coefficient_matrix_count
        || source_rows != base::SOURCE_ROWS
        || right_bits != krylov::WIDTH
        || contribution_words != PANEL_WORDS
        || answer.fnv1a64_le.size() != 16
        || !std::filesystem::is_regular_file(root / "result.json")) {
        throw std::runtime_error("contribution manifest binding differs");
    }
    return answer;
}

void run_shard(
    const std::filesystem::path &operator_path,
    const std::filesystem::path &anchor_root,
    const std::filesystem::path &permutation_path,
    const std::filesystem::path &diagonal_path,
    const std::filesystem::path &request_path,
    const std::filesystem::path &slice_manifest_path,
    const std::filesystem::path &slice_payload_path,
    const std::string &state_path,
    const std::filesystem::path &output_root,
    int batch) {
#if !defined(MCELIECEX_ENABLE_LSTAR_REJ_RECONSTRUCTION) \
    || MCELIECEX_ENABLE_LSTAR_REJ_RECONSTRUCTION != 1
    throw std::runtime_error("Lstar REJ reconstruction build is disabled");
#else
    if (std::filesystem::exists(output_root)) {
        throw std::runtime_error("refusing existing reconstruction shard root");
    }
    if (batch < 64 || batch > krylov::WIDTH
        || batch % 64 || krylov::WIDTH % batch) {
        throw std::runtime_error("reconstruction batch differs");
    }
    const ReconstructionRequest request = read_reconstruction_request(request_path);
    const RelationSlice slice = read_slice_manifest(slice_manifest_path, request);
    const std::size_t relation_words = static_cast<std::size_t>(
        (slice.end - slice.start) * MATRIX_WORDS);
    std::vector<u64> relations = read_words_exact(slice_payload_path, relation_words);
    if (hex_u64(checksum_words(relations)) != slice.payload_fnv1a64_le) {
        throw std::runtime_error("relation slice FNV differs");
    }
    krylov::State state;
    if (slice.start == 0) {
        if (state_path != "-") {
            throw std::runtime_error("initial reconstruction shard state differs");
        }
        state.panel = krylov::initial_panel();
    } else {
        if (state_path == "-") {
            throw std::runtime_error("noninitial reconstruction state is absent");
        }
        state = krylov::read_state(state_path);
    }
    if (state.completed_steps != slice.start) {
        throw std::runtime_error("reconstruction checkpoint order differs");
    }
    HostOperator host = load_operator(
        operator_path, anchor_root, permutation_path, diagonal_path);

    u64 *device_current = nullptr;
    u64 *device_next = nullptr;
    u64 *device_candidates = nullptr;
    u64 *device_table = nullptr;
    CUDA_CHECK(cudaMalloc(&device_current, PANEL_BYTES));
    CUDA_CHECK(cudaMalloc(&device_next, PANEL_BYTES));
    CUDA_CHECK(cudaMalloc(&device_candidates, PANEL_BYTES));
    constexpr std::size_t table_words =
        krylov::WORDS * 16U * 16U * krylov::WORDS;
    CUDA_CHECK(cudaMalloc(&device_table, table_words * sizeof(u64)));
    const std::string input_state_fnv = hex_u64(checksum_words(state.panel));
    CUDA_CHECK(cudaMemcpy(
        device_current, state.panel.data(), PANEL_BYTES, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(device_candidates, 0, PANEL_BYTES));
    state.panel.clear();
    state.panel.shrink_to_fit();
    DeviceOperator device(host, batch);
    std::size_t free_after_setup = 0, total_bytes = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_after_setup, &total_bytes));
    if (free_after_setup < DEVICE_RESERVE) {
        throw std::runtime_error("reconstruction device reserve below 256 MiB");
    }
    if (!std::filesystem::create_directory(output_root)
        || ::chmod(output_root.c_str(), 0700) != 0) {
        throw std::runtime_error("cannot create reconstruction shard root");
    }

    double accumulation_seconds = 0;
    double operator_seconds = 0;
    double max_rounding_error = 0;
    u64 max_coefficient = 0;
    const auto started = Clock::now();
    for (u64 coefficient = slice.start; coefficient < slice.end; ++coefficient) {
        const u64 *matrix = relations.data()
            + static_cast<std::size_t>(coefficient - slice.start) * MATRIX_WORDS;
        const std::vector<u64> table = relation_nibble_table(matrix);
        CUDA_CHECK(cudaMemcpy(
            device_table, table.data(), table.size() * sizeof(u64),
            cudaMemcpyHostToDevice));
        accumulation_seconds += base::timed_launch([&] {
            accumulate_relation_kernel<<<
                launch_blocks(base::SOURCE_ROWS), THREADS>>>(
                    device_current, base::SOURCE_ROWS,
                    device_table, device_candidates);
            CUDA_CHECK(cudaGetLastError());
        });
        if (coefficient + 1 < slice.end) {
            const base::ToeplitzTiming timing =
                device.apply(device_current, device_next);
            operator_seconds += timing.seconds;
            max_rounding_error = std::max(
                max_rounding_error, timing.max_error);
            max_coefficient = std::max(max_coefficient, timing.max_coefficient);
            std::swap(device_current, device_next);
        }
        const u64 completed = coefficient + 1;
        if (completed == slice.end || (completed - slice.start) % 64 == 0) {
            const double elapsed = std::chrono::duration<double>(
                Clock::now() - started).count();
            std::cout
                << "{\"schema\":\"mceliecex-tii254-d6-lstar-rej-reconstruction-heartbeat-v1\","
                << "\"start_coefficient\":" << slice.start << ','
                << "\"completed_coefficient\":" << completed << ','
                << "\"end_coefficient\":" << slice.end << ','
                << "\"elapsed_seconds\":" << std::setprecision(17)
                << elapsed << "}\n" << std::flush;
        }
    }

    std::vector<u64> contribution(PANEL_WORDS);
    CUDA_CHECK(cudaMemcpy(
        contribution.data(), device_candidates, PANEL_BYTES,
        cudaMemcpyDeviceToHost));
    const std::string contribution_fnv =
        hex_u64(checksum_words(contribution));
    const auto payload_temporary = output_root / "contribution.bin.tmp";
    const auto payload_path = output_root / "contribution.bin";
    write_words(payload_temporary, contribution);
    fsync_path(payload_temporary, false);
    publish_temporary_new(payload_temporary, payload_path);
    const auto manifest_temporary = output_root / "contribution.txt.tmp";
    const auto manifest_path = output_root / "contribution.txt";
    write_contribution_manifest(
        manifest_temporary, request, slice, contribution_fnv);
    fsync_path(manifest_temporary, false);
    publish_temporary_new(manifest_temporary, manifest_path);
    const double wall_seconds = std::chrono::duration<double>(
        Clock::now() - started).count();
    std::ostringstream result;
    result << std::setprecision(17)
           << "{\n"
           << "  \"schema\": \"" << SHARD_SCHEMA << "\",\n"
           << "  \"terminal\": \"tii254_d6_lstar_rej_reconstruction_shard_complete_unpromoted\",\n"
           << "  \"request_identity\": \"" << krylov::REQUEST_IDENTITY << "\",\n"
           << "  \"reconstruction_binding_sha256\": \""
           << request.binding_sha256 << "\",\n"
           << "  \"relation_slice_sha256\": \""
           << slice.payload_sha256 << "\",\n"
           << "  \"input_state_sha256\": \""
           << slice.input_state_sha256 << "\",\n"
           << "  \"input_state_inventory_sha256\": \""
           << slice.input_state_inventory_sha256 << "\",\n"
           << "  \"start_coefficient\": " << slice.start << ",\n"
           << "  \"end_coefficient\": " << slice.end << ",\n"
           << "  \"coefficient_count\": " << (slice.end - slice.start) << ",\n"
           << "  \"source_rows\": " << base::SOURCE_ROWS << ",\n"
           << "  \"right_bits\": " << krylov::WIDTH << ",\n"
           << "  \"batch\": " << batch << ",\n"
           << "  \"contribution_bytes\": " << PANEL_BYTES << ",\n"
           << "  \"contribution_fnv1a64_le\": \""
           << contribution_fnv << "\",\n"
           << "  \"input_state_fnv1a64_le\": \""
           << input_state_fnv
           << "\",\n"
           << "  \"accumulation_seconds_total\": "
           << accumulation_seconds << ",\n"
           << "  \"operator_seconds_total\": " << operator_seconds << ",\n"
           << "  \"wall_seconds\": " << wall_seconds << ",\n"
           << "  \"max_rounding_error\": " << max_rounding_error << ",\n"
           << "  \"max_integer_convolution_coefficient\": "
           << max_coefficient << ",\n"
           << "  \"device_free_bytes_after_setup\": "
           << free_after_setup << ",\n"
           << "  \"claim_boundary\": \"one exact interval contribution only; complete partition combination, A Q, independent literal E(JQ), observer profile, locator, polynomial, and key replay remain required\"\n"
           << "}\n";
    write_json_new(
        output_root / "result.json.tmp",
        output_root / "result.json",
        result.str());
    fsync_path(output_root, true);

    cudaFree(device_table);
    cudaFree(device_candidates);
    cudaFree(device_next);
    cudaFree(device_current);
#endif
}

void run_combine(
    const std::filesystem::path &operator_path,
    const std::filesystem::path &anchor_root,
    const std::filesystem::path &permutation_path,
    const std::filesystem::path &diagonal_path,
    const std::filesystem::path &request_path,
    const std::filesystem::path &output_root,
    int batch,
    const std::vector<std::filesystem::path> &contribution_roots) {
#if !defined(MCELIECEX_ENABLE_LSTAR_REJ_RECONSTRUCTION) \
    || MCELIECEX_ENABLE_LSTAR_REJ_RECONSTRUCTION != 1
    throw std::runtime_error("Lstar REJ reconstruction build is disabled");
#else
    if (std::filesystem::exists(output_root) || contribution_roots.empty()) {
        throw std::runtime_error("refusing reconstruction combination");
    }
    const ReconstructionRequest request = read_reconstruction_request(request_path);
    std::vector<ContributionManifest> contributions;
    contributions.reserve(contribution_roots.size());
    for (const auto &root : contribution_roots) {
        contributions.push_back(read_contribution_manifest(root, request));
    }
    std::sort(
        contributions.begin(), contributions.end(),
        [](const ContributionManifest &left, const ContributionManifest &right) {
            return left.start < right.start;
        });
    u64 cursor = 0;
    for (const ContributionManifest &contribution : contributions) {
        if (contribution.start != cursor) {
            throw std::runtime_error(
                "contribution intervals do not form an exact partition");
        }
        cursor = contribution.end;
    }
    if (cursor != request.coefficient_matrix_count) {
        throw std::runtime_error(
            "contribution intervals do not cover every coefficient");
    }
    std::vector<u64> candidates(PANEL_WORDS, 0);
    std::vector<std::string> contribution_fnvs;
    contribution_fnvs.reserve(contributions.size());
    for (const ContributionManifest &record : contributions) {
        std::vector<u64> contribution = read_words_exact(
            record.root / "contribution.bin", PANEL_WORDS);
        const std::string observed_fnv = hex_u64(checksum_words(contribution));
        if (observed_fnv != record.fnv1a64_le) {
            throw std::runtime_error("contribution payload FNV differs");
        }
        contribution_fnvs.push_back(observed_fnv);
        for (std::size_t word = 0; word < PANEL_WORDS; ++word) {
            candidates[word] ^= contribution[word];
        }
    }
    const unsigned int rank = panel_rank(candidates);
    const std::string candidate_fnv = hex_u64(checksum_words(candidates));
    HostOperator host = load_operator(
        operator_path, anchor_root, permutation_path, diagonal_path);
    u64 *device_candidates = nullptr;
    u64 *device_residual = nullptr;
    CUDA_CHECK(cudaMalloc(&device_candidates, PANEL_BYTES));
    CUDA_CHECK(cudaMalloc(&device_residual, PANEL_BYTES));
    CUDA_CHECK(cudaMemcpy(
        device_candidates, candidates.data(), PANEL_BYTES,
        cudaMemcpyHostToDevice));
    DeviceOperator device(host, batch);
    std::size_t free_after_setup = 0, total_bytes = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_after_setup, &total_bytes));
    if (free_after_setup < DEVICE_RESERVE) {
        throw std::runtime_error("combine device reserve below 256 MiB");
    }
    const auto started = Clock::now();
    const base::ToeplitzTiming timing =
        device.apply(device_candidates, device_residual);
    std::vector<u64> residual(PANEL_WORDS);
    CUDA_CHECK(cudaMemcpy(
        residual.data(), device_residual, PANEL_BYTES,
        cudaMemcpyDeviceToHost));
    const ResidualStats stats = residual_stats(residual);
    if (!std::filesystem::create_directory(output_root)
        || ::chmod(output_root.c_str(), 0700) != 0) {
        throw std::runtime_error("cannot create reconstruction combine root");
    }
    const auto candidates_temporary = output_root / "candidates.bin.tmp";
    const auto candidates_path = output_root / "candidates.bin";
    write_words(candidates_temporary, candidates);
    fsync_path(candidates_temporary, false);
    publish_temporary_new(candidates_temporary, candidates_path);
    const double wall_seconds = std::chrono::duration<double>(
        Clock::now() - started).count();
    std::ostringstream result;
    result << std::setprecision(17)
           << "{\n"
           << "  \"schema\": \"" << COMBINE_SCHEMA << "\",\n"
           << "  \"terminal\": \""
           << (stats.nonzero_words == 0
                   ? "tii254_d6_lstar_rej_reconstruction_AQ_zero_unpromoted"
                   : "tii254_d6_lstar_rej_reconstruction_AQ_nonzero_refusal")
           << "\",\n"
           << "  \"request_identity\": \"" << krylov::REQUEST_IDENTITY << "\",\n"
           << "  \"reconstruction_binding_sha256\": \""
           << request.binding_sha256 << "\",\n"
           << "  \"contribution_count\": " << contributions.size() << ",\n"
           << "  \"coefficient_matrix_count\": "
           << request.coefficient_matrix_count << ",\n"
           << "  \"contribution_intervals\": [";
    for (std::size_t index = 0; index < contributions.size(); ++index) {
        if (index) result << ',';
        result << '[' << contributions[index].start << ','
               << contributions[index].end << ']';
    }
    result << "],\n"
           << "  \"contribution_fnv1a64_le\": [";
    for (std::size_t index = 0; index < contribution_fnvs.size(); ++index) {
        if (index) result << ',';
        result << "\"" << contribution_fnvs[index] << "\"";
    }
    result << "],\n"
           << "  \"source_rows\": " << base::SOURCE_ROWS << ",\n"
           << "  \"right_bits\": " << krylov::WIDTH << ",\n"
           << "  \"batch\": " << batch << ",\n"
           << "  \"candidate_rank\": " << rank << ",\n"
           << "  \"candidate_bytes\": " << PANEL_BYTES << ",\n"
           << "  \"candidate_fnv1a64_le\": \"" << candidate_fnv << "\",\n"
           << "  \"A_Q_zero\": "
           << (stats.nonzero_words == 0 ? "true" : "false") << ",\n"
           << "  \"A_Q_residual_nonzero_words\": "
           << stats.nonzero_words << ",\n"
           << "  \"A_Q_residual_hamming_weight\": "
           << stats.hamming_weight << ",\n"
           << "  \"A_Q_residual_fnv1a64_le\": \""
           << stats.fnv1a64_le << "\",\n"
           << "  \"A_Q_seconds\": " << timing.seconds << ",\n"
           << "  \"A_Q_max_rounding_error\": " << timing.max_error << ",\n"
           << "  \"A_Q_max_integer_convolution_coefficient\": "
           << timing.max_coefficient << ",\n"
           << "  \"wall_seconds\": " << wall_seconds << ",\n"
           << "  \"device_free_bytes_after_setup\": "
           << free_after_setup << ",\n"
           << "  \"claim_boundary\": \"native A Q replay only; independent literal E(JQ), observer profile, canonical containment, locator, polynomial, and key replay remain required\"\n"
           << "}\n";
    write_json_new(
        output_root / "result.json.tmp",
        output_root / "result.json",
        result.str());
    fsync_path(output_root, true);
    cudaFree(device_residual);
    cudaFree(device_candidates);
#endif
}

}  // namespace mceliecex_tii254_lstar_rej_reconstruction

int main(int argc, char **argv) {
    try {
        namespace reconstruction =
            mceliecex_tii254_lstar_rej_reconstruction;
        if (argc == 2 && std::string(argv[1]) == "self-test") {
            reconstruction::run_accumulation_self_test();
            return 0;
        }
        if (argc == 12 && std::string(argv[1]) == "shard") {
            reconstruction::run_shard(
                argv[2], argv[3], argv[4], argv[5], argv[6], argv[7], argv[8],
                argv[9], argv[10], std::stoi(argv[11]));
            return 0;
        }
        if (argc >= 11 && std::string(argv[1]) == "combine") {
            if (std::string(argv[7]) != "--contributions") {
                throw std::runtime_error("combine contribution marker differs");
            }
            std::vector<std::filesystem::path> contributions;
            for (int index = 10; index < argc; ++index) {
                contributions.emplace_back(argv[index]);
            }
            reconstruction::run_combine(
                argv[2], argv[3], argv[4], argv[5], argv[6], argv[8],
                std::stoi(argv[9]), contributions);
            return 0;
        }
        std::cerr
            << "usage:\n  " << argv[0]
            << " shard OPERATOR.txt ANCHOR_ROOT PERM.u32le DIAGONAL.bin"
            << " REQUEST.txt SLICE.txt SLICE.bin STATE.bin|-"
            << " NEW_OUTPUT_DIR BATCH\n  " << argv[0]
            << " combine OPERATOR.txt ANCHOR_ROOT PERM.u32le DIAGONAL.bin"
            << " REQUEST.txt --contributions NEW_OUTPUT_DIR BATCH"
            << " CONTRIBUTION_ROOT...\n";
        return 2;
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
