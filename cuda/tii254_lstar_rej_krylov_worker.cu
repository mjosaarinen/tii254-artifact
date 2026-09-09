// Restartable projected-sequence worker for the corrected TII-254 operator
//
//     A = R E J : F2^15,527,170 -> F2^15,527,170.
//
// The default build exposes only a four-step restart-equivalence gate.  A
// production segment is compiled in only with
// MCELIECEX_ENABLE_LSTAR_REJ_PRODUCTION=1 and remains externally gated by a
// separately authenticated GH200 transition result.  This file does not
// implement PM basis, reconstruction, relation admission, or key recovery.

#ifdef MCELIECEX_TII254_LSTAR_REJ_ANCHOR5
#define MCELIECEX_TII254_LSTAR_REJ_VARIABLE_ANCHOR 1
#endif
#define MCELIECEX_TII254_LSTAR_REJ_EMBEDDED 1
#include "tii254_lstar_rej_cufft_worker.cu"
#undef MCELIECEX_TII254_LSTAR_REJ_EMBEDDED

#include <sys/stat.h>
#include <unordered_set>

namespace mceliecex_tii254_lstar_rej_krylov {

namespace base = mceliecex_tii254_lstar_rej;
namespace anchor = mceliecex_tii254_lstar_anchor;
#ifdef MCELIECEX_TII254_LSTAR_REJ_ANCHOR5
namespace anchor_request_policy = mceliecex_tii254_lstar_anchor_v2;
#endif

#ifdef MCELIECEX_TII254_LSTAR_REJ_ANCHOR5
constexpr const char *REQUEST_IDENTITY =
    "040219f22202ac1c125e5dccb4add89b38f69b31838ef8b56f44e351e448727e";
#else
constexpr const char *REQUEST_IDENTITY =
    "15bd29473bd36a97f3e45e38b221a8584f90add19fd02b7679c7f20cf7371f06";
#endif
constexpr const char *PUBLIC_SHA256 =
    "d1b7c7d808d2f129ecbbf6ea3c1d69a0a37c6e811ff8a1d856da131d46d2ccd6";
constexpr const char *CELL_IDENTITY =
    "0365b0066cb24f496c0133ba282ccbccee6bc235e808856a26f430617abcccf7";
constexpr const char *PROFILE_IDENTITY =
    "72c944e9de9a936da2699ff4ef22664f91e15b9d1d15cc8f34d73f53954e94be";
#ifdef MCELIECEX_TII254_LSTAR_REJ_ANCHOR5
constexpr const char *CODEC_IDENTITY =
    "f0905e6344f452540d7446d79336dad11125be8d6d246c9fca6d7d6330a51a0c";
#else
constexpr const char *CODEC_IDENTITY =
    "000d6544e8a6bfea28ac2307e38f718b2b64abe20c3e3dd7d352d30b1cd49bde";
#endif
constexpr const char *OBSERVER_IDENTITY =
    "7032512443d9b1676dfa97932e80e28c1541fe305664829b0896011d4a65b678";
constexpr const char *TOEPLITZ_SHA256 =
    "7fe8e995dcd8f5754b60bd4fb7b562b09d04f9582f8382a4383b6d0fb2315d73";
constexpr const char *PERMUTATION_SHA256 =
    "0224eda39f53a11dcadc81beaf9da84e903726ae16bdb9f99cee6f4a0a4f34dc";
#ifdef MCELIECEX_TII254_LSTAR_TWO_SIDED_TOEPLITZ
#ifndef MCELIECEX_TII254_LSTAR_REJ_ANCHOR5
#error "two-sided Toeplitz Krylov is defined only for the label-5 request"
#endif
constexpr const char *RIGHT_START_GENERATOR =
    "information_theoretic_binary_toeplitz_seed_v1";
constexpr const char *LEFT_PROJECTION_GENERATOR =
    "information_theoretic_binary_toeplitz_seed_v1";
constexpr const char *STATE_SCHEMA =
    "mceliecex-tii254-d6-lstar-anchor5-two-sided-toeplitz-krylov-state-v1";
constexpr const char *SEGMENT_SCHEMA =
    "mceliecex-tii254-d6-lstar-anchor5-two-sided-toeplitz-krylov-segment-result-v1";
constexpr const char *HEARTBEAT_SCHEMA =
    "mceliecex-tii254-d6-lstar-anchor5-two-sided-toeplitz-krylov-heartbeat-v1";
#elif defined(MCELIECEX_TII254_LSTAR_REJ_ANCHOR5)
constexpr const char *RIGHT_START_GENERATOR =
    "splitmix64_counter_words_v1";
constexpr const char *INITIAL_PANEL_FNV1A64_LE = "986a8771e6adc8f6";
constexpr const char *STATE_SCHEMA =
    "mceliecex-tii254-d6-lstar-anchor5-rej-krylov-state-v2";
constexpr const char *SEGMENT_SCHEMA =
    "mceliecex-tii254-d6-lstar-anchor5-rej-krylov-segment-result-v2";
constexpr const char *HEARTBEAT_SCHEMA =
    "mceliecex-tii254-d6-lstar-anchor5-rej-krylov-heartbeat-v2";
constexpr u64 SEED_Y = UINT64_C(0x0eb58a6cd0f02f02);
#else
constexpr const char *RIGHT_START_GENERATOR =
    "splitmix64_counter_words_v1";
constexpr const char *INITIAL_PANEL_FNV1A64_LE = "fd6bc95d91031029";
constexpr const char *STATE_SCHEMA =
    "mceliecex-tii254-d6-lstar-rej-krylov-state-v1";
constexpr const char *SEGMENT_SCHEMA =
    "mceliecex-tii254-d6-lstar-rej-krylov-segment-result-v1";
constexpr u64 SEED_Y = UINT64_C(0x243f6a8885a308d3);
constexpr const char *HEARTBEAT_SCHEMA =
    "mceliecex-tii254-d6-lstar-rej-krylov-heartbeat-v1";
#endif
#ifndef MCELIECEX_TII254_LSTAR_TWO_SIDED_TOEPLITZ
constexpr u64 SEED_U = UINT64_C(0x13198a2e03707344);
#endif
constexpr int WIDTH = 512;
constexpr int WORDS = WIDTH / 64;
constexpr u64 LEFT_BITS = 512;
constexpr u64 TOTAL_STEPS = 60'718;
[[maybe_unused]] constexpr u64 PRODUCTION_SEGMENT_STEPS = 4'096;
constexpr u64 HEARTBEAT_STEPS = 128;
constexpr std::size_t DEVICE_RESERVE = std::size_t{256} << 20;

template <class Value>
void read_named(std::istream &stream, const char *expected, Value &value) {
    std::string key;
    if (!(stream >> key) || key != expected || !(stream >> value)) {
        throw std::runtime_error(
            std::string("expected Lstar REJ state key ") + expected);
    }
}

struct State {
    u64 completed_steps = 0;
    std::vector<u64> panel;
#ifdef MCELIECEX_TII254_LSTAR_TWO_SIDED_TOEPLITZ
    std::string right_seed_fnv1a64_le;
    std::string left_seed_fnv1a64_le;
#endif
};

#ifndef MCELIECEX_TII254_LSTAR_TWO_SIDED_TOEPLITZ
std::vector<u64> initial_panel() {
    const std::size_t count =
        static_cast<std::size_t>(base::SOURCE_ROWS) * WORDS;
    std::vector<u64> panel(count);
    for (std::size_t index = 0; index < count; ++index) {
        panel[index] = base::splitmix_word(SEED_Y, static_cast<u64>(index));
    }
    return panel;
}
#endif

#ifdef MCELIECEX_TII254_LSTAR_TWO_SIDED_TOEPLITZ
u64 checksum_bytes(const std::vector<unsigned char> &bytes) {
    u64 value = UINT64_C(1469598103934665603);
    for (const unsigned char byte : bytes) {
        value ^= byte;
        value *= UINT64_C(1099511628211);
    }
    return value;
}

std::vector<u64> pack_seed_words(
    const std::vector<unsigned char> &seed) {
    const u64 seed_bits = base::SOURCE_ROWS + WIDTH - 1;
    std::vector<u64> words(
        static_cast<std::size_t>((seed_bits + 63) / 64 + 1), 0);
    for (std::size_t index = 0; index < seed.size(); ++index) {
        words[index / 8] |= static_cast<u64>(seed[index])
            << (8 * (index % 8));
    }
    return words;
}

__device__ __forceinline__ u64 toeplitz_seed_window(
    const u64 *seed,
    u64 start_bit) {
    const u64 word = start_bit >> 6;
    const unsigned shift = static_cast<unsigned>(start_bit & 63U);
    const u64 low = seed[word];
    if (shift == 0) return low;
    return (low >> shift) | (seed[word + 1] << (64U - shift));
}

__global__ void materialize_toeplitz_right_start_kernel(
    const u64 *seed,
    u64 *panel) {
    const u64 total = base::SOURCE_ROWS * WORDS;
    const u64 stride = static_cast<u64>(blockDim.x) * gridDim.x;
    for (u64 index = static_cast<u64>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < total; index += stride) {
        const u64 row = index / WORDS;
        const u64 word = index % WORDS;
        const u64 start = row + WIDTH - 64 * (word + 1);
        panel[index] = __brevll(toeplitz_seed_window(seed, start));
    }
}
#endif

State read_state(const std::filesystem::path &path) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) throw std::runtime_error("cannot open Lstar REJ state");
    std::string schema;
    std::getline(stream, schema);
    if (schema != STATE_SCHEMA) {
        throw std::runtime_error("Lstar REJ state schema differs");
    }
    std::string request_identity;
    std::string public_sha256;
    std::string cell_identity;
    std::string profile_identity;
    std::string codec_identity;
    std::string observer_identity;
    std::string toeplitz_sha256;
    std::string permutation_sha256;
    std::string right_start_generator;
    u64 source_rows = 0;
    int right_bits = 0;
    u64 total_steps = 0;
#ifndef MCELIECEX_TII254_LSTAR_TWO_SIDED_TOEPLITZ
    u64 seed_y = 0;
#else
    std::string left_projection_generator;
#endif
    u64 panel_words = 0;
    std::string expected_fnv;
    State state;
    read_named(stream, "request_identity", request_identity);
    read_named(stream, "authenticated_public_sha256", public_sha256);
    read_named(stream, "cell_identity", cell_identity);
    read_named(stream, "profile_identity", profile_identity);
    read_named(stream, "codec_identity", codec_identity);
    read_named(stream, "observer_descriptor_identity", observer_identity);
    read_named(stream, "toeplitz_diagonal_sha256", toeplitz_sha256);
    read_named(stream, "literal_original_to_slice_sha256", permutation_sha256);
    read_named(stream, "source_rows", source_rows);
    read_named(stream, "right_bits", right_bits);
    read_named(stream, "total_steps", total_steps);
    read_named(stream, "completed_steps", state.completed_steps);
    read_named(stream, "right_start_generator", right_start_generator);
#ifdef MCELIECEX_TII254_LSTAR_TWO_SIDED_TOEPLITZ
    read_named(
        stream, "left_projection_generator", left_projection_generator);
    read_named(
        stream, "right_seed_fnv1a64_le", state.right_seed_fnv1a64_le);
    read_named(
        stream, "left_seed_fnv1a64_le", state.left_seed_fnv1a64_le);
#else
    read_named(stream, "seed_y", seed_y);
#endif
    read_named(stream, "panel_words", panel_words);
    read_named(stream, "panel_fnv1a64_le", expected_fnv);
    std::string marker;
    if (!(stream >> marker) || marker != "data_le" || stream.get() != '\n') {
        throw std::runtime_error("Lstar REJ state data marker differs");
    }
    const u64 expected_words = base::SOURCE_ROWS * static_cast<u64>(WORDS);
    if (request_identity != REQUEST_IDENTITY
        || public_sha256 != PUBLIC_SHA256
        || cell_identity != CELL_IDENTITY
        || profile_identity != PROFILE_IDENTITY
        || codec_identity != CODEC_IDENTITY
        || observer_identity != OBSERVER_IDENTITY
        || toeplitz_sha256 != TOEPLITZ_SHA256
        || permutation_sha256 != PERMUTATION_SHA256
        || source_rows != base::SOURCE_ROWS
        || right_bits != WIDTH
        || total_steps != TOTAL_STEPS
        || state.completed_steps >= TOTAL_STEPS
        || right_start_generator != RIGHT_START_GENERATOR
#ifdef MCELIECEX_TII254_LSTAR_TWO_SIDED_TOEPLITZ
        || left_projection_generator != LEFT_PROJECTION_GENERATOR
        || state.right_seed_fnv1a64_le.size() != 16
        || state.left_seed_fnv1a64_le.size() != 16
#else
        || seed_y != SEED_Y
#endif
        || panel_words != expected_words
        || panel_words > std::numeric_limits<std::size_t>::max()) {
        throw std::runtime_error("Lstar REJ state binding differs");
    }
    state.panel.resize(static_cast<std::size_t>(panel_words));
    stream.read(
        reinterpret_cast<char *>(state.panel.data()),
        static_cast<std::streamsize>(state.panel.size() * sizeof(u64)));
    if (!stream || stream.peek() != std::char_traits<char>::eof()) {
        throw std::runtime_error("Lstar REJ state payload is truncated");
    }
    if (expected_fnv != hex_u64(checksum_words(state.panel))) {
        throw std::runtime_error("Lstar REJ state checksum differs");
    }
    return state;
}

void write_state(const std::filesystem::path &path, const State &state) {
    std::ofstream stream(path, std::ios::binary | std::ios::out | std::ios::trunc);
    if (!stream) throw std::runtime_error("cannot create Lstar REJ state temporary");
    stream << STATE_SCHEMA << '\n'
           << "request_identity " << REQUEST_IDENTITY << '\n'
           << "authenticated_public_sha256 " << PUBLIC_SHA256 << '\n'
           << "cell_identity " << CELL_IDENTITY << '\n'
           << "profile_identity " << PROFILE_IDENTITY << '\n'
           << "codec_identity " << CODEC_IDENTITY << '\n'
           << "observer_descriptor_identity " << OBSERVER_IDENTITY << '\n'
           << "toeplitz_diagonal_sha256 " << TOEPLITZ_SHA256 << '\n'
           << "literal_original_to_slice_sha256 " << PERMUTATION_SHA256 << '\n'
           << "source_rows " << base::SOURCE_ROWS << '\n'
           << "right_bits " << WIDTH << '\n'
           << "total_steps " << TOTAL_STEPS << '\n'
           << "completed_steps " << state.completed_steps << '\n'
           << "right_start_generator " << RIGHT_START_GENERATOR << '\n'
#ifdef MCELIECEX_TII254_LSTAR_TWO_SIDED_TOEPLITZ
           << "left_projection_generator " << LEFT_PROJECTION_GENERATOR << '\n'
           << "right_seed_fnv1a64_le " << state.right_seed_fnv1a64_le << '\n'
           << "left_seed_fnv1a64_le " << state.left_seed_fnv1a64_le << '\n'
#else
           << "seed_y " << SEED_Y << '\n'
#endif
           << "panel_words " << state.panel.size() << '\n'
           << "panel_fnv1a64_le " << hex_u64(checksum_words(state.panel)) << '\n'
           << "data_le\n";
    stream.write(
        reinterpret_cast<const char *>(state.panel.data()),
        static_cast<std::streamsize>(state.panel.size() * sizeof(u64)));
    stream.flush();
    if (!stream) throw std::runtime_error("cannot write Lstar REJ state");
}

#ifndef MCELIECEX_TII254_LSTAR_TWO_SIDED_TOEPLITZ
std::vector<u64> projection_rows(
    const std::filesystem::path &path) {
    const auto payload = anchor::read_all(path);
    if (payload.size() != LEFT_BITS * 4) {
        throw std::runtime_error("Lstar REJ projection payload size differs");
    }
    std::vector<u64> expected;
    expected.reserve(static_cast<std::size_t>(LEFT_BITS));
    std::unordered_set<u64> used;
    used.reserve(static_cast<std::size_t>(2 * LEFT_BITS));
    for (u64 probe = 0; expected.size() < LEFT_BITS; ++probe) {
        const u64 row = base::splitmix_word(SEED_U, probe) % base::SOURCE_ROWS;
        if (used.insert(row).second) expected.push_back(row);
    }
    std::vector<u64> observed(static_cast<std::size_t>(LEFT_BITS));
    for (std::size_t index = 0; index < observed.size(); ++index) {
        const std::size_t offset = 4 * index;
        observed[index] = static_cast<u64>(payload[offset])
            | (static_cast<u64>(payload[offset + 1]) << 8)
            | (static_cast<u64>(payload[offset + 2]) << 16)
            | (static_cast<u64>(payload[offset + 3]) << 24);
    }
    if (observed != expected) {
        throw std::runtime_error("Lstar REJ projection convention differs");
    }
    return observed;
}

__global__ void gather_projection_kernel(
    const u64 *panel,
    const u64 *rows,
    u64 *term) {
    constexpr u64 items = LEFT_BITS * static_cast<u64>(WORDS);
    const u64 stride = static_cast<u64>(blockDim.x) * gridDim.x;
    for (u64 index = static_cast<u64>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < items; index += stride) {
        const u64 row = index / WORDS;
        const int word = static_cast<int>(index % WORDS);
        term[index] = panel[rows[row] * WORDS + word];
    }
}
#endif

void run_segment(
    const std::filesystem::path &operator_path,
    const std::filesystem::path &anchor_root,
    const std::filesystem::path &permutation_path,
    const std::filesystem::path &diagonal_path,
    const std::filesystem::path &projection_or_left_seed_path,
    const std::string &right_seed_path,
    const std::string &input_state_path,
    const std::filesystem::path &output_root,
    u64 requested_steps,
    int batch,
    bool production_mode) {
    if (!requested_steps || batch < 64 || batch > WIDTH
        || batch % 64 || WIDTH % batch) {
        throw std::runtime_error("Lstar REJ segment shape differs");
    }
    if (production_mode) {
#if !defined(MCELIECEX_ENABLE_LSTAR_REJ_PRODUCTION) \
    || MCELIECEX_ENABLE_LSTAR_REJ_PRODUCTION != 1
        throw std::runtime_error("Lstar REJ production build is disabled");
#else
        if (requested_steps != PRODUCTION_SEGMENT_STEPS) {
            throw std::runtime_error("production segment order differs");
        }
#endif
    } else if (requested_steps > 4) {
        throw std::runtime_error("restart gate is limited to four steps");
    }
    if (std::filesystem::exists(output_root)) {
        throw std::runtime_error("refusing existing Lstar REJ segment root");
    }

    Request operator_request = ::read_request(operator_path);
    if (operator_request.public_sha256 != PROFILE_IDENTITY
        || operator_request.expected_columns != base::FORM_ROWS
        || operator_request.expected_rows != base::LITERAL_ROWS
        || operator_request.expected_nonzeros != 6'223'124'835ULL
        || operator_request.words != 1) {
        throw std::runtime_error("current-profile literal operator binding differs");
    }
    operator_request.words = WORDS;
    const auto anchor_input = anchor_root / "input";
    const auto anchor_request =
#ifdef MCELIECEX_TII254_LSTAR_REJ_ANCHOR5
        anchor_request_policy::read_request(
#else
        anchor::read_request(
#endif
            anchor_input / "request-r512.txt");
    if (anchor_request.source_rows != base::SOURCE_ROWS
        || anchor_request.literal_rows != base::LITERAL_ROWS
        || anchor_request.right_words != WORDS
        || anchor_request.codec_identity != CODEC_IDENTITY) {
        throw std::runtime_error("Lstar REJ anchor binding differs");
    }
    const anchor::Codec codec =
#ifdef MCELIECEX_TII254_LSTAR_REJ_ANCHOR5
        anchor_request_policy::load_codec(anchor_request, anchor_input);
#else
        anchor::load_codec(
        anchor_request,
        anchor_input / "complement-selector.u32le",
        anchor_input / "nonanchor-t64-positions.u32le",
        anchor_input / "common-e4-reduced.sparse",
        anchor_input / "correction.matrix",
        anchor_input / "pivot-common-sources.u32le",
        anchor_input / "free-common-sources.u32le",
        anchor_input / "retained-relations.bin");
#endif
    const auto permutation = anchor::read_u32s(
        permutation_path, base::LITERAL_ROWS);
    std::vector<unsigned char> seen(
        static_cast<std::size_t>(base::LITERAL_ROWS), 0);
    for (const std::uint32_t row : permutation) {
        if (row >= base::LITERAL_ROWS || seen[row]++) {
            throw std::runtime_error("Lstar REJ literal permutation differs");
        }
    }
    const auto diagonal = base::read_diagonal(diagonal_path);
    const SlicePlan plan = build_slice_plan(operator_request, false);
    const auto positions = point_position_tables(operator_request);
#ifdef MCELIECEX_TII254_LSTAR_TWO_SIDED_TOEPLITZ
    if (right_seed_path.empty()) {
        throw std::runtime_error("two-sided right seed path is empty");
    }
    constexpr u64 seed_bits = base::SOURCE_ROWS + WIDTH - 1;
    const auto right_seed = base::read_binary_toeplitz_seed(
        right_seed_path, seed_bits);
    const auto left_seed = base::read_binary_toeplitz_seed(
        projection_or_left_seed_path, seed_bits);
    const std::string right_seed_fnv = hex_u64(checksum_bytes(right_seed));
    const std::string left_seed_fnv = hex_u64(checksum_bytes(left_seed));
    const auto right_seed_words = pack_seed_words(right_seed);
#else
    if (!right_seed_path.empty()) {
        throw std::runtime_error("fixed-start right seed path is not empty");
    }
    const auto host_projection = projection_rows(projection_or_left_seed_path);
#endif
    CUDA_CHECK(cudaMemcpyToSymbol(
        DEVICE_CHOOSE, HOST_CHOOSE.values.data(), sizeof(HOST_CHOOSE.values)));

    State state;
    if (input_state_path == "-") {
#ifdef MCELIECEX_TII254_LSTAR_TWO_SIDED_TOEPLITZ
        state.right_seed_fnv1a64_le = right_seed_fnv;
        state.left_seed_fnv1a64_le = left_seed_fnv;
#else
        state.panel = initial_panel();
        if (hex_u64(checksum_words(state.panel)) != INITIAL_PANEL_FNV1A64_LE) {
            throw std::runtime_error("production right-start CPU binding differs");
        }
#endif
    } else {
        state = read_state(input_state_path);
#ifdef MCELIECEX_TII254_LSTAR_TWO_SIDED_TOEPLITZ
        if (state.right_seed_fnv1a64_le != right_seed_fnv
            || state.left_seed_fnv1a64_le != left_seed_fnv) {
            throw std::runtime_error("two-sided restart seed binding differs");
        }
#endif
    }
    const u64 start_step = state.completed_steps;
    const u64 step_count = std::min(requested_steps, TOTAL_STEPS - start_step);
    if (!step_count) throw std::runtime_error("Lstar REJ request is complete");

    const std::size_t source_words =
        static_cast<std::size_t>(base::SOURCE_ROWS) * WORDS;
    const std::size_t literal_words =
        static_cast<std::size_t>(base::LITERAL_ROWS) * WORDS;
    const std::size_t form_words =
        static_cast<std::size_t>(base::FORM_ROWS) * WORDS;
    const std::size_t common_words =
        static_cast<std::size_t>(anchor_request.common_rows) * WORDS;
    const std::size_t term_words = static_cast<std::size_t>(LEFT_BITS) * WORDS;

    std::size_t free_before = 0;
    std::size_t total_bytes = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_before, &total_bytes));
    anchor::DeviceCodec device_codec = anchor::upload(codec);
    std::uint32_t *device_permutation = anchor::copy_device(permutation);
    unsigned char *device_positions = anchor::copy_device(positions);
#ifndef MCELIECEX_TII254_LSTAR_TWO_SIDED_TOEPLITZ
    u64 *device_projection = anchor::copy_device(host_projection);
#else
    cufftDoubleComplex *left_spectrum = nullptr;
#endif
    u64 *device_state_a = nullptr;
    u64 *device_state_b = nullptr;
    u64 *device_literal_original = nullptr;
    u64 *device_literal_slice = nullptr;
    u64 *device_form = nullptr;
    u64 *device_common = nullptr;
    u64 *device_term = nullptr;
    CUDA_CHECK(cudaMalloc(&device_state_a, source_words * sizeof(u64)));
    CUDA_CHECK(cudaMalloc(&device_state_b, source_words * sizeof(u64)));
    CUDA_CHECK(cudaMalloc(
        &device_literal_original, literal_words * sizeof(u64)));
    CUDA_CHECK(cudaMalloc(&device_literal_slice, literal_words * sizeof(u64)));
    CUDA_CHECK(cudaMalloc(&device_form, form_words * sizeof(u64)));
    CUDA_CHECK(cudaMalloc(&device_common, common_words * sizeof(u64)));
    CUDA_CHECK(cudaMalloc(&device_term, term_words * sizeof(u64)));
#ifdef MCELIECEX_TII254_LSTAR_TWO_SIDED_TOEPLITZ
    if (input_state_path == "-") {
        u64 *device_seed = nullptr;
        CUDA_CHECK(cudaMalloc(
            &device_seed, right_seed_words.size() * sizeof(u64)));
        CUDA_CHECK(cudaMemcpy(
            device_seed, right_seed_words.data(),
            right_seed_words.size() * sizeof(u64), cudaMemcpyHostToDevice));
        materialize_toeplitz_right_start_kernel<<<
            launch_blocks(base::SOURCE_ROWS * WORDS), base::REJ_THREADS>>>(
                device_seed, device_state_a);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaFree(device_seed));
    } else {
        CUDA_CHECK(cudaMemcpy(
            device_state_a, state.panel.data(), source_words * sizeof(u64),
            cudaMemcpyHostToDevice));
    }
#else
    CUDA_CHECK(cudaMemcpy(
        device_state_a, state.panel.data(), source_words * sizeof(u64),
        cudaMemcpyHostToDevice));
#endif
    base::DeviceToeplitz toeplitz(diagonal, WIDTH, batch);
#ifdef MCELIECEX_TII254_LSTAR_TWO_SIDED_TOEPLITZ
    double left_setup_seconds = 0;
    left_spectrum = base::build_transposed_left_spectrum(
        left_seed, base::SOURCE_ROWS, WIDTH, left_setup_seconds);
#endif
    std::size_t free_after_setup = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_after_setup, &total_bytes));
    if (free_after_setup < DEVICE_RESERVE) {
        throw std::runtime_error("Lstar REJ device reserve below 256 MiB");
    }
    state.panel.clear();
    state.panel.shrink_to_fit();

    if (!std::filesystem::create_directory(output_root)
        || ::chmod(output_root.c_str(), 0700) != 0) {
        throw std::runtime_error("cannot create protected Lstar REJ segment root");
    }
    const auto sequence_temporary = output_root / "sequence.bin.tmp";
    const auto sequence_path = output_root / "sequence.bin";
    const auto state_temporary = output_root / "state.bin.tmp";
    const auto state_path = output_root / "state.bin";
    std::ofstream sequence(
        sequence_temporary, std::ios::binary | std::ios::out | std::ios::trunc);
    if (!sequence) throw std::runtime_error("cannot create sequence temporary");

    u64 *device_state = device_state_a;
    u64 *device_next = device_state_b;
    std::vector<u64> term(term_words);
    u64 sequence_fnv = UINT64_C(1469598103934665603);
    double j_seconds = 0;
    double scatter_seconds = 0;
    double e_seconds = 0;
    double r_seconds = 0;
    double projection_seconds = 0;
    double max_rounding_error = 0;
    u64 max_coefficient = 0;
#ifdef MCELIECEX_TII254_LSTAR_TWO_SIDED_TOEPLITZ
    double left_max_rounding_error = 0;
    u64 left_max_coefficient = 0;
#endif
    const auto started = Clock::now();

    for (u64 local_step = 0; local_step < step_count; ++local_step) {
        j_seconds += base::timed_launch([&] {
            base::launch_j_forward(
                anchor_request, codec, device_codec,
                device_state, device_literal_original, device_common);
        });
        scatter_seconds += base::timed_launch([&] {
            base::scatter_original_to_slice_kernel<<<
                launch_blocks(static_cast<u64>(literal_words)), THREADS>>>(
                    device_literal_original, device_permutation,
                    base::LITERAL_ROWS, WORDS, device_literal_slice);
            CUDA_CHECK(cudaGetLastError());
        });
        e_seconds += base::timed_launch([&] {
            base::launch_e_forward(
                operator_request, plan, device_positions,
                device_literal_slice, device_form);
        });
        const base::ToeplitzTiming r_timing =
            toeplitz.apply(false, device_form, device_next);
        r_seconds += r_timing.seconds;
        max_rounding_error = std::max(
            max_rounding_error, r_timing.max_error);
        max_coefficient = std::max(max_coefficient, r_timing.max_coefficient);
#ifdef MCELIECEX_TII254_LSTAR_TWO_SIDED_TOEPLITZ
        const base::ToeplitzTiming left_timing = toeplitz.apply_rectangular(
            left_spectrum, false,
            device_next, base::SOURCE_ROWS, 0,
            device_term, LEFT_BITS, 0,
            nullptr, 0);
        projection_seconds += left_timing.seconds;
        left_max_rounding_error = std::max(
            left_max_rounding_error, left_timing.max_error);
        left_max_coefficient = std::max(
            left_max_coefficient, left_timing.max_coefficient);
#else
        projection_seconds += base::timed_launch([&] {
            gather_projection_kernel<<<
                launch_blocks(static_cast<u64>(term_words)), THREADS>>>(
                device_next, device_projection, device_term);
            CUDA_CHECK(cudaGetLastError());
        });
#endif
        CUDA_CHECK(cudaMemcpy(
            term.data(), device_term, term_words * sizeof(u64),
            cudaMemcpyDeviceToHost));
        sequence.write(
            reinterpret_cast<const char *>(term.data()),
            static_cast<std::streamsize>(term_words * sizeof(u64)));
        if (!sequence) throw std::runtime_error("cannot append projected term");
        sequence_fnv = update_checksum_words(sequence_fnv, term);
        std::swap(device_state, device_next);
        const u64 completed = start_step + local_step + 1;
        if (completed == start_step + step_count
            || (completed - start_step) % HEARTBEAT_STEPS == 0) {
            const double elapsed = std::chrono::duration<double>(
                Clock::now() - started).count();
            std::cout
                << "{\"schema\":\"" << HEARTBEAT_SCHEMA << "\","
                << "\"request_identity\":\"" << REQUEST_IDENTITY << "\","
                << "\"completed_steps\":" << completed << ','
                << "\"segment_end_step\":" << (start_step + step_count) << ','
                << "\"elapsed_seconds\":" << std::setprecision(17)
                << elapsed << "}\n" << std::flush;
        }
    }
    sequence.flush();
    sequence.close();

    state.completed_steps = start_step + step_count;
    state.panel.resize(source_words);
    CUDA_CHECK(cudaMemcpy(
        state.panel.data(), device_state, source_words * sizeof(u64),
        cudaMemcpyDeviceToHost));
    write_state(state_temporary, state);
    fsync_path(sequence_temporary, false);
    fsync_path(state_temporary, false);
    publish_temporary_new(sequence_temporary, sequence_path);
    publish_temporary_new(state_temporary, state_path);

    std::size_t free_after = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_after, &total_bytes));
    int device_index = 0;
    CUDA_CHECK(cudaGetDevice(&device_index));
    cudaDeviceProp properties{};
    CUDA_CHECK(cudaGetDeviceProperties(&properties, device_index));
    const double wall_seconds = std::chrono::duration<double>(
        Clock::now() - started).count();
    std::ostringstream result;
    result << std::setprecision(17)
           << "{\n"
           << "  \"schema\": \"" << SEGMENT_SCHEMA << "\",\n"
           << "  \"terminal\": \"tii254_d6_lstar_rej_krylov_segment_complete_unpromoted\",\n"
           << "  \"claim_boundary\": \"restartable projected terms and bound state only; no PM basis, reconstruction, relation, locator, polynomial, key, completeness, or negative claim\",\n"
           << "  \"mode\": \""
           << (production_mode ? "production-segment" : "restart-gate-only")
           << "\",\n"
           << "  \"request_identity\": \"" << REQUEST_IDENTITY << "\",\n"
           << "  \"authenticated_public_sha256\": \"" << PUBLIC_SHA256 << "\",\n"
           << "  \"cell_identity\": \"" << CELL_IDENTITY << "\",\n"
           << "  \"profile_identity\": \"" << PROFILE_IDENTITY << "\",\n"
           << "  \"codec_identity\": \"" << CODEC_IDENTITY << "\",\n"
           << "  \"observer_descriptor_identity\": \""
           << OBSERVER_IDENTITY << "\",\n"
           << "  \"toeplitz_diagonal_sha256\": \""
           << TOEPLITZ_SHA256 << "\",\n"
           << "  \"literal_original_to_slice_sha256\": \""
           << PERMUTATION_SHA256 << "\",\n"
           << "  \"operator\": \"A=R E J\",\n"
           << "  \"origin\": \"U^T A^(i+1) Y\",\n"
           << "  \"binary_sequence_layout\": \"cado_m_by_n_row_major_le\",\n"
           << "  \"start_step\": " << start_step << ",\n"
           << "  \"term_count\": " << step_count << ",\n"
           << "  \"end_step\": " << state.completed_steps << ",\n"
           << "  \"total_steps\": " << TOTAL_STEPS << ",\n"
           << "  \"source_rows\": " << base::SOURCE_ROWS << ",\n"
           << "  \"left_bits\": " << LEFT_BITS << ",\n"
           << "  \"right_bits\": " << WIDTH << ",\n"
#ifdef MCELIECEX_TII254_LSTAR_TWO_SIDED_TOEPLITZ
           << "  \"right_seed_fnv1a64_le\": \""
           << state.right_seed_fnv1a64_le << "\",\n"
           << "  \"left_seed_fnv1a64_le\": \""
           << state.left_seed_fnv1a64_le << "\",\n"
#else
           << "  \"seed_y\": " << SEED_Y << ",\n"
#endif
           << "  \"right_start_generator\": \""
           << RIGHT_START_GENERATOR << "\",\n"
#ifdef MCELIECEX_TII254_LSTAR_TWO_SIDED_TOEPLITZ
           << "  \"left_projection_generator\": \""
           << LEFT_PROJECTION_GENERATOR << "\",\n"
#else
           << "  \"seed_u\": " << SEED_U << ",\n"
#endif
           << "  \"batch\": " << batch << ",\n"
           << "  \"gpu_name\": \"" << json_escape(properties.name) << "\",\n"
           << "  \"device_total_bytes\": " << total_bytes << ",\n"
           << "  \"device_free_bytes_before\": " << free_before << ",\n"
           << "  \"device_free_bytes_after_setup\": " << free_after_setup << ",\n"
           << "  \"device_free_bytes_after\": " << free_after << ",\n"
           << "  \"cufft_workspace_bytes\": "
           << toeplitz.plans.workspace_bytes << ",\n"
           << "  \"toeplitz_setup_seconds\": " << toeplitz.setup_seconds << ",\n"
#ifdef MCELIECEX_TII254_LSTAR_TWO_SIDED_TOEPLITZ
           << "  \"left_toeplitz_setup_seconds\": "
           << left_setup_seconds << ",\n"
#endif
           << "  \"j_seconds_total\": " << j_seconds << ",\n"
           << "  \"scatter_seconds_total\": " << scatter_seconds << ",\n"
           << "  \"e_seconds_total\": " << e_seconds << ",\n"
           << "  \"r_seconds_total\": " << r_seconds << ",\n"
           << "  \"projection_seconds_total\": " << projection_seconds << ",\n"
           << "  \"wall_seconds\": " << wall_seconds << ",\n"
           << "  \"r_max_rounding_error\": " << max_rounding_error << ",\n"
           << "  \"r_max_coefficient\": " << max_coefficient << ",\n"
#ifdef MCELIECEX_TII254_LSTAR_TWO_SIDED_TOEPLITZ
           << "  \"left_max_rounding_error\": "
           << left_max_rounding_error << ",\n"
           << "  \"left_max_coefficient\": "
           << left_max_coefficient << ",\n"
#else
           << "  \"projection_rows_fnv1a64_le\": \""
           << hex_u64(checksum_words(host_projection)) << "\",\n"
#endif
           << "  \"sequence_fnv1a64_le\": \""
           << hex_u64(sequence_fnv) << "\",\n"
           << "  \"state_fnv1a64_le\": \""
           << hex_u64(checksum_words(state.panel)) << "\",\n"
           << "  \"sequence_file\": \"sequence.bin\",\n"
           << "  \"sequence_bytes\": "
           << std::filesystem::file_size(sequence_path) << ",\n"
           << "  \"state_file\": \"state.bin\",\n"
           << "  \"successor_promotion_authorized\": false\n"
           << "}\n";
    const auto result_temporary = output_root / "result.json.tmp";
    const auto result_path = output_root / "result.json";
    {
        std::ofstream stream(result_temporary, std::ios::out | std::ios::trunc);
        stream << result.str();
        stream.flush();
        if (!stream) throw std::runtime_error("cannot write segment result");
    }
    fsync_path(result_temporary, false);
    publish_temporary_new(result_temporary, result_path);
    fsync_path(output_root, true);

    cudaFree(device_term);
    cudaFree(device_common);
    cudaFree(device_form);
    cudaFree(device_literal_slice);
    cudaFree(device_literal_original);
    cudaFree(device_state_b);
    cudaFree(device_state_a);
#ifdef MCELIECEX_TII254_LSTAR_TWO_SIDED_TOEPLITZ
    cudaFree(left_spectrum);
#else
    cudaFree(device_projection);
#endif
    cudaFree(device_positions);
    cudaFree(device_permutation);
    device_codec.release();
}

}  // namespace mceliecex_tii254_lstar_rej_krylov

#if !defined(MCELIECEX_TII254_LSTAR_REJ_KRYLOV_EMBEDDED)
int main(int argc, char **argv) {
    try {
        if (argc != 12) {
            std::cerr
                << "usage: " << argv[0]
                << " OPERATOR.txt ANCHOR_PACKAGE LITERAL_TO_SLICE.u32le"
                << " DIAGONAL.bin PROJECTION_ROWS.u32le INPUT_STATE_OR_DASH"
                << " OUTPUT_DIR STEPS BATCH"
                << " {restart-gate-only|production-segment} segment-only\n";
            return 2;
        }
        if (std::string(argv[11]) != "segment-only") {
            throw std::runtime_error("final segment-only sentinel differs");
        }
        const std::string mode = argv[10];
        if (mode != "restart-gate-only" && mode != "production-segment") {
            throw std::runtime_error("Lstar REJ segment mode differs");
        }
        mceliecex_tii254_lstar_rej_krylov::run_segment(
            argv[1], argv[2], argv[3], argv[4], argv[5], "", argv[6], argv[7],
            std::stoull(argv[8]), std::stoi(argv[9]),
            mode == "production-segment");
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "TII-254 Lstar REJ Krylov segment refused: "
                  << error.what() << '\n';
        return 1;
    }
}
#endif
