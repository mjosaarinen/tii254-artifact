// CPU/CUDA differential for variable-information-minor L_star anchors.
//
// The arithmetic kernels are inherited unchanged from the sealed label-2
// worker.  Only the request reader and result schema are new.

#define main mceliecex_holdout_cuda_embedded_main
#include "holdout_cuda_worker.cu"
#undef main

#define MCELIECEX_HOLDOUT_CUDA_ALREADY_EMBEDDED 1
#define main mceliecex_tii254_lstar_anchor_embedded_main
#include "tii254_lstar_anchor_codec_worker.cu"
#undef main
#undef MCELIECEX_HOLDOUT_CUDA_ALREADY_EMBEDDED

#include "tii254_lstar_anchor_codec_v2_support.cuh"

namespace mceliecex_tii254_lstar_anchor_v2 {

constexpr const char *RESULT_SCHEMA =
    "mceliecex-tii254-d6-lstar-variable-anchor-cuda-result-v2";

void write_result(
    const std::filesystem::path &path,
    const Request &request,
    const Codec &codec,
    const char *mode,
    const Run &forward,
    const Run &transpose,
    bool compared,
    bool equal,
    const std::string &device) {
    std::ostringstream output;
    output << std::setprecision(17)
           << "{\n"
           << "  \"schema\": \"" << RESULT_SCHEMA << "\",\n"
           << "  \"terminal\": \"tii254_d6_lstar_variable_anchor_"
           << mode << "_pass\",\n"
           << "  \"claim_boundary\": \"variable anchor codec differential only; no sequence, relation, locator, polynomial, or key\",\n"
           << "  \"codec_identity\": \"" << request.codec_identity << "\",\n"
           << "  \"mode\": \"" << mode << "\",\n"
           << "  \"source_rows\": " << request.source_rows << ",\n"
           << "  \"literal_rows\": " << request.literal_rows << ",\n"
           << "  \"removed_rows\": " << request.removed_rows << ",\n"
           << "  \"free_rows\": " << request.free_rows << ",\n"
           << "  \"nonanchor_rows\": " << request.nonanchor_rows << ",\n"
           << "  \"anchor_literal_offset\": "
           << request.anchor_literal_offset << ",\n"
           << "  \"right_bits\": " << request.right_bits << ",\n"
           << "  \"common_incidences\": "
           << codec.common.row_indices.size() << ",\n"
           << "  \"forward_seconds\": " << forward.seconds << ",\n"
           << "  \"transpose_seconds\": " << transpose.seconds << ",\n"
           << "  \"forward_fnv1a64_le\": \""
           << hex_u64(checksum_words(forward.output)) << "\",\n"
           << "  \"transpose_fnv1a64_le\": \""
           << hex_u64(checksum_words(transpose.output)) << "\",\n"
           << "  \"adjoint_identity\": true,\n"
           << "  \"cuda_comparison_performed\": "
           << (compared ? "true" : "false") << ",\n"
           << "  \"cpu_cuda_equal\": ";
    if (compared) {
        output << (equal ? "true" : "false");
    } else {
        output << "null";
    }
    output << ",\n  \"device\": \"" << device << "\"\n}\n";
    write_new(path, output.str());
}

void run(
    const std::string &mode,
    const std::filesystem::path &root,
    int width,
    const std::filesystem::path &result_path) {
    if (width != 64 && width != 512) {
        throw std::runtime_error("width must be 64 or 512");
    }
    const std::filesystem::path input = root / "input";
    const Request request = read_request(
        input / (std::string("request-r") + std::to_string(width) + ".txt"));
    const Codec codec = load_codec(request, input);
    const auto source = random_words(
        static_cast<std::size_t>(request.source_rows) * request.right_words,
        UINT64_C(0xc4d6a254e9173b01));
    const auto literal_dual = random_words(
        static_cast<std::size_t>(request.literal_rows) * request.right_words,
        UINT64_C(0x72e4f93b1a65c820));
    Run forward;
    Run transpose;
    const bool compared = mode == "differential";
    bool equal = false;
    std::string device = "CPU reference";
    if (mode == "cpu") {
        forward = base::cpu_forward(request, codec, source);
        transpose = base::cpu_transpose(request, codec, literal_dual);
    } else if (mode == "cuda" || mode == "differential") {
        cudaDeviceProp properties{};
        device_name(properties);
        device = properties.name;
        if (mode == "differential") {
            const Run cpu_forward = base::cpu_forward(request, codec, source);
            const Run cpu_transpose = base::cpu_transpose(
                request, codec, literal_dual);
            forward = base::gpu_forward(request, codec, source);
            transpose = base::gpu_transpose(request, codec, literal_dual);
            equal = forward.output == cpu_forward.output
                && transpose.output == cpu_transpose.output;
            if (!equal) {
                throw std::runtime_error("variable anchor CPU/CUDA differential differs");
            }
        } else {
            forward = base::gpu_forward(request, codec, source);
            transpose = base::gpu_transpose(request, codec, literal_dual);
        }
    } else {
        throw std::runtime_error("unknown variable anchor benchmark mode");
    }
    if (
        base::lane_dot(
            forward.output, literal_dual,
            request.literal_rows, request.right_words)
        != base::lane_dot(
            source, transpose.output,
            request.source_rows, request.right_words)
    ) {
        throw std::runtime_error("variable anchor adjoint identity differs");
    }
    mceliecex_tii254_lstar_anchor_v2::write_result(
        result_path, request, codec, mode.c_str(), forward, transpose,
        compared, equal, device);
}

}  // namespace mceliecex_tii254_lstar_anchor_v2

int main(int argc, char **argv) {
    try {
        if (argc == 5) {
            const int width = std::stoi(argv[3]);
            mceliecex_tii254_lstar_anchor_v2::run(
                argv[1], argv[2], width, argv[4]);
            return 0;
        }
        std::cerr << "usage: " << argv[0]
                  << " (cpu|cuda|differential) PACKAGE_ROOT (64|512) RESULT.json\n";
        return 2;
    } catch (const std::exception &error) {
        std::cerr << "TII-254 variable L_star anchor codec refused: "
                  << error.what() << '\n';
        return 1;
    }
}
