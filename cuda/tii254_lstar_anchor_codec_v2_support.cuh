#pragma once

// Variable-information-minor request reader for successor L_star anchors.
//
// Include tii254_lstar_anchor_codec_worker.cu first.  This header deliberately
// reuses its already tested sparse/common/correction kernels while replacing
// only the label-2-specific request-size policy.  The source dimension remains
// invariant because each additional deleted anchor coordinate removes one
// common-image degree of freedom and exposes one additional nonanchor axis.

namespace mceliecex_tii254_lstar_anchor_v2 {

namespace base = mceliecex_tii254_lstar_anchor;
using Request = base::Request;
using Codec = base::Codec;
using Run = base::Run;

constexpr const char *REQUEST_SCHEMA =
    "mceliecex-tii254-d6-lstar-anchor-cuda-request-v2";
constexpr const char *SOURCE_LAYOUT =
    "retained_relations_then_nonanchor_lstar_complement_axes_then_free_common_image_kernel_coordinates";
constexpr u64 T64_ROWS = 15'654'333;
constexpr u64 COMPLEMENT_ROWS = 15'654'269;
constexpr u64 SOURCE_ROWS = 15'527'170;
constexpr u64 LITERAL_ROWS = 20'271'300;
constexpr u64 RETAINED_ROWS = 64;
constexpr u64 ANCHOR_ROWS = 230'300;
constexpr u64 COMMON_ROWS = 103'137;

Request read_request(const std::filesystem::path &path) {
    std::ifstream stream(path);
    if (!stream) {
        throw std::runtime_error("cannot open variable L_star anchor request");
    }
    std::string schema;
    std::getline(stream, schema);
    if (schema != REQUEST_SCHEMA) {
        throw std::runtime_error("variable anchor request schema differs");
    }
    Request request;
    base::read_named(stream, "codec_identity", request.codec_identity);
    base::read_named(stream, "source_rows", request.source_rows);
    base::read_named(stream, "t64_rows", request.t64_rows);
    base::read_named(stream, "literal_rows", request.literal_rows);
    base::read_named(stream, "retained_rows", request.retained_rows);
    base::read_named(stream, "nonanchor_rows", request.nonanchor_rows);
    base::read_named(stream, "anchor_rows", request.anchor_rows);
    base::read_named(stream, "common_rows", request.common_rows);
    base::read_named(stream, "removed_rows", request.removed_rows);
    base::read_named(stream, "free_rows", request.free_rows);
    base::read_named(stream, "right_bits", request.right_bits);
    base::read_named(stream, "anchor_literal_offset", request.anchor_literal_offset);
    base::read_named(stream, "source_layout", request.source_layout);
    std::string trailing;
    if (stream >> trailing) {
        throw std::runtime_error("variable anchor request has trailing tokens");
    }
    request.right_words = (request.right_bits + 63) / 64;
    const bool identity_is_hex = request.codec_identity.size() == 64
        && std::all_of(
            request.codec_identity.begin(), request.codec_identity.end(),
            [](unsigned char value) {
                return (value >= '0' && value <= '9')
                    || (value >= 'a' && value <= 'f');
            });
    if (!identity_is_hex
        || request.source_layout != SOURCE_LAYOUT
        || request.source_rows != SOURCE_ROWS
        || request.t64_rows != T64_ROWS
        || request.literal_rows != LITERAL_ROWS
        || request.retained_rows != RETAINED_ROWS
        || request.anchor_rows != ANCHOR_ROWS
        || request.common_rows != COMMON_ROWS
        || !request.removed_rows
        || request.removed_rows >= COMMON_ROWS
        || request.free_rows != COMMON_ROWS - request.removed_rows
        || request.nonanchor_rows
            != COMPLEMENT_ROWS - (ANCHOR_ROWS - request.removed_rows)
        || request.source_rows
            != request.retained_rows + request.nonanchor_rows + request.free_rows
        || request.anchor_literal_offset > request.literal_rows
        || request.anchor_rows
            > request.literal_rows - request.anchor_literal_offset
        || (request.right_bits != 64 && request.right_bits != 512)) {
        throw std::runtime_error("variable anchor request dimensions differ");
    }
    return request;
}

Codec load_codec(const Request &request, const std::filesystem::path &input) {
    return base::load_codec(
        request,
        input / "complement-selector.u32le",
        input / "nonanchor-t64-positions.u32le",
        input / "common-e4-reduced.sparse",
        input / "correction.matrix",
        input / "pivot-common-sources.u32le",
        input / "free-common-sources.u32le",
        input / "retained-relations.bin");
}

}  // namespace mceliecex_tii254_lstar_anchor_v2
