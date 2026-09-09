// Complete device-resident TII-254 L_star composition gate.
//
//     A   = R E J
//     A^T = J^T E^T R^T
//
// J is an exact L_star anchor codec, E is the literal current-profile
// operator, and R is the systematic Toeplitz graph compressor.  The default
// build retains the sealed label-2 request.  Defining
// MCELIECEX_TII254_LSTAR_REJ_VARIABLE_ANCHOR switches only the request policy
// and independent CPU hashes to the variable-minor successor.  This
// executable is a differential/timing gate only; it has no Krylov entry point.

#define main mceliecex_holdout_cuda_embedded_main
#include "holdout_cuda_worker.cu"
#undef main

#define MCELIECEX_HOLDOUT_CUDA_ALREADY_EMBEDDED 1
#define main mceliecex_tii254_lstar_anchor_embedded_main
#include "tii254_lstar_anchor_codec_worker.cu"
#undef main
#undef MCELIECEX_HOLDOUT_CUDA_ALREADY_EMBEDDED

#ifdef MCELIECEX_TII254_LSTAR_REJ_VARIABLE_ANCHOR
#include "tii254_lstar_anchor_codec_v2_support.cuh"
#endif

#include <cufft.h>
#include <functional>
#include <map>

namespace mceliecex_tii254_lstar_rej {

namespace anchor = mceliecex_tii254_lstar_anchor;
#ifdef MCELIECEX_TII254_LSTAR_REJ_VARIABLE_ANCHOR
namespace anchor_request_policy = mceliecex_tii254_lstar_anchor_v2;
#endif

constexpr const char *PROFILE_IDENTITY =
    "72c944e9de9a936da2699ff4ef22664f91e15b9d1d15cc8f34d73f53954e94be";
#ifdef MCELIECEX_TII254_LSTAR_REJ_VARIABLE_ANCHOR
constexpr const char *RESULT_SCHEMA =
    "mceliecex-tii254-d6-lstar-variable-anchor-rej-cufft-result-v2";
constexpr const char *RESULT_TERMINAL =
    "tii254_d6_lstar_variable_anchor_rej_device_fnv_adjoint_gate_pass";
#else
constexpr const char *RESULT_SCHEMA =
    "mceliecex-tii254-d6-lstar-rej-cufft-result-v1";
constexpr const char *RESULT_TERMINAL =
    "tii254_d6_lstar_rej_device_fnv_adjoint_gate_pass";
#endif
constexpr u64 SOURCE_ROWS = 15'527'170;
constexpr u64 LITERAL_ROWS = 20'271'300;
constexpr u64 FORM_ROWS = 15'890'700;
constexpr u64 TAIL_ROWS = 363'530;
constexpr u64 DIAGONAL_BITS = 15'890'699;
constexpr u64 FFT_LENGTH = UINT64_C(1) << 24;
constexpr u64 FFT_REAL_STRIDE = FFT_LENGTH + 2;
constexpr u64 FFT_FREQUENCIES = FFT_LENGTH / 2 + 1;
constexpr u64 SOURCE_SEED = UINT64_C(0x6c7374617252454a);
constexpr u64 DUAL_SEED = UINT64_C(0x72656a6475616c31);
constexpr int REJ_THREADS = 256;

#define REJ_CUFFT_CHECK(call) do { \
    const cufftResult status_ = (call); \
    if (status_ != CUFFT_SUCCESS) { \
        throw std::runtime_error("cuFFT failure code " \
            + std::to_string(static_cast<int>(status_))); \
    } \
} while (false)

u64 splitmix_word(u64 seed, u64 index) {
    u64 value = seed + (index + 1) * UINT64_C(0x9e3779b97f4a7c15);
    value = (value ^ (value >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    value = (value ^ (value >> 27)) * UINT64_C(0x94d049bb133111eb);
    return value ^ (value >> 31);
}

std::vector<unsigned char> read_diagonal(const std::filesystem::path &path) {
    const auto payload = anchor::read_all(path);
    if (payload.size() != (DIAGONAL_BITS + 7) / 8) {
        throw std::runtime_error("systematic Toeplitz diagonal size differs");
    }
    if (DIAGONAL_BITS % 8
        && (payload.back() >> (DIAGONAL_BITS % 8)) != 0) {
        throw std::runtime_error("systematic Toeplitz diagonal padding differs");
    }
    return payload;
}

std::vector<unsigned char> read_binary_toeplitz_seed(
    const std::filesystem::path &path,
    u64 seed_bits) {
    if (!seed_bits) {
        throw std::runtime_error("binary Toeplitz seed length is zero");
    }
    const auto payload = anchor::read_all(path);
    if (payload.size() != (seed_bits + 7) / 8) {
        throw std::runtime_error("binary Toeplitz seed size differs");
    }
    if (seed_bits % 8
        && (payload.back() >> (seed_bits % 8)) != 0) {
        throw std::runtime_error("binary Toeplitz seed padding differs");
    }
    return payload;
}

void write_words_new(
    const std::filesystem::path &path, const std::vector<u64> &words) {
    if (std::filesystem::exists(path)) {
        throw std::runtime_error("refusing existing packed output: " + path.string());
    }
    std::ofstream stream(path, std::ios::binary | std::ios::out | std::ios::trunc);
    if (!stream) throw std::runtime_error("cannot create packed output");
    if (!words.empty()) {
        stream.write(
            reinterpret_cast<const char *>(words.data()),
            static_cast<std::streamsize>(words.size() * sizeof(u64)));
    }
    stream.flush();
    if (!stream) throw std::runtime_error("cannot write packed output");
}

__global__ void scatter_original_to_slice_kernel(
    const u64 *original,
    const std::uint32_t *original_to_slice,
    u64 rows,
    int words,
    u64 *slice) {
    const u64 items = rows * static_cast<u64>(words);
    const u64 stride = static_cast<u64>(blockDim.x) * gridDim.x;
    for (u64 item = static_cast<u64>(blockIdx.x) * blockDim.x + threadIdx.x;
         item < items; item += stride) {
        const u64 row = item / words;
        const int word = static_cast<int>(item % words);
        slice[static_cast<u64>(original_to_slice[row]) * words + word]
            = original[item];
    }
}

__global__ void gather_slice_to_original_kernel(
    const u64 *slice,
    const std::uint32_t *original_to_slice,
    u64 rows,
    int words,
    u64 *original) {
    const u64 items = rows * static_cast<u64>(words);
    const u64 stride = static_cast<u64>(blockDim.x) * gridDim.x;
    for (u64 item = static_cast<u64>(blockIdx.x) * blockDim.x + threadIdx.x;
         item < items; item += stride) {
        const u64 row = item / words;
        const int word = static_cast<int>(item % words);
        original[item]
            = slice[static_cast<u64>(original_to_slice[row]) * words + word];
    }
}

__global__ void fill_circulant_kernel(
    double *circulant,
    const unsigned char *diagonal) {
    const u64 stride = static_cast<u64>(blockDim.x) * gridDim.x;
    for (u64 index = static_cast<u64>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < FFT_LENGTH; index += stride) {
        u64 diagonal_index = 0;
        bool present = false;
        if (index < SOURCE_ROWS) {
            diagonal_index = TAIL_ROWS - 1 + index;
            present = true;
        } else if (index > FFT_LENGTH - TAIL_ROWS) {
            const u64 distance = FFT_LENGTH - index;
            diagonal_index = TAIL_ROWS - 1 - distance;
            present = true;
        }
        unsigned bit = 0;
        if (present) {
            bit = (diagonal[diagonal_index >> 3]
                >> (diagonal_index & 7)) & 1U;
        }
        circulant[index] = static_cast<double>(bit);
    }
}

__device__ __forceinline__ unsigned packed_seed_bit(
    const unsigned char *seed,
    u64 index) {
    return (seed[index >> 3] >> (index & 7)) & 1U;
}

// For U[i,a]=sigma[i-a+b-1], the transposed map U^T is the Toeplitz
// convolution whose natural diagonal is reverse(sigma).  This kernel builds
// its length-FFT_LENGTH circulant embedding without materialising U.
__global__ void fill_transposed_left_circulant_kernel(
    double *circulant,
    const unsigned char *sigma,
    u64 input_rows,
    u64 output_rows) {
    const u64 seed_bits = input_rows + output_rows - 1;
    const u64 stride = static_cast<u64>(blockDim.x) * gridDim.x;
    for (u64 index = static_cast<u64>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < FFT_LENGTH; index += stride) {
        u64 diagonal_index = 0;
        bool present = false;
        if (index < output_rows) {
            diagonal_index = input_rows - 1 + index;
            present = true;
        } else if (index > FFT_LENGTH - input_rows) {
            const u64 distance = FFT_LENGTH - index;
            diagonal_index = input_rows - 1 - distance;
            present = true;
        }
        unsigned bit = 0;
        if (present) {
            bit = packed_seed_bit(
                sigma, seed_bits - 1 - diagonal_index);
        }
        circulant[index] = static_cast<double>(bit);
    }
}

__global__ void unpack_panel_kernel(
    double *buffer,
    const u64 *panel,
    u64 rows,
    u64 panel_row_offset,
    int panel_words,
    int group_base,
    int local_groups) {
    const u64 items = rows * static_cast<u64>(local_groups);
    const u64 stride = static_cast<u64>(blockDim.x) * gridDim.x;
    for (u64 item = static_cast<u64>(blockIdx.x) * blockDim.x + threadIdx.x;
         item < items; item += stride) {
        const int local_group = static_cast<int>(item / rows);
        const u64 row = item - static_cast<u64>(local_group) * rows;
        const u64 word = panel[(panel_row_offset + row) * panel_words
            + group_base + local_group];
        for (int bit = 0; bit < 64; ++bit) {
            buffer[(64 * static_cast<u64>(local_group) + bit)
                    * FFT_REAL_STRIDE + row]
                = static_cast<double>((word >> bit) & 1U);
        }
    }
}

__global__ void multiply_spectra_kernel(
    cufftDoubleComplex *panels,
    const cufftDoubleComplex *circulant,
    u64 lanes,
    bool transpose) {
    const u64 items = FFT_FREQUENCIES * lanes;
    const u64 stride = static_cast<u64>(blockDim.x) * gridDim.x;
    for (u64 item = static_cast<u64>(blockIdx.x) * blockDim.x + threadIdx.x;
         item < items; item += stride) {
        const u64 frequency = item % FFT_FREQUENCIES;
        const cufftDoubleComplex left = panels[item];
        cufftDoubleComplex right = circulant[frequency];
        if (transpose) right.y = -right.y;
        panels[item] = {
            left.x * right.x - left.y * right.y,
            left.x * right.y + left.y * right.x,
        };
    }
}

__global__ void pack_panel_kernel(
    u64 *output,
    const double *buffer,
    u64 rows,
    u64 output_row_offset,
    int output_words,
    int group_base,
    int local_groups,
    const u64 *xor_panel,
    u64 xor_row_offset,
    unsigned long long *max_error_bits,
    unsigned long long *max_coefficient) {
    __shared__ unsigned long long error_by_thread[REJ_THREADS];
    __shared__ unsigned long long coefficient_by_thread[REJ_THREADS];
    const u64 items = rows * static_cast<u64>(local_groups);
    const u64 stride = static_cast<u64>(blockDim.x) * gridDim.x;
    const double inverse_length = 1.0 / static_cast<double>(FFT_LENGTH);
    double thread_error = 0;
    unsigned long long thread_coefficient = 0;
    for (u64 item = static_cast<u64>(blockIdx.x) * blockDim.x + threadIdx.x;
         item < items; item += stride) {
        const int local_group = static_cast<int>(item / rows);
        const u64 row = item - static_cast<u64>(local_group) * rows;
        u64 word = 0;
        for (int bit = 0; bit < 64; ++bit) {
            const double scaled = buffer[
                (64 * static_cast<u64>(local_group) + bit)
                    * FFT_REAL_STRIDE + row] * inverse_length;
            const long long coefficient = __double2ll_rn(scaled);
            thread_error = fmax(
                thread_error,
                fabs(scaled - static_cast<double>(coefficient)));
            const unsigned long long magnitude = coefficient < 0
                ? static_cast<unsigned long long>(-coefficient)
                : static_cast<unsigned long long>(coefficient);
            thread_coefficient = max(thread_coefficient, magnitude);
            word |= (static_cast<u64>(coefficient) & 1U) << bit;
        }
        const int output_group = group_base + local_group;
        if (xor_panel != nullptr) {
            word ^= xor_panel[(xor_row_offset + row) * output_words
                + output_group];
        }
        output[(output_row_offset + row) * output_words + output_group] = word;
    }
    error_by_thread[threadIdx.x] = __double_as_longlong(thread_error);
    coefficient_by_thread[threadIdx.x] = thread_coefficient;
    __syncthreads();
    for (int offset = blockDim.x / 2; offset != 0; offset >>= 1) {
        if (threadIdx.x < offset) {
            error_by_thread[threadIdx.x] = max(
                error_by_thread[threadIdx.x],
                error_by_thread[threadIdx.x + offset]);
            coefficient_by_thread[threadIdx.x] = max(
                coefficient_by_thread[threadIdx.x],
                coefficient_by_thread[threadIdx.x + offset]);
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        atomicMax(max_error_bits, error_by_thread[0]);
        atomicMax(max_coefficient, coefficient_by_thread[0]);
    }
}

struct CufftPlans {
    cufftHandle forward{};
    cufftHandle inverse{};
    void *workspace = nullptr;
    std::size_t workspace_bytes = 0;

    explicit CufftPlans(int batch) {
        int shape[1] = {static_cast<int>(FFT_LENGTH)};
        int real_embed[1] = {static_cast<int>(FFT_LENGTH)};
        int complex_embed[1] = {static_cast<int>(FFT_FREQUENCIES)};
        std::size_t forward_work = 0;
        std::size_t inverse_work = 0;
        REJ_CUFFT_CHECK(cufftCreate(&forward));
        REJ_CUFFT_CHECK(cufftCreate(&inverse));
        REJ_CUFFT_CHECK(cufftSetAutoAllocation(forward, 0));
        REJ_CUFFT_CHECK(cufftSetAutoAllocation(inverse, 0));
        REJ_CUFFT_CHECK(cufftMakePlanMany(
            forward, 1, shape,
            real_embed, 1, static_cast<int>(FFT_REAL_STRIDE),
            complex_embed, 1, static_cast<int>(FFT_FREQUENCIES),
            CUFFT_D2Z, batch, &forward_work));
        REJ_CUFFT_CHECK(cufftMakePlanMany(
            inverse, 1, shape,
            complex_embed, 1, static_cast<int>(FFT_FREQUENCIES),
            real_embed, 1, static_cast<int>(FFT_REAL_STRIDE),
            CUFFT_Z2D, batch, &inverse_work));
        workspace_bytes = std::max(forward_work, inverse_work);
        if (workspace_bytes) CUDA_CHECK(cudaMalloc(&workspace, workspace_bytes));
        REJ_CUFFT_CHECK(cufftSetWorkArea(forward, workspace));
        REJ_CUFFT_CHECK(cufftSetWorkArea(inverse, workspace));
    }

    ~CufftPlans() {
        if (forward) cufftDestroy(forward);
        if (inverse) cufftDestroy(inverse);
        cudaFree(workspace);
    }

    CufftPlans(const CufftPlans &) = delete;
    CufftPlans &operator=(const CufftPlans &) = delete;
};

struct ToeplitzTiming {
    double seconds = 0;
    double max_error = 0;
    u64 max_coefficient = 0;
};

struct DeviceToeplitz {
    int width;
    int words;
    int batch;
    CufftPlans plans;
    cufftDoubleComplex *circulant = nullptr;
    double *buffer = nullptr;
    unsigned long long *error = nullptr;
    unsigned long long *coefficient = nullptr;
    double setup_seconds = 0;

    DeviceToeplitz(
        const std::vector<unsigned char> &diagonal,
        int width_value,
        int batch_value)
        : width(width_value), words(width_value / 64), batch(batch_value),
          plans(batch_value) {
        unsigned char *device_diagonal = nullptr;
        double *device_circulant = nullptr;
        CUDA_CHECK(cudaMalloc(&device_diagonal, diagonal.size()));
        CUDA_CHECK(cudaMemcpy(
            device_diagonal, diagonal.data(), diagonal.size(),
            cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMalloc(
            &device_circulant, FFT_REAL_STRIDE * sizeof(double)));
        CUDA_CHECK(cudaMemset(
            device_circulant, 0, FFT_REAL_STRIDE * sizeof(double)));
        fill_circulant_kernel<<<launch_blocks(FFT_LENGTH), REJ_THREADS>>>(
            device_circulant, device_diagonal);
        CUDA_CHECK(cudaGetLastError());
        cufftHandle plan{};
        REJ_CUFFT_CHECK(cufftPlan1d(
            &plan, static_cast<int>(FFT_LENGTH), CUFFT_D2Z, 1));
        const auto started = Clock::now();
        REJ_CUFFT_CHECK(cufftExecD2Z(
            plan,
            reinterpret_cast<cufftDoubleReal *>(device_circulant),
            reinterpret_cast<cufftDoubleComplex *>(device_circulant)));
        CUDA_CHECK(cudaDeviceSynchronize());
        setup_seconds = std::chrono::duration<double>(Clock::now() - started).count();
        cufftDestroy(plan);
        cudaFree(device_diagonal);
        circulant = reinterpret_cast<cufftDoubleComplex *>(device_circulant);
        CUDA_CHECK(cudaMalloc(
            &buffer,
            static_cast<std::size_t>(batch) * FFT_REAL_STRIDE * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&error, sizeof(*error)));
        CUDA_CHECK(cudaMalloc(&coefficient, sizeof(*coefficient)));
    }

    ~DeviceToeplitz() {
        cudaFree(coefficient);
        cudaFree(error);
        cudaFree(buffer);
        cudaFree(circulant);
    }

    // Apply any Toeplitz matrix using this object's already allocated FFT
    // plans, workspace, and batch buffer.  This is the allocation-sharing
    // boundary used by a possible information-theoretic left projection: it
    // needs only a second circulant spectrum, never a second 34 GiB
    // buffer/workspace pair.
    ToeplitzTiming apply_rectangular(
        const cufftDoubleComplex *spectrum,
        bool conjugate_spectrum,
        const u64 *input,
        u64 input_rows,
        u64 input_row_offset,
        u64 *output,
        u64 output_rows,
        u64 output_row_offset,
        const u64 *xor_panel,
        u64 xor_row_offset) {
        CUDA_CHECK(cudaMemset(error, 0, sizeof(*error)));
        CUDA_CHECK(cudaMemset(coefficient, 0, sizeof(*coefficient)));
        cudaEvent_t before{}, after{};
        CUDA_CHECK(cudaEventCreate(&before));
        CUDA_CHECK(cudaEventCreate(&after));
        CUDA_CHECK(cudaEventRecord(before));
        const int local_groups = batch / 64;
        for (int lane_base = 0; lane_base < width; lane_base += batch) {
            const int group_base = lane_base / 64;
            CUDA_CHECK(cudaMemset(
                buffer, 0,
                static_cast<std::size_t>(batch)
                    * FFT_REAL_STRIDE * sizeof(double)));
            unpack_panel_kernel<<<
                launch_blocks(input_rows * local_groups), REJ_THREADS>>>(
                    buffer, input, input_rows, input_row_offset, words,
                    group_base, local_groups);
            CUDA_CHECK(cudaGetLastError());
            REJ_CUFFT_CHECK(cufftExecD2Z(
                plans.forward,
                reinterpret_cast<cufftDoubleReal *>(buffer),
                reinterpret_cast<cufftDoubleComplex *>(buffer)));
            multiply_spectra_kernel<<<
                launch_blocks(FFT_FREQUENCIES * static_cast<u64>(batch)),
                REJ_THREADS>>>(
                    reinterpret_cast<cufftDoubleComplex *>(buffer),
                    spectrum,
                    static_cast<u64>(batch),
                    conjugate_spectrum);
            CUDA_CHECK(cudaGetLastError());
            REJ_CUFFT_CHECK(cufftExecZ2D(
                plans.inverse,
                reinterpret_cast<cufftDoubleComplex *>(buffer),
                reinterpret_cast<cufftDoubleReal *>(buffer)));
            pack_panel_kernel<<<
                launch_blocks(output_rows * local_groups), REJ_THREADS>>>(
                    output, buffer, output_rows, output_row_offset, words,
                    group_base, local_groups,
                    xor_panel, xor_row_offset,
                    error, coefficient);
            CUDA_CHECK(cudaGetLastError());
        }
        CUDA_CHECK(cudaEventRecord(after));
        CUDA_CHECK(cudaEventSynchronize(after));
        float milliseconds = 0;
        CUDA_CHECK(cudaEventElapsedTime(&milliseconds, before, after));
        unsigned long long error_bits = 0;
        unsigned long long maximum = 0;
        CUDA_CHECK(cudaMemcpy(
            &error_bits, error, sizeof(error_bits), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(
            &maximum, coefficient, sizeof(maximum), cudaMemcpyDeviceToHost));
        cudaEventDestroy(after);
        cudaEventDestroy(before);
        double observed_error = 0;
        std::memcpy(&observed_error, &error_bits, sizeof(observed_error));
        if (!(observed_error < 0.25)) {
            throw std::runtime_error("complete-composition cuFFT rounding refused");
        }
        return {
            milliseconds / 1000.0,
            observed_error,
            static_cast<u64>(maximum),
        };
    }

    ToeplitzTiming apply(bool transpose, const u64 *input, u64 *output) {
        if (transpose) {
            CUDA_CHECK(cudaMemcpy(
                output, input,
                static_cast<std::size_t>(SOURCE_ROWS) * words * sizeof(u64),
                cudaMemcpyDeviceToDevice));
        }
        return apply_rectangular(
            circulant,
            transpose,
            input,
            transpose ? SOURCE_ROWS : TAIL_ROWS,
            transpose ? 0 : SOURCE_ROWS,
            output,
            transpose ? TAIL_ROWS : SOURCE_ROWS,
            transpose ? SOURCE_ROWS : 0,
            transpose ? nullptr : input,
            0);
    }
};

// Build only the spectrum for an N-by-b implicit Toeplitz left panel.  The
// caller owns the returned device allocation.  It is deliberately separate
// from DeviceToeplitz's large batch buffer so a two-sided supplier can reuse
// the existing plans and workspace in apply_rectangular().
cufftDoubleComplex *build_transposed_left_spectrum(
    const std::vector<unsigned char> &seed,
    u64 input_rows,
    u64 output_rows,
    double &setup_seconds) {
    if (!input_rows || !output_rows
        || input_rows + output_rows - 1 > FFT_LENGTH
        || seed.size() != (input_rows + output_rows - 1 + 7) / 8) {
        throw std::runtime_error("Toeplitz left spectrum shape differs");
    }
    unsigned char *device_seed = nullptr;
    double *device_circulant = nullptr;
    cufftHandle plan{};
    try {
        CUDA_CHECK(cudaMalloc(&device_seed, seed.size()));
        CUDA_CHECK(cudaMemcpy(
            device_seed, seed.data(), seed.size(), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMalloc(
            &device_circulant, FFT_REAL_STRIDE * sizeof(double)));
        CUDA_CHECK(cudaMemset(
            device_circulant, 0, FFT_REAL_STRIDE * sizeof(double)));
        fill_transposed_left_circulant_kernel<<<
            launch_blocks(FFT_LENGTH), REJ_THREADS>>>(
                device_circulant, device_seed, input_rows, output_rows);
        CUDA_CHECK(cudaGetLastError());
        REJ_CUFFT_CHECK(cufftPlan1d(
            &plan, static_cast<int>(FFT_LENGTH), CUFFT_D2Z, 1));
        const auto started = Clock::now();
        REJ_CUFFT_CHECK(cufftExecD2Z(
            plan,
            reinterpret_cast<cufftDoubleReal *>(device_circulant),
            reinterpret_cast<cufftDoubleComplex *>(device_circulant)));
        CUDA_CHECK(cudaDeviceSynchronize());
        setup_seconds = std::chrono::duration<double>(
            Clock::now() - started).count();
        cufftDestroy(plan);
        plan = {};
        cudaFree(device_seed);
        return reinterpret_cast<cufftDoubleComplex *>(device_circulant);
    } catch (...) {
        if (plan) cufftDestroy(plan);
        cudaFree(device_circulant);
        cudaFree(device_seed);
        throw;
    }
}

double timed_launch(const std::function<void()> &launch) {
    cudaEvent_t before{}, after{};
    CUDA_CHECK(cudaEventCreate(&before));
    CUDA_CHECK(cudaEventCreate(&after));
    CUDA_CHECK(cudaEventRecord(before));
    launch();
    CUDA_CHECK(cudaEventRecord(after));
    CUDA_CHECK(cudaEventSynchronize(after));
    float milliseconds = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, before, after));
    cudaEventDestroy(after);
    cudaEventDestroy(before);
    return milliseconds / 1000.0;
}

void launch_j_forward(
    const anchor::Request &request,
    const anchor::Codec &codec,
    const anchor::DeviceCodec &device,
    const u64 *source,
    u64 *literal,
    u64 *common) {
    CUDA_CHECK(cudaMemset(
        literal, 0,
        static_cast<std::size_t>(request.literal_rows)
            * request.right_words * sizeof(u64)));
    CUDA_CHECK(cudaMemset(
        common, 0,
        static_cast<std::size_t>(request.common_rows)
            * request.right_words * sizeof(u64)));
    anchor::nonanchor_forward_kernel<<<
        launch_blocks(request.nonanchor_rows * request.right_words), THREADS>>>(
            source, device.nonanchor, request.retained_rows,
            request.nonanchor_rows, request.right_words, literal);
    anchor::fill_free_kernel<<<
        launch_blocks(request.free_rows * request.right_words), THREADS>>>(
            source, device.free_sources,
            request.retained_rows + request.nonanchor_rows,
            request.free_rows, request.right_words, common);
    anchor::correction_forward_kernel<<<
        launch_blocks(request.removed_rows * request.right_words), THREADS>>>(
            source, device.correction, device.pivot_sources,
            request.retained_rows + request.nonanchor_rows,
            request.removed_rows, request.free_rows,
            codec.correction.words, request.right_words, common);
    anchor::common_forward_kernel<<<
        launch_blocks(request.anchor_rows * request.right_words), THREADS>>>(
            common, device.common_row_offsets, device.common_row_columns,
            request.anchor_rows, request.anchor_literal_offset,
            request.right_words, literal);
    anchor::retained_forward_kernel<<<
        launch_blocks(request.retained_rows * request.right_words), THREADS>>>(
            source, device.retained_offsets, device.retained_positions,
            request.retained_rows, request.right_words, literal);
    CUDA_CHECK(cudaGetLastError());
}

void launch_j_transpose(
    const anchor::Request &request,
    const anchor::Codec &codec,
    const anchor::DeviceCodec &device,
    const u64 *literal,
    u64 *source,
    u64 *common) {
    CUDA_CHECK(cudaMemset(
        source, 0,
        static_cast<std::size_t>(request.source_rows)
            * request.right_words * sizeof(u64)));
    anchor::nonanchor_transpose_kernel<<<
        launch_blocks(request.nonanchor_rows * request.right_words), THREADS>>>(
            literal, device.nonanchor, request.retained_rows,
            request.nonanchor_rows, request.right_words, source);
    anchor::common_transpose_kernel<<<
        launch_blocks(request.common_rows * request.right_words), THREADS>>>(
            literal, device.common_offsets, device.common_rows,
            request.common_rows, request.anchor_literal_offset,
            request.right_words, common);
    anchor::correction_transpose_kernel<<<
        launch_blocks(request.free_rows * request.right_words), THREADS>>>(
            common, device.correction_transpose,
            device.pivot_sources, device.free_sources,
            request.retained_rows, request.nonanchor_rows,
            request.free_rows, codec.correction.transpose_words,
            request.right_words, source);
    anchor::retained_transpose_kernel<<<
        launch_blocks(request.retained_rows * request.right_words), THREADS>>>(
            literal, device.retained_offsets, device.retained_positions,
            request.retained_rows, request.right_words, source);
    CUDA_CHECK(cudaGetLastError());
}

void launch_e_forward(
    const Request &request,
    const SlicePlan &plan,
    const unsigned char *positions,
    const u64 *literal_slice,
    u64 *form) {
    CUDA_CHECK(cudaMemset(
        form, 0,
        static_cast<std::size_t>(request.expected_columns)
            * request.words * sizeof(u64)));
    for (std::size_t point_index = 0;
         point_index < request.points.size(); ++point_index) {
        const Point &point = request.points[point_index];
        for (const SliceClass &slice_class : plan.points[point_index].classes) {
            launch_mt_slice_recompute_class_raw(
                request, point, slice_class,
                literal_slice + slice_class.range_offset * request.words,
                form, positions, point_index);
        }
    }
}

void launch_e_transpose(
    const Request &request,
    const SlicePlan &plan,
    const unsigned char *positions,
    const u64 *form,
    u64 *literal_slice) {
    // The current recompute kernels assign every covered row.  Clear anyway
    // so the wrapper's E^T output contract remains explicit and future
    // sparse/holey slice plans cannot expose stale forward-scatter data.
    CUDA_CHECK(cudaMemset(
        literal_slice, 0,
        static_cast<std::size_t>(request.expected_rows)
            * request.words * sizeof(u64)));
    for (std::size_t point_index = 0;
         point_index < request.points.size(); ++point_index) {
        const Point &point = request.points[point_index];
        for (const SliceClass &slice_class : plan.points[point_index].classes) {
            launch_m_slice_recompute_class_raw(
                request, point, slice_class,
                form,
                literal_slice + slice_class.range_offset * request.words,
                positions, point_index);
        }
    }
}

struct Expected {
    const char *source;
    const char *literal_forward;
    const char *form_forward;
    const char *rej_forward;
    const char *dual;
    const char *form_transpose;
    const char *literal_transpose;
    const char *rej_transpose;
};

Expected expected_for(int width) {
#ifdef MCELIECEX_TII254_LSTAR_REJ_VARIABLE_ANCHOR
    if (width == 64) {
        return {
            "c7c6bb87cb80f636", "925232a12278fc4d", "f64482655eb191e0",
            "dff3c8a259c5b23a", "faab72bf6bbb742b", "aa87b7e54a529d18",
            "c862d4f17368caeb", "c55a9a46caf7a5d1",
        };
    }
    if (width == 512) {
        return {
            "b8eda24c13770428", "89dd36835df8a9f0", "203d3133c0e89585",
            "857bac659a9a6571", "6b8ffb3263ac5a19", "c0a7755ec3f7651e",
            "73b8ec6684d9b542", "5353db9e62c44205",
        };
    }
#else
    if (width == 64) {
        return {
            "c7c6bb87cb80f636", "54b9dd44aceeb17a", "d50663af1808eefe",
            "6018ff5579de6444", "faab72bf6bbb742b", "aa87b7e54a529d18",
            "c862d4f17368caeb", "4c8f8a02b7cefe8d",
        };
    }
    if (width == 512) {
        return {
            "b8eda24c13770428", "851b415dd0984f0d", "4131259dfbee7d32",
            "f6e2afc3baf81866", "6b8ffb3263ac5a19", "c0a7755ec3f7651e",
            "73b8ec6684d9b542", "1aa7ea0054329404",
        };
    }
#endif
    throw std::runtime_error("no independent CPU oracle for width");
}

std::string snapshot(
    const u64 *device,
    std::size_t words,
    const std::filesystem::path &path,
    bool emit,
    std::vector<u64> *retained = nullptr) {
    std::vector<u64> host(words);
    CUDA_CHECK(cudaMemcpy(
        host.data(), device, words * sizeof(u64), cudaMemcpyDeviceToHost));
    const std::string hash = hex_u64(checksum_words(host));
    if (emit) write_words_new(path, host);
    if (retained != nullptr) *retained = std::move(host);
    return hash;
}

void require_hash(
    const char *stage, const std::string &observed, const char *expected) {
    if (observed != expected) {
        throw std::runtime_error(
            std::string("independent CPU/CUDA mismatch at ") + stage
            + ": " + observed + " != " + expected);
    }
}

void run(
    const std::filesystem::path &operator_path,
    const std::filesystem::path &anchor_root,
    const std::filesystem::path &permutation_path,
    const std::filesystem::path &diagonal_path,
    int width,
    int batch,
    bool emit,
    const std::filesystem::path &output_root) {
    if ((width != 64 && width != 512)
        || batch < 64 || batch > width || batch % 64 || width % batch) {
        throw std::runtime_error(
            "width must be 64 or 512 and batch must divide it in 64-lane groups");
    }
    if (std::filesystem::exists(output_root)
        || !std::filesystem::create_directory(output_root)) {
        throw std::runtime_error("refusing existing or unavailable output root");
    }
    Request operator_request = ::read_request(operator_path);
    if (operator_request.public_sha256 != PROFILE_IDENTITY
        || operator_request.expected_columns != FORM_ROWS
        || operator_request.expected_rows != LITERAL_ROWS
        || operator_request.expected_nonzeros != 6'223'124'835ULL
        || operator_request.words != 1) {
        throw std::runtime_error("current-profile literal operator binding differs");
    }
    operator_request.words = width / 64;
    const auto input_root = anchor_root / "input";
    const auto anchor_request =
#ifdef MCELIECEX_TII254_LSTAR_REJ_VARIABLE_ANCHOR
        anchor_request_policy::read_request(
#else
        anchor::read_request(
#endif
            input_root / (std::string("request-r") + std::to_string(width) + ".txt"));
    if (anchor_request.source_rows != SOURCE_ROWS
        || anchor_request.literal_rows != LITERAL_ROWS
        || anchor_request.right_words != operator_request.words) {
        throw std::runtime_error("anchor/operator panel dimensions differ");
    }
    const anchor::Codec codec =
#ifdef MCELIECEX_TII254_LSTAR_REJ_VARIABLE_ANCHOR
        anchor_request_policy::load_codec(anchor_request, input_root);
#else
        anchor::load_codec(
        anchor_request,
        input_root / "complement-selector.u32le",
        input_root / "nonanchor-t64-positions.u32le",
        input_root / "common-e4-reduced.sparse",
        input_root / "correction.matrix",
        input_root / "pivot-common-sources.u32le",
        input_root / "free-common-sources.u32le",
        input_root / "retained-relations.bin");
#endif
    const auto permutation = anchor::read_u32s(permutation_path, LITERAL_ROWS);
    std::vector<unsigned char> seen(static_cast<std::size_t>(LITERAL_ROWS), 0);
    for (const std::uint32_t row : permutation) {
        if (row >= LITERAL_ROWS || seen[row]++) {
            throw std::runtime_error("literal original/slice payload is not a permutation");
        }
    }
    const auto diagonal = read_diagonal(diagonal_path);
    const SlicePlan plan = build_slice_plan(operator_request, false);
    const auto positions = point_position_tables(operator_request);
    CUDA_CHECK(cudaMemcpyToSymbol(
        DEVICE_CHOOSE, HOST_CHOOSE.values.data(), sizeof(HOST_CHOOSE.values)));

    const int words = operator_request.words;
    const std::size_t source_words = static_cast<std::size_t>(SOURCE_ROWS) * words;
    const std::size_t literal_words = static_cast<std::size_t>(LITERAL_ROWS) * words;
    const std::size_t form_words = static_cast<std::size_t>(FORM_ROWS) * words;
    const std::size_t common_words = static_cast<std::size_t>(anchor_request.common_rows) * words;
    std::vector<u64> source = random_words(source_words, SOURCE_SEED);
    std::vector<u64> dual(source_words);
    for (std::size_t index = 0; index < dual.size(); ++index) {
        dual[index] = splitmix_word(DUAL_SEED, static_cast<u64>(index));
    }

    std::size_t free_before = 0;
    std::size_t total_bytes = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_before, &total_bytes));
    anchor::DeviceCodec device_codec = anchor::upload(codec);
    std::uint32_t *device_permutation = anchor::copy_device(permutation);
    unsigned char *device_positions = anchor::copy_device(positions);
    u64 *device_source = nullptr;
    u64 *device_literal_original = nullptr;
    u64 *device_literal_slice = nullptr;
    u64 *device_form = nullptr;
    u64 *device_compressed = nullptr;
    u64 *device_common = nullptr;
    CUDA_CHECK(cudaMalloc(&device_source, source_words * sizeof(u64)));
    CUDA_CHECK(cudaMalloc(&device_literal_original, literal_words * sizeof(u64)));
    CUDA_CHECK(cudaMalloc(&device_literal_slice, literal_words * sizeof(u64)));
    CUDA_CHECK(cudaMalloc(&device_form, form_words * sizeof(u64)));
    CUDA_CHECK(cudaMalloc(&device_compressed, source_words * sizeof(u64)));
    CUDA_CHECK(cudaMalloc(&device_common, common_words * sizeof(u64)));
    CUDA_CHECK(cudaMemcpy(
        device_source, source.data(), source_words * sizeof(u64),
        cudaMemcpyHostToDevice));
    DeviceToeplitz toeplitz(diagonal, width, batch);
    std::size_t free_after_setup = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_after_setup, &total_bytes));

    const Expected expected = expected_for(width);
    std::map<std::string, std::string> hashes;
    std::map<std::string, double> timings;
    hashes["source"] = hex_u64(checksum_words(source));
    require_hash("source", hashes["source"], expected.source);
    if (emit) write_words_new(output_root / "source.bin", source);

    timings["j_forward"] = timed_launch([&] {
        launch_j_forward(
            anchor_request, codec, device_codec,
            device_source, device_literal_original, device_common);
    });
    hashes["literal_forward"] = snapshot(
        device_literal_original, literal_words,
        output_root / "literal-forward.bin", emit);
    require_hash(
        "literal_forward", hashes["literal_forward"], expected.literal_forward);

    timings["original_to_slice"] = timed_launch([&] {
        scatter_original_to_slice_kernel<<<
            launch_blocks(static_cast<u64>(literal_words)), THREADS>>>(
                device_literal_original, device_permutation,
                LITERAL_ROWS, words, device_literal_slice);
        CUDA_CHECK(cudaGetLastError());
    });
    timings["e_forward"] = timed_launch([&] {
        launch_e_forward(
            operator_request, plan, device_positions,
            device_literal_slice, device_form);
    });
    hashes["form_forward"] = snapshot(
        device_form, form_words, output_root / "form-forward.bin", emit);
    require_hash("form_forward", hashes["form_forward"], expected.form_forward);

    const ToeplitzTiming r_forward = toeplitz.apply(
        false, device_form, device_compressed);
    timings["r_forward"] = r_forward.seconds;
    std::vector<u64> forward;
    hashes["rej_forward"] = snapshot(
        device_compressed, source_words,
        output_root / "rej-forward.bin", emit, &forward);
    require_hash("rej_forward", hashes["rej_forward"], expected.rej_forward);

    hashes["dual"] = hex_u64(checksum_words(dual));
    require_hash("dual", hashes["dual"], expected.dual);
    if (emit) write_words_new(output_root / "range-dual.bin", dual);
    CUDA_CHECK(cudaMemcpy(
        device_compressed, dual.data(), source_words * sizeof(u64),
        cudaMemcpyHostToDevice));
    const ToeplitzTiming r_transpose = toeplitz.apply(
        true, device_compressed, device_form);
    timings["r_transpose"] = r_transpose.seconds;
    hashes["form_transpose"] = snapshot(
        device_form, form_words,
        output_root / "form-transpose.bin", emit);
    require_hash(
        "form_transpose", hashes["form_transpose"], expected.form_transpose);

    timings["e_transpose"] = timed_launch([&] {
        launch_e_transpose(
            operator_request, plan, device_positions,
            device_form, device_literal_slice);
    });
    timings["slice_to_original"] = timed_launch([&] {
        gather_slice_to_original_kernel<<<
            launch_blocks(static_cast<u64>(literal_words)), THREADS>>>(
                device_literal_slice, device_permutation,
                LITERAL_ROWS, words, device_literal_original);
        CUDA_CHECK(cudaGetLastError());
    });
    hashes["literal_transpose"] = snapshot(
        device_literal_original, literal_words,
        output_root / "literal-transpose.bin", emit);
    require_hash(
        "literal_transpose", hashes["literal_transpose"],
        expected.literal_transpose);

    timings["j_transpose"] = timed_launch([&] {
        launch_j_transpose(
            anchor_request, codec, device_codec,
            device_literal_original, device_source, device_common);
    });
    std::vector<u64> transpose;
    hashes["rej_transpose"] = snapshot(
        device_source, source_words,
        output_root / "rej-transpose.bin", emit, &transpose);
    require_hash(
        "rej_transpose", hashes["rej_transpose"], expected.rej_transpose);

    const auto lhs = anchor::lane_dot(forward, dual, SOURCE_ROWS, words);
    const auto rhs = anchor::lane_dot(source, transpose, SOURCE_ROWS, words);
    if (lhs != rhs) {
        throw std::runtime_error("complete R E J packed adjoint identity differs");
    }
    std::size_t free_after = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_after, &total_bytes));
    int device_index = 0;
    CUDA_CHECK(cudaGetDevice(&device_index));
    cudaDeviceProp properties{};
    CUDA_CHECK(cudaGetDeviceProperties(&properties, device_index));
    int runtime_version = 0;
    int cufft_version = 0;
    CUDA_CHECK(cudaRuntimeGetVersion(&runtime_version));
    REJ_CUFFT_CHECK(cufftGetVersion(&cufft_version));

    std::ostringstream result;
    result << "{\n"
           << "  \"schema\": \"" << RESULT_SCHEMA << "\",\n"
           << "  \"terminal\": \"" << RESULT_TERMINAL << "\",\n"
           << "  \"claim_boundary\": \"device FNV-1a stage gate, packed adjoint, and timing only; enclosing batch must replay emitted SHA-256 values; no Krylov, relation, locator, polynomial, or key\",\n"
           << "  \"profile_identity\": \"" << PROFILE_IDENTITY << "\",\n"
           << "  \"codec_identity\": \"" << anchor_request.codec_identity << "\",\n"
           << "  \"width\": " << width << ",\n"
           << "  \"batch\": " << batch << ",\n"
           << "  \"emitted_stage_panels\": " << (emit ? "true" : "false") << ",\n"
           << "  \"device\": \"" << properties.name << "\",\n"
           << "  \"compute_capability\": \"" << properties.major << '.'
           << properties.minor << "\",\n"
           << "  \"cuda_runtime_version\": " << runtime_version << ",\n"
           << "  \"cufft_version\": " << cufft_version << ",\n"
           << "  \"device_total_bytes\": " << total_bytes << ",\n"
           << "  \"device_free_bytes_before\": " << free_before << ",\n"
           << "  \"device_free_bytes_after_setup\": " << free_after_setup << ",\n"
           << "  \"device_free_bytes_after\": " << free_after << ",\n"
           << "  \"cufft_workspace_bytes\": " << toeplitz.plans.workspace_bytes << ",\n"
           << "  \"toeplitz_setup_seconds\": " << std::setprecision(9)
           << toeplitz.setup_seconds << ",\n"
           << "  \"timings_seconds\": {";
    bool first = true;
    for (const auto &[key, value] : timings) {
        result << (first ? "\n" : ",\n")
               << "    \"" << key << "\": " << std::setprecision(9) << value;
        first = false;
    }
    result << "\n  },\n  \"fnv1a64_le\": {";
    first = true;
    for (const auto &[key, value] : hashes) {
        result << (first ? "\n" : ",\n")
               << "    \"" << key << "\": \"" << value << "\"";
        first = false;
    }
    result << "\n  },\n"
           << "  \"r_forward_max_rounding_error\": "
           << r_forward.max_error << ",\n"
           << "  \"r_transpose_max_rounding_error\": "
           << r_transpose.max_error << ",\n"
           << "  \"r_forward_max_coefficient\": "
           << r_forward.max_coefficient << ",\n"
           << "  \"r_transpose_max_coefficient\": "
           << r_transpose.max_coefficient << ",\n"
           << "  \"all_stage_cpu_cuda_fnv1a64_equal\": true,\n"
           << "  \"all_lane_group_adjoint_identity\": true,\n"
           << "  \"successor_krylov_authorized\": false\n"
           << "}\n";
    write_new(output_root / "result.json", result.str());

    cudaFree(device_common);
    cudaFree(device_compressed);
    cudaFree(device_form);
    cudaFree(device_literal_slice);
    cudaFree(device_literal_original);
    cudaFree(device_source);
    cudaFree(device_positions);
    cudaFree(device_permutation);
    device_codec.release();
}

}  // namespace mceliecex_tii254_lstar_rej

#ifndef MCELIECEX_TII254_LSTAR_REJ_EMBEDDED
int main(int argc, char **argv) {
    try {
        if (argc != 10) {
            std::cerr
                << "usage: " << argv[0]
                << " OPERATOR.txt ANCHOR_PACKAGE LITERAL_TO_SLICE.u32le"
                << " DIAGONAL.bin WIDTH BATCH EMIT_PANELS OUTPUT_DIR gate-only\n";
            return 2;
        }
        const int width = std::stoi(argv[5]);
        const int batch = std::stoi(argv[6]);
        const std::string emit_text = argv[7];
        if (emit_text != "1") {
            throw std::runtime_error(
                "EMIT_PANELS must be one for the SHA-256-checked gate");
        }
        if (std::string(argv[9]) != "gate-only") {
            throw std::runtime_error("final gate-only sentinel differs");
        }
        mceliecex_tii254_lstar_rej::run(
            argv[1], argv[2], argv[3], argv[4],
            width, batch, true, argv[8]);
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "TII-254 L_star complete R E J gate refused: "
                  << error.what() << '\n';
        return 1;
    }
}
#endif
