// shadow_mmq.dll - llama.cpp's MMQ (int8 tensor-core, ggml-cuda/mmq.cuh, MIT) behind a small C interface for Shadow's prompt path.
// Built by build.bat (nvcc + MSVC).  The ggml runtime hooks MMQ's host code needs (device info, a device pool) are provided here; the kernels are ggml-cuda's own.
// All calls are asynchronous on the stream passed in (a CUstream / cudaStream_t) and use the primary context the engine already has current.
//
//   size_t shadow_mmq_y_bytes(int64_t k, int64_t ntok)                                      bytes of the quantized-activation buffer for ntok rows of k floats
//   int    shadow_mmq_quantize(const float* x, void* yq, int64_t k, int64_t ntok, int64_t xs, int type, void* stream)
//                                                                                          x: ntok rows xs floats apart; yq: shadow_mmq_y_bytes(k, ntok) bytes
//   int    shadow_mmq_gemm(int type, const void* w, const void* yq, float* dst, int64_t rows, int64_t k, int64_t ntok, int64_t dst_stride, void* stream)
//                                                                                          dst[t * dst_stride + r] = sum_k w[r][k] * x[t][k]; w: GGUF layout, `rows` rows of k values
// type = the ggml_type number (Q4_K 12, Q5_K 13, Q6_K 14, IQ4_NL 20, IQ4_XS 23, ...).  Return 0 on success, nonzero for an unsupported type or a CUDA error.
#include "common.cuh"
#include "mmq.cuh"
#include "quantize.cuh"

#include <cstdio>
#include <cstdlib>
#include <cstdarg>
#include <mutex>

#define EXPORT extern "C" __declspec(dllexport)

// ---- the ggml-cuda.cu hooks
extern "C" void ggml_abort(const char * file, int line, const char * fmt, ...) {
    va_list ap; va_start(ap, fmt); fprintf(stderr, "shadow_mmq: ggml_abort %s:%d: ", file, line); vfprintf(stderr, fmt, ap); fprintf(stderr, "\n"); va_end(ap); exit(1);
}
void ggml_cuda_error(const char * stmt, const char * func, const char * file, int line, const char * msg) {
    fprintf(stderr, "shadow_mmq: CUDA error: %s in %s at %s:%d: %s\n", stmt, func, file, line, msg); exit(1);
}
static ggml_cuda_device_info g_info;
const ggml_cuda_device_info & ggml_cuda_info() {
    static std::once_flag once;
    std::call_once(once, []() {
        cudaDeviceProp p; cudaGetDeviceProperties(&p, 0);
        g_info.device_count = 1; g_info.physical_device_count = 1;
        auto & d = g_info.devices[0];
        d.cc = 100 * p.major + 10 * p.minor; d.nsm = p.multiProcessorCount; d.smpb = p.sharedMemPerBlock; d.smpbo = p.sharedMemPerBlockOptin;
        d.integrated = false; d.vmm = false; d.vmm_granularity = 0; d.total_vram = p.totalGlobalMem; d.warp_size = p.warpSize;
        d.supports_cooperative_launch = false; d.physical_device = 0; d.physical_share_count = 1; d.virtual_index = 0;
    });
    return g_info;
}
void ggml_cuda_set_device(int) {}
int ggml_cuda_get_device() { return 0; }
ggml_backend_cuda_context::~ggml_backend_cuda_context() {}

// the stream-k fixup scratch: one buffer, handed out again on every call (all calls are on one stream, so the uses are ordered)
struct scratch_pool : ggml_cuda_pool {
    void * buf = nullptr; size_t cap = 0;
    void * alloc(size_t size, size_t * actual) override {
        if (size > cap) { if (buf) cudaFree(buf); cap = size > (size_t) 16 << 20 ? size : (size_t) 16 << 20; if (cudaMalloc(&buf, cap) != cudaSuccess) { fprintf(stderr, "shadow_mmq: scratch alloc failed\n"); exit(1); } }
        *actual = cap; return buf;
    }
    void free(void *, size_t) override {}
};
std::unique_ptr<ggml_cuda_pool> ggml_backend_cuda_context::new_pool_for_device(int, int) { return std::make_unique<scratch_pool>(); }

static ggml_backend_cuda_context & ctx() { static ggml_backend_cuda_context c(0); return c; }

EXPORT int shadow_mmq_version() { return 1; }

EXPORT size_t shadow_mmq_y_bytes(int64_t k, int64_t ntok) {
    const int64_t k_pad = GGML_PAD(k, MATRIX_ROW_PADDING);
    return (size_t) ntok * k_pad * sizeof(block_q8_1_mmq) / QK8_1_MMQ + (size_t) 128 * sizeof(block_q8_1_mmq);
}

EXPORT int shadow_mmq_quantize(const float * x, void * yq, int64_t k, int64_t ntok, int64_t xs, int type, void * stream) {
    if (k % 4 != 0) return 2;
    const int64_t k_pad = GGML_PAD(k, MATRIX_ROW_PADDING);
    quantize_mmq_q8_1_cuda(x, nullptr, yq, (ggml_type) type, k, xs, xs * ntok, xs * ntok, k_pad, ntok, 1, 1, (cudaStream_t) stream);
    return cudaGetLastError() == cudaSuccess ? 0 : 3;
}

template <ggml_type T>
static void run(const mmq_args & a, cudaStream_t s) { mul_mat_q_case<T>(ctx(), a, s); }

EXPORT int shadow_mmq_gemm(int type, const void * w, const void * yq, float * dst, int64_t rows, int64_t k, int64_t ntok, int64_t dst_stride, void * stream) {
    const ggml_type t = (ggml_type) type;
    const int64_t qk = t == GGML_TYPE_IQ4_NL ? 32 : 256;   // the supported types: K-quants and IQ4_XS are 256-value blocks
    if (k % qk != 0) return 2;
    const int64_t k_pad = GGML_PAD(k, MATRIX_ROW_PADDING);
    const int64_t s01 = k / qk;                                                       // row stride in blocks
    const int64_t s12 = ntok * k_pad * sizeof(block_q8_1) / (QK8_1 * sizeof(int));    // the activation channel stride, in ints
    const mmq_args args = {
        (const char *) w, t, (const int *) yq, nullptr, nullptr, dst, nullptr,
        k, rows, ntok, s01, ntok, dst_stride,
        1, 1, s01 * rows, s12, dst_stride * ntok,
        1, 1, s01 * rows, s12, dst_stride * ntok,
        ntok, ntok};
    cudaStream_t s = (cudaStream_t) stream;
    switch (t) {
        case GGML_TYPE_Q4_K:  run<GGML_TYPE_Q4_K>(args, s); break;
        case GGML_TYPE_Q5_K:  run<GGML_TYPE_Q5_K>(args, s); break;
        case GGML_TYPE_Q6_K:  run<GGML_TYPE_Q6_K>(args, s); break;
        case GGML_TYPE_IQ4_XS: run<GGML_TYPE_IQ4_XS>(args, s); break;
        default: return 1;
    }
    return cudaGetLastError() == cudaSuccess ? 0 : 3;
}
