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
#include "mmvq.cuh"

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
// Per-device state (STATUS 113): a pair of sessions on two cards loads this one DLL in one process, each stage with its own primary context current on the calling thread.  Everything that was
// one static for device 0 is now looked up by `cudaGetDevice()` (the runtime follows the driver context the engine made current): the device properties, the ggml context and its
// stream-k scratch buffer, which must live in the memory of the device whose kernels use it.
static ggml_cuda_device_info g_info;
const ggml_cuda_device_info & ggml_cuda_info() {
    static std::once_flag once;
    std::call_once(once, []() {
        int n = 0; cudaGetDeviceCount(&n);
        if (n > GGML_CUDA_MAX_DEVICES) n = GGML_CUDA_MAX_DEVICES;
        g_info.device_count = n; g_info.physical_device_count = n;
        for (int i = 0; i < n; i++) {
            cudaDeviceProp p; cudaGetDeviceProperties(&p, i);
            auto & d = g_info.devices[i];
            d.cc = 100 * p.major + 10 * p.minor; d.nsm = p.multiProcessorCount; d.smpb = p.sharedMemPerBlock; d.smpbo = p.sharedMemPerBlockOptin;
            d.integrated = false; d.vmm = false; d.vmm_granularity = 0; d.total_vram = p.totalGlobalMem; d.warp_size = p.warpSize;
            d.supports_cooperative_launch = false; d.physical_device = i; d.physical_share_count = 1; d.virtual_index = i;
        }
    });
    return g_info;
}
void ggml_cuda_set_device(int device) { cudaSetDevice(device); }
int ggml_cuda_get_device() { int d = 0; cudaGetDevice(&d); return d; }
ggml_backend_cuda_context::~ggml_backend_cuda_context() {}

// the stream-k fixup scratch: one buffer a device, handed out again on every call (all of a device's calls are on one stream, so the uses are ordered)
struct scratch_pool : ggml_cuda_pool {
    int device = 0;
    void * buf = nullptr; size_t cap = 0;
    explicit scratch_pool(int dev) : device(dev) {}
    void * alloc(size_t size, size_t * actual) override {
        if (size > cap) {
            int prev = 0; cudaGetDevice(&prev); cudaSetDevice(device);
            if (buf) cudaFree(buf);
            cap = size > (size_t) 16 << 20 ? size : (size_t) 16 << 20;
            if (cudaMalloc(&buf, cap) != cudaSuccess) { fprintf(stderr, "shadow_mmq: scratch alloc failed\n"); exit(1); }
            cudaSetDevice(prev);
        }
        *actual = cap; return buf;
    }
    void free(void *, size_t) override {}
};
std::unique_ptr<ggml_cuda_pool> ggml_backend_cuda_context::new_pool_for_device(int device, int) { return std::make_unique<scratch_pool>(device); }

static ggml_backend_cuda_context & ctx() {
    static std::mutex mu;
    static ggml_backend_cuda_context * c[GGML_CUDA_MAX_DEVICES] = {};
    const int d = ggml_cuda_get_device();
    std::lock_guard<std::mutex> lock(mu);
    if (!c[d]) c[d] = new ggml_backend_cuda_context(d);
    return *c[d];
}

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

// ---- ggml's host-side helpers mmvq.cu links against.  Only `ggml_cuda_mul_mat_vec_q` (the ggml_tensor entry, which Shadow does not call) and `ggml_cuda_should_use_mmvq` use most of them;
// `ggml_blck_size` is read by the launchers.  Correct for the types the exports below accept, harmless stubs for the rest.
extern "C" {
int64_t ggml_blck_size(enum ggml_type t) {
    switch (t) {
        case GGML_TYPE_Q8_0: case GGML_TYPE_IQ4_NL: case GGML_TYPE_Q8_1: case GGML_TYPE_Q4_0: case GGML_TYPE_Q4_1: case GGML_TYPE_Q5_0: case GGML_TYPE_Q5_1: return 32;
        case GGML_TYPE_Q4_K: case GGML_TYPE_Q5_K: case GGML_TYPE_Q6_K: case GGML_TYPE_IQ4_XS: case GGML_TYPE_Q2_K: case GGML_TYPE_Q3_K: return 256;
        default: return 1;
    }
}
size_t ggml_type_size(enum ggml_type t) {
    switch (t) {
        case GGML_TYPE_F32: return 4;
        case GGML_TYPE_Q8_0: return 34;
        case GGML_TYPE_Q8_1: return 36;
        case GGML_TYPE_IQ4_NL: return 18;
        case GGML_TYPE_Q4_K: return 144;
        case GGML_TYPE_Q5_K: return 176;
        case GGML_TYPE_Q6_K: return 210;
        case GGML_TYPE_IQ4_XS: return 136;
        default: return 0;
    }
}
bool ggml_is_quantized(enum ggml_type t) { return t != GGML_TYPE_F32 && t != GGML_TYPE_F16 && t != GGML_TYPE_BF16; }
int64_t ggml_nelements(const struct ggml_tensor * x) { return x->ne[0] * x->ne[1] * x->ne[2] * x->ne[3]; }
size_t ggml_nbytes(const struct ggml_tensor * x) { return (size_t) (ggml_nelements(x) / ggml_blck_size(x->type) * ggml_type_size(x->type)); }
bool ggml_is_contiguous(const struct ggml_tensor *) { return true; }
bool ggml_is_contiguously_allocated(const struct ggml_tensor *) { return true; }
bool ggml_are_same_stride(const struct ggml_tensor *, const struct ggml_tensor *) { return true; }
size_t ggml_backend_buffer_get_alloc_size(ggml_backend_buffer_t, const struct ggml_tensor * t) { return ggml_nbytes(t); }
enum ggml_backend_buffer_usage ggml_backend_buffer_get_usage(ggml_backend_buffer_t) { return GGML_BACKEND_BUFFER_USAGE_ANY; }
}

// llama.cpp's mul_mat_vec_q (ggml-cuda/mmvq.cu) for 1..=8 inputs, the decode / verify-pass projections (STATUS 109): the inputs are quantized to plain q8_1 blocks
// (`shadow_mmvq_quantize`), then one pass over the weights multiplies them with all of the inputs (`shadow_mmvq_gemv`).  Types: Q8_0 8, Q4_K 12, Q5_K 13, Q6_K 14, IQ4_NL 20, IQ4_XS 23.
//   size_t shadow_mmvq_y_bytes(int64_t k, int64_t ntok)
//   int    shadow_mmvq_quantize(const float* x, void* yq, int64_t k, int64_t ntok, int64_t xs, void* stream)
//   int    shadow_mmvq_gemv(int type, const void* w, const void* yq, float* dst, int64_t rows, int64_t k, int64_t ntok, int64_t dst_stride, void* stream)
EXPORT size_t shadow_mmvq_y_bytes(int64_t k, int64_t ntok) {
    const int64_t k_pad = GGML_PAD(k, MATRIX_ROW_PADDING);
    return (size_t) ntok * (k_pad / QK8_1) * sizeof(block_q8_1) + 4096;
}

EXPORT int shadow_mmvq_quantize(const float * x, void * yq, int64_t k, int64_t ntok, int64_t xs, void * stream) {
    if (k % QK8_1 != 0 || ntok < 1 || ntok > MMVQ_MAX_BATCH_SIZE) return 2;
    const int64_t k_pad = GGML_PAD(k, MATRIX_ROW_PADDING);
    quantize_row_q8_1_cuda(x, nullptr, yq, GGML_TYPE_F32, k, xs, xs * ntok, xs * ntok, k_pad, ntok, 1, 1, (cudaStream_t) stream);
    return cudaGetLastError() == cudaSuccess ? 0 : 3;
}

EXPORT int shadow_mmvq_gemv(int type, const void * w, const void * yq, float * dst, int64_t rows, int64_t k, int64_t ntok, int64_t dst_stride, void * stream) {
    if (ntok < 1 || ntok > MMVQ_MAX_BATCH_SIZE || k % QK8_1 != 0 || dst_stride < rows) return 2;
    switch ((ggml_type) type) {
        case GGML_TYPE_Q8_0: case GGML_TYPE_Q4_K: case GGML_TYPE_Q5_K: case GGML_TYPE_Q6_K: case GGML_TYPE_IQ4_NL: case GGML_TYPE_IQ4_XS: break;
        default: return 1;
    }
    ggml_tensor src0 = {}, src1 = {}, d = {};
    src0.type = (ggml_type) type; src0.ne[0] = k; src0.ne[1] = rows;
    src1.type = GGML_TYPE_Q8_1; src1.ne[0] = k; src1.ne[1] = ntok;
    d.type = GGML_TYPE_F32; d.ne[0] = dst_stride; d.ne[1] = ntok;
    const int64_t k_pad = GGML_PAD(k, MATRIX_ROW_PADDING);
    ggml_cuda_op_mul_mat_vec_q(ctx(), &src0, &src1, &d, (const char *) w, nullptr, (const char *) yq, dst, 0, rows, ntok, k_pad, (cudaStream_t) stream);
    return cudaGetLastError() == cudaSuccess ? 0 : 3;
}

// grouped experts (the MoE gate/up projections of a long prompt)
#include "moe_part.inc"
