// Standalone timing of llama.cpp's MMQ (int8 tensor-core) on a MoE-shaped product: NEXP experts, TOK tokens each, weights IQ4_NL [ROWS x K], activations f32 -> q8_1_mmq.
// The ggml runtime hooks MMQ needs (device info, device pool) are stubbed here; the kernels are ggml-cuda's own (MIT, llama.cpp 3cf03257f).
//   mmq_bench.exe [rows K tok nexp]
#include "common.cuh"
#include "mmq.cuh"
#include "quantize.cuh"

#include <cstdio>
#include <cstdlib>
#include <cstdarg>
#include <vector>
#include <random>
#include <cmath>

// ---- stubs for ggml-cuda.cu
extern "C" void ggml_abort(const char * file, int line, const char * fmt, ...) {
    va_list ap; va_start(ap, fmt); fprintf(stderr, "ggml_abort %s:%d: ", file, line); vfprintf(stderr, fmt, ap); fprintf(stderr, "\n"); va_end(ap); exit(1);
}
static ggml_cuda_device_info g_info;
static bool g_info_init = false;
const ggml_cuda_device_info & ggml_cuda_info() {
    if (!g_info_init) {
        cudaDeviceProp p; cudaGetDeviceProperties(&p, 0);
        g_info.device_count = 1; g_info.physical_device_count = 1;
        auto & d = g_info.devices[0];
        d.cc = 100 * p.major + 10 * p.minor + GGML_CUDA_CC_OFFSET_AMD * 0; d.nsm = p.multiProcessorCount; d.smpb = p.sharedMemPerBlock; d.smpbo = p.sharedMemPerBlockOptin;
        d.integrated = false; d.vmm = false; d.vmm_granularity = 0; d.total_vram = p.totalGlobalMem; d.warp_size = p.warpSize;
        d.supports_cooperative_launch = false; d.physical_device = 0; d.physical_share_count = 1; d.virtual_index = 0;
        g_info_init = true;
    }
    return g_info;
}
void ggml_cuda_error(const char * stmt, const char * func, const char * file, int line, const char * msg) { fprintf(stderr, "CUDA error: %s in %s at %s:%d: %s\n", stmt, func, file, line, msg); exit(1); }
ggml_backend_cuda_context::~ggml_backend_cuda_context() {}
void ggml_cuda_set_device(int) {}
int ggml_cuda_get_device() { return 0; }
struct simple_pool : ggml_cuda_pool {
    void * alloc(size_t size, size_t * actual) override { void * p = nullptr; cudaMalloc(&p, size); *actual = size; return p; }
    void free(void * p, size_t) override { cudaFree(p); }
};
std::unique_ptr<ggml_cuda_pool> ggml_backend_cuda_context::new_pool_for_device(int, int) { return std::make_unique<simple_pool>(); }

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { printf("CUDA error %s at %d: %s\n", #x, __LINE__, cudaGetErrorString(e)); exit(1); } } while (0)

int main(int argc, char ** argv) {
    const int64_t rows = argc > 1 ? atoi(argv[1]) : 2560, K = argc > 2 ? atoi(argv[2]) : 640, tok = argc > 3 ? atoi(argv[3]) : 40, nexp = argc > 4 ? atoi(argv[4]) : 512;
    const ggml_type type = BENCH_TYPE;
    const int64_t qk = BENCH_QK, bytes_blk = sizeof(BENCH_BLOCK);
    const int64_t blocks_per_row = K / qk;
    printf("%s: rows %lld K %lld tokens/expert %lld experts %lld; block %lld bytes; cc %d, %d SMs\n", BENCH_NAME, (long long) rows, (long long) K, (long long) tok, (long long) nexp, (long long) bytes_blk,
           ggml_cuda_info().devices[0].cc, ggml_cuda_info().devices[0].nsm);

    // weights: random nibbles, d = 0.01 as half
    std::mt19937 rng(1);
    std::vector<uint8_t> hx((size_t) nexp * rows * blocks_per_row * bytes_blk);
    for (auto & b : hx) b = (uint8_t) rng();
    const uint16_t dh = 0x2066;   // ~0.0156 in half
    for (size_t i = 0; i < hx.size(); i += bytes_blk) { hx[i] = dh & 0xff; hx[i + 1] = dh >> 8; }
    // activations f32 [nexp][tok][K]
    std::vector<float> hy((size_t) nexp * tok * K);
    std::normal_distribution<float> nd(0.f, 1.f);
    for (auto & v : hy) v = nd(rng);

    char * dx; float * dy; float * ddst; char * dq;
    CK(cudaMalloc(&dx, hx.size())); CK(cudaMemcpy(dx, hx.data(), hx.size(), cudaMemcpyHostToDevice));
    CK(cudaMalloc(&dy, hy.size() * 4)); CK(cudaMemcpy(dy, hy.data(), hy.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMalloc(&ddst, (size_t) nexp * tok * rows * 4));
    const int64_t K_pad = GGML_PAD(K, MATRIX_ROW_PADDING);
    const size_t nq = (size_t) nexp * tok * K_pad * sizeof(block_q8_1_mmq) / QK8_1_MMQ + (size_t) 128 * sizeof(block_q8_1_mmq);
    CK(cudaMalloc(&dq, nq)); CK(cudaMemset(dq, 0, nq));

    ggml_backend_cuda_context ctx(0);
    cudaStream_t stream = 0;

    const int64_t s01 = blocks_per_row, s02 = rows * blocks_per_row;
    const int64_t s12 = tok * K_pad * sizeof(block_q8_1) / (QK8_1 * sizeof(int)), s13 = nexp * s12;
    const mmq_args args = {
        dx, type, (const int *) dq, nullptr, nullptr, ddst, nullptr,
        K, rows, tok, s01, tok, rows,
        nexp, nexp, s02, s12, (int64_t) tok * rows,
        1, 1, s02 * nexp, s13, (int64_t) tok * rows * nexp,
        tok, tok};

    auto run = [&]() {
        quantize_mmq_q8_1_cuda(dy, nullptr, dq, type, K, K, (int64_t) tok * K, (int64_t) nexp * tok * K, K_pad, tok, nexp, 1, stream);
        mul_mat_q_case<BENCH_TYPE>(ctx, args, stream);
    };
    run(); CK(cudaGetLastError()); CK(cudaDeviceSynchronize());

    // timing: quantize alone, then the product alone, then both
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    const int reps = 20;
    auto time = [&](auto fn) { cudaEventRecord(e0); for (int i = 0; i < reps; ++i) fn(); cudaEventRecord(e1); cudaEventSynchronize(e1); float ms; cudaEventElapsedTime(&ms, e0, e1); return ms / reps; };
    float t_q = time([&]() { quantize_mmq_q8_1_cuda(dy, nullptr, dq, type, K, K, (int64_t) tok * K, (int64_t) nexp * tok * K, K_pad, tok, nexp, 1, stream); });
    float t_m = time([&]() { mul_mat_q_case<BENCH_TYPE>(ctx, args, stream); });
    float t_a = time(run);
    const double flop = 2.0 * rows * K * tok * nexp;
    printf("quantize %.3f ms, product %.3f ms, both %.3f ms; product %.1f TFLOP/s, both %.1f TFLOP/s (%.1f GFLOP)\n", t_q, t_m, t_a, flop / t_m / 1e9, flop / t_a / 1e9, flop / 1e9);

#if BENCH_IS_IQ4NL
    // check against a plain float reference on a few channels
    std::vector<float> hd((size_t) nexp * tok * rows);
    CK(cudaMemcpy(hd.data(), ddst, hd.size() * 4, cudaMemcpyDeviceToHost));
    static const int8_t kv[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};
    double max_err = 0, max_ref = 0, sum_sq_err = 0, sum_sq_ref = 0;
    for (int ch : std::initializer_list<int>{0, (int) (nexp / 2), (int) (nexp - 1)}) {
        for (int t = 0; t < std::min<int>(tok, 8); ++t) {
            for (int r : {0, 1, 7, (int) rows / 2, (int) rows - 1}) {
                double acc = 0;
                const uint8_t * w = &hx[((size_t) ch * rows + r) * blocks_per_row * bytes_blk];
                for (int64_t b = 0; b < blocks_per_row; ++b) {
                    const uint8_t * blk = w + b * bytes_blk;
                    uint16_t h = blk[0] | (blk[1] << 8);
                    // half -> float
                    int e = (h >> 10) & 31, m = h & 1023; float d = e == 0 ? m / 1024.f / 16384.f : ldexpf(1.f + m / 1024.f, e - 15);
                    for (int i = 0; i < 16; ++i) {
                        acc += (double) d * kv[blk[2 + i] & 15] * hy[((size_t) ch * tok + t) * K + b * 32 + i];
                        acc += (double) d * kv[blk[2 + i] >> 4] * hy[((size_t) ch * tok + t) * K + b * 32 + 16 + i];
                    }
                }
                const double got = hd[((size_t) ch * tok + t) * rows + r];
                max_err = std::max(max_err, std::fabs(got - acc)); max_ref = std::max(max_ref, std::fabs(acc));
                sum_sq_err += (got - acc) * (got - acc); sum_sq_ref += acc * acc;
            }
        }
    }
    printf("vs float reference: max |err| %.4f (max |ref| %.2f), relative RMS error %.4f\n", max_err, max_ref, std::sqrt(sum_sq_err / sum_sq_ref));
#endif
    return 0;
}
