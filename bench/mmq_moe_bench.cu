// Times native/mmq/shadow_mmq.dll's `shadow_mmq_moe` (llama.cpp's int8 MMQ tile loop driven by the planner's group tables, quantize + product for the gate and up matrices) on synthetic
// tables in the planner's layout (the same ones as mma_bench.cu builds), to see whether its per-launch time in the engine (5-7 ms) is the kernel's or the engine's.
//
//   mmq_moe_bench.exe [ty rows K m dist chunk]    ty 18 IQ3_XXS | 21 IQ3_S | 22 IQ2_S; dist u | r; chunk 32 | 48 | 64
#include <windows.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <vector>
#include <random>
#include <algorithm>
#include <cmath>
#include <cuda_runtime.h>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { printf("CUDA error %s at %d: %s\n", #x, __LINE__, cudaGetErrorString(e)); exit(1); } } while (0)

typedef size_t (*BytesFn)(int64_t, int64_t, int64_t);
typedef int (*MoeFn)(int, unsigned long long, unsigned long long, int64_t, int64_t, int64_t, unsigned long long, int64_t, int64_t, int64_t, unsigned long long, int64_t, unsigned long long, unsigned long long);

int main(int argc, char** argv) {
    const int ty = argc > 1 ? atoi(argv[1]) : 18;
    const unsigned int rows = argc > 2 ? atoi(argv[2]) : 640, K = argc > 3 ? atoi(argv[3]) : 2560, m = argc > 4 ? atoi(argv[4]) : 2048;
    const char dist = argc > 5 ? argv[5][0] : 'r';
    const unsigned int CH = argc > 6 ? atoi(argv[6]) : 32;
    unsigned int qk = 256, bytes = ty == 18 ? 98 : ty == 21 ? 110 : 82;
    const unsigned int NE = 512, TOPK = 10, h = K, ff = rows;
    const unsigned int rb = (K / qk) * bytes;
    const size_t mat_bytes = (size_t) rows * rb;

    HMODULE lib = LoadLibraryA("J:\\Shadow\\native\\mmq\\shadow_mmq.dll");
    if (!lib) { printf("no dll\n"); return 1; }
    BytesFn moe_bytes = (BytesFn) GetProcAddress(lib, "shadow_mmq_moe_bytes");
    MoeFn moe = (MoeFn) GetProcAddress(lib, "shadow_mmq_moe");
    if (!moe || !moe_bytes) { printf("no entry\n"); return 1; }

    std::mt19937 rng(7);
    std::vector<double> wgt(NE);
    std::lognormal_distribution<double> ln(0.0, dist == 'r' ? 1.1 : 0.0);
    double tot = 0;
    for (auto& w : wgt) { w = ln(rng); tot += w; }
    std::vector<unsigned int> cnt(NE);
    unsigned int sum = 0;
    for (unsigned int e = 0; e < NE; e++) { cnt[e] = (unsigned int) (wgt[e] / tot * m * TOPK); sum += cnt[e]; }
    for (unsigned int e = 0; sum < m * TOPK; e = (e + 1) % NE) { cnt[e]++; sum++; }
    for (auto& c : cnt) c = std::min<unsigned int>(c, m);
    struct G { unsigned int e, n; };
    std::vector<G> groups;
    for (unsigned int e = 0; e < NE; e++)
        for (unsigned int t = 0; t < cnt[e]; t += CH) groups.push_back({e, std::min<unsigned int>(CH, cnt[e] - t)});
    const unsigned int n = (unsigned int) groups.size();
    unsigned long long pairs = 0; for (auto c : cnt) pairs += c;
    printf("type %d chunk %u: rows %u K %u m %u %c counts: groups %u, pairs %llu, fill %.0f%%\n", ty, CH, rows, K, m, dist, n, pairs, 100.0 * pairs / ((double) n * CH));

    std::vector<uint8_t> hw((size_t) NE * 2 * mat_bytes);
    for (auto& b : hw) b = (uint8_t) rng();
    for (size_t i = 0; i + 1 < hw.size(); i += bytes) { hw[i] = 0x66; hw[i + 1] = 0x20; }
    uint8_t* dw; CK(cudaMalloc(&dw, hw.size())); CK(cudaMemcpy(dw, hw.data(), hw.size(), cudaMemcpyHostToDevice));
    std::vector<float> hx((size_t) m * h);
    std::normal_distribution<float> nd(0.f, 1.f);
    for (auto& v : hx) v = nd(rng);
    float *dx, *dy; CK(cudaMalloc(&dx, hx.size() * 4)); CK(cudaMemcpy(dx, hx.data(), hx.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMalloc(&dy, (size_t) 2 * m * TOPK * ff * 4));
    std::vector<uint64_t> wtab(2 * n);
    const unsigned int rec = 1 + 2 * CH;
    std::vector<unsigned int> gtab((size_t) 2 * n * rec, 0);
    std::vector<unsigned int> next(NE, 0);
    unsigned int pair = 0;
    for (unsigned int g = 0; g < n; g++) {
        wtab[g] = (uint64_t) (dw + ((size_t) groups[g].e * 2 + 0) * mat_bytes);
        wtab[n + g] = (uint64_t) (dw + ((size_t) groups[g].e * 2 + 1) * mat_bytes);
        for (int half = 0; half < 2; half++) {
            unsigned int* r = &gtab[(size_t) (half * n + g) * rec];
            r[0] = groups[g].n;
            for (unsigned int t = 0; t < groups[g].n; t++) {
                const unsigned int tok = (unsigned int) ((groups[g].e * 2654435761u + (next[groups[g].e] + t) * 40503u) % m);
                r[1 + t] = tok;
                r[1 + CH + t] = (pair + t) + half * m * TOPK;
            }
        }
        pair += groups[g].n;
        next[groups[g].e] += groups[g].n;
    }
    uint64_t* dwt; unsigned int* dgt;
    CK(cudaMalloc(&dwt, wtab.size() * 8)); CK(cudaMemcpy(dwt, wtab.data(), wtab.size() * 8, cudaMemcpyHostToDevice));
    CK(cudaMalloc(&dgt, gtab.size() * 4)); CK(cudaMemcpy(dgt, gtab.data(), gtab.size() * 4, cudaMemcpyHostToDevice));
    const size_t sb = moe_bytes(K, n, CH);
    void* scratch; CK(cudaMalloc(&scratch, sb));
    auto run = [&]() { return moe(ty, (unsigned long long) dwt, (unsigned long long) dgt, 2 * n, n, CH, (unsigned long long) dx, h, K, rows, (unsigned long long) dy, ff,
                                  (unsigned long long) scratch, 0ull); };
    int rc = run(); CK(cudaDeviceSynchronize());
    if (rc) { printf("moe returned %d\n", rc); return 1; }
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    const int reps = 20;
    cudaEventRecord(e0);
    for (int i = 0; i < reps; i++) run();
    cudaEventRecord(e1); CK(cudaEventSynchronize(e1));
    float ms; cudaEventElapsedTime(&ms, e0, e1); ms /= reps;
    const double flop = 2.0 * rows * K * (double) pairs * 2;
    printf("quantize + gate/up products: %.3f ms (%.1f TFLOP/s of useful work, %.2f us a group)\n", ms, flop / ms / 1e9, ms * 1e3 / (2.0 * n));
    return 0;
}
