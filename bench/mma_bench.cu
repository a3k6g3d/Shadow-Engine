// Times the engine's own grouped MoE kernels (kernels/matvec.cu `mma_grouped_*` / `mma_grouped64_*`) on synthetic tables in the layout the planner writes, so a variant of the kernel
// can be measured in seconds instead of in an engine run: NE experts (random quantized bytes), a gate and an up matrix each, groups of CH tokens (a record: count, CH input rows, CH
// output rows), the launch of `routed_grouped` (grid (rows / 64, 2 * groups), 128 threads, one launch for gate + up).
//
//   nvcc -O3 -arch=sm_120 -I<kernels dir> -DKERNEL_FILE=\"matvec.cu\" -DQT=IQ3XXS -DENTRY=mma_grouped_iq3_xxs -DCH=32 -o mma_bench.exe mma_bench.cu
//   mma_bench.exe [rows K m dist]      rows 640, K 2560, m rows of the sub-chunk (2048), dist: u (uniform counts) | r (ragged: lognormal, median ~ 0.65 x the mean)
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <vector>
#include <random>
#include <algorithm>
#include <cmath>
#include "prelude.cuh"
#include "iq_tables.cuh"
#include "quants.cuh"
#include KERNEL_FILE

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { printf("CUDA error %s at %d: %s\n", #x, __LINE__, cudaGetErrorString(e)); exit(1); } } while (0)

int main(int argc, char** argv) {
    const unsigned int rows = argc > 1 ? atoi(argv[1]) : 640, K = argc > 2 ? atoi(argv[2]) : 2560, m = argc > 3 ? atoi(argv[3]) : 2048;
    const char dist = argc > 4 ? argv[4][0] : 'r';
    const unsigned int NE = 512, TOPK = 10, h = K, ff = rows;
    const unsigned int rb = (K / QT::QK) * QT::BYTES;
    const size_t mat_bytes = (size_t) rows * rb;
    printf("%s ch %d: rows %u K %u (%u bytes a row, %.2f MB a matrix), m %u, %c counts\n", ENTRY_NAME, CH, rows, K, rb, mat_bytes / 1e6, m, dist);

    // counts per expert, summing to m * TOPK pairs
    std::mt19937 rng(7);
    std::vector<double> wgt(NE);
    std::lognormal_distribution<double> ln(0.0, dist == 'r' ? 1.1 : 0.0);
    double tot = 0;
    for (auto& w : wgt) { w = ln(rng); tot += w; }
    std::vector<unsigned int> cnt(NE);
    unsigned int sum = 0;
    for (unsigned int e = 0; e < NE; e++) { cnt[e] = (unsigned int) (wgt[e] / tot * m * TOPK); sum += cnt[e]; }
    for (unsigned int e = 0; sum < m * TOPK; e = (e + 1) % NE) { cnt[e]++; sum++; }
    for (auto& c : cnt) c = std::min<unsigned int>(c, m);   // an expert sees a token once
    // groups
    struct G { unsigned int e, n; };
    std::vector<G> groups;
    for (unsigned int e = 0; e < NE; e++)
        for (unsigned int t = 0; t < cnt[e]; t += CH) groups.push_back({e, std::min<unsigned int>(CH, cnt[e] - t)});
    const unsigned int n = (unsigned int) groups.size();
    unsigned long long pairs = 0; for (auto c : cnt) pairs += c;
    printf("groups %u, pairs %llu, fill %.0f%%\n", n, pairs, 100.0 * pairs / ((double) n * CH));

    // weights: random bytes, a half-precision scale where the formats keep one first
    std::vector<uint8_t> hw((size_t) NE * 2 * mat_bytes);
    for (auto& b : hw) b = (uint8_t) rng();
    for (size_t i = 0; i + 1 < hw.size(); i += QT::BYTES) { hw[i] = 0x66; hw[i + 1] = 0x20; }
    uint8_t* dw; CK(cudaMalloc(&dw, hw.size())); CK(cudaMemcpy(dw, hw.data(), hw.size(), cudaMemcpyHostToDevice));
    // x: m rows of h floats; y: 2 * m * TOPK rows of ff floats (gate then up halves)
    std::vector<float> hx((size_t) m * h);
    std::normal_distribution<float> nd(0.f, 1.f);
    for (auto& v : hx) v = nd(rng);
    float *dx, *dy; CK(cudaMalloc(&dx, hx.size() * 4)); CK(cudaMemcpy(dx, hx.data(), hx.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMalloc(&dy, (size_t) 2 * m * TOPK * ff * 4));
    // tables: wtab[2n] (gate groups then up groups), gtab records of 1 + 2 CH u32
    std::vector<uint64_t> wtab(2 * n);
    const unsigned int rec = 1 + 2 * CH;
    std::vector<unsigned int> gtab((size_t) 2 * n * rec, 0);
    std::vector<unsigned int> next(NE, 0);
    unsigned int pair = 0;   // every (token, expert) pair its own output row, so the output is race free
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

    dim3 grid((rows + 63) / 64, 2 * n, 1);
    auto launch = [&]() { ENTRY<<<grid, 128>>>(dwt, dgt, dx, h, dy, ff, rows, K); };
    launch(); CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
    // a checksum of the output, to compare variants bit for bit
    std::vector<float> hy((size_t) 2 * m * TOPK * ff);
    CK(cudaMemcpy(hy.data(), dy, hy.size() * 4, cudaMemcpyDeviceToHost));
    uint64_t hsh = 1469598103934665603ull;
    for (float v : hy) { uint32_t b; memcpy(&b, &v, 4); hsh = (hsh ^ b) * 1099511628211ull; }
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    const int reps = 20;
    cudaEventRecord(e0);
    for (int i = 0; i < reps; i++) {
        // alternate the weights' base so the L2 does not keep them (a 2 x 512 expert set is far bigger than it anyway)
        launch();
    }
    cudaEventRecord(e1); CK(cudaEventSynchronize(e1));
    float ms; cudaEventElapsedTime(&ms, e0, e1); ms /= reps;
    const double flop = 2.0 * rows * K * (double) pairs * 2;
    printf("launch %.3f ms (%.1f TFLOP/s of useful work, %.2f us a group), output hash %016llx\n", ms, flop / ms / 1e9, ms * 1e3 / (2.0 * n), (unsigned long long) hsh);
    return 0;
}
