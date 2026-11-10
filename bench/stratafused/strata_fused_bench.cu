// Times Strata 0.1.44's fused int8 tensor-core MoE kernels (MIT; src/prefill/moe_fused_iq.cu, moe_fused.cu) on experts shaped like Shadow's, so they can be compared with Shadow's own
// grouped kernels (bench/mma_bench.cu) before anything is ported (docs/PLAN_RAM_TIER_AND_LEVERS_2026-11-09.md, Phase C1).
//
//   build.bat                          (strata_fused_bench.exe)
//   strata_fused_bench.exe [gu down m dist]   gu: iq2_s | iq3_xxs | iq3_s;  down: iq4_nl;  m: tokens of the chunk (2048);  dist: u | r (ragged, as mma_bench)
//
// One layer: NE = 512 experts, k = 10 per token, n_embd 2560, n_ff 640; an expert's blob = [gate 640 rows | up 640 rows | down 2560 rows] in the raw GGUF block layout (random bytes: the
// timing does not depend on the values).  Reported: the activation quantization, the grouping, and the experts (gate/up + SwiGLU + int8 H + down), per layer, in ms.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <vector>
#include <random>
#include <algorithm>
#include <cmath>
#include <cuda_runtime.h>
#include "strata/prefill/moe_fused_iq.hpp"

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { printf("CUDA error %s at %d: %s\n", #x, __LINE__, cudaGetErrorString(e)); exit(1); } } while (0)
using namespace strata::prefill::fused;

int main(int argc, char** argv) {
    _putenv_s("STRATA_PF_FUSED", "1");
    const char* gu = argc > 1 ? argv[1] : "iq3_xxs";
    const char* dn = argc > 2 ? argv[2] : "iq4_nl";
    const int m = argc > 3 ? atoi(argv[3]) : 2048;
    const char dist = argc > 4 ? argv[4][0] : 'r';
    const int NE = 512, K = 10, H = 2560, FF = 640;
    int gu_type, gu_bytes, gu_vals;           // ggml type, bytes per block, values per block
    if (!strcmp(gu, "iq2_s")) { gu_type = 22; gu_bytes = 82; gu_vals = 256; }
    else if (!strcmp(gu, "iq3_xxs")) { gu_type = 18; gu_bytes = 98; gu_vals = 256; }
    else if (!strcmp(gu, "iq3_s")) { gu_type = 21; gu_bytes = 110; gu_vals = 256; }
    else { printf("gate/up type %s not handled\n", gu); return 1; }
    const int d_type = 20, d_bytes = 18, d_vals = 32;   // IQ4_NL
    NativeGeom g;
    g.gu_type = gu_type; g.d_type = d_type;
    g.gu_row = (size_t) (H / gu_vals) * gu_bytes;
    g.d_row = (size_t) (FF / d_vals) * d_bytes;
    g.up_off = g.gu_row * FF;
    g.down_off = 2 * g.gu_row * FF;
    const size_t blob = g.down_off + g.d_row * H;
    const size_t stride = (blob + 255) / 256 * 256;
    printf("gate/up %s (%zu B a row), down %s (%zu B a row), blob %.2f MB, m %d tokens, k %d, %d experts, %c counts\n", gu, g.gu_row, dn, g.d_row, blob / 1e6, m, K, NE, dist);
    if (!native_supported(gu_type, d_type)) { printf("native_supported(%d, %d) is false (needs STRATA_PF_FUSED=1 and sm_80+)\n", gu_type, d_type); return 1; }

    // routing: each token takes K distinct experts, drawn with lognormal weights (ragged) or uniformly
    std::mt19937 rng(7);
    std::lognormal_distribution<double> ln(0.0, dist == 'r' ? 1.1 : 0.0);
    std::vector<double> w(NE);
    for (auto& x : w) x = ln(rng);
    std::discrete_distribution<int> pick(w.begin(), w.end());
    const int64_t n = (int64_t) m * K;
    std::vector<int32_t> ids(n);
    for (int t = 0; t < m; t++) {
        int chosen[16]; int c = 0;
        while (c < K) {
            int e = pick(rng); bool dup = false;
            for (int i = 0; i < c; i++) dup |= chosen[i] == e;
            if (!dup) chosen[c++] = e;
        }
        for (int i = 0; i < K; i++) ids[(size_t) t * K + i] = chosen[i];
    }
    std::vector<int> cnt(NE, 0);
    for (auto e : ids) cnt[e]++;
    int mx = 0, nz = 0; for (int c : cnt) { mx = std::max(mx, c); nz += c > 0; }
    printf("routed rows %lld, experts used %d, busiest %d, mean %.1f\n", (long long) n, nz, mx, (double) n / nz);

    // weights: NE blobs of random bytes, a sane half-precision scale at the start of each block
    std::vector<uint8_t> hw((size_t) NE * stride);
    for (auto& b : hw) b = (uint8_t) rng();
    for (int e = 0; e < NE; e++) {   // every block's half-precision scale (the first two bytes of a block in all four formats) = 0x2066 (~0.016)
        uint8_t* p = hw.data() + (size_t) e * stride;
        for (size_t o = 0; o < 2 * g.gu_row * FF; o += gu_bytes) { p[o] = 0x66; p[o + 1] = 0x20; }
        for (size_t o = g.down_off; o < blob; o += d_bytes) { p[o] = 0x66; p[o + 1] = 0x20; }
    }
    uint8_t* dw; CK(cudaMalloc(&dw, hw.size())); CK(cudaMemcpy(dw, hw.data(), hw.size(), cudaMemcpyHostToDevice));
    std::vector<float> hx((size_t) m * H);
    std::normal_distribution<float> nd(0.f, 1.f);
    for (auto& v : hx) v = nd(rng);
    float* dx; CK(cudaMalloc(&dx, hx.size() * 4)); CK(cudaMemcpy(dx, hx.data(), hx.size() * 4, cudaMemcpyHostToDevice));
    int32_t *dids, *dslot, *dsrc;
    CK(cudaMalloc(&dids, n * 4)); CK(cudaMemcpy(dids, ids.data(), n * 4, cudaMemcpyHostToDevice));
    CK(cudaMalloc(&dslot, n * 4)); CK(cudaMalloc(&dsrc, n * 4));
    void *xa, *scratch, *ha; float* dm;
    CK(cudaMalloc(&xa, act_bytes(m, H)));
    CK(cudaMalloc(&scratch, group_bytes(n, NE)));
    CK(cudaMalloc(&ha, act_bytes(n, FF)));
    CK(cudaMalloc((void**) &dm, (size_t) n * H * 4));

    cudaEvent_t ev[5]; for (auto& e : ev) cudaEventCreate(&e);
    auto run = [&](float* t) {
        cudaEventRecord(ev[0]);
        quantize_act_native(dx, m, H, xa, nullptr);
        cudaEventRecord(ev[1]);
        group(dids, n, K, NE, scratch, dslot, dsrc, nullptr);
        cudaEventRecord(ev[2]);
        for (int e0 = 0; e0 < NE; e0 += kMaxBatch) {
            Batch b; b.e0 = e0; b.e1 = std::min(NE, e0 + kMaxBatch);
            for (int e = b.e0; e < b.e1; e++) b.blob[e - e0] = dw + (size_t) e * stride;
            experts_native(b, g, NE, n, scratch, xa, dsrc, ha, dm, nullptr);
        }
        cudaEventRecord(ev[3]);
        CK(cudaEventSynchronize(ev[3]));
        cudaEventElapsedTime(&t[0], ev[0], ev[1]); cudaEventElapsedTime(&t[1], ev[1], ev[2]); cudaEventElapsedTime(&t[2], ev[2], ev[3]);
    };
    float t[3];
    run(t); CK(cudaGetLastError());
    // a checksum of the down output, to see that the kernels ran (not a correctness test)
    std::vector<float> hy((size_t) n * H);
    CK(cudaMemcpy(hy.data(), dm, hy.size() * 4, cudaMemcpyDeviceToHost));
    double s = 0; size_t nan = 0; for (float v : hy) { if (v != v) nan++; else s += std::fabs(v); }
    printf("down output: mean |y| %.4g, %zu NaN\n", s / hy.size(), nan);
    const int reps = 10;
    double a = 0, b = 0, c = 0;
    for (int i = 0; i < reps; i++) { run(t); a += t[0]; b += t[1]; c += t[2]; }
    printf("per layer, %d tokens: quantize %.3f ms, group %.3f ms, experts (gate/up + SwiGLU + down) %.3f ms\n", m, a / reps, b / reps, c / reps);
    const double flop = 2.0 * (2.0 * FF * H + (double) H * FF) * n;
    printf("useful work %.1f TFLOP/s on the experts; per token %.1f us\n", flop / (c / reps) / 1e9, (c / reps) * 1e3 / m);
    return 0;
}
