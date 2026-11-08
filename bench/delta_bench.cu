// The prompt path's DeltaNet recurrence kernels on synthetic inputs of the model's shape (48 value heads, 16 key heads, head dimension 128, one sub-chunk of m tokens): checks that
// `delta_steps128_kh` gives the same bits as `delta_steps128_f4` (outputs and the state after) and times both, so a variant can be measured in seconds.
//
//   nvcc -O3 -arch=sm_120 -std=c++17 -I<kernels dir> -o delta_bench.exe delta_bench.cu        (bench/mmabench/build_delta.bat)
//   delta_bench.exe [m]
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <vector>
#include <random>
#include <cmath>
#include "prelude.cuh"
#include "iq_tables.cuh"
#include "quants.cuh"
#include "forward.cu"

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { printf("CUDA error %s at %d: %s\n", #x, __LINE__, cudaGetErrorString(e)); exit(1); } } while (0)

int main(int argc, char** argv) {
    const unsigned int m = argc > 1 ? atoi(argv[1]) : 2048;
    const unsigned int kh = 16, vh = 48, d = 128, kd = kh * d, vd = vh * d, ch = 2 * kd + vd;
    std::mt19937 rng(11);
    std::uniform_real_distribution<float> u(-1.0f, 1.0f);
    std::normal_distribution<float> nrm(0.0f, 1.0f);
    std::vector<float> hx((size_t) m * ch), hb((size_t) m * vh), ha((size_t) m * vh), hA(vh), hdtb(vh), hS((size_t) vh * d * d);
    // q and k as the model leaves them (L2-normalized per head, then scaled), v free
    for (unsigned int t = 0; t < m; t++) {
        for (unsigned int part = 0; part < 2; part++)
            for (unsigned int h = 0; h < kh; h++) {
                double ss = 0;
                float* row = &hx[(size_t) t * ch + part * kd + h * d];
                for (unsigned int i = 0; i < d; i++) { row[i] = nrm(rng); ss += row[i] * row[i]; }
                for (unsigned int i = 0; i < d; i++) row[i] = (float) (row[i] / sqrt(ss)) * (part == 0 ? 11.3137f * 0.0884f : 1.0f);
            }
        for (unsigned int i = 0; i < vd; i++) hx[(size_t) t * ch + 2 * kd + i] = 0.5f * nrm(rng);
    }
    for (auto& x : hb) x = 2.0f * nrm(rng);
    for (auto& x : ha) x = nrm(rng);
    for (auto& x : hA) x = -0.5f - 0.5f * (u(rng) + 1.0f);
    for (auto& x : hdtb) x = 0.3f * nrm(rng);
    for (auto& x : hS) x = 0.05f * nrm(rng);

    float *x, *b, *a, *A, *dtb, *S1, *S2, *o1, *o2;
    CK(cudaMalloc(&x, hx.size() * 4)); CK(cudaMalloc(&b, hb.size() * 4)); CK(cudaMalloc(&a, ha.size() * 4));
    CK(cudaMalloc(&A, vh * 4)); CK(cudaMalloc(&dtb, vh * 4));
    CK(cudaMalloc(&S1, hS.size() * 4)); CK(cudaMalloc(&S2, hS.size() * 4));
    CK(cudaMalloc(&o1, (size_t) m * vd * 4)); CK(cudaMalloc(&o2, (size_t) m * vd * 4));
    CK(cudaMemcpy(x, hx.data(), hx.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(b, hb.data(), hb.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(a, ha.data(), ha.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(A, hA.data(), vh * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dtb, hdtb.data(), vh * 4, cudaMemcpyHostToDevice));
    const float *q = x, *k = x + kd, *v = x + 2 * kd;

    auto run_f4 = [&](float* S, float* o) { delta_steps128_f4<<<vh, 512>>>(q, k, v, ch, b, a, vh, A, dtb, S, o, vd, kh, m); };
    auto run_kh = [&](float* S, float* o) { delta_steps128_kh<<<kh * 4, 128>>>(q, k, v, ch, b, a, vh, A, dtb, S, o, vd, kh, m); };

    // identical bits, from the same starting state, over m tokens and over a ragged tail (m - 3, m - 8 + 1)
    int bad = 0;
    for (unsigned int mm : {m, m - 3, m - 7, 8u, 5u, 1u}) {
        CK(cudaMemcpy(S1, hS.data(), hS.size() * 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(S2, hS.data(), hS.size() * 4, cudaMemcpyHostToDevice));
        CK(cudaMemset(o1, 0, (size_t) m * vd * 4)); CK(cudaMemset(o2, 0, (size_t) m * vd * 4));
        delta_steps128_f4<<<vh, 512>>>(q, k, v, ch, b, a, vh, A, dtb, S1, o1, vd, kh, mm);
        delta_steps128_kh<<<kh * 4, 128>>>(q, k, v, ch, b, a, vh, A, dtb, S2, o2, vd, kh, mm);
        CK(cudaDeviceSynchronize());
        for (int var = 2; var >= 1; var--) {   // the 2-warp and 1-warp blocks against the 4-warp one
            float *S3, *o3; CK(cudaMalloc(&S3, hS.size() * 4)); CK(cudaMalloc(&o3, (size_t) m * vd * 4));
            CK(cudaMemcpy(S3, hS.data(), hS.size() * 4, cudaMemcpyHostToDevice)); CK(cudaMemset(o3, 0, (size_t) m * vd * 4));
            if (var == 2) delta_steps128_kh2<<<kh * 8, 64>>>(q, k, v, ch, b, a, vh, A, dtb, S3, o3, vd, kh, mm);
            else delta_steps128_kh1<<<kh * 16, 32>>>(q, k, v, ch, b, a, vh, A, dtb, S3, o3, vd, kh, mm);
            CK(cudaDeviceSynchronize());
            std::vector<float> r3((size_t) m * vd), r2((size_t) m * vd), s3(hS.size()), s2(hS.size());
            CK(cudaMemcpy(r3.data(), o3, r3.size() * 4, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(r2.data(), o2, r2.size() * 4, cudaMemcpyDeviceToHost));
            CK(cudaMemcpy(s3.data(), S3, s3.size() * 4, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(s2.data(), S2, s2.size() * 4, cudaMemcpyDeviceToHost));
            size_t bo = 0, bs = 0;
            for (size_t i = 0; i < r3.size(); i++) bo += memcmp(&r3[i], &r2[i], 4) != 0;
            for (size_t i = 0; i < s3.size(); i++) bs += memcmp(&s3[i], &s2[i], 4) != 0;
            printf("   m %5u: kh%d against kh: outputs differing %zu, state %zu%s\n", mm, var, bo, bs, (bo || bs) ? "   <-- MISMATCH" : "");
            bad += (bo || bs);
            cudaFree(S3); cudaFree(o3);
        }
        std::vector<float> r1((size_t) m * vd), r2((size_t) m * vd), s1(hS.size()), s2(hS.size());
        CK(cudaMemcpy(r1.data(), o1, r1.size() * 4, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(r2.data(), o2, r2.size() * 4, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(s1.data(), S1, s1.size() * 4, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(s2.data(), S2, s2.size() * 4, cudaMemcpyDeviceToHost));
        size_t bo = 0, bs = 0;
        for (size_t i = 0; i < r1.size(); i++) bo += memcmp(&r1[i], &r2[i], 4) != 0;
        for (size_t i = 0; i < s1.size(); i++) bs += memcmp(&s1[i], &s2[i], 4) != 0;
        printf("m %5u: outputs differing in %zu of %zu words, state in %zu of %zu%s\n", mm, bo, r1.size(), bs, s1.size(), (bo || bs) ? "   <-- MISMATCH" : "");
        bad += (bo || bs);
    }

    // timings: alternate the two so a clock step affects both
    cudaEvent_t e0, e1;
    CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    CK(cudaMemcpy(S1, hS.data(), hS.size() * 4, cudaMemcpyHostToDevice));
    for (int w = 0; w < 3; w++) { run_f4(S1, o1); run_kh(S1, o2); }
    CK(cudaDeviceSynchronize());
    float tf = 0, tk = 0, tk2 = 0, tk1 = 0;
    const int reps = 30;
    for (int r = 0; r < reps; r++) {
        CK(cudaEventRecord(e0)); run_f4(S1, o1); CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1));
        float ms; CK(cudaEventElapsedTime(&ms, e0, e1)); tf += ms;
        CK(cudaEventRecord(e0)); run_kh(S1, o2); CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1));
        CK(cudaEventElapsedTime(&ms, e0, e1)); tk += ms;
        CK(cudaEventRecord(e0)); delta_steps128_kh2<<<kh * 8, 64>>>(q, k, v, ch, b, a, vh, A, dtb, S1, o2, vd, kh, m); CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1));
        CK(cudaEventElapsedTime(&ms, e0, e1)); tk2 += ms;
        CK(cudaEventRecord(e0)); delta_steps128_kh1<<<kh * 16, 32>>>(q, k, v, ch, b, a, vh, A, dtb, S1, o2, vd, kh, m); CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1));
        CK(cudaEventElapsedTime(&ms, e0, e1)); tk1 += ms;
    }
    printf("m %u: f4 %.3f ms; kh (4 warps, 64 blocks) %.3f ms (%.2fx); kh2 (2 warps, 128 blocks) %.3f ms (%.2fx); kh1 (1 warp, 256 blocks) %.3f ms (%.2fx); a 48-head layer\n",
           m, tf / reps, tk / reps, tf / tk, tk2 / reps, tf / tk2, tk1 / reps, tf / tk1);
    return bad ? 1 : 0;
}
