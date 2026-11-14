// EXL3 (ExLlamaV3) trellis kernels: y = x W for a projection stored as a blob (crates/shadow-model/src/exl3.rs has the format and the
// CPU reference these are checked against): f16 suh [cols], f16 svh [rows], a 4-byte marker, then the trellis, 16 K u16 words a
// 16 x 16 tile, tiles in [cols/16][rows/16] order.  The weight is diag(suh) H W_hat H diag(svh) with H the 128 x 128 Hadamard on
// every 128-block of the input and of the output index, so a block of 256 threads (8 warps, a warp a tile column) owns 128
// outputs, rotates the input 128 at a time in shared memory, accumulates the 8 tile columns x 8 tile rows of that slice, and
// rotates the 128 outputs at the end.  Lane L of a warp decodes weights 8 L .. 8 L + 7 of a tile (the mma B-fragment order):
// row 2 (L mod 4) + (j mod 2) + 8 ((j >> 1) mod 2), column 2 (L >> 3) + ((L >> 2) mod 2) + 8 (j >> 2).  Only the codebook "mul1"
// (the one every published model uses) is made.  Needs prelude.cuh only; no cuda_fp16.h.

static __device__ __forceinline__ float exl3_cb2(u32 state) {
    u32 x = state * 0x83DCD12Du;
    u32 sum;
    asm("dp4a.u32.u32 %0, %1, %2, %3;" : "=r"(sum) : "r"(x), "r"(0x01010101u), "r"(0x6400u));
    float r;
    asm("{ .reg .b16 a, b, c, d; mov.b16 a, %1; mov.b16 b, 0x1eee; mov.b16 c, 0xc931; fma.rn.f16 d, a, b, c; cvt.f32.f16 %0, d; }"
        : "=f"(r) : "h"((u16) sum));
    return r;
}

// the 8 weights of lane `lane` of a tile (`tile`: 8 K u32 words).  The 8 windows of a lane are consecutive in the ring (K bits apart,
// 16 bits wide): 7 K + 16 bits from `sb`, in at most 4 words, loaded once (modulo the ring).
template <int K>
static __device__ __forceinline__ void exl3_lane_vals(const u32* tile, int lane, float* v) {
    constexpr int NW = (31 + 7 * K + 16 + 31) / 32;
    constexpr int RING = 256 * K;
    const int sb = ((lane * 8 + 1) * K + RING - 16) % RING;
    const int w0 = sb >> 5, off = sb & 31;
    u32 w[NW + 1];
    #pragma unroll
    for (int i = 0; i < NW; i++) {
        int wi = w0 + i;
        w[i] = tile[wi >= 8 * K ? wi - 8 * K : wi];
    }
    w[NW] = 0;
    #pragma unroll
    for (int j = 0; j < 8; j++) {
        int pos = off + j * K;
        int wi = pos >> 5, sh = pos & 31;
        u64 m = ((u64) w[wi] << 32) | w[wi + 1];
        v[j] = exl3_cb2((u32) (m >> (48 - sh)) & 0xffffu);
    }
}

// y_j = x_j W for j < m (m <= M), x_j = x + j xs, y_j = y + j ys; one block a 128-output slice (blockIdx.x)
template <int K, int M>
static __device__ __forceinline__ void exl3_mv(const u8* w, const float* x, u32 xs, float* y, u32 ys, u32 rows, u32 cols, u32 m) {
    __shared__ float sx[M][128];
    __shared__ float sy[M][128];
    const u32 ob = blockIdx.x;
    const u16* suh = (const u16*) w;
    const u16* svh = (const u16*) (w + 2 * cols);
    const u32* tr = (const u32*) (w + 2 * cols + 2 * rows + 4);
    const u32 nt = rows / 16;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int col_lo = 2 * (lane >> 3) + ((lane >> 2) & 1);
    float acc_lo[M], acc_hi[M];
    #pragma unroll
    for (int i = 0; i < M; i++) { acc_lo[i] = 0.0f; acc_hi[i] = 0.0f; }
    for (u32 b = 0; b < cols / 128; b++) {
        for (u32 idx = threadIdx.x; idx < m * 128; idx += 256) {
            u32 j = idx >> 7, i = idx & 127;
            sx[j][i] = x[(u64) j * xs + b * 128 + i] * f16_to_f32(suh[b * 128 + i]);
        }
        __syncthreads();
        for (int h = 1; h < 128; h <<= 1) {
            for (u32 idx = threadIdx.x; idx < m * 64; idx += 256) {
                u32 j = idx >> 6, p = idx & 63;
                u32 i0 = ((p / h) * 2 * h) + (p % h);
                float a = sx[j][i0], c = sx[j][i0 + h];
                sx[j][i0] = a + c;
                sx[j][i0 + h] = a - c;
            }
            __syncthreads();
        }
        for (int kk = 0; kk < 8; kk++) {
            const u32* tile = tr + ((u64) (b * 8 + kk) * nt + ob * 8 + warp) * (8 * K);
            float v[8];
            exl3_lane_vals<K>(tile, lane, v);
            #pragma unroll
            for (int j = 0; j < 8; j++) {
                int r = kk * 16 + (lane & 3) * 2 + (j & 1) + 8 * ((j >> 1) & 1);
                #pragma unroll
                for (int i = 0; i < M; i++) {
                    if ((u32) i < m) {
                        float xv = sx[i][r];
                        if (j < 4) acc_lo[i] += xv * v[j]; else acc_hi[i] += xv * v[j];
                    }
                }
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for (int i = 0; i < M; i++) {
        acc_lo[i] += __shfl_xor_sync(0xffffffffu, acc_lo[i], 1);
        acc_lo[i] += __shfl_xor_sync(0xffffffffu, acc_lo[i], 2);
        acc_hi[i] += __shfl_xor_sync(0xffffffffu, acc_hi[i], 1);
        acc_hi[i] += __shfl_xor_sync(0xffffffffu, acc_hi[i], 2);
        if ((lane & 3) == 0 && (u32) i < m) {
            sy[i][warp * 16 + col_lo] = acc_lo[i] * 0.08838834764831845f;
            sy[i][warp * 16 + col_lo + 8] = acc_hi[i] * 0.08838834764831845f;
        }
    }
    __syncthreads();
    for (int h = 1; h < 128; h <<= 1) {
        for (u32 idx = threadIdx.x; idx < m * 64; idx += 256) {
            u32 j = idx >> 6, p = idx & 63;
            u32 i0 = ((p / h) * 2 * h) + (p % h);
            float a = sy[j][i0], c = sy[j][i0 + h];
            sy[j][i0] = a + c;
            sy[j][i0 + h] = a - c;
        }
        __syncthreads();
    }
    for (u32 idx = threadIdx.x; idx < m * 128; idx += 256) {
        u32 j = idx >> 7, i = idx & 127;
        y[(u64) j * ys + ob * 128 + i] = sy[j][i] * 0.08838834764831845f * f16_to_f32(svh[ob * 128 + i]);
    }
}

// v2, one input (M = 1): a block of 256 KS threads owns 128 outputs; KS k-slices of 8 warps (a warp a tile column) walk the tile rows
// kt = slice, slice + KS, ...; the whole input is rotated into shared memory first (one warp a 128-block, no barriers: the
// butterflies run in registers and shuffles, in the same stage order as v1), the slices' partial sums are added in slice order
// (deterministic), and warp 0 rotates the 128 outputs.  Dynamic shared memory: 4 cols bytes.
template <int K, int KS>
static __device__ __forceinline__ void exl3_mv2(const u8* w, const float* x, float* y, u32 rows, u32 cols) {
    extern __shared__ float exl3_sx[];
    __shared__ float red[KS][128];
    constexpr int NWARP = 8 * KS;
    const u32 ob = blockIdx.x;
    const u16* suh = (const u16*) w;
    const u16* svh = (const u16*) (w + 2 * cols);
    const u32* tr = (const u32*) (w + 2 * cols + 2 * rows + 4);
    const u32 nt = rows / 16;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    for (u32 b = warp; b < cols / 128; b += NWARP) {
        float v[4];
        #pragma unroll
        for (int e = 0; e < 4; e++) v[e] = x[b * 128 + lane + 32 * e] * f16_to_f32(suh[b * 128 + lane + 32 * e]);
        #pragma unroll
        for (int m = 1; m < 32; m <<= 1) {
            #pragma unroll
            for (int e = 0; e < 4; e++) { float p = __shfl_xor_sync(0xffffffffu, v[e], m); v[e] = (lane & m) ? p - v[e] : v[e] + p; }
        }
        { float a = v[0], c = v[1]; v[0] = a + c; v[1] = a - c; a = v[2]; c = v[3]; v[2] = a + c; v[3] = a - c; }
        { float a = v[0], c = v[2]; v[0] = a + c; v[2] = a - c; a = v[1]; c = v[3]; v[1] = a + c; v[3] = a - c; }
        #pragma unroll
        for (int e = 0; e < 4; e++) exl3_sx[b * 128 + lane + 32 * e] = v[e];
    }
    __syncthreads();
    const int tc = warp & 7, sl = warp >> 3;
    float lo = 0.0f, hi = 0.0f;
    #pragma unroll 2
    for (u32 kt = sl; kt < cols / 16; kt += KS) {
        const u32* tile = tr + ((u64) kt * nt + ob * 8 + tc) * (8 * K);
        float v[8];
        exl3_lane_vals<K>(tile, lane, v);
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            float xv = exl3_sx[kt * 16 + (lane & 3) * 2 + (j & 1) + 8 * ((j >> 1) & 1)];
            if (j < 4) lo += xv * v[j]; else hi += xv * v[j];
        }
    }
    lo += __shfl_xor_sync(0xffffffffu, lo, 1);
    lo += __shfl_xor_sync(0xffffffffu, lo, 2);
    hi += __shfl_xor_sync(0xffffffffu, hi, 1);
    hi += __shfl_xor_sync(0xffffffffu, hi, 2);
    if ((lane & 3) == 0) {
        const int col_lo = 2 * (lane >> 3) + ((lane >> 2) & 1);
        red[sl][tc * 16 + col_lo] = lo;
        red[sl][tc * 16 + col_lo + 8] = hi;
    }
    __syncthreads();
    if (warp == 0) {
        float v[4];
        #pragma unroll
        for (int e = 0; e < 4; e++) {
            float t = red[0][lane + 32 * e];
            #pragma unroll
            for (int q = 1; q < KS; q++) t += red[q][lane + 32 * e];
            v[e] = t * 0.08838834764831845f;
        }
        #pragma unroll
        for (int m = 1; m < 32; m <<= 1) {
            #pragma unroll
            for (int e = 0; e < 4; e++) { float p = __shfl_xor_sync(0xffffffffu, v[e], m); v[e] = (lane & m) ? p - v[e] : v[e] + p; }
        }
        { float a = v[0], c = v[1]; v[0] = a + c; v[1] = a - c; a = v[2]; c = v[3]; v[2] = a + c; v[3] = a - c; }
        { float a = v[0], c = v[2]; v[0] = a + c; v[2] = a - c; a = v[1]; c = v[3]; v[1] = a + c; v[3] = a - c; }
        #pragma unroll
        for (int e = 0; e < 4; e++) y[ob * 128 + lane + 32 * e] = v[e] * 0.08838834764831845f * f16_to_f32(svh[ob * 128 + lane + 32 * e]);
    }
}

// v2, m <= M inputs: the same slices, but a slice walks its own 128-blocks of the input (block b belongs to slice b mod KS) and rotates
// one block of all m inputs at a time (warp j of the slice rotates input j, in registers and shuffles), then its 8 warps accumulate
// the 8 tile rows of that block for all m inputs; the slice's barrier is a named one (id 1 + slice).  Shared memory: a block of
// every input a slice, M x 128 floats, plus the partial sums.
template <int K, int M, int KS>
static __device__ __forceinline__ void exl3_mvc2(const u8* w, const float* x, u32 xs, float* y, u32 ys, u32 rows, u32 cols, u32 m) {
    __shared__ float sxs[KS][M][128];
    __shared__ float red[KS][M][128];
    const u32 ob = blockIdx.x;
    const u16* suh = (const u16*) w;
    const u16* svh = (const u16*) (w + 2 * cols);
    const u32* tr = (const u32*) (w + 2 * cols + 2 * rows + 4);
    const u32 nt = rows / 16;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int tc = warp & 7, sl = warp >> 3;
    float acc_lo[M], acc_hi[M];
    #pragma unroll
    for (int i = 0; i < M; i++) { acc_lo[i] = 0.0f; acc_hi[i] = 0.0f; }
    for (u32 b = sl; b < cols / 128; b += KS) {
        if ((u32) tc < m) {
            float v[4];
            #pragma unroll
            for (int e = 0; e < 4; e++) v[e] = x[(u64) tc * xs + b * 128 + lane + 32 * e] * f16_to_f32(suh[b * 128 + lane + 32 * e]);
            #pragma unroll
            for (int q = 1; q < 32; q <<= 1) {
                #pragma unroll
                for (int e = 0; e < 4; e++) { float p = __shfl_xor_sync(0xffffffffu, v[e], q); v[e] = (lane & q) ? p - v[e] : v[e] + p; }
            }
            { float a = v[0], c = v[1]; v[0] = a + c; v[1] = a - c; a = v[2]; c = v[3]; v[2] = a + c; v[3] = a - c; }
            { float a = v[0], c = v[2]; v[0] = a + c; v[2] = a - c; a = v[1]; c = v[3]; v[1] = a + c; v[3] = a - c; }
            #pragma unroll
            for (int e = 0; e < 4; e++) sxs[sl][tc][lane + 32 * e] = v[e];
        }
        asm volatile("bar.sync %0, 256;" :: "r"(1 + sl));
        #pragma unroll 2
        for (int kk = 0; kk < 8; kk++) {
            const u32* tile = tr + ((u64) (b * 8 + kk) * nt + ob * 8 + tc) * (8 * K);
            float v[8];
            exl3_lane_vals<K>(tile, lane, v);
            #pragma unroll
            for (int i = 0; i < M; i++) {
                if ((u32) i < m) {
                    const float2 xa = *(const float2*) &sxs[sl][i][kk * 16 + (lane & 3) * 2];
                    const float2 xb = *(const float2*) &sxs[sl][i][kk * 16 + (lane & 3) * 2 + 8];
                    // j: (j & 1) picks the element of a pair, (j >> 1) & 1 the pair; j < 4 is the low column, j >= 4 the high one
                    acc_lo[i] += xa.x * v[0]; acc_lo[i] += xa.y * v[1]; acc_lo[i] += xb.x * v[2]; acc_lo[i] += xb.y * v[3];
                    acc_hi[i] += xa.x * v[4]; acc_hi[i] += xa.y * v[5]; acc_hi[i] += xb.x * v[6]; acc_hi[i] += xb.y * v[7];
                }
            }
        }
        asm volatile("bar.sync %0, 256;" :: "r"(1 + sl));
    }
    #pragma unroll
    for (int i = 0; i < M; i++) {
        acc_lo[i] += __shfl_xor_sync(0xffffffffu, acc_lo[i], 1);
        acc_lo[i] += __shfl_xor_sync(0xffffffffu, acc_lo[i], 2);
        acc_hi[i] += __shfl_xor_sync(0xffffffffu, acc_hi[i], 1);
        acc_hi[i] += __shfl_xor_sync(0xffffffffu, acc_hi[i], 2);
        if ((lane & 3) == 0) {
            const int col_lo = 2 * (lane >> 3) + ((lane >> 2) & 1);
            red[sl][i][tc * 16 + col_lo] = acc_lo[i];
            red[sl][i][tc * 16 + col_lo + 8] = acc_hi[i];
        }
    }
    __syncthreads();
    if (warp < (int) m) {
        float v[4];
        #pragma unroll
        for (int e = 0; e < 4; e++) {
            float t = red[0][warp][lane + 32 * e];
            #pragma unroll
            for (int q = 1; q < KS; q++) t += red[q][warp][lane + 32 * e];
            v[e] = t * 0.08838834764831845f;
        }
        #pragma unroll
        for (int q = 1; q < 32; q <<= 1) {
            #pragma unroll
            for (int e = 0; e < 4; e++) { float p = __shfl_xor_sync(0xffffffffu, v[e], q); v[e] = (lane & q) ? p - v[e] : v[e] + p; }
        }
        { float a = v[0], c = v[1]; v[0] = a + c; v[1] = a - c; a = v[2]; c = v[3]; v[2] = a + c; v[3] = a - c; }
        { float a = v[0], c = v[2]; v[0] = a + c; v[2] = a - c; a = v[1]; c = v[3]; v[1] = a + c; v[3] = a - c; }
        #pragma unroll
        for (int e = 0; e < 4; e++) y[(u64) warp * ys + ob * 128 + lane + 32 * e] = v[e] * 0.08838834764831845f * f16_to_f32(svh[ob * 128 + lane + 32 * e]);
    }
}

// all 65,536 codebook values (a test of the decode against the CPU reference's)
extern "C" __global__ void exl3_codebook(float* out, u32 n) {
    u32 i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = exl3_cb2(i);
}


extern "C" __global__ void matvec_exl3_1(const u8* w, const float* x, float* y, u32 rows, u32 cols) { exl3_mv<1, 1>(w, x, 0, y, 0, rows, cols, 1); }
extern "C" __global__ void matvec_cols_exl3_1(const u8* w, const float* x, u32 xs, float* y, u32 ys, u32 rows, u32 cols, u32 m) { exl3_mv<1, 8>(w, x, xs, y, ys, rows, cols, m); }
extern "C" __global__ void matvec_batch_exl3_1(const u64* wtab, const float* x, u32 xs, float* y, u32 ys, u32 rows, u32 cols) { exl3_mv<1, 1>((const u8*) wtab[blockIdx.y], x + (u64) blockIdx.y * xs, 0, y + (u64) blockIdx.y * ys, 0, rows, cols, 1); }
extern "C" __global__ void __launch_bounds__(512) matvec_cols_exl3_s2_1(const u8* w, const float* x, u32 xs, float* y, u32 ys, u32 rows, u32 cols, u32 m) { exl3_mvc2<1, 8, 2>(w, x, xs, y, ys, rows, cols, m); }
extern "C" __global__ void __launch_bounds__(1024) matvec_cols_exl3_s4_1(const u8* w, const float* x, u32 xs, float* y, u32 ys, u32 rows, u32 cols, u32 m) { exl3_mvc2<1, 8, 4>(w, x, xs, y, ys, rows, cols, m); }
extern "C" __global__ void __launch_bounds__(256) matvec_exl3_s1_1(const u8* w, const float* x, float* y, u32 rows, u32 cols) { exl3_mv2<1, 1>(w, x, y, rows, cols); }
extern "C" __global__ void __launch_bounds__(256) matvec_batch_exl3_s1_1(const u64* wtab, const float* x, u32 xs, float* y, u32 ys, u32 rows, u32 cols) { exl3_mv2<1, 1>((const u8*) wtab[blockIdx.y], x + (u64) blockIdx.y * xs, y + (u64) blockIdx.y * ys, rows, cols); }
extern "C" __global__ void __launch_bounds__(512) matvec_exl3_s2_1(const u8* w, const float* x, float* y, u32 rows, u32 cols) { exl3_mv2<1, 2>(w, x, y, rows, cols); }
extern "C" __global__ void __launch_bounds__(512) matvec_batch_exl3_s2_1(const u64* wtab, const float* x, u32 xs, float* y, u32 ys, u32 rows, u32 cols) { exl3_mv2<1, 2>((const u8*) wtab[blockIdx.y], x + (u64) blockIdx.y * xs, y + (u64) blockIdx.y * ys, rows, cols); }
extern "C" __global__ void __launch_bounds__(1024) matvec_exl3_s4_1(const u8* w, const float* x, float* y, u32 rows, u32 cols) { exl3_mv2<1, 4>(w, x, y, rows, cols); }
extern "C" __global__ void __launch_bounds__(1024) matvec_batch_exl3_s4_1(const u64* wtab, const float* x, u32 xs, float* y, u32 ys, u32 rows, u32 cols) { exl3_mv2<1, 4>((const u8*) wtab[blockIdx.y], x + (u64) blockIdx.y * xs, y + (u64) blockIdx.y * ys, rows, cols); }
extern "C" __global__ void matvec_exl3_2(const u8* w, const float* x, float* y, u32 rows, u32 cols) { exl3_mv<2, 1>(w, x, 0, y, 0, rows, cols, 1); }
extern "C" __global__ void matvec_cols_exl3_2(const u8* w, const float* x, u32 xs, float* y, u32 ys, u32 rows, u32 cols, u32 m) { exl3_mv<2, 8>(w, x, xs, y, ys, rows, cols, m); }
extern "C" __global__ void matvec_batch_exl3_2(const u64* wtab, const float* x, u32 xs, float* y, u32 ys, u32 rows, u32 cols) { exl3_mv<2, 1>((const u8*) wtab[blockIdx.y], x + (u64) blockIdx.y * xs, 0, y + (u64) blockIdx.y * ys, 0, rows, cols, 1); }
extern "C" __global__ void __launch_bounds__(512) matvec_cols_exl3_s2_2(const u8* w, const float* x, u32 xs, float* y, u32 ys, u32 rows, u32 cols, u32 m) { exl3_mvc2<2, 8, 2>(w, x, xs, y, ys, rows, cols, m); }
extern "C" __global__ void __launch_bounds__(1024) matvec_cols_exl3_s4_2(const u8* w, const float* x, u32 xs, float* y, u32 ys, u32 rows, u32 cols, u32 m) { exl3_mvc2<2, 8, 4>(w, x, xs, y, ys, rows, cols, m); }
extern "C" __global__ void __launch_bounds__(256) matvec_exl3_s1_2(const u8* w, const float* x, float* y, u32 rows, u32 cols) { exl3_mv2<2, 1>(w, x, y, rows, cols); }
extern "C" __global__ void __launch_bounds__(256) matvec_batch_exl3_s1_2(const u64* wtab, const float* x, u32 xs, float* y, u32 ys, u32 rows, u32 cols) { exl3_mv2<2, 1>((const u8*) wtab[blockIdx.y], x + (u64) blockIdx.y * xs, y + (u64) blockIdx.y * ys, rows, cols); }
extern "C" __global__ void __launch_bounds__(512) matvec_exl3_s2_2(const u8* w, const float* x, float* y, u32 rows, u32 cols) { exl3_mv2<2, 2>(w, x, y, rows, cols); }
extern "C" __global__ void __launch_bounds__(512) matvec_batch_exl3_s2_2(const u64* wtab, const float* x, u32 xs, float* y, u32 ys, u32 rows, u32 cols) { exl3_mv2<2, 2>((const u8*) wtab[blockIdx.y], x + (u64) blockIdx.y * xs, y + (u64) blockIdx.y * ys, rows, cols); }
extern "C" __global__ void __launch_bounds__(1024) matvec_exl3_s4_2(const u8* w, const float* x, float* y, u32 rows, u32 cols) { exl3_mv2<2, 4>(w, x, y, rows, cols); }
extern "C" __global__ void __launch_bounds__(1024) matvec_batch_exl3_s4_2(const u64* wtab, const float* x, u32 xs, float* y, u32 ys, u32 rows, u32 cols) { exl3_mv2<2, 4>((const u8*) wtab[blockIdx.y], x + (u64) blockIdx.y * xs, y + (u64) blockIdx.y * ys, rows, cols); }
extern "C" __global__ void matvec_exl3_3(const u8* w, const float* x, float* y, u32 rows, u32 cols) { exl3_mv<3, 1>(w, x, 0, y, 0, rows, cols, 1); }
extern "C" __global__ void matvec_cols_exl3_3(const u8* w, const float* x, u32 xs, float* y, u32 ys, u32 rows, u32 cols, u32 m) { exl3_mv<3, 8>(w, x, xs, y, ys, rows, cols, m); }
extern "C" __global__ void matvec_batch_exl3_3(const u64* wtab, const float* x, u32 xs, float* y, u32 ys, u32 rows, u32 cols) { exl3_mv<3, 1>((const u8*) wtab[blockIdx.y], x + (u64) blockIdx.y * xs, 0, y + (u64) blockIdx.y * ys, 0, rows, cols, 1); }
extern "C" __global__ void __launch_bounds__(512) matvec_cols_exl3_s2_3(const u8* w, const float* x, u32 xs, float* y, u32 ys, u32 rows, u32 cols, u32 m) { exl3_mvc2<3, 8, 2>(w, x, xs, y, ys, rows, cols, m); }
extern "C" __global__ void __launch_bounds__(1024) matvec_cols_exl3_s4_3(const u8* w, const float* x, u32 xs, float* y, u32 ys, u32 rows, u32 cols, u32 m) { exl3_mvc2<3, 8, 4>(w, x, xs, y, ys, rows, cols, m); }
extern "C" __global__ void __launch_bounds__(256) matvec_exl3_s1_3(const u8* w, const float* x, float* y, u32 rows, u32 cols) { exl3_mv2<3, 1>(w, x, y, rows, cols); }
extern "C" __global__ void __launch_bounds__(256) matvec_batch_exl3_s1_3(const u64* wtab, const float* x, u32 xs, float* y, u32 ys, u32 rows, u32 cols) { exl3_mv2<3, 1>((const u8*) wtab[blockIdx.y], x + (u64) blockIdx.y * xs, y + (u64) blockIdx.y * ys, rows, cols); }
extern "C" __global__ void __launch_bounds__(512) matvec_exl3_s2_3(const u8* w, const float* x, float* y, u32 rows, u32 cols) { exl3_mv2<3, 2>(w, x, y, rows, cols); }
extern "C" __global__ void __launch_bounds__(512) matvec_batch_exl3_s2_3(const u64* wtab, const float* x, u32 xs, float* y, u32 ys, u32 rows, u32 cols) { exl3_mv2<3, 2>((const u8*) wtab[blockIdx.y], x + (u64) blockIdx.y * xs, y + (u64) blockIdx.y * ys, rows, cols); }
extern "C" __global__ void __launch_bounds__(1024) matvec_exl3_s4_3(const u8* w, const float* x, float* y, u32 rows, u32 cols) { exl3_mv2<3, 4>(w, x, y, rows, cols); }
extern "C" __global__ void __launch_bounds__(1024) matvec_batch_exl3_s4_3(const u64* wtab, const float* x, u32 xs, float* y, u32 ys, u32 rows, u32 cols) { exl3_mv2<3, 4>((const u8*) wtab[blockIdx.y], x + (u64) blockIdx.y * xs, y + (u64) blockIdx.y * ys, rows, cols); }
extern "C" __global__ void matvec_exl3_4(const u8* w, const float* x, float* y, u32 rows, u32 cols) { exl3_mv<4, 1>(w, x, 0, y, 0, rows, cols, 1); }
extern "C" __global__ void matvec_cols_exl3_4(const u8* w, const float* x, u32 xs, float* y, u32 ys, u32 rows, u32 cols, u32 m) { exl3_mv<4, 8>(w, x, xs, y, ys, rows, cols, m); }
extern "C" __global__ void matvec_batch_exl3_4(const u64* wtab, const float* x, u32 xs, float* y, u32 ys, u32 rows, u32 cols) { exl3_mv<4, 1>((const u8*) wtab[blockIdx.y], x + (u64) blockIdx.y * xs, 0, y + (u64) blockIdx.y * ys, 0, rows, cols, 1); }
extern "C" __global__ void __launch_bounds__(512) matvec_cols_exl3_s2_4(const u8* w, const float* x, u32 xs, float* y, u32 ys, u32 rows, u32 cols, u32 m) { exl3_mvc2<4, 8, 2>(w, x, xs, y, ys, rows, cols, m); }
extern "C" __global__ void __launch_bounds__(1024) matvec_cols_exl3_s4_4(const u8* w, const float* x, u32 xs, float* y, u32 ys, u32 rows, u32 cols, u32 m) { exl3_mvc2<4, 8, 4>(w, x, xs, y, ys, rows, cols, m); }
extern "C" __global__ void __launch_bounds__(256) matvec_exl3_s1_4(const u8* w, const float* x, float* y, u32 rows, u32 cols) { exl3_mv2<4, 1>(w, x, y, rows, cols); }
extern "C" __global__ void __launch_bounds__(256) matvec_batch_exl3_s1_4(const u64* wtab, const float* x, u32 xs, float* y, u32 ys, u32 rows, u32 cols) { exl3_mv2<4, 1>((const u8*) wtab[blockIdx.y], x + (u64) blockIdx.y * xs, y + (u64) blockIdx.y * ys, rows, cols); }
extern "C" __global__ void __launch_bounds__(512) matvec_exl3_s2_4(const u8* w, const float* x, float* y, u32 rows, u32 cols) { exl3_mv2<4, 2>(w, x, y, rows, cols); }
extern "C" __global__ void __launch_bounds__(512) matvec_batch_exl3_s2_4(const u64* wtab, const float* x, u32 xs, float* y, u32 ys, u32 rows, u32 cols) { exl3_mv2<4, 2>((const u8*) wtab[blockIdx.y], x + (u64) blockIdx.y * xs, y + (u64) blockIdx.y * ys, rows, cols); }
extern "C" __global__ void __launch_bounds__(1024) matvec_exl3_s4_4(const u8* w, const float* x, float* y, u32 rows, u32 cols) { exl3_mv2<4, 4>(w, x, y, rows, cols); }
extern "C" __global__ void __launch_bounds__(1024) matvec_batch_exl3_s4_4(const u64* wtab, const float* x, u32 xs, float* y, u32 ys, u32 rows, u32 cols) { exl3_mv2<4, 4>((const u8*) wtab[blockIdx.y], x + (u64) blockIdx.y * xs, y + (u64) blockIdx.y * ys, rows, cols); }
extern "C" __global__ void matvec_exl3_5(const u8* w, const float* x, float* y, u32 rows, u32 cols) { exl3_mv<5, 1>(w, x, 0, y, 0, rows, cols, 1); }
extern "C" __global__ void matvec_cols_exl3_5(const u8* w, const float* x, u32 xs, float* y, u32 ys, u32 rows, u32 cols, u32 m) { exl3_mv<5, 8>(w, x, xs, y, ys, rows, cols, m); }
extern "C" __global__ void matvec_batch_exl3_5(const u64* wtab, const float* x, u32 xs, float* y, u32 ys, u32 rows, u32 cols) { exl3_mv<5, 1>((const u8*) wtab[blockIdx.y], x + (u64) blockIdx.y * xs, 0, y + (u64) blockIdx.y * ys, 0, rows, cols, 1); }
extern "C" __global__ void __launch_bounds__(512) matvec_cols_exl3_s2_5(const u8* w, const float* x, u32 xs, float* y, u32 ys, u32 rows, u32 cols, u32 m) { exl3_mvc2<5, 8, 2>(w, x, xs, y, ys, rows, cols, m); }
extern "C" __global__ void __launch_bounds__(1024) matvec_cols_exl3_s4_5(const u8* w, const float* x, u32 xs, float* y, u32 ys, u32 rows, u32 cols, u32 m) { exl3_mvc2<5, 8, 4>(w, x, xs, y, ys, rows, cols, m); }
extern "C" __global__ void __launch_bounds__(256) matvec_exl3_s1_5(const u8* w, const float* x, float* y, u32 rows, u32 cols) { exl3_mv2<5, 1>(w, x, y, rows, cols); }
extern "C" __global__ void __launch_bounds__(256) matvec_batch_exl3_s1_5(const u64* wtab, const float* x, u32 xs, float* y, u32 ys, u32 rows, u32 cols) { exl3_mv2<5, 1>((const u8*) wtab[blockIdx.y], x + (u64) blockIdx.y * xs, y + (u64) blockIdx.y * ys, rows, cols); }
extern "C" __global__ void __launch_bounds__(512) matvec_exl3_s2_5(const u8* w, const float* x, float* y, u32 rows, u32 cols) { exl3_mv2<5, 2>(w, x, y, rows, cols); }
extern "C" __global__ void __launch_bounds__(512) matvec_batch_exl3_s2_5(const u64* wtab, const float* x, u32 xs, float* y, u32 ys, u32 rows, u32 cols) { exl3_mv2<5, 2>((const u8*) wtab[blockIdx.y], x + (u64) blockIdx.y * xs, y + (u64) blockIdx.y * ys, rows, cols); }
extern "C" __global__ void __launch_bounds__(1024) matvec_exl3_s4_5(const u8* w, const float* x, float* y, u32 rows, u32 cols) { exl3_mv2<5, 4>(w, x, y, rows, cols); }
extern "C" __global__ void __launch_bounds__(1024) matvec_batch_exl3_s4_5(const u64* wtab, const float* x, u32 xs, float* y, u32 ys, u32 rows, u32 cols) { exl3_mv2<5, 4>((const u8*) wtab[blockIdx.y], x + (u64) blockIdx.y * xs, y + (u64) blockIdx.y * ys, rows, cols); }
extern "C" __global__ void matvec_exl3_6(const u8* w, const float* x, float* y, u32 rows, u32 cols) { exl3_mv<6, 1>(w, x, 0, y, 0, rows, cols, 1); }
extern "C" __global__ void matvec_cols_exl3_6(const u8* w, const float* x, u32 xs, float* y, u32 ys, u32 rows, u32 cols, u32 m) { exl3_mv<6, 8>(w, x, xs, y, ys, rows, cols, m); }
extern "C" __global__ void matvec_batch_exl3_6(const u64* wtab, const float* x, u32 xs, float* y, u32 ys, u32 rows, u32 cols) { exl3_mv<6, 1>((const u8*) wtab[blockIdx.y], x + (u64) blockIdx.y * xs, 0, y + (u64) blockIdx.y * ys, 0, rows, cols, 1); }
extern "C" __global__ void __launch_bounds__(512) matvec_cols_exl3_s2_6(const u8* w, const float* x, u32 xs, float* y, u32 ys, u32 rows, u32 cols, u32 m) { exl3_mvc2<6, 8, 2>(w, x, xs, y, ys, rows, cols, m); }
extern "C" __global__ void __launch_bounds__(1024) matvec_cols_exl3_s4_6(const u8* w, const float* x, u32 xs, float* y, u32 ys, u32 rows, u32 cols, u32 m) { exl3_mvc2<6, 8, 4>(w, x, xs, y, ys, rows, cols, m); }
extern "C" __global__ void __launch_bounds__(256) matvec_exl3_s1_6(const u8* w, const float* x, float* y, u32 rows, u32 cols) { exl3_mv2<6, 1>(w, x, y, rows, cols); }
extern "C" __global__ void __launch_bounds__(256) matvec_batch_exl3_s1_6(const u64* wtab, const float* x, u32 xs, float* y, u32 ys, u32 rows, u32 cols) { exl3_mv2<6, 1>((const u8*) wtab[blockIdx.y], x + (u64) blockIdx.y * xs, y + (u64) blockIdx.y * ys, rows, cols); }
extern "C" __global__ void __launch_bounds__(512) matvec_exl3_s2_6(const u8* w, const float* x, float* y, u32 rows, u32 cols) { exl3_mv2<6, 2>(w, x, y, rows, cols); }
extern "C" __global__ void __launch_bounds__(512) matvec_batch_exl3_s2_6(const u64* wtab, const float* x, u32 xs, float* y, u32 ys, u32 rows, u32 cols) { exl3_mv2<6, 2>((const u8*) wtab[blockIdx.y], x + (u64) blockIdx.y * xs, y + (u64) blockIdx.y * ys, rows, cols); }
extern "C" __global__ void __launch_bounds__(1024) matvec_exl3_s4_6(const u8* w, const float* x, float* y, u32 rows, u32 cols) { exl3_mv2<6, 4>(w, x, y, rows, cols); }
extern "C" __global__ void __launch_bounds__(1024) matvec_batch_exl3_s4_6(const u64* wtab, const float* x, u32 xs, float* y, u32 ys, u32 rows, u32 cols) { exl3_mv2<6, 4>((const u8*) wtab[blockIdx.y], x + (u64) blockIdx.y * xs, y + (u64) blockIdx.y * ys, rows, cols); }
extern "C" __global__ void matvec_exl3_7(const u8* w, const float* x, float* y, u32 rows, u32 cols) { exl3_mv<7, 1>(w, x, 0, y, 0, rows, cols, 1); }
extern "C" __global__ void matvec_cols_exl3_7(const u8* w, const float* x, u32 xs, float* y, u32 ys, u32 rows, u32 cols, u32 m) { exl3_mv<7, 8>(w, x, xs, y, ys, rows, cols, m); }
extern "C" __global__ void matvec_batch_exl3_7(const u64* wtab, const float* x, u32 xs, float* y, u32 ys, u32 rows, u32 cols) { exl3_mv<7, 1>((const u8*) wtab[blockIdx.y], x + (u64) blockIdx.y * xs, 0, y + (u64) blockIdx.y * ys, 0, rows, cols, 1); }
extern "C" __global__ void __launch_bounds__(512) matvec_cols_exl3_s2_7(const u8* w, const float* x, u32 xs, float* y, u32 ys, u32 rows, u32 cols, u32 m) { exl3_mvc2<7, 8, 2>(w, x, xs, y, ys, rows, cols, m); }
extern "C" __global__ void __launch_bounds__(1024) matvec_cols_exl3_s4_7(const u8* w, const float* x, u32 xs, float* y, u32 ys, u32 rows, u32 cols, u32 m) { exl3_mvc2<7, 8, 4>(w, x, xs, y, ys, rows, cols, m); }
extern "C" __global__ void __launch_bounds__(256) matvec_exl3_s1_7(const u8* w, const float* x, float* y, u32 rows, u32 cols) { exl3_mv2<7, 1>(w, x, y, rows, cols); }
extern "C" __global__ void __launch_bounds__(256) matvec_batch_exl3_s1_7(const u64* wtab, const float* x, u32 xs, float* y, u32 ys, u32 rows, u32 cols) { exl3_mv2<7, 1>((const u8*) wtab[blockIdx.y], x + (u64) blockIdx.y * xs, y + (u64) blockIdx.y * ys, rows, cols); }
extern "C" __global__ void __launch_bounds__(512) matvec_exl3_s2_7(const u8* w, const float* x, float* y, u32 rows, u32 cols) { exl3_mv2<7, 2>(w, x, y, rows, cols); }
extern "C" __global__ void __launch_bounds__(512) matvec_batch_exl3_s2_7(const u64* wtab, const float* x, u32 xs, float* y, u32 ys, u32 rows, u32 cols) { exl3_mv2<7, 2>((const u8*) wtab[blockIdx.y], x + (u64) blockIdx.y * xs, y + (u64) blockIdx.y * ys, rows, cols); }
extern "C" __global__ void __launch_bounds__(1024) matvec_exl3_s4_7(const u8* w, const float* x, float* y, u32 rows, u32 cols) { exl3_mv2<7, 4>(w, x, y, rows, cols); }
extern "C" __global__ void __launch_bounds__(1024) matvec_batch_exl3_s4_7(const u64* wtab, const float* x, u32 xs, float* y, u32 ys, u32 rows, u32 cols) { exl3_mv2<7, 4>((const u8*) wtab[blockIdx.y], x + (u64) blockIdx.y * xs, y + (u64) blockIdx.y * ys, rows, cols); }
extern "C" __global__ void matvec_exl3_8(const u8* w, const float* x, float* y, u32 rows, u32 cols) { exl3_mv<8, 1>(w, x, 0, y, 0, rows, cols, 1); }
extern "C" __global__ void matvec_cols_exl3_8(const u8* w, const float* x, u32 xs, float* y, u32 ys, u32 rows, u32 cols, u32 m) { exl3_mv<8, 8>(w, x, xs, y, ys, rows, cols, m); }
extern "C" __global__ void matvec_batch_exl3_8(const u64* wtab, const float* x, u32 xs, float* y, u32 ys, u32 rows, u32 cols) { exl3_mv<8, 1>((const u8*) wtab[blockIdx.y], x + (u64) blockIdx.y * xs, 0, y + (u64) blockIdx.y * ys, 0, rows, cols, 1); }
extern "C" __global__ void __launch_bounds__(512) matvec_cols_exl3_s2_8(const u8* w, const float* x, u32 xs, float* y, u32 ys, u32 rows, u32 cols, u32 m) { exl3_mvc2<8, 8, 2>(w, x, xs, y, ys, rows, cols, m); }
extern "C" __global__ void __launch_bounds__(1024) matvec_cols_exl3_s4_8(const u8* w, const float* x, u32 xs, float* y, u32 ys, u32 rows, u32 cols, u32 m) { exl3_mvc2<8, 8, 4>(w, x, xs, y, ys, rows, cols, m); }
extern "C" __global__ void __launch_bounds__(256) matvec_exl3_s1_8(const u8* w, const float* x, float* y, u32 rows, u32 cols) { exl3_mv2<8, 1>(w, x, y, rows, cols); }
extern "C" __global__ void __launch_bounds__(256) matvec_batch_exl3_s1_8(const u64* wtab, const float* x, u32 xs, float* y, u32 ys, u32 rows, u32 cols) { exl3_mv2<8, 1>((const u8*) wtab[blockIdx.y], x + (u64) blockIdx.y * xs, y + (u64) blockIdx.y * ys, rows, cols); }
extern "C" __global__ void __launch_bounds__(512) matvec_exl3_s2_8(const u8* w, const float* x, float* y, u32 rows, u32 cols) { exl3_mv2<8, 2>(w, x, y, rows, cols); }
extern "C" __global__ void __launch_bounds__(512) matvec_batch_exl3_s2_8(const u64* wtab, const float* x, u32 xs, float* y, u32 ys, u32 rows, u32 cols) { exl3_mv2<8, 2>((const u8*) wtab[blockIdx.y], x + (u64) blockIdx.y * xs, y + (u64) blockIdx.y * ys, rows, cols); }
extern "C" __global__ void __launch_bounds__(1024) matvec_exl3_s4_8(const u8* w, const float* x, float* y, u32 rows, u32 cols) { exl3_mv2<8, 4>(w, x, y, rows, cols); }
extern "C" __global__ void __launch_bounds__(1024) matvec_batch_exl3_s4_8(const u64* wtab, const float* x, u32 xs, float* y, u32 ys, u32 rows, u32 cols) { exl3_mv2<8, 4>((const u8*) wtab[blockIdx.y], x + (u64) blockIdx.y * xs, y + (u64) blockIdx.y * ys, rows, cols); }
