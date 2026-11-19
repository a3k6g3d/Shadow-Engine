// EXL3 (ExLlamaV3) trellis kernels: y = x W for a projection stored as a blob (crates/shadowzero-model/src/exl3.rs has the format and the
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

// the Hadamard rotation of 128 values, lane L holding i = L + 32 e (the stage order of the other kernels: h = 1 .. 64)
static __device__ __forceinline__ void exl3_wht4(float* v, int lane) {
    #pragma unroll
    for (int m = 1; m < 32; m <<= 1) {
        #pragma unroll
        for (int e = 0; e < 4; e++) { float p = __shfl_xor_sync(0xffffffffu, v[e], m); v[e] = (lane & m) ? p - v[e] : v[e] + p; }
    }
    { float a = v[0], c = v[1]; v[0] = a + c; v[1] = a - c; a = v[2]; c = v[3]; v[2] = a + c; v[3] = a - c; }
    { float a = v[0], c = v[2]; v[0] = a + c; v[2] = a - c; a = v[1]; c = v[3]; v[1] = a + c; v[3] = a - c; }
}

static __device__ __forceinline__ u32 exl3_extract16(u32 hi, u32 lo, int sh) {
    u64 m = ((u64) hi << 32) | lo;
    return (u32) (m >> (48 - sh)) & 0xffffu;
}

// A tile's bit stream for KB = 2 K: even KB an integer rate K (weight t's window ends at bit (t + 1) K), odd KB the rate K + 0.5 of the "mul1" codebook, where positions
// alternate K and K + 1 bits (weight 2 q + 1 ends at (q + 1) (2 K + 1), weight 2 q at q (2 K + 1) + K; exllamav3's dq8_half).  A tile is 4 KB u32 words of 128 KB bits.
template <int KB>
struct Rt {
    static constexpr bool H = (KB & 1) != 0;
    static constexpr int KA = KB / 2;
    static constexpr int WORDS = 4 * KB;
    static constexpr int RING = 128 * KB;
    // start bit of window j of a lane (weights 8 L + j) relative to window 0's
    static __device__ __forceinline__ constexpr int off(int j) { return H ? (j >> 1) * (2 * KA + 1) + (j & 1) * (KA + 1) : j * KA; }
    static constexpr int SPAN = 16 + (H ? 3 * (2 * KA + 1) + KA + 1 : 7 * KA);   // bits from window 0's start to window 7's end
    static constexpr int NW = (31 + SPAN + 31) / 32;
    // start bit of window 0 of lane L (before the wrap): the end of weight 8 L less 16
    static __device__ __forceinline__ int start(int lane) {
        const int end0 = H ? 4 * lane * (2 * KA + 1) + KA : (8 * lane + 1) * KA;
        return (end0 + RING - 16) % RING;
    }
};

template <int KB>
static __device__ __forceinline__ void exl3_load_tile(const u32* tile, int lane, u32* w, int& off) {
    using R = Rt<KB>;
    const int sb = R::start(lane);
    const int w0 = sb >> 5;
    off = sb & 31;
#ifndef EXL3_NO_COALESCE
    if constexpr (R::WORDS == 16) {
        u32 tw = (lane < R::WORDS) ? tile[lane] : 0u;
        #pragma unroll
        for (int i = 0; i < R::NW; i++) {
            w[i] = __shfl_sync(0xffffffffu, tw, (w0 + i) & (R::WORDS - 1));
        }
    } else {
        #pragma unroll
        for (int i = 0; i < R::NW; i++) {
            int wi = w0 + i;
            w[i] = tile[wi >= R::WORDS ? wi - R::WORDS : wi];
        }
    }
#else
    #pragma unroll
    for (int i = 0; i < R::NW; i++) {
        int wi = w0 + i;
        w[i] = tile[wi >= R::WORDS ? wi - R::WORDS : wi];
    }
#endif
    w[R::NW] = 0;
}

// the 8 weights of lane `lane` of a tile (`tile`: 4 KB u32 words).  The 8 windows of a lane are consecutive in the ring: `SPAN` bits from the start of window 0,
// in at most 4 words, loaded once (modulo the ring).
template <int KB>
static __device__ __forceinline__ void exl3_lane_vals(const u32* tile, int lane, float* v) {
    using R = Rt<KB>;
    int off;
    u32 w[R::NW + 1];
    exl3_load_tile<KB>(tile, lane, w, off);
    #pragma unroll
    for (int j = 0; j < 8; j++) {
        int pos = off + R::off(j);
        int wi = pos >> 5, sh = pos & 31;
        v[j] = exl3_cb2(exl3_extract16(w[wi], w[wi + 1], sh));
    }
}

// y_j = x_j W for j < m (m <= M), x_j = x + j xs, y_j = y + j ys; one block a 128-output slice (blockIdx.x)
template <int KB, int M>
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
            const u32* tile = tr + ((u64) (b * 8 + kk) * nt + ob * 8 + warp) * (4 * KB);
            float v[8];
            exl3_lane_vals<KB>(tile, lane, v);
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
template <int KB, int KS>
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
    pdl_wait();
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
        const u32* tile = tr + ((u64) kt * nt + ob * 8 + tc) * (4 * KB);
        float v[8];
        exl3_lane_vals<KB>(tile, lane, v);
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
    pdl_launch_dependents();
}

// v2, m <= M inputs: the same slices, but a slice walks its own 128-blocks of the input (block b belongs to slice b mod KS) and rotates
// one block of all m inputs at a time (warp j of the slice rotates input j, in registers and shuffles), then its 8 warps accumulate
// the 8 tile rows of that block for all m inputs; the slice's barrier is a named one (id 1 + slice).  Shared memory: a block of
// every input a slice, M x 128 floats, plus the partial sums.
// SK: split over blocks too (grid.y = nsplit): block (x, y) walks the 128-blocks b = y KS + slice, + nsplit KS, ..., and writes its 128 partial sums of each input to
// part[(y 8 + j) rows + ...] instead of rotating them; `exl3_splitk_combine` adds the nsplit partials in order and rotates (for the shapes with few output blocks)
template <int KB, int M, int KS, bool G, bool SK = false>
static __device__ __forceinline__ void exl3_mvc2(const u8* w, const float* x, u32 xs, float* y, u32 ys, u32 rows, u32 cols, u32 m, const u32* gi) {
    pdl_wait();
    if (G) { m = gi[0]; if (m == 0) return; } // a routed expert's group: n <= 8 inputs at x + gi[1 + j] xs, outputs at y + gi[9 + j] ys
    __shared__ union {
        float sxs[KS][M][128];
        float red[KS][M][128];
    } sh;
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
    for (u32 b = (SK ? blockIdx.y * KS + sl : sl); b < cols / 128; b += (SK ? gridDim.y * KS : KS)) {
        if ((u32) tc < m) {
            float v[4];
            #pragma unroll
            for (int e = 0; e < 4; e++) v[e] = x[(u64) (G ? gi[1 + tc] : (u32) tc) * xs + b * 128 + lane + 32 * e] * f16_to_f32(suh[b * 128 + lane + 32 * e]);
            exl3_wht4(v, lane);
            #pragma unroll
            for (int e = 0; e < 4; e++) sh.sxs[sl][tc][lane + 32 * e] = v[e];
        }
        asm volatile("bar.sync %0, 256;" :: "r"(1 + sl));
        #pragma unroll 2
        for (int kk = 0; kk < 8; kk++) {
            const u32* tile = tr + ((u64) (b * 8 + kk) * nt + ob * 8 + tc) * (4 * KB);
            float v[8];
            exl3_lane_vals<KB>(tile, lane, v);
            #pragma unroll
            for (int i = 0; i < M; i++) {
                if ((u32) i < m) {
                    const float2 xa = *(const float2*) &sh.sxs[sl][i][kk * 16 + (lane & 3) * 2];
                    const float2 xb = *(const float2*) &sh.sxs[sl][i][kk * 16 + (lane & 3) * 2 + 8];
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
        if ((u32) i < m) {
            acc_lo[i] += __shfl_xor_sync(0xffffffffu, acc_lo[i], 1);
            acc_lo[i] += __shfl_xor_sync(0xffffffffu, acc_lo[i], 2);
            acc_hi[i] += __shfl_xor_sync(0xffffffffu, acc_hi[i], 1);
            acc_hi[i] += __shfl_xor_sync(0xffffffffu, acc_hi[i], 2);
            if ((lane & 3) == 0) {
                const int col_lo = 2 * (lane >> 3) + ((lane >> 2) & 1);
                sh.red[sl][i][tc * 16 + col_lo] = acc_lo[i];
                sh.red[sl][i][tc * 16 + col_lo + 8] = acc_hi[i];
            }
        }
    }
    __syncthreads();
    if (SK) {
        if (warp < (int) m) {
            #pragma unroll
            for (int e = 0; e < 4; e++) {
                float t = sh.red[0][warp][lane + 32 * e];
                #pragma unroll
                for (int q = 1; q < KS; q++) t += sh.red[q][warp][lane + 32 * e];
                y[((u64) blockIdx.y * 8 + warp) * rows + ob * 128 + lane + 32 * e] = t;
            }
        }
        return;
    }
    if (warp < (int) m) {
        float v[4];
        #pragma unroll
        for (int e = 0; e < 4; e++) {
            float t = sh.red[0][warp][lane + 32 * e];
            #pragma unroll
            for (int q = 1; q < KS; q++) t += sh.red[q][warp][lane + 32 * e];
            v[e] = t * 0.08838834764831845f;
        }
        exl3_wht4(v, lane);
        #pragma unroll
        for (int e = 0; e < 4; e++) y[(u64) (G ? gi[9 + warp] : (u32) warp) * ys + ob * 128 + lane + 32 * e] = v[e] * 0.08838834764831845f * f16_to_f32(svh[ob * 128 + lane + 32 * e]);
    }
    pdl_launch_dependents();
}

// Pre-rotated small-batch decode (m <= M, contiguous rotated f32 rows xr[m][cols] from exl3_rotate_rows_f32):
// stages 512-col chunks of xr into 8 KiB SMEM (unioned with red) for zero L1 contention and high SM occupancy.
template <int KB, int M, int KS, bool SPARSE = false>
static __device__ __forceinline__ void exl3_mvc2_pre(const u8* __restrict__ w, const float* __restrict__ xr, float* __restrict__ y, u32 ys, u32 rows, u32 cols, u32 m, const u32* __restrict__ sel_blks = nullptr) {
    pdl_wait();
    if (SPARSE && blockIdx.x >= sel_blks[0]) return;
    __shared__ union {
        float sx[M][512];
        float red[KS][M][128];
    } sh;
    const u32 out_blk = blockIdx.x;
    const u32 ob = SPARSE ? sel_blks[1u + out_blk] : out_blk;
    const u16* svh = (const u16*) (w + 2 * cols);
    const u32* tr = (const u32*) (w + 2 * cols + 2 * rows + 4);
    const u32 nt = rows / 16;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int tc = warp & 7, sl = warp >> 3;
    float acc_lo[M], acc_hi[M];
    #pragma unroll
    for (int i = 0; i < M; i++) { acc_lo[i] = 0.0f; acc_hi[i] = 0.0f; }
    const u32 xoff = (lane & 3) * 2;
    for (u32 c0 = 0; c0 < cols; c0 += 512u) {
        const u32 c_len = (cols - c0 < 512u) ? (cols - c0) : 512u;
        const u32 nf4 = c_len >> 2;
        const u32 n_kt = c_len >> 4;
        const u32 tot4 = m * nf4;
        for (u32 idx = threadIdx.x; idx < tot4; idx += 256u * KS) {
            const u32 r = (c_len == 512u) ? (idx >> 7) : (idx / nf4);
            const u32 f4 = (c_len == 512u) ? (idx & 127u) : (idx - r * nf4);
            *(float4*) &sh.sx[r][f4 * 4] = *(const float4*) (xr + (u64) r * cols + c0 + f4 * 4);
        }
        __syncthreads();
        const u32 kt0 = c0 >> 4;
        if (m == 4u) {
            #pragma unroll 2
            for (u32 tk = sl; tk < n_kt; tk += KS) {
                const u32* tile = tr + ((u64) (kt0 + tk) * nt + ob * 8 + tc) * (4 * KB);
                float v[8];
                exl3_lane_vals<KB>(tile, lane, v);
                const u32 soff = tk * 16u + xoff;
                #pragma unroll
                for (int i = 0; i < 4; i++) {
                    const float2 xa = *(const float2*) &sh.sx[i][soff];
                    const float2 xb = *(const float2*) &sh.sx[i][soff + 8];
                    acc_lo[i] += xa.x * v[0]; acc_lo[i] += xa.y * v[1]; acc_lo[i] += xb.x * v[2]; acc_lo[i] += xb.y * v[3];
                    acc_hi[i] += xa.x * v[4]; acc_hi[i] += xa.y * v[5]; acc_hi[i] += xb.x * v[6]; acc_hi[i] += xb.y * v[7];
                }
            }
        } else if (m == 3u) {
            #pragma unroll 2
            for (u32 tk = sl; tk < n_kt; tk += KS) {
                const u32* tile = tr + ((u64) (kt0 + tk) * nt + ob * 8 + tc) * (4 * KB);
                float v[8];
                exl3_lane_vals<KB>(tile, lane, v);
                const u32 soff = tk * 16u + xoff;
                #pragma unroll
                for (int i = 0; i < 3; i++) {
                    const float2 xa = *(const float2*) &sh.sx[i][soff];
                    const float2 xb = *(const float2*) &sh.sx[i][soff + 8];
                    acc_lo[i] += xa.x * v[0]; acc_lo[i] += xa.y * v[1]; acc_lo[i] += xb.x * v[2]; acc_lo[i] += xb.y * v[3];
                    acc_hi[i] += xa.x * v[4]; acc_hi[i] += xa.y * v[5]; acc_hi[i] += xb.x * v[6]; acc_hi[i] += xb.y * v[7];
                }
            }
        } else {
            #pragma unroll 2
            for (u32 tk = sl; tk < n_kt; tk += KS) {
                const u32* tile = tr + ((u64) (kt0 + tk) * nt + ob * 8 + tc) * (4 * KB);
                float v[8];
                exl3_lane_vals<KB>(tile, lane, v);
                const u32 soff = tk * 16u + xoff;
                #pragma unroll
                for (int i = 0; i < M; i++) {
                    if ((u32) i < m) {
                        const float2 xa = *(const float2*) &sh.sx[i][soff];
                        const float2 xb = *(const float2*) &sh.sx[i][soff + 8];
                        acc_lo[i] += xa.x * v[0]; acc_lo[i] += xa.y * v[1]; acc_lo[i] += xb.x * v[2]; acc_lo[i] += xb.y * v[3];
                        acc_hi[i] += xa.x * v[4]; acc_hi[i] += xa.y * v[5]; acc_hi[i] += xb.x * v[6]; acc_hi[i] += xb.y * v[7];
                    }
                }
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for (int i = 0; i < M; i++) {
        if ((u32) i < m) {
            acc_lo[i] += __shfl_xor_sync(0xffffffffu, acc_lo[i], 1);
            acc_lo[i] += __shfl_xor_sync(0xffffffffu, acc_lo[i], 2);
            acc_hi[i] += __shfl_xor_sync(0xffffffffu, acc_hi[i], 1);
            acc_hi[i] += __shfl_xor_sync(0xffffffffu, acc_hi[i], 2);
            if ((lane & 3) == 0) {
                const int col_lo = 2 * (lane >> 3) + ((lane >> 2) & 1);
                sh.red[sl][i][tc * 16 + col_lo] = acc_lo[i];
                sh.red[sl][i][tc * 16 + col_lo + 8] = acc_hi[i];
            }
        }
    }
    __syncthreads();
    if (warp < (int) m) {
        float v[4];
        #pragma unroll
        for (int e = 0; e < 4; e++) {
            float t = sh.red[0][warp][lane + 32 * e];
            #pragma unroll
            for (int q = 1; q < KS; q++) t += sh.red[q][warp][lane + 32 * e];
            v[e] = t * 0.08838834764831845f;
        }
        exl3_wht4(v, lane);
        #pragma unroll
        for (int e = 0; e < 4; e++) y[(u64) warp * ys + out_blk * 128 + lane + 32 * e] = v[e] * 0.08838834764831845f * f16_to_f32(svh[ob * 128 + lane + 32 * e]);
    }
    pdl_launch_dependents();
}

// The tensor-core prompt kernel: y_j = x_j W for 64 tokens a block (blockIdx.y), 128 outputs a block (blockIdx.x).  A warp is a tile column; the
// trellis lane order is the mma B-fragment order (lane L decodes B[k = 2 (L mod 4) + (j mod 2) + 8 ((j >> 1) mod 2)][n = L >> 2] of n-half j >> 2), so a lane's
// 8 decoded f16 values are two n8 fragments as they are.  The input is rotated (x suh, the 128-block Hadamard, 1 / sqrt(128)) a 128-block at a time into
// shared memory as f16 (a warp 8 token rows, in registers and shuffles), the products go through `mma.m16n8k16` with f32 accumulators, and the 128 outputs
// of a token are rotated at the end (x 1 / sqrt(128) x svh).
static __device__ __forceinline__ u32 exl3_cb2_h(u32 state) {
    u32 x = state * 0x83DCD12Du;
    u32 sum;
    asm("dp4a.u32.u32 %0, %1, %2, %3;" : "=r"(sum) : "r"(x), "r"(0x01010101u), "r"(0x6400u));
    u32 r;
    asm("{ .reg .b16 a, b, c, d; mov.b16 a, %1; mov.b16 b, 0x1eee; mov.b16 c, 0xc931; fma.rn.f16 d, a, b, c; cvt.u32.u16 %0, d; }" : "=r"(r) : "h"((u16) sum));
    return r;
}

template <int KB>
static __device__ __forceinline__ void exl3_lane_h(const u32* tile, int lane, u32* hv) {
    using R = Rt<KB>;
    int off;
    u32 w[R::NW + 1];
    exl3_load_tile<KB>(tile, lane, w, off);
    #pragma unroll
    for (int j = 0; j < 8; j++) {
        int pos = off + R::off(j);
        int wi = pos >> 5, sh = pos & 31;
        hv[j] = exl3_cb2_h(exl3_extract16(w[wi], w[wi + 1], sh));
    }
}

static __device__ __forceinline__ u32 exl3_f16_bits(float f) {
    u32 r;
    asm("{ .reg .b16 h; cvt.rn.f16.f32 h, %1; cvt.u32.u16 %0, h; }" : "=r"(r) : "f"(f));
    return r;
}

// G: a routed expert's group of up to 64 tokens (record gi: n, 64 input rows, 64 output rows; the matrix w is the group's)
// PRE: the input is `xr`, already rotated and rounded to f16 by `exl3_rotate_rows` (the rotation is done once a matrix, not once a 128-output block)
template <int KB, bool G, bool PRE = false>
static __device__ __forceinline__ void exl3_mma(const u8* w, const float* x, u32 xs, float* y, u32 ys, u32 rows, u32 cols, u32 m, const u32* gi) {
    if (G) m = gi[0];
    if (G && m == 0) return;  // an empty group of the device planner (its tables are sized for the most groups there could be, STATUS 163): its weight address is 0
    __shared__ __align__(16) u8 smem[64 * 132 * 4];
    u16* sa = (u16*) smem;      // [64][136] f16 (the input's rotated 128-block)
    float* sy = (float*) smem;  // [64][132] f32 (the accumulated outputs; the same storage, after the last block)
    const u32 ob = blockIdx.x, tok0 = G ? 0 : blockIdx.y * 64;
    const u16* suh = (const u16*) w;
    const u16* svh = (const u16*) (w + 2 * cols);
    const u32* tr = (const u32*) (w + 2 * cols + 2 * rows + 4);
    const u32 nt = rows / 16;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int g = lane >> 2, t = lane & 3;
    float acc[4][2][4];
    #pragma unroll
    for (int a = 0; a < 4; a++)
        #pragma unroll
        for (int h = 0; h < 2; h++)
            #pragma unroll
            for (int i = 0; i < 4; i++) acc[a][h][i] = 0.0f;
    for (u32 b = 0; b < cols / 128; b++) {
        if (PRE) {
            // 64 rows x 128 f16 = 1,024 16-byte pieces, four a thread
            const uint4* xr = (const uint4*) x;
            #pragma unroll
            for (int q = 0; q < 4; q++) {
                const int idx = threadIdx.x + 256 * q, r = idx >> 4, c16 = idx & 15;
                const u32 tok = tok0 + r;
                uint4 val = make_uint4(0, 0, 0, 0);
                if (tok < m) val = xr[(((G ? (u64) blockIdx.y * 64 + r : (u64) tok)) * cols + b * 128) / 8 + c16];
                *(uint4*) (sa + r * 136 + c16 * 8) = val;
            }
        }
        if (!PRE) for (int rr = 0; rr < 8; rr++) {
            const int r = warp * 8 + rr;
            const u32 tok = tok0 + r;
            if (tok < m) {
                float v[4];
                #pragma unroll
                for (int e = 0; e < 4; e++) v[e] = x[(u64) (G ? gi[1 + r] : tok) * xs + b * 128 + lane + 32 * e] * f16_to_f32(suh[b * 128 + lane + 32 * e]) * 0.08838834764831845f;
                exl3_wht4(v, lane);
                #pragma unroll
                for (int e = 0; e < 4; e++) sa[r * 136 + lane + 32 * e] = (u16) exl3_f16_bits(v[e]);
            } else {
                #pragma unroll
                for (int e = 0; e < 4; e++) sa[r * 136 + lane + 32 * e] = 0;
            }
        }
        __syncthreads();
        #pragma unroll 2
        for (int kk = 0; kk < 8; kk++) {
            const u32* tile = tr + ((u64) (b * 8 + kk) * nt + ob * 8 + warp) * (4 * KB);
            u32 hv[8];
            exl3_lane_h<KB>(tile, lane, hv);
            const u32 b0l = hv[0] | (hv[1] << 16), b1l = hv[2] | (hv[3] << 16), b0h = hv[4] | (hv[5] << 16), b1h = hv[6] | (hv[7] << 16);
            #pragma unroll
            for (int tt = 0; tt < 4; tt++) {
                const u16* ar = sa + (tt * 16 + g) * 136 + kk * 16 + 2 * t;
                const u32 a0 = *(const u32*) ar, a1 = *(const u32*) (ar + 8 * 136), a2 = *(const u32*) (ar + 8), a3 = *(const u32*) (ar + 8 * 136 + 8);
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                             : "+f"(acc[tt][0][0]), "+f"(acc[tt][0][1]), "+f"(acc[tt][0][2]), "+f"(acc[tt][0][3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0l), "r"(b1l));
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                             : "+f"(acc[tt][1][0]), "+f"(acc[tt][1][1]), "+f"(acc[tt][1][2]), "+f"(acc[tt][1][3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0h), "r"(b1h));
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for (int tt = 0; tt < 4; tt++) {
        #pragma unroll
        for (int h = 0; h < 2; h++) {
            const int c = warp * 16 + h * 8 + 2 * t;
            sy[(tt * 16 + g) * 132 + c] = acc[tt][h][0];
            sy[(tt * 16 + g) * 132 + c + 1] = acc[tt][h][1];
            sy[(tt * 16 + g + 8) * 132 + c] = acc[tt][h][2];
            sy[(tt * 16 + g + 8) * 132 + c + 1] = acc[tt][h][3];
        }
    }
    __syncthreads();
    for (int rr = 0; rr < 8; rr++) {
        const int r = warp * 8 + rr;
        const u32 tok = tok0 + r;
        if (tok >= m) break;
        float v[4];
        #pragma unroll
        for (int e = 0; e < 4; e++) v[e] = sy[r * 132 + lane + 32 * e];
        exl3_wht4(v, lane);
        #pragma unroll
        for (int e = 0; e < 4; e++) y[(u64) (G ? gi[65 + r] : tok) * ys + ob * 128 + lane + 32 * e] = v[e] * 0.08838834764831845f * f16_to_f32(svh[ob * 128 + lane + 32 * e]);
    }
}

// The input rotation of a matrix, once: xr[j][i] = f16(H (x_j suh) / sqrt(128)) on every 128-block of the m rows (a warp a row and block), the
// same arithmetic as the rotation inside `exl3_mma`.  `w` is the matrix's blob (suh is its first `cols` f16).
extern "C" __global__ void exl3_rotate_rows(const u8* w, const float* x, u32 xs, u16* xr, u32 cols, u32 m) {
    const u32 nb = cols / 128;
    const u32 job = blockIdx.x * 8 + (threadIdx.x >> 5);
    const int lane = threadIdx.x & 31;
    if (job >= m * nb) return;
    const u32 tok = job / nb, b = job % nb;
    const u16* suh = (const u16*) w;
    float v[4];
    #pragma unroll
    for (int e = 0; e < 4; e++) v[e] = x[(u64) tok * xs + b * 128 + lane + 32 * e] * f16_to_f32(suh[b * 128 + lane + 32 * e]) * 0.08838834764831845f;
    exl3_wht4(v, lane);
    #pragma unroll
    for (int e = 0; e < 4; e++) xr[(u64) tok * cols + b * 128 + lane + 32 * e] = (u16) exl3_f16_bits(v[e]);
}

// The f32 input rotation for pre-rotated decode (`exl3_mvc2_pre`): xr[j][i] = H(x_j * suh), without 1/sqrt(128) (applied at the output).
extern "C" __global__ void exl3_rotate_rows_f32(const u8* w, const float* x, u32 xs, float* xr, u32 cols, u32 m) {
    const u32 nb = cols / 128;
    const u32 job = blockIdx.x * 8 + (threadIdx.x >> 5);
    const int lane = threadIdx.x & 31;
    if (job >= m * nb) return;
    const u32 tok = job / nb, b = job % nb;
    const u16* suh = (const u16*) w;
    pdl_wait();
    float v[4];
    #pragma unroll
    for (int e = 0; e < 4; e++) v[e] = x[(u64) tok * xs + b * 128 + lane + 32 * e] * f16_to_f32(suh[b * 128 + lane + 32 * e]);
    exl3_wht4(v, lane);
    #pragma unroll
    for (int e = 0; e < 4; e++) xr[(u64) tok * cols + b * 128 + lane + 32 * e] = v[e];
    pdl_launch_dependents();
}

// The input rotation of a batch of routed-expert groups, once a group (each expert has its own suh): group g = blockIdx.y of the table slice, whose n <= 64 tokens
// (rows gi[1 + r] of x) become f16 rows 64 g + r of xr; a warp a (token, 128-block) job, 8 a block (grid.x = 8 jobs x ceil(64 cols / 128 / 8)).
extern "C" __global__ void exl3_rotate_groups(const u64* wtab, const u32* gtab, const float* x, u32 xs, u16* xr, u32 cols) {
    const u32 nb = cols / 128;
    const u32* gi = gtab + (u64) blockIdx.y * 129;
    const u32 job = blockIdx.x * 8 + (threadIdx.x >> 5);
    const int lane = threadIdx.x & 31;
    const u32 tok = job / nb, b = job % nb;
    if (tok >= gi[0]) return;
    const u16* suh = (const u16*) wtab[blockIdx.y];
    float v[4];
    #pragma unroll
    for (int e = 0; e < 4; e++) v[e] = x[(u64) gi[1 + tok] * xs + b * 128 + lane + 32 * e] * f16_to_f32(suh[b * 128 + lane + 32 * e]) * 0.08838834764831845f;
    exl3_wht4(v, lane);
    #pragma unroll
    for (int e = 0; e < 4; e++) xr[((u64) blockIdx.y * 64 + tok) * cols + b * 128 + lane + 32 * e] = (u16) exl3_f16_bits(v[e]);
}

// The second half of a split-K launch: y_j = rotate(sum over the nsplit partials in order) for j < m, a warp an input, a block 128 outputs.
extern "C" __global__ void exl3_splitk_combine(const u8* w, const float* part, float* y, u32 ys, u32 rows, u32 cols, u32 m, u32 nsplit) {
    const u32 ob = blockIdx.x;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    if ((u32) warp >= m) return;
    const u16* svh = (const u16*) (w + 2 * cols);
    pdl_wait();
    float v[4];
    #pragma unroll
    for (int e = 0; e < 4; e++) {
        float t = part[(u64) warp * rows + ob * 128 + lane + 32 * e];
        for (u32 q = 1; q < nsplit; q++) t += part[((u64) q * 8 + warp) * rows + ob * 128 + lane + 32 * e];
        v[e] = t * 0.08838834764831845f;
    }
    exl3_wht4(v, lane);
    #pragma unroll
    for (int e = 0; e < 4; e++) y[(u64) warp * ys + ob * 128 + lane + 32 * e] = v[e] * 0.08838834764831845f * f16_to_f32(svh[ob * 128 + lane + 32 * e]);
    pdl_launch_dependents();
}

// all 65,536 codebook values (a test of the decode against the CPU reference's)
extern "C" __global__ void exl3_codebook(float* out, u32 n) {
    u32 i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = exl3_cb2(i);
}


extern "C" __global__ void matvec_exl3_1(const u8* w, const float* x, float* y, u32 rows, u32 cols) { exl3_mv<2, 1>(w, x, 0, y, 0, rows, cols, 1); }
extern "C" __global__ void matvec_cols_exl3_1(const u8* w, const float* x, u32 xs, float* y, u32 ys, u32 rows, u32 cols, u32 m) { exl3_mv<2, 8>(w, x, xs, y, ys, rows, cols, m); }
extern "C" __global__ void matvec_batch_exl3_1(const u64* wtab, const float* x, u32 xs, float* y, u32 ys, u32 rows, u32 cols) { exl3_mv<2, 1>((const u8*) wtab[blockIdx.y], x + (u64) blockIdx.y * xs, 0, y + (u64) blockIdx.y * ys, 0, rows, cols, 1); }
extern "C" __global__ void __launch_bounds__(512) matvec_cols_exl3_s2_1(const u8* w, const float* x, u32 xs, float* y, u32 ys, u32 rows, u32 cols, u32 m) { exl3_mvc2<2, 8, 2, false>(w, x, xs, y, ys, rows, cols, m, nullptr); }
extern "C" __global__ void __launch_bounds__(1024) matvec_cols_exl3_s4_1(const u8* w, const float* x, u32 xs, float* y, u32 ys, u32 rows, u32 cols, u32 m) { exl3_mvc2<2, 8, 4, false>(w, x, xs, y, ys, rows, cols, m, nullptr); }
extern "C" __global__ void __launch_bounds__(1024) matvec_cols_exl3_m4s4_1(const u8* w, const float* x, u32 xs, float* y, u32 ys, u32 rows, u32 cols, u32 m) { exl3_mvc2<2, 4, 4, false>(w, x, xs, y, ys, rows, cols, m, nullptr); }
extern "C" __global__ void __launch_bounds__(1024) matvec_cols_exl3_sk4_1(const u8* w, const float* x, u32 xs, float* part, u32 rows, u32 cols, u32 m) { exl3_mvc2<2, 8, 4, false, true>(w, x, xs, part, 0, rows, cols, m, nullptr); }
extern "C" __global__ void __launch_bounds__(1024) matvec_grouped_exl3_m2s4_1(const u64* wtab, const u32* gtab, const float* x, u32 xs, float* y, u32 ys, u32 rows, u32 cols) { exl3_mvc2<2, 2, 4, true>((const u8*) wtab[blockIdx.y], x, xs, y, ys, rows, cols, 0, gtab + (u64) blockIdx.y * 17); }
extern "C" __global__ void __launch_bounds__(1024) matvec_grouped_exl3_m4s4_1(const u64* wtab, const u32* gtab, const float* x, u32 xs, float* y, u32 ys, u32 rows, u32 cols) { exl3_mvc2<2, 4, 4, true>((const u8*) wtab[blockIdx.y], x, xs, y, ys, rows, cols, 0, gtab + (u64) blockIdx.y * 17); }
extern "C" __global__ void __launch_bounds__(1024) matvec_grouped_exl3_s4_1(const u64* wtab, const u32* gtab, const float* x, u32 xs, float* y, u32 ys, u32 rows, u32 cols) { exl3_mvc2<2, 8, 4, true>((const u8*) wtab[blockIdx.y], x, xs, y, ys, rows, cols, 0, gtab + (u64) blockIdx.y * 17); }
extern "C" __global__ void __launch_bounds__(256) mma_cols_exl3_1(const u8* w, const float* x, u32 xs, float* y, u32 ys, u32 rows, u32 cols, u32 m) { exl3_mma<2, false>(w, x, xs, y, ys, rows, cols, m, nullptr); }
extern "C" __global__ void __launch_bounds__(256) mma_grouped_exl3_1(const u64* wtab, const u32* gtab, const float* x, u32 xs, float* y, u32 ys, u32 rows, u32 cols) { exl3_mma<2, true>((const u8*) wtab[blockIdx.y], x, xs, y, ys, rows, cols, 0, gtab + (u64) blockIdx.y * 129); }
extern "C" __global__ void __launch_bounds__(256) mma_grouped_exl3_pre_1(const u64* wtab, const u32* gtab, const float* x, u32 xs, float* y, u32 ys, u32 rows, u32 cols) { exl3_mma<2, true, true>((const u8*) wtab[blockIdx.y], x, xs, y, ys, rows, cols, 0, gtab + (u64) blockIdx.y * 129); }
extern "C" __global__ void __launch_bounds__(256) mma_cols_exl3_pre_1(const u8* w, const float* x, u32 xs, float* y, u32 ys, u32 rows, u32 cols, u32 m) { exl3_mma<2, false, true>(w, x, xs, y, ys, rows, cols, m, nullptr); }
extern "C" __global__ void __launch_bounds__(256) matvec_exl3_s1_1(const u8* w, const float* x, float* y, u32 rows, u32 cols) { exl3_mv2<2, 1>(w, x, y, rows, cols); }
extern "C" __global__ void __launch_bounds__(256) matvec_batch_exl3_s1_1(const u64* wtab, const float* x, u32 xs, float* y, u32 ys, u32 rows, u32 cols) { exl3_mv2<2, 1>((const u8*) wtab[blockIdx.y], x + (u64) blockIdx.y * xs, y + (u64) blockIdx.y * ys, rows, cols); }
extern "C" __global__ void __launch_bounds__(512) matvec_exl3_s2_1(const u8* w, const float* x, float* y, u32 rows, u32 cols) { exl3_mv2<2, 2>(w, x, y, rows, cols); }
extern "C" __global__ void __launch_bounds__(512) matvec_batch_exl3_s2_1(const u64* wtab, const float* x, u32 xs, float* y, u32 ys, u32 rows, u32 cols) { exl3_mv2<2, 2>((const u8*) wtab[blockIdx.y], x + (u64) blockIdx.y * xs, y + (u64) blockIdx.y * ys, rows, cols); }
extern "C" __global__ void __launch_bounds__(1024) matvec_exl3_s4_1(const u8* w, const float* x, float* y, u32 rows, u32 cols) { exl3_mv2<2, 4>(w, x, y, rows, cols); }
extern "C" __global__ void __launch_bounds__(1024) matvec_batch_exl3_s4_1(const u64* wtab, const float* x, u32 xs, float* y, u32 ys, u32 rows, u32 cols) { exl3_mv2<2, 4>((const u8*) wtab[blockIdx.y], x + (u64) blockIdx.y * xs, y + (u64) blockIdx.y * ys, rows, cols); }
extern "C" __global__ void matvec_exl3_2(const u8* w, const float* x, float* y, u32 rows, u32 cols) { exl3_mv<4, 1>(w, x, 0, y, 0, rows, cols, 1); }
extern "C" __global__ void matvec_cols_exl3_2(const u8* w, const float* x, u32 xs, float* y, u32 ys, u32 rows, u32 cols, u32 m) { exl3_mv<4, 8>(w, x, xs, y, ys, rows, cols, m); }
extern "C" __global__ void matvec_batch_exl3_2(const u64* wtab, const float* x, u32 xs, float* y, u32 ys, u32 rows, u32 cols) { exl3_mv<4, 1>((const u8*) wtab[blockIdx.y], x + (u64) blockIdx.y * xs, 0, y + (u64) blockIdx.y * ys, 0, rows, cols, 1); }
extern "C" __global__ void __launch_bounds__(512) matvec_cols_exl3_s2_2(const u8* w, const float* x, u32 xs, float* y, u32 ys, u32 rows, u32 cols, u32 m) { exl3_mvc2<4, 8, 2, false>(w, x, xs, y, ys, rows, cols, m, nullptr); }
extern "C" __global__ void __launch_bounds__(1024) matvec_cols_exl3_s4_2(const u8* w, const float* x, u32 xs, float* y, u32 ys, u32 rows, u32 cols, u32 m) { exl3_mvc2<4, 8, 4, false>(w, x, xs, y, ys, rows, cols, m, nullptr); }
extern "C" __global__ void __launch_bounds__(1024) matvec_cols_exl3_m4s4_2(const u8* w, const float* x, u32 xs, float* y, u32 ys, u32 rows, u32 cols, u32 m) { exl3_mvc2<4, 4, 4, false>(w, x, xs, y, ys, rows, cols, m, nullptr); }
extern "C" __global__ void __launch_bounds__(1024) matvec_cols_exl3_sk4_2(const u8* w, const float* x, u32 xs, float* part, u32 rows, u32 cols, u32 m) { exl3_mvc2<4, 8, 4, false, true>(w, x, xs, part, 0, rows, cols, m, nullptr); }
extern "C" __global__ void __launch_bounds__(1024) matvec_grouped_exl3_m2s4_2(const u64* wtab, const u32* gtab, const float* x, u32 xs, float* y, u32 ys, u32 rows, u32 cols) { exl3_mvc2<4, 2, 4, true>((const u8*) wtab[blockIdx.y], x, xs, y, ys, rows, cols, 0, gtab + (u64) blockIdx.y * 17); }
extern "C" __global__ void __launch_bounds__(1024) matvec_grouped_exl3_m4s4_2(const u64* wtab, const u32* gtab, const float* x, u32 xs, float* y, u32 ys, u32 rows, u32 cols) { exl3_mvc2<4, 4, 4, true>((const u8*) wtab[blockIdx.y], x, xs, y, ys, rows, cols, 0, gtab + (u64) blockIdx.y * 17); }
extern "C" __global__ void __launch_bounds__(1024) matvec_grouped_exl3_s4_2(const u64* wtab, const u32* gtab, const float* x, u32 xs, float* y, u32 ys, u32 rows, u32 cols) { exl3_mvc2<4, 8, 4, true>((const u8*) wtab[blockIdx.y], x, xs, y, ys, rows, cols, 0, gtab + (u64) blockIdx.y * 17); }
extern "C" __global__ void __launch_bounds__(256) mma_cols_exl3_2(const u8* w, const float* x, u32 xs, float* y, u32 ys, u32 rows, u32 cols, u32 m) { exl3_mma<4, false>(w, x, xs, y, ys, rows, cols, m, nullptr); }
extern "C" __global__ void __launch_bounds__(256) mma_grouped_exl3_2(const u64* wtab, const u32* gtab, const float* x, u32 xs, float* y, u32 ys, u32 rows, u32 cols) { exl3_mma<4, true>((const u8*) wtab[blockIdx.y], x, xs, y, ys, rows, cols, 0, gtab + (u64) blockIdx.y * 129); }
extern "C" __global__ void __launch_bounds__(256) mma_grouped_exl3_pre_2(const u64* wtab, const u32* gtab, const float* x, u32 xs, float* y, u32 ys, u32 rows, u32 cols) { exl3_mma<4, true, true>((const u8*) wtab[blockIdx.y], x, xs, y, ys, rows, cols, 0, gtab + (u64) blockIdx.y * 129); }
extern "C" __global__ void __launch_bounds__(256) mma_cols_exl3_pre_2(const u8* w, const float* x, u32 xs, float* y, u32 ys, u32 rows, u32 cols, u32 m) { exl3_mma<4, false, true>(w, x, xs, y, ys, rows, cols, m, nullptr); }
extern "C" __global__ void __launch_bounds__(256) matvec_exl3_s1_2(const u8* w, const float* x, float* y, u32 rows, u32 cols) { exl3_mv2<4, 1>(w, x, y, rows, cols); }
extern "C" __global__ void __launch_bounds__(256) matvec_batch_exl3_s1_2(const u64* wtab, const float* x, u32 xs, float* y, u32 ys, u32 rows, u32 cols) { exl3_mv2<4, 1>((const u8*) wtab[blockIdx.y], x + (u64) blockIdx.y * xs, y + (u64) blockIdx.y * ys, rows, cols); }
extern "C" __global__ void __launch_bounds__(512) matvec_exl3_s2_2(const u8* w, const float* x, float* y, u32 rows, u32 cols) { exl3_mv2<4, 2>(w, x, y, rows, cols); }
extern "C" __global__ void __launch_bounds__(512) matvec_batch_exl3_s2_2(const u64* wtab, const float* x, u32 xs, float* y, u32 ys, u32 rows, u32 cols) { exl3_mv2<4, 2>((const u8*) wtab[blockIdx.y], x + (u64) blockIdx.y * xs, y + (u64) blockIdx.y * ys, rows, cols); }
extern "C" __global__ void __launch_bounds__(1024) matvec_exl3_s4_2(const u8* w, const float* x, float* y, u32 rows, u32 cols) { exl3_mv2<4, 4>(w, x, y, rows, cols); }
extern "C" __global__ void __launch_bounds__(1024) matvec_batch_exl3_s4_2(const u64* wtab, const float* x, u32 xs, float* y, u32 ys, u32 rows, u32 cols) { exl3_mv2<4, 4>((const u8*) wtab[blockIdx.y], x + (u64) blockIdx.y * xs, y + (u64) blockIdx.y * ys, rows, cols); }
extern "C" __global__ void matvec_exl3_3(const u8* w, const float* x, float* y, u32 rows, u32 cols) { exl3_mv<6, 1>(w, x, 0, y, 0, rows, cols, 1); }
extern "C" __global__ void matvec_cols_exl3_3(const u8* w, const float* x, u32 xs, float* y, u32 ys, u32 rows, u32 cols, u32 m) { exl3_mv<6, 8>(w, x, xs, y, ys, rows, cols, m); }
extern "C" __global__ void matvec_batch_exl3_3(const u64* wtab, const float* x, u32 xs, float* y, u32 ys, u32 rows, u32 cols) { exl3_mv<6, 1>((const u8*) wtab[blockIdx.y], x + (u64) blockIdx.y * xs, 0, y + (u64) blockIdx.y * ys, 0, rows, cols, 1); }
extern "C" __global__ void __launch_bounds__(512) matvec_cols_exl3_s2_3(const u8* w, const float* x, u32 xs, float* y, u32 ys, u32 rows, u32 cols, u32 m) { exl3_mvc2<6, 8, 2, false>(w, x, xs, y, ys, rows, cols, m, nullptr); }
extern "C" __global__ void __launch_bounds__(1024) matvec_cols_exl3_s4_3(const u8* w, const float* x, u32 xs, float* y, u32 ys, u32 rows, u32 cols, u32 m) { exl3_mvc2<6, 8, 4, false>(w, x, xs, y, ys, rows, cols, m, nullptr); }
extern "C" __global__ void __launch_bounds__(1024) matvec_cols_exl3_m4s4_3(const u8* w, const float* x, u32 xs, float* y, u32 ys, u32 rows, u32 cols, u32 m) { exl3_mvc2<6, 4, 4, false>(w, x, xs, y, ys, rows, cols, m, nullptr); }
extern "C" __global__ void __launch_bounds__(1024) matvec_cols_exl3_sk4_3(const u8* w, const float* x, u32 xs, float* part, u32 rows, u32 cols, u32 m) { exl3_mvc2<6, 8, 4, false, true>(w, x, xs, part, 0, rows, cols, m, nullptr); }
extern "C" __global__ void __launch_bounds__(1024) matvec_grouped_exl3_m2s4_3(const u64* wtab, const u32* gtab, const float* x, u32 xs, float* y, u32 ys, u32 rows, u32 cols) { exl3_mvc2<6, 2, 4, true>((const u8*) wtab[blockIdx.y], x, xs, y, ys, rows, cols, 0, gtab + (u64) blockIdx.y * 17); }
extern "C" __global__ void __launch_bounds__(1024) matvec_grouped_exl3_m4s4_3(const u64* wtab, const u32* gtab, const float* x, u32 xs, float* y, u32 ys, u32 rows, u32 cols) { exl3_mvc2<6, 4, 4, true>((const u8*) wtab[blockIdx.y], x, xs, y, ys, rows, cols, 0, gtab + (u64) blockIdx.y * 17); }
extern "C" __global__ void __launch_bounds__(1024) matvec_grouped_exl3_s4_3(const u64* wtab, const u32* gtab, const float* x, u32 xs, float* y, u32 ys, u32 rows, u32 cols) { exl3_mvc2<6, 8, 4, true>((const u8*) wtab[blockIdx.y], x, xs, y, ys, rows, cols, 0, gtab + (u64) blockIdx.y * 17); }
extern "C" __global__ void __launch_bounds__(256) mma_cols_exl3_3(const u8* w, const float* x, u32 xs, float* y, u32 ys, u32 rows, u32 cols, u32 m) { exl3_mma<6, false>(w, x, xs, y, ys, rows, cols, m, nullptr); }
extern "C" __global__ void __launch_bounds__(256) mma_grouped_exl3_3(const u64* wtab, const u32* gtab, const float* x, u32 xs, float* y, u32 ys, u32 rows, u32 cols) { exl3_mma<6, true>((const u8*) wtab[blockIdx.y], x, xs, y, ys, rows, cols, 0, gtab + (u64) blockIdx.y * 129); }
extern "C" __global__ void __launch_bounds__(256) mma_grouped_exl3_pre_3(const u64* wtab, const u32* gtab, const float* x, u32 xs, float* y, u32 ys, u32 rows, u32 cols) { exl3_mma<6, true, true>((const u8*) wtab[blockIdx.y], x, xs, y, ys, rows, cols, 0, gtab + (u64) blockIdx.y * 129); }
extern "C" __global__ void __launch_bounds__(256) mma_cols_exl3_pre_3(const u8* w, const float* x, u32 xs, float* y, u32 ys, u32 rows, u32 cols, u32 m) { exl3_mma<6, false, true>(w, x, xs, y, ys, rows, cols, m, nullptr); }
extern "C" __global__ void __launch_bounds__(256) matvec_exl3_s1_3(const u8* w, const float* x, float* y, u32 rows, u32 cols) { exl3_mv2<6, 1>(w, x, y, rows, cols); }
extern "C" __global__ void __launch_bounds__(256) matvec_batch_exl3_s1_3(const u64* wtab, const float* x, u32 xs, float* y, u32 ys, u32 rows, u32 cols) { exl3_mv2<6, 1>((const u8*) wtab[blockIdx.y], x + (u64) blockIdx.y * xs, y + (u64) blockIdx.y * ys, rows, cols); }
extern "C" __global__ void __launch_bounds__(512) matvec_exl3_s2_3(const u8* w, const float* x, float* y, u32 rows, u32 cols) { exl3_mv2<6, 2>(w, x, y, rows, cols); }
extern "C" __global__ void __launch_bounds__(512) matvec_batch_exl3_s2_3(const u64* wtab, const float* x, u32 xs, float* y, u32 ys, u32 rows, u32 cols) { exl3_mv2<6, 2>((const u8*) wtab[blockIdx.y], x + (u64) blockIdx.y * xs, y + (u64) blockIdx.y * ys, rows, cols); }
extern "C" __global__ void __launch_bounds__(1024) matvec_exl3_s4_3(const u8* w, const float* x, float* y, u32 rows, u32 cols) { exl3_mv2<6, 4>(w, x, y, rows, cols); }
extern "C" __global__ void __launch_bounds__(1024) matvec_batch_exl3_s4_3(const u64* wtab, const float* x, u32 xs, float* y, u32 ys, u32 rows, u32 cols) { exl3_mv2<6, 4>((const u8*) wtab[blockIdx.y], x + (u64) blockIdx.y * xs, y + (u64) blockIdx.y * ys, rows, cols); }
extern "C" __global__ void matvec_exl3_4(const u8* w, const float* x, float* y, u32 rows, u32 cols) { exl3_mv<8, 1>(w, x, 0, y, 0, rows, cols, 1); }
extern "C" __global__ void matvec_cols_exl3_4(const u8* w, const float* x, u32 xs, float* y, u32 ys, u32 rows, u32 cols, u32 m) { exl3_mv<8, 8>(w, x, xs, y, ys, rows, cols, m); }
extern "C" __global__ void matvec_batch_exl3_4(const u64* wtab, const float* x, u32 xs, float* y, u32 ys, u32 rows, u32 cols) { exl3_mv<8, 1>((const u8*) wtab[blockIdx.y], x + (u64) blockIdx.y * xs, 0, y + (u64) blockIdx.y * ys, 0, rows, cols, 1); }
extern "C" __global__ void __launch_bounds__(512) matvec_cols_exl3_s2_4(const u8* w, const float* x, u32 xs, float* y, u32 ys, u32 rows, u32 cols, u32 m) { exl3_mvc2<8, 8, 2, false>(w, x, xs, y, ys, rows, cols, m, nullptr); }
extern "C" __global__ void __launch_bounds__(1024) matvec_cols_exl3_s4_4(const u8* w, const float* x, u32 xs, float* y, u32 ys, u32 rows, u32 cols, u32 m) { exl3_mvc2<8, 8, 4, false>(w, x, xs, y, ys, rows, cols, m, nullptr); }
extern "C" __global__ void __launch_bounds__(1024) matvec_cols_exl3_m4s4_4(const u8* w, const float* x, u32 xs, float* y, u32 ys, u32 rows, u32 cols, u32 m) { exl3_mvc2<8, 4, 4, false>(w, x, xs, y, ys, rows, cols, m, nullptr); }
extern "C" __global__ void __launch_bounds__(1024) matvec_cols_exl3_sk4_4(const u8* w, const float* x, u32 xs, float* part, u32 rows, u32 cols, u32 m) { exl3_mvc2<8, 8, 4, false, true>(w, x, xs, part, 0, rows, cols, m, nullptr); }
extern "C" __global__ void __launch_bounds__(1024) matvec_grouped_exl3_m2s4_4(const u64* wtab, const u32* gtab, const float* x, u32 xs, float* y, u32 ys, u32 rows, u32 cols) { exl3_mvc2<8, 2, 4, true>((const u8*) wtab[blockIdx.y], x, xs, y, ys, rows, cols, 0, gtab + (u64) blockIdx.y * 17); }
extern "C" __global__ void __launch_bounds__(1024) matvec_grouped_exl3_m4s4_4(const u64* wtab, const u32* gtab, const float* x, u32 xs, float* y, u32 ys, u32 rows, u32 cols) { exl3_mvc2<8, 4, 4, true>((const u8*) wtab[blockIdx.y], x, xs, y, ys, rows, cols, 0, gtab + (u64) blockIdx.y * 17); }
extern "C" __global__ void __launch_bounds__(1024) matvec_grouped_exl3_s4_4(const u64* wtab, const u32* gtab, const float* x, u32 xs, float* y, u32 ys, u32 rows, u32 cols) { exl3_mvc2<8, 8, 4, true>((const u8*) wtab[blockIdx.y], x, xs, y, ys, rows, cols, 0, gtab + (u64) blockIdx.y * 17); }
extern "C" __global__ void __launch_bounds__(256) mma_cols_exl3_4(const u8* w, const float* x, u32 xs, float* y, u32 ys, u32 rows, u32 cols, u32 m) { exl3_mma<8, false>(w, x, xs, y, ys, rows, cols, m, nullptr); }
extern "C" __global__ void __launch_bounds__(256) mma_grouped_exl3_4(const u64* wtab, const u32* gtab, const float* x, u32 xs, float* y, u32 ys, u32 rows, u32 cols) { exl3_mma<8, true>((const u8*) wtab[blockIdx.y], x, xs, y, ys, rows, cols, 0, gtab + (u64) blockIdx.y * 129); }
extern "C" __global__ void __launch_bounds__(256) mma_grouped_exl3_pre_4(const u64* wtab, const u32* gtab, const float* x, u32 xs, float* y, u32 ys, u32 rows, u32 cols) { exl3_mma<8, true, true>((const u8*) wtab[blockIdx.y], x, xs, y, ys, rows, cols, 0, gtab + (u64) blockIdx.y * 129); }
extern "C" __global__ void __launch_bounds__(256) mma_cols_exl3_pre_4(const u8* w, const float* x, u32 xs, float* y, u32 ys, u32 rows, u32 cols, u32 m) { exl3_mma<8, false, true>(w, x, xs, y, ys, rows, cols, m, nullptr); }
extern "C" __global__ void __launch_bounds__(256) matvec_exl3_s1_4(const u8* w, const float* x, float* y, u32 rows, u32 cols) { exl3_mv2<8, 1>(w, x, y, rows, cols); }
extern "C" __global__ void __launch_bounds__(256) matvec_batch_exl3_s1_4(const u64* wtab, const float* x, u32 xs, float* y, u32 ys, u32 rows, u32 cols) { exl3_mv2<8, 1>((const u8*) wtab[blockIdx.y], x + (u64) blockIdx.y * xs, y + (u64) blockIdx.y * ys, rows, cols); }
extern "C" __global__ void __launch_bounds__(512) matvec_exl3_s2_4(const u8* w, const float* x, float* y, u32 rows, u32 cols) { exl3_mv2<8, 2>(w, x, y, rows, cols); }
extern "C" __global__ void __launch_bounds__(512) matvec_batch_exl3_s2_4(const u64* wtab, const float* x, u32 xs, float* y, u32 ys, u32 rows, u32 cols) { exl3_mv2<8, 2>((const u8*) wtab[blockIdx.y], x + (u64) blockIdx.y * xs, y + (u64) blockIdx.y * ys, rows, cols); }
extern "C" __global__ void __launch_bounds__(1024) matvec_exl3_s4_4(const u8* w, const float* x, float* y, u32 rows, u32 cols) { exl3_mv2<8, 4>(w, x, y, rows, cols); }
extern "C" __global__ void __launch_bounds__(1024) matvec_batch_exl3_s4_4(const u64* wtab, const float* x, u32 xs, float* y, u32 ys, u32 rows, u32 cols) { exl3_mv2<8, 4>((const u8*) wtab[blockIdx.y], x + (u64) blockIdx.y * xs, y + (u64) blockIdx.y * ys, rows, cols); }
extern "C" __global__ void matvec_exl3_5(const u8* w, const float* x, float* y, u32 rows, u32 cols) { exl3_mv<10, 1>(w, x, 0, y, 0, rows, cols, 1); }
extern "C" __global__ void matvec_cols_exl3_5(const u8* w, const float* x, u32 xs, float* y, u32 ys, u32 rows, u32 cols, u32 m) { exl3_mv<10, 8>(w, x, xs, y, ys, rows, cols, m); }
extern "C" __global__ void matvec_batch_exl3_5(const u64* wtab, const float* x, u32 xs, float* y, u32 ys, u32 rows, u32 cols) { exl3_mv<10, 1>((const u8*) wtab[blockIdx.y], x + (u64) blockIdx.y * xs, 0, y + (u64) blockIdx.y * ys, 0, rows, cols, 1); }
extern "C" __global__ void __launch_bounds__(512) matvec_cols_exl3_s2_5(const u8* w, const float* x, u32 xs, float* y, u32 ys, u32 rows, u32 cols, u32 m) { exl3_mvc2<10, 8, 2, false>(w, x, xs, y, ys, rows, cols, m, nullptr); }
extern "C" __global__ void __launch_bounds__(1024) matvec_cols_exl3_s4_5(const u8* w, const float* x, u32 xs, float* y, u32 ys, u32 rows, u32 cols, u32 m) { exl3_mvc2<10, 8, 4, false>(w, x, xs, y, ys, rows, cols, m, nullptr); }
extern "C" __global__ void __launch_bounds__(1024) matvec_cols_exl3_m4s4_5(const u8* w, const float* x, u32 xs, float* y, u32 ys, u32 rows, u32 cols, u32 m) { exl3_mvc2<10, 4, 4, false>(w, x, xs, y, ys, rows, cols, m, nullptr); }
extern "C" __global__ void __launch_bounds__(1024) matvec_cols_exl3_sk4_5(const u8* w, const float* x, u32 xs, float* part, u32 rows, u32 cols, u32 m) { exl3_mvc2<10, 8, 4, false, true>(w, x, xs, part, 0, rows, cols, m, nullptr); }
extern "C" __global__ void __launch_bounds__(1024) matvec_grouped_exl3_m2s4_5(const u64* wtab, const u32* gtab, const float* x, u32 xs, float* y, u32 ys, u32 rows, u32 cols) { exl3_mvc2<10, 2, 4, true>((const u8*) wtab[blockIdx.y], x, xs, y, ys, rows, cols, 0, gtab + (u64) blockIdx.y * 17); }
extern "C" __global__ void __launch_bounds__(1024) matvec_grouped_exl3_m4s4_5(const u64* wtab, const u32* gtab, const float* x, u32 xs, float* y, u32 ys, u32 rows, u32 cols) { exl3_mvc2<10, 4, 4, true>((const u8*) wtab[blockIdx.y], x, xs, y, ys, rows, cols, 0, gtab + (u64) blockIdx.y * 17); }
extern "C" __global__ void __launch_bounds__(1024) matvec_grouped_exl3_s4_5(const u64* wtab, const u32* gtab, const float* x, u32 xs, float* y, u32 ys, u32 rows, u32 cols) { exl3_mvc2<10, 8, 4, true>((const u8*) wtab[blockIdx.y], x, xs, y, ys, rows, cols, 0, gtab + (u64) blockIdx.y * 17); }
extern "C" __global__ void __launch_bounds__(256) mma_cols_exl3_5(const u8* w, const float* x, u32 xs, float* y, u32 ys, u32 rows, u32 cols, u32 m) { exl3_mma<10, false>(w, x, xs, y, ys, rows, cols, m, nullptr); }
extern "C" __global__ void __launch_bounds__(256) mma_grouped_exl3_5(const u64* wtab, const u32* gtab, const float* x, u32 xs, float* y, u32 ys, u32 rows, u32 cols) { exl3_mma<10, true>((const u8*) wtab[blockIdx.y], x, xs, y, ys, rows, cols, 0, gtab + (u64) blockIdx.y * 129); }
extern "C" __global__ void __launch_bounds__(256) mma_grouped_exl3_pre_5(const u64* wtab, const u32* gtab, const float* x, u32 xs, float* y, u32 ys, u32 rows, u32 cols) { exl3_mma<10, true, true>((const u8*) wtab[blockIdx.y], x, xs, y, ys, rows, cols, 0, gtab + (u64) blockIdx.y * 129); }
extern "C" __global__ void __launch_bounds__(256) mma_cols_exl3_pre_5(const u8* w, const float* x, u32 xs, float* y, u32 ys, u32 rows, u32 cols, u32 m) { exl3_mma<10, false, true>(w, x, xs, y, ys, rows, cols, m, nullptr); }
extern "C" __global__ void __launch_bounds__(256) matvec_exl3_s1_5(const u8* w, const float* x, float* y, u32 rows, u32 cols) { exl3_mv2<10, 1>(w, x, y, rows, cols); }
extern "C" __global__ void __launch_bounds__(256) matvec_batch_exl3_s1_5(const u64* wtab, const float* x, u32 xs, float* y, u32 ys, u32 rows, u32 cols) { exl3_mv2<10, 1>((const u8*) wtab[blockIdx.y], x + (u64) blockIdx.y * xs, y + (u64) blockIdx.y * ys, rows, cols); }
extern "C" __global__ void __launch_bounds__(512) matvec_exl3_s2_5(const u8* w, const float* x, float* y, u32 rows, u32 cols) { exl3_mv2<10, 2>(w, x, y, rows, cols); }
extern "C" __global__ void __launch_bounds__(512) matvec_batch_exl3_s2_5(const u64* wtab, const float* x, u32 xs, float* y, u32 ys, u32 rows, u32 cols) { exl3_mv2<10, 2>((const u8*) wtab[blockIdx.y], x + (u64) blockIdx.y * xs, y + (u64) blockIdx.y * ys, rows, cols); }
extern "C" __global__ void __launch_bounds__(1024) matvec_exl3_s4_5(const u8* w, const float* x, float* y, u32 rows, u32 cols) { exl3_mv2<10, 4>(w, x, y, rows, cols); }
extern "C" __global__ void __launch_bounds__(1024) matvec_batch_exl3_s4_5(const u64* wtab, const float* x, u32 xs, float* y, u32 ys, u32 rows, u32 cols) { exl3_mv2<10, 4>((const u8*) wtab[blockIdx.y], x + (u64) blockIdx.y * xs, y + (u64) blockIdx.y * ys, rows, cols); }
extern "C" __global__ void matvec_exl3_6(const u8* w, const float* x, float* y, u32 rows, u32 cols) { exl3_mv<12, 1>(w, x, 0, y, 0, rows, cols, 1); }
extern "C" __global__ void matvec_cols_exl3_6(const u8* w, const float* x, u32 xs, float* y, u32 ys, u32 rows, u32 cols, u32 m) { exl3_mv<12, 8>(w, x, xs, y, ys, rows, cols, m); }
extern "C" __global__ void matvec_batch_exl3_6(const u64* wtab, const float* x, u32 xs, float* y, u32 ys, u32 rows, u32 cols) { exl3_mv<12, 1>((const u8*) wtab[blockIdx.y], x + (u64) blockIdx.y * xs, 0, y + (u64) blockIdx.y * ys, 0, rows, cols, 1); }
extern "C" __global__ void __launch_bounds__(512) matvec_cols_exl3_s2_6(const u8* w, const float* x, u32 xs, float* y, u32 ys, u32 rows, u32 cols, u32 m) { exl3_mvc2<12, 8, 2, false>(w, x, xs, y, ys, rows, cols, m, nullptr); }
extern "C" __global__ void __launch_bounds__(1024) matvec_cols_exl3_s4_6(const u8* w, const float* x, u32 xs, float* y, u32 ys, u32 rows, u32 cols, u32 m) { exl3_mvc2<12, 8, 4, false>(w, x, xs, y, ys, rows, cols, m, nullptr); }
extern "C" __global__ void __launch_bounds__(1024) matvec_cols_exl3_m4s4_6(const u8* w, const float* x, u32 xs, float* y, u32 ys, u32 rows, u32 cols, u32 m) { exl3_mvc2<12, 4, 4, false>(w, x, xs, y, ys, rows, cols, m, nullptr); }
extern "C" __global__ void __launch_bounds__(1024) matvec_cols_exl3_sk4_6(const u8* w, const float* x, u32 xs, float* part, u32 rows, u32 cols, u32 m) { exl3_mvc2<12, 8, 4, false, true>(w, x, xs, part, 0, rows, cols, m, nullptr); }
extern "C" __global__ void __launch_bounds__(1024) matvec_grouped_exl3_m2s4_6(const u64* wtab, const u32* gtab, const float* x, u32 xs, float* y, u32 ys, u32 rows, u32 cols) { exl3_mvc2<12, 2, 4, true>((const u8*) wtab[blockIdx.y], x, xs, y, ys, rows, cols, 0, gtab + (u64) blockIdx.y * 17); }
extern "C" __global__ void __launch_bounds__(1024) matvec_grouped_exl3_m4s4_6(const u64* wtab, const u32* gtab, const float* x, u32 xs, float* y, u32 ys, u32 rows, u32 cols) { exl3_mvc2<12, 4, 4, true>((const u8*) wtab[blockIdx.y], x, xs, y, ys, rows, cols, 0, gtab + (u64) blockIdx.y * 17); }
extern "C" __global__ void __launch_bounds__(1024) matvec_grouped_exl3_s4_6(const u64* wtab, const u32* gtab, const float* x, u32 xs, float* y, u32 ys, u32 rows, u32 cols) { exl3_mvc2<12, 8, 4, true>((const u8*) wtab[blockIdx.y], x, xs, y, ys, rows, cols, 0, gtab + (u64) blockIdx.y * 17); }
extern "C" __global__ void __launch_bounds__(256) mma_cols_exl3_6(const u8* w, const float* x, u32 xs, float* y, u32 ys, u32 rows, u32 cols, u32 m) { exl3_mma<12, false>(w, x, xs, y, ys, rows, cols, m, nullptr); }
extern "C" __global__ void __launch_bounds__(256) mma_grouped_exl3_6(const u64* wtab, const u32* gtab, const float* x, u32 xs, float* y, u32 ys, u32 rows, u32 cols) { exl3_mma<12, true>((const u8*) wtab[blockIdx.y], x, xs, y, ys, rows, cols, 0, gtab + (u64) blockIdx.y * 129); }
extern "C" __global__ void __launch_bounds__(256) mma_grouped_exl3_pre_6(const u64* wtab, const u32* gtab, const float* x, u32 xs, float* y, u32 ys, u32 rows, u32 cols) { exl3_mma<12, true, true>((const u8*) wtab[blockIdx.y], x, xs, y, ys, rows, cols, 0, gtab + (u64) blockIdx.y * 129); }
extern "C" __global__ void __launch_bounds__(256) mma_cols_exl3_pre_6(const u8* w, const float* x, u32 xs, float* y, u32 ys, u32 rows, u32 cols, u32 m) { exl3_mma<12, false, true>(w, x, xs, y, ys, rows, cols, m, nullptr); }
extern "C" __global__ void __launch_bounds__(256) matvec_exl3_s1_6(const u8* w, const float* x, float* y, u32 rows, u32 cols) { exl3_mv2<12, 1>(w, x, y, rows, cols); }
extern "C" __global__ void __launch_bounds__(256) matvec_batch_exl3_s1_6(const u64* wtab, const float* x, u32 xs, float* y, u32 ys, u32 rows, u32 cols) { exl3_mv2<12, 1>((const u8*) wtab[blockIdx.y], x + (u64) blockIdx.y * xs, y + (u64) blockIdx.y * ys, rows, cols); }
extern "C" __global__ void __launch_bounds__(512) matvec_exl3_s2_6(const u8* w, const float* x, float* y, u32 rows, u32 cols) { exl3_mv2<12, 2>(w, x, y, rows, cols); }
extern "C" __global__ void __launch_bounds__(512) matvec_batch_exl3_s2_6(const u64* wtab, const float* x, u32 xs, float* y, u32 ys, u32 rows, u32 cols) { exl3_mv2<12, 2>((const u8*) wtab[blockIdx.y], x + (u64) blockIdx.y * xs, y + (u64) blockIdx.y * ys, rows, cols); }
extern "C" __global__ void __launch_bounds__(1024) matvec_exl3_s4_6(const u8* w, const float* x, float* y, u32 rows, u32 cols) { exl3_mv2<12, 4>(w, x, y, rows, cols); }
extern "C" __global__ void __launch_bounds__(1024) matvec_batch_exl3_s4_6(const u64* wtab, const float* x, u32 xs, float* y, u32 ys, u32 rows, u32 cols) { exl3_mv2<12, 4>((const u8*) wtab[blockIdx.y], x + (u64) blockIdx.y * xs, y + (u64) blockIdx.y * ys, rows, cols); }
extern "C" __global__ void matvec_exl3_7(const u8* w, const float* x, float* y, u32 rows, u32 cols) { exl3_mv<14, 1>(w, x, 0, y, 0, rows, cols, 1); }
extern "C" __global__ void matvec_cols_exl3_7(const u8* w, const float* x, u32 xs, float* y, u32 ys, u32 rows, u32 cols, u32 m) { exl3_mv<14, 8>(w, x, xs, y, ys, rows, cols, m); }
extern "C" __global__ void matvec_batch_exl3_7(const u64* wtab, const float* x, u32 xs, float* y, u32 ys, u32 rows, u32 cols) { exl3_mv<14, 1>((const u8*) wtab[blockIdx.y], x + (u64) blockIdx.y * xs, 0, y + (u64) blockIdx.y * ys, 0, rows, cols, 1); }
extern "C" __global__ void __launch_bounds__(512) matvec_cols_exl3_s2_7(const u8* w, const float* x, u32 xs, float* y, u32 ys, u32 rows, u32 cols, u32 m) { exl3_mvc2<14, 8, 2, false>(w, x, xs, y, ys, rows, cols, m, nullptr); }
extern "C" __global__ void __launch_bounds__(1024) matvec_cols_exl3_s4_7(const u8* w, const float* x, u32 xs, float* y, u32 ys, u32 rows, u32 cols, u32 m) { exl3_mvc2<14, 8, 4, false>(w, x, xs, y, ys, rows, cols, m, nullptr); }
extern "C" __global__ void __launch_bounds__(1024) matvec_cols_exl3_m4s4_7(const u8* w, const float* x, u32 xs, float* y, u32 ys, u32 rows, u32 cols, u32 m) { exl3_mvc2<14, 4, 4, false>(w, x, xs, y, ys, rows, cols, m, nullptr); }
extern "C" __global__ void __launch_bounds__(1024) matvec_cols_exl3_sk4_7(const u8* w, const float* x, u32 xs, float* part, u32 rows, u32 cols, u32 m) { exl3_mvc2<14, 8, 4, false, true>(w, x, xs, part, 0, rows, cols, m, nullptr); }
extern "C" __global__ void __launch_bounds__(1024) matvec_grouped_exl3_m2s4_7(const u64* wtab, const u32* gtab, const float* x, u32 xs, float* y, u32 ys, u32 rows, u32 cols) { exl3_mvc2<14, 2, 4, true>((const u8*) wtab[blockIdx.y], x, xs, y, ys, rows, cols, 0, gtab + (u64) blockIdx.y * 17); }
extern "C" __global__ void __launch_bounds__(1024) matvec_grouped_exl3_m4s4_7(const u64* wtab, const u32* gtab, const float* x, u32 xs, float* y, u32 ys, u32 rows, u32 cols) { exl3_mvc2<14, 4, 4, true>((const u8*) wtab[blockIdx.y], x, xs, y, ys, rows, cols, 0, gtab + (u64) blockIdx.y * 17); }
extern "C" __global__ void __launch_bounds__(1024) matvec_grouped_exl3_s4_7(const u64* wtab, const u32* gtab, const float* x, u32 xs, float* y, u32 ys, u32 rows, u32 cols) { exl3_mvc2<14, 8, 4, true>((const u8*) wtab[blockIdx.y], x, xs, y, ys, rows, cols, 0, gtab + (u64) blockIdx.y * 17); }
extern "C" __global__ void __launch_bounds__(256) mma_cols_exl3_7(const u8* w, const float* x, u32 xs, float* y, u32 ys, u32 rows, u32 cols, u32 m) { exl3_mma<14, false>(w, x, xs, y, ys, rows, cols, m, nullptr); }
extern "C" __global__ void __launch_bounds__(256) mma_grouped_exl3_7(const u64* wtab, const u32* gtab, const float* x, u32 xs, float* y, u32 ys, u32 rows, u32 cols) { exl3_mma<14, true>((const u8*) wtab[blockIdx.y], x, xs, y, ys, rows, cols, 0, gtab + (u64) blockIdx.y * 129); }
extern "C" __global__ void __launch_bounds__(256) mma_grouped_exl3_pre_7(const u64* wtab, const u32* gtab, const float* x, u32 xs, float* y, u32 ys, u32 rows, u32 cols) { exl3_mma<14, true, true>((const u8*) wtab[blockIdx.y], x, xs, y, ys, rows, cols, 0, gtab + (u64) blockIdx.y * 129); }
extern "C" __global__ void __launch_bounds__(256) mma_cols_exl3_pre_7(const u8* w, const float* x, u32 xs, float* y, u32 ys, u32 rows, u32 cols, u32 m) { exl3_mma<14, false, true>(w, x, xs, y, ys, rows, cols, m, nullptr); }
extern "C" __global__ void __launch_bounds__(256) matvec_exl3_s1_7(const u8* w, const float* x, float* y, u32 rows, u32 cols) { exl3_mv2<14, 1>(w, x, y, rows, cols); }
extern "C" __global__ void __launch_bounds__(256) matvec_batch_exl3_s1_7(const u64* wtab, const float* x, u32 xs, float* y, u32 ys, u32 rows, u32 cols) { exl3_mv2<14, 1>((const u8*) wtab[blockIdx.y], x + (u64) blockIdx.y * xs, y + (u64) blockIdx.y * ys, rows, cols); }
extern "C" __global__ void __launch_bounds__(512) matvec_exl3_s2_7(const u8* w, const float* x, float* y, u32 rows, u32 cols) { exl3_mv2<14, 2>(w, x, y, rows, cols); }
extern "C" __global__ void __launch_bounds__(512) matvec_batch_exl3_s2_7(const u64* wtab, const float* x, u32 xs, float* y, u32 ys, u32 rows, u32 cols) { exl3_mv2<14, 2>((const u8*) wtab[blockIdx.y], x + (u64) blockIdx.y * xs, y + (u64) blockIdx.y * ys, rows, cols); }
extern "C" __global__ void __launch_bounds__(1024) matvec_exl3_s4_7(const u8* w, const float* x, float* y, u32 rows, u32 cols) { exl3_mv2<14, 4>(w, x, y, rows, cols); }
extern "C" __global__ void __launch_bounds__(1024) matvec_batch_exl3_s4_7(const u64* wtab, const float* x, u32 xs, float* y, u32 ys, u32 rows, u32 cols) { exl3_mv2<14, 4>((const u8*) wtab[blockIdx.y], x + (u64) blockIdx.y * xs, y + (u64) blockIdx.y * ys, rows, cols); }
extern "C" __global__ void matvec_exl3_8(const u8* w, const float* x, float* y, u32 rows, u32 cols) { exl3_mv<16, 1>(w, x, 0, y, 0, rows, cols, 1); }
extern "C" __global__ void matvec_cols_exl3_8(const u8* w, const float* x, u32 xs, float* y, u32 ys, u32 rows, u32 cols, u32 m) { exl3_mv<16, 8>(w, x, xs, y, ys, rows, cols, m); }
extern "C" __global__ void matvec_batch_exl3_8(const u64* wtab, const float* x, u32 xs, float* y, u32 ys, u32 rows, u32 cols) { exl3_mv<16, 1>((const u8*) wtab[blockIdx.y], x + (u64) blockIdx.y * xs, 0, y + (u64) blockIdx.y * ys, 0, rows, cols, 1); }
extern "C" __global__ void __launch_bounds__(512) matvec_cols_exl3_s2_8(const u8* w, const float* x, u32 xs, float* y, u32 ys, u32 rows, u32 cols, u32 m) { exl3_mvc2<16, 8, 2, false>(w, x, xs, y, ys, rows, cols, m, nullptr); }
extern "C" __global__ void __launch_bounds__(1024) matvec_cols_exl3_s4_8(const u8* w, const float* x, u32 xs, float* y, u32 ys, u32 rows, u32 cols, u32 m) { exl3_mvc2<16, 8, 4, false>(w, x, xs, y, ys, rows, cols, m, nullptr); }
extern "C" __global__ void __launch_bounds__(1024) matvec_cols_exl3_m4s4_8(const u8* w, const float* x, u32 xs, float* y, u32 ys, u32 rows, u32 cols, u32 m) { exl3_mvc2<16, 4, 4, false>(w, x, xs, y, ys, rows, cols, m, nullptr); }
extern "C" __global__ void __launch_bounds__(1024) matvec_cols_exl3_sk4_8(const u8* w, const float* x, u32 xs, float* part, u32 rows, u32 cols, u32 m) { exl3_mvc2<16, 8, 4, false, true>(w, x, xs, part, 0, rows, cols, m, nullptr); }
extern "C" __global__ void __launch_bounds__(1024) matvec_grouped_exl3_m2s4_8(const u64* wtab, const u32* gtab, const float* x, u32 xs, float* y, u32 ys, u32 rows, u32 cols) { exl3_mvc2<16, 2, 4, true>((const u8*) wtab[blockIdx.y], x, xs, y, ys, rows, cols, 0, gtab + (u64) blockIdx.y * 17); }
extern "C" __global__ void __launch_bounds__(1024) matvec_grouped_exl3_m4s4_8(const u64* wtab, const u32* gtab, const float* x, u32 xs, float* y, u32 ys, u32 rows, u32 cols) { exl3_mvc2<16, 4, 4, true>((const u8*) wtab[blockIdx.y], x, xs, y, ys, rows, cols, 0, gtab + (u64) blockIdx.y * 17); }
extern "C" __global__ void __launch_bounds__(1024) matvec_grouped_exl3_s4_8(const u64* wtab, const u32* gtab, const float* x, u32 xs, float* y, u32 ys, u32 rows, u32 cols) { exl3_mvc2<16, 8, 4, true>((const u8*) wtab[blockIdx.y], x, xs, y, ys, rows, cols, 0, gtab + (u64) blockIdx.y * 17); }
extern "C" __global__ void __launch_bounds__(256) mma_cols_exl3_8(const u8* w, const float* x, u32 xs, float* y, u32 ys, u32 rows, u32 cols, u32 m) { exl3_mma<16, false>(w, x, xs, y, ys, rows, cols, m, nullptr); }
extern "C" __global__ void __launch_bounds__(256) mma_grouped_exl3_8(const u64* wtab, const u32* gtab, const float* x, u32 xs, float* y, u32 ys, u32 rows, u32 cols) { exl3_mma<16, true>((const u8*) wtab[blockIdx.y], x, xs, y, ys, rows, cols, 0, gtab + (u64) blockIdx.y * 129); }
extern "C" __global__ void __launch_bounds__(256) mma_grouped_exl3_pre_8(const u64* wtab, const u32* gtab, const float* x, u32 xs, float* y, u32 ys, u32 rows, u32 cols) { exl3_mma<16, true, true>((const u8*) wtab[blockIdx.y], x, xs, y, ys, rows, cols, 0, gtab + (u64) blockIdx.y * 129); }
extern "C" __global__ void __launch_bounds__(256) mma_cols_exl3_pre_8(const u8* w, const float* x, u32 xs, float* y, u32 ys, u32 rows, u32 cols, u32 m) { exl3_mma<16, false, true>(w, x, xs, y, ys, rows, cols, m, nullptr); }
extern "C" __global__ void __launch_bounds__(256) matvec_exl3_s1_8(const u8* w, const float* x, float* y, u32 rows, u32 cols) { exl3_mv2<16, 1>(w, x, y, rows, cols); }
extern "C" __global__ void __launch_bounds__(256) matvec_batch_exl3_s1_8(const u64* wtab, const float* x, u32 xs, float* y, u32 ys, u32 rows, u32 cols) { exl3_mv2<16, 1>((const u8*) wtab[blockIdx.y], x + (u64) blockIdx.y * xs, y + (u64) blockIdx.y * ys, rows, cols); }
extern "C" __global__ void __launch_bounds__(512) matvec_exl3_s2_8(const u8* w, const float* x, float* y, u32 rows, u32 cols) { exl3_mv2<16, 2>(w, x, y, rows, cols); }
extern "C" __global__ void __launch_bounds__(512) matvec_batch_exl3_s2_8(const u64* wtab, const float* x, u32 xs, float* y, u32 ys, u32 rows, u32 cols) { exl3_mv2<16, 2>((const u8*) wtab[blockIdx.y], x + (u64) blockIdx.y * xs, y + (u64) blockIdx.y * ys, rows, cols); }
extern "C" __global__ void __launch_bounds__(1024) matvec_exl3_s4_8(const u8* w, const float* x, float* y, u32 rows, u32 cols) { exl3_mv2<16, 4>(w, x, y, rows, cols); }
extern "C" __global__ void __launch_bounds__(1024) matvec_batch_exl3_s4_8(const u64* wtab, const float* x, u32 xs, float* y, u32 ys, u32 rows, u32 cols) { exl3_mv2<16, 4>((const u8*) wtab[blockIdx.y], x + (u64) blockIdx.y * xs, y + (u64) blockIdx.y * ys, rows, cols); }
extern "C" __global__ void matvec_exl3_1h(const u8* w, const float* x, float* y, u32 rows, u32 cols) { exl3_mv<3, 1>(w, x, 0, y, 0, rows, cols, 1); }
extern "C" __global__ void matvec_cols_exl3_1h(const u8* w, const float* x, u32 xs, float* y, u32 ys, u32 rows, u32 cols, u32 m) { exl3_mv<3, 8>(w, x, xs, y, ys, rows, cols, m); }
extern "C" __global__ void matvec_batch_exl3_1h(const u64* wtab, const float* x, u32 xs, float* y, u32 ys, u32 rows, u32 cols) { exl3_mv<3, 1>((const u8*) wtab[blockIdx.y], x + (u64) blockIdx.y * xs, 0, y + (u64) blockIdx.y * ys, 0, rows, cols, 1); }
extern "C" __global__ void __launch_bounds__(512) matvec_cols_exl3_s2_1h(const u8* w, const float* x, u32 xs, float* y, u32 ys, u32 rows, u32 cols, u32 m) { exl3_mvc2<3, 8, 2, false>(w, x, xs, y, ys, rows, cols, m, nullptr); }
extern "C" __global__ void __launch_bounds__(1024) matvec_cols_exl3_s4_1h(const u8* w, const float* x, u32 xs, float* y, u32 ys, u32 rows, u32 cols, u32 m) { exl3_mvc2<3, 8, 4, false>(w, x, xs, y, ys, rows, cols, m, nullptr); }
extern "C" __global__ void __launch_bounds__(1024) matvec_cols_exl3_m4s4_1h(const u8* w, const float* x, u32 xs, float* y, u32 ys, u32 rows, u32 cols, u32 m) { exl3_mvc2<3, 4, 4, false>(w, x, xs, y, ys, rows, cols, m, nullptr); }
extern "C" __global__ void __launch_bounds__(1024) matvec_cols_exl3_sk4_1h(const u8* w, const float* x, u32 xs, float* part, u32 rows, u32 cols, u32 m) { exl3_mvc2<3, 8, 4, false, true>(w, x, xs, part, 0, rows, cols, m, nullptr); }
extern "C" __global__ void __launch_bounds__(1024) matvec_grouped_exl3_m2s4_1h(const u64* wtab, const u32* gtab, const float* x, u32 xs, float* y, u32 ys, u32 rows, u32 cols) { exl3_mvc2<3, 2, 4, true>((const u8*) wtab[blockIdx.y], x, xs, y, ys, rows, cols, 0, gtab + (u64) blockIdx.y * 17); }
extern "C" __global__ void __launch_bounds__(1024) matvec_grouped_exl3_m4s4_1h(const u64* wtab, const u32* gtab, const float* x, u32 xs, float* y, u32 ys, u32 rows, u32 cols) { exl3_mvc2<3, 4, 4, true>((const u8*) wtab[blockIdx.y], x, xs, y, ys, rows, cols, 0, gtab + (u64) blockIdx.y * 17); }
extern "C" __global__ void __launch_bounds__(1024) matvec_grouped_exl3_s4_1h(const u64* wtab, const u32* gtab, const float* x, u32 xs, float* y, u32 ys, u32 rows, u32 cols) { exl3_mvc2<3, 8, 4, true>((const u8*) wtab[blockIdx.y], x, xs, y, ys, rows, cols, 0, gtab + (u64) blockIdx.y * 17); }
extern "C" __global__ void __launch_bounds__(256) mma_cols_exl3_1h(const u8* w, const float* x, u32 xs, float* y, u32 ys, u32 rows, u32 cols, u32 m) { exl3_mma<3, false>(w, x, xs, y, ys, rows, cols, m, nullptr); }
extern "C" __global__ void __launch_bounds__(256) mma_grouped_exl3_1h(const u64* wtab, const u32* gtab, const float* x, u32 xs, float* y, u32 ys, u32 rows, u32 cols) { exl3_mma<3, true>((const u8*) wtab[blockIdx.y], x, xs, y, ys, rows, cols, 0, gtab + (u64) blockIdx.y * 129); }
extern "C" __global__ void __launch_bounds__(256) mma_grouped_exl3_pre_1h(const u64* wtab, const u32* gtab, const float* x, u32 xs, float* y, u32 ys, u32 rows, u32 cols) { exl3_mma<3, true, true>((const u8*) wtab[blockIdx.y], x, xs, y, ys, rows, cols, 0, gtab + (u64) blockIdx.y * 129); }
extern "C" __global__ void __launch_bounds__(256) mma_cols_exl3_pre_1h(const u8* w, const float* x, u32 xs, float* y, u32 ys, u32 rows, u32 cols, u32 m) { exl3_mma<3, false, true>(w, x, xs, y, ys, rows, cols, m, nullptr); }
extern "C" __global__ void __launch_bounds__(256) matvec_exl3_s1_1h(const u8* w, const float* x, float* y, u32 rows, u32 cols) { exl3_mv2<3, 1>(w, x, y, rows, cols); }
extern "C" __global__ void __launch_bounds__(256) matvec_batch_exl3_s1_1h(const u64* wtab, const float* x, u32 xs, float* y, u32 ys, u32 rows, u32 cols) { exl3_mv2<3, 1>((const u8*) wtab[blockIdx.y], x + (u64) blockIdx.y * xs, y + (u64) blockIdx.y * ys, rows, cols); }
extern "C" __global__ void __launch_bounds__(512) matvec_exl3_s2_1h(const u8* w, const float* x, float* y, u32 rows, u32 cols) { exl3_mv2<3, 2>(w, x, y, rows, cols); }
extern "C" __global__ void __launch_bounds__(512) matvec_batch_exl3_s2_1h(const u64* wtab, const float* x, u32 xs, float* y, u32 ys, u32 rows, u32 cols) { exl3_mv2<3, 2>((const u8*) wtab[blockIdx.y], x + (u64) blockIdx.y * xs, y + (u64) blockIdx.y * ys, rows, cols); }
extern "C" __global__ void __launch_bounds__(1024) matvec_exl3_s4_1h(const u8* w, const float* x, float* y, u32 rows, u32 cols) { exl3_mv2<3, 4>(w, x, y, rows, cols); }
extern "C" __global__ void __launch_bounds__(1024) matvec_batch_exl3_s4_1h(const u64* wtab, const float* x, u32 xs, float* y, u32 ys, u32 rows, u32 cols) { exl3_mv2<3, 4>((const u8*) wtab[blockIdx.y], x + (u64) blockIdx.y * xs, y + (u64) blockIdx.y * ys, rows, cols); }
extern "C" __global__ void matvec_exl3_2h(const u8* w, const float* x, float* y, u32 rows, u32 cols) { exl3_mv<5, 1>(w, x, 0, y, 0, rows, cols, 1); }
extern "C" __global__ void matvec_cols_exl3_2h(const u8* w, const float* x, u32 xs, float* y, u32 ys, u32 rows, u32 cols, u32 m) { exl3_mv<5, 8>(w, x, xs, y, ys, rows, cols, m); }
extern "C" __global__ void matvec_batch_exl3_2h(const u64* wtab, const float* x, u32 xs, float* y, u32 ys, u32 rows, u32 cols) { exl3_mv<5, 1>((const u8*) wtab[blockIdx.y], x + (u64) blockIdx.y * xs, 0, y + (u64) blockIdx.y * ys, 0, rows, cols, 1); }
extern "C" __global__ void __launch_bounds__(512) matvec_cols_exl3_s2_2h(const u8* w, const float* x, u32 xs, float* y, u32 ys, u32 rows, u32 cols, u32 m) { exl3_mvc2<5, 8, 2, false>(w, x, xs, y, ys, rows, cols, m, nullptr); }
extern "C" __global__ void __launch_bounds__(1024) matvec_cols_exl3_s4_2h(const u8* w, const float* x, u32 xs, float* y, u32 ys, u32 rows, u32 cols, u32 m) { exl3_mvc2<5, 8, 4, false>(w, x, xs, y, ys, rows, cols, m, nullptr); }
extern "C" __global__ void __launch_bounds__(1024) matvec_cols_exl3_m4s4_2h(const u8* w, const float* x, u32 xs, float* y, u32 ys, u32 rows, u32 cols, u32 m) { exl3_mvc2<5, 4, 4, false>(w, x, xs, y, ys, rows, cols, m, nullptr); }
extern "C" __global__ void __launch_bounds__(1024) matvec_cols_exl3_sk4_2h(const u8* w, const float* x, u32 xs, float* part, u32 rows, u32 cols, u32 m) { exl3_mvc2<5, 8, 4, false, true>(w, x, xs, part, 0, rows, cols, m, nullptr); }
extern "C" __global__ void __launch_bounds__(1024) matvec_grouped_exl3_m2s4_2h(const u64* wtab, const u32* gtab, const float* x, u32 xs, float* y, u32 ys, u32 rows, u32 cols) { exl3_mvc2<5, 2, 4, true>((const u8*) wtab[blockIdx.y], x, xs, y, ys, rows, cols, 0, gtab + (u64) blockIdx.y * 17); }
extern "C" __global__ void __launch_bounds__(1024) matvec_grouped_exl3_m4s4_2h(const u64* wtab, const u32* gtab, const float* x, u32 xs, float* y, u32 ys, u32 rows, u32 cols) { exl3_mvc2<5, 4, 4, true>((const u8*) wtab[blockIdx.y], x, xs, y, ys, rows, cols, 0, gtab + (u64) blockIdx.y * 17); }
extern "C" __global__ void __launch_bounds__(1024) matvec_grouped_exl3_s4_2h(const u64* wtab, const u32* gtab, const float* x, u32 xs, float* y, u32 ys, u32 rows, u32 cols) { exl3_mvc2<5, 8, 4, true>((const u8*) wtab[blockIdx.y], x, xs, y, ys, rows, cols, 0, gtab + (u64) blockIdx.y * 17); }
extern "C" __global__ void __launch_bounds__(256) mma_cols_exl3_2h(const u8* w, const float* x, u32 xs, float* y, u32 ys, u32 rows, u32 cols, u32 m) { exl3_mma<5, false>(w, x, xs, y, ys, rows, cols, m, nullptr); }
extern "C" __global__ void __launch_bounds__(256) mma_grouped_exl3_2h(const u64* wtab, const u32* gtab, const float* x, u32 xs, float* y, u32 ys, u32 rows, u32 cols) { exl3_mma<5, true>((const u8*) wtab[blockIdx.y], x, xs, y, ys, rows, cols, 0, gtab + (u64) blockIdx.y * 129); }
extern "C" __global__ void __launch_bounds__(256) mma_grouped_exl3_pre_2h(const u64* wtab, const u32* gtab, const float* x, u32 xs, float* y, u32 ys, u32 rows, u32 cols) { exl3_mma<5, true, true>((const u8*) wtab[blockIdx.y], x, xs, y, ys, rows, cols, 0, gtab + (u64) blockIdx.y * 129); }
extern "C" __global__ void __launch_bounds__(256) mma_cols_exl3_pre_2h(const u8* w, const float* x, u32 xs, float* y, u32 ys, u32 rows, u32 cols, u32 m) { exl3_mma<5, false, true>(w, x, xs, y, ys, rows, cols, m, nullptr); }
extern "C" __global__ void __launch_bounds__(256) matvec_exl3_s1_2h(const u8* w, const float* x, float* y, u32 rows, u32 cols) { exl3_mv2<5, 1>(w, x, y, rows, cols); }
extern "C" __global__ void __launch_bounds__(256) matvec_batch_exl3_s1_2h(const u64* wtab, const float* x, u32 xs, float* y, u32 ys, u32 rows, u32 cols) { exl3_mv2<5, 1>((const u8*) wtab[blockIdx.y], x + (u64) blockIdx.y * xs, y + (u64) blockIdx.y * ys, rows, cols); }
extern "C" __global__ void __launch_bounds__(512) matvec_exl3_s2_2h(const u8* w, const float* x, float* y, u32 rows, u32 cols) { exl3_mv2<5, 2>(w, x, y, rows, cols); }
extern "C" __global__ void __launch_bounds__(512) matvec_batch_exl3_s2_2h(const u64* wtab, const float* x, u32 xs, float* y, u32 ys, u32 rows, u32 cols) { exl3_mv2<5, 2>((const u8*) wtab[blockIdx.y], x + (u64) blockIdx.y * xs, y + (u64) blockIdx.y * ys, rows, cols); }
extern "C" __global__ void __launch_bounds__(1024) matvec_exl3_s4_2h(const u8* w, const float* x, float* y, u32 rows, u32 cols) { exl3_mv2<5, 4>(w, x, y, rows, cols); }
extern "C" __global__ void __launch_bounds__(1024) matvec_batch_exl3_s4_2h(const u64* wtab, const float* x, u32 xs, float* y, u32 ys, u32 rows, u32 cols) { exl3_mv2<5, 4>((const u8*) wtab[blockIdx.y], x + (u64) blockIdx.y * xs, y + (u64) blockIdx.y * ys, rows, cols); }
extern "C" __global__ void matvec_exl3_3h(const u8* w, const float* x, float* y, u32 rows, u32 cols) { exl3_mv<7, 1>(w, x, 0, y, 0, rows, cols, 1); }
extern "C" __global__ void matvec_cols_exl3_3h(const u8* w, const float* x, u32 xs, float* y, u32 ys, u32 rows, u32 cols, u32 m) { exl3_mv<7, 8>(w, x, xs, y, ys, rows, cols, m); }
extern "C" __global__ void matvec_batch_exl3_3h(const u64* wtab, const float* x, u32 xs, float* y, u32 ys, u32 rows, u32 cols) { exl3_mv<7, 1>((const u8*) wtab[blockIdx.y], x + (u64) blockIdx.y * xs, 0, y + (u64) blockIdx.y * ys, 0, rows, cols, 1); }
extern "C" __global__ void __launch_bounds__(512) matvec_cols_exl3_s2_3h(const u8* w, const float* x, u32 xs, float* y, u32 ys, u32 rows, u32 cols, u32 m) { exl3_mvc2<7, 8, 2, false>(w, x, xs, y, ys, rows, cols, m, nullptr); }
extern "C" __global__ void __launch_bounds__(1024) matvec_cols_exl3_s4_3h(const u8* w, const float* x, u32 xs, float* y, u32 ys, u32 rows, u32 cols, u32 m) { exl3_mvc2<7, 8, 4, false>(w, x, xs, y, ys, rows, cols, m, nullptr); }
extern "C" __global__ void __launch_bounds__(1024) matvec_cols_exl3_m4s4_3h(const u8* w, const float* x, u32 xs, float* y, u32 ys, u32 rows, u32 cols, u32 m) { exl3_mvc2<7, 4, 4, false>(w, x, xs, y, ys, rows, cols, m, nullptr); }
extern "C" __global__ void __launch_bounds__(1024) matvec_cols_exl3_sk4_3h(const u8* w, const float* x, u32 xs, float* part, u32 rows, u32 cols, u32 m) { exl3_mvc2<7, 8, 4, false, true>(w, x, xs, part, 0, rows, cols, m, nullptr); }
extern "C" __global__ void __launch_bounds__(1024) matvec_grouped_exl3_m2s4_3h(const u64* wtab, const u32* gtab, const float* x, u32 xs, float* y, u32 ys, u32 rows, u32 cols) { exl3_mvc2<7, 2, 4, true>((const u8*) wtab[blockIdx.y], x, xs, y, ys, rows, cols, 0, gtab + (u64) blockIdx.y * 17); }
extern "C" __global__ void __launch_bounds__(1024) matvec_grouped_exl3_m4s4_3h(const u64* wtab, const u32* gtab, const float* x, u32 xs, float* y, u32 ys, u32 rows, u32 cols) { exl3_mvc2<7, 4, 4, true>((const u8*) wtab[blockIdx.y], x, xs, y, ys, rows, cols, 0, gtab + (u64) blockIdx.y * 17); }
extern "C" __global__ void __launch_bounds__(1024) matvec_grouped_exl3_s4_3h(const u64* wtab, const u32* gtab, const float* x, u32 xs, float* y, u32 ys, u32 rows, u32 cols) { exl3_mvc2<7, 8, 4, true>((const u8*) wtab[blockIdx.y], x, xs, y, ys, rows, cols, 0, gtab + (u64) blockIdx.y * 17); }
extern "C" __global__ void __launch_bounds__(256) mma_cols_exl3_3h(const u8* w, const float* x, u32 xs, float* y, u32 ys, u32 rows, u32 cols, u32 m) { exl3_mma<7, false>(w, x, xs, y, ys, rows, cols, m, nullptr); }
extern "C" __global__ void __launch_bounds__(256) mma_grouped_exl3_3h(const u64* wtab, const u32* gtab, const float* x, u32 xs, float* y, u32 ys, u32 rows, u32 cols) { exl3_mma<7, true>((const u8*) wtab[blockIdx.y], x, xs, y, ys, rows, cols, 0, gtab + (u64) blockIdx.y * 129); }
extern "C" __global__ void __launch_bounds__(256) mma_grouped_exl3_pre_3h(const u64* wtab, const u32* gtab, const float* x, u32 xs, float* y, u32 ys, u32 rows, u32 cols) { exl3_mma<7, true, true>((const u8*) wtab[blockIdx.y], x, xs, y, ys, rows, cols, 0, gtab + (u64) blockIdx.y * 129); }
extern "C" __global__ void __launch_bounds__(256) mma_cols_exl3_pre_3h(const u8* w, const float* x, u32 xs, float* y, u32 ys, u32 rows, u32 cols, u32 m) { exl3_mma<7, false, true>(w, x, xs, y, ys, rows, cols, m, nullptr); }
extern "C" __global__ void __launch_bounds__(256) matvec_exl3_s1_3h(const u8* w, const float* x, float* y, u32 rows, u32 cols) { exl3_mv2<7, 1>(w, x, y, rows, cols); }
extern "C" __global__ void __launch_bounds__(256) matvec_batch_exl3_s1_3h(const u64* wtab, const float* x, u32 xs, float* y, u32 ys, u32 rows, u32 cols) { exl3_mv2<7, 1>((const u8*) wtab[blockIdx.y], x + (u64) blockIdx.y * xs, y + (u64) blockIdx.y * ys, rows, cols); }
extern "C" __global__ void __launch_bounds__(512) matvec_exl3_s2_3h(const u8* w, const float* x, float* y, u32 rows, u32 cols) { exl3_mv2<7, 2>(w, x, y, rows, cols); }
extern "C" __global__ void __launch_bounds__(512) matvec_batch_exl3_s2_3h(const u64* wtab, const float* x, u32 xs, float* y, u32 ys, u32 rows, u32 cols) { exl3_mv2<7, 2>((const u8*) wtab[blockIdx.y], x + (u64) blockIdx.y * xs, y + (u64) blockIdx.y * ys, rows, cols); }
extern "C" __global__ void __launch_bounds__(1024) matvec_exl3_s4_3h(const u8* w, const float* x, float* y, u32 rows, u32 cols) { exl3_mv2<7, 4>(w, x, y, rows, cols); }
extern "C" __global__ void __launch_bounds__(1024) matvec_batch_exl3_s4_3h(const u64* wtab, const float* x, u32 xs, float* y, u32 ys, u32 rows, u32 cols) { exl3_mv2<7, 4>((const u8*) wtab[blockIdx.y], x + (u64) blockIdx.y * xs, y + (u64) blockIdx.y * ys, rows, cols); }
extern "C" __global__ void __launch_bounds__(256) matvec_cols_exl3_pre_m4s1_1(const u8* w, const float* xr, float* y, u32 ys, u32 rows, u32 cols, u32 m) { exl3_mvc2_pre<2, 4, 1>(w, xr, y, ys, rows, cols, m); }
extern "C" __global__ void __launch_bounds__(1024) matvec_cols_exl3_sk4_m4_1(const u8* w, const float* x, u32 xs, float* part, u32 rows, u32 cols, u32 m) { exl3_mvc2<2, 4, 4, false, true>(w, x, xs, part, 0, rows, cols, m, nullptr); }
extern "C" __global__ void __launch_bounds__(256) matvec_cols_exl3_pre_m4s1_2(const u8* w, const float* xr, float* y, u32 ys, u32 rows, u32 cols, u32 m) { exl3_mvc2_pre<4, 4, 1>(w, xr, y, ys, rows, cols, m); }
extern "C" __global__ void __launch_bounds__(1024) matvec_cols_exl3_sk4_m4_2(const u8* w, const float* x, u32 xs, float* part, u32 rows, u32 cols, u32 m) { exl3_mvc2<4, 4, 4, false, true>(w, x, xs, part, 0, rows, cols, m, nullptr); }
extern "C" __global__ void __launch_bounds__(256) matvec_cols_exl3_pre_m4s1_3(const u8* w, const float* xr, float* y, u32 ys, u32 rows, u32 cols, u32 m) { exl3_mvc2_pre<6, 4, 1>(w, xr, y, ys, rows, cols, m); }
extern "C" __global__ void __launch_bounds__(1024) matvec_cols_exl3_sk4_m4_3(const u8* w, const float* x, u32 xs, float* part, u32 rows, u32 cols, u32 m) { exl3_mvc2<6, 4, 4, false, true>(w, x, xs, part, 0, rows, cols, m, nullptr); }
extern "C" __global__ void __launch_bounds__(256) matvec_cols_exl3_pre_m4s1_4(const u8* w, const float* xr, float* y, u32 ys, u32 rows, u32 cols, u32 m) { exl3_mvc2_pre<8, 4, 1>(w, xr, y, ys, rows, cols, m); }
extern "C" __global__ void __launch_bounds__(1024) matvec_cols_exl3_sk4_m4_4(const u8* w, const float* x, u32 xs, float* part, u32 rows, u32 cols, u32 m) { exl3_mvc2<8, 4, 4, false, true>(w, x, xs, part, 0, rows, cols, m, nullptr); }
extern "C" __global__ void __launch_bounds__(256) matvec_cols_exl3_pre_m4s1_5(const u8* w, const float* xr, float* y, u32 ys, u32 rows, u32 cols, u32 m) { exl3_mvc2_pre<10, 4, 1>(w, xr, y, ys, rows, cols, m); }
extern "C" __global__ void __launch_bounds__(1024) matvec_cols_exl3_sk4_m4_5(const u8* w, const float* x, u32 xs, float* part, u32 rows, u32 cols, u32 m) { exl3_mvc2<10, 4, 4, false, true>(w, x, xs, part, 0, rows, cols, m, nullptr); }
extern "C" __global__ void __launch_bounds__(256) matvec_cols_exl3_pre_m4s1_6(const u8* w, const float* xr, float* y, u32 ys, u32 rows, u32 cols, u32 m) { exl3_mvc2_pre<12, 4, 1>(w, xr, y, ys, rows, cols, m); }
extern "C" __global__ void __launch_bounds__(1024) matvec_cols_exl3_sk4_m4_6(const u8* w, const float* x, u32 xs, float* part, u32 rows, u32 cols, u32 m) { exl3_mvc2<12, 4, 4, false, true>(w, x, xs, part, 0, rows, cols, m, nullptr); }
extern "C" __global__ void __launch_bounds__(256) matvec_cols_exl3_pre_m4s1_7(const u8* w, const float* xr, float* y, u32 ys, u32 rows, u32 cols, u32 m) { exl3_mvc2_pre<14, 4, 1>(w, xr, y, ys, rows, cols, m); }
extern "C" __global__ void __launch_bounds__(1024) matvec_cols_exl3_sk4_m4_7(const u8* w, const float* x, u32 xs, float* part, u32 rows, u32 cols, u32 m) { exl3_mvc2<14, 4, 4, false, true>(w, x, xs, part, 0, rows, cols, m, nullptr); }
extern "C" __global__ void __launch_bounds__(256) matvec_cols_exl3_pre_m4s1_8(const u8* w, const float* xr, float* y, u32 ys, u32 rows, u32 cols, u32 m) { exl3_mvc2_pre<16, 4, 1>(w, xr, y, ys, rows, cols, m); }
extern "C" __global__ void __launch_bounds__(1024) matvec_cols_exl3_sk4_m4_8(const u8* w, const float* x, u32 xs, float* part, u32 rows, u32 cols, u32 m) { exl3_mvc2<16, 4, 4, false, true>(w, x, xs, part, 0, rows, cols, m, nullptr); }
extern "C" __global__ void __launch_bounds__(256) matvec_cols_exl3_pre_m4s1_1h(const u8* w, const float* xr, float* y, u32 ys, u32 rows, u32 cols, u32 m) { exl3_mvc2_pre<3, 4, 1>(w, xr, y, ys, rows, cols, m); }
extern "C" __global__ void __launch_bounds__(1024) matvec_cols_exl3_sk4_m4_1h(const u8* w, const float* x, u32 xs, float* part, u32 rows, u32 cols, u32 m) { exl3_mvc2<3, 4, 4, false, true>(w, x, xs, part, 0, rows, cols, m, nullptr); }
extern "C" __global__ void __launch_bounds__(256) matvec_cols_exl3_pre_m4s1_2h(const u8* w, const float* xr, float* y, u32 ys, u32 rows, u32 cols, u32 m) { exl3_mvc2_pre<5, 4, 1>(w, xr, y, ys, rows, cols, m); }
extern "C" __global__ void __launch_bounds__(1024) matvec_cols_exl3_sk4_m4_2h(const u8* w, const float* x, u32 xs, float* part, u32 rows, u32 cols, u32 m) { exl3_mvc2<5, 4, 4, false, true>(w, x, xs, part, 0, rows, cols, m, nullptr); }
extern "C" __global__ void __launch_bounds__(256) matvec_cols_exl3_pre_m4s1_3h(const u8* w, const float* xr, float* y, u32 ys, u32 rows, u32 cols, u32 m) { exl3_mvc2_pre<7, 4, 1>(w, xr, y, ys, rows, cols, m); }
extern "C" __global__ void __launch_bounds__(1024) matvec_cols_exl3_sk4_m4_3h(const u8* w, const float* x, u32 xs, float* part, u32 rows, u32 cols, u32 m) { exl3_mvc2<7, 4, 4, false, true>(w, x, xs, part, 0, rows, cols, m, nullptr); }

// ------------------------------------------------------------------------------------ Q2 proxy builder + sparse block EXL3 rescore
// Dequantizes 128x128 EXL3 tile (ob, b), applies output Hadamard + svh/128 (so y = xr * W_rot without further H_out),
// and packs 2-bit symmetric Lloyd-Max {-1.5, -0.5, +0.5, +1.5}*delta codes into tile-interleaved SOA layout for dp4a.
template <int KB>
static __device__ __forceinline__ void exl3_build_q2_proxy_impl(const u8* __restrict__ w, u8* __restrict__ w_q2,
                                                                u16* __restrict__ w_sc, u32 rows, u32 cols) {
    __shared__ float ws[16][132];
    __shared__ u16 wh[128][132];
    const u32 ob = blockIdx.x, b = blockIdx.y;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const u32 n_b = cols / 128u, nt = rows / 16u;
    const u16* svh = (const u16*) (w + 2 * cols);
    const u32* tr = (const u32*) (w + 2 * cols + 2 * rows + 4);
    const int col_lo = 2 * (lane >> 3) + ((lane >> 2) & 1);
    for (int kk = 0; kk < 8; kk++) {
        const u32* tile = tr + ((u64) (b * 8 + kk) * nt + ob * 8 + warp) * (4 * KB);
        float v[8];
        exl3_lane_vals<KB>(tile, lane, v);
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            int r_in = (lane & 3) * 2 + (j & 1) + 8 * ((j >> 1) & 1);
            int c_out = warp * 16 + col_lo + (j >= 4 ? 8 : 0);
            ws[r_in][c_out] = v[j];
        }
        __syncthreads();
        #pragma unroll
        for (int u = 0; u < 2; u++) {
            int r_in = warp + 8 * u;
            float rv[4];
            #pragma unroll
            for (int e = 0; e < 4; e++) rv[e] = ws[r_in][lane + 32 * e];
            exl3_wht4(rv, lane);
            #pragma unroll
            for (int e = 0; e < 4; e++) {
                int c_out = lane + 32 * e;
                float val = rv[e] * 0.0078125f * f16_to_f32(svh[ob * 128 + c_out]);
                wh[c_out][kk * 16 + r_in] = f32_to_f16(val);
            }
        }
        __syncthreads();
    }
    if (tid < 128) {
        const u32 r = (u32) tid;
        float sq = 0.0f;
        for (u32 c = 0; c < 128u; c++) {
            float fv = f16_to_f32(wh[r][c]);
            sq = fmaf(fv, fv, sq);
        }
        float delta = 0.9957f * sqrtf(sq * 0.0078125f);
        u16 sh = f32_to_f16(0.5f * delta);
        float thr = 2.0f * f16_to_f32(sh);
        w_sc[((u64) ob * n_b + b) * 128u + r] = sh;
        for (u32 hb = 0; hb < 2u; hb++) {
            u32 words[4];
            #pragma unroll
            for (u32 wi = 0; wi < 4u; wi++) {
                u32 c0 = hb * 64u + wi * 16u;
                u32 pw = 0;
                #pragma unroll
                for (u32 sl = 0; sl < 4u; sl++) {
                    #pragma unroll
                    for (u32 bi = 0; bi < 4u; bi++) {
                        float fv = f16_to_f32(wh[r][c0 + sl * 4u + bi]);
                        u32 q = (fv >= 0.0f) ? (fv >= thr ? 3u : 2u) : (fv <= -thr ? 0u : 1u);
                        pw |= (q << (bi * 8u + sl * 2u));
                    }
                }
                words[wi] = pw;
            }
            ((uint4*) w_q2)[((u64) ob * (n_b * 2u) + b * 2u + hb) * 128u + r] = make_uint4(words[0], words[1], words[2], words[3]);
        }
    }
}

extern "C" __global__ void __launch_bounds__(256) exl3_build_q2_proxy_2(const u8* w, u8* w_q2, u16* w_sc, u32 rows, u32 cols) { exl3_build_q2_proxy_impl<4>(w, w_q2, w_sc, rows, cols); }
extern "C" __global__ void __launch_bounds__(256) exl3_build_q2_proxy_2h(const u8* w, u8* w_q2, u16* w_sc, u32 rows, u32 cols) { exl3_build_q2_proxy_impl<5>(w, w_q2, w_sc, rows, cols); }
extern "C" __global__ void __launch_bounds__(256) exl3_build_q2_proxy_3(const u8* w, u8* w_q2, u16* w_sc, u32 rows, u32 cols) { exl3_build_q2_proxy_impl<6>(w, w_q2, w_sc, rows, cols); }
extern "C" __global__ void __launch_bounds__(256) exl3_build_q2_proxy_3h(const u8* w, u8* w_q2, u16* w_sc, u32 rows, u32 cols) { exl3_build_q2_proxy_impl<7>(w, w_q2, w_sc, rows, cols); }
extern "C" __global__ void __launch_bounds__(256) exl3_build_q2_proxy_4(const u8* w, u8* w_q2, u16* w_sc, u32 rows, u32 cols) { exl3_build_q2_proxy_impl<8>(w, w_q2, w_sc, rows, cols); }
extern "C" __global__ void __launch_bounds__(256) exl3_build_q2_proxy_5(const u8* w, u8* w_q2, u16* w_sc, u32 rows, u32 cols) { exl3_build_q2_proxy_impl<10>(w, w_q2, w_sc, rows, cols); }
extern "C" __global__ void __launch_bounds__(256) exl3_build_q2_proxy_6(const u8* w, u8* w_q2, u16* w_sc, u32 rows, u32 cols) { exl3_build_q2_proxy_impl<12>(w, w_q2, w_sc, rows, cols); }
extern "C" __global__ void __launch_bounds__(256) exl3_build_q2_proxy_8(const u8* w, u8* w_q2, u16* w_sc, u32 rows, u32 cols) { exl3_build_q2_proxy_impl<16>(w, w_q2, w_sc, rows, cols); }

extern "C" __global__ void __launch_bounds__(256) matvec_cols_exl3_pre_sparse_m4s1_2(const u8* w, const float* xr, float* y, u32 ys, u32 rows, u32 cols, u32 m, const u32* sel_blks) { exl3_mvc2_pre<4, 4, 1, true>(w, xr, y, ys, rows, cols, m, sel_blks); }
extern "C" __global__ void __launch_bounds__(256) matvec_cols_exl3_pre_sparse_m4s1_2h(const u8* w, const float* xr, float* y, u32 ys, u32 rows, u32 cols, u32 m, const u32* sel_blks) { exl3_mvc2_pre<5, 4, 1, true>(w, xr, y, ys, rows, cols, m, sel_blks); }
extern "C" __global__ void __launch_bounds__(256) matvec_cols_exl3_pre_sparse_m4s1_3(const u8* w, const float* xr, float* y, u32 ys, u32 rows, u32 cols, u32 m, const u32* sel_blks) { exl3_mvc2_pre<6, 4, 1, true>(w, xr, y, ys, rows, cols, m, sel_blks); }
extern "C" __global__ void __launch_bounds__(256) matvec_cols_exl3_pre_sparse_m4s1_3h(const u8* w, const float* xr, float* y, u32 ys, u32 rows, u32 cols, u32 m, const u32* sel_blks) { exl3_mvc2_pre<7, 4, 1, true>(w, xr, y, ys, rows, cols, m, sel_blks); }
extern "C" __global__ void __launch_bounds__(256) matvec_cols_exl3_pre_sparse_m4s1_4(const u8* w, const float* xr, float* y, u32 ys, u32 rows, u32 cols, u32 m, const u32* sel_blks) { exl3_mvc2_pre<8, 4, 1, true>(w, xr, y, ys, rows, cols, m, sel_blks); }
extern "C" __global__ void __launch_bounds__(256) matvec_cols_exl3_pre_sparse_m4s1_5(const u8* w, const float* xr, float* y, u32 ys, u32 rows, u32 cols, u32 m, const u32* sel_blks) { exl3_mvc2_pre<10, 4, 1, true>(w, xr, y, ys, rows, cols, m, sel_blks); }
extern "C" __global__ void __launch_bounds__(256) matvec_cols_exl3_pre_sparse_m4s1_6(const u8* w, const float* xr, float* y, u32 ys, u32 rows, u32 cols, u32 m, const u32* sel_blks) { exl3_mvc2_pre<12, 4, 1, true>(w, xr, y, ys, rows, cols, m, sel_blks); }
extern "C" __global__ void __launch_bounds__(256) matvec_cols_exl3_pre_sparse_m4s1_8(const u8* w, const float* xr, float* y, u32 ys, u32 rows, u32 cols, u32 m, const u32* sel_blks) { exl3_mvc2_pre<16, 4, 1, true>(w, xr, y, ys, rows, cols, m, sel_blks); }

extern "C" __global__ void __launch_bounds__(256) matvec_cols_exl3_pre_sparse_m8s1_2(const u8* w, const float* xr, float* y, u32 ys, u32 rows, u32 cols, u32 m, const u32* sel_blks) { exl3_mvc2_pre<4, 8, 1, true>(w, xr, y, ys, rows, cols, m, sel_blks); }
extern "C" __global__ void __launch_bounds__(256) matvec_cols_exl3_pre_sparse_m8s1_2h(const u8* w, const float* xr, float* y, u32 ys, u32 rows, u32 cols, u32 m, const u32* sel_blks) { exl3_mvc2_pre<5, 8, 1, true>(w, xr, y, ys, rows, cols, m, sel_blks); }
extern "C" __global__ void __launch_bounds__(256) matvec_cols_exl3_pre_sparse_m8s1_3(const u8* w, const float* xr, float* y, u32 ys, u32 rows, u32 cols, u32 m, const u32* sel_blks) { exl3_mvc2_pre<6, 8, 1, true>(w, xr, y, ys, rows, cols, m, sel_blks); }
extern "C" __global__ void __launch_bounds__(256) matvec_cols_exl3_pre_sparse_m8s1_3h(const u8* w, const float* xr, float* y, u32 ys, u32 rows, u32 cols, u32 m, const u32* sel_blks) { exl3_mvc2_pre<7, 8, 1, true>(w, xr, y, ys, rows, cols, m, sel_blks); }
extern "C" __global__ void __launch_bounds__(256) matvec_cols_exl3_pre_sparse_m8s1_4(const u8* w, const float* xr, float* y, u32 ys, u32 rows, u32 cols, u32 m, const u32* sel_blks) { exl3_mvc2_pre<8, 8, 1, true>(w, xr, y, ys, rows, cols, m, sel_blks); }
extern "C" __global__ void __launch_bounds__(256) matvec_cols_exl3_pre_sparse_m8s1_5(const u8* w, const float* xr, float* y, u32 ys, u32 rows, u32 cols, u32 m, const u32* sel_blks) { exl3_mvc2_pre<10, 8, 1, true>(w, xr, y, ys, rows, cols, m, sel_blks); }
extern "C" __global__ void __launch_bounds__(256) matvec_cols_exl3_pre_sparse_m8s1_6(const u8* w, const float* xr, float* y, u32 ys, u32 rows, u32 cols, u32 m, const u32* sel_blks) { exl3_mvc2_pre<12, 8, 1, true>(w, xr, y, ys, rows, cols, m, sel_blks); }
extern "C" __global__ void __launch_bounds__(256) matvec_cols_exl3_pre_sparse_m8s1_8(const u8* w, const float* xr, float* y, u32 ys, u32 rows, u32 cols, u32 m, const u32* sel_blks) { exl3_mvc2_pre<16, 8, 1, true>(w, xr, y, ys, rows, cols, m, sel_blks); }
