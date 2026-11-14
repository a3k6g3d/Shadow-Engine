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

// the 8 weights of lane `lane` of a tile (`tile`: 8 K u32 words)
template <int K>
static __device__ __forceinline__ void exl3_lane_vals(const u32* tile, int lane, float* v) {
    #pragma unroll
    for (int j = 0; j < 8; j++) {
        int t = lane * 8 + j;
        int s = ((t + 1) * K + 256 * K - 16) % (256 * K);
        int wi = s >> 5, sh = s & 31;
        u32 a = tile[wi], b = tile[(wi + 1) % (8 * K)];
        u64 m = ((u64) a << 32) | b;
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

// all 65,536 codebook values (a test of the decode against the CPU reference's)
extern "C" __global__ void exl3_codebook(float* out, u32 n) {
    u32 i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = exl3_cb2(i);
}


extern "C" __global__ void matvec_exl3_1(const u8* w, const float* x, float* y, u32 rows, u32 cols) { exl3_mv<1, 1>(w, x, 0, y, 0, rows, cols, 1); }
extern "C" __global__ void matvec_cols_exl3_1(const u8* w, const float* x, u32 xs, float* y, u32 ys, u32 rows, u32 cols, u32 m) { exl3_mv<1, 8>(w, x, xs, y, ys, rows, cols, m); }
extern "C" __global__ void matvec_batch_exl3_1(const u64* wtab, const float* x, u32 xs, float* y, u32 ys, u32 rows, u32 cols) { exl3_mv<1, 1>((const u8*) wtab[blockIdx.y], x + (u64) blockIdx.y * xs, 0, y + (u64) blockIdx.y * ys, 0, rows, cols, 1); }
extern "C" __global__ void matvec_exl3_2(const u8* w, const float* x, float* y, u32 rows, u32 cols) { exl3_mv<2, 1>(w, x, 0, y, 0, rows, cols, 1); }
extern "C" __global__ void matvec_cols_exl3_2(const u8* w, const float* x, u32 xs, float* y, u32 ys, u32 rows, u32 cols, u32 m) { exl3_mv<2, 8>(w, x, xs, y, ys, rows, cols, m); }
extern "C" __global__ void matvec_batch_exl3_2(const u64* wtab, const float* x, u32 xs, float* y, u32 ys, u32 rows, u32 cols) { exl3_mv<2, 1>((const u8*) wtab[blockIdx.y], x + (u64) blockIdx.y * xs, 0, y + (u64) blockIdx.y * ys, 0, rows, cols, 1); }
extern "C" __global__ void matvec_exl3_3(const u8* w, const float* x, float* y, u32 rows, u32 cols) { exl3_mv<3, 1>(w, x, 0, y, 0, rows, cols, 1); }
extern "C" __global__ void matvec_cols_exl3_3(const u8* w, const float* x, u32 xs, float* y, u32 ys, u32 rows, u32 cols, u32 m) { exl3_mv<3, 8>(w, x, xs, y, ys, rows, cols, m); }
extern "C" __global__ void matvec_batch_exl3_3(const u64* wtab, const float* x, u32 xs, float* y, u32 ys, u32 rows, u32 cols) { exl3_mv<3, 1>((const u8*) wtab[blockIdx.y], x + (u64) blockIdx.y * xs, 0, y + (u64) blockIdx.y * ys, 0, rows, cols, 1); }
extern "C" __global__ void matvec_exl3_4(const u8* w, const float* x, float* y, u32 rows, u32 cols) { exl3_mv<4, 1>(w, x, 0, y, 0, rows, cols, 1); }
extern "C" __global__ void matvec_cols_exl3_4(const u8* w, const float* x, u32 xs, float* y, u32 ys, u32 rows, u32 cols, u32 m) { exl3_mv<4, 8>(w, x, xs, y, ys, rows, cols, m); }
extern "C" __global__ void matvec_batch_exl3_4(const u64* wtab, const float* x, u32 xs, float* y, u32 ys, u32 rows, u32 cols) { exl3_mv<4, 1>((const u8*) wtab[blockIdx.y], x + (u64) blockIdx.y * xs, 0, y + (u64) blockIdx.y * ys, 0, rows, cols, 1); }
extern "C" __global__ void matvec_exl3_5(const u8* w, const float* x, float* y, u32 rows, u32 cols) { exl3_mv<5, 1>(w, x, 0, y, 0, rows, cols, 1); }
extern "C" __global__ void matvec_cols_exl3_5(const u8* w, const float* x, u32 xs, float* y, u32 ys, u32 rows, u32 cols, u32 m) { exl3_mv<5, 8>(w, x, xs, y, ys, rows, cols, m); }
extern "C" __global__ void matvec_batch_exl3_5(const u64* wtab, const float* x, u32 xs, float* y, u32 ys, u32 rows, u32 cols) { exl3_mv<5, 1>((const u8*) wtab[blockIdx.y], x + (u64) blockIdx.y * xs, 0, y + (u64) blockIdx.y * ys, 0, rows, cols, 1); }
extern "C" __global__ void matvec_exl3_6(const u8* w, const float* x, float* y, u32 rows, u32 cols) { exl3_mv<6, 1>(w, x, 0, y, 0, rows, cols, 1); }
extern "C" __global__ void matvec_cols_exl3_6(const u8* w, const float* x, u32 xs, float* y, u32 ys, u32 rows, u32 cols, u32 m) { exl3_mv<6, 8>(w, x, xs, y, ys, rows, cols, m); }
extern "C" __global__ void matvec_batch_exl3_6(const u64* wtab, const float* x, u32 xs, float* y, u32 ys, u32 rows, u32 cols) { exl3_mv<6, 1>((const u8*) wtab[blockIdx.y], x + (u64) blockIdx.y * xs, 0, y + (u64) blockIdx.y * ys, 0, rows, cols, 1); }
extern "C" __global__ void matvec_exl3_7(const u8* w, const float* x, float* y, u32 rows, u32 cols) { exl3_mv<7, 1>(w, x, 0, y, 0, rows, cols, 1); }
extern "C" __global__ void matvec_cols_exl3_7(const u8* w, const float* x, u32 xs, float* y, u32 ys, u32 rows, u32 cols, u32 m) { exl3_mv<7, 8>(w, x, xs, y, ys, rows, cols, m); }
extern "C" __global__ void matvec_batch_exl3_7(const u64* wtab, const float* x, u32 xs, float* y, u32 ys, u32 rows, u32 cols) { exl3_mv<7, 1>((const u8*) wtab[blockIdx.y], x + (u64) blockIdx.y * xs, 0, y + (u64) blockIdx.y * ys, 0, rows, cols, 1); }
extern "C" __global__ void matvec_exl3_8(const u8* w, const float* x, float* y, u32 rows, u32 cols) { exl3_mv<8, 1>(w, x, 0, y, 0, rows, cols, 1); }
extern "C" __global__ void matvec_cols_exl3_8(const u8* w, const float* x, u32 xs, float* y, u32 ys, u32 rows, u32 cols, u32 m) { exl3_mv<8, 8>(w, x, xs, y, ys, rows, cols, m); }
extern "C" __global__ void matvec_batch_exl3_8(const u64* wtab, const float* x, u32 xs, float* y, u32 ys, u32 rows, u32 cols) { exl3_mv<8, 1>((const u8*) wtab[blockIdx.y], x + (u64) blockIdx.y * xs, 0, y + (u64) blockIdx.y * ys, 0, rows, cols, 1); }
