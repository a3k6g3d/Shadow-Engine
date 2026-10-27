// y = W x for a matrix stored as ggml `[cols, rows]` (each of the `rows` rows is `cols` weights, whole blocks,
// contiguous), x and y f32.  One warp per row: each lane decodes every 32nd weight of the row with the quants.cuh
// decoder and accumulates in f32; the warp then sums its lanes.  Correctness first: the per-weight decode re-reads
// block scales, so this is the reference GPU matvec, not the fast one.  Launch: block = 256 threads (8 rows),
// grid = ceil(rows / 8).

template <typename Q>
static __device__ __forceinline__ void matvec_rows(const u8* w, const float* x, float* y, unsigned int rows,
                                                   unsigned int cols) {
    unsigned int row = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32, lane = threadIdx.x % 32;
    if (row >= rows) return;
    const u8* r = w + (u64) row * (cols / Q::QK) * Q::BYTES;
    float acc = 0.0f;
    for (unsigned int i = lane; i < cols; i += 32) acc += Q::at(r + (i / Q::QK) * Q::BYTES, i % Q::QK) * x[i];
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, off);
    if (lane == 0) y[row] = acc;
}

extern "C" __global__ void matvec_q4_k(const u8* w, const float* x, float* y, unsigned int rows, unsigned int cols) { matvec_rows<Q4K>(w, x, y, rows, cols); }
extern "C" __global__ void matvec_q5_k(const u8* w, const float* x, float* y, unsigned int rows, unsigned int cols) { matvec_rows<Q5K>(w, x, y, rows, cols); }
extern "C" __global__ void matvec_q6_k(const u8* w, const float* x, float* y, unsigned int rows, unsigned int cols) { matvec_rows<Q6K>(w, x, y, rows, cols); }
extern "C" __global__ void matvec_iq4_nl(const u8* w, const float* x, float* y, unsigned int rows, unsigned int cols) { matvec_rows<IQ4NL>(w, x, y, rows, cols); }
extern "C" __global__ void matvec_iq4_xs(const u8* w, const float* x, float* y, unsigned int rows, unsigned int cols) { matvec_rows<IQ4XS>(w, x, y, rows, cols); }
extern "C" __global__ void matvec_q2_0(const u8* w, const float* x, float* y, unsigned int rows, unsigned int cols) { matvec_rows<Q20>(w, x, y, rows, cols); }
extern "C" __global__ void matvec_iq2_s(const u8* w, const float* x, float* y, unsigned int rows, unsigned int cols) { matvec_rows<IQ2S>(w, x, y, rows, cols); }
extern "C" __global__ void matvec_iq3_xxs(const u8* w, const float* x, float* y, unsigned int rows, unsigned int cols) { matvec_rows<IQ3XXS>(w, x, y, rows, cols); }
extern "C" __global__ void matvec_iq3_s(const u8* w, const float* x, float* y, unsigned int rows, unsigned int cols) { matvec_rows<IQ3S>(w, x, y, rows, cols); }
extern "C" __global__ void matvec_bf16(const u8* w, const float* x, float* y, unsigned int rows, unsigned int cols) { matvec_rows<BF16>(w, x, y, rows, cols); }
extern "C" __global__ void matvec_f32(const u8* w, const float* x, float* y, unsigned int rows, unsigned int cols) { matvec_rows<F32>(w, x, y, rows, cols); }

// ------------------------------------------------------------------------------------------------ the fast kernels
// Same rows, same weights, decoded 8 consecutive weights at a time: `Fast<Q>::dot8(row, i, x)` is the dot product of
// weights i..i+8 of a row with x[0..8], where each weight is computed exactly as `Q::at` computes it (same operations,
// same order) but a block's scale, grid index and sign byte are decoded once per 8 weights rather than once per
// weight.  Only the order of the f32 sum differs from the reference kernel.  Lane l of a warp takes chunks l, l+32, ...
// of its row, so neighbouring lanes read neighbouring bytes.  Needs cols % 8 == 0 (the host falls back otherwise).

template <typename Q> struct Fast {
    static __device__ __forceinline__ float dot8(const u8* r, unsigned int i, const float* x) {
        float s = 0.0f;
#pragma unroll
        for (unsigned int k = 0; k < 8; k++) s += Q::at(r + ((i + k) / Q::QK) * Q::BYTES, (i + k) % Q::QK) * x[k];
        return s;
    }
};

template <> struct Fast<IQ4NL> {
    static __device__ __forceinline__ float dot8(const u8* r, unsigned int i, const float* x) {
        const u8* b = r + (i / 32) * 18;
        unsigned int e0 = i % 32;
        float d = qf16(b), s = 0.0f;
        const u8* q = b + 2 + (e0 & 15);
#pragma unroll
        for (unsigned int k = 0; k < 8; k++) {
            u8 v = e0 < 16 ? (q[k] & 0x0f) : (q[k] >> 4);
            s += __fmul_rn(d, (float) kIQ4NL[v]) * x[k];
        }
        return s;
    }
};

template <> struct Fast<Q20> {
    static __device__ __forceinline__ float dot8(const u8* r, unsigned int i, const float* x) {
        const u8* b = r + (i / 64) * 18;
        unsigned int e0 = i % 64;
        float d = qf16(b), s = 0.0f;
#pragma unroll
        for (unsigned int k = 0; k < 8; k++) {
            unsigned int e = e0 + k;
            u8 q = (b[2 + e / 4] >> ((e % 4) * 2)) & 0x03;
            s += __fmul_rn((float) ((int) q - 1), d) * x[k];
        }
        return s;
    }
};

template <> struct Fast<IQ2S> {
    static __device__ __forceinline__ float dot8(const u8* r, unsigned int i, const float* x) {
        const u8* b = r + (i / 256) * 82;
        unsigned int e0 = i % 256, j = e0 / 8;
        unsigned int idx = b[2 + j] | (((b[66 + j / 4] >> (2 * (j % 4))) & 3) << 8);
        unsigned int sc = (b[74 + e0 / 32] >> (4 * ((e0 / 16) % 2))) & 0x0f;
        float db = __fmul_rn(__fmul_rn(qf16(b), __fadd_rn(0.5f, (float) sc)), 0.25f), s = 0.0f;
        u8 signs = b[34 + j];
        u64 row = *(const u64*) kIQ2SGrid[idx];
#pragma unroll
        for (unsigned int k = 0; k < 8; k++)
            s += qsign(__fmul_rn(db, (float) (u8) (row >> (8 * k))), signs >> k) * x[k];
        return s;
    }
};

template <> struct Fast<IQ3XXS> {
    static __device__ __forceinline__ float dot8(const u8* r, unsigned int i, const float* x) {
        const u8* b = r + (i / 256) * 98;
        unsigned int e0 = i % 256, g = e0 / 32, kk = (e0 % 32) / 8;
        const u8* sp = b + 66 + g * 4;
        unsigned int s32 = (unsigned int) sp[0] | ((unsigned int) sp[1] << 8) | ((unsigned int) sp[2] << 16)
                         | ((unsigned int) sp[3] << 24);
        float db = __fmul_rn(__fmul_rn(qf16(b), __fadd_rn(0.5f, (float) (s32 >> 28))), 0.5f), s = 0.0f;
        u8 signs = kKSigns[(s32 >> (7 * kk)) & 0x7f];
        u32 r0 = *(const u32*) kIQ3XXSGrid[b[2 + e0 / 4]], r1 = *(const u32*) kIQ3XXSGrid[b[2 + e0 / 4 + 1]];
#pragma unroll
        for (unsigned int k = 0; k < 8; k++) {
            u8 v = (u8) ((k < 4 ? r0 : r1) >> (8 * (k % 4)));
            s += qsign(__fmul_rn(db, (float) v), signs >> k) * x[k];
        }
        return s;
    }
};

template <> struct Fast<IQ3S> {
    static __device__ __forceinline__ float dot8(const u8* r, unsigned int i, const float* x) {
        const u8* b = r + (i / 256) * 110;
        unsigned int e0 = i % 256, j0 = e0 / 4, g = e0 / 32;
        unsigned int sc = (b[106 + g / 2] >> (4 * (g % 2))) & 0x0f;
        float db = __fmul_rn(qf16(b), (float) (1 + 2 * sc)), s = 0.0f;
        u8 signs = b[74 + e0 / 8];
        u32 rr[2];
#pragma unroll
        for (unsigned int h = 0; h < 2; h++) {
            unsigned int j = j0 + h;
            rr[h] = *(const u32*) kIQ3SGrid[b[2 + j] | (((b[66 + j / 8] >> (j % 8)) & 1) << 8)];
        }
#pragma unroll
        for (unsigned int k = 0; k < 8; k++) {
            u8 v = (u8) (rr[k / 4] >> (8 * (k % 4)));
            s += qsign(__fmul_rn(db, (float) v), signs >> k) * x[k];
        }
        return s;
    }
};

template <typename Q>
static __device__ __forceinline__ void matvec_fast_rows(const u8* w, const float* x, float* y, unsigned int rows,
                                                        unsigned int cols) {
    unsigned int row = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32, lane = threadIdx.x % 32;
    if (row >= rows) return;
    const u8* r = w + (u64) row * (cols / Q::QK) * Q::BYTES;
    float acc = 0.0f;
    for (unsigned int c = lane; c < cols / 8; c += 32) acc += Fast<Q>::dot8(r, c * 8, x + c * 8);
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, off);
    if (lane == 0) y[row] = acc;
}

// Several matrices of one shape at once (one token's routed experts): blockIdx.y = n picks matrix wtab[n], input
// x + n * xs and output y + n * ys.  Launch: block = 256 threads (8 rows), grid = (ceil(rows / 8), matrices).
template <typename Q>
static __device__ __forceinline__ void matvec_batch_rows(const u64* wtab, const float* x, unsigned int xs, float* y,
                                                         unsigned int ys, unsigned int rows, unsigned int cols) {
    unsigned int n = blockIdx.y;
    matvec_fast_rows<Q>((const u8*) wtab[n], x + (u64) n * xs, y + (u64) n * ys, rows, cols);
}

extern "C" __global__ void matvec_fast_q4_k(const u8* w, const float* x, float* y, unsigned int rows, unsigned int cols) { matvec_fast_rows<Q4K>(w, x, y, rows, cols); }
extern "C" __global__ void matvec_batch_q4_k(const u64* wtab, const float* x, unsigned int xs, float* y, unsigned int ys, unsigned int rows, unsigned int cols) { matvec_batch_rows<Q4K>(wtab, x, xs, y, ys, rows, cols); }
extern "C" __global__ void matvec_fast_q5_k(const u8* w, const float* x, float* y, unsigned int rows, unsigned int cols) { matvec_fast_rows<Q5K>(w, x, y, rows, cols); }
extern "C" __global__ void matvec_batch_q5_k(const u64* wtab, const float* x, unsigned int xs, float* y, unsigned int ys, unsigned int rows, unsigned int cols) { matvec_batch_rows<Q5K>(wtab, x, xs, y, ys, rows, cols); }
extern "C" __global__ void matvec_fast_q6_k(const u8* w, const float* x, float* y, unsigned int rows, unsigned int cols) { matvec_fast_rows<Q6K>(w, x, y, rows, cols); }
extern "C" __global__ void matvec_batch_q6_k(const u64* wtab, const float* x, unsigned int xs, float* y, unsigned int ys, unsigned int rows, unsigned int cols) { matvec_batch_rows<Q6K>(wtab, x, xs, y, ys, rows, cols); }
extern "C" __global__ void matvec_fast_iq4_nl(const u8* w, const float* x, float* y, unsigned int rows, unsigned int cols) { matvec_fast_rows<IQ4NL>(w, x, y, rows, cols); }
extern "C" __global__ void matvec_batch_iq4_nl(const u64* wtab, const float* x, unsigned int xs, float* y, unsigned int ys, unsigned int rows, unsigned int cols) { matvec_batch_rows<IQ4NL>(wtab, x, xs, y, ys, rows, cols); }
extern "C" __global__ void matvec_fast_iq4_xs(const u8* w, const float* x, float* y, unsigned int rows, unsigned int cols) { matvec_fast_rows<IQ4XS>(w, x, y, rows, cols); }
extern "C" __global__ void matvec_batch_iq4_xs(const u64* wtab, const float* x, unsigned int xs, float* y, unsigned int ys, unsigned int rows, unsigned int cols) { matvec_batch_rows<IQ4XS>(wtab, x, xs, y, ys, rows, cols); }
extern "C" __global__ void matvec_fast_q2_0(const u8* w, const float* x, float* y, unsigned int rows, unsigned int cols) { matvec_fast_rows<Q20>(w, x, y, rows, cols); }
extern "C" __global__ void matvec_batch_q2_0(const u64* wtab, const float* x, unsigned int xs, float* y, unsigned int ys, unsigned int rows, unsigned int cols) { matvec_batch_rows<Q20>(wtab, x, xs, y, ys, rows, cols); }
extern "C" __global__ void matvec_fast_iq2_s(const u8* w, const float* x, float* y, unsigned int rows, unsigned int cols) { matvec_fast_rows<IQ2S>(w, x, y, rows, cols); }
extern "C" __global__ void matvec_batch_iq2_s(const u64* wtab, const float* x, unsigned int xs, float* y, unsigned int ys, unsigned int rows, unsigned int cols) { matvec_batch_rows<IQ2S>(wtab, x, xs, y, ys, rows, cols); }
extern "C" __global__ void matvec_fast_iq3_xxs(const u8* w, const float* x, float* y, unsigned int rows, unsigned int cols) { matvec_fast_rows<IQ3XXS>(w, x, y, rows, cols); }
extern "C" __global__ void matvec_batch_iq3_xxs(const u64* wtab, const float* x, unsigned int xs, float* y, unsigned int ys, unsigned int rows, unsigned int cols) { matvec_batch_rows<IQ3XXS>(wtab, x, xs, y, ys, rows, cols); }
extern "C" __global__ void matvec_fast_iq3_s(const u8* w, const float* x, float* y, unsigned int rows, unsigned int cols) { matvec_fast_rows<IQ3S>(w, x, y, rows, cols); }
extern "C" __global__ void matvec_batch_iq3_s(const u64* wtab, const float* x, unsigned int xs, float* y, unsigned int ys, unsigned int rows, unsigned int cols) { matvec_batch_rows<IQ3S>(wtab, x, xs, y, ys, rows, cols); }
extern "C" __global__ void matvec_fast_bf16(const u8* w, const float* x, float* y, unsigned int rows, unsigned int cols) { matvec_fast_rows<BF16>(w, x, y, rows, cols); }
extern "C" __global__ void matvec_batch_bf16(const u64* wtab, const float* x, unsigned int xs, float* y, unsigned int ys, unsigned int rows, unsigned int cols) { matvec_batch_rows<BF16>(wtab, x, xs, y, ys, rows, cols); }
extern "C" __global__ void matvec_fast_f32(const u8* w, const float* x, float* y, unsigned int rows, unsigned int cols) { matvec_fast_rows<F32>(w, x, y, rows, cols); }
extern "C" __global__ void matvec_batch_f32(const u64* wtab, const float* x, unsigned int xs, float* y, unsigned int ys, unsigned int rows, unsigned int cols) { matvec_batch_rows<F32>(wtab, x, xs, y, ys, rows, cols); }
