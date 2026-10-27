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
extern "C" __global__ void matvec_nvfp4(const u8* w, const float* x, float* y, unsigned int rows, unsigned int cols) { matvec_rows<NVFP4>(w, x, y, rows, cols); }
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

template <> struct Fast<Q4K> {
    static __device__ __forceinline__ float dot8(const u8* r, unsigned int i, const float* x) {
        const u8* b = r + (i / 256) * 144;
        unsigned int e0 = i % 256, ig = e0 / 32, c0 = e0 % 32, sh = (ig % 2) * 4;
        u8 sc, mn;
        q_scale_min_k4(b + 4, ig, &sc, &mn);
        float ds = __fmul_rn(qf16(b), (float) sc), dm = __fmul_rn(qf16(b + 2), (float) mn), s = 0.0f;
        const u8* q = b + 16 + (ig / 2) * 32 + c0;
        u32 q0 = *(const u32*) q, q1 = *(const u32*) (q + 4);
#pragma unroll
        for (unsigned int k = 0; k < 4; k++) {
            u8 v = (u8) ((q0 >> (8 * k + sh)) & 0x0f);
            s += __fsub_rn(__fmul_rn(ds, (float) v), dm) * x[k];
        }
#pragma unroll
        for (unsigned int k = 0; k < 4; k++) {
            u8 v = (u8) ((q1 >> (8 * k + sh)) & 0x0f);
            s += __fsub_rn(__fmul_rn(ds, (float) v), dm) * x[4 + k];
        }
        return s;
    }
};

template <> struct Fast<Q5K> {
    static __device__ __forceinline__ float dot8(const u8* r, unsigned int i, const float* x) {
        const u8* b = r + (i / 256) * 176;
        unsigned int e0 = i % 256, ig = e0 / 32, c0 = e0 % 32, sh = (ig % 2) * 4;
        u8 sc, mn;
        q_scale_min_k4(b + 4, ig, &sc, &mn);
        float ds = __fmul_rn(qf16(b), (float) sc), dm = __fmul_rn(qf16(b + 2), (float) mn), s = 0.0f;
        const u8* ql = b + 48 + (ig / 2) * 32 + c0;
        const u8* qh = b + 16 + c0;
        u32 ql0 = *(const u32*) ql, ql1 = *(const u32*) (ql + 4);
        u32 qh0 = *(const u32*) qh, qh1 = *(const u32*) (qh + 4);
#pragma unroll
        for (unsigned int k = 0; k < 4; k++) {
            u8 v = (u8) (((ql0 >> (8 * k + sh)) & 0x0f) | (((qh0 >> (8 * k + ig)) & 1u) << 4));
            s += __fsub_rn(__fmul_rn(ds, (float) v), dm) * x[k];
        }
#pragma unroll
        for (unsigned int k = 0; k < 4; k++) {
            u8 v = (u8) (((ql1 >> (8 * k + sh)) & 0x0f) | (((qh1 >> (8 * k + ig)) & 1u) << 4));
            s += __fsub_rn(__fmul_rn(ds, (float) v), dm) * x[4 + k];
        }
        return s;
    }
};

template <> struct Fast<Q6K> {
    static __device__ __forceinline__ float dot8(const u8* r, unsigned int i, const float* x) {
        const u8* b = r + (i / 256) * 210;
        unsigned int e0 = i % 256, r_idx = e0 / 32, c0 = e0 % 32;
        unsigned int g = r_idx / 4, h = (r_idx % 4) / 2, k_idx = r_idx % 2, s_idx = r_idx % 4;
        float d = __fmul_rn(qf16(b + 208), (float) (signed char) b[192 + e0 / 16]), s = 0.0f;
        const u16* ql = (const u16*) (b + g * 64 + k_idx * 32 + c0);
        const u16* qh = (const u16*) (b + 128 + g * 32 + c0);
#pragma unroll
        for (unsigned int k = 0; k < 4; k++) {
            u16 wl = ql[k], wh = qh[k];
            u8 lo0 = (wl >> (h * 4)) & 0x0f, hi0 = (wh >> (2 * s_idx)) & 0x03;
            u8 lo1 = (wl >> (8 + h * 4)) & 0x0f, hi1 = (wh >> (8 + 2 * s_idx)) & 0x03;
            int q0 = (int) (signed char) (u8) (lo0 | (hi0 << 4)) - 32;
            int q1 = (int) (signed char) (u8) (lo1 | (hi1 << 4)) - 32;
            s += __fmul_rn(d, (float) q0) * x[2 * k];
            s += __fmul_rn(d, (float) q1) * x[2 * k + 1];
        }
        return s;
    }
};

template <> struct Fast<IQ4NL> {
    static __device__ __forceinline__ float dot8(const u8* r, unsigned int i, const float* x) {
        const u8* b = r + (i / 32) * 18;
        unsigned int e0 = i % 32, sh = (e0 >= 16) ? 4u : 0u;
        float d = qf16(b), s = 0.0f;
        const u16* q = (const u16*) (b + 2 + (e0 & 15));
#pragma unroll
        for (unsigned int k = 0; k < 4; k++) {
            u16 w = q[k];
            s += __fmul_rn(d, iq4nl_val((w >> sh) & 0x0f)) * x[2 * k];
            s += __fmul_rn(d, iq4nl_val((w >> (8 + sh)) & 0x0f)) * x[2 * k + 1];
        }
        return s;
    }
};

template <> struct Fast<IQ4XS> {
    static __device__ __forceinline__ float dot8(const u8* r, unsigned int i, const float* x) {
        const u8* b = r + (i / 256) * 136;
        unsigned int e0 = i % 256, g = e0 / 32, c0 = e0 % 32, sh_nib = (c0 >= 16) ? 4u : 0u;
        u32 hdr = *(const u32*) b;
        u16 sh = (u16) (hdr >> 16);
        u8 lo = (b[4 + g / 2] >> (4 * (g % 2))) & 0x0f;
        u8 hi = (sh >> (2 * g)) & 3;
        float dl = __fmul_rn(f16_to_f32((u16) hdr), (float) ((int) (u8) (lo | (hi << 4)) - 32)), s = 0.0f;
        const u8* q = b + 8 + g * 16 + (c0 & 15);
        u32 q0 = *(const u32*) q, q1 = *(const u32*) (q + 4);
#pragma unroll
        for (unsigned int k = 0; k < 4; k++) {
            s += __fmul_rn(dl, iq4nl_val((q0 >> (8 * k + sh_nib)) & 0x0f)) * x[k];
        }
#pragma unroll
        for (unsigned int k = 0; k < 4; k++) {
            s += __fmul_rn(dl, iq4nl_val((q1 >> (8 * k + sh_nib)) & 0x0f)) * x[4 + k];
        }
        return s;
    }
};

template <> struct Fast<Q20> {
    static __device__ __forceinline__ float dot8(const u8* r, unsigned int i, const float* x) {
        const u8* b = r + (i / 64) * 18;
        unsigned int e0 = i % 64;
        float d = qf16(b), s = 0.0f;
        u16 q16 = *(const u16*) (b + 2 + e0 / 4);
#pragma unroll
        for (unsigned int k = 0; k < 8; k++) {
            u8 q = (q16 >> (2 * k)) & 0x03;
            s += __fmul_rn((float) ((int) q - 1), d) * x[k];
        }
        return s;
    }
};

template <> struct Fast<NVFP4> {
    static __device__ __forceinline__ float dot8(const u8* r, unsigned int i, const float* x) {
        const u8* b = r + (i / 64) * 36;
        unsigned int e0 = i % 64, g = e0 / 16, hi = (e0 & 8) >> 1;
        float d = ue4m3_half(b[g]), s = 0.0f;
        const u8* q = b + 4 + g * 8;
        u32 q0 = *(const u32*) q, q1 = *(const u32*) (q + 4);
#pragma unroll
        for (unsigned int k = 0; k < 4; k++) {
            u8 v0 = (u8) ((q0 >> (8 * k + hi)) & 0x0f);
            s += __fmul_rn(d, e2m1_double(v0)) * x[k];
        }
#pragma unroll
        for (unsigned int k = 0; k < 4; k++) {
            u8 v1 = (u8) ((q1 >> (8 * k + hi)) & 0x0f);
            s += __fmul_rn(d, e2m1_double(v1)) * x[4 + k];
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
        const u16* sp = (const u16*) (b + 66 + g * 4);
        unsigned int s32 = (unsigned int) sp[0] | ((unsigned int) sp[1] << 16);
        float db = __fmul_rn(__fmul_rn(qf16(b), __fadd_rn(0.5f, (float) (s32 >> 28))), 0.5f), s = 0.0f;
        u8 signs = kKSigns[(s32 >> (7 * kk)) & 0x7f];
        u16 q2 = *(const u16*) (b + 2 + e0 / 4);
        u32 r0 = *(const u32*) kIQ3XXSGrid[q2 & 0xff], r1 = *(const u32*) kIQ3XXSGrid[q2 >> 8];
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
        u8 signs = b[74 + e0 / 8], qh = b[66 + g];
        u16 q2 = *(const u16*) (b + 2 + j0);
        unsigned int sh0 = j0 % 8;
        u32 rr[2] = {
            *(const u32*) kIQ3SGrid[(q2 & 0xff) | (((qh >> sh0) & 1u) << 8)],
            *(const u32*) kIQ3SGrid[(q2 >> 8) | (((qh >> (sh0 + 1)) & 1u) << 8)],
        };
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

template <typename Q>
static __device__ __forceinline__ void matvec_pair_fast_rows(const u8* w0, float* y0, unsigned int r0,
                                                             const u8* w1, float* y1, unsigned int r1,
                                                             const float* x, unsigned int cols) {
    unsigned int row = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32, lane = threadIdx.x % 32;
    const u8* w;
    float* y;
    if (row < r0) {
        w = w0; y = y0;
    } else if (row < r0 + r1) {
        row -= r0; w = w1; y = y1;
    } else {
        return;
    }
    const u8* r = w + (u64) row * (cols / Q::QK) * Q::BYTES;
    float acc = 0.0f;
    for (unsigned int c = lane; c < cols / 8; c += 32) acc += Fast<Q>::dot8(r, c * 8, x + c * 8);
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, off);
    if (lane == 0) y[row] = acc;
}

// Two matrices of different quant types sharing x: rows 0..r0 decode as Q0, rows r0..r0+r1 as Q1.  With r0 % 8 == 0
// every 8-row block is one type, so no block mixes decoders.
template <typename Q0, typename Q1>
static __device__ __forceinline__ void matvec_pair2_fast_rows(const u8* w0, float* y0, unsigned int r0,
                                                              const u8* w1, float* y1, unsigned int r1,
                                                              const float* x, unsigned int cols) {
    unsigned int row = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32, lane = threadIdx.x % 32;
    float acc = 0.0f;
    if (row < r0) {
        const u8* r = w0 + (u64) row * (cols / Q0::QK) * Q0::BYTES;
        for (unsigned int c = lane; c < cols / 8; c += 32) acc += Fast<Q0>::dot8(r, c * 8, x + c * 8);
        for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, off);
        if (lane == 0) y0[row] = acc;
    } else if (row < r0 + r1) {
        row -= r0;
        const u8* r = w1 + (u64) row * (cols / Q1::QK) * Q1::BYTES;
        for (unsigned int c = lane; c < cols / 8; c += 32) acc += Fast<Q1>::dot8(r, c * 8, x + c * 8);
        for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, off);
        if (lane == 0) y1[row] = acc;
    }
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
extern "C" __global__ void matvec_fast_nvfp4(const u8* w, const float* x, float* y, unsigned int rows, unsigned int cols) { matvec_fast_rows<NVFP4>(w, x, y, rows, cols); }
extern "C" __global__ void matvec_fast_nvfp4_s(const u8* w, const float* x, float* y, unsigned int rows, unsigned int cols, float scale) {
    unsigned int row = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32, lane = threadIdx.x % 32;
    if (row >= rows) return;
    const u8* r = w + (u64) row * (cols / NVFP4::QK) * NVFP4::BYTES;
    float acc = 0.0f;
    for (unsigned int c = lane; c < cols / 8; c += 32) acc += Fast<NVFP4>::dot8(r, c * 8, x + c * 8);
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, off);
    if (lane == 0) y[row] = acc * scale;
}
extern "C" __global__ void matvec_qkv_nvfp4(const u8* wq, float sq, float* yq, unsigned int rq,
                                            const u8* wk, float sk, float* yk, unsigned int rk,
                                            const u8* wv, float sv, float* yv, unsigned int rv,
                                            const float* x, unsigned int cols) {
    unsigned int row = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32, lane = threadIdx.x % 32;
    const u8* w;
    float scale;
    float* y;
    if (row < rq) {
        w = wq; scale = sq; y = yq;
    } else if (row < rq + rk) {
        row -= rq; w = wk; scale = sk; y = yk;
    } else if (row < rq + rk + rv) {
        row -= rq + rk; w = wv; scale = sv; y = yv;
    } else {
        return;
    }
    const u8* r = w + (u64) row * (cols / NVFP4::QK) * NVFP4::BYTES;
    float acc = 0.0f;
    for (unsigned int c = lane; c < cols / 8; c += 32) acc += Fast<NVFP4>::dot8(r, c * 8, x + c * 8);
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, off);
    if (lane == 0) y[row] = acc * scale;
}
extern "C" __global__ void matvec_batch_nvfp4(const u64* wtab, const float* x, unsigned int xs, float* y, unsigned int ys, unsigned int rows, unsigned int cols) { matvec_batch_rows<NVFP4>(wtab, x, xs, y, ys, rows, cols); }
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
extern "C" __global__ void matvec_pair_bf16(const u8* w0, float* y0, unsigned int r0,
                                            const u8* w1, float* y1, unsigned int r1,
                                            const float* x, unsigned int cols) {
    unsigned int row = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32, lane = threadIdx.x % 32;
    const u8* w;
    float* y;
    if (row < r0) {
        w = w0; y = y0;
    } else if (row < r0 + r1) {
        row -= r0; w = w1; y = y1;
    } else {
        return;
    }
    const u8* r = w + (u64) row * cols * BF16::BYTES;
    float acc = 0.0f;
    for (unsigned int i = lane; i < cols; i += 32) acc += BF16::at(r + (u64) i * BF16::BYTES, 0) * x[i];
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, off);
    if (lane == 0) y[row] = acc;
}
extern "C" __global__ void hc_down_inject_bf16(const u8* w_down, float* down, unsigned int r_down,
                                               const u8* w_inj, float* g, unsigned int r_inj,
                                               const float* rn, unsigned int cols, float div) {
    unsigned int row = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32, lane = threadIdx.x % 32;
    bool is_inj = false;
    const u8* w;
    if (row < r_down) {
        w = w_down;
    } else if (row < r_down + r_inj) {
        row -= r_down; w = w_inj; is_inj = true;
    } else {
        return;
    }
    const u8* r = w + (u64) row * cols * BF16::BYTES;
    float acc = 0.0f;
    for (unsigned int i = lane; i < cols; i += 32) acc += BF16::at(r + (u64) i * BF16::BYTES, 0) * rn[i];
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, off);
    if (lane == 0) {
        float v = acc / div;
        float sig = (float) (1.0 / (1.0 + exp(-(double) v)));
        if (is_inj) g[row] = 2.0f * sig;
        else down[row] = v * sig;
    }
}
extern "C" __global__ void matvec_fast_f32(const u8* w, const float* x, float* y, unsigned int rows, unsigned int cols) { matvec_fast_rows<F32>(w, x, y, rows, cols); }
extern "C" __global__ void matvec_batch_f32(const u64* wtab, const float* x, unsigned int xs, float* y, unsigned int ys, unsigned int rows, unsigned int cols) { matvec_batch_rows<F32>(wtab, x, xs, y, ys, rows, cols); }
extern "C" __global__ void matvec_pair_q4_k(const u8* w0, float* y0, unsigned int r0, const u8* w1, float* y1, unsigned int r1, const float* x, unsigned int cols) { matvec_pair_fast_rows<Q4K>(w0, y0, r0, w1, y1, r1, x, cols); }
extern "C" __global__ void matvec_pair_q5_k(const u8* w0, float* y0, unsigned int r0, const u8* w1, float* y1, unsigned int r1, const float* x, unsigned int cols) { matvec_pair_fast_rows<Q5K>(w0, y0, r0, w1, y1, r1, x, cols); }
extern "C" __global__ void matvec_pair_q6_k(const u8* w0, float* y0, unsigned int r0, const u8* w1, float* y1, unsigned int r1, const float* x, unsigned int cols) { matvec_pair_fast_rows<Q6K>(w0, y0, r0, w1, y1, r1, x, cols); }
extern "C" __global__ void matvec_pair_iq4_nl(const u8* w0, float* y0, unsigned int r0, const u8* w1, float* y1, unsigned int r1, const float* x, unsigned int cols) { matvec_pair_fast_rows<IQ4NL>(w0, y0, r0, w1, y1, r1, x, cols); }
extern "C" __global__ void matvec_pair_iq4_xs(const u8* w0, float* y0, unsigned int r0, const u8* w1, float* y1, unsigned int r1, const float* x, unsigned int cols) { matvec_pair_fast_rows<IQ4XS>(w0, y0, r0, w1, y1, r1, x, cols); }
extern "C" __global__ void matvec_pair_nvfp4(const u8* w0, float* y0, unsigned int r0, const u8* w1, float* y1, unsigned int r1, const float* x, unsigned int cols) { matvec_pair_fast_rows<NVFP4>(w0, y0, r0, w1, y1, r1, x, cols); }
extern "C" __global__ void matvec_pair_q2_0(const u8* w0, float* y0, unsigned int r0, const u8* w1, float* y1, unsigned int r1, const float* x, unsigned int cols) { matvec_pair_fast_rows<Q20>(w0, y0, r0, w1, y1, r1, x, cols); }
extern "C" __global__ void matvec_pair_iq2_s(const u8* w0, float* y0, unsigned int r0, const u8* w1, float* y1, unsigned int r1, const float* x, unsigned int cols) { matvec_pair_fast_rows<IQ2S>(w0, y0, r0, w1, y1, r1, x, cols); }
extern "C" __global__ void matvec_pair_iq3_xxs(const u8* w0, float* y0, unsigned int r0, const u8* w1, float* y1, unsigned int r1, const float* x, unsigned int cols) { matvec_pair_fast_rows<IQ3XXS>(w0, y0, r0, w1, y1, r1, x, cols); }
extern "C" __global__ void matvec_pair_iq3_s(const u8* w0, float* y0, unsigned int r0, const u8* w1, float* y1, unsigned int r1, const float* x, unsigned int cols) { matvec_pair_fast_rows<IQ3S>(w0, y0, r0, w1, y1, r1, x, cols); }
extern "C" __global__ void matvec_pair_q6_k_q4_k(const u8* w0, float* y0, unsigned int r0, const u8* w1, float* y1, unsigned int r1, const float* x, unsigned int cols) { matvec_pair2_fast_rows<Q6K, Q4K>(w0, y0, r0, w1, y1, r1, x, cols); }
extern "C" __global__ void matvec_pair_q4_k_iq4_xs(const u8* w0, float* y0, unsigned int r0, const u8* w1, float* y1, unsigned int r1, const float* x, unsigned int cols) { matvec_pair2_fast_rows<Q4K, IQ4XS>(w0, y0, r0, w1, y1, r1, x, cols); }
extern "C" __global__ void matvec_pair_iq4_xs_q6_k(const u8* w0, float* y0, unsigned int r0, const u8* w1, float* y1, unsigned int r1, const float* x, unsigned int cols) { matvec_pair2_fast_rows<IQ4XS, Q6K>(w0, y0, r0, w1, y1, r1, x, cols); }
extern "C" __global__ void matvec_pair_f32(const u8* w0, float* y0, unsigned int r0,
                                           const u8* w1, float* y1, unsigned int r1,
                                           const float* x, unsigned int cols) {
    unsigned int row = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32, lane = threadIdx.x % 32;
    const u8* w;
    float* y;
    if (row < r0) {
        w = w0; y = y0;
    } else if (row < r0 + r1) {
        row -= r0; w = w1; y = y1;
    } else {
        return;
    }
    const float* r = (const float*) (w + (u64) row * cols * F32::BYTES);
    float acc = 0.0f;
    for (unsigned int i = lane; i < cols; i += 32) acc += r[i] * x[i];
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, off);
    if (lane == 0) y[row] = acc;
}
