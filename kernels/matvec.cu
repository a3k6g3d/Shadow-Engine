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
extern "C" __global__ void matvec_q8_0(const u8* w, const float* x, float* y, unsigned int rows, unsigned int cols) { matvec_rows<Q80>(w, x, y, rows, cols); }
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

// dotm<M>: dot8 for M inputs at once (x_j = x + j * xs), each weight decoded once; acc[j] += that input's dot8, with
// the same operations in the same order per input, so the sums are dot8's to the bit.  Written from each dot8 below.
#define DOTM_ACC(a, b) { float w_ = (a); _Pragma("unroll") for (unsigned int j_ = 0; j_ < M; j_++) s[j_] += w_ * x[(u64) j_ * xs + (b)]; }

template <typename Q> struct Fast {
    static __device__ __forceinline__ float dot8(const u8* r, unsigned int i, const float* x) {
        float s = 0.0f;
#pragma unroll
        for (unsigned int k = 0; k < 8; k++) s += Q::at(r + ((i + k) / Q::QK) * Q::BYTES, (i + k) % Q::QK) * x[k];
        return s;
    }
    template <unsigned int M>
    static __device__ __forceinline__ void dotm(const u8* r, unsigned int i, const float* x, unsigned int xs, float* acc) {
        float s[M] = {};
#pragma unroll
        for (unsigned int k = 0; k < 8; k++) DOTM_ACC(Q::at(r + ((i + k) / Q::QK) * Q::BYTES, (i + k) % Q::QK), k);
        for (unsigned int j = 0; j < M; j++) acc[j] += s[j];
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
    template <unsigned int M>
    static __device__ __forceinline__ void dotm(const u8* r, unsigned int i, const float* x, unsigned int xs, float* acc) {
        const u8* b = r + (i / 256) * 144;
        unsigned int e0 = i % 256, ig = e0 / 32, c0 = e0 % 32, sh = (ig % 2) * 4;
        u8 sc, mn;
        q_scale_min_k4(b + 4, ig, &sc, &mn);
        float ds = __fmul_rn(qf16(b), (float) sc), dm = __fmul_rn(qf16(b + 2), (float) mn);
        float s[M] = {};
        const u8* q = b + 16 + (ig / 2) * 32 + c0;
        u32 q0 = *(const u32*) q, q1 = *(const u32*) (q + 4);
#pragma unroll
        for (unsigned int k = 0; k < 4; k++) {
            u8 v = (u8) ((q0 >> (8 * k + sh)) & 0x0f);
            DOTM_ACC(__fsub_rn(__fmul_rn(ds, (float) v), dm), k);
        }
#pragma unroll
        for (unsigned int k = 0; k < 4; k++) {
            u8 v = (u8) ((q1 >> (8 * k + sh)) & 0x0f);
            DOTM_ACC(__fsub_rn(__fmul_rn(ds, (float) v), dm), 4 + k);
        }
        for (unsigned int j = 0; j < M; j++) acc[j] += s[j];
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
    template <unsigned int M>
    static __device__ __forceinline__ void dotm(const u8* r, unsigned int i, const float* x, unsigned int xs, float* acc) {
        const u8* b = r + (i / 256) * 176;
        unsigned int e0 = i % 256, ig = e0 / 32, c0 = e0 % 32, sh = (ig % 2) * 4;
        u8 sc, mn;
        q_scale_min_k4(b + 4, ig, &sc, &mn);
        float ds = __fmul_rn(qf16(b), (float) sc), dm = __fmul_rn(qf16(b + 2), (float) mn);
        float s[M] = {};
        const u8* ql = b + 48 + (ig / 2) * 32 + c0;
        const u8* qh = b + 16 + c0;
        u32 ql0 = *(const u32*) ql, ql1 = *(const u32*) (ql + 4);
        u32 qh0 = *(const u32*) qh, qh1 = *(const u32*) (qh + 4);
#pragma unroll
        for (unsigned int k = 0; k < 4; k++) {
            u8 v = (u8) (((ql0 >> (8 * k + sh)) & 0x0f) | (((qh0 >> (8 * k + ig)) & 1u) << 4));
            DOTM_ACC(__fsub_rn(__fmul_rn(ds, (float) v), dm), k);
        }
#pragma unroll
        for (unsigned int k = 0; k < 4; k++) {
            u8 v = (u8) (((ql1 >> (8 * k + sh)) & 0x0f) | (((qh1 >> (8 * k + ig)) & 1u) << 4));
            DOTM_ACC(__fsub_rn(__fmul_rn(ds, (float) v), dm), 4 + k);
        }
        for (unsigned int j = 0; j < M; j++) acc[j] += s[j];
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
    template <unsigned int M>
    static __device__ __forceinline__ void dotm(const u8* r, unsigned int i, const float* x, unsigned int xs, float* acc) {
        const u8* b = r + (i / 256) * 210;
        unsigned int e0 = i % 256, r_idx = e0 / 32, c0 = e0 % 32;
        unsigned int g = r_idx / 4, h = (r_idx % 4) / 2, k_idx = r_idx % 2, s_idx = r_idx % 4;
        float d = __fmul_rn(qf16(b + 208), (float) (signed char) b[192 + e0 / 16]);
        float s[M] = {};
        const u16* ql = (const u16*) (b + g * 64 + k_idx * 32 + c0);
        const u16* qh = (const u16*) (b + 128 + g * 32 + c0);
#pragma unroll
        for (unsigned int k = 0; k < 4; k++) {
            u16 wl = ql[k], wh = qh[k];
            u8 lo0 = (wl >> (h * 4)) & 0x0f, hi0 = (wh >> (2 * s_idx)) & 0x03;
            u8 lo1 = (wl >> (8 + h * 4)) & 0x0f, hi1 = (wh >> (8 + 2 * s_idx)) & 0x03;
            int q0 = (int) (signed char) (u8) (lo0 | (hi0 << 4)) - 32;
            int q1 = (int) (signed char) (u8) (lo1 | (hi1 << 4)) - 32;
            DOTM_ACC(__fmul_rn(d, (float) q0), 2 * k);
            DOTM_ACC(__fmul_rn(d, (float) q1), 2 * k + 1);
        }
        for (unsigned int j = 0; j < M; j++) acc[j] += s[j];
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
    template <unsigned int M>
    static __device__ __forceinline__ void dotm(const u8* r, unsigned int i, const float* x, unsigned int xs, float* acc) {
        const u8* b = r + (i / 32) * 18;
        unsigned int e0 = i % 32, sh = (e0 >= 16) ? 4u : 0u;
        float d = qf16(b);
        float s[M] = {};
        const u16* q = (const u16*) (b + 2 + (e0 & 15));
#pragma unroll
        for (unsigned int k = 0; k < 4; k++) {
            u16 w = q[k];
            DOTM_ACC(__fmul_rn(d, iq4nl_val((w >> sh) & 0x0f)), 2 * k);
            DOTM_ACC(__fmul_rn(d, iq4nl_val((w >> (8 + sh)) & 0x0f)), 2 * k + 1);
        }
        for (unsigned int j = 0; j < M; j++) acc[j] += s[j];
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
    template <unsigned int M>
    static __device__ __forceinline__ void dotm(const u8* r, unsigned int i, const float* x, unsigned int xs, float* acc) {
        const u8* b = r + (i / 256) * 136;
        unsigned int e0 = i % 256, g = e0 / 32, c0 = e0 % 32, sh_nib = (c0 >= 16) ? 4u : 0u;
        u32 hdr = *(const u32*) b;
        u16 sh = (u16) (hdr >> 16);
        u8 lo = (b[4 + g / 2] >> (4 * (g % 2))) & 0x0f;
        u8 hi = (sh >> (2 * g)) & 3;
        float dl = __fmul_rn(f16_to_f32((u16) hdr), (float) ((int) (u8) (lo | (hi << 4)) - 32));
        float s[M] = {};
        const u8* q = b + 8 + g * 16 + (c0 & 15);
        u32 q0 = *(const u32*) q, q1 = *(const u32*) (q + 4);
#pragma unroll
        for (unsigned int k = 0; k < 4; k++) {
            DOTM_ACC(__fmul_rn(dl, iq4nl_val((q0 >> (8 * k + sh_nib)) & 0x0f)), k);
        }
#pragma unroll
        for (unsigned int k = 0; k < 4; k++) {
            DOTM_ACC(__fmul_rn(dl, iq4nl_val((q1 >> (8 * k + sh_nib)) & 0x0f)), 4 + k);
        }
        for (unsigned int j = 0; j < M; j++) acc[j] += s[j];
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
    template <unsigned int M>
    static __device__ __forceinline__ void dotm(const u8* r, unsigned int i, const float* x, unsigned int xs, float* acc) {
        const u8* b = r + (i / 64) * 18;
        unsigned int e0 = i % 64;
        float d = qf16(b);
        float s[M] = {};
        u16 q16 = *(const u16*) (b + 2 + e0 / 4);
#pragma unroll
        for (unsigned int k = 0; k < 8; k++) {
            u8 q = (q16 >> (2 * k)) & 0x03;
            DOTM_ACC(__fmul_rn((float) ((int) q - 1), d), k);
        }
        for (unsigned int j = 0; j < M; j++) acc[j] += s[j];
    }
};

template <> struct Fast<Q80> {
    static __device__ __forceinline__ float dot8(const u8* r, unsigned int i, const float* x) {
        const u8* b = r + (i / 32) * 34;
        float d = qf16(b), s = 0.0f;
        const signed char* q = (const signed char*) (b + 2 + (i % 32));
#pragma unroll
        for (unsigned int k = 0; k < 8; k++) s += (float) q[k] * x[k];
        return __fmul_rn(d, s);
    }
    template <unsigned int M>
    static __device__ __forceinline__ void dotm(const u8* r, unsigned int i, const float* x, unsigned int xs, float* acc) {
        const u8* b = r + (i / 32) * 34;
        float d = qf16(b);
        float s[M] = {};
        const signed char* q = (const signed char*) (b + 2 + (i % 32));
#pragma unroll
        for (unsigned int k = 0; k < 8; k++) DOTM_ACC((float) q[k], k);
        for (unsigned int j = 0; j < M; j++) acc[j] += __fmul_rn(d, s[j]);
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
    template <unsigned int M>
    static __device__ __forceinline__ void dotm(const u8* r, unsigned int i, const float* x, unsigned int xs, float* acc) {
        const u8* b = r + (i / 64) * 36;
        unsigned int e0 = i % 64, g = e0 / 16, hi = (e0 & 8) >> 1;
        float d = ue4m3_half(b[g]);
        float s[M] = {};
        const u8* q = b + 4 + g * 8;
        u32 q0 = *(const u32*) q, q1 = *(const u32*) (q + 4);
#pragma unroll
        for (unsigned int k = 0; k < 4; k++) {
            u8 v0 = (u8) ((q0 >> (8 * k + hi)) & 0x0f);
            DOTM_ACC(__fmul_rn(d, e2m1_double(v0)), k);
        }
#pragma unroll
        for (unsigned int k = 0; k < 4; k++) {
            u8 v1 = (u8) ((q1 >> (8 * k + hi)) & 0x0f);
            DOTM_ACC(__fmul_rn(d, e2m1_double(v1)), 4 + k);
        }
        for (unsigned int j = 0; j < M; j++) acc[j] += s[j];
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
    template <unsigned int M>
    static __device__ __forceinline__ void dotm(const u8* r, unsigned int i, const float* x, unsigned int xs, float* acc) {
        const u8* b = r + (i / 256) * 82;
        unsigned int e0 = i % 256, j = e0 / 8;
        unsigned int idx = b[2 + j] | (((b[66 + j / 4] >> (2 * (j % 4))) & 3) << 8);
        unsigned int sc = (b[74 + e0 / 32] >> (4 * ((e0 / 16) % 2))) & 0x0f;
        float db = __fmul_rn(__fmul_rn(qf16(b), __fadd_rn(0.5f, (float) sc)), 0.25f);
        float s[M] = {};
        u8 signs = b[34 + j];
        u64 row = *(const u64*) kIQ2SGrid[idx];
#pragma unroll
        for (unsigned int k = 0; k < 8; k++)
            DOTM_ACC(qsign(__fmul_rn(db, (float) (u8) (row >> (8 * k))), signs >> k), k);
        for (unsigned int j = 0; j < M; j++) acc[j] += s[j];
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
    template <unsigned int M>
    static __device__ __forceinline__ void dotm(const u8* r, unsigned int i, const float* x, unsigned int xs, float* acc) {
        const u8* b = r + (i / 256) * 98;
        unsigned int e0 = i % 256, g = e0 / 32, kk = (e0 % 32) / 8;
        const u16* sp = (const u16*) (b + 66 + g * 4);
        unsigned int s32 = (unsigned int) sp[0] | ((unsigned int) sp[1] << 16);
        float db = __fmul_rn(__fmul_rn(qf16(b), __fadd_rn(0.5f, (float) (s32 >> 28))), 0.5f);
        float s[M] = {};
        u8 signs = kKSigns[(s32 >> (7 * kk)) & 0x7f];
        u16 q2 = *(const u16*) (b + 2 + e0 / 4);
        u32 r0 = *(const u32*) kIQ3XXSGrid[q2 & 0xff], r1 = *(const u32*) kIQ3XXSGrid[q2 >> 8];
#pragma unroll
        for (unsigned int k = 0; k < 8; k++) {
            u8 v = (u8) ((k < 4 ? r0 : r1) >> (8 * (k % 4)));
            DOTM_ACC(qsign(__fmul_rn(db, (float) v), signs >> k), k);
        }
        for (unsigned int j = 0; j < M; j++) acc[j] += s[j];
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
    template <unsigned int M>
    static __device__ __forceinline__ void dotm(const u8* r, unsigned int i, const float* x, unsigned int xs, float* acc) {
        const u8* b = r + (i / 256) * 110;
        unsigned int e0 = i % 256, j0 = e0 / 4, g = e0 / 32;
        unsigned int sc = (b[106 + g / 2] >> (4 * (g % 2))) & 0x0f;
        float db = __fmul_rn(qf16(b), (float) (1 + 2 * sc));
        float s[M] = {};
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
            DOTM_ACC(qsign(__fmul_rn(db, (float) v), signs >> k), k);
        }
        for (unsigned int j = 0; j < M; j++) acc[j] += s[j];
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
extern "C" __global__ void matvec_fast_q8_0(const u8* w, const float* x, float* y, unsigned int rows, unsigned int cols) { matvec_fast_rows<Q80>(w, x, y, rows, cols); }
extern "C" __global__ void matvec_batch_q8_0(const u64* wtab, const float* x, unsigned int xs, float* y, unsigned int ys, unsigned int rows, unsigned int cols) { matvec_batch_rows<Q80>(wtab, x, xs, y, ys, rows, cols); }
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
        float sig = 1.0f / (1.0f + expf(-v));
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
extern "C" __global__ void matvec_pair_q8_0(const u8* w0, float* y0, unsigned int r0, const u8* w1, float* y1, unsigned int r1, const float* x, unsigned int cols) { matvec_pair_fast_rows<Q80>(w0, y0, r0, w1, y1, r1, x, cols); }
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

// ------------------------------------------------------------------------------------------- several inputs at once
// y_j = W x_j for j < m <= 8 inputs (x_j at x + j * xs, y_j at y + j * ys): each warp reads and decodes its row's
// weights once (dotm) and dots them with every input, so a batch of tokens (prefill, a speculative verify pass) pays
// for the weights once.  Same per-input sums as matvec_fast_rows, bit for bit.
// Each warp takes COLS_ROWS consecutive rows, so every input chunk it loads serves COLS_ROWS rows.  The inputs must be
// 16-byte aligned with xs a multiple of 4 (float4 loads).
#define COLS_ROWS 4
// rows a warp takes in the plain matvec_cols kernels (the reference-order ones below keep COLS_ROWS): fewer rows a warp = more warps = more
// loads in flight on a small matrix; each row's sums are the same lane-ordered sums, so every output is unchanged
#ifndef COLS_PLAIN
#define COLS_PLAIN 1
#endif

template <typename Q, unsigned int M>
static __device__ __forceinline__ void matvec_cols_m(const u8* w, const float* x, unsigned int xs, float* y,
                                                     unsigned int ys, unsigned int row0, unsigned int nr,
                                                     unsigned int lane, unsigned int cols) {
    const u64 rb = (u64) (cols / Q::QK) * Q::BYTES;
    float acc[COLS_PLAIN][M];
#pragma unroll
    for (unsigned int k = 0; k < COLS_PLAIN; k++)
#pragma unroll
        for (unsigned int j = 0; j < M; j++) acc[k][j] = 0.0f;
    for (unsigned int c = lane; c < cols / 8; c += 32) {
        // this chunk of every input, loaded once into registers and used for all COLS_PLAIN rows
        float xr[M * 8];
#pragma unroll
        for (unsigned int j = 0; j < M; j++) {
            const float4* p = (const float4*) (x + (u64) j * xs + c * 8);
            float4 a = p[0], b = p[1];
            xr[j * 8 + 0] = a.x; xr[j * 8 + 1] = a.y; xr[j * 8 + 2] = a.z; xr[j * 8 + 3] = a.w;
            xr[j * 8 + 4] = b.x; xr[j * 8 + 5] = b.y; xr[j * 8 + 6] = b.z; xr[j * 8 + 7] = b.w;
        }
#pragma unroll
        for (unsigned int k = 0; k < COLS_PLAIN; k++)
            if (k < nr) Fast<Q>::template dotm<M>(w + (row0 + k) * rb, c * 8, xr, 8, acc[k]);
    }
#pragma unroll
    for (unsigned int k = 0; k < COLS_PLAIN; k++) {
        if (k >= nr) break;
#pragma unroll
        for (unsigned int j = 0; j < M; j++) {
            float a = acc[k][j];
            for (int off = 16; off > 0; off >>= 1) a += __shfl_down_sync(0xffffffffu, a, off);
            if (lane == 0) y[(u64) j * ys + row0 + k] = a;
        }
    }
}

// Launch: block = 256 threads (8 warps x COLS_PLAIN rows = 16 rows), grid = ceil(rows / 16).
template <typename Q>
static __device__ __forceinline__ void matvec_cols_rows_b(const u8* w, const float* x, unsigned int xs, float* y,
                                                          unsigned int ys, unsigned int rows, unsigned int cols,
                                                          unsigned int m, unsigned int bx) {
    unsigned int row0 = (bx * (blockDim.x / 32) + threadIdx.x / 32) * COLS_PLAIN, lane = threadIdx.x % 32;
    if (row0 >= rows) return;
    unsigned int nr = min(rows - row0, (unsigned int) COLS_PLAIN);
    switch (m) {
        case 1: matvec_cols_m<Q, 1>(w, x, xs, y, ys, row0, nr, lane, cols); break;
        case 2: matvec_cols_m<Q, 2>(w, x, xs, y, ys, row0, nr, lane, cols); break;
        case 3: matvec_cols_m<Q, 3>(w, x, xs, y, ys, row0, nr, lane, cols); break;
        case 4: matvec_cols_m<Q, 4>(w, x, xs, y, ys, row0, nr, lane, cols); break;
        case 5: matvec_cols_m<Q, 5>(w, x, xs, y, ys, row0, nr, lane, cols); break;
        case 6: matvec_cols_m<Q, 6>(w, x, xs, y, ys, row0, nr, lane, cols); break;
        case 7: matvec_cols_m<Q, 7>(w, x, xs, y, ys, row0, nr, lane, cols); break;
        default: matvec_cols_m<Q, 8>(w, x, xs, y, ys, row0, nr, lane, cols); break;
    }
}

template <typename Q>
static __device__ __forceinline__ void matvec_cols_rows(const u8* w, const float* x, unsigned int xs, float* y,
                                                        unsigned int ys, unsigned int rows, unsigned int cols,
                                                        unsigned int m) {
    matvec_cols_rows_b<Q>(w, x, xs, y, ys, rows, cols, m, blockIdx.x);
}

extern "C" __global__ void matvec_cols_q4_k(const u8* w, const float* x, unsigned int xs, float* y, unsigned int ys, unsigned int rows, unsigned int cols, unsigned int m) { matvec_cols_rows<Q4K>(w, x, xs, y, ys, rows, cols, m); }
extern "C" __global__ void matvec_cols_q5_k(const u8* w, const float* x, unsigned int xs, float* y, unsigned int ys, unsigned int rows, unsigned int cols, unsigned int m) { matvec_cols_rows<Q5K>(w, x, xs, y, ys, rows, cols, m); }
extern "C" __global__ void matvec_cols_q6_k(const u8* w, const float* x, unsigned int xs, float* y, unsigned int ys, unsigned int rows, unsigned int cols, unsigned int m) { matvec_cols_rows<Q6K>(w, x, xs, y, ys, rows, cols, m); }
extern "C" __global__ void matvec_cols_iq4_nl(const u8* w, const float* x, unsigned int xs, float* y, unsigned int ys, unsigned int rows, unsigned int cols, unsigned int m) { matvec_cols_rows<IQ4NL>(w, x, xs, y, ys, rows, cols, m); }
extern "C" __global__ void matvec_cols_iq4_xs(const u8* w, const float* x, unsigned int xs, float* y, unsigned int ys, unsigned int rows, unsigned int cols, unsigned int m) { matvec_cols_rows<IQ4XS>(w, x, xs, y, ys, rows, cols, m); }
extern "C" __global__ void matvec_cols_nvfp4(const u8* w, const float* x, unsigned int xs, float* y, unsigned int ys, unsigned int rows, unsigned int cols, unsigned int m) { matvec_cols_rows<NVFP4>(w, x, xs, y, ys, rows, cols, m); }
extern "C" __global__ void matvec_cols_q2_0(const u8* w, const float* x, unsigned int xs, float* y, unsigned int ys, unsigned int rows, unsigned int cols, unsigned int m) { matvec_cols_rows<Q20>(w, x, xs, y, ys, rows, cols, m); }
extern "C" __global__ void matvec_cols_q8_0(const u8* w, const float* x, unsigned int xs, float* y, unsigned int ys, unsigned int rows, unsigned int cols, unsigned int m) { matvec_cols_rows<Q80>(w, x, xs, y, ys, rows, cols, m); }
extern "C" __global__ void matvec_cols_iq2_s(const u8* w, const float* x, unsigned int xs, float* y, unsigned int ys, unsigned int rows, unsigned int cols, unsigned int m) { matvec_cols_rows<IQ2S>(w, x, xs, y, ys, rows, cols, m); }
extern "C" __global__ void matvec_cols_iq3_xxs(const u8* w, const float* x, unsigned int xs, float* y, unsigned int ys, unsigned int rows, unsigned int cols, unsigned int m) { matvec_cols_rows<IQ3XXS>(w, x, xs, y, ys, rows, cols, m); }
extern "C" __global__ void matvec_cols_iq3_s(const u8* w, const float* x, unsigned int xs, float* y, unsigned int ys, unsigned int rows, unsigned int cols, unsigned int m) { matvec_cols_rows<IQ3S>(w, x, xs, y, ys, rows, cols, m); }
extern "C" __global__ void matvec_cols_bf16(const u8* w, const float* x, unsigned int xs, float* y, unsigned int ys, unsigned int rows, unsigned int cols, unsigned int m) { matvec_cols_rows<BF16>(w, x, xs, y, ys, rows, cols, m); }
extern "C" __global__ void matvec_cols_f32(const u8* w, const float* x, unsigned int xs, float* y, unsigned int ys, unsigned int rows, unsigned int cols, unsigned int m) { matvec_cols_rows<F32>(w, x, xs, y, ys, rows, cols, m); }

// ------------------------------------------------------------------ tensor cores for a long prompt's projections (inexact)
// Y[token][row] = sum_k W[row][k] * X[token][k] for 32 tokens x 64 rows a block: each warp takes 16 rows and all 32 tokens, as 4
// m16n8k16 FP16 multiply-accumulates (FP32 accumulate) per 16 values of k.  The weight tile (64 rows x 64 k) is decoded to FP16 in
// shared memory with the reference decode (the same float value w = d * q, then rounded to FP16), the input tile converted to FP16 on
// the way in.  Not the reference arithmetic (FP16 rounding of both operands, a different order of sum): the host uses it only in the
// default (inexact) mode.  Needs cols % 64 == 0.  Launch: block = 128 threads, grid = (ceil(rows / 64), ceil(m / 32)).
#define MMA_ROWS 64
#define MMA_TOK 32
#define MMA_K 64
#define MMA_PAD 8

static __device__ __forceinline__ unsigned int pack_half2(float lo, float hi) {
    unsigned int r;
    asm("cvt.rn.f16x2.f32 %0, %2, %1;" : "=r"(r) : "f"(lo), "f"(hi));
    return r;
}

template <typename Q> struct Dec8 {};

// 8 weights i..i+8 of a row, as floats exactly as `Fast<Q>::dotm` computes them
template <> struct Dec8<Q6K> {
    static __device__ __forceinline__ void dec8(const u8* r, unsigned int i, float* o) {
        const u8* b = r + (i / 256) * 210;
        unsigned int e0 = i % 256, r_idx = e0 / 32, c0 = e0 % 32;
        unsigned int g = r_idx / 4, h = (r_idx % 4) / 2, k_idx = r_idx % 2, s_idx = r_idx % 4;
        float d = __fmul_rn(qf16(b + 208), (float) (signed char) b[192 + e0 / 16]);
        const u16* ql = (const u16*) (b + g * 64 + k_idx * 32 + c0);
        const u16* qh = (const u16*) (b + 128 + g * 32 + c0);
#pragma unroll
        for (unsigned int k = 0; k < 4; k++) {
            u16 wl = ql[k], wh = qh[k];
            u8 lo0 = (wl >> (h * 4)) & 0x0f, hi0 = (wh >> (2 * s_idx)) & 0x03;
            u8 lo1 = (wl >> (8 + h * 4)) & 0x0f, hi1 = (wh >> (8 + 2 * s_idx)) & 0x03;
            int q0 = (int) (signed char) (u8) (lo0 | (hi0 << 4)) - 32;
            int q1 = (int) (signed char) (u8) (lo1 | (hi1 << 4)) - 32;
            o[2 * k] = __fmul_rn(d, (float) q0);
            o[2 * k + 1] = __fmul_rn(d, (float) q1);
        }
    }
};

template <> struct Dec8<IQ4XS> {
    static __device__ __forceinline__ void dec8(const u8* r, unsigned int i, float* o) {
        const u8* b = r + (i / 256) * 136;
        unsigned int e0 = i % 256, g = e0 / 32, c0 = e0 % 32, sh_nib = (c0 >= 16) ? 4u : 0u;
        u32 hdr = *(const u32*) b;
        u16 sh = (u16) (hdr >> 16);
        u8 lo = (b[4 + g / 2] >> (4 * (g % 2))) & 0x0f;
        u8 hi = (sh >> (2 * g)) & 3;
        float dl = __fmul_rn(f16_to_f32((u16) hdr), (float) ((int) (u8) (lo | (hi << 4)) - 32));
        const u8* q = b + 8 + g * 16 + (c0 & 15);
        u32 q0 = *(const u32*) q, q1 = *(const u32*) (q + 4);
#pragma unroll
        for (unsigned int k = 0; k < 4; k++) o[k] = __fmul_rn(dl, iq4nl_val((q0 >> (8 * k + sh_nib)) & 0x0f));
#pragma unroll
        for (unsigned int k = 0; k < 4; k++) o[4 + k] = __fmul_rn(dl, iq4nl_val((q1 >> (8 * k + sh_nib)) & 0x0f));
    }
};

template <typename Q>
static __device__ __forceinline__ void mma_cols(const u8* w, const float* x, unsigned int xs, float* y, unsigned int ys,
                                                unsigned int rows, unsigned int cols, unsigned int m) {
    __shared__ __align__(16) unsigned short ws[MMA_ROWS][MMA_K + MMA_PAD];
    __shared__ __align__(16) unsigned short xt[MMA_TOK][MMA_K + MMA_PAD];
    const unsigned int tid = threadIdx.x, warp = tid / 32, lane = tid % 32, grp = lane / 4, tig = lane % 4;
    const unsigned int row_base = blockIdx.x * MMA_ROWS, tok_base = blockIdx.y * MMA_TOK;
    const u64 rb = (u64) (cols / Q::QK) * Q::BYTES;
    float acc[4][4];
#pragma unroll
    for (unsigned int n = 0; n < 4; n++)
#pragma unroll
        for (unsigned int e = 0; e < 4; e++) acc[n][e] = 0.0f;
    for (unsigned int kc = 0; kc < cols; kc += MMA_K) {
#pragma unroll
        for (unsigned int t = 0; t < 4; t++) {
            unsigned int ch = tid + 128 * t, row = ch / 8, kk = (ch % 8) * 8;
            float v[8];
            if (row_base + row < rows) {
                Dec8<Q>::dec8(w + (u64) (row_base + row) * rb, kc + kk, v);
            } else {
#pragma unroll
                for (unsigned int e = 0; e < 8; e++) v[e] = 0.0f;
            }
            uint4 p;
            p.x = pack_half2(v[0], v[1]); p.y = pack_half2(v[2], v[3]); p.z = pack_half2(v[4], v[5]); p.w = pack_half2(v[6], v[7]);
            *(uint4*) &ws[row][kk] = p;
        }
#pragma unroll
        for (unsigned int t = 0; t < 4; t++) {
            unsigned int idx = tid + 128 * t, tok = idx / 16, kq = (idx % 16) * 4;
            float4 f = make_float4(0.f, 0.f, 0.f, 0.f);
            if (tok_base + tok < m) f = *(const float4*) (x + (u64) (tok_base + tok) * xs + kc + kq);
            uint2 p;
            p.x = pack_half2(f.x, f.y); p.y = pack_half2(f.z, f.w);
            *(uint2*) &xt[tok][kq] = p;
        }
        __syncthreads();
#pragma unroll
        for (unsigned int ks = 0; ks < MMA_K / 16; ks++) {
            const unsigned int k0 = ks * 16 + tig * 2;
            unsigned int a0 = *(const unsigned int*) &ws[warp * 16 + grp][k0];
            unsigned int a1 = *(const unsigned int*) &ws[warp * 16 + grp + 8][k0];
            unsigned int a2 = *(const unsigned int*) &ws[warp * 16 + grp][k0 + 8];
            unsigned int a3 = *(const unsigned int*) &ws[warp * 16 + grp + 8][k0 + 8];
#pragma unroll
            for (unsigned int n = 0; n < 4; n++) {
                unsigned int b0 = *(const unsigned int*) &xt[n * 8 + grp][k0];
                unsigned int b1 = *(const unsigned int*) &xt[n * 8 + grp][k0 + 8];
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                             : "+f"(acc[n][0]), "+f"(acc[n][1]), "+f"(acc[n][2]), "+f"(acc[n][3])
                             : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (unsigned int n = 0; n < 4; n++)
#pragma unroll
        for (unsigned int e = 0; e < 4; e++) {
            unsigned int row = row_base + warp * 16 + grp + (e >= 2 ? 8 : 0), tok = tok_base + n * 8 + tig * 2 + (e & 1);
            if (row < rows && tok < m) y[(u64) tok * ys + row] = acc[n][e];
        }
}
extern "C" __global__ void mma_cols_q6_k(const u8* w, const float* x, unsigned int xs, float* y, unsigned int ys, unsigned int rows, unsigned int cols, unsigned int m) { mma_cols<Q6K>(w, x, xs, y, ys, rows, cols, m); }
extern "C" __global__ void mma_cols_iq4_xs(const u8* w, const float* x, unsigned int xs, float* y, unsigned int ys, unsigned int rows, unsigned int cols, unsigned int m) { mma_cols<IQ4XS>(w, x, xs, y, ys, rows, cols, m); }

// ------------------------------------------------------------------ several inputs at once, in the reference order
// One token runs BF16 and F32 matrices through the reference order (matvec_rows, matvec_pair_bf16/f32 and
// hc_down_inject_bf16: lane l adds the products of weights l, l+32, ... one at a time), not dot8's, so a batched step
// that matches it bit for bit needs the same order: this is that order for m <= 8 inputs, COLS_ROWS rows per warp.
// Rows [0, r0) are w0's (outputs y0 + j * ys0), rows [r0, r0 + r1) w1's (y1 + j * ys1).  mode 0 writes the sums; mode
// 1 is hc_down_inject's epilogue (v = sum / div: w0's rows get v * sigmoid(v), w1's 2 * sigmoid(v)).
template <typename Q, unsigned int M, unsigned int RW>
static __device__ __forceinline__ void cols_ref_m(const u8* w0, float* y0, unsigned int r0, unsigned int ys0,
                                                  const u8* w1, float* y1, unsigned int ys1, const float* x,
                                                  unsigned int xs, unsigned int row0, unsigned int nr, unsigned int lane,
                                                  unsigned int cols, unsigned int mode, float div) {
    const u8* wr[RW];
#pragma unroll
    for (unsigned int k = 0; k < RW; k++) {
        unsigned int row = row0 + k;
        wr[k] = row < r0 ? w0 + (u64) row * cols * Q::BYTES : w1 + (u64) (row - r0) * cols * Q::BYTES;
    }
    float acc[RW][M];
#pragma unroll
    for (unsigned int k = 0; k < RW; k++)
#pragma unroll
        for (unsigned int j = 0; j < M; j++) acc[k][j] = 0.0f;
    for (unsigned int i = lane; i < cols; i += 32) {
        float xr[M];
#pragma unroll
        for (unsigned int j = 0; j < M; j++) xr[j] = x[(u64) j * xs + i];
#pragma unroll
        for (unsigned int k = 0; k < RW; k++)
            if (k < nr) {
                float wv = Q::at(wr[k] + (u64) i * Q::BYTES, 0);
#pragma unroll
                for (unsigned int j = 0; j < M; j++) acc[k][j] += wv * xr[j];
            }
    }
#pragma unroll
    for (unsigned int k = 0; k < RW; k++) {
        if (k >= nr) break;
        unsigned int row = row0 + k;
#pragma unroll
        for (unsigned int j = 0; j < M; j++) {
            float a = acc[k][j];
            for (int off = 16; off > 0; off >>= 1) a += __shfl_down_sync(0xffffffffu, a, off);
            if (lane != 0) continue;
            if (mode == 1) {
                float v = a / div;
                float sig = 1.0f / (1.0f + expf(-v));
                a = row < r0 ? v * sig : 2.0f * sig;
            }
            if (row < r0) y0[(u64) j * ys0 + row] = a;
            else y1[(u64) j * ys1 + row - r0] = a;
        }
    }
}

// Rows a warp takes: COLS_ROWS, or 1 for a matrix of fewer than COLS_REF_SPLIT rows (so a short, wide one - the hc
// down projections' 324 rows - still spreads over the GPU; each row's sum is the same either way).
#define COLS_REF_SPLIT 2048
// Launch: block = 256 threads, grid = ceil((r0 + r1) / (8 * rows a warp)).
template <typename Q>
static __device__ __forceinline__ void cols_ref_rows(const u8* w0, float* y0, unsigned int r0, unsigned int ys0,
                                                     const u8* w1, float* y1, unsigned int r1, unsigned int ys1,
                                                     const float* x, unsigned int xs, unsigned int cols,
                                                     unsigned int mode, float div, unsigned int m) {
    const bool one = r0 + r1 < COLS_REF_SPLIT;
    const unsigned int rw = one ? 1 : COLS_ROWS;
    unsigned int row0 = (blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32) * rw, lane = threadIdx.x % 32;
    if (row0 >= r0 + r1) return;
    unsigned int nr = min(r0 + r1 - row0, rw);
#define CR(M) do { if (one) cols_ref_m<Q, M, 1>(w0, y0, r0, ys0, w1, y1, ys1, x, xs, row0, nr, lane, cols, mode, div);                    else cols_ref_m<Q, M, COLS_ROWS>(w0, y0, r0, ys0, w1, y1, ys1, x, xs, row0, nr, lane, cols, mode, div); } while (0)
    switch (m) {
        case 1: CR(1); break;
        case 2: CR(2); break;
        case 3: CR(3); break;
        case 4: CR(4); break;
        case 5: CR(5); break;
        case 6: CR(6); break;
        case 7: CR(7); break;
        default: CR(8); break;
    }
#undef CR
}

extern "C" __global__ void matvec_cols_ref_bf16(const u8* w, const float* x, unsigned int xs, float* y, unsigned int ys, unsigned int rows, unsigned int cols, unsigned int m) { cols_ref_rows<BF16>(w, y, rows, ys, w, y, 0, ys, x, xs, cols, 0, 1.0f, m); }
extern "C" __global__ void matvec_cols_ref_f32(const u8* w, const float* x, unsigned int xs, float* y, unsigned int ys, unsigned int rows, unsigned int cols, unsigned int m) { cols_ref_rows<F32>(w, y, rows, ys, w, y, 0, ys, x, xs, cols, 0, 1.0f, m); }
// hc_down_inject_bf16 for m inputs rn + j * rs: down + j * ds, g + j * gs.
extern "C" __global__ void hc_down_inject_cols_bf16(const u8* w_down, float* down, unsigned int r_down, unsigned int ds,
                                                    const u8* w_inj, float* g, unsigned int r_inj, unsigned int gs,
                                                    const float* rn, unsigned int rs, unsigned int cols, float div,
                                                    unsigned int m) {
    cols_ref_rows<BF16>(w_down, down, r_down, ds, w_inj, g, r_inj, gs, rn, rs, cols, 1, div, m);
}

// ------------------------------------------------------------------------------------- matrices on their own inputs
// A batch's routed experts: group g = blockIdx.y is matrix wtab[g] on its own inputs.  gtab[17 g] = n (<= 8), then
// 8 input rows and 8 output rows: input j at x + xi[j] * xs, output j at y + yi[j] * ys.  Same sums as
// matvec_cols_m (dotm), so each one is matvec_fast_rows' on that input, bit for bit.  The inputs must be 16-byte
// aligned with xs a multiple of 4.  Launch: block = 256 threads (8 * GROUPED_RW rows), grid = (ceil(rows / (8 *
// GROUPED_RW)), groups).
template <typename Q, unsigned int M, unsigned int RW>
static __device__ __forceinline__ void grouped_m(const u8* w, const float* x, unsigned int xs, float* y,
                                                 unsigned int ys, const unsigned int* gi, unsigned int row0,
                                                 unsigned int nr, unsigned int lane, unsigned int cols) {
    const u64 rb = (u64) (cols / Q::QK) * Q::BYTES;
    const float* xp[M];
    float* yp[M];
#pragma unroll
    for (unsigned int j = 0; j < M; j++) {
        xp[j] = x + (u64) gi[1 + j] * xs;
        yp[j] = y + (u64) gi[9 + j] * ys;
    }
    float acc[RW][M];
#pragma unroll
    for (unsigned int k = 0; k < RW; k++)
#pragma unroll
        for (unsigned int j = 0; j < M; j++) acc[k][j] = 0.0f;
    for (unsigned int c = lane; c < cols / 8; c += 32) {
        float xr[M * 8];
#pragma unroll
        for (unsigned int j = 0; j < M; j++) {
            const float4* p = (const float4*) (xp[j] + c * 8);
            float4 a = p[0], b = p[1];
            xr[j * 8 + 0] = a.x; xr[j * 8 + 1] = a.y; xr[j * 8 + 2] = a.z; xr[j * 8 + 3] = a.w;
            xr[j * 8 + 4] = b.x; xr[j * 8 + 5] = b.y; xr[j * 8 + 6] = b.z; xr[j * 8 + 7] = b.w;
        }
#pragma unroll
        for (unsigned int k = 0; k < RW; k++)
            if (k < nr) Fast<Q>::template dotm<M>(w + (row0 + k) * rb, c * 8, xr, 8, acc[k]);
    }
#pragma unroll
    for (unsigned int k = 0; k < RW; k++) {
        if (k >= nr) break;
#pragma unroll
        for (unsigned int j = 0; j < M; j++) {
            float a = acc[k][j];
            for (int off = 16; off > 0; off >>= 1) a += __shfl_down_sync(0xffffffffu, a, off);
            if (lane == 0) yp[j][row0 + k] = a;
        }
    }
}

#define GROUPED_RW 1
template <typename Q>
static __device__ __forceinline__ void grouped_rows(const u64* wtab, const unsigned int* gtab, const float* x,
                                                    unsigned int xs, float* y, unsigned int ys, unsigned int rows,
                                                    unsigned int cols) {
    const unsigned int* gi = gtab + (u64) blockIdx.y * 17;
    const u8* w = (const u8*) wtab[blockIdx.y];
    // one row a warp (GROUPED_RW): a routed expert has 640 or 2560 rows and a group few inputs, so the GPU needs the
    // warps more than the input reuse of several rows a warp
    unsigned int row0 = (blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32) * GROUPED_RW, lane = threadIdx.x % 32;
    if (row0 >= rows) return;
    unsigned int nr = min(rows - row0, (unsigned int) GROUPED_RW);
    switch (gi[0]) {
        case 1: grouped_m<Q, 1, GROUPED_RW>(w, x, xs, y, ys, gi, row0, nr, lane, cols); break;
        case 2: grouped_m<Q, 2, GROUPED_RW>(w, x, xs, y, ys, gi, row0, nr, lane, cols); break;
        case 3: grouped_m<Q, 3, GROUPED_RW>(w, x, xs, y, ys, gi, row0, nr, lane, cols); break;
        case 4: grouped_m<Q, 4, GROUPED_RW>(w, x, xs, y, ys, gi, row0, nr, lane, cols); break;
        case 5: grouped_m<Q, 5, GROUPED_RW>(w, x, xs, y, ys, gi, row0, nr, lane, cols); break;
        case 6: grouped_m<Q, 6, GROUPED_RW>(w, x, xs, y, ys, gi, row0, nr, lane, cols); break;
        case 7: grouped_m<Q, 7, GROUPED_RW>(w, x, xs, y, ys, gi, row0, nr, lane, cols); break;
        default: grouped_m<Q, 8, GROUPED_RW>(w, x, xs, y, ys, gi, row0, nr, lane, cols); break;
    }
}

extern "C" __global__ void matvec_grouped_q4_k(const u64* wtab, const unsigned int* gtab, const float* x, unsigned int xs, float* y, unsigned int ys, unsigned int rows, unsigned int cols) { grouped_rows<Q4K>(wtab, gtab, x, xs, y, ys, rows, cols); }
extern "C" __global__ void matvec_grouped_q5_k(const u64* wtab, const unsigned int* gtab, const float* x, unsigned int xs, float* y, unsigned int ys, unsigned int rows, unsigned int cols) { grouped_rows<Q5K>(wtab, gtab, x, xs, y, ys, rows, cols); }
extern "C" __global__ void matvec_grouped_q6_k(const u64* wtab, const unsigned int* gtab, const float* x, unsigned int xs, float* y, unsigned int ys, unsigned int rows, unsigned int cols) { grouped_rows<Q6K>(wtab, gtab, x, xs, y, ys, rows, cols); }
extern "C" __global__ void matvec_grouped_iq4_nl(const u64* wtab, const unsigned int* gtab, const float* x, unsigned int xs, float* y, unsigned int ys, unsigned int rows, unsigned int cols) { grouped_rows<IQ4NL>(wtab, gtab, x, xs, y, ys, rows, cols); }
extern "C" __global__ void matvec_grouped_iq4_xs(const u64* wtab, const unsigned int* gtab, const float* x, unsigned int xs, float* y, unsigned int ys, unsigned int rows, unsigned int cols) { grouped_rows<IQ4XS>(wtab, gtab, x, xs, y, ys, rows, cols); }
extern "C" __global__ void matvec_grouped_nvfp4(const u64* wtab, const unsigned int* gtab, const float* x, unsigned int xs, float* y, unsigned int ys, unsigned int rows, unsigned int cols) { grouped_rows<NVFP4>(wtab, gtab, x, xs, y, ys, rows, cols); }
extern "C" __global__ void matvec_grouped_q2_0(const u64* wtab, const unsigned int* gtab, const float* x, unsigned int xs, float* y, unsigned int ys, unsigned int rows, unsigned int cols) { grouped_rows<Q20>(wtab, gtab, x, xs, y, ys, rows, cols); }
extern "C" __global__ void matvec_grouped_q8_0(const u64* wtab, const unsigned int* gtab, const float* x, unsigned int xs, float* y, unsigned int ys, unsigned int rows, unsigned int cols) { grouped_rows<Q80>(wtab, gtab, x, xs, y, ys, rows, cols); }
extern "C" __global__ void matvec_grouped_iq2_s(const u64* wtab, const unsigned int* gtab, const float* x, unsigned int xs, float* y, unsigned int ys, unsigned int rows, unsigned int cols) { grouped_rows<IQ2S>(wtab, gtab, x, xs, y, ys, rows, cols); }
extern "C" __global__ void matvec_grouped_iq3_xxs(const u64* wtab, const unsigned int* gtab, const float* x, unsigned int xs, float* y, unsigned int ys, unsigned int rows, unsigned int cols) { grouped_rows<IQ3XXS>(wtab, gtab, x, xs, y, ys, rows, cols); }
extern "C" __global__ void matvec_grouped_iq3_s(const u64* wtab, const unsigned int* gtab, const float* x, unsigned int xs, float* y, unsigned int ys, unsigned int rows, unsigned int cols) { grouped_rows<IQ3S>(wtab, gtab, x, xs, y, ys, rows, cols); }

// ------------------------------------------------------------------------- BF16 hc projections in the dot8 order
// hc_down_inject_bf16 with matvec_fast_rows' sums (dot8 over 8 consecutive weights a lane) instead of the reference
// order: vector loads, same math, only the float sum order differs.  One row a warp; grid = ceil((r_down + r_inj) / 8).
static __device__ __forceinline__ float hc_epilogue(float a, float div, bool inj) {
    float v = a / div;
    float sig = 1.0f / (1.0f + expf(-v));
    return inj ? 2.0f * sig : v * sig;
}

// Four warps share a row (the 324 rows would fill a GPU's SMs only a quarter): warp w takes chunks 32 w + lane + 128 i,
// each warp reduces its lanes, and the four partials are added in order w = 0..3.  Two rows a block of 256 threads;
// grid = ceil((r_down + r_inj) / 2).
extern "C" __global__ void hc_down_inject_fast_bf16(const u8* w_down, float* down, unsigned int r_down,
                                                    const u8* w_inj, float* g, unsigned int r_inj,
                                                    const float* rn, unsigned int cols, float div) {
    __shared__ float red[2][4];
    const unsigned int sub = threadIdx.x / 128, wq = (threadIdx.x / 32) % 4, lane = threadIdx.x % 32;
    unsigned int row = blockIdx.x * 2 + sub;
    const bool valid = row < r_down + r_inj;
    bool inj = false;
    const u8* w = w_down;
    if (valid && row >= r_down) {
        row -= r_down; w = w_inj; inj = true;
    }
    float acc = 0.0f;
    if (valid) {
        const u8* r = w + (u64) row * cols * BF16::BYTES;
        for (unsigned int c = wq * 32 + lane; c < cols / 8; c += 128) acc += Fast<BF16>::dot8(r, c * 8, rn + c * 8);
    }
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, off);
    if (lane == 0) red[sub][wq] = acc;
    __syncthreads();
    if (valid && wq == 0 && lane == 0) {
        float a = red[sub][0] + red[sub][1] + red[sub][2] + red[sub][3];
        (inj ? g : down)[row] = hc_epilogue(a, div, inj);
    }
}

// The same for m <= 8 inputs (rn + j * rs; down + j * ds, g + j * gs), dotm so each input's sums are the one-input
// kernel's bit for bit.  One row a warp; grid = ceil((r_down + r_inj) / 8).
template <unsigned int M>
static __device__ __forceinline__ void hc_cols_fast_m(const u8* r, const float* rn, unsigned int rs, unsigned int cols,
                                                      unsigned int wq, unsigned int lane, bool valid, float* acc) {
#pragma unroll
    for (unsigned int j = 0; j < M; j++) acc[j] = 0.0f;
    if (valid) {
        for (unsigned int c = wq * 32 + lane; c < cols / 8; c += 128) {
            float xr[M * 8];
#pragma unroll
            for (unsigned int j = 0; j < M; j++) {
                const float4* p = (const float4*) (rn + (u64) j * rs + c * 8);
                float4 a = p[0], b = p[1];
                xr[j * 8 + 0] = a.x; xr[j * 8 + 1] = a.y; xr[j * 8 + 2] = a.z; xr[j * 8 + 3] = a.w;
                xr[j * 8 + 4] = b.x; xr[j * 8 + 5] = b.y; xr[j * 8 + 6] = b.z; xr[j * 8 + 7] = b.w;
            }
            Fast<BF16>::template dotm<M>(r, c * 8, xr, 8, acc);
        }
    }
#pragma unroll
    for (unsigned int j = 0; j < M; j++)
        for (int off = 16; off > 0; off >>= 1) acc[j] += __shfl_down_sync(0xffffffffu, acc[j], off);
}

extern "C" __global__ void hc_down_inject_cols_fast_bf16(const u8* w_down, float* down, unsigned int r_down, unsigned int ds,
                                                         const u8* w_inj, float* g, unsigned int r_inj, unsigned int gs,
                                                         const float* rn, unsigned int rs, unsigned int cols, float div,
                                                         unsigned int m) {
    __shared__ float red[2][4][8];
    const unsigned int sub = threadIdx.x / 128, wq = (threadIdx.x / 32) % 4, lane = threadIdx.x % 32;
    unsigned int row = blockIdx.x * 2 + sub;
    const bool valid = row < r_down + r_inj;
    bool inj = false;
    const u8* w = w_down;
    if (valid && row >= r_down) {
        row -= r_down; w = w_inj; inj = true;
    }
    const u8* r = w + (u64) row * cols * BF16::BYTES;
    float acc[8];
    switch (m) {
        case 1: hc_cols_fast_m<1>(r, rn, rs, cols, wq, lane, valid, acc); break;
        case 2: hc_cols_fast_m<2>(r, rn, rs, cols, wq, lane, valid, acc); break;
        case 3: hc_cols_fast_m<3>(r, rn, rs, cols, wq, lane, valid, acc); break;
        case 4: hc_cols_fast_m<4>(r, rn, rs, cols, wq, lane, valid, acc); break;
        case 5: hc_cols_fast_m<5>(r, rn, rs, cols, wq, lane, valid, acc); break;
        case 6: hc_cols_fast_m<6>(r, rn, rs, cols, wq, lane, valid, acc); break;
        case 7: hc_cols_fast_m<7>(r, rn, rs, cols, wq, lane, valid, acc); break;
        default: hc_cols_fast_m<8>(r, rn, rs, cols, wq, lane, valid, acc); break;
    }
    if (lane == 0)
        for (unsigned int j = 0; j < m && j < 8; j++) red[sub][wq][j] = acc[j];
    __syncthreads();
    if (valid && wq == 0 && lane == 0) {
        for (unsigned int j = 0; j < m && j < 8; j++) {
            float a = red[sub][0][j] + red[sub][1][j] + red[sub][2][j] + red[sub][3][j];
            if (inj) g[(u64) j * gs + row] = hc_epilogue(a, div, true);
            else down[(u64) j * ds + row] = hc_epilogue(a, div, false);
        }
    }
}

// matvec_fast_rows over the rows ids[i] of a matrix (the draft head: 106k of the 248k vocabulary rows): y[i] = W[ids[i]] x.
template <typename Q>
static __device__ __forceinline__ void matvec_ids_rows(const u8* w, const unsigned int* ids, const float* x, float* y,
                                                       unsigned int n, unsigned int cols) {
    unsigned int i = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32, lane = threadIdx.x % 32;
    if (i >= n) return;
    const u8* r = w + (u64) ids[i] * (cols / Q::QK) * Q::BYTES;
    float acc = 0.0f;
    for (unsigned int c = lane; c < cols / 8; c += 32) acc += Fast<Q>::dot8(r, c * 8, x + c * 8);
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, off);
    if (lane == 0) y[i] = acc;
}
extern "C" __global__ void matvec_ids_q6_k(const u8* w, const unsigned int* ids, const float* x, float* y, unsigned int n, unsigned int cols) { matvec_ids_rows<Q6K>(w, ids, x, y, n, cols); }
extern "C" __global__ void matvec_ids_q4_k(const u8* w, const unsigned int* ids, const float* x, float* y, unsigned int n, unsigned int cols) { matvec_ids_rows<Q4K>(w, ids, x, y, n, cols); }
extern "C" __global__ void matvec_ids_iq4_xs(const u8* w, const unsigned int* ids, const float* x, float* y, unsigned int n, unsigned int cols) { matvec_ids_rows<IQ4XS>(w, ids, x, y, n, cols); }

// matvec_ids_rows for M inputs at once (x_j = x + j * xs, y_j = y + j * ys): each selected weight row is decoded once.
template <typename Q, unsigned int M>
static __device__ __forceinline__ void matvec_ids_cols_m(const u8* w, const unsigned int* ids, const float* x, unsigned int xs,
                                                         float* y, unsigned int ys, unsigned int n, unsigned int cols) {
    unsigned int i = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32, lane = threadIdx.x % 32;
    if (i >= n) return;
    const u8* r = w + (u64) ids[i] * (cols / Q::QK) * Q::BYTES;
    float acc[M];
#pragma unroll
    for (unsigned int j = 0; j < M; j++) acc[j] = 0.0f;
    for (unsigned int c = lane; c < cols / 8; c += 32) {
        float xr[M * 8];
#pragma unroll
        for (unsigned int j = 0; j < M; j++) {
            const float4* p = (const float4*) (x + (u64) j * xs + c * 8);
            float4 a = p[0], b = p[1];
            xr[j * 8 + 0] = a.x; xr[j * 8 + 1] = a.y; xr[j * 8 + 2] = a.z; xr[j * 8 + 3] = a.w;
            xr[j * 8 + 4] = b.x; xr[j * 8 + 5] = b.y; xr[j * 8 + 6] = b.z; xr[j * 8 + 7] = b.w;
        }
        Fast<Q>::template dotm<M>(r, c * 8, xr, 8, acc);
    }
#pragma unroll
    for (unsigned int j = 0; j < M; j++) {
        float a = acc[j];
        for (int off = 16; off > 0; off >>= 1) a += __shfl_down_sync(0xffffffffu, a, off);
        if (lane == 0) y[(u64) j * ys + i] = a;
    }
}

template <typename Q>
static __device__ __forceinline__ void matvec_ids_cols(const u8* w, const unsigned int* ids, const float* x, unsigned int xs,
                                                       float* y, unsigned int ys, unsigned int n, unsigned int cols, unsigned int m) {
    switch (m) {
        case 1: matvec_ids_cols_m<Q, 1>(w, ids, x, xs, y, ys, n, cols); break;
        case 2: matvec_ids_cols_m<Q, 2>(w, ids, x, xs, y, ys, n, cols); break;
        case 3: matvec_ids_cols_m<Q, 3>(w, ids, x, xs, y, ys, n, cols); break;
        case 4: matvec_ids_cols_m<Q, 4>(w, ids, x, xs, y, ys, n, cols); break;
        case 5: matvec_ids_cols_m<Q, 5>(w, ids, x, xs, y, ys, n, cols); break;
        case 6: matvec_ids_cols_m<Q, 6>(w, ids, x, xs, y, ys, n, cols); break;
        case 7: matvec_ids_cols_m<Q, 7>(w, ids, x, xs, y, ys, n, cols); break;
        default: matvec_ids_cols_m<Q, 8>(w, ids, x, xs, y, ys, n, cols); break;
    }
}
extern "C" __global__ void matvec_ids_cols_q6_k(const u8* w, const unsigned int* ids, const float* x, unsigned int xs, float* y, unsigned int ys, unsigned int n, unsigned int cols, unsigned int m) { matvec_ids_cols<Q6K>(w, ids, x, xs, y, ys, n, cols, m); }
extern "C" __global__ void matvec_ids_cols_q4_k(const u8* w, const unsigned int* ids, const float* x, unsigned int xs, float* y, unsigned int ys, unsigned int n, unsigned int cols, unsigned int m) { matvec_ids_cols<Q4K>(w, ids, x, xs, y, ys, n, cols, m); }
extern "C" __global__ void matvec_ids_cols_iq4_xs(const u8* w, const unsigned int* ids, const float* x, unsigned int xs, float* y, unsigned int ys, unsigned int n, unsigned int cols, unsigned int m) { matvec_ids_cols<IQ4XS>(w, ids, x, xs, y, ys, n, cols, m); }

// ----------------------------------------------------------------------------- several matrices, one launch
// The projections of one input (a DeltaNet layer's qkv / gate / beta / alpha; the router, its shared-expert gate and
// the shared expert's gate and up) as jobs in one grid: job j owns blocks [first_j, first_{j+1}).  Each runs
// matvec_cols_rows on its own matrix and output, so every output is the one-matrix kernel's bit for bit.
struct ColsJob {
    u64 w;
    u64 y;
    unsigned int ys, rows, type, first;
};

extern "C" __global__ void matvec_cols_multi(const ColsJob* jobs, unsigned int njobs, const float* x, unsigned int xs,
                                             unsigned int cols, unsigned int m, unsigned int j0) {
    unsigned int j = 0;
    while (j + 1 < njobs && blockIdx.x >= jobs[j + 1].first) j++;
    const ColsJob jb = jobs[j];
    const unsigned int bx = blockIdx.x - jb.first;
    const u8* w = (const u8*) jb.w;
    float* y = (float*) jb.y + (u64) j0 * jb.ys;
    switch (jb.type) {
        case 0: matvec_cols_rows_b<Q4K>(w, x, xs, y, jb.ys, jb.rows, cols, m, bx); break;
        case 1: matvec_cols_rows_b<Q5K>(w, x, xs, y, jb.ys, jb.rows, cols, m, bx); break;
        case 2: matvec_cols_rows_b<Q6K>(w, x, xs, y, jb.ys, jb.rows, cols, m, bx); break;
        case 3: matvec_cols_rows_b<IQ4XS>(w, x, xs, y, jb.ys, jb.rows, cols, m, bx); break;
        case 4: matvec_cols_rows_b<BF16>(w, x, xs, y, jb.ys, jb.rows, cols, m, bx); break;
        case 5: matvec_cols_rows_b<F32>(w, x, xs, y, jb.ys, jb.rows, cols, m, bx); break;
        default: matvec_cols_rows_b<IQ4NL>(w, x, xs, y, jb.ys, jb.rows, cols, m, bx); break;
    }
}

static __device__ __forceinline__ float sigmoidf_ref(float x) { return 1.0f / (1.0f + expf(-x)); }  // as forward.cu's

// ------------------------------------------------------------------------------------ hc up-projection + mix
// u[j][i] = (sum over streams s of sigmoid(W_up[s h + i] . down[j]) * rn[j][s h + i]) / streams, in one launch for
// m <= 8 inputs: the up-projection never exists as a vector.  Each of the 4 dots is dotm over the 320 columns (the
// sums of matvec_cols_bf16), the streams are added in order, as hc_mix does.  One output a warp; grid = ceil(h / 8).
template <unsigned int M>
static __device__ __forceinline__ void hc_up_mix_m(const u8* w_up, const float* down, unsigned int ds, const float* rn,
                                                   unsigned int rs, float* u, unsigned int us, unsigned int h,
                                                   unsigned int streams, unsigned int rank, unsigned int i,
                                                   unsigned int lane) {
    float mix[M];
#pragma unroll
    for (unsigned int j = 0; j < M; j++) mix[j] = 0.0f;
    for (unsigned int s = 0; s < streams; ++s) {
        const u8* r = w_up + ((u64) s * h + i) * rank * BF16::BYTES;
        float acc[M];
#pragma unroll
        for (unsigned int j = 0; j < M; j++) acc[j] = 0.0f;
        for (unsigned int c = lane; c < rank / 8; c += 32) {
            float xr[M * 8];
#pragma unroll
            for (unsigned int j = 0; j < M; j++) {
                const float4* p = (const float4*) (down + (u64) j * ds + c * 8);
                float4 a = p[0], b = p[1];
                xr[j * 8 + 0] = a.x; xr[j * 8 + 1] = a.y; xr[j * 8 + 2] = a.z; xr[j * 8 + 3] = a.w;
                xr[j * 8 + 4] = b.x; xr[j * 8 + 5] = b.y; xr[j * 8 + 6] = b.z; xr[j * 8 + 7] = b.w;
            }
            Fast<BF16>::template dotm<M>(r, c * 8, xr, 8, acc);
        }
#pragma unroll
        for (unsigned int j = 0; j < M; j++) {
            float a = acc[j];
            for (int off = 16; off > 0; off >>= 1) a += __shfl_down_sync(0xffffffffu, a, off);
            if (lane == 0) mix[j] += sigmoidf_ref(a) * rn[(u64) j * rs + (u64) s * h + i];
        }
    }
    if (lane == 0)
#pragma unroll
        for (unsigned int j = 0; j < M; j++) u[(u64) j * us + i] = mix[j] / (float) streams;
}

extern "C" __global__ void hc_up_mix_cols(const u8* w_up, const float* down, unsigned int ds, const float* rn,
                                          unsigned int rs, float* u, unsigned int us, unsigned int h,
                                          unsigned int streams, unsigned int rank, unsigned int m) {
    const unsigned int i = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32, lane = threadIdx.x % 32;
    if (i >= h) return;
    switch (m) {
        case 1: hc_up_mix_m<1>(w_up, down, ds, rn, rs, u, us, h, streams, rank, i, lane); break;
        case 2: hc_up_mix_m<2>(w_up, down, ds, rn, rs, u, us, h, streams, rank, i, lane); break;
        case 3: hc_up_mix_m<3>(w_up, down, ds, rn, rs, u, us, h, streams, rank, i, lane); break;
        case 4: hc_up_mix_m<4>(w_up, down, ds, rn, rs, u, us, h, streams, rank, i, lane); break;
        case 5: hc_up_mix_m<5>(w_up, down, ds, rn, rs, u, us, h, streams, rank, i, lane); break;
        case 6: hc_up_mix_m<6>(w_up, down, ds, rn, rs, u, us, h, streams, rank, i, lane); break;
        case 7: hc_up_mix_m<7>(w_up, down, ds, rn, rs, u, us, h, streams, rank, i, lane); break;
        default: hc_up_mix_m<8>(w_up, down, ds, rn, rs, u, us, h, streams, rank, i, lane); break;
    }
}
