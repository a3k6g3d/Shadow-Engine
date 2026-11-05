// Everything in a forward pass that is not a matrix-vector product (matvec.cu does those): norms, the
// hyper-connection glue, the DeltaNet step, attention with its sparse indexer, the MoE glue and the n-gram memory.
// Each kernel follows the CPU reference (crates/shadow-model/src/forward.rs) step for step, including where the
// reference widens to double (norm sums, dot products, exp/log), so the GPU session can be held to the reference
// fixtures stream by stream.  Kernels take no doubles as arguments (the launcher has none); positions are u32 and the
// RoPE base is passed as a float (1e7 is exact in float).

// float transcendentals (expf, log1pf): FP64 is 1/64 rate on this GPU, and expf is good to ~1e-7 relative
static __device__ __forceinline__ float sigmoidf_ref(float x) { return 1.0f / (1.0f + expf(-x)); }
static __device__ __forceinline__ float siluf_ref(float x) { return x * sigmoidf_ref(x); }
static __device__ __forceinline__ float softplusf_ref(float x) { return x > 20.0f ? x : log1pf(expf(x)); }

// block-wide sum of a double, blockDim.x <= 1024, a power of two
static __device__ double block_sum(double v) {
    __shared__ double buf[1024];
    buf[threadIdx.x] = v;
    __syncthreads();
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) buf[threadIdx.x] += buf[threadIdx.x + s];
        __syncthreads();
    }
    double r = buf[0];
    __syncthreads();
    return r;
}

static __device__ double block_max(double v) {
    __shared__ double buf[1024];
    buf[threadIdx.x] = v;
    __syncthreads();
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) buf[threadIdx.x] = fmax(buf[threadIdx.x], buf[threadIdx.x + s]);
        __syncthreads();
    }
    double r = buf[0];
    __syncthreads();
    return r;
}

// RMSNorm of one group of `n` values in place in shared-free style: y[i] = (float)(x[i] * r) * w[i]
static __device__ void rmsnorm_one(const float* x, const float* w, float* y, unsigned int n, float eps) {
    double ss = 0.0;
    for (unsigned int i = threadIdx.x; i < n; i += blockDim.x) ss += (double) x[i] * (double) x[i];
    double ms = block_sum(ss) / (double) n;
    double r = 1.0 / sqrt(ms + (double) eps);
    for (unsigned int i = threadIdx.x; i < n; i += blockDim.x) y[i] = (float) ((double) x[i] * r) * w[i];
}

// NeoX rotary embedding of x[0 .. dims) at `pos`, one thread per pair, in place
// Picture rows have a 2-D position (t, h, w) instead of one index.  `g_rope_tab` (set by the host, null when no picture is in the conversation)
// maps a token index to its three positions, [3 * index + {t, h, w}]; pair i of the rotation takes t, h or w as i % 3 is 0, 1 or 2 (the
// interleaved sections 11, 11, 10 of the model's 32 pairs: llama.cpp's IMROPE).  With a null table this is the plain one-position rotation.
extern "C" __device__ const int* g_rope_tab = nullptr;

static __device__ void rope_one(float* x, unsigned int pos, unsigned int dims, float theta) {
    unsigned int half = dims / 2;
    unsigned int pp[3] = {pos, pos, pos};
    const int* tab = g_rope_tab;
    if (tab != nullptr) {
        pp[0] = (unsigned int) tab[3 * (u64) pos];
        pp[1] = (unsigned int) tab[3 * (u64) pos + 1];
        pp[2] = (unsigned int) tab[3 * (u64) pos + 2];
    }
    for (unsigned int i = threadIdx.x; i < half; i += blockDim.x) {
        double f = (double) pp[i % 3] / pow((double) theta, 2.0 * (double) i / (double) dims);
        double cs = cos(f), sn = sin(f);
        double a = x[i], b = x[i + half];
        x[i] = (float) (a * cs - b * sn);
        x[i + half] = (float) (b * cs + a * sn);
    }
}

// ------------------------------------------------------------------------------------------------ norms and glue
// grid = number of groups, block = 256
extern "C" __global__ void rmsnorm_groups(const float* x, const float* w, float* y, unsigned int group, float eps) {
    unsigned int g = blockIdx.x;
    rmsnorm_one(x + (u64) g * group, w + (u64) g * group, y + (u64) g * group, group, eps);
}

// `rmsnorm_groups` for many rows: grid (groups, rows), a row `groups * group` floats (the same arithmetic as one launch a row).
extern "C" __global__ void rmsnorm_groups_rows(const float* x, const float* w, float* y, unsigned int group, float eps, unsigned int groups) {
    const unsigned int g = blockIdx.x;
    const u64 off = (u64) blockIdx.y * groups * group + (u64) g * group;
    rmsnorm_one(x + off, w + (u64) g * group, y + off, group, eps);
}

// r[i] += y[i], then u = rmsnorm(r, w, eps) in one block of 256 threads
extern "C" __global__ void add_rmsnorm1(float* r, const float* y, const float* w, float* u, unsigned int n, float eps) {
    double ss = 0.0;
    for (unsigned int i = threadIdx.x; i < n; i += blockDim.x) {
        float v = r[i] + y[i];
        r[i] = v;
        ss += (double) v * (double) v;
    }
    double ms = block_sum(ss) / (double) n;
    double scale = 1.0 / sqrt(ms + (double) eps);
    for (unsigned int i = threadIdx.x; i < n; i += blockDim.x) u[i] = (float) ((double) r[i] * scale) * w[i];
}

// x = silu(x / div)
extern "C" __global__ void silu_div(float* x, unsigned int n, float div) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = siluf_ref(x[i] / div);
}

// y = mul * sigmoid(x / div)
extern "C" __global__ void sigmoid_div(const float* x, float* y, unsigned int n, float div, float mul) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = mul * sigmoidf_ref(x[i] / div);
}

// u[i] = (sum over streams of sigmoid(up[s*h+i]) * rn[s*h+i]) / streams, summed in stream order as the reference does
static __device__ __forceinline__ void hc_mix_one(const float* rn, const float* up, float* u, unsigned int h,
                                                  unsigned int streams) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= h) return;
    float acc = 0.0f;
    for (unsigned int s = 0; s < streams; ++s) acc += sigmoidf_ref(up[s * h + i]) * rn[s * h + i];
    u[i] = acc / (float) streams;
}

extern "C" __global__ void hc_mix(const float* rn, const float* up, float* u, unsigned int h, unsigned int streams) {
    hc_mix_one(rn, up, u, h, streams);
}

// hc_mix for rows y = blockIdx.y: rn and up rows h * streams apart, u rows h apart.  grid = (ceil(h / 256), rows).
extern "C" __global__ void hc_mix_rows(const float* rn, const float* up, float* u, unsigned int h, unsigned int streams) {
    const u64 y = blockIdx.y, rs = (u64) h * streams;
    hc_mix_one(rn + y * rs, up + y * rs, u + y * h, h, streams);
}

// r[s*h+i] += g[s] * y[i]
extern "C" __global__ void hc_write(float* r, const float* g, const float* y, unsigned int h, unsigned int streams) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= h * streams) return;
    r[i] += g[i / h] * y[i % h];
}

// Fuses hc_write (r[s*h+i] += g[s] * y[i]) and the next hc_read's rmsnorm_groups (rn[s*h+i] = rmsnorm(r_s, w_s)).
// grid = streams, block = 256.
static __device__ __forceinline__ void hc_write_rmsnorm_one(float* r, const float* g, const float* y, const float* w,
                                                            float* rn, unsigned int group, float eps, unsigned int s) {
    float gs = g[s];
    float* rs = r + (u64) s * group;
    const float* ws = w + (u64) s * group;
    float* ys = rn + (u64) s * group;
    double ss = 0.0;
    for (unsigned int i = threadIdx.x; i < group; i += blockDim.x) {
        float v = rs[i] + gs * y[i];
        rs[i] = v;
        ss += (double) v * (double) v;
    }
    double ms = block_sum(ss) / (double) group;
    double scale = 1.0 / sqrt(ms + (double) eps);
    for (unsigned int i = threadIdx.x; i < group; i += blockDim.x) {
        ys[i] = (float) ((double) rs[i] * scale) * ws[i];
    }
}

extern "C" __global__ void hc_write_rmsnorm_groups(float* r, const float* g, const float* y, const float* w, float* rn,
                                                   unsigned int group, float eps) {
    hc_write_rmsnorm_one(r, g, y, w, rn, group, eps, blockIdx.x);
}

// For rows j = blockIdx.y (r and rn rows streams * group apart, g rows streams, y rows group); grid = (streams, rows).
extern "C" __global__ void hc_write_rmsnorm_rows(float* r, const float* g, const float* y, const float* w, float* rn,
                                                 unsigned int group, float eps, unsigned int streams) {
    const u64 j = blockIdx.y, rs = (u64) streams * group;
    hc_write_rmsnorm_one(r + j * rs, g + j * streams, y + j * group, w, rn + j * rs, group, eps, blockIdx.x);
}

// r[s*h+i] = x[i]
extern "C" __global__ void broadcast(const float* x, float* r, unsigned int h, unsigned int streams) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < h * streams) r[i] = x[i % h];
}

// a += b
extern "C" __global__ void add(float* a, const float* b, unsigned int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) a[i] += b[i];
}

extern "C" __global__ void copy(float* dst, const float* src, unsigned int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = src[i];
}

// 32-bit words, bit for bit (tables of addresses and weights, staged in mapped host memory)
extern "C" __global__ void copy_words(unsigned int* dst, const unsigned int* src, unsigned int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = src[i];
}

// x[i] *= s
extern "C" __global__ void scale_vec(float* x, unsigned int n, float s) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] *= s;
}

// ------------------------------------------------------------------------------------------------ DeltaNet
// One step of the depthwise causal conv (kernel k) with its state of the k-1 previous inputs (oldest first), then SiLU;
// the state shifts by one and takes x.  One thread per channel.
extern "C" __global__ void conv_silu_step(const float* x, float* state, const float* w, float* out, unsigned int ch,
                                          unsigned int k) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= ch) return;
    double acc = (double) w[i * k + (k - 1)] * (double) x[i];
    for (unsigned int t = 0; t + 1 < k; ++t) acc += (double) w[i * k + t] * (double) state[t * ch + i];
    out[i] = siluf_ref((float) acc);
    for (unsigned int t = 0; t + 2 < k; ++t) state[t * ch + i] = state[(t + 1) * ch + i];
    state[(k - 2) * ch + i] = x[i];
}

// conv_silu_step for m consecutive tokens (x and out rows ch apart): each channel walks the tokens in order, so the
// window and the outputs are conv_silu_step's on each token in turn.
extern "C" __global__ void conv_silu_steps(const float* x, float* state, const float* w, float* out, unsigned int ch,
                                           unsigned int k, unsigned int m) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= ch) return;
    for (unsigned int j = 0; j < m; ++j) {
        const float xi = x[(u64) j * ch + i];
        double acc = (double) w[i * k + (k - 1)] * (double) xi;
        for (unsigned int t = 0; t + 1 < k; ++t) acc += (double) w[i * k + t] * (double) state[t * ch + i];
        out[(u64) j * ch + i] = siluf_ref((float) acc);
        for (unsigned int t = 0; t + 2 < k; ++t) state[t * ch + i] = state[(t + 1) * ch + i];
        state[(k - 2) * ch + i] = xi;
    }
}

// Each `d`-wide head: x = x / sqrt(sum x^2 + 1e-6), then x /= div.  grid = heads, block = 128.
extern "C" __global__ void l2norm_heads(float* x, unsigned int d, float div) {
    float* h = x + (u64) blockIdx.x * d;
    double ss = 0.0;
    for (unsigned int i = threadIdx.x; i < d; i += blockDim.x) ss += (double) h[i] * (double) h[i];
    double n = 1.0 / sqrt(block_sum(ss) + 1e-6);
    for (unsigned int i = threadIdx.x; i < d; i += blockDim.x) {
        float v = (float) ((double) h[i] * n);
        h[i] = div == 1.0f ? v : v / div;
    }
}

// Blocks 0..kh-1 normalize Q heads (divided by q_div); blocks kh..2*kh-1 normalize K heads (div = 1.0).
extern "C" __global__ void l2norm_qk_rows(float* x, unsigned int kh, unsigned int d, float q_div, unsigned int xs);

extern "C" __global__ void l2norm_qk_heads(float* x, unsigned int kh, unsigned int d, float q_div) {
    float* h = x + (u64) blockIdx.x * d;
    float div = blockIdx.x < kh ? q_div : 1.0f;
    double ss = 0.0;
    for (unsigned int i = threadIdx.x; i < d; i += blockDim.x) ss += (double) h[i] * (double) h[i];
    double n = 1.0 / sqrt(block_sum(ss) + 1e-6);
    for (unsigned int i = threadIdx.x; i < d; i += blockDim.x) {
        float v = (float) ((double) h[i] * n);
        h[i] = div == 1.0f ? v : v / div;
    }
}

// l2norm_qk_heads on rows y = blockIdx.y, xs apart; grid = (2 kh, rows).
extern "C" __global__ void l2norm_qk_rows(float* x, unsigned int kh, unsigned int d, float q_div, unsigned int xs) {
    float* h = x + (u64) blockIdx.y * xs + (u64) blockIdx.x * d;
    float div = blockIdx.x < kh ? q_div : 1.0f;
    double ss = 0.0;
    for (unsigned int i = threadIdx.x; i < d; i += blockDim.x) ss += (double) h[i] * (double) h[i];
    double n = 1.0 / sqrt(block_sum(ss) + 1e-6);
    for (unsigned int i = threadIdx.x; i < d; i += blockDim.x) {
        float v = (float) ((double) h[i] * n);
        h[i] = div == 1.0f ? v : v / div;
    }
}

// The gated delta rule for one token, one value head per block, one thread per value index j (blockDim = d):
// S *= decay; d_j = (v_j - S^T k)_j * beta; S += k d^T; o_j = (S^T q)_j.  S is [d][d], key index major.  Value head h
// pairs with key head h % kh (the GGUF tiles value heads).
extern "C" __global__ void delta_step(const float* q, const float* k, const float* v, const float* b, const float* a,
                                      const float* A, const float* dtb, float* S, float* o, unsigned int kh,
                                      unsigned int d) {
    __shared__ float qs[256], ks[256];
    unsigned int h = blockIdx.x, j = threadIdx.x, kx = h % kh;
    for (unsigned int i = threadIdx.x; i < d; i += blockDim.x) {
        qs[i] = q[kx * d + i];
        ks[i] = k[kx * d + i];
    }
    __syncthreads();
    if (j >= d) return;
    float beta = sigmoidf_ref(b[h]);
    float decay = expf(A[h] * softplusf_ref(a[h] + dtb[h]));
    float* s = S + (u64) h * d * d;
    // the 128-term sums: float FMAs in runs of 16 (FP64 runs at 1/64 of FP32 on this GPU), the runs added in double
    double mem = 0.0;
    for (unsigned int i0 = 0; i0 < d; i0 += 16) {
        float part = 0.0f;
        for (unsigned int i = i0; i < i0 + 16 && i < d; ++i) {
            s[i * d + j] *= decay;
            part = fmaf(s[i * d + j], ks[i], part);
        }
        mem += (double) part;
    }
    float delta = (float) (((double) v[h * d + j] - mem) * (double) beta);
    double acc = 0.0;
    for (unsigned int i0 = 0; i0 < d; i0 += 16) {
        float part = 0.0f;
        for (unsigned int i = i0; i < i0 + 16 && i < d; ++i) {
            s[i * d + j] += ks[i] * delta;
            part = fmaf(s[i * d + j], qs[i], part);
        }
        acc += (double) part;
    }
    o[h * d + j] = (float) acc;
}

// delta_step for m consecutive tokens of one sequence, d = 128: each thread keeps its column of S in registers
// between the tokens, so S is read and written once instead of m times.  Token t's q, k, v start at q / k / v +
// t * xs, its b and a at + t * bs, its output at o + t * os.  The same operations in the same order as delta_step on
// each token in turn, so the same bits.  grid = value heads, block = 128.
extern "C" __global__ void delta_steps128(const float* q, const float* k, const float* v, unsigned int xs, const float* b,
                                          const float* a, unsigned int bs, const float* A, const float* dtb, float* S,
                                          float* o, unsigned int os, unsigned int kh, unsigned int m) {
    const unsigned int d = 128;
    __shared__ float qs[128], ks[128];
    unsigned int h = blockIdx.x, j = threadIdx.x, kx = h % kh;
    float* s = S + (u64) h * d * d;
    float col[128];
#pragma unroll
    for (unsigned int i = 0; i < d; ++i) col[i] = s[i * d + j];
    for (unsigned int t = 0; t < m; ++t) {
        __syncthreads();
        qs[j] = q[(u64) t * xs + kx * d + j];
        ks[j] = k[(u64) t * xs + kx * d + j];
        __syncthreads();
        float beta = sigmoidf_ref(b[(u64) t * bs + h]);
        float decay = expf(A[h] * softplusf_ref(a[(u64) t * bs + h] + dtb[h]));
        double mem = 0.0;
#pragma unroll
        for (unsigned int i0 = 0; i0 < d; i0 += 16) {
            float part = 0.0f;
#pragma unroll
            for (unsigned int i = i0; i < i0 + 16; ++i) {
                col[i] *= decay;
                part = fmaf(col[i], ks[i], part);
            }
            mem += (double) part;
        }
        float delta = (float) (((double) v[(u64) t * xs + h * d + j] - mem) * (double) beta);
        double acc = 0.0;
#pragma unroll
        for (unsigned int i0 = 0; i0 < d; i0 += 16) {
            float part = 0.0f;
#pragma unroll
            for (unsigned int i = i0; i < i0 + 16; ++i) {
                col[i] += ks[i] * delta;
                part = fmaf(col[i], qs[i], part);
            }
            acc += (double) part;
        }
        o[(u64) t * os + h * d + j] = (float) acc;
    }
#pragma unroll
    for (unsigned int i = 0; i < d; ++i) s[i * d + j] = col[i];
}

// delta_steps128 with 8 threads a column instead of 1: thread (c, j) of the 1024-thread block keeps rows 16c..16c+15 of column j of S (16 registers, not 128), so
// the 16-term fma chains of one token run in parallel and the 8 partial sums are added in the same order (c = 0..7, in double) by every thread of the column:
// the same operations on the same values in the same order, so the same bits as delta_steps128.  grid = value heads, block = 1024.
extern "C" __global__ void delta_steps128_x8(const float* q, const float* k, const float* v, unsigned int xs, const float* b,
                                             const float* a, unsigned int bs, const float* A, const float* dtb, float* S,
                                             float* o, unsigned int os, unsigned int kh, unsigned int m) {
    const unsigned int d = 128;
    __shared__ float qs[128], ks[128], pm[8][128], pa[8][128];
    const unsigned int h = blockIdx.x, c = threadIdx.x / 128, j = threadIdx.x % 128, kx = h % kh;
    float* s = S + (u64) h * d * d;
    float col[16];
#pragma unroll
    for (unsigned int r = 0; r < 16; ++r) col[r] = s[(16 * c + r) * d + j];
    for (unsigned int t = 0; t < m; ++t) {
        __syncthreads();
        if (c == 0) {
            qs[j] = q[(u64) t * xs + kx * d + j];
            ks[j] = k[(u64) t * xs + kx * d + j];
        }
        __syncthreads();
        const float beta = sigmoidf_ref(b[(u64) t * bs + h]);
        const float decay = expf(A[h] * softplusf_ref(a[(u64) t * bs + h] + dtb[h]));
        float part = 0.0f;
#pragma unroll
        for (unsigned int r = 0; r < 16; ++r) {
            col[r] *= decay;
            part = fmaf(col[r], ks[16 * c + r], part);
        }
        pm[c][j] = part;
        __syncthreads();
        double mem = 0.0;
#pragma unroll
        for (unsigned int cc = 0; cc < 8; ++cc) mem += (double) pm[cc][j];
        const float delta = (float) (((double) v[(u64) t * xs + h * d + j] - mem) * (double) beta);
        part = 0.0f;
#pragma unroll
        for (unsigned int r = 0; r < 16; ++r) {
            col[r] += ks[16 * c + r] * delta;
            part = fmaf(col[r], qs[16 * c + r], part);
        }
        pa[c][j] = part;
        __syncthreads();
        if (c == 0) {
            double acc = 0.0;
#pragma unroll
            for (unsigned int cc = 0; cc < 8; ++cc) acc += (double) pa[cc][j];
            o[(u64) t * os + h * d + j] = (float) acc;
        }
    }
#pragma unroll
    for (unsigned int r = 0; r < 16; ++r) s[(16 * c + r) * d + j] = col[r];
}

// ---- the DeltaNet recurrence of a long prompt as a chunked scan (gated delta rule, WY form), 64 tokens a chunk.  Not bit-exact with delta_steps128 (other rounding), the same maths:
//   S_t = g_t S_{t-1} + k_t u_t^T,  u_t = beta_t (v_t - g_t S_{t-1}^T k_t),  o_t = S_t^T q_t,  gamma_t = prod of the chunk's g up to t, ratio(t,s) = gamma_t / gamma_s
//   (I + A) U = diag(beta) (V - diag(gamma) K S_0),  A[t][s] = beta_t ratio(t,s) (k_s . k_t) for s < t;  T = (I + A)^-1 diag(beta)
//   O = diag(gamma) Q S_0 + Mo U,  Mo[t][s] = ratio(t,s) (k_s . q_t) for s <= t;  S_64 = gamma_64 S_0 + K^T diag(gamma_64 / gamma_s) U
// delta_chunk_prep builds T, Mo, gamma, gamma_64/gamma_s for every (chunk, value head) at once (nothing in it depends on S); delta_chunk_run walks the chunks of a head in order,
// 16 columns of S a block.  Tokens past m in the last chunk have beta = 0 and g = 1 (they change nothing).  Scratch per (chunk, head): DSCR floats.
#define DSCR 8448
extern "C" __global__ void delta_chunk_prep(const float* q, const float* k, const float* b, const float* a, const float* A, const float* dtb, float* scr,
                                            unsigned int xs, unsigned int bs, unsigned int kh, unsigned int vh, unsigned int m) {
    __shared__ float Ls[64], lgs[64], bt[64];
    __shared__ float Am[64][65], T2[64][65];
    const unsigned int n = blockIdx.x, h = blockIdx.y, kx = h % kh, tid = threadIdx.x, t0 = n * 64;
    float* out = scr + ((u64) n * vh + h) * DSCR;
    if (tid < 64) {
        bool ok = t0 + tid < m;
        bt[tid] = ok ? sigmoidf_ref(b[(u64) (t0 + tid) * bs + h]) : 0.0f;
        lgs[tid] = ok ? A[h] * softplusf_ref(a[(u64) (t0 + tid) * bs + h] + dtb[h]) : 0.0f;
    }
    __syncthreads();
    if (tid == 0) {
        float acc = 0.0f;
        for (unsigned int t = 0; t < 64; ++t) { acc += lgs[t]; Ls[t] = acc; }
    }
    __syncthreads();
    {
        const unsigned int t = tid / 4, s0 = (tid % 4) * 16;
        const unsigned int tt = min(t0 + t, m - 1);
        const float4* kt = (const float4*) (k + (u64) tt * xs + kx * 128);
        const float4* qt = (const float4*) (q + (u64) tt * xs + kx * 128);
        float kk[16], qk[16];
#pragma unroll
        for (unsigned int e = 0; e < 16; ++e) { kk[e] = 0.0f; qk[e] = 0.0f; }
        for (unsigned int i4 = 0; i4 < 32; ++i4) {
            float4 kv = kt[i4], qv = qt[i4];
#pragma unroll
            for (unsigned int e = 0; e < 16; ++e) {
                const unsigned int ts = min(t0 + s0 + e, m - 1);
                float4 sv = ((const float4*) (k + (u64) ts * xs + kx * 128))[i4];
                kk[e] = fmaf(kv.x, sv.x, fmaf(kv.y, sv.y, fmaf(kv.z, sv.z, fmaf(kv.w, sv.w, kk[e]))));
                qk[e] = fmaf(qv.x, sv.x, fmaf(qv.y, sv.y, fmaf(qv.z, sv.z, fmaf(qv.w, sv.w, qk[e]))));
            }
        }
#pragma unroll
        for (unsigned int e = 0; e < 16; ++e) {
            const unsigned int s = s0 + e;
            const float ratio = s <= t ? expf(Ls[t] - Ls[s]) : 0.0f;
            Am[t][s] = s < t ? bt[t] * ratio * kk[e] : 0.0f;
            out[4096 + t * 64 + s] = s <= t ? ratio * qk[e] : 0.0f;
        }
    }
    __syncthreads();
    if (tid < 64) {
        const unsigned int c = tid;
        for (unsigned int t = 0; t < 64; ++t) {
            float x = 0.0f;
            if (t >= c) {
                x = t == c ? bt[t] : 0.0f;
                for (unsigned int s = c; s < t; ++s) x -= Am[t][s] * T2[s][c];
            }
            T2[t][c] = x;
        }
    }
    __syncthreads();
    for (unsigned int e = tid; e < 4096; e += 256) out[e] = T2[e / 64][e % 64];
    if (tid < 64) {
        out[8192 + tid] = expf(Ls[tid]);
        out[8256 + tid] = expf(Ls[63] - Ls[tid]);
    }
    if (tid == 0) out[8320] = expf(Ls[63]);
}

extern "C" __global__ void delta_chunk_run(const float* q, const float* k, const float* v, const float* scr, float* S, float* o,
                                           unsigned int xs, unsigned int os, unsigned int kh, unsigned int vh, unsigned int m) {
    __shared__ __align__(16) float Ss[128][16];
    __shared__ __align__(16) float Rs[64][16], Us[64][16];
    const unsigned int h = blockIdx.x, j0 = blockIdx.y * 16, kx = h % kh, tid = threadIdx.x;
    const unsigned int t = tid / 4, g = tid % 4;
    float* s = S + (u64) h * 128 * 128;
    for (unsigned int e = tid; e < 128 * 16; e += 256) Ss[e / 16][e % 16] = s[(e / 16) * 128 + j0 + e % 16];
    const unsigned int nch = (m + 63) / 64;
    for (unsigned int n = 0; n < nch; ++n) {
        const float* sc = scr + ((u64) n * vh + h) * DSCR;
        const unsigned int t0 = n * 64, tt = min(t0 + t, m - 1);
        const float4* kt = (const float4*) (k + (u64) tt * xs + kx * 128);
        const float4* qt = (const float4*) (q + (u64) tt * xs + kx * 128);
        __syncthreads();
        float acc[4] = {0.f, 0.f, 0.f, 0.f}, accq[4] = {0.f, 0.f, 0.f, 0.f};
        for (unsigned int i4 = 0; i4 < 32; ++i4) {
            float4 kv = kt[i4], qv = qt[i4];
            float ka[4] = {kv.x, kv.y, kv.z, kv.w}, qa[4] = {qv.x, qv.y, qv.z, qv.w};
#pragma unroll
            for (unsigned int ii = 0; ii < 4; ++ii) {
                float4 sv = *(const float4*) &Ss[4 * i4 + ii][4 * g];
                acc[0] = fmaf(ka[ii], sv.x, acc[0]); acc[1] = fmaf(ka[ii], sv.y, acc[1]); acc[2] = fmaf(ka[ii], sv.z, acc[2]); acc[3] = fmaf(ka[ii], sv.w, acc[3]);
                accq[0] = fmaf(qa[ii], sv.x, accq[0]); accq[1] = fmaf(qa[ii], sv.y, accq[1]); accq[2] = fmaf(qa[ii], sv.z, accq[2]); accq[3] = fmaf(qa[ii], sv.w, accq[3]);
            }
        }
        const float gam = sc[8192 + t];
        float4 vv = *(const float4*) (v + (u64) tt * xs + h * 128 + j0 + 4 * g);
        *(float4*) &Rs[t][4 * g] = make_float4(vv.x - gam * acc[0], vv.y - gam * acc[1], vv.z - gam * acc[2], vv.w - gam * acc[3]);
        __syncthreads();
        float u[4] = {0.f, 0.f, 0.f, 0.f};
        for (unsigned int s2 = 0; s2 <= t; ++s2) {
            const float w = sc[t * 64 + s2];
            float4 r = *(const float4*) &Rs[s2][4 * g];
            u[0] = fmaf(w, r.x, u[0]); u[1] = fmaf(w, r.y, u[1]); u[2] = fmaf(w, r.z, u[2]); u[3] = fmaf(w, r.w, u[3]);
        }
        *(float4*) &Us[t][4 * g] = make_float4(u[0], u[1], u[2], u[3]);
        __syncthreads();
        float oo[4] = {gam * accq[0], gam * accq[1], gam * accq[2], gam * accq[3]};
        for (unsigned int s2 = 0; s2 <= t; ++s2) {
            const float w = sc[4096 + t * 64 + s2];
            float4 r = *(const float4*) &Us[s2][4 * g];
            oo[0] = fmaf(w, r.x, oo[0]); oo[1] = fmaf(w, r.y, oo[1]); oo[2] = fmaf(w, r.z, oo[2]); oo[3] = fmaf(w, r.w, oo[3]);
        }
        if (t0 + t < m) *(float4*) (o + (u64) (t0 + t) * os + h * 128 + j0 + 4 * g) = make_float4(oo[0], oo[1], oo[2], oo[3]);
        {
            const unsigned int i = tid / 2, cb = (tid % 2) * 8;
            const float gc = sc[8320];
            float a8[8];
#pragma unroll
            for (unsigned int c = 0; c < 8; ++c) a8[c] = gc * Ss[i][cb + c];
            for (unsigned int s2 = 0; s2 < 64; ++s2) {
                const unsigned int ts = min(t0 + s2, m - 1);
                const float kw = k[(u64) ts * xs + kx * 128 + i] * sc[8256 + s2];
#pragma unroll
                for (unsigned int c = 0; c < 8; ++c) a8[c] = fmaf(kw, Us[s2][cb + c], a8[c]);
            }
#pragma unroll
            for (unsigned int c = 0; c < 8; ++c) Ss[i][cb + c] = a8[c];
        }
    }
    __syncthreads();
    for (unsigned int e = tid; e < 128 * 16; e += 256) s[(e / 16) * 128 + j0 + e % 16] = Ss[e / 16][e % 16];
}

// delta_steps128 with gated_rmsnorm_heads folded in: after each token's recurrence the head's 128 outputs are
// normalized and gated in the block (the same operations in the same order as gated_rmsnorm_heads, whose block is
// these 128 threads too).  z rows os apart; o is written normalized.
extern "C" __global__ void delta_steps128_gn(const float* q, const float* k, const float* v, unsigned int xs, const float* b,
                                             const float* a, unsigned int bs, const float* A, const float* dtb, float* S,
                                             float* o, unsigned int os, unsigned int kh, unsigned int m, const float* z,
                                             const float* nw, float eps) {
    const unsigned int d = 128;
    __shared__ float qs[128], ks[128];
    unsigned int h = blockIdx.x, j = threadIdx.x, kx = h % kh;
    float* s = S + (u64) h * d * d;
    float col[128];
#pragma unroll
    for (unsigned int i = 0; i < d; ++i) col[i] = s[i * d + j];
    for (unsigned int t = 0; t < m; ++t) {
        __syncthreads();
        qs[j] = q[(u64) t * xs + kx * d + j];
        ks[j] = k[(u64) t * xs + kx * d + j];
        __syncthreads();
        float beta = sigmoidf_ref(b[(u64) t * bs + h]);
        float decay = expf(A[h] * softplusf_ref(a[(u64) t * bs + h] + dtb[h]));
        double mem = 0.0;
#pragma unroll
        for (unsigned int i0 = 0; i0 < d; i0 += 16) {
            float part = 0.0f;
#pragma unroll
            for (unsigned int i = i0; i < i0 + 16; ++i) {
                col[i] *= decay;
                part = fmaf(col[i], ks[i], part);
            }
            mem += (double) part;
        }
        float delta = (float) (((double) v[(u64) t * xs + h * d + j] - mem) * (double) beta);
        double acc = 0.0;
#pragma unroll
        for (unsigned int i0 = 0; i0 < d; i0 += 16) {
            float part = 0.0f;
#pragma unroll
            for (unsigned int i = i0; i < i0 + 16; ++i) {
                col[i] += ks[i] * delta;
                part = fmaf(col[i], qs[i], part);
            }
            acc += (double) part;
        }
        const float ov = (float) acc;
        double r = 1.0 / sqrt(block_sum((double) ov * (double) ov) / (double) d + (double) eps);
        o[(u64) t * os + h * d + j] = (float) ((double) ov * r) * nw[j] * sigmoidf_ref(z[(u64) t * os + h * d + j]);
    }
#pragma unroll
    for (unsigned int i = 0; i < d; ++i) s[i * d + j] = col[i];
}

// Each head: RMSNorm(o_h) * w * sigmoid(z_h).  grid = heads, block = 128.
extern "C" __global__ void gated_rmsnorm_heads(float* o, const float* w, const float* z, unsigned int d, float eps) {
    float* h = o + (u64) blockIdx.x * d;
    const float* zh = z + (u64) blockIdx.x * d;
    double ss = 0.0;
    for (unsigned int i = threadIdx.x; i < d; i += blockDim.x) ss += (double) h[i] * (double) h[i];
    double r = 1.0 / sqrt(block_sum(ss) / (double) d + (double) eps);
    for (unsigned int i = threadIdx.x; i < d; i += blockDim.x) h[i] = (float) ((double) h[i] * r) * w[i] * sigmoidf_ref(zh[i]);
}

// gated_rmsnorm_heads on rows y = blockIdx.y (o and z rows os apart); grid = (heads, rows).
extern "C" __global__ void gated_rmsnorm_rows(float* o, const float* w, const float* z, unsigned int d, float eps,
                                              unsigned int os) {
    float* h = o + (u64) blockIdx.y * os + (u64) blockIdx.x * d;
    const float* zh = z + (u64) blockIdx.y * os + (u64) blockIdx.x * d;
    double ss = 0.0;
    for (unsigned int i = threadIdx.x; i < d; i += blockDim.x) ss += (double) h[i] * (double) h[i];
    double r = 1.0 / sqrt(block_sum(ss) / (double) d + (double) eps);
    for (unsigned int i = threadIdx.x; i < d; i += blockDim.x) h[i] = (float) ((double) h[i] * r) * w[i] * sigmoidf_ref(zh[i]);
}

// ------------------------------------------------------------------------------------------------ attention
// Each head of `in` (head h starts at in + h*in_stride): RMSNorm with w, then RoPE on the first rope_dims at pos,
// written to out + h*out_stride.  grid = heads, block = 128.
extern "C" __global__ void head_norm_rope(const float* in, unsigned int in_stride, float* out, unsigned int out_stride,
                                          const float* w, unsigned int d, float eps, unsigned int pos,
                                          unsigned int rope_dims, float theta) {
    const float* x = in + (u64) blockIdx.x * in_stride;
    float* y = out + (u64) blockIdx.x * out_stride;
    rmsnorm_one(x, w, y, d, eps);
    __syncthreads();
    rope_one(y, pos, rope_dims, theta);
}

// `head_norm_rope` for tokens r = blockIdx.y at positions pos0 + r: input rows `in_rs` and output rows `out_rs` floats apart.
// grid = (heads, rows).
extern "C" __global__ void head_norm_rope_rows(const float* in, unsigned int in_stride, unsigned int in_rs, float* out,
                                               unsigned int out_stride, unsigned int out_rs, const float* w, unsigned int d,
                                               float eps, unsigned int pos0, unsigned int rope_dims, float theta) {
    const u64 r = blockIdx.y;
    const float* x = in + r * in_rs + (u64) blockIdx.x * in_stride;
    float* y = out + r * out_rs + (u64) blockIdx.x * out_stride;
    rmsnorm_one(x, w, y, d, eps);
    __syncthreads();
    rope_one(y, pos0 + (unsigned int) r, rope_dims, theta);
}

// Blocks 0..nh-1 normalize+RoPE query heads (qg -> q, with wq); blocks nh..nh+nkv-1 normalize+RoPE key heads (kk -> kc, with wk).
extern "C" __global__ void head_norm_rope_qk(const float* qg, float* q, const float* wq,
                                             const float* kk, float* kc, const float* wk,
                                             unsigned int nh, unsigned int d, float eps,
                                             unsigned int pos, unsigned int rope_dims, float theta) {
    unsigned int b = blockIdx.x;
    const float* x = b < nh ? (qg + (u64) b * d) : (kk + (u64) (b - nh) * d);
    float* y = b < nh ? (q + (u64) b * d) : (kc + (u64) (b - nh) * d);
    const float* w = b < nh ? wq : wk;
    rmsnorm_one(x, w, y, d, eps);
    __syncthreads();
    rope_one(y, pos, rope_dims, theta);
}

// Qwen gated attention: blocks 0..nh-1 normalize+RoPE query heads (qg[h*2*d..h*2*d+d] -> q[h*d..]) AND split the gate
// half (qg[h*2*d+d..h*2*d+2*d] -> gate[h*d..]); blocks nh..nh+nkv-1 normalize+RoPE key heads (kk -> kc).
static __device__ __forceinline__ void head_norm_rope_qgk_one(const float* qg, float* q, float* gate, const float* wq,
                                                              const float* kk, float* kc, const float* wk,
                                                              unsigned int nh, unsigned int d, float eps,
                                                              unsigned int pos, unsigned int rope_dims, float theta,
                                                              unsigned int b) {
    if (b < nh) {
        const float* x = qg + (u64) b * 2 * d;
        float* y = q + (u64) b * d;
        float* g = gate + (u64) b * d;
        for (unsigned int i = threadIdx.x; i < d; i += blockDim.x) g[i] = x[d + i];
        rmsnorm_one(x, wq, y, d, eps);
        __syncthreads();
        rope_one(y, pos, rope_dims, theta);
    } else {
        unsigned int kh = b - nh;
        const float* x = kk + (u64) kh * d;
        float* y = kc + (u64) kh * d;
        rmsnorm_one(x, wk, y, d, eps);
        __syncthreads();
        rope_one(y, pos, rope_dims, theta);
    }
}

extern "C" __global__ void head_norm_rope_qgk(const float* qg, float* q, float* gate, const float* wq,
                                              const float* kk, float* kc, const float* wk,
                                              unsigned int nh, unsigned int d, float eps,
                                              unsigned int pos, unsigned int rope_dims, float theta) {
    head_norm_rope_qgk_one(qg, q, gate, wq, kk, kc, wk, nh, d, eps, pos, rope_dims, theta, blockIdx.x);
}

// For tokens j = blockIdx.y at positions pos0 + j: qg rows 2 nh d apart, q and gate rows nh d, kk rows nkv d, and the
// keys into the cache at kc + (pos0 + j) * nkv * d.  grid = (nh + nkv, rows).
extern "C" __global__ void head_norm_rope_qgk_rows(const float* qg, float* q, float* gate, const float* wq,
                                                   const float* kk, float* kc, const float* wk, unsigned int nh,
                                                   unsigned int nkv, unsigned int d, float eps, unsigned int pos0,
                                                   unsigned int rope_dims, float theta) {
    const u64 j = blockIdx.y, qd = (u64) nh * d, kd = (u64) nkv * d;
    head_norm_rope_qgk_one(qg + j * 2 * qd, q + j * qd, gate + j * qd, wq, kk + j * kd, kc + (pos0 + j) * kd, wk,
                           nh, d, eps, pos0 + (unsigned int) j, rope_dims, theta, blockIdx.x);
}

// gate[h*hd + j] = qg[h*2*hd + hd + j]: the gate half of each query head
extern "C" __global__ void split_gate(const float* qg, float* gate, unsigned int heads, unsigned int hd) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= heads * hd) return;
    gate[i] = qg[(i / hd) * 2 * hd + hd + i % hd];
}

// Softmax attention of each query head over the positions in `toks` (all 0..ntok-1 when toks is null), GQA, then
// times sigmoid(gate).  Scores, exponentials and the weighted sum in double, as the reference.  grid = heads,
// block = 256; `scratch` holds heads x stride doubles.
static __device__ __forceinline__ void attention_head(const float* q, const float* kc, const float* vc, const u32* toks,
                                                      unsigned int ntok, const float* gate, float* o, double* scratch,
                                                      unsigned int stride, unsigned int nkv, unsigned int hd,
                                                      unsigned int group, unsigned int h) {
    unsigned int g = h / group;
    const float* qh = q + (u64) h * hd;
    double* sc = scratch + (u64) h * stride;
    double scale = 1.0 / sqrt((double) hd);
    double mx = -1e300;
    // The same sums in the same order, with far fewer memory instructions (the kernel was bound by the L1/L2 queue: a load of
    // q[i] and one of kt[i] for every multiply-add, and sc[t] and toks[t] for every one of the weighted sum): q staged in shared
    // memory, K read 16 bytes at a time, the tile's scores and positions staged in shared memory.
    __shared__ float qs[512];
    __shared__ float scs[256];
    __shared__ unsigned int ps[256];
    const bool fast = hd <= 512 && (hd & 15) == 0 && hd <= blockDim.x && blockDim.x <= 256;
    if (fast) {
        for (unsigned int i = threadIdx.x; i < hd; i += blockDim.x) qs[i] = qh[i];
        __syncthreads();
    }
    for (unsigned int t = threadIdx.x; t < ntok; t += blockDim.x) {
        unsigned int p = toks ? toks[t] : t;
        const float* kt = kc + ((u64) p * nkv + g) * hd;
        // the 256-term dot in float runs of 16 added in double (FP64 is 1/64 rate on this GPU)
        double s = 0.0;
        if (fast) {
            const float4* k4 = (const float4*) kt;
            for (unsigned int i0 = 0; i0 < hd; i0 += 16) {
                float part = 0.0f;
                #pragma unroll
                for (unsigned int c = 0; c < 4; ++c) {
                    float4 kv = k4[i0 / 4 + c];
                    part = fmaf(qs[i0 + 4 * c], kv.x, part);
                    part = fmaf(qs[i0 + 4 * c + 1], kv.y, part);
                    part = fmaf(qs[i0 + 4 * c + 2], kv.z, part);
                    part = fmaf(qs[i0 + 4 * c + 3], kv.w, part);
                }
                s += (double) part;
            }
        } else {
            for (unsigned int i0 = 0; i0 < hd; i0 += 16) {
                float part = 0.0f;
                for (unsigned int i = i0; i < i0 + 16 && i < hd; ++i) part = fmaf(qh[i], kt[i], part);
                s += (double) part;
            }
        }
        s *= scale;
        sc[t] = s;
        mx = fmax(mx, s);
    }
    mx = block_max(mx);
    double z = 0.0;
    for (unsigned int t = threadIdx.x; t < ntok; t += blockDim.x) {
        double e = (double) expf((float) (sc[t] - mx));
        sc[t] = e;
        z += e;
    }
    z = block_sum(z);
    __syncthreads();
    if (fast) {
        // one output element a thread; tiles of 256 positions (a multiple of the runs of 16) through shared memory
        const unsigned int j = threadIdx.x;
        double acc = 0.0;
        for (unsigned int tile = 0; tile < ntok; tile += 256) {
            unsigned int t = tile + threadIdx.x;
            __syncthreads();
            if (t < ntok) {
                scs[threadIdx.x] = (float) sc[t];
                ps[threadIdx.x] = toks ? toks[t] : t;
            }
            __syncthreads();
            if (j < hd) {
                const unsigned int n = min(256u, ntok - tile);
                for (unsigned int r0 = 0; r0 < n; r0 += 16) {
                    float part = 0.0f;
                    for (unsigned int r = r0; r < r0 + 16 && r < n; ++r)
                        part = fmaf(scs[r], vc[((u64) ps[r] * nkv + g) * hd + j], part);
                    acc += (double) part;
                }
            }
        }
        if (j < hd) o[h * hd + j] = (float) (acc / z) * sigmoidf_ref(gate[h * hd + j]);
        return;
    }
    for (unsigned int j = threadIdx.x; j < hd; j += blockDim.x) {
        double acc = 0.0;
        for (unsigned int t0 = 0; t0 < ntok; t0 += 16) {
            float part = 0.0f;
            for (unsigned int t = t0; t < t0 + 16 && t < ntok; ++t) {
                unsigned int p = toks ? toks[t] : t;
                part = fmaf((float) sc[t], vc[((u64) p * nkv + g) * hd + j], part);
            }
            acc += (double) part;
        }
        o[h * hd + j] = (float) (acc / z) * sigmoidf_ref(gate[h * hd + j]);
    }
}

extern "C" __global__ void attention_heads(const float* q, const float* kc, const float* vc, const u32* toks,
                                           unsigned int ntok, const float* gate, float* o, double* scratch,
                                           unsigned int stride, unsigned int nkv, unsigned int hd, unsigned int group) {
    attention_head(q, kc, vc, toks, ntok, gate, o, scratch, stride, nkv, hd, group, blockIdx.x);
}

// Tokens j = blockIdx.y attending over positions 0 .. ntok0 + j (all of them: no sparse selection): q, gate and o rows
// nh * hd apart, scratch rows nh * stride doubles.  grid = (nh, rows).
extern "C" __global__ void attention_heads_rows(const float* q, const float* kc, const float* vc, unsigned int ntok0,
                                                const float* gate, float* o, double* scratch, unsigned int stride,
                                                unsigned int nkv, unsigned int hd, unsigned int group, unsigned int nh) {
    const u64 j = blockIdx.y, qd = (u64) nh * hd;
    attention_head(q + j * qd, kc, vc, nullptr, ntok0 + (unsigned int) j, gate + j * qd, o + j * qd,
                   scratch + j * nh * stride, stride, nkv, hd, group, blockIdx.x);
}

// Tokens j = blockIdx.y attending over their own selected positions: toks + j * tstride, ntoks[j] of them (the sparse
// selection).  q, gate and o rows nh * hd apart, scratch rows nh * stride doubles.  grid = (nh, rows).
extern "C" __global__ void attention_heads_sel_rows(const float* q, const float* kc, const float* vc, const u32* toks,
                                                    unsigned int tstride, const u32* ntoks, const float* gate, float* o,
                                                    double* scratch, unsigned int stride, unsigned int nkv, unsigned int hd,
                                                    unsigned int group, unsigned int nh) {
    const u64 j = blockIdx.y, qd = (u64) nh * hd;
    attention_head(q + j * qd, kc, vc, toks + j * tstride, ntoks[j], gate + j * qd, o + j * qd,
                   scratch + j * nh * stride, stride, nkv, hd, group, blockIdx.x);
}

// `attention_heads_sel_rows` with the heads of a GQA group sharing their K and V reads: block (x, row) takes the heads
// [part * hb, min(group, (part + 1) * hb)) of kv head g, x = g * ceil(group / hb) + part, so each selected position's K row (and V
// element) is loaded once for up to hb heads instead of once a head.  Every head's sums are the same fmaf runs of 16 added in double in
// the same order, its softmax the same block reductions: the same numbers as `attention_head`.  hd a multiple of 16, at most 256;
// blockDim 256; hb <= 8.  grid = (nkv * ceil(group / hb), rows).
#define HBMAX 8
extern "C" __global__ void attention_group_sel_rows(const float* q, const float* kc, const float* vc, const u32* toks,
                                                    unsigned int tstride, const u32* ntoks, const float* gate, float* o,
                                                    double* scratch, unsigned int stride, unsigned int nkv, unsigned int hd,
                                                    unsigned int group, unsigned int nh, unsigned int hb) {
    const u64 row = blockIdx.y;
    const unsigned int nsplit = (group + hb - 1) / hb;
    const unsigned int g = blockIdx.x / nsplit, part = blockIdx.x % nsplit;
    const unsigned int h0 = g * group + part * hb;                   // the first head of this block
    const unsigned int nhh = min(hb, group - part * hb);              // how many
    const float* qrow = q + row * nh * hd;
    const float* gaterow = gate + row * nh * hd;
    float* orow = o + row * nh * hd;
    const u32* tk = toks + row * tstride;
    const unsigned int ntok = ntoks[row];
    double* scr = scratch + row * nh * stride;
    __shared__ float qs[HBMAX][256];
    __shared__ float scs[HBMAX][256];
    __shared__ unsigned int ps[256];
    for (unsigned int idx = threadIdx.x; idx < nhh * hd; idx += blockDim.x) qs[idx / hd][idx % hd] = qrow[(u64) h0 * hd + idx];
    __syncthreads();
    const double scale = 1.0 / sqrt((double) hd);
    double mx[HBMAX];
    #pragma unroll
    for (unsigned int a = 0; a < HBMAX; ++a) mx[a] = -1e300;
    for (unsigned int t = threadIdx.x; t < ntok; t += blockDim.x) {
        const float4* k4 = (const float4*) (kc + ((u64) tk[t] * nkv + g) * hd);
        double s[HBMAX];
        #pragma unroll
        for (unsigned int a = 0; a < HBMAX; ++a) s[a] = 0.0;
        for (unsigned int i0 = 0; i0 < hd; i0 += 16) {
            float kv[16];
            #pragma unroll
            for (unsigned int c = 0; c < 4; ++c) {
                float4 v = k4[i0 / 4 + c];
                kv[4 * c] = v.x; kv[4 * c + 1] = v.y; kv[4 * c + 2] = v.z; kv[4 * c + 3] = v.w;
            }
            #pragma unroll
            for (unsigned int a = 0; a < HBMAX; ++a) {
                if (a < nhh) {
                    float p = 0.0f;
                    #pragma unroll
                    for (unsigned int i = 0; i < 16; ++i) p = fmaf(qs[a][i0 + i], kv[i], p);
                    s[a] += (double) p;
                }
            }
        }
        #pragma unroll
        for (unsigned int a = 0; a < HBMAX; ++a) {
            if (a < nhh) {
                s[a] *= scale;
                scr[(u64) (h0 + a) * stride + t] = s[a];
                mx[a] = fmax(mx[a], s[a]);
            }
        }
    }
    double z[HBMAX];
    #pragma unroll
    for (unsigned int a = 0; a < HBMAX; ++a) {
        z[a] = 1.0;
        if (a < nhh) {
            double m = block_max(mx[a]);
            double zt = 0.0;
            double* sc = scr + (u64) (h0 + a) * stride;
            for (unsigned int t = threadIdx.x; t < ntok; t += blockDim.x) {
                double e = (double) expf((float) (sc[t] - m));
                sc[t] = e;
                zt += e;
            }
            z[a] = block_sum(zt);
        }
    }
    __syncthreads();
    const unsigned int j = threadIdx.x;
    double acc[HBMAX];
    #pragma unroll
    for (unsigned int a = 0; a < HBMAX; ++a) acc[a] = 0.0;
    for (unsigned int tile = 0; tile < ntok; tile += 256) {
        __syncthreads();
        const unsigned int n = min(256u, ntok - tile);
        for (unsigned int idx = threadIdx.x; idx < nhh * 256; idx += blockDim.x) {
            const unsigned int a = idx / 256, r = idx % 256;
            if (r < n) scs[a][r] = (float) scr[(u64) (h0 + a) * stride + tile + r];
        }
        if (threadIdx.x < n) ps[threadIdx.x] = tk[tile + threadIdx.x];
        __syncthreads();
        if (j < hd) {
            for (unsigned int r0 = 0; r0 < n; r0 += 16) {
                float p[HBMAX];
                #pragma unroll
                for (unsigned int a = 0; a < HBMAX; ++a) p[a] = 0.0f;
                for (unsigned int r = r0; r < r0 + 16 && r < n; ++r) {
                    const float v = vc[((u64) ps[r] * nkv + g) * hd + j];
                    #pragma unroll
                    for (unsigned int a = 0; a < HBMAX; ++a) if (a < nhh) p[a] = fmaf(scs[a][r], v, p[a]);
                }
                #pragma unroll
                for (unsigned int a = 0; a < HBMAX; ++a) acc[a] += (double) p[a];
            }
        }
    }
    if (j < hd) {
        #pragma unroll
        for (unsigned int a = 0; a < HBMAX; ++a)
            if (a < nhh) orow[(u64) (h0 + a) * hd + j] = (float) (acc[a] / z[a]) * sigmoidf_ref(gaterow[(u64) (h0 + a) * hd + j]);
    }
}

// `attention_group_sel_rows` with scores, softmax and the weighted sum in float (the card does FP64 at 1/64 of FP32 and the double kernel was FP64-bound: 74% of the pipe, DRAM 2%):
// inexact against the reference (1e-6 relative), so for `--tensor-prompt`; same arguments (the scratch is the same buffer, read as floats).
extern "C" __global__ void attention_group_sel_rows_f32(const float* q, const float* kc, const float* vc, const u32* toks,
                                                    unsigned int tstride, const u32* ntoks, const float* gate, float* o,
                                                    double* scratch_d, unsigned int stride, unsigned int nkv, unsigned int hd,
                                                    unsigned int group, unsigned int nh, unsigned int hb) {
    const u64 row = blockIdx.y;
    const unsigned int nsplit = (group + hb - 1) / hb;
    const unsigned int g = blockIdx.x / nsplit, part = blockIdx.x % nsplit;
    const unsigned int h0 = g * group + part * hb;                   // the first head of this block
    const unsigned int nhh = min(hb, group - part * hb);              // how many
    const float* qrow = q + row * nh * hd;
    const float* gaterow = gate + row * nh * hd;
    float* orow = o + row * nh * hd;
    const u32* tk = toks + row * tstride;
    const unsigned int ntok = ntoks[row];
    float* scr = (float*) scratch_d + row * nh * stride;
    __shared__ float qs[HBMAX][256];
    __shared__ float scs[HBMAX][256];
    __shared__ unsigned int ps[256];
    for (unsigned int idx = threadIdx.x; idx < nhh * hd; idx += blockDim.x) qs[idx / hd][idx % hd] = qrow[(u64) h0 * hd + idx];
    __syncthreads();
    const float scale = 1.0f / sqrtf((float) hd);
    float mx[HBMAX];
    #pragma unroll
    for (unsigned int a = 0; a < HBMAX; ++a) mx[a] = -1e30f;
    for (unsigned int t = threadIdx.x; t < ntok; t += blockDim.x) {
        const float4* k4 = (const float4*) (kc + ((u64) tk[t] * nkv + g) * hd);
        float s[HBMAX];
        #pragma unroll
        for (unsigned int a = 0; a < HBMAX; ++a) s[a] = 0.0f;
        for (unsigned int i0 = 0; i0 < hd; i0 += 16) {
            float kv[16];
            #pragma unroll
            for (unsigned int c = 0; c < 4; ++c) {
                float4 v = k4[i0 / 4 + c];
                kv[4 * c] = v.x; kv[4 * c + 1] = v.y; kv[4 * c + 2] = v.z; kv[4 * c + 3] = v.w;
            }
            #pragma unroll
            for (unsigned int a = 0; a < HBMAX; ++a) {
                if (a < nhh) {
                    float p = 0.0f;
                    #pragma unroll
                    for (unsigned int i = 0; i < 16; ++i) p = fmaf(qs[a][i0 + i], kv[i], p);
                    s[a] += p;
                }
            }
        }
        #pragma unroll
        for (unsigned int a = 0; a < HBMAX; ++a) {
            if (a < nhh) {
                s[a] *= scale;
                scr[(u64) (h0 + a) * stride + t] = s[a];
                mx[a] = fmaxf(mx[a], s[a]);
            }
        }
    }
    float z[HBMAX];
    #pragma unroll
    for (unsigned int a = 0; a < HBMAX; ++a) {
        z[a] = 1.0f;
        if (a < nhh) {
            float m = (float) block_max((double) mx[a]);
            float zt = 0.0f;
            float* sc = scr + (u64) (h0 + a) * stride;
            for (unsigned int t = threadIdx.x; t < ntok; t += blockDim.x) {
                float e = expf(sc[t] - m);
                sc[t] = e;
                zt += e;
            }
            z[a] = (float) block_sum((double) zt);
        }
    }
    __syncthreads();
    const unsigned int j = threadIdx.x;
    float acc[HBMAX];
    #pragma unroll
    for (unsigned int a = 0; a < HBMAX; ++a) acc[a] = 0.0f;
    for (unsigned int tile = 0; tile < ntok; tile += 256) {
        __syncthreads();
        const unsigned int n = min(256u, ntok - tile);
        for (unsigned int idx = threadIdx.x; idx < nhh * 256; idx += blockDim.x) {
            const unsigned int a = idx / 256, r = idx % 256;
            if (r < n) scs[a][r] = scr[(u64) (h0 + a) * stride + tile + r];
        }
        if (threadIdx.x < n) ps[threadIdx.x] = tk[tile + threadIdx.x];
        __syncthreads();
        if (j < hd) {
            for (unsigned int r0 = 0; r0 < n; r0 += 16) {
                float p[HBMAX];
                #pragma unroll
                for (unsigned int a = 0; a < HBMAX; ++a) p[a] = 0.0f;
                for (unsigned int r = r0; r < r0 + 16 && r < n; ++r) {
                    const float v = vc[((u64) ps[r] * nkv + g) * hd + j];
                    #pragma unroll
                    for (unsigned int a = 0; a < HBMAX; ++a) if (a < nhh) p[a] = fmaf(scs[a][r], v, p[a]);
                }
                #pragma unroll
                for (unsigned int a = 0; a < HBMAX; ++a) acc[a] += p[a];
            }
        }
    }
    if (j < hd) {
        #pragma unroll
        for (unsigned int a = 0; a < HBMAX; ++a)
            if (a < nhh) orow[(u64) (h0 + a) * hd + j] = (acc[a] / z[a]) * sigmoidf_ref(gaterow[(u64) (h0 + a) * hd + j]);
    }
}

// ---- `attention_group_sel_rows_f32` rebuilt around the profile (Nsight Compute, 9.5k prompt: 6% of the FP32 FMA rate, LSU requests at 71% of peak = bound by load issue; the shared
// 8 KB double buffers of block_sum/block_max held it to 2 blocks an SM): 4 heads a block (the launch's hb must be 4), scores 4 tokens a thread (every q load from shared memory feeds
// 4x the FMAs), the max and sum by warp shuffles, exp folded into the weighted sum's tile load (no separate pass over the scores), and the weighted sum with each thread holding
// 4 columns of V as a float4 for 4 heads (one q-weight load per 16 FMAs; the tile's rows split over 4 thread groups, summed at the end).
// Same arguments as `attention_group_sel_rows_f32`.
static __device__ __forceinline__ void block4_reduce(float* v, float* w, bool is_max) {
    #pragma unroll
    for (unsigned int a = 0; a < 4; ++a)
        #pragma unroll
        for (unsigned int o = 16; o > 0; o >>= 1) {
            const float x = __shfl_xor_sync(0xffffffffu, v[a], o);
            v[a] = is_max ? fmaxf(v[a], x) : v[a] + x;
        }
    if ((threadIdx.x & 31) == 0) {
        #pragma unroll
        for (unsigned int a = 0; a < 4; ++a) w[a * 8 + (threadIdx.x >> 5)] = v[a];
    }
    __syncthreads();
    #pragma unroll
    for (unsigned int a = 0; a < 4; ++a) {
        float r = w[a * 8];
        #pragma unroll
        for (unsigned int k = 1; k < 8; ++k) r = is_max ? fmaxf(r, w[a * 8 + k]) : r + w[a * 8 + k];
        v[a] = r;
    }
    __syncthreads();
}

extern "C" __global__ void __launch_bounds__(256) attention_group_sel_rows_v2(const float* q, const float* kc, const float* vc, const u32* toks,
                                                    unsigned int tstride, const u32* ntoks, const float* gate, float* o,
                                                    double* scratch_d, unsigned int stride, unsigned int nkv, unsigned int hd,
                                                    unsigned int group, unsigned int nh, unsigned int hb) {
    const u64 row = blockIdx.y;
    const unsigned int nsplit = (group + 3) / 4;
    const unsigned int g = blockIdx.x / nsplit, part = blockIdx.x % nsplit;
    const unsigned int h0 = g * group + part * 4;
    const unsigned int nhh = min(4u, group - part * 4);
    const float* qrow = q + row * nh * hd;
    const float* gaterow = gate + row * nh * hd;
    float* orow = o + row * nh * hd;
    const u32* tk = toks + row * tstride;
    const unsigned int ntok = ntoks[row];
    float* scr = (float*) scratch_d + row * nh * stride;
    __shared__ __align__(16) float qs[4 * 256];
    __shared__ __align__(16) float scs[256 * 4];
    __shared__ __align__(16) float red[3 * 4 * 256];
    __shared__ unsigned int ps[256];
    __shared__ float wred[32];
    const unsigned int tid = threadIdx.x;
    for (unsigned int idx = tid; idx < 4 * 256; idx += 256) qs[idx] = 0.0f;
    __syncthreads();
    for (unsigned int idx = tid; idx < nhh * hd; idx += 256) qs[(idx / hd) * 256 + idx % hd] = qrow[(u64) h0 * hd + idx];
    __syncthreads();
    const float scale = 1.0f / sqrtf((float) hd);
    float mx[4] = {-1e30f, -1e30f, -1e30f, -1e30f};
    for (unsigned int base = 0; base < ntok; base += 1024) {
        const float4* k4[4];
        bool ok[4];
        #pragma unroll
        for (unsigned int u = 0; u < 4; ++u) {
            const unsigned int t = base + tid + 256 * u;
            ok[u] = t < ntok;
            k4[u] = (const float4*) (kc + ((u64) (ok[u] ? tk[t] : 0u) * nkv + g) * hd);
        }
        float s[4][4];
        #pragma unroll
        for (unsigned int u = 0; u < 4; ++u)
            #pragma unroll
            for (unsigned int a = 0; a < 4; ++a) s[u][a] = 0.0f;
        for (unsigned int i0 = 0; i0 < hd; i0 += 16) {
            float4 kv[4][4];
            #pragma unroll
            for (unsigned int u = 0; u < 4; ++u)
                #pragma unroll
                for (unsigned int c = 0; c < 4; ++c) kv[u][c] = k4[u][i0 / 4 + c];
            #pragma unroll
            for (unsigned int a = 0; a < 4; ++a) {
                if (a < nhh) {
                    #pragma unroll
                    for (unsigned int c = 0; c < 4; ++c) {
                        const float4 qq = *(const float4*) &qs[a * 256 + i0 + 4 * c];
                        #pragma unroll
                        for (unsigned int u = 0; u < 4; ++u) {
                            s[u][a] = fmaf(qq.x, kv[u][c].x, s[u][a]);
                            s[u][a] = fmaf(qq.y, kv[u][c].y, s[u][a]);
                            s[u][a] = fmaf(qq.z, kv[u][c].z, s[u][a]);
                            s[u][a] = fmaf(qq.w, kv[u][c].w, s[u][a]);
                        }
                    }
                }
            }
        }
        #pragma unroll
        for (unsigned int u = 0; u < 4; ++u) {
            if (ok[u]) {
                const unsigned int t = base + tid + 256 * u;
                #pragma unroll
                for (unsigned int a = 0; a < 4; ++a) {
                    if (a < nhh) {
                        s[u][a] *= scale;
                        scr[(u64) (h0 + a) * stride + t] = s[u][a];
                        mx[a] = fmaxf(mx[a], s[u][a]);
                    }
                }
            }
        }
    }
    block4_reduce(mx, wred, true);
    float zt[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    const unsigned int grp = tid >> 6, c4 = (tid & 63) * 4;
    float4 acc[4];
    #pragma unroll
    for (unsigned int a = 0; a < 4; ++a) acc[a] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    for (unsigned int tile = 0; tile < ntok; tile += 256) {
        __syncthreads();
        const unsigned int n = min(256u, ntok - tile);
        #pragma unroll
        for (unsigned int a = 0; a < 4; ++a) {
            float e = 0.0f;
            if (a < nhh && tid < n) {
                e = expf(scr[(u64) (h0 + a) * stride + tile + tid] - mx[a]);
                zt[a] += e;
            }
            scs[tid * 4 + a] = e;
        }
        ps[tid] = tid < n ? tk[tile + tid] : 0u;
        __syncthreads();
        if (c4 < hd) {
            const unsigned int npad = (n + 31) & ~31u;
            for (unsigned int rb = 0; rb < npad; rb += 32) {
                float4 v[8], w[8];
                #pragma unroll
                for (unsigned int k = 0; k < 8; ++k) {
                    const unsigned int r = rb + grp * 8 + k;
                    v[k] = *(const float4*) (vc + ((u64) ps[r] * nkv + g) * hd + c4);
                    w[k] = *(const float4*) &scs[r * 4];
                }
                #pragma unroll
                for (unsigned int k = 0; k < 8; ++k) {
                    acc[0].x = fmaf(w[k].x, v[k].x, acc[0].x); acc[0].y = fmaf(w[k].x, v[k].y, acc[0].y);
                    acc[0].z = fmaf(w[k].x, v[k].z, acc[0].z); acc[0].w = fmaf(w[k].x, v[k].w, acc[0].w);
                    acc[1].x = fmaf(w[k].y, v[k].x, acc[1].x); acc[1].y = fmaf(w[k].y, v[k].y, acc[1].y);
                    acc[1].z = fmaf(w[k].y, v[k].z, acc[1].z); acc[1].w = fmaf(w[k].y, v[k].w, acc[1].w);
                    acc[2].x = fmaf(w[k].z, v[k].x, acc[2].x); acc[2].y = fmaf(w[k].z, v[k].y, acc[2].y);
                    acc[2].z = fmaf(w[k].z, v[k].z, acc[2].z); acc[2].w = fmaf(w[k].z, v[k].w, acc[2].w);
                    acc[3].x = fmaf(w[k].w, v[k].x, acc[3].x); acc[3].y = fmaf(w[k].w, v[k].y, acc[3].y);
                    acc[3].z = fmaf(w[k].w, v[k].z, acc[3].z); acc[3].w = fmaf(w[k].w, v[k].w, acc[3].w);
                }
            }
        }
    }
    block4_reduce(zt, wred, false);
    // the 4 row groups' partial sums: groups 1..3 hand theirs to group 0
    if (grp > 0 && c4 < hd) {
        #pragma unroll
        for (unsigned int a = 0; a < 4; ++a) *(float4*) &red[((grp - 1) * 4 + a) * 256 + c4] = acc[a];
    }
    __syncthreads();
    if (grp == 0 && c4 < hd) {
        #pragma unroll
        for (unsigned int a = 0; a < 4; ++a) {
            if (a < nhh) {
                float4 r = acc[a];
                #pragma unroll
                for (unsigned int k = 0; k < 3; ++k) {
                    const float4 x = *(const float4*) &red[(k * 4 + a) * 256 + c4];
                    r.x += x.x; r.y += x.y; r.z += x.z; r.w += x.w;
                }
                const float inv = 1.0f / zt[a];
                const float* gt = gaterow + (u64) (h0 + a) * hd + c4;
                float* op = orow + (u64) (h0 + a) * hd + c4;
                op[0] = (r.x * inv) * sigmoidf_ref(gt[0]);
                op[1] = (r.y * inv) * sigmoidf_ref(gt[1]);
                op[2] = (r.z * inv) * sigmoidf_ref(gt[2]);
                op[3] = (r.w * inv) * sigmoidf_ref(gt[3]);
            }
        }
    }
}

// ---- `attention_group_sel_rows_v2` with all of a KV group's heads (up to 12) in one block: the K and V rows of the selected positions are read once for the group, not once for
// each block of 4 heads (v2's profile: 75% of L2 peak, the K and V reads three times over).  Scores 2 tokens a thread, 12 heads: a q load from shared memory feeds 8 FMAs.
// The weighted sum: a float4 of V columns for 12 heads a thread (48 accumulators), 4 row groups, summed one group at a time through the dead q buffer.
// Same arguments as `attention_group_sel_rows_v2`; the launch has one block a (group, row) and `group` <= 12.
#define V3H 12
static __device__ __forceinline__ void block12_reduce(float* v, float* w, bool is_max) {
    #pragma unroll
    for (unsigned int a = 0; a < V3H; ++a)
        #pragma unroll
        for (unsigned int o = 16; o > 0; o >>= 1) {
            const float x = __shfl_xor_sync(0xffffffffu, v[a], o);
            v[a] = is_max ? fmaxf(v[a], x) : v[a] + x;
        }
    if ((threadIdx.x & 31) == 0) {
        #pragma unroll
        for (unsigned int a = 0; a < V3H; ++a) w[a * 8 + (threadIdx.x >> 5)] = v[a];
    }
    __syncthreads();
    #pragma unroll
    for (unsigned int a = 0; a < V3H; ++a) {
        float r = w[a * 8];
        #pragma unroll
        for (unsigned int k = 1; k < 8; ++k) r = is_max ? fmaxf(r, w[a * 8 + k]) : r + w[a * 8 + k];
        v[a] = r;
    }
    __syncthreads();
}

extern "C" __global__ void __launch_bounds__(256) attention_group_sel_rows_v3(const float* q, const float* kc, const float* vc, const u32* toks,
                                                    unsigned int tstride, const u32* ntoks, const float* gate, float* o,
                                                    double* scratch_d, unsigned int stride, unsigned int nkv, unsigned int hd,
                                                    unsigned int group, unsigned int nh, unsigned int hb) {
    const u64 row = blockIdx.y;
    const unsigned int g = blockIdx.x;
    const unsigned int h0 = g * group;
    const unsigned int nhh = group;                                  // <= 12
    const float* qrow = q + row * nh * hd;
    const float* gaterow = gate + row * nh * hd;
    float* orow = o + row * nh * hd;
    const u32* tk = toks + row * tstride;
    const unsigned int ntok = ntoks[row];
    float* scr = (float*) scratch_d + row * nh * stride;
    __shared__ __align__(16) float qs[V3H * 256];                     // also the cross-group sum buffer at the end
    __shared__ __align__(16) float scs[256 * V3H];
    __shared__ unsigned int ps[256];
    __shared__ float wred[V3H * 8];
    const unsigned int tid = threadIdx.x;
    for (unsigned int idx = tid; idx < V3H * 256; idx += 256) qs[idx] = 0.0f;
    __syncthreads();
    for (unsigned int idx = tid; idx < nhh * hd; idx += 256) qs[(idx / hd) * 256 + idx % hd] = qrow[(u64) h0 * hd + idx];
    __syncthreads();
    const float scale = 1.0f / sqrtf((float) hd);
    float mx[V3H];
    #pragma unroll
    for (unsigned int a = 0; a < V3H; ++a) mx[a] = -1e30f;
    for (unsigned int base = 0; base < ntok; base += 512) {
        const float4* k4[2];
        bool ok[2];
        #pragma unroll
        for (unsigned int u = 0; u < 2; ++u) {
            const unsigned int t = base + tid + 256 * u;
            ok[u] = t < ntok;
            k4[u] = (const float4*) (kc + ((u64) (ok[u] ? tk[t] : 0u) * nkv + g) * hd);
        }
        float s[2][V3H];
        #pragma unroll
        for (unsigned int u = 0; u < 2; ++u)
            #pragma unroll
            for (unsigned int a = 0; a < V3H; ++a) s[u][a] = 0.0f;
        for (unsigned int i0 = 0; i0 < hd; i0 += 16) {
            float4 kv[2][4];
            #pragma unroll
            for (unsigned int u = 0; u < 2; ++u)
                #pragma unroll
                for (unsigned int c = 0; c < 4; ++c) kv[u][c] = k4[u][i0 / 4 + c];
            #pragma unroll
            for (unsigned int a = 0; a < V3H; ++a) {
                if (a < nhh) {
                    #pragma unroll
                    for (unsigned int c = 0; c < 4; ++c) {
                        const float4 qq = *(const float4*) &qs[a * 256 + i0 + 4 * c];
                        #pragma unroll
                        for (unsigned int u = 0; u < 2; ++u) {
                            s[u][a] = fmaf(qq.x, kv[u][c].x, s[u][a]);
                            s[u][a] = fmaf(qq.y, kv[u][c].y, s[u][a]);
                            s[u][a] = fmaf(qq.z, kv[u][c].z, s[u][a]);
                            s[u][a] = fmaf(qq.w, kv[u][c].w, s[u][a]);
                        }
                    }
                }
            }
        }
        #pragma unroll
        for (unsigned int u = 0; u < 2; ++u) {
            if (ok[u]) {
                const unsigned int t = base + tid + 256 * u;
                #pragma unroll
                for (unsigned int a = 0; a < V3H; ++a) {
                    if (a < nhh) {
                        s[u][a] *= scale;
                        scr[(u64) (h0 + a) * stride + t] = s[u][a];
                        mx[a] = fmaxf(mx[a], s[u][a]);
                    }
                }
            }
        }
    }
    block12_reduce(mx, wred, true);
    float zt[V3H];
    #pragma unroll
    for (unsigned int a = 0; a < V3H; ++a) zt[a] = 0.0f;
    const unsigned int grp = tid >> 6, c4 = (tid & 63) * 4;
    float4 acc[V3H];
    #pragma unroll
    for (unsigned int a = 0; a < V3H; ++a) acc[a] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    for (unsigned int tile = 0; tile < ntok; tile += 256) {
        __syncthreads();
        const unsigned int n = min(256u, ntok - tile);
        #pragma unroll
        for (unsigned int a = 0; a < V3H; ++a) {
            float e = 0.0f;
            if (a < nhh && tid < n) {
                e = expf(scr[(u64) (h0 + a) * stride + tile + tid] - mx[a]);
                zt[a] += e;
            }
            scs[tid * V3H + a] = e;
        }
        ps[tid] = tid < n ? tk[tile + tid] : 0u;
        __syncthreads();
        if (c4 < hd) {
            const unsigned int npad = (n + 31) & ~31u;
            for (unsigned int rb = 0; rb < npad; rb += 32) {
                #pragma unroll
                for (unsigned int kb = 0; kb < 8; kb += 4) {
                    float4 v[4];
                    #pragma unroll
                    for (unsigned int k = 0; k < 4; ++k) {
                        const unsigned int r = rb + grp * 8 + kb + k;
                        v[k] = *(const float4*) (vc + ((u64) ps[r] * nkv + g) * hd + c4);
                    }
                    #pragma unroll
                    for (unsigned int k = 0; k < 4; ++k) {
                        const unsigned int r = rb + grp * 8 + kb + k;
                        #pragma unroll
                        for (unsigned int h4 = 0; h4 < V3H / 4; ++h4) {
                            const float4 w = *(const float4*) &scs[r * V3H + h4 * 4];
                            acc[h4 * 4 + 0].x = fmaf(w.x, v[k].x, acc[h4 * 4 + 0].x); acc[h4 * 4 + 0].y = fmaf(w.x, v[k].y, acc[h4 * 4 + 0].y);
                            acc[h4 * 4 + 0].z = fmaf(w.x, v[k].z, acc[h4 * 4 + 0].z); acc[h4 * 4 + 0].w = fmaf(w.x, v[k].w, acc[h4 * 4 + 0].w);
                            acc[h4 * 4 + 1].x = fmaf(w.y, v[k].x, acc[h4 * 4 + 1].x); acc[h4 * 4 + 1].y = fmaf(w.y, v[k].y, acc[h4 * 4 + 1].y);
                            acc[h4 * 4 + 1].z = fmaf(w.y, v[k].z, acc[h4 * 4 + 1].z); acc[h4 * 4 + 1].w = fmaf(w.y, v[k].w, acc[h4 * 4 + 1].w);
                            acc[h4 * 4 + 2].x = fmaf(w.z, v[k].x, acc[h4 * 4 + 2].x); acc[h4 * 4 + 2].y = fmaf(w.z, v[k].y, acc[h4 * 4 + 2].y);
                            acc[h4 * 4 + 2].z = fmaf(w.z, v[k].z, acc[h4 * 4 + 2].z); acc[h4 * 4 + 2].w = fmaf(w.z, v[k].w, acc[h4 * 4 + 2].w);
                            acc[h4 * 4 + 3].x = fmaf(w.w, v[k].x, acc[h4 * 4 + 3].x); acc[h4 * 4 + 3].y = fmaf(w.w, v[k].y, acc[h4 * 4 + 3].y);
                            acc[h4 * 4 + 3].z = fmaf(w.w, v[k].z, acc[h4 * 4 + 3].z); acc[h4 * 4 + 3].w = fmaf(w.w, v[k].w, acc[h4 * 4 + 3].w);
                        }
                    }
                }
            }
        }
    }
    block12_reduce(zt, wred, false);
    // the 4 row groups' partial sums, one group at a time through the (dead) q buffer
    #pragma unroll
    for (unsigned int k = 1; k < 4; ++k) {
        __syncthreads();
        if (grp == k && c4 < hd) {
            #pragma unroll
            for (unsigned int a = 0; a < V3H; ++a) *(float4*) &qs[a * 256 + c4] = acc[a];
        }
        __syncthreads();
        if (grp == 0 && c4 < hd) {
            #pragma unroll
            for (unsigned int a = 0; a < V3H; ++a) {
                const float4 x = *(const float4*) &qs[a * 256 + c4];
                acc[a].x += x.x; acc[a].y += x.y; acc[a].z += x.z; acc[a].w += x.w;
            }
        }
    }
    if (grp == 0 && c4 < hd) {
        #pragma unroll
        for (unsigned int a = 0; a < V3H; ++a) {
            if (a < nhh) {
                const float inv = 1.0f / zt[a];
                const float* gt = gaterow + (u64) (h0 + a) * hd + c4;
                float* op = orow + (u64) (h0 + a) * hd + c4;
                op[0] = (acc[a].x * inv) * sigmoidf_ref(gt[0]);
                op[1] = (acc[a].y * inv) * sigmoidf_ref(gt[1]);
                op[2] = (acc[a].z * inv) * sigmoidf_ref(gt[2]);
                op[3] = (acc[a].w * inv) * sigmoidf_ref(gt[3]);
            }
        }
    }
}

// `attention_heads_rows` for a prompt's first positions (every earlier token is attended) in float, a block a (row, key group, `hb` heads) as `attention_group_sel_rows_f32`
// shares each key row's loads across the `hb` heads (the one-head-a-block kernel read every key row once per head of its group).  Same arguments as the sparse kernel
// without the token lists: row r attends positions 0 .. ntok0 + r - 1.
extern "C" __global__ void attention_group_dense_rows_f32(const float* q, const float* kc, const float* vc, unsigned int ntok0, const float* gate, float* o,
                                                    double* scratch_d, unsigned int stride, unsigned int nkv, unsigned int hd,
                                                    unsigned int group, unsigned int nh, unsigned int hb) {
    const u64 row = blockIdx.y;
    const unsigned int nsplit = (group + hb - 1) / hb;
    const unsigned int g = blockIdx.x / nsplit, part = blockIdx.x % nsplit;
    const unsigned int h0 = g * group + part * hb;                   // the first head of this block
    const unsigned int nhh = min(hb, group - part * hb);              // how many
    const float* qrow = q + row * nh * hd;
    const float* gaterow = gate + row * nh * hd;
    float* orow = o + row * nh * hd;
    const unsigned int ntok = ntok0 + (unsigned int) row;
    float* scr = (float*) scratch_d + row * nh * stride;
    __shared__ float qs[HBMAX][256];
    __shared__ float scs[HBMAX][256];
    __shared__ unsigned int ps[256];
    for (unsigned int idx = threadIdx.x; idx < nhh * hd; idx += blockDim.x) qs[idx / hd][idx % hd] = qrow[(u64) h0 * hd + idx];
    __syncthreads();
    const float scale = 1.0f / sqrtf((float) hd);
    float mx[HBMAX];
    #pragma unroll
    for (unsigned int a = 0; a < HBMAX; ++a) mx[a] = -1e30f;
    for (unsigned int t = threadIdx.x; t < ntok; t += blockDim.x) {
        const float4* k4 = (const float4*) (kc + ((u64) t * nkv + g) * hd);
        float s[HBMAX];
        #pragma unroll
        for (unsigned int a = 0; a < HBMAX; ++a) s[a] = 0.0f;
        for (unsigned int i0 = 0; i0 < hd; i0 += 16) {
            float kv[16];
            #pragma unroll
            for (unsigned int c = 0; c < 4; ++c) {
                float4 v = k4[i0 / 4 + c];
                kv[4 * c] = v.x; kv[4 * c + 1] = v.y; kv[4 * c + 2] = v.z; kv[4 * c + 3] = v.w;
            }
            #pragma unroll
            for (unsigned int a = 0; a < HBMAX; ++a) {
                if (a < nhh) {
                    float p = 0.0f;
                    #pragma unroll
                    for (unsigned int i = 0; i < 16; ++i) p = fmaf(qs[a][i0 + i], kv[i], p);
                    s[a] += p;
                }
            }
        }
        #pragma unroll
        for (unsigned int a = 0; a < HBMAX; ++a) {
            if (a < nhh) {
                s[a] *= scale;
                scr[(u64) (h0 + a) * stride + t] = s[a];
                mx[a] = fmaxf(mx[a], s[a]);
            }
        }
    }
    float z[HBMAX];
    #pragma unroll
    for (unsigned int a = 0; a < HBMAX; ++a) {
        z[a] = 1.0f;
        if (a < nhh) {
            float m = (float) block_max((double) mx[a]);
            float zt = 0.0f;
            float* sc = scr + (u64) (h0 + a) * stride;
            for (unsigned int t = threadIdx.x; t < ntok; t += blockDim.x) {
                float e = expf(sc[t] - m);
                sc[t] = e;
                zt += e;
            }
            z[a] = (float) block_sum((double) zt);
        }
    }
    __syncthreads();
    const unsigned int j = threadIdx.x;
    float acc[HBMAX];
    #pragma unroll
    for (unsigned int a = 0; a < HBMAX; ++a) acc[a] = 0.0f;
    for (unsigned int tile = 0; tile < ntok; tile += 256) {
        __syncthreads();
        const unsigned int n = min(256u, ntok - tile);
        for (unsigned int idx = threadIdx.x; idx < nhh * 256; idx += blockDim.x) {
            const unsigned int a = idx / 256, r = idx % 256;
            if (r < n) scs[a][r] = scr[(u64) (h0 + a) * stride + tile + r];
        }
        if (threadIdx.x < n) ps[threadIdx.x] = tile + threadIdx.x;
        __syncthreads();
        if (j < hd) {
            for (unsigned int r0 = 0; r0 < n; r0 += 16) {
                float p[HBMAX];
                #pragma unroll
                for (unsigned int a = 0; a < HBMAX; ++a) p[a] = 0.0f;
                for (unsigned int r = r0; r < r0 + 16 && r < n; ++r) {
                    const float v = vc[((u64) ps[r] * nkv + g) * hd + j];
                    #pragma unroll
                    for (unsigned int a = 0; a < HBMAX; ++a) if (a < nhh) p[a] = fmaf(scs[a][r], v, p[a]);
                }
                #pragma unroll
                for (unsigned int a = 0; a < HBMAX; ++a) acc[a] += p[a];
            }
        }
    }
    if (j < hd) {
        #pragma unroll
        for (unsigned int a = 0; a < HBMAX; ++a)
            if (a < nhh) orow[(u64) (h0 + a) * hd + j] = (acc[a] / z[a]) * sigmoidf_ref(gaterow[(u64) (h0 + a) * hd + j]);
    }
}

// Softmax attention of each query head over positions start_pos .. start_pos + ntok, GQA, with a learned per-head
// sink logit included in the softmax max and denominator (zero value contribution) and no output gate (Flint).
extern "C" __global__ void attention_sinks_heads(const float* q, const float* kc, const float* vc,
                                                 const float* sinks, unsigned int start_pos, unsigned int ntok,
                                                 float* o, double* scratch, unsigned int stride,
                                                 unsigned int nkv, unsigned int hd, unsigned int group) {
    unsigned int h = blockIdx.x, g = h / group;
    const float* qh = q + (u64) h * hd;
    double* sc = scratch + (u64) h * stride;
    double scale = 1.0 / sqrt((double) hd);
    double sink = (double) sinks[h];
    double mx = sink;
    for (unsigned int t = threadIdx.x; t < ntok; t += blockDim.x) {
        unsigned int p = start_pos + t;
        const float* kt = kc + ((u64) p * nkv + g) * hd;
        double s = 0.0;
        for (unsigned int i = 0; i < hd; ++i) s += (double) qh[i] * (double) kt[i];
        s *= scale;
        sc[t] = s;
        mx = fmax(mx, s);
    }
    mx = block_max(mx);
    double z = 0.0;
    for (unsigned int t = threadIdx.x; t < ntok; t += blockDim.x) {
        double e = exp(sc[t] - mx);
        sc[t] = e;
        z += e;
    }
    z = block_sum(z) + exp(sink - mx);
    __syncthreads();
    for (unsigned int j = threadIdx.x; j < hd; j += blockDim.x) {
        double acc = 0.0;
        for (unsigned int t = 0; t < ntok; ++t) {
            unsigned int p = start_pos + t;
            acc += sc[t] * (double) vc[((u64) p * nkv + g) * hd + j];
        }
        o[h * hd + j] = (float) (acc / z);
    }
}

// The indexer's key for complete block b: the mean of its `ratio` raw keys (summed in order, then divided), RMSNorm
// with w, RoPE at the block's first position.  One block of 128 threads.
static __device__ __forceinline__ void pool_block_one(const float* raw, float* blocks, const float* w, unsigned int b,
                                                      unsigned int ratio, unsigned int id, float eps, unsigned int rope_dims,
                                                      float theta) {
    __shared__ float pooled[256];
    for (unsigned int i = threadIdx.x; i < id; i += blockDim.x) {
        float acc = 0.0f;
        for (unsigned int r = 0; r < ratio; ++r) acc += raw[((u64) b * ratio + r) * id + i];
        pooled[i] = acc / (float) ratio;
    }
    __syncthreads();
    float* out = blocks + (u64) b * id;
    rmsnorm_one(pooled, w, out, id, eps);
    __syncthreads();
    rope_one(out, b * ratio, rope_dims, theta);
}

extern "C" __global__ void pool_block(const float* raw, float* blocks, const float* w, unsigned int b,
                                      unsigned int ratio, unsigned int id, float eps, unsigned int rope_dims,
                                      float theta) {
    pool_block_one(raw, blocks, w, b, ratio, id, eps, rope_dims, theta);
}

// `pool_block` for the blocks b0 .. b0 + gridDim.x - 1 in one launch (a prompt pass completes hundreds a layer, which took one launch each): the same numbers a block.
extern "C" __global__ void pool_blocks(const float* raw, float* blocks, const float* w, unsigned int b0,
                                       unsigned int ratio, unsigned int id, float eps, unsigned int rope_dims,
                                       float theta) {
    pool_block_one(raw, blocks, w, b0 + blockIdx.x, ratio, id, eps, rope_dims, theta);
}

// score[b] = (sum over heads of max(0, q_h . key_b)) / sqrt(id); the dots in double, rounded to float, as the reference.
// One thread per block.
// `index_scores` for tokens r = blockIdx.y (row r has (n0 + r) / ratio complete blocks): iq rows `iq_rs` floats apart, score rows
// `sc_rs` floats apart.  grid = (ceil(max blocks / 256), rows), block = 256.
extern "C" __global__ void index_scores_rows(const float* iq, const float* blocks, float* scores, unsigned int n0,
                                             unsigned int ratio, unsigned int heads, unsigned int id,
                                             unsigned int iq_rs, unsigned int sc_rs) {
    const u64 r = blockIdx.y;
    unsigned int nb = (n0 + (unsigned int) r) / ratio;
    unsigned int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= nb) return;
    const float* q = iq + r * iq_rs;
    const float* kb = blocks + (u64) b * id;
    float s = 0.0f;
    for (unsigned int h = 0; h < heads; ++h) {
        double d = 0.0;
        for (unsigned int i = 0; i < id; ++i) d += (double) q[h * id + i] * (double) kb[i];
        s += fmaxf((float) d, 0.0f);
    }
    scores[r * sc_rs + b] = s / sqrtf((float) id);
}

extern "C" __global__ void index_scores(const float* iq, const float* blocks, float* scores, unsigned int nb,
                                        unsigned int heads, unsigned int id) {
    unsigned int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= nb) return;
    const float* kb = blocks + (u64) b * id;
    float s = 0.0f;
    for (unsigned int h = 0; h < heads; ++h) {
        double d = 0.0;
        for (unsigned int i = 0; i < id; ++i) d += (double) iq[h * id + i] * (double) kb[i];
        s += fmaxf((float) d, 0.0f);
    }
    scores[b] = s / sqrtf((float) id);
}

// ------------------------------------------------------------------------------------------------ MoE glue
// out = silu(g) * u
extern "C" __global__ void swiglu(const float* g, const float* u, float* out, unsigned int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = siluf_ref(g[i]) * u[i];
}

// out[j*ff + c] = silu(g[j*ff + c] * gs[j]) * (u[j*ff + c] * us[j]) for expert j in 0..k, channel c in 0..ff
extern "C" __global__ void swiglu_scaled(const float* g, const float* u, const float* gs, const float* us,
                                         float* out, unsigned int ff, unsigned int k) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= ff * k) return;
    unsigned int j = i / ff;
    out[i] = siluf_ref(g[i] * gs[j]) * (u[i] * us[j]);
}

// y += a * x
extern "C" __global__ void axpy(float* y, const float* x, unsigned int n, float a) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] += a * x[i];
}

// y += sum_j w[j] * x[j * n ..], j = 0, 1, ... k-1 in that order: k axpy calls in one launch (the routed experts'
// outputs, weighted, into the MoE sum), with the same per-element arithmetic as axpy.
extern "C" __global__ void axpy_many(float* y, const float* x, const float* w, unsigned int n, unsigned int k) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float acc = y[i];
    for (unsigned int j = 0; j < k; j++) acc += w[j] * x[(u64) j * n + i];
    y[i] = acc;
}

// y += sum_j w[j] * (x[j * n ..] * s[j]), j = 0, 1, ... k-1 in that order (with per-expert tensor scale s[j])
extern "C" __global__ void axpy_many_scaled(float* y, const float* x, const float* w, const float* s,
                                            unsigned int n, unsigned int k) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float acc = y[i];
    for (unsigned int j = 0; j < k; j++) acc += w[j] * (x[(u64) j * n + i] * s[j]);
    y[i] = acc;
}

// y[i] = (sum_j w[j] * (eo[j * n + i] * s[j])) + sh_eo[i]: zeroes, accumulates routed experts, adds shared expert
extern "C" __global__ void moe_combine_flint(float* y, const float* eo, const float* w, const float* s,
                                             const float* sh_eo, unsigned int n, unsigned int k) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float acc = 0.0f;
    for (unsigned int j = 0; j < k; j++) acc += w[j] * (eo[(u64) j * n + i] * s[j]);
    y[i] = acc + sh_eo[i];
}

// y += sigmoid(gl[0]) * x  (the shared expert's gate, a one-row matvec already in gl)
extern "C" __global__ void axpy_sigmoid(float* y, const float* x, const float* gl, unsigned int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] += sigmoidf_ref(gl[0]) * x[i];
}

// Fuses my.zero() + axpy_many + axpy_sigmoid + copy into one launch for Qwen's MoE combine.
extern "C" __global__ void moe_combine_qwen(float* y, const float* eo, const float* w, const float* sh_eo,
                                            const float* gl, unsigned int n, unsigned int k) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float acc = 0.0f;
    for (unsigned int j = 0; j < k; j++) acc += w[j] * eo[(u64) j * n + i];
    y[i] = acc + sigmoidf_ref(gl[0]) * sh_eo[i];
}

// moe_combine_qwen for tokens j = blockIdx.y: y rows n apart, eo rows k * n, weights rows k, shared rows n, gate
// logits rows 1, and token j's expert count cnt[j] (<= k).  grid = (ceil(n / 256), rows).
extern "C" __global__ void moe_combine_qwen_rows(float* y, const float* eo, const float* w, const float* sh_eo,
                                                 const float* gl, const unsigned int* cnt, unsigned int n, unsigned int k) {
    const u64 j = blockIdx.y;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float* e = eo + j * k * n;
    const float* wj = w + j * k;
    float acc = 0.0f;
    for (unsigned int t = 0; t < cnt[j]; t++) acc += wj[t] * e[(u64) t * n + i];
    y[j * n + i] = acc + sigmoidf_ref(gl[j]) * sh_eo[j * n + i];
}

// moe_combine_qwen with eo's rows in another order: expert j's output is row perm[j] (the sum still runs over j).
extern "C" __global__ void moe_combine_qwen_perm(float* y, const float* eo, const float* w, const unsigned int* perm,
                                                 const float* sh_eo, const float* gl, unsigned int n, unsigned int k) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float acc = 0.0f;
    for (unsigned int j = 0; j < k; j++) acc += w[j] * eo[(u64) perm[j] * n + i];
    y[i] = acc + sigmoidf_ref(gl[0]) * sh_eo[i];
}

// ------------------------------------------------------------------------------------------------ PLE
// Per stream s (one block each): g = (key_s . qry_s) / sqrt(h) in double, signed square root, sigmoid; gv_s = sg * val.
extern "C" __global__ void ple_gate(const float* key, const float* qry, const float* val, float* gv, unsigned int h) {
    unsigned int s = blockIdx.x;
    double dot = 0.0;
    for (unsigned int i = threadIdx.x; i < h; i += blockDim.x) dot += (double) key[s * h + i] * (double) qry[s * h + i];
    dot = block_sum(dot) / sqrt((double) h);
    double g = sqrt(fmax(fabs(dot), 1e-6)) * copysign(1.0, dot);   // Rust's signum: +0 -> 1, -0 -> -1
    float sg = sigmoidf_ref((float) g);
    for (unsigned int i = threadIdx.x; i < h; i += blockDim.x) gv[s * h + i] = sg * val[i];
}

// out[i] += silu(depthwise causal conv, kernel k, dilation dil, over gvn and its state of the last (k-1)*dil inputs);
// the state shifts by one and takes gvn.  One thread per channel.
extern "C" __global__ void ple_conv_step(const float* gvn, float* state, const float* w, float* out, unsigned int ch,
                                         unsigned int k, unsigned int dil) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= ch) return;
    unsigned int hist = (k - 1) * dil;
    double acc = (double) w[i * k + (k - 1)] * (double) gvn[i];
    for (unsigned int t = 0; t + 1 < k; ++t) {
        unsigned int back = (k - 1 - t) * dil;
        acc += (double) w[i * k + t] * (double) state[(hist - back) * ch + i];
    }
    out[i] += siluf_ref((float) acc);
    for (unsigned int t = 0; t + 1 < hist; ++t) state[t * ch + i] = state[(t + 1) * ch + i];
    state[(hist - 1) * ch + i] = gvn[i];
}

// ------------------------------------------------------------------------------------------------ draft argmax
// The draft head's pick over a token subset (ids, or 0..n-1 when null): out[0] = the id with the largest logit (the
// smallest id on a tie), out[1] = its softmax probability over the subset.  One block of 1024 threads.
extern "C" __global__ void argmax_subset(const float* logits, const unsigned int* ids, unsigned int n, float* out) {
    __shared__ double bv[1024];
    __shared__ unsigned int bi[1024];
    double best = -1e300;
    unsigned int bid = 0xffffffffu;
    for (unsigned int i = threadIdx.x; i < n; i += blockDim.x) {
        unsigned int id = ids ? ids[i] : i;
        double v = logits[id];
        if (v > best || (v == best && id < bid)) { best = v; bid = id; }
    }
    bv[threadIdx.x] = best;
    bi[threadIdx.x] = bid;
    __syncthreads();
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            double v = bv[threadIdx.x + s];
            unsigned int id = bi[threadIdx.x + s];
            if (v > bv[threadIdx.x] || (v == bv[threadIdx.x] && id < bi[threadIdx.x])) { bv[threadIdx.x] = v; bi[threadIdx.x] = id; }
        }
        __syncthreads();
    }
    const double top = bv[0];
    const unsigned int tid = bi[0];
    __syncthreads();
    double z = 0.0;
    for (unsigned int i = threadIdx.x; i < n; i += blockDim.x) z += exp((double) logits[ids ? ids[i] : i] - top);
    z = block_sum(z);
    if (threadIdx.x == 0) {
        out[0] = (float) tid;
        out[1] = (float) (1.0 / z);
    }
}

// argmax_subset over logits computed only for the subset: logits[i] belongs to token ids[i]; out[0] is the token id.
extern "C" __global__ void argmax_compact(const float* logits, const unsigned int* ids, unsigned int n, float* out) {
    __shared__ double bv[1024];
    __shared__ unsigned int bi[1024];
    double best = -1e300;
    unsigned int bid = 0xffffffffu;
    for (unsigned int i = threadIdx.x; i < n; i += blockDim.x) {
        double v = logits[i];
        unsigned int id = ids[i];
        if (v > best || (v == best && id < bid)) { best = v; bid = id; }
    }
    bv[threadIdx.x] = best;
    bi[threadIdx.x] = bid;
    __syncthreads();
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            double v = bv[threadIdx.x + s];
            unsigned int id = bi[threadIdx.x + s];
            if (v > bv[threadIdx.x] || (v == bv[threadIdx.x] && id < bi[threadIdx.x])) { bv[threadIdx.x] = v; bi[threadIdx.x] = id; }
        }
        __syncthreads();
    }
    const double top = bv[0];
    const unsigned int tid = bi[0];
    __syncthreads();
    double z = 0.0;
    for (unsigned int i = threadIdx.x; i < n; i += blockDim.x) z += exp((double) logits[i] - top);
    z = block_sum(z);
    if (threadIdx.x == 0) {
        out[0] = (float) tid;
        out[1] = (float) (1.0 / z);
    }
}

// ------------------------------------------------------------------------------------------------ DFlash 2 drafter
// A 5-layer block-diffusion drafter (docs/STATUS.md): BF16 weights (the top 16 bits of a float), float activations.
static __device__ __forceinline__ float bf16f(const unsigned short* p, u64 i) { return __uint_as_float(((unsigned int) p[i]) << 16); }

// feat[row][slot * h + i] = mean over the cs streams of the residual r[row][s * h + i]; grid = (ceil(h / 256), rows)
extern "C" __global__ void df_capture(const float* r, float* feat, unsigned int slot, unsigned int h, unsigned int cs,
                                      unsigned int nfeat) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y;
    if (i >= h) return;
    const float* rj = r + (u64) j * h * cs;
    float s = 0.0f;
    for (unsigned int k = 0; k < cs; ++k) s += rj[(u64) k * h + i];
    feat[(u64) j * nfeat + slot * h + i] = s / (float) cs;
}

// y[row] = rmsnorm(x[row]) * w (BF16, shared by the rows); grid = rows, block 256.  In place is fine.
extern "C" __global__ void df_rmsnorm(const float* x, const unsigned short* w, float* y, unsigned int n, float eps) {
    const float* xr = x + (u64) blockIdx.x * n;
    float* yr = y + (u64) blockIdx.x * n;
    double ss = 0.0;
    for (unsigned int i = threadIdx.x; i < n; i += blockDim.x) ss += (double) xr[i] * (double) xr[i];
    double r = 1.0 / sqrt(block_sum(ss) / (double) n + (double) eps);
    for (unsigned int i = threadIdx.x; i < n; i += blockDim.x) yr[i] = (float) ((double) xr[i] * r) * bf16f(w, i);
}

// res[row] += d[row]; y[row] = rmsnorm(res[row]) * w; grid = rows, block 256
extern "C" __global__ void df_add_rmsnorm(float* res, const float* d, const unsigned short* w, float* y, unsigned int n,
                                          float eps) {
    float* rr = res + (u64) blockIdx.x * n;
    const float* dr = d + (u64) blockIdx.x * n;
    float* yr = y + (u64) blockIdx.x * n;
    double ss = 0.0;
    for (unsigned int i = threadIdx.x; i < n; i += blockDim.x) {
        float v = rr[i] + dr[i];
        rr[i] = v;
        ss += (double) v * (double) v;
    }
    double r = 1.0 / sqrt(block_sum(ss) / (double) n + (double) eps);
    for (unsigned int i = threadIdx.x; i < n; i += blockDim.x) yr[i] = (float) ((double) rr[i] * r) * bf16f(w, i);
}

// The grouped dynamic 2-tap convolution along a block's rows (DFlashGroupedConv): with the kernel projection's output
// delta (row t at delta + t * 4 * ngroups: [side][tap][group]) and base [side][tap][channel],
// y[t][c] = (base0[c] + d0[g]) x[t][c] + (t >= 1) (base1[c] + d1[g]) x[t-1][c], g = c / group.  grid = (ceil(h / 256), rows)
extern "C" __global__ void df_conv(const float* x, const float* delta, const unsigned short* base, float* y, unsigned int side,
                                   unsigned int h, unsigned int group, unsigned int ngroups) {
    unsigned int c = blockIdx.x * blockDim.x + threadIdx.x, t = blockIdx.y;
    if (c >= h) return;
    unsigned int g = c / group;
    const float* d = delta + (u64) t * (4 * ngroups) + side * 2 * ngroups;
    float c0 = bf16f(base, (u64) (side * 2 + 0) * h + c) + d[g];
    float out = c0 * x[(u64) t * h + c];
    if (t >= 1) {
        float c1 = bf16f(base, (u64) (side * 2 + 1) * h + c) + d[ngroups + g];
        out += c1 * x[(u64) (t - 1) * h + c];
    }
    y[(u64) t * h + c] = out;
}

// NeoX RoPE over full 128-wide heads, row j at position pos0 + j; grid = (heads, rows), block 64
extern "C" __global__ void df_rope(float* x, unsigned int pos0, unsigned int stride, float theta) {
    float* xr = x + (u64) blockIdx.y * stride + blockIdx.x * 128;
    unsigned int i = threadIdx.x;
    float f = (float) (pos0 + blockIdx.y) * powf(theta, -(float) i / 64.0f);
    float cs, sn;
    sincosf(f, &sn, &cs);
    float a = xr[i], b = xr[i + 64];
    xr[i] = a * cs - b * sn;
    xr[i + 64] = b * cs + a * sn;
}

// ring[((pos0 + j) % cap) * width ..] = src[j * width ..]; grid = (ceil(width / 256), rows)
extern "C" __global__ void df_ring_put(const float* src, float* ring, unsigned int pos0, unsigned int cap, unsigned int width) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= width) return;
    ring[(u64) ((pos0 + blockIdx.y) % cap) * width + i] = src[(u64) blockIdx.y * width + i];
}

// g[i] = silu(g[i]) * u[i]
extern "C" __global__ void df_silu_mul(float* g, const float* u, unsigned int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) g[i] = siluf_ref(g[i]) * u[i];
}

// Attention of the block's `rows` queries (32 heads of 128, 8 KV heads of 128) over the keys of positions
// [max(0, qpos + 1 - window), n + rows) kept in the rings (position p in slot p % ring): the block sees all of itself.
// grid = (32, rows), block 128, dynamic shared = (window + rows) floats
extern "C" __global__ void df_attention(const float* q, const float* kc, const float* vc, float* out, unsigned int n,
                                        unsigned int rows, unsigned int window, unsigned int ring, float scale) {
    extern __shared__ float sc[];
    __shared__ float qv[128];
    __shared__ float red[128];
    unsigned int h = blockIdx.x, t = blockIdx.y, kvh = h / 4, d = threadIdx.x;
    unsigned int qpos = n + t, hi = n + rows - 1;
    unsigned int lo = qpos + 1 > window ? qpos + 1 - window : 0;
    qv[d] = q[(u64) t * 4096 + h * 128 + d];
    __syncthreads();
    unsigned int len = hi - lo + 1;
    unsigned int warp = d / 32, lane = d % 32;
    for (unsigned int j = warp; j < len; j += 4) {
        const float* kr = kc + (u64) ((lo + j) % ring) * 1024 + kvh * 128;
        float s = 0.0f;
        for (unsigned int i = lane; i < 128; i += 32) s += qv[i] * kr[i];
        for (int off = 16; off > 0; off >>= 1) s += __shfl_down_sync(0xffffffffu, s, off);
        if (lane == 0) sc[j] = s * scale;
    }
    __syncthreads();
    float mx = -3.0e38f;
    for (unsigned int j = d; j < len; j += 128) mx = fmaxf(mx, sc[j]);
    red[d] = mx;
    __syncthreads();
    for (unsigned int s = 64; s > 0; s >>= 1) {
        if (d < s) red[d] = fmaxf(red[d], red[d + s]);
        __syncthreads();
    }
    mx = red[0];
    __syncthreads();
    float sm = 0.0f;
    for (unsigned int j = d; j < len; j += 128) {
        float e = expf(sc[j] - mx);
        sc[j] = e;
        sm += e;
    }
    red[d] = sm;
    __syncthreads();
    for (unsigned int s = 64; s > 0; s >>= 1) {
        if (d < s) red[d] += red[d + s];
        __syncthreads();
    }
    sm = red[0];
    float acc = 0.0f;
    for (unsigned int j = 0; j < len; ++j) acc += sc[j] * vc[(u64) ((lo + j) % ring) * 1024 + kvh * 128 + d];
    out[(u64) t * 4096 + h * 128 + d] = acc / sm;
}

// The 16 largest of each row of logits (vocab values, rows ls apart), largest first, ties to the lower id; with `map`
// the logits are over the vocabulary rows map[i] and the ids written are map[i].  grid = rows, block 256.
extern "C" __global__ void df_top16(const float* logits, unsigned int vocab, unsigned int ls, const unsigned int* map,
                                    unsigned int* ids, float* vals) {
    __shared__ float sv[256 * 16];
    __shared__ unsigned int si[256 * 16];
    __shared__ unsigned int pos[256];
    __shared__ float bv[256];
    __shared__ unsigned int bi[256], bt[256];
    const float* row = logits + (u64) blockIdx.x * ls;
    float lv[16];
    unsigned int li[16];
#pragma unroll
    for (int k = 0; k < 16; ++k) { lv[k] = -3.0e38f; li[k] = 0xffffffffu; }
    for (unsigned int i = threadIdx.x; i < vocab; i += 256) {
        float v = row[i];
        if (v > lv[15]) {
            lv[15] = v;
            li[15] = i;
#pragma unroll
            for (int k = 15; k > 0; --k) {
                if (lv[k] > lv[k - 1]) {
                    float tv = lv[k]; lv[k] = lv[k - 1]; lv[k - 1] = tv;
                    unsigned int ti = li[k]; li[k] = li[k - 1]; li[k - 1] = ti;
                }
            }
        }
    }
#pragma unroll
    for (int k = 0; k < 16; ++k) { sv[threadIdx.x * 16 + k] = lv[k]; si[threadIdx.x * 16 + k] = li[k]; }
    pos[threadIdx.x] = 0;
    __syncthreads();
    for (int r = 0; r < 16; ++r) {
        unsigned int p = pos[threadIdx.x];
        bv[threadIdx.x] = p < 16 ? sv[threadIdx.x * 16 + p] : -3.0e38f;
        bi[threadIdx.x] = p < 16 ? si[threadIdx.x * 16 + p] : 0xffffffffu;
        bt[threadIdx.x] = threadIdx.x;
        __syncthreads();
        for (unsigned int s = 128; s > 0; s >>= 1) {
            if (threadIdx.x < s) {
                float a = bv[threadIdx.x], b = bv[threadIdx.x + s];
                unsigned int ia = bi[threadIdx.x], ib = bi[threadIdx.x + s];
                if (b > a || (b == a && ib < ia)) {
                    bv[threadIdx.x] = b; bi[threadIdx.x] = ib; bt[threadIdx.x] = bt[threadIdx.x + s];
                }
            }
            __syncthreads();
        }
        if (threadIdx.x == 0) {
            ids[blockIdx.x * 16 + r] = map ? map[bi[0]] : bi[0];
            vals[blockIdx.x * 16 + r] = bv[0];
            pos[bt[0]] += 1;
        }
        __syncthreads();
    }
}

// The selector's greedy walk (DFlash2's CandidateSelector): slots e = 0 .. slots-1, 16 candidates cand[e][c] with
// logits unary[e][c] and projected hidden hp[e][rank].  Slot e scores candidate c as
//   unary[e][c] + < pred[p_e] * hp[e], succ[cand[e][c]] >,   p_0 = anchor, p_e = the token chosen for slot e - 1,
// and takes the best.  One block of 512 threads (a warp a candidate); out[e] = the token chosen, out[slots + e] = the
// float bits of its probability among the 16 (softmax of the scores).
extern "C" __global__ void df_walk(const unsigned int* cand, const float* unary, const float* hp, const unsigned short* pred,
                                   const unsigned short* succ, unsigned int anchor, unsigned int slots, unsigned int rank,
                                   unsigned int* out) {
    __shared__ float score[16];
    __shared__ float pv[256];
    __shared__ unsigned int ptok;
    unsigned int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    if (threadIdx.x == 0) ptok = anchor;
    __syncthreads();
    for (unsigned int e = 0; e < slots; ++e) {
        for (unsigned int i = threadIdx.x; i < rank; i += 512) pv[i] = bf16f(pred, (u64) ptok * rank + i) * hp[e * rank + i];
        __syncthreads();
        unsigned int tok = cand[e * 16 + warp];
        float s = 0.0f;
        for (unsigned int i = lane; i < rank; i += 32) s += pv[i] * bf16f(succ, (u64) tok * rank + i);
        for (int off = 16; off > 0; off >>= 1) s += __shfl_down_sync(0xffffffffu, s, off);
        if (lane == 0) score[warp] = unary[e * 16 + warp] + s;
        __syncthreads();
        if (threadIdx.x == 0) {
            unsigned int best = 0;
            for (unsigned int c = 1; c < 16; ++c) if (score[c] > score[best]) best = c;
            ptok = cand[e * 16 + best];
            out[e] = ptok;
            float z = 0.0f;
            for (unsigned int c = 0; c < 16; ++c) z += expf(score[c] - score[best]);
            out[slots + e] = __float_as_uint(1.0f / z);
        }
        __syncthreads();
    }
}

// The top `keep` blocks of each row's index scores, as the token lists `attention_*_sel_rows` read: what the host did after a download of the
// scores (select_nth by score, best first, then lower block first; the chosen blocks' tokens in ascending order, then the tokens of the
// incomplete block).  Row r = blockIdx.x has n = n0 + r tokens, nb = n / ratio complete blocks (nb >= keep).  toks rows `tstride` words apart,
// lens[r] = the row's token count.  Scores are not NaN.  block = 256 threads.
extern "C" __global__ void select_blocks_rows(const float* scores, unsigned int* toks, unsigned int* lens, unsigned int sc_rs,
                                              unsigned int tstride, unsigned int n0, unsigned int ratio, unsigned int keep) {
    __shared__ unsigned int cnt;
    __shared__ unsigned int scan[256];
    const u64 r = blockIdx.x;
    const unsigned int tid = threadIdx.x;
    const unsigned int n = n0 + (unsigned int) r;
    const unsigned int nb = n / ratio;
    const float* sc = scores + r * sc_rs;
    unsigned int* out = toks + r * tstride;
    // smaller key = better (higher score); -0.0 is 0.0 as in the host's comparison
    #define SEL_KEY(b, d) { float v_ = sc[b]; if (v_ == 0.0f) v_ = 0.0f; unsigned int u_ = __float_as_uint(v_); u_ = (u_ & 0x80000000u) ? ~u_ : (u_ | 0x80000000u); d = ~u_; }
    // the keep-th smallest key by radix selection: `prefix` ends as its value, k as how many of the keys equal to it are taken (lowest blocks first)
    unsigned int prefix = 0, k = keep;
    for (int bit = 31; bit >= 0; --bit) {
        if (tid == 0) cnt = 0;
        __syncthreads();
        const unsigned int hi = bit == 31 ? 0u : (0xFFFFFFFFu << (bit + 1));
        unsigned int local = 0;
        for (unsigned int b = tid; b < nb; b += 256) {
            unsigned int d;
            SEL_KEY(b, d);
            if ((d & hi) == prefix && !((d >> bit) & 1u)) ++local;
        }
        atomicAdd(&cnt, local);
        __syncthreads();
        const unsigned int c = cnt;
        __syncthreads();
        if (c < k) {
            k -= c;
            prefix |= 1u << bit;
        }
    }
    unsigned int eq_before = 0, ch_before = 0;
    for (unsigned int c0 = 0; c0 < nb; c0 += 256) {
        const unsigned int b = c0 + tid;
        unsigned int d = 0;
        if (b < nb) { SEL_KEY(b, d); }
        const unsigned int lt = b < nb && d < prefix;
        const unsigned int eq = b < nb && d == prefix;
        scan[tid] = eq;
        __syncthreads();
        for (unsigned int off = 1; off < 256; off <<= 1) {
            const unsigned int v = tid >= off ? scan[tid - off] : 0u;
            __syncthreads();
            scan[tid] += v;
            __syncthreads();
        }
        const unsigned int eq_incl = scan[tid], eq_total = scan[255];
        __syncthreads();
        const unsigned int chosen = lt || (eq && eq_before + eq_incl - eq < k);
        scan[tid] = chosen;
        __syncthreads();
        for (unsigned int off = 1; off < 256; off <<= 1) {
            const unsigned int v = tid >= off ? scan[tid - off] : 0u;
            __syncthreads();
            scan[tid] += v;
            __syncthreads();
        }
        const unsigned int ch_incl = scan[tid], ch_total = scan[255];
        if (chosen) {
            const unsigned int q = ch_before + ch_incl - 1;
            for (unsigned int i = 0; i < ratio; ++i) out[q * ratio + i] = b * ratio + i;
        }
        eq_before += eq_total;
        ch_before += ch_total;
        __syncthreads();
    }
    #undef SEL_KEY
    const unsigned int tn = n - nb * ratio;
    for (unsigned int t = tid; t < tn; t += 256) out[keep * ratio + t] = nb * ratio + t;
    if (tid == 0) lens[r] = keep * ratio + tn;
}

// `index_scores_rows` with the dot products in float (`--tensor-prompt`'s inexact family): the double version is limited by the card's FP64 rate (1/64 of FP32).
extern "C" __global__ void index_scores_rows_f32(const float* iq, const float* blocks, float* scores, unsigned int n0,
                                                 unsigned int ratio, unsigned int heads, unsigned int id,
                                                 unsigned int iq_rs, unsigned int sc_rs) {
    const u64 r = blockIdx.y;
    unsigned int nb = (n0 + (unsigned int) r) / ratio;
    unsigned int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= nb) return;
    const float* q = iq + r * iq_rs;
    const float* kb = blocks + (u64) b * id;
    float s = 0.0f;
    for (unsigned int h = 0; h < heads; ++h) {
        float d = 0.0f;
        for (unsigned int i = 0; i < id; ++i) d = fmaf(q[h * id + i], kb[i], d);
        s += fmaxf(d, 0.0f);
    }
    scores[r * sc_rs + b] = s / sqrtf((float) id);
}

// `index_scores_rows_f32` register-blocked (Nsight Compute: L1/TEX at 98.6%, two loads for every FMA): a block of 128 threads scores 4 rows, a thread 2 compressed blocks of
// keys (b and b + 128), so each key float4 serves 4 rows and each query float4 (shared memory, a broadcast) serves 2 blocks: 8 FMAs a shared load.  Every dot product is
// summed in the same order as the one-thread-a-block kernel (i = 0 .. id - 1 by fmaf), so the scores are bit for bit the same.  heads <= 4, id a multiple of 4 and <= 256;
// the launch is (ceil(max_nb / 256), ceil(rows / 4)) blocks of 128 threads.
extern "C" __global__ void __launch_bounds__(128) index_scores_rows_f32v2(const float* iq, const float* blocks, float* scores, unsigned int n0,
                                                 unsigned int ratio, unsigned int heads, unsigned int id,
                                                 unsigned int iq_rs, unsigned int sc_rs, unsigned int rows) {
    __shared__ __align__(16) float qs[4][4 * 256];
    const unsigned int tid = threadIdx.x;
    const unsigned int r0 = blockIdx.y * 4;
    const unsigned int nr = min(4u, rows - r0);
    for (unsigned int idx = tid; idx < 4 * heads * id; idx += 128) {
        const unsigned int r = idx / (heads * id), k = idx % (heads * id);
        qs[r][k] = r < nr ? iq[(u64) (r0 + r) * iq_rs + k] : 0.0f;
    }
    __syncthreads();
    unsigned int nbr[4];
    unsigned int nb_max = 0;
    #pragma unroll
    for (unsigned int r = 0; r < 4; ++r) {
        nbr[r] = r < nr ? (n0 + r0 + r) / ratio : 0u;
        nb_max = max(nb_max, nbr[r]);
    }
    const unsigned int bA = blockIdx.x * 256 + tid, bB = bA + 128;
    if (bA >= nb_max) return;
    const bool hasB = bB < nb_max;
    const float4* kA = (const float4*) (blocks + (u64) bA * id);
    const float4* kB = (const float4*) (blocks + (u64) (hasB ? bB : bA) * id);
    float dA[4][4], dB[4][4];                                         // [row][head]
    #pragma unroll
    for (unsigned int r = 0; r < 4; ++r)
        #pragma unroll
        for (unsigned int h = 0; h < 4; ++h) { dA[r][h] = 0.0f; dB[r][h] = 0.0f; }
    for (unsigned int i4 = 0; i4 < id / 4; ++i4) {
        const float4 a = kA[i4], b = kB[i4];
        #pragma unroll
        for (unsigned int h = 0; h < 4; ++h) {
            if (h < heads) {
                #pragma unroll
                for (unsigned int r = 0; r < 4; ++r) {
                    const float4 q = *(const float4*) &qs[r][h * id + 4 * i4];
                    dA[r][h] = fmaf(q.x, a.x, dA[r][h]); dA[r][h] = fmaf(q.y, a.y, dA[r][h]); dA[r][h] = fmaf(q.z, a.z, dA[r][h]); dA[r][h] = fmaf(q.w, a.w, dA[r][h]);
                    dB[r][h] = fmaf(q.x, b.x, dB[r][h]); dB[r][h] = fmaf(q.y, b.y, dB[r][h]); dB[r][h] = fmaf(q.z, b.z, dB[r][h]); dB[r][h] = fmaf(q.w, b.w, dB[r][h]);
                }
            }
        }
    }
    const float inv = sqrtf((float) id);
    #pragma unroll
    for (unsigned int r = 0; r < 4; ++r) {
        if (r < nr) {
            float sA = 0.0f, sB = 0.0f;
            #pragma unroll
            for (unsigned int h = 0; h < 4; ++h) {
                if (h < heads) { sA += fmaxf(dA[r][h], 0.0f); sB += fmaxf(dB[r][h], 0.0f); }
            }
            if (bA < nbr[r]) scores[(u64) (r0 + r) * sc_rs + bA] = sA / inv;
            if (hasB && bB < nbr[r]) scores[(u64) (r0 + r) * sc_rs + bB] = sB / inv;
        }
    }
}

// ---------------------------------------------------------------------------------------------------------------------------------
// Device-side routing and planning of a long prompt's MoE (docs/PREFILL2_SPEC.md, stage 1): what the host did between the router's logits arriving and the
// grouped expert kernels launching, so the host need not wait for the logits and can queue a whole pass ahead.

extern "C" __global__ void zero_words(unsigned int* p, unsigned int n) {
    for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) p[i] = 0u;
}

extern "C" __global__ void fill_words(unsigned int* p, unsigned int n, unsigned int v) {
    for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) p[i] = v;
}

// `shadow_model::forward::route` for token j = one thread: the f64 exponentials of (logit - max), their sum in index order, the top k by (score descending, index
// ascending) kept by insertion, the weights (ex / z) / (sum of the top k's ex / z).  Writes the k expert numbers (tidx) and weights (as f32, the form the combine
// reads) of token j and counts the token among each chosen expert's users (cnt).  The CUDA exp() is not the host library's: weights can differ in the last bit of
// a double, which is below an f32's rounding except by chance, so this is a fast-mode kernel, not an `--exact` one.
// the one-thread-a-token router (`SHADOW_ROUTE_1T=1`)
extern "C" __global__ void route_topk_rows_1t(const float* logits, unsigned int* tidx, float* weights, unsigned int* cnt, unsigned int ne, unsigned int k,
                                           unsigned int m) {
    const unsigned int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= m) return;
    const float* lg = logits + (u64) j * ne;
    float mx = -3.402823466e+38f;
    for (unsigned int i = 0; i < ne; ++i) mx = fmaxf(mx, lg[i]);
    double ex[512];
    double z = 0.0;
    for (unsigned int i = 0; i < ne; ++i) {
        ex[i] = exp((double) (lg[i] - mx));
        z += ex[i];
    }
    unsigned int top[16];
    unsigned int n = 0;
    for (unsigned int i = 0; i < ne; ++i) {
        // before(a, b): a ranks ahead of b
        if (n == k && !(ex[i] > ex[top[k - 1]] || (ex[i] == ex[top[k - 1]] && i < top[k - 1]))) continue;
        unsigned int at = 0;
        while (at < n && (ex[top[at]] > ex[i] || (ex[top[at]] == ex[i] && top[at] < i))) ++at;
        if (n < k) ++n;
        for (unsigned int t = n - 1; t > at; --t) top[t] = top[t - 1];
        top[at] = i;
    }
    double tz = 0.0;
    for (unsigned int t = 0; t < n; ++t) tz += ex[top[t]] / z;
    for (unsigned int t = 0; t < n; ++t) {
        tidx[(u64) j * k + t] = top[t];
        weights[(u64) j * k + t] = (float) ((ex[top[t]] / z) / tz);
        atomicAdd(&cnt[top[t]], 1u);
    }
}

extern "C" __global__ void __launch_bounds__(256) route_topk_rows(const float* logits, unsigned int* tidx, float* weights, unsigned int* cnt, unsigned int ne, unsigned int k,
                                           unsigned int m) {
    // a warp a token (blocks of 8 warps; ne <= 512, k <= 16): the logits' max and the exponentials in parallel, the sum z sequentially in index order by lane 0
    // (the same double sum as one thread gave), the top k by k rounds of a warp-wide best-of (greater score, then smaller index: the order the insertion kept)
    __shared__ double exs[8][512];
    const unsigned int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    const unsigned int j = blockIdx.x * 8 + w;
    if (j >= m) return;                               // whole warps leave together
    double* ex = exs[w];
    const float* lg = logits + (u64) j * ne;
    float mx = -3.402823466e+38f;
    for (unsigned int i = lane; i < ne; i += 32) mx = fmaxf(mx, lg[i]);
    #pragma unroll
    for (unsigned int o = 16; o > 0; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
    for (unsigned int i = lane; i < ne; i += 32) ex[i] = exp((double) (lg[i] - mx));
    __syncwarp();
    double z = 0.0;
    if (lane == 0) {
        for (unsigned int i = 0; i < ne; ++i) z += ex[i];
    }
    z = __shfl_sync(0xffffffffu, z, 0);
    unsigned int taken = 0;                           // bit q: expert lane + 32 q chosen
    unsigned int top[16];
    for (unsigned int t = 0; t < k; ++t) {
        double bv = -1.0;
        unsigned int bi = 0xffffffffu;
        for (unsigned int q = 0; q * 32 + lane < ne; ++q) {
            if (taken >> q & 1u) continue;
            const double v = ex[q * 32 + lane];
            const unsigned int i = q * 32 + lane;
            if (v > bv || (v == bv && i < bi)) { bv = v; bi = i; }
        }
        #pragma unroll
        for (unsigned int o = 16; o > 0; o >>= 1) {
            const double ov = __shfl_xor_sync(0xffffffffu, bv, o);
            const unsigned int oi = __shfl_xor_sync(0xffffffffu, bi, o);
            if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
        }
        top[t] = bi;
        if ((bi & 31u) == lane) taken |= 1u << (bi >> 5);
    }
    double tz = 0.0;
    for (unsigned int t = 0; t < k; ++t) tz += ex[top[t]] / z;
    if (lane < k) {
        // top[] is the same on every lane; lane t writes the t-th
        unsigned int mine = top[0];
        #pragma unroll
        for (unsigned int t = 1; t < 16; ++t) if (t == lane) mine = top[t];
        tidx[(u64) j * k + lane] = mine;
        weights[(u64) j * k + lane] = (float) ((ex[mine] / z) / tz);
        atomicAdd(&cnt[mine], 1u);
    }
}

// One block (ne <= 1024 threads): the first group of each expert within its phase.  An expert's cnt tokens make ceil(cnt / 32) groups; phase 0 holds the experts
// with copied[e] == 0 (resident in the VRAM cache), phase 1 the others, each in expert order.
extern "C" __global__ void plan_scan(const unsigned int* cnt, const unsigned char* copied, unsigned int* gstart, unsigned int ne) {
    __shared__ unsigned int s0[1024], s1[1024];
    const unsigned int e = threadIdx.x;
    unsigned int g = e < ne ? (cnt[e] + 31u) / 32u : 0u;
    const unsigned int c = e < ne ? copied[e] : 0u;
    s0[e] = c ? 0u : g;
    s1[e] = c ? g : 0u;
    __syncthreads();
    for (unsigned int off = 1; off < 1024; off <<= 1) {
        const unsigned int a = e >= off ? s0[e - off] : 0u, b = e >= off ? s1[e - off] : 0u;
        __syncthreads();
        s0[e] += a;
        s1[e] += b;
        __syncthreads();
    }
    if (e < ne) gstart[e] = c ? s1[e] - g : s0[e] - g;
}

// One block (256 threads) an expert: its tokens in token order, 32 a group, written as the records and matrix addresses `mma_grouped_*` read (65 u32 a record:
// count, 32 input rows, 32 output rows; part 0 = gate: in row j, out row pair; part 1 = up: in j, out up_off + pair; part 2 = down: in pair, out pair;
// pair = j * k + the expert's rank in token j's list).  `base` + a0 / a1 the addresses of phase 0 / 1 (3 parts x G entries of 8 bytes), + r0 / r1 the records.
// The records' counts were zeroed, so groups past the last one are empty and the kernels skip them.
extern "C" __global__ void plan_fill(const unsigned int* tidx, const unsigned int* cnt, const unsigned int* gstart, const unsigned char* copied,
                                     const u64* eaddr, unsigned char* base, unsigned int ne, unsigned int k, unsigned int m, unsigned int G,
                                     unsigned int up_off, unsigned int a0, unsigned int a1, unsigned int r0, unsigned int r1) {
    __shared__ unsigned int scan[256];
    const unsigned int e = blockIdx.x, tid = threadIdx.x;
    const unsigned int total = cnt[e];
    if (total == 0) return;
    const unsigned int ph = copied[e];
    u64* addr = (u64*) (base + (ph ? a1 : a0));
    unsigned int* rec = (unsigned int*) (base + (ph ? r1 : r0));
    const unsigned int g0 = gstart[e];
    unsigned int seen = 0;
    for (unsigned int t0 = 0; t0 < m; t0 += 256) {
        const unsigned int j = t0 + tid;
        int rank = -1;
        if (j < m) {
            for (unsigned int i = 0; i < k; ++i) if (tidx[(u64) j * k + i] == e) { rank = (int) i; break; }
        }
        scan[tid] = rank >= 0 ? 1u : 0u;
        __syncthreads();
        for (unsigned int off = 1; off < 256; off <<= 1) {
            const unsigned int v = tid >= off ? scan[tid - off] : 0u;
            __syncthreads();
            scan[tid] += v;
            __syncthreads();
        }
        const unsigned int incl = scan[tid], chunk_total = scan[255];
        if (rank >= 0) {
            const unsigned int p = seen + incl - 1, g = g0 + p / 32, s = p % 32;
            const unsigned int pair = j * k + (unsigned int) rank;
            for (unsigned int part = 0; part < 3; ++part) {
                unsigned int* r = rec + ((u64) part * G + g) * 65;
                r[1 + s] = part == 2 ? pair : j;
                r[33 + s] = part == 1 ? up_off + pair : pair;
                if (s == 0) {
                    r[0] = min(32u, total - 32u * (p / 32));
                    addr[(u64) part * G + g] = eaddr[(u64) part * ne + e];
                }
            }
        }
        seen += chunk_total;
        __syncthreads();
    }
}
