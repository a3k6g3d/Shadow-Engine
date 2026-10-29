// Everything in a forward pass that is not a matrix-vector product (matvec.cu does those): norms, the
// hyper-connection glue, the DeltaNet step, attention with its sparse indexer, the MoE glue and the n-gram memory.
// Each kernel follows the CPU reference (crates/shadow-model/src/forward.rs) step for step, including where the
// reference widens to double (norm sums, dot products, exp/log), so the GPU session can be held to the reference
// fixtures stream by stream.  Kernels take no doubles as arguments (the launcher has none); positions are u32 and the
// RoPE base is passed as a float (1e7 is exact in float).

static __device__ __forceinline__ float sigmoidf_ref(float x) { return (float) (1.0 / (1.0 + exp(-(double) x))); }
static __device__ __forceinline__ float siluf_ref(float x) { return x * sigmoidf_ref(x); }
static __device__ __forceinline__ float softplusf_ref(float x) { return x > 20.0f ? x : (float) log1p(exp((double) x)); }

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
static __device__ void rope_one(float* x, unsigned int pos, unsigned int dims, float theta) {
    unsigned int half = dims / 2;
    for (unsigned int i = threadIdx.x; i < half; i += blockDim.x) {
        double f = (double) pos / pow((double) theta, 2.0 * (double) i / (double) dims);
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
    float decay = (float) exp((double) A[h] * (double) softplusf_ref(a[h] + dtb[h]));
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
        float decay = (float) exp((double) A[h] * (double) softplusf_ref(a[(u64) t * bs + h] + dtb[h]));
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
    for (unsigned int t = threadIdx.x; t < ntok; t += blockDim.x) {
        unsigned int p = toks ? toks[t] : t;
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
    z = block_sum(z);
    __syncthreads();
    for (unsigned int j = threadIdx.x; j < hd; j += blockDim.x) {
        double acc = 0.0;
        for (unsigned int t = 0; t < ntok; ++t) {
            unsigned int p = toks ? toks[t] : t;
            acc += sc[t] * (double) vc[((u64) p * nkv + g) * hd + j];
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
extern "C" __global__ void pool_block(const float* raw, float* blocks, const float* w, unsigned int b,
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

// score[b] = (sum over heads of max(0, q_h . key_b)) / sqrt(id); the dots in double, rounded to float, as the reference.
// One thread per block.
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
