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
extern "C" __global__ void hc_mix(const float* rn, const float* up, float* u, unsigned int h, unsigned int streams) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= h) return;
    float acc = 0.0f;
    for (unsigned int s = 0; s < streams; ++s) acc += sigmoidf_ref(up[s * h + i]) * rn[s * h + i];
    u[i] = acc / (float) streams;
}

// r[s*h+i] += g[s] * y[i]
extern "C" __global__ void hc_write(float* r, const float* g, const float* y, unsigned int h, unsigned int streams) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= h * streams) return;
    r[i] += g[i / h] * y[i % h];
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
    double mem = 0.0;
    for (unsigned int i = 0; i < d; ++i) {
        s[i * d + j] *= decay;
        mem += (double) s[i * d + j] * (double) ks[i];
    }
    float delta = (float) (((double) v[h * d + j] - mem) * (double) beta);
    double acc = 0.0;
    for (unsigned int i = 0; i < d; ++i) {
        s[i * d + j] += ks[i] * delta;
        acc += (double) s[i * d + j] * (double) qs[i];
    }
    o[h * d + j] = (float) acc;
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

// gate[h*hd + j] = qg[h*2*hd + hd + j]: the gate half of each query head
extern "C" __global__ void split_gate(const float* qg, float* gate, unsigned int heads, unsigned int hd) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= heads * hd) return;
    gate[i] = qg[(i / hd) * 2 * hd + hd + i % hd];
}

// Softmax attention of each query head over the positions in `toks` (all 0..ntok-1 when toks is null), GQA, then
// times sigmoid(gate).  Scores, exponentials and the weighted sum in double, as the reference.  grid = heads,
// block = 256; `scratch` holds heads x stride doubles.
extern "C" __global__ void attention_heads(const float* q, const float* kc, const float* vc, const u32* toks,
                                           unsigned int ntok, const float* gate, float* o, double* scratch,
                                           unsigned int stride, unsigned int nkv, unsigned int hd, unsigned int group) {
    unsigned int h = blockIdx.x, g = h / group;
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
