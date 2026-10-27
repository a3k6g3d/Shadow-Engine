// Dequantization on the GPU, one quant block per thread block, one weight per thread, for the types with no lookup
// grid: Q4_K, Q5_K, Q6_K, IQ4_NL, IQ4_XS, Q2_0, BF16.  These are the oracle-checked building blocks of the matvec
// kernels to come: crates/shadow-cuda/tests/dequant.rs holds them to the Rust reference (shadow_model::quant) bit for
// bit, so the arithmetic is written in the reference's order with __fmul_rn / __fsub_rn, which NVRTC may not fuse
// into an FMA.  Launch: grid = number of blocks, block = weights per block (256, or 32 / 64 for IQ4_NL / Q2_0).

__constant__ signed char kIQ4NL[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};

static __device__ __forceinline__ float f16_at(const u8* b) { return f16_to_f32((u16) b[0] | ((u16) b[1] << 8)); }

static __device__ __forceinline__ void scale_min_k4(const u8* s, int i, u8* sc, u8* mn) {
    if (i < 4) {
        *sc = s[i] & 0x3f;
        *mn = s[4 + i] & 0x3f;
    } else {
        *sc = (s[8 + i - 4] & 0x0f) | ((s[i - 4] >> 2) & 0x30);
        *mn = (s[8 + i - 4] >> 4) | ((s[4 + i - 4] >> 2) & 0x30);
    }
}

extern "C" __global__ void dequant_q4_k(const u8* blocks, float* out, unsigned int n_blocks) {
    const u8* b = blocks + (u64) blockIdx.x * 144;
    unsigned int e = threadIdx.x, i = e / 32, c = e % 32;
    u8 sc, mn;
    scale_min_k4(b + 4, i, &sc, &mn);
    float ds = __fmul_rn(f16_at(b), (float) sc), dm = __fmul_rn(f16_at(b + 2), (float) mn);
    u8 q = (b[16 + (i / 2) * 32 + c] >> ((i % 2) * 4)) & 0x0f;
    out[(u64) blockIdx.x * 256 + e] = __fsub_rn(__fmul_rn(ds, (float) q), dm);
}

extern "C" __global__ void dequant_q5_k(const u8* blocks, float* out, unsigned int n_blocks) {
    const u8* b = blocks + (u64) blockIdx.x * 176;
    unsigned int e = threadIdx.x, i = e / 32, c = e % 32;
    u8 sc, mn;
    scale_min_k4(b + 4, i, &sc, &mn);
    float ds = __fmul_rn(f16_at(b), (float) sc), dm = __fmul_rn(f16_at(b + 2), (float) mn);
    u8 q = ((b[48 + (i / 2) * 32 + c] >> ((i % 2) * 4)) & 0x0f) | (((b[16 + c] >> i) & 1) << 4);
    out[(u64) blockIdx.x * 256 + e] = __fsub_rn(__fmul_rn(ds, (float) q), dm);
}

extern "C" __global__ void dequant_q6_k(const u8* blocks, float* out, unsigned int n_blocks) {
    const u8* b = blocks + (u64) blockIdx.x * 210;
    unsigned int e = threadIdx.x, r = e / 32, c = e % 32;
    unsigned int g = r / 4, h = (r % 4) / 2, k = r % 2, s = r % 4;
    u8 lo = (b[g * 64 + k * 32 + c] >> (h * 4)) & 0x0f;
    u8 hi = (b[128 + g * 32 + c] >> (2 * s)) & 0x03;
    int q = (int) (signed char) (u8) (lo | (hi << 4)) - 32;
    float d = __fmul_rn(f16_at(b + 208), (float) (signed char) b[192 + e / 16]);
    out[(u64) blockIdx.x * 256 + e] = __fmul_rn(d, (float) q);
}

extern "C" __global__ void dequant_iq4_nl(const u8* blocks, float* out, unsigned int n_blocks) {
    const u8* b = blocks + (u64) blockIdx.x * 18;
    unsigned int e = threadIdx.x;
    u8 q = e < 16 ? (b[2 + e] & 0x0f) : (b[2 + e - 16] >> 4);
    out[(u64) blockIdx.x * 32 + e] = __fmul_rn(f16_at(b), (float) kIQ4NL[q]);
}

extern "C" __global__ void dequant_iq4_xs(const u8* blocks, float* out, unsigned int n_blocks) {
    const u8* b = blocks + (u64) blockIdx.x * 136;
    unsigned int e = threadIdx.x, g = e / 32, c = e % 32;
    u16 sh = (u16) b[2] | ((u16) b[3] << 8);
    u8 lo = (b[4 + g / 2] >> (4 * (g % 2))) & 0x0f;
    u8 hi = (sh >> (2 * g)) & 3;
    float dl = __fmul_rn(f16_at(b), (float) ((int) (u8) (lo | (hi << 4)) - 32));
    u8 qb = b[8 + g * 16 + (c % 16)];
    u8 q = c < 16 ? (qb & 0x0f) : (qb >> 4);
    out[(u64) blockIdx.x * 256 + e] = __fmul_rn(dl, (float) kIQ4NL[q]);
}

extern "C" __global__ void dequant_q2_0(const u8* blocks, float* out, unsigned int n_blocks) {
    const u8* b = blocks + (u64) blockIdx.x * 18;
    unsigned int j = threadIdx.x;
    u8 q = (b[2 + j / 4] >> ((j % 4) * 2)) & 0x03;
    out[(u64) blockIdx.x * 64 + j] = __fmul_rn((float) ((int) q - 1), f16_at(b));
}

extern "C" __global__ void dequant_bf16(const u8* blocks, float* out, unsigned int n_blocks) {
    u64 i = (u64) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_blocks) out[i] = bf16_to_f32((u16) blocks[2 * i] | ((u16) blocks[2 * i + 1] << 8));
}
