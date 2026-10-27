// Dequantization of the grid-based types - IQ2_S, IQ3_XXS, IQ3_S, 94 of the model's 144 routed-expert tensors - one
// quant block per thread block, one weight per thread.  The grids come from iq_tables.cuh (a header of this file in
// the registry).  Same rule as dequant.cu: the reference's arithmetic order, unfused, so the GPU is held to the same
// bits as shadow_model::quant.  Launch: grid = number of blocks, block = 256.

static __device__ __forceinline__ float half_at(const u8* b) { return f16_to_f32((u16) b[0] | ((u16) b[1] << 8)); }

static __device__ __forceinline__ float signed_by(float v, unsigned int bit) { return (bit & 1) ? -v : v; }

extern "C" __global__ void dequant_iq2_s(const u8* blocks, float* out, unsigned int n_blocks) {
    const u8* b = blocks + (u64) blockIdx.x * 82;
    unsigned int e = threadIdx.x, j = e / 8, k = e % 8;
    unsigned int idx = b[2 + j] | (((b[66 + j / 4] >> (2 * (j % 4))) & 3) << 8);
    unsigned int s = (b[74 + e / 32] >> (4 * ((e / 16) % 2))) & 0x0f;
    float db = __fmul_rn(__fmul_rn(half_at(b), __fadd_rn(0.5f, (float) s)), 0.25f);
    float v = __fmul_rn(db, (float) kIQ2SGrid[idx][k]);
    out[(u64) blockIdx.x * 256 + e] = signed_by(v, b[34 + e / 8] >> (e % 8));
}

extern "C" __global__ void dequant_iq3_xxs(const u8* blocks, float* out, unsigned int n_blocks) {
    const u8* b = blocks + (u64) blockIdx.x * 98;
    unsigned int e = threadIdx.x, g = e / 32, k = (e % 32) / 8, m = e % 8;
    const u8* sp = b + 66 + g * 4;
    unsigned int s32 = (unsigned int) sp[0] | ((unsigned int) sp[1] << 8) | ((unsigned int) sp[2] << 16)
                     | ((unsigned int) sp[3] << 24);
    float db = __fmul_rn(__fmul_rn(half_at(b), __fadd_rn(0.5f, (float) (s32 >> 28))), 0.5f);
    u8 signs = kKSigns[(s32 >> (7 * k)) & 0x7f];
    float v = __fmul_rn(db, (float) kIQ3XXSGrid[b[2 + e / 4]][e % 4]);
    out[(u64) blockIdx.x * 256 + e] = signed_by(v, signs >> m);
}

extern "C" __global__ void dequant_iq3_s(const u8* blocks, float* out, unsigned int n_blocks) {
    const u8* b = blocks + (u64) blockIdx.x * 110;
    unsigned int e = threadIdx.x, j = e / 4, g = e / 32;
    unsigned int idx = b[2 + j] | (((b[66 + j / 8] >> (j % 8)) & 1) << 8);
    unsigned int s = (b[106 + g / 2] >> (4 * (g % 2))) & 0x0f;
    float db = __fmul_rn(half_at(b), (float) (1 + 2 * s));
    float v = __fmul_rn(db, (float) kIQ3SGrid[idx][e % 4]);
    out[(u64) blockIdx.x * 256 + e] = signed_by(v, b[74 + e / 8] >> (e % 8));
}
