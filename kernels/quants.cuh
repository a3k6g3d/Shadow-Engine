// Every quant format of the model as a device-side decoder: `Q::at(block, e)` is weight e of one block, computed in
// the reference's arithmetic order, unfused (__fmul_rn / __fsub_rn / __fadd_rn), so dequant.cu can be held to the Rust
// reference bit for bit.  QK = weights per block, BYTES = bytes per block.  Needs iq_tables.cuh before it.
//
// Per-weight decoding re-reads a block's scales for every weight: right for correctness, not for speed.  The fast
// matvec kernels will decode a block's scales once per warp; these stay as their oracle.

// IQ4_NL's 16 non-linear levels {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113} packed in
// four 32-bit immediate constants so warp lanes with different nibbles never serialize on __constant__ memory.
static __device__ __forceinline__ float iq4nl_val(unsigned int q) {
    unsigned int w = (q < 8u) ? ((q < 4u) ? 0xbfad9881u : 0xf6eaddcfu)
                              : ((q < 12u) ? 0x26190d01u : 0x71594535u);
    signed char b = (signed char) ((w >> ((q & 3u) * 8u)) & 0xffu);
    return (float) b;
}

static __device__ __forceinline__ float qf16(const u8* b) { return f16_to_f32((u16) b[0] | ((u16) b[1] << 8)); }

static __device__ __forceinline__ float qsign(float v, unsigned int bit) { return (bit & 1) ? -v : v; }

static __device__ __forceinline__ void q_scale_min_k4(const u8* s, unsigned int i, u8* sc, u8* mn) {
    if (i < 4) {
        *sc = s[i] & 0x3f;
        *mn = s[4 + i] & 0x3f;
    } else {
        *sc = (s[8 + i - 4] & 0x0f) | ((s[i - 4] >> 2) & 0x30);
        *mn = (s[8 + i - 4] >> 4) | ((s[4 + i - 4] >> 2) & 0x30);
    }
}

struct Q4K {
    static const unsigned int QK = 256, BYTES = 144;
    static __device__ __forceinline__ float at(const u8* b, unsigned int e) {
        unsigned int i = e / 32, c = e % 32;
        u8 sc, mn;
        q_scale_min_k4(b + 4, i, &sc, &mn);
        float ds = __fmul_rn(qf16(b), (float) sc), dm = __fmul_rn(qf16(b + 2), (float) mn);
        u8 q = (b[16 + (i / 2) * 32 + c] >> ((i % 2) * 4)) & 0x0f;
        return __fsub_rn(__fmul_rn(ds, (float) q), dm);
    }
};

struct Q5K {
    static const unsigned int QK = 256, BYTES = 176;
    static __device__ __forceinline__ float at(const u8* b, unsigned int e) {
        unsigned int i = e / 32, c = e % 32;
        u8 sc, mn;
        q_scale_min_k4(b + 4, i, &sc, &mn);
        float ds = __fmul_rn(qf16(b), (float) sc), dm = __fmul_rn(qf16(b + 2), (float) mn);
        u8 q = ((b[48 + (i / 2) * 32 + c] >> ((i % 2) * 4)) & 0x0f) | (((b[16 + c] >> i) & 1) << 4);
        return __fsub_rn(__fmul_rn(ds, (float) q), dm);
    }
};

struct Q6K {
    static const unsigned int QK = 256, BYTES = 210;
    static __device__ __forceinline__ float at(const u8* b, unsigned int e) {
        unsigned int r = e / 32, c = e % 32;
        unsigned int g = r / 4, h = (r % 4) / 2, k = r % 2, s = r % 4;
        u8 lo = (b[g * 64 + k * 32 + c] >> (h * 4)) & 0x0f;
        u8 hi = (b[128 + g * 32 + c] >> (2 * s)) & 0x03;
        int q = (int) (signed char) (u8) (lo | (hi << 4)) - 32;
        float d = __fmul_rn(qf16(b + 208), (float) (signed char) b[192 + e / 16]);
        return __fmul_rn(d, (float) q);
    }
};

struct IQ4NL {
    static const unsigned int QK = 32, BYTES = 18;
    static __device__ __forceinline__ float at(const u8* b, unsigned int e) {
        u8 q = e < 16 ? (b[2 + e] & 0x0f) : (b[2 + e - 16] >> 4);
        return __fmul_rn(qf16(b), iq4nl_val(q));
    }
};

struct IQ4XS {
    static const unsigned int QK = 256, BYTES = 136;
    static __device__ __forceinline__ float at(const u8* b, unsigned int e) {
        unsigned int g = e / 32, c = e % 32;
        u16 sh = (u16) b[2] | ((u16) b[3] << 8);
        u8 lo = (b[4 + g / 2] >> (4 * (g % 2))) & 0x0f;
        u8 hi = (sh >> (2 * g)) & 3;
        float dl = __fmul_rn(qf16(b), (float) ((int) (u8) (lo | (hi << 4)) - 32));
        u8 qb = b[8 + g * 16 + (c % 16)];
        return __fmul_rn(dl, iq4nl_val(c < 16 ? (qb & 0x0f) : (qb >> 4)));
    }
};

struct Q20 {
    static const unsigned int QK = 64, BYTES = 18;
    static __device__ __forceinline__ float at(const u8* b, unsigned int e) {
        u8 q = (b[2 + e / 4] >> ((e % 4) * 2)) & 0x03;
        return __fmul_rn((float) ((int) q - 1), qf16(b));
    }
};

static __device__ __forceinline__ float ue4m3_half(u8 sb) {
    if (sb == 0 || sb == 0x7f) return 0.0f;
    unsigned int exp = (sb >> 3) & 0x0f, man = sb & 0x07;
    if (exp == 0) return __fmul_rn((float) man, 1.0f / 1024.0f);
    return __fmul_rn((float) (8 + man), __uint_as_float((116 + exp) << 23));
}

static __device__ __forceinline__ float e2m1_double(u8 q) {
    int mag = (int) ((0xc8643210u >> ((q & 7u) * 4u)) & 0x0fu);
    return (float) ((q & 8u) ? -mag : mag);
}

struct NVFP4 {
    static const unsigned int QK = 64, BYTES = 36;
    static __device__ __forceinline__ float at(const u8* b, unsigned int e) {
        unsigned int g = e / 16, c = e % 16;
        u8 qb = b[4 + g * 8 + (c % 8)];
        u8 q = c < 8 ? (qb & 0x0f) : (qb >> 4);
        return __fmul_rn(ue4m3_half(b[g]), e2m1_double(q));
    }
};

struct IQ2S {
    static const unsigned int QK = 256, BYTES = 82;
    static __device__ __forceinline__ float at(const u8* b, unsigned int e) {
        unsigned int j = e / 8;
        unsigned int idx = b[2 + j] | (((b[66 + j / 4] >> (2 * (j % 4))) & 3) << 8);
        unsigned int s = (b[74 + e / 32] >> (4 * ((e / 16) % 2))) & 0x0f;
        float db = __fmul_rn(__fmul_rn(qf16(b), __fadd_rn(0.5f, (float) s)), 0.25f);
        return qsign(__fmul_rn(db, (float) kIQ2SGrid[idx][e % 8]), b[34 + e / 8] >> (e % 8));
    }
};

struct IQ3XXS {
    static const unsigned int QK = 256, BYTES = 98;
    static __device__ __forceinline__ float at(const u8* b, unsigned int e) {
        unsigned int g = e / 32, k = (e % 32) / 8;
        const u8* sp = b + 66 + g * 4;
        unsigned int s32 = (unsigned int) sp[0] | ((unsigned int) sp[1] << 8) | ((unsigned int) sp[2] << 16)
                         | ((unsigned int) sp[3] << 24);
        float db = __fmul_rn(__fmul_rn(qf16(b), __fadd_rn(0.5f, (float) (s32 >> 28))), 0.5f);
        u8 signs = kKSigns[(s32 >> (7 * k)) & 0x7f];
        return qsign(__fmul_rn(db, (float) kIQ3XXSGrid[b[2 + e / 4]][e % 4]), signs >> (e % 8));
    }
};

struct IQ3S {
    static const unsigned int QK = 256, BYTES = 110;
    static __device__ __forceinline__ float at(const u8* b, unsigned int e) {
        unsigned int j = e / 4, g = e / 32;
        unsigned int idx = b[2 + j] | (((b[66 + j / 8] >> (j % 8)) & 1) << 8);
        unsigned int s = (b[106 + g / 2] >> (4 * (g % 2))) & 0x0f;
        float db = __fmul_rn(qf16(b), (float) (1 + 2 * s));
        return qsign(__fmul_rn(db, (float) kIQ3SGrid[idx][e % 4]), b[74 + e / 8] >> (e % 8));
    }
};

struct BF16 {
    static const unsigned int QK = 1, BYTES = 2;
    static __device__ __forceinline__ float at(const u8* b, unsigned int) {
        return bf16_to_f32((u16) b[0] | ((u16) b[1] << 8));
    }
};

struct F32 {
    static const unsigned int QK = 1, BYTES = 4;
    static __device__ __forceinline__ float at(const u8* b, unsigned int) {
        return __uint_as_float((u32) b[0] | ((u32) b[1] << 8) | ((u32) b[2] << 16) | ((u32) b[3] << 24));
    }
};
