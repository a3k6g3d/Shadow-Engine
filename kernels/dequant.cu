// Dequantization on the GPU, one quant block per thread block, one weight per thread: the decoders of quants.cuh
// laid out as plain kernels, so crates/shadow-cuda/tests/dequant.rs can hold each one to the Rust reference
// (shadow_model::quant, itself bit-exact against ggml's) bit for bit.  Launch: grid = number of blocks, block = QK.
// BF16 is one weight per thread over any count.

template <typename Q>
static __device__ __forceinline__ void dequant_block(const u8* blocks, float* out) {
    out[(u64) blockIdx.x * Q::QK + threadIdx.x] = Q::at(blocks + (u64) blockIdx.x * Q::BYTES, threadIdx.x);
}

extern "C" __global__ void dequant_q4_k(const u8* blocks, float* out, unsigned int n_blocks) { dequant_block<Q4K>(blocks, out); }
extern "C" __global__ void dequant_q5_k(const u8* blocks, float* out, unsigned int n_blocks) { dequant_block<Q5K>(blocks, out); }
extern "C" __global__ void dequant_q6_k(const u8* blocks, float* out, unsigned int n_blocks) { dequant_block<Q6K>(blocks, out); }
extern "C" __global__ void dequant_iq4_nl(const u8* blocks, float* out, unsigned int n_blocks) { dequant_block<IQ4NL>(blocks, out); }
extern "C" __global__ void dequant_iq4_xs(const u8* blocks, float* out, unsigned int n_blocks) { dequant_block<IQ4XS>(blocks, out); }
extern "C" __global__ void dequant_nvfp4(const u8* blocks, float* out, unsigned int n_blocks) { dequant_block<NVFP4>(blocks, out); }
extern "C" __global__ void dequant_q2_0(const u8* blocks, float* out, unsigned int n_blocks) { dequant_block<Q20>(blocks, out); }
extern "C" __global__ void dequant_q8_0(const u8* blocks, float* out, unsigned int n_blocks) { dequant_block<Q80>(blocks, out); }
extern "C" __global__ void dequant_iq2_s(const u8* blocks, float* out, unsigned int n_blocks) { dequant_block<IQ2S>(blocks, out); }
extern "C" __global__ void dequant_iq3_xxs(const u8* blocks, float* out, unsigned int n_blocks) { dequant_block<IQ3XXS>(blocks, out); }
extern "C" __global__ void dequant_iq3_s(const u8* blocks, float* out, unsigned int n_blocks) { dequant_block<IQ3S>(blocks, out); }

extern "C" __global__ void dequant_bf16(const u8* blocks, float* out, unsigned int n_blocks) {
    u64 i = (u64) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_blocks) out[i] = BF16::at(blocks + 2 * i, 0);
}
