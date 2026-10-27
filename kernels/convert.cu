// The prelude's half-precision conversions, exposed so a test can run every one of the 65,536 bit patterns through
// them and compare with the Rust reference.  Every dequant kernel stands on these four functions.

extern "C" __global__ void f16_to_f32_all(const u16* in, float* out, unsigned int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = f16_to_f32(in[i]);
}

extern "C" __global__ void bf16_to_f32_all(const u16* in, float* out, unsigned int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = bf16_to_f32(in[i]);
}

extern "C" __global__ void f32_to_f16_all(const float* in, u16* out, unsigned int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = f32_to_f16(in[i]);
}

extern "C" __global__ void f32_to_bf16_all(const float* in, u16* out, unsigned int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = f32_to_bf16(in[i]);
}
