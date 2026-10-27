// The stage 0 kernel: proves the path (NVRTC compile -> CUBIN -> cuModuleLoadData -> cuLaunchKernel) end to end,
// and nothing else.  Every kernel file follows the same rules: self-contained (no toolkit #include - a user's PC has
// the NVRTC DLLs, not the CUDA headers), entry points `extern "C"` so their names are not mangled, and every entry
// point's parameter list is mirrored in crates/shadow-cuda/src/kernels.rs, which the drift test checks.

extern "C" __global__ void vector_add(const float* a, const float* b, float* out, unsigned int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] + b[i];
}
