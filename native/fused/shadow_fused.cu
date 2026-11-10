// shadow_fused.dll: Strata 0.1.44's fused int8 tensor-core MoE kernels (MIT, src/moe_fused*.cu, LICENSE.strata) behind a C interface for Shadow's grouped-expert prompt path
// (docs/STATUS.md 86-87, Phase C).  The Strata sources are vendored with three patches, all marked "SHADOW PATCH": a device table of expert pointers (`Batch::tab`), the pair table
// (`pair_offset`) and the down store into the pair's row.  No cudart linkage in the Rust side: like shadow_mmq.dll this runs on the primary context the engine already holds.
//
// One layer's experts, in this order, all on `stream`, nothing waits on the host:
//   sf_quantize(x, rows, 2560, xa)                                  the layer's input, int8 per 32 values (once per layer)
//   sf_mask(ids, n, copied, phase, ids2)                            the routing ids with the other phase's experts set to -1 (the resident experts run while the others copy)
//   sf_group(ids2, n, k, ne, scratch, slot, src)                    counts, offsets, 64-row tiles, rows and their pairs (scratch: sf_group_bytes)
//   sf_experts(tab, ..., n, scratch, xa, src, ha, dm)               gate/up + SwiGLU + int8 H + down, the down rows into dm[pair]
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <cuda_runtime.h>
#include "strata/prefill/moe_fused_iq.hpp"

using namespace strata::prefill::fused;

#define SF_API extern "C" __declspec(dllexport)

SF_API int sf_init() {
    _putenv_s("STRATA_PF_FUSED", "1");   // the native packs' kernels are opt-in in Strata; here they are the point
    return 0;
}

/// 1 when the kernels cover this gate/up + down pair of ggml types on this device.
SF_API int sf_supported(int gu_type, int d_type) { return native_supported(gu_type, d_type) ? 1 : 0; }

SF_API size_t sf_act_bytes(int64_t rows, int64_t cols) { return act_bytes(rows, cols); }
SF_API size_t sf_group_bytes(int64_t n, int n_expert) { return group_bytes(n, n_expert); }

SF_API void sf_quantize(const float* x, int64_t rows, int64_t cols, void* xa, void* stream) { quantize_act_native(x, rows, cols, xa, stream); }

__global__ void mask_kernel(const int32_t* __restrict__ ids, int64_t n, const uint8_t* __restrict__ copied, int phase, int32_t* __restrict__ out) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int e = ids[i];
    out[i] = (e >= 0 && (copied[e] != 0) == (phase != 0)) ? e : -1;
}

/// out[i] = ids[i] when expert ids[i] belongs to `phase` (0: resident, copied[e] == 0; 1: copied), else -1 (the grouping skips it).
SF_API void sf_mask(const int32_t* ids, int64_t n, const uint8_t* copied, int phase, int32_t* out, void* stream) {
    if (n > 0) mask_kernel<<<(unsigned) ((n + 255) / 256), 256, 0, (cudaStream_t) stream>>>(ids, n, copied, phase, out);
}

SF_API void sf_group(const int32_t* ids, int64_t n, int k, int n_expert, void* scratch, int32_t* slot, int32_t* src, void* stream) { group(ids, n, k, n_expert, scratch, slot, src, stream); }

/// `tab`: DEVICE table of `n_expert` pointers (u64), expert e's blob [gate | up | down] in raw GGUF blocks (2-byte aligned).  Launches in batches of up to kMaxBatch experts.
SF_API void sf_experts(const uint64_t* tab, int gu_type, int d_type, uint64_t gu_row, uint64_t d_row, uint64_t up_off, uint64_t down_off, int n_expert, int64_t n,
                       const void* scratch, const void* xa, const int32_t* src, void* ha, float* dm, void* stream) {
    NativeGeom g;
    g.gu_type = gu_type; g.d_type = d_type; g.gu_row = (size_t) gu_row; g.d_row = (size_t) d_row; g.up_off = (size_t) up_off; g.down_off = (size_t) down_off;
    for (int e0 = 0; e0 < n_expert; e0 += kMaxBatch) {
        Batch b;
        b.e0 = e0;
        b.e1 = std::min(n_expert, e0 + kMaxBatch);
        b.tab = reinterpret_cast<const uint8_t* const*>(tab);
        experts_native(b, g, n_expert, n, scratch, xa, src, ha, dm, stream);
    }
}
