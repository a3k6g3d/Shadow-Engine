
// ---------------------------------------------------------------------------------------------------------------------------------------------------------
// Grouped experts of a long prompt (the MoE gate/up projections): Shadow's device planner (plan_fill) leaves, for each group of <= J tokens of one expert, a record
// (1 + 2 * J words: count, J input rows, J output rows) and the address of the expert's matrix; the groups of the gate then the up matrices follow each other (2 * G
// of them) and share their tokens, so the activations are quantized once for G groups: column g * J + j of the quantized activations is input row j of group g.
// One block a (J-wide group, tile of I weight rows); the block's weights are read from the group's own address, so experts may lie anywhere (cache slots, the stage
// halves).  The tile loop is ggml's (mul_mat_q_process_tile).
template <ggml_type type, int J, bool fallback>
__launch_bounds__(ggml_cuda_mmq_get_nthreads(type, J, fallback), ggml_cuda_mmq_get_occupancy(type, J, fallback))
static __global__ void shadow_moe_q(
        const unsigned long long * __restrict__ xtab, const int * __restrict__ y, const unsigned int * __restrict__ rec, float * __restrict__ dst,
        const int nrows_x, const int stride_row_x, const int blocks_per_ne00, const int ncols_y, const int stride_col_dst, const int ny_groups) {
    if (ggml_cuda_mmq_get_config(type, J, fallback).type == GGML_TYPE_COUNT) {
        NO_DEVICE_CODE;
        return;
    }
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    constexpr int nwarps    = ggml_cuda_mmq_get_nthreads(type, J, fallback) / warp_size;
    constexpr int I         = ggml_cuda_mmq_get_I(type, J, fallback);

    const int zt = blockIdx.z, it = blockIdx.x;
    const unsigned int * r = rec + (size_t) zt * (1 + 2 * J);
    const int cnt = (int) r[0];
    if (cnt == 0) {
        return;
    }
    extern __shared__ int ids_dst_shared[];
#pragma unroll
    for (int j0 = 0; j0 < J; j0 += nwarps*warp_size) {
        const int j = j0 + threadIdx.y*warp_size + threadIdx.x;
        if (j0 + nwarps*warp_size > J && j >= J) {
            break;
        }
        ids_dst_shared[j] = j < cnt ? (int) r[1 + J + j] : 0;
    }
    __syncthreads();
    const int * yt = y + (size_t) ((zt % ny_groups) * J) * (sizeof(block_q8_1_mmq) / sizeof(int));
    mul_mat_q_process_tile<type, J, fallback, false>(
        (const char *) xtab[zt], it*I*stride_row_x, yt, ids_dst_shared, dst + it*I, nullptr, nullptr, stride_row_x, ncols_y, stride_col_dst,
        nrows_x - it*I - 1, cnt - 1, 0, blocks_per_ne00);
}

template <ggml_type type, int J, bool fallback>
static int launch_moe(const unsigned long long * xtab, const int * y, const unsigned int * rec, float * dst, int64_t rows, int64_t k, int64_t ncols_y,
                      int64_t dst_stride, int64_t ngroups, int64_t ny_groups, cudaStream_t s) {
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    const int warp_size = ggml_cuda_info().devices[ggml_cuda_get_device()].warp_size;
    const ggml_cuda_mmq_config config = ggml_cuda_mmq_get_config(type, J, fallback, cc);
    if (config.type == GGML_TYPE_COUNT) return 4;
    const int nwarps = config.nthreads / warp_size;
    const int nbytes_shared = mmq_get_nbytes_shared(config, cc);
    const dim3 block_dims(warp_size, nwarps, 1);
    CUDA_SET_SHARED_MEMORY_LIMIT((shadow_moe_q<type, J, fallback>), nbytes_shared);
    const dim3 grid((unsigned) ((rows + config.I - 1) / config.I), 1, (unsigned) ngroups);
    const int bpr = (int) (k / ggml_cuda_type_traits<type>::qk);
    shadow_moe_q<type, J, fallback><<<grid, block_dims, nbytes_shared, s>>>(xtab, y, rec, dst, (int) rows, bpr, bpr, (int) ncols_y, (int) dst_stride, (int) ny_groups);
    return cudaGetLastError() == cudaSuccess ? 0 : 3;
}

template <ggml_type type>
static int dispatch_moe(int64_t chunk, const unsigned long long * xtab, const int * y, const unsigned int * rec, float * dst, int64_t rows, int64_t k, int64_t ncols_y,
                        int64_t dst_stride, int64_t ngroups, int64_t ny_groups, cudaStream_t s) {
    const bool fb = rows % 128 != 0;
    if (chunk == 32) return fb ? launch_moe<type, 32, true>(xtab, y, rec, dst, rows, k, ncols_y, dst_stride, ngroups, ny_groups, s)
                               : launch_moe<type, 32, false>(xtab, y, rec, dst, rows, k, ncols_y, dst_stride, ngroups, ny_groups, s);
    if (chunk == 48) return fb ? launch_moe<type, 48, true>(xtab, y, rec, dst, rows, k, ncols_y, dst_stride, ngroups, ny_groups, s)
                               : launch_moe<type, 48, false>(xtab, y, rec, dst, rows, k, ncols_y, dst_stride, ngroups, ny_groups, s);
    if (chunk == 64) return fb ? launch_moe<type, 64, true>(xtab, y, rec, dst, rows, k, ncols_y, dst_stride, ngroups, ny_groups, s)
                               : launch_moe<type, 64, false>(xtab, y, rec, dst, rows, k, ncols_y, dst_stride, ngroups, ny_groups, s);
    return 1;
}

// ids[g * chunk + j] = group g's input row j (0 past the group's count): the gather list of the quantization
static __global__ void moe_ids(const unsigned int * rec, int * ids, int ncols, int chunk) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= ncols) return;
    const unsigned int * r = rec + (size_t) (c / chunk) * (1 + 2 * chunk);
    const int j = c % chunk;
    ids[c] = j < (int) r[0] ? (int) r[1 + j] : 0;
}

// bytes of scratch: the gather list (256-aligned), then the quantized activations
EXPORT size_t shadow_mmq_moe_bytes(int64_t k, int64_t ny_groups, int64_t chunk) {
    return (size_t) GGML_PAD(ny_groups * chunk * 4, 256) + shadow_mmq_y_bytes(k, ny_groups * chunk);
}

// groups: the 2 * G (gate then up) or G (down) groups at xtab (u64 matrix addresses) and rec; ny_groups = G; x: rows of k floats xs apart, named by the records' input
// rows; dst row r (the records' output rows) at dst + r * dst_stride; chunk = 32 or 64 tokens a group.
EXPORT int shadow_mmq_moe(int type, unsigned long long xtab, unsigned long long rec, int64_t ngroups, int64_t ny_groups, int64_t chunk, unsigned long long x, int64_t xs,
                          int64_t k, int64_t rows, unsigned long long dst, int64_t dst_stride, unsigned long long scratch, void * stream) {
    if (k % 256 != 0 || xs % 4 != 0 || (chunk != 32 && chunk != 48 && chunk != 64)) return 2;
    cudaStream_t s = (cudaStream_t) stream;
    const int64_t ncols = ny_groups * chunk;
    int * ids = (int *) scratch;
    void * yq = (void *) (scratch + GGML_PAD(ncols * 4, 256));
    moe_ids<<<(unsigned) ((ncols + 255) / 256), 256, 0, s>>>((const unsigned int *) rec, ids, (int) ncols, (int) chunk);
    const int64_t k_pad = GGML_PAD(k, MATRIX_ROW_PADDING);
    quantize_mmq_q8_1_cuda((const float *) x, ids, yq, (ggml_type) type, k, xs, 0, 0, k_pad, ncols, 1, 1, s);
    if (cudaGetLastError() != cudaSuccess) return 3;
    const unsigned long long * xt = (const unsigned long long *) xtab;
    const unsigned int * rc = (const unsigned int *) rec;
    const int * y = (const int *) yq;
    float * d = (float *) dst;
    switch ((ggml_type) type) {
        case GGML_TYPE_IQ2_S:   return dispatch_moe<GGML_TYPE_IQ2_S>(chunk, xt, y, rc, d, rows, k, ncols, dst_stride, ngroups, ny_groups, s);
        case GGML_TYPE_IQ3_XXS: return dispatch_moe<GGML_TYPE_IQ3_XXS>(chunk, xt, y, rc, d, rows, k, ncols, dst_stride, ngroups, ny_groups, s);
        case GGML_TYPE_IQ3_S:   return dispatch_moe<GGML_TYPE_IQ3_S>(chunk, xt, y, rc, d, rows, k, ncols, dst_stride, ngroups, ny_groups, s);
        default: return 1;
    }
}
