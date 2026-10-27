// y = W x for a matrix stored as ggml `[cols, rows]` (each of the `rows` rows is `cols` weights, whole blocks,
// contiguous), x and y f32.  One warp per row: each lane decodes every 32nd weight of the row with the quants.cuh
// decoder and accumulates in f32; the warp then sums its lanes.  Correctness first: the per-weight decode re-reads
// block scales, so this is the reference GPU matvec, not the fast one.  Launch: block = 256 threads (8 rows),
// grid = ceil(rows / 8).

template <typename Q>
static __device__ __forceinline__ void matvec_rows(const u8* w, const float* x, float* y, unsigned int rows,
                                                   unsigned int cols) {
    unsigned int row = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32, lane = threadIdx.x % 32;
    if (row >= rows) return;
    const u8* r = w + (u64) row * (cols / Q::QK) * Q::BYTES;
    float acc = 0.0f;
    for (unsigned int i = lane; i < cols; i += 32) acc += Q::at(r + (i / Q::QK) * Q::BYTES, i % Q::QK) * x[i];
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, off);
    if (lane == 0) y[row] = acc;
}

extern "C" __global__ void matvec_q4_k(const u8* w, const float* x, float* y, unsigned int rows, unsigned int cols) { matvec_rows<Q4K>(w, x, y, rows, cols); }
extern "C" __global__ void matvec_q5_k(const u8* w, const float* x, float* y, unsigned int rows, unsigned int cols) { matvec_rows<Q5K>(w, x, y, rows, cols); }
extern "C" __global__ void matvec_q6_k(const u8* w, const float* x, float* y, unsigned int rows, unsigned int cols) { matvec_rows<Q6K>(w, x, y, rows, cols); }
extern "C" __global__ void matvec_iq4_nl(const u8* w, const float* x, float* y, unsigned int rows, unsigned int cols) { matvec_rows<IQ4NL>(w, x, y, rows, cols); }
extern "C" __global__ void matvec_iq4_xs(const u8* w, const float* x, float* y, unsigned int rows, unsigned int cols) { matvec_rows<IQ4XS>(w, x, y, rows, cols); }
extern "C" __global__ void matvec_q2_0(const u8* w, const float* x, float* y, unsigned int rows, unsigned int cols) { matvec_rows<Q20>(w, x, y, rows, cols); }
extern "C" __global__ void matvec_iq2_s(const u8* w, const float* x, float* y, unsigned int rows, unsigned int cols) { matvec_rows<IQ2S>(w, x, y, rows, cols); }
extern "C" __global__ void matvec_iq3_xxs(const u8* w, const float* x, float* y, unsigned int rows, unsigned int cols) { matvec_rows<IQ3XXS>(w, x, y, rows, cols); }
extern "C" __global__ void matvec_iq3_s(const u8* w, const float* x, float* y, unsigned int rows, unsigned int cols) { matvec_rows<IQ3S>(w, x, y, rows, cols); }
extern "C" __global__ void matvec_bf16(const u8* w, const float* x, float* y, unsigned int rows, unsigned int cols) { matvec_rows<BF16>(w, x, y, rows, cols); }
extern "C" __global__ void matvec_f32(const u8* w, const float* x, float* y, unsigned int rows, unsigned int cols) { matvec_rows<F32>(w, x, y, rows, cols); }
