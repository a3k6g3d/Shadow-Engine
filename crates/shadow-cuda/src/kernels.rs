//! The kernel registry: every `.cu` file under `kernels/`, carried in the binary, and every entry point's parameter
//! list written out in Rust.  A launch is checked against that list, argument by argument, before it reaches the
//! driver; and `tests/drift.rs` parses the `.cu` files and checks the list against them, so the two cannot drift.

use crate::driver::{Context, CuDevicePtr, CudaError, Module, Stream};
use crate::nvrtc;
use std::ffi::c_void;

/// A kernel parameter's type, as the `.cu` file declares it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum P {
    /// `T*`
    Ptr,
    /// `const T*`
    ConstPtr,
    U32,
    I32,
    U64,
    F32,
}

/// A launch argument.  A device pointer satisfies `Ptr` and `ConstPtr`; scalars must match exactly.
#[derive(Debug, Clone, Copy)]
pub enum Arg {
    Ptr(CuDevicePtr),
    U32(u32),
    I32(i32),
    U64(u64),
    F32(f32),
}

impl Arg {
    fn fits(&self, p: P) -> bool {
        matches!((self, p), (Arg::Ptr(_), P::Ptr | P::ConstPtr) | (Arg::U32(_), P::U32) | (Arg::I32(_), P::I32)
                 | (Arg::U64(_), P::U64) | (Arg::F32(_), P::F32))
    }
}

pub struct Entry {
    pub name: &'static str,
    pub params: &'static [P],
}

/// One `.cu` file: its name under `kernels/`, its source, the headers it needs (prepended after the prelude, in
/// order), and its entry points.
pub struct File {
    pub file: &'static str,
    pub source: &'static str,
    pub headers: &'static [Header],
    pub entries: &'static [Entry],
}

/// A `.cuh` under `kernels/` that a file lists; carried in the binary like the files.
pub struct Header {
    pub file: &'static str,
    pub source: &'static str,
}

pub const PRELUDE: &str = include_str!("../../../kernels/prelude.cuh");

pub const IQ_TABLES: Header = Header { file: "iq_tables.cuh", source: include_str!("../../../kernels/iq_tables.cuh") };
pub const QUANTS: Header = Header { file: "quants.cuh", source: include_str!("../../../kernels/quants.cuh") };

const UNARY: &[P] = &[P::ConstPtr, P::Ptr, P::U32];
const MATVEC: &[P] = &[P::ConstPtr, P::ConstPtr, P::Ptr, P::U32, P::U32];

/// Per quant type of the model: (ggml type name, weights per block, bytes per block, dequant entry, matvec entry).
pub const QUANT_KERNELS: &[(&str, u32, u32, &str, &str)] = &[
    ("Q4_K", 256, 144, "dequant_q4_k", "matvec_q4_k"),
    ("Q5_K", 256, 176, "dequant_q5_k", "matvec_q5_k"),
    ("Q6_K", 256, 210, "dequant_q6_k", "matvec_q6_k"),
    ("IQ4_NL", 32, 18, "dequant_iq4_nl", "matvec_iq4_nl"),
    ("IQ4_XS", 256, 136, "dequant_iq4_xs", "matvec_iq4_xs"),
    ("Q2_0", 64, 18, "dequant_q2_0", "matvec_q2_0"),
    ("Q8_0", 32, 34, "dequant_q8_0", "matvec_q8_0"),
    ("IQ2_S", 256, 82, "dequant_iq2_s", "matvec_iq2_s"),
    ("IQ3_XXS", 256, 98, "dequant_iq3_xxs", "matvec_iq3_xxs"),
    ("IQ3_S", 256, 110, "dequant_iq3_s", "matvec_iq3_s"),
    ("BF16", 1, 2, "dequant_bf16", "matvec_bf16"),
    ("F32", 1, 4, "", "matvec_f32"),
];

const MATVEC_BATCH: &[P] = &[P::ConstPtr, P::ConstPtr, P::U32, P::Ptr, P::U32, P::U32, P::U32];
const MATVEC_PAIR: &[P] = &[P::ConstPtr, P::Ptr, P::U32, P::ConstPtr, P::Ptr, P::U32, P::ConstPtr, P::U32];

/// Per quant type: (ggml type name, fast matvec entry, batched matvec entry) - matvec.cu's 8-weights-at-a-time
/// kernels.  Same signature as the reference matvec for the first; the second takes a device table of matrix
/// addresses and input/output strides, one matrix per blockIdx.y.
pub const FAST_KERNELS: &[(&str, &str, &str)] = &[
    ("Q4_K", "matvec_fast_q4_k", "matvec_batch_q4_k"),
    ("Q5_K", "matvec_fast_q5_k", "matvec_batch_q5_k"),
    ("Q6_K", "matvec_fast_q6_k", "matvec_batch_q6_k"),
    ("IQ4_NL", "matvec_fast_iq4_nl", "matvec_batch_iq4_nl"),
    ("IQ4_XS", "matvec_fast_iq4_xs", "matvec_batch_iq4_xs"),
    ("Q2_0", "matvec_fast_q2_0", "matvec_batch_q2_0"),
    ("Q8_0", "matvec_fast_q8_0", "matvec_batch_q8_0"),
    ("IQ2_S", "matvec_fast_iq2_s", "matvec_batch_iq2_s"),
    ("IQ3_XXS", "matvec_fast_iq3_xxs", "matvec_batch_iq3_xxs"),
    ("IQ3_S", "matvec_fast_iq3_s", "matvec_batch_iq3_s"),
    ("BF16", "matvec_fast_bf16", "matvec_batch_bf16"),
    ("F32", "matvec_fast_f32", "matvec_batch_f32"),
];

/// Per quant type: (ggml type name, paired matvec entry) for two matrices sharing input `x`.
pub const PAIR_KERNELS: &[(&str, &str)] = &[
    ("Q4_K", "matvec_pair_q4_k"),
    ("Q5_K", "matvec_pair_q5_k"),
    ("Q6_K", "matvec_pair_q6_k"),
    ("IQ4_NL", "matvec_pair_iq4_nl"),
    ("IQ4_XS", "matvec_pair_iq4_xs"),
    ("Q2_0", "matvec_pair_q2_0"),
    ("Q8_0", "matvec_pair_q8_0"),
    ("IQ2_S", "matvec_pair_iq2_s"),
    ("IQ3_XXS", "matvec_pair_iq3_xxs"),
    ("IQ3_S", "matvec_pair_iq3_s"),
    ("BF16", "matvec_pair_bf16"),
    ("F32", "matvec_pair_f32"),
];

/// (type of the first matrix, type of the second, entry): pairs of different quant types sharing `x`.  The first
/// matrix's row count must be a multiple of 8 so no 8-row block mixes the two decoders.
pub const PAIR2_KERNELS: &[(&str, &str, &str)] = &[
    ("Q6_K", "Q4_K", "matvec_pair_q6_k_q4_k"),
    ("Q4_K", "IQ4_XS", "matvec_pair_q4_k_iq4_xs"),
    ("IQ4_XS", "Q6_K", "matvec_pair_iq4_xs_q6_k"),
];

/// matvec_cols_*: w, x, x stride, y, y stride, rows, cols, inputs.
const MATVEC_COLS: &[P] = &[P::ConstPtr, P::ConstPtr, P::U32, P::Ptr, P::U32, P::U32, P::U32, P::U32];

const fn cols(name: &'static str) -> Entry {
    Entry { name, params: MATVEC_COLS }
}

const fn batch(name: &'static str) -> Entry {
    Entry { name, params: MATVEC_BATCH }
}

const fn pair(name: &'static str) -> Entry {
    Entry { name, params: MATVEC_PAIR }
}

const fn unary(name: &'static str) -> Entry {
    Entry { name, params: UNARY }
}

const fn matvec(name: &'static str) -> Entry {
    Entry { name, params: MATVEC }
}

const fn e(name: &'static str, params: &'static [P]) -> Entry {
    Entry { name, params }
}

use P::{ConstPtr as C, Ptr as W, F32 as F, U32 as U};

pub const FILES: &[File] = &[
    File {
        file: "vector_add.cu",
        source: include_str!("../../../kernels/vector_add.cu"),
        headers: &[],
        entries: &[Entry { name: "vector_add", params: &[P::ConstPtr, P::ConstPtr, P::Ptr, P::U32] }],
    },
    File {
        file: "convert.cu",
        source: include_str!("../../../kernels/convert.cu"),
        headers: &[],
        entries: &[unary("f16_to_f32_all"), unary("bf16_to_f32_all"), unary("f32_to_f16_all"), unary("f32_to_bf16_all")],
    },
    File {
        file: "dequant.cu",
        source: include_str!("../../../kernels/dequant.cu"),
        headers: &[IQ_TABLES, QUANTS],
        entries: &[
            unary("dequant_q4_k"), unary("dequant_q5_k"), unary("dequant_q6_k"), unary("dequant_iq4_nl"),
            unary("dequant_iq4_xs"), unary("dequant_q2_0"), unary("dequant_q8_0"),
            unary("dequant_iq2_s"), unary("dequant_iq3_xxs"), unary("dequant_iq3_s"), unary("dequant_bf16"),
        ],
    },
    File {
        file: "matvec.cu",
        source: include_str!("../../../kernels/matvec.cu"),
        headers: &[IQ_TABLES, QUANTS],
        entries: &[
            matvec("matvec_q4_k"), matvec("matvec_q5_k"), matvec("matvec_q6_k"), matvec("matvec_iq4_nl"),
            matvec("matvec_iq4_xs"), matvec("matvec_q2_0"), matvec("matvec_q8_0"),
            matvec("matvec_iq2_s"), matvec("matvec_iq3_xxs"), matvec("matvec_iq3_s"), matvec("matvec_bf16"), matvec("matvec_f32"),
            matvec("matvec_fast_q4_k"), batch("matvec_batch_q4_k"), matvec("matvec_fast_q5_k"), batch("matvec_batch_q5_k"),
            matvec("matvec_fast_q6_k"), batch("matvec_batch_q6_k"), matvec("matvec_fast_iq4_nl"),
            batch("matvec_batch_iq4_nl"), matvec("matvec_fast_iq4_xs"), batch("matvec_batch_iq4_xs"),
            matvec("matvec_fast_q2_0"), batch("matvec_batch_q2_0"),
            matvec("matvec_fast_q8_0"), batch("matvec_batch_q8_0"),
            matvec("matvec_fast_iq2_s"), batch("matvec_batch_iq2_s"), matvec("matvec_fast_iq3_xxs"), batch("matvec_batch_iq3_xxs"),
            matvec("matvec_fast_iq3_s"), batch("matvec_batch_iq3_s"), matvec("matvec_fast_bf16"),
            batch("matvec_batch_bf16"),
            e("matvec_pair_bf16", &[C, W, U, C, W, U, C, U]),
            e("hc_down_inject_bf16", &[C, W, U, C, W, U, C, U, F]),
            matvec("matvec_fast_f32"), batch("matvec_batch_f32"),
            cols("matvec_cols_q4_k"), cols("matvec_cols_q5_k"), cols("matvec_cols_q6_k"), cols("matvec_cols_iq4_nl"), cols("matvec_cols_iq4_xs"),
            cols("matvec_cols_q2_0"), cols("matvec_cols_q8_0"), cols("matvec_cols_iq2_s"), cols("matvec_cols_iq3_xxs"),
            cols("matvec_cols_iq3_s"), cols("matvec_cols_bf16"), cols("matvec_cols_f32"),
            cols("matvec_cols_ref_bf16"), cols("matvec_cols_ref_f32"), cols("mma_cols_q6_k"), cols("mma_cols64_q6_k"), cols("mma_cols_bf16"), cols("mma_cols64_bf16"), cols("mma_cols_f32x3"), e("mma_hc_down_inject_bf16", &[C, W, U, U, C, W, U, U, C, U, U, F, U]), cols("mma_cols_q4_k"), cols("mma_cols64_q4_k"), cols("mma_cols_q5_k"), cols("mma_cols64_q5_k"), cols("mma_cols_iq4_xs"), cols("mma_cols64_iq4_xs"), cols("mma_cols_iq4_nl"), cols("mma_cols64_iq4_nl"), cols("mma_cols_q2_0"), cols("mma_cols64_q2_0"), cols("mma_cols_iq2_s"), cols("mma_cols64_iq2_s"), cols("mma_cols_iq3_xxs"), cols("mma_cols64_iq3_xxs"), cols("mma_cols_iq3_s"), cols("mma_cols64_iq3_s"),
            e("matvec_grouped_q4_k", &[C, C, C, U, W, U, U, U]), e("matvec_grouped_q5_k", &[C, C, C, U, W, U, U, U]), e("matvec_grouped_q6_k", &[C, C, C, U, W, U, U, U]), e("matvec_grouped_iq4_nl", &[C, C, C, U, W, U, U, U]), e("matvec_grouped_iq4_xs", &[C, C, C, U, W, U, U, U]), e("matvec_grouped_q2_0", &[C, C, C, U, W, U, U, U]), e("matvec_grouped_q8_0", &[C, C, C, U, W, U, U, U]), e("matvec_grouped_iq2_s", &[C, C, C, U, W, U, U, U]), e("matvec_grouped_iq3_xxs", &[C, C, C, U, W, U, U, U]), e("matvec_grouped_iq3_s", &[C, C, C, U, W, U, U, U]), e("mma_grouped_gu_q6_k", &[C, C, C, U, W, U, U, U]), e("mma_grouped64_gu_q6_k", &[C, C, C, U, W, U, U, U]), e("mma_grouped_gu_iq4_xs", &[C, C, C, U, W, U, U, U]), e("mma_grouped64_gu_iq4_xs", &[C, C, C, U, W, U, U, U]), e("mma_grouped_gu_iq4_nl", &[C, C, C, U, W, U, U, U]), e("mma_grouped64_gu_iq4_nl", &[C, C, C, U, W, U, U, U]), e("mma_grouped_gu_q2_0", &[C, C, C, U, W, U, U, U]), e("mma_grouped64_gu_q2_0", &[C, C, C, U, W, U, U, U]), e("mma_grouped_gu_iq2_s", &[C, C, C, U, W, U, U, U]), e("mma_grouped64_gu_iq2_s", &[C, C, C, U, W, U, U, U]), e("mma_grouped_gu_iq3_xxs", &[C, C, C, U, W, U, U, U]), e("mma_grouped64_gu_iq3_xxs", &[C, C, C, U, W, U, U, U]), e("mma_grouped_gu_iq3_s", &[C, C, C, U, W, U, U, U]), e("mma_grouped64_gu_iq3_s", &[C, C, C, U, W, U, U, U]), e("mma_grouped48_q6_k", &[C, C, C, U, W, U, U, U]), e("mma_grouped48_iq4_xs", &[C, C, C, U, W, U, U, U]), e("mma_grouped48_iq4_nl", &[C, C, C, U, W, U, U, U]), e("mma_grouped48_q2_0", &[C, C, C, U, W, U, U, U]), e("mma_grouped48_iq2_s", &[C, C, C, U, W, U, U, U]), e("mma_grouped48_iq3_xxs", &[C, C, C, U, W, U, U, U]), e("mma_grouped48_iq3_s", &[C, C, C, U, W, U, U, U]), e("mma_grouped16_q6_k", &[C, C, C, U, W, U, U, U]), e("mma_grouped16_iq4_xs", &[C, C, C, U, W, U, U, U]), e("mma_grouped16_iq4_nl", &[C, C, C, U, W, U, U, U]), e("mma_grouped16_q2_0", &[C, C, C, U, W, U, U, U]), e("mma_grouped16_iq2_s", &[C, C, C, U, W, U, U, U]), e("mma_grouped16_iq3_xxs", &[C, C, C, U, W, U, U, U]), e("mma_grouped16_iq3_s", &[C, C, C, U, W, U, U, U]), e("mma_grouped_q6_k", &[C, C, C, U, W, U, U, U]), e("mma_grouped64_q6_k", &[C, C, C, U, W, U, U, U]), e("mma_grouped_iq4_xs", &[C, C, C, U, W, U, U, U]), e("mma_grouped64_iq4_xs", &[C, C, C, U, W, U, U, U]), e("mma_grouped_iq4_nl", &[C, C, C, U, W, U, U, U]), e("mma_grouped64_iq4_nl", &[C, C, C, U, W, U, U, U]), e("mma_grouped_q2_0", &[C, C, C, U, W, U, U, U]), e("mma_grouped64_q2_0", &[C, C, C, U, W, U, U, U]), e("mma_grouped_iq2_s", &[C, C, C, U, W, U, U, U]), e("mma_grouped64_iq2_s", &[C, C, C, U, W, U, U, U]), e("mma_grouped_iq3_xxs", &[C, C, C, U, W, U, U, U]), e("mma_grouped64_iq3_xxs", &[C, C, C, U, W, U, U, U]), e("mma_grouped_iq3_s", &[C, C, C, U, W, U, U, U]), e("mma_grouped64_iq3_s", &[C, C, C, U, W, U, U, U]),
            e("hc_down_inject_cols_bf16", &[C, W, U, U, C, W, U, U, C, U, U, F, U]),
            e("hc_down_inject_fast_bf16", &[C, W, U, C, W, U, C, U, F]),
            e("matvec_cols_multi", &[C, U, C, U, U, U, U]),
            e("hc_up_mix_cols", &[C, C, U, C, U, W, U, U, U, U, U]),
            e("matvec_ids_q6_k", &[C, C, C, W, U, U]),
            e("matvec_ids_q4_k", &[C, C, C, W, U, U]),
            e("matvec_ids_iq4_xs", &[C, C, C, W, U, U]),
            e("matvec_ids_cols_q6_k", &[C, C, C, U, W, U, U, U, U]),
            e("matvec_ids_cols_q4_k", &[C, C, C, U, W, U, U, U, U]),
            e("matvec_ids_cols_iq4_xs", &[C, C, C, U, W, U, U, U, U]),
            e("hc_down_inject_cols_fast_bf16", &[C, W, U, U, C, W, U, U, C, U, U, F, U]),
            pair("matvec_pair_q4_k"), pair("matvec_pair_q5_k"), pair("matvec_pair_q6_k"),
            pair("matvec_pair_iq4_nl"), pair("matvec_pair_iq4_xs"),
            pair("matvec_pair_q2_0"), pair("matvec_pair_q8_0"), pair("matvec_pair_iq2_s"), pair("matvec_pair_iq3_xxs"),
            pair("matvec_pair_iq3_s"), pair("matvec_pair_f32"),
            pair("matvec_pair_q6_k_q4_k"), pair("matvec_pair_q4_k_iq4_xs"), pair("matvec_pair_iq4_xs_q6_k"),
        ],
    },
    File {
        file: "forward.cu",
        source: include_str!("../../../kernels/forward.cu"),
        headers: &[],
        entries: &[
            e("rmsnorm_groups", &[C, C, W, U, F]),
            e("rmsnorm_groups_rows", &[C, C, W, U, F, U]), e("rmsnorm_groups_rows_f", &[C, C, W, U, F, U]),
            e("add_rmsnorm1", &[W, C, C, W, U, F]),
            e("silu_div", &[W, U, F]),
            e("sigmoid_div", &[C, W, U, F, F]),
            e("hc_mix", &[C, C, W, U, U]),
            e("hc_mix_rows", &[C, C, W, U, U]),
            e("hc_write_rmsnorm_rows", &[W, C, C, C, W, U, F, U]), e("hc_write_rmsnorm_rows_f", &[W, C, C, C, W, U, F, U]),
            e("conv_silu_steps", &[C, W, C, W, U, U, U]),
            e("conv_silu_steps_par", &[C, C, C, W, U, U, U]), e("conv_silu_steps_par_f", &[C, C, C, W, U, U, U]),
            e("conv_state_update", &[C, W, U, U, U]),
            e("l2norm_qk_rows", &[W, U, U, F, U]), e("l2norm_qk_rows_f", &[W, U, U, F, U]),
            e("gated_rmsnorm_rows", &[W, C, C, U, F, U]),
            e("gated_rmsnorm_rows_w", &[W, C, C, U, F, U, U]), e("gated_rmsnorm_rows_wf", &[W, C, C, U, F, U, U]),
            e("head_norm_rope_qgk_rows", &[C, W, W, C, C, W, C, U, U, U, F, U, U, F]), e("head_norm_rope_qgk_rows_f", &[C, W, W, C, C, W, C, U, U, U, F, U, U, F]), e("rope_pow_table", &[F, U]), e("ple_gate_rows_f", &[C, C, C, W, U, U]), e("ple_conv_rows_f", &[C, C, C, C, W, U, U, U]), e("ple_state_update", &[C, W, U, U, U]),
            e("attention_heads_rows", &[C, C, C, U, C, W, W, U, U, U, U, U]),
            e("attention_heads_sel_rows", &[C, C, C, C, U, C, C, W, W, U, U, U, U, U]),
            e("attention_group_sel_rows", &[C, C, C, C, U, C, C, W, W, U, U, U, U, U, U]),
            e("attention_group_sel_rows_f32", &[C, C, C, C, U, C, C, W, W, U, U, U, U, U, U]),
            e("attention_group_sel_rows_v2", &[C, C, C, C, U, C, C, W, W, U, U, U, U, U, U]),
            e("attention_group_sel_rows_v3", &[C, C, C, C, U, C, C, W, W, U, U, U, U, U, U]), e("attention_group_sel_rows_tc", &[C, C, C, C, U, C, C, W, W, U, U, U, U, U, U]), e("attention_group_sel_rows_tc2", &[C, C, C, C, U, C, C, W, W, U, U, U, U, U, U]),
            e("attention_group_dense_rows_f32", &[C, C, C, U, C, W, W, U, U, U, U, U, U]), e("attention_group_dense_rows_tc", &[C, C, C, U, C, W, W, U, U, U, U, U, U]),
            e("index_scores_rows", &[C, C, W, U, U, U, U, U, U]),
            e("index_scores_rows_f32", &[C, C, W, U, U, U, U, U, U]),
            e("index_scores_rows_f32v2", &[C, C, W, U, U, U, U, U, U, U]),
            e("select_blocks_rows", &[C, W, W, U, U, U, U, U]),
            e("zero_words", &[W, U]),
            e("fill_words", &[W, U, U]),
            e("route_topk_rows", &[C, W, W, W, U, U, U]),
            e("route_topk_rows_1t", &[C, W, W, W, U, U, U]),
            e("route_uniform_rows", &[C, W, W, W, U, U, U]),
            e("plan_scan_w", &[C, C, W, U]),
            e("plan_fill_w", &[C, C, C, C, C, W, C, U, U, U, U]),
            e("plan_scan", &[C, C, W, U, U]),
            e("plan_fill", &[C, C, C, C, C, W, U, U, U, U, U, U, U, U, U, U]),
            e("head_norm_rope_rows", &[C, U, U, W, U, U, C, U, F, U, U, F]),
            e("moe_combine_qwen_rows", &[W, C, C, C, C, C, U, U]),
            e("hc_write", &[W, C, C, U, U]),
            e("hc_write_rmsnorm_groups", &[W, C, C, C, W, U, F]),
            e("broadcast", &[C, W, U, U]),
            e("broadcast_rows", &[C, W, U, U, U]),
            e("add", &[W, C, U]),
            e("copy", &[W, C, U]),
            e("copy_words", &[W, C, U]),
            e("scale_vec", &[W, U, F]),
            e("conv_silu_step", &[C, W, C, W, U, U]),
            e("l2norm_heads", &[W, U, F]),
            e("l2norm_qk_heads", &[W, U, U, F]),
            e("delta_step", &[C, C, C, C, C, C, C, W, W, U, U]),
            e("delta_steps128", &[C, C, C, U, C, C, U, C, C, W, W, U, U, U]), e("delta_steps128_f", &[C, C, C, U, C, C, U, C, C, W, W, U, U, U]), e("delta_steps128_fp", &[C, C, C, U, C, C, U, C, C, W, W, U, U, U]), e("delta_steps128_f4", &[C, C, C, U, C, C, U, C, C, W, W, U, U, U]),
            e("delta_steps128_kh", &[C, C, C, U, C, C, U, C, C, W, W, U, U, U]), e("delta_steps128_kh2", &[C, C, C, U, C, C, U, C, C, W, W, U, U, U]), e("delta_steps128_kh1", &[C, C, C, U, C, C, U, C, C, W, W, U, U, U]),
            e("delta_chunk_prep", &[C, C, C, C, C, C, W, U, U, U, U, U]),
            e("delta_chunk_run", &[C, C, C, C, W, W, U, U, U, U, U]),
            e("delta_steps128_x8", &[C, C, C, U, C, C, U, C, C, W, W, U, U, U]),
            e("delta_steps128_gn", &[C, C, C, U, C, C, U, C, C, W, W, U, U, U, C, C, F]),
            e("df_capture", &[C, W, U, U, U, U]),
            e("df_rmsnorm", &[C, C, W, U, F]),
            e("df_add_rmsnorm", &[W, C, C, W, U, F]),
            e("df_conv", &[C, C, C, W, U, U, U, U]),
            e("df_rope", &[W, U, U, F]),
            e("df_ring_put", &[C, W, U, U, U]),
            e("df_silu_mul", &[W, C, U]),
            e("df_attention", &[C, C, C, W, U, U, U, U, F, U]),
            e("df_top16", &[C, U, U, C, W, W]),
            e("df_walk", &[C, C, C, C, C, U, U, U, W]),
            e("gated_rmsnorm_heads", &[W, C, C, U, F]),
            e("head_norm_rope", &[C, U, W, U, C, U, F, U, U, F]),
            e("head_norm_rope_qk", &[C, W, C, C, W, C, U, U, F, U, U, F]),
            e("head_norm_rope_qgk", &[C, W, W, C, C, W, C, U, U, F, U, U, F]),
            e("split_gate", &[C, W, U, U]),
            e("attention_heads", &[C, C, C, C, U, C, W, W, U, U, U, U]),
            e("pool_block", &[C, W, C, U, U, U, F, U, F]),
            e("pool_blocks", &[C, W, C, U, U, U, F, U, F]),
            e("index_scores", &[C, C, W, U, U, U]),
            e("swiglu", &[C, C, W, U]),
            e("axpy", &[W, C, U, F]),
            e("axpy_many", &[W, C, C, U, U]),
            e("axpy_many_scaled", &[W, C, C, C, U, U]),
            e("moe_combine_qwen", &[W, C, C, C, C, U, U]),
            e("moe_combine_qwen_perm", &[W, C, C, C, C, C, U, U]),
            e("axpy_sigmoid", &[W, C, C, U]),
            e("argmax_subset", &[C, C, U, W]),
            e("argmax_compact", &[C, C, U, W]),
            e("ple_gate", &[C, C, C, W, U]),
            e("ple_conv_step", &[C, W, C, W, U, U, U]),
        ],
    },
];

pub fn file(name: &str) -> Option<&'static File> {
    FILES.iter().find(|f| f.file == name)
}

/// The whole program NVRTC compiles for one file: the prelude, its headers, then the file.
pub fn program_source(f: &File) -> String {
    let mut s = String::from(PRELUDE);
    for h in f.headers {
        s += &format!("\n#line 1 \"{}\"\n{}", h.file, h.source);
    }
    s + &format!("\n#line 1 \"{}\"\n{}", f.file, f.source)
}

/// Kernel launches through `Program::launch` since the process started, and the host nanoseconds spent in them.
pub static LAUNCHES: (std::sync::atomic::AtomicU64, std::sync::atomic::AtomicU64) =
    (std::sync::atomic::AtomicU64::new(0), std::sync::atomic::AtomicU64::new(0));

/// One file compiled and loaded on one GPU.
pub struct Program<'c> {
    pub file: &'static File,
    pub module: Module<'c>,
    pub compiled: nvrtc::Compiled,
    /// `SHADOW_KTIME=1`: every launch between two events, for `kernel_times` (profiling only: events cost time).
    timed: Option<std::cell::RefCell<Vec<(&'static str, crate::driver::Event<'c>, crate::driver::Event<'c>)>>>,
}

impl<'c> Program<'c> {
    pub fn load(ctx: &'c Context, file: &'static File) -> Result<Program<'c>, String> {
        let compiled = nvrtc::compile(file.file, &program_source(file), ctx.cc, &nvrtc::cache_dir())?;
        let module = ctx.load_module(&compiled.image).map_err(|e| e.to_string())?;
        let timed = std::env::var("SHADOW_KTIME").is_ok_and(|v| v == "1").then(Default::default);
        Ok(Program { file, module, compiled, timed })
    }

    /// Launches `name` with `args`, after checking them against the registry.  Safe to call: the count and the types
    /// are checked here; what is still the caller's promise is that each device pointer is big enough for the kernel,
    /// which is why the pointer-taking wrappers above this (one per kernel) are where the sizes get checked.
    ///
    /// # Safety
    /// Every `Arg::Ptr` must point to device memory of this context, large enough for what the kernel reads and
    /// writes at this grid size.
    pub unsafe fn launch(&self, name: &str, grid: (u32, u32, u32), block: (u32, u32, u32), shared: u32,
                         stream: Option<&Stream>, args: &[Arg]) -> Result<(), String> {
        let t0 = std::time::Instant::now();
        let entry = self.file.entries.iter().find(|e| e.name == name)
            .ok_or_else(|| format!("{} has no entry point {name} in the registry", self.file.file))?;
        if entry.params.len() != args.len() {
            return Err(format!("{name} takes {} arguments, {} given", entry.params.len(), args.len()));
        }
        for (i, (a, p)) in args.iter().zip(entry.params).enumerate() {
            if !a.fits(*p) {
                return Err(format!("{name} argument {i}: {a:?} does not fit a {p:?} parameter"));
            }
        }
        // Each argument as its own storage, then an array of pointers to them: cuLaunchKernel's kernelParams.
        let mut vals: Vec<[u8; 8]> = args.iter().map(|a| match *a {
            Arg::Ptr(p) | Arg::U64(p) => p.to_le_bytes(),
            Arg::U32(v) => (v as u64).to_le_bytes(),
            Arg::I32(v) => (v as u32 as u64).to_le_bytes(),
            Arg::F32(v) => (v.to_bits() as u64).to_le_bytes(),
        }).collect();
        let mut ptrs: Vec<*mut c_void> = vals.iter_mut().map(|v| v.as_mut_ptr() as *mut c_void).collect();
        let f = self.module.function(name).map_err(|e: CudaError| e.to_string())?;
        // SAFETY: the count and the types were checked against the registry (little-endian: a 4-byte parameter reads
        // the low half of its 8-byte slot); the pointers' sizes are the caller's promise.
        let ev = match self.timed.as_ref() {
            Some(_) => {
                let ctx = self.module.ctx();
                let (a, b) = (ctx.event().map_err(|e| e.to_string())?, ctx.event().map_err(|e| e.to_string())?);
                a.record(stream).map_err(|e| e.to_string())?;
                Some((a, b))
            }
            None => None,
        };
        let r = unsafe { f.launch(grid, block, shared, stream, &mut ptrs) }.map_err(|e| e.to_string());
        if let (Some((a, b)), Some(t)) = (ev, self.timed.as_ref()) {
            b.record(stream).map_err(|e| e.to_string())?;
            t.borrow_mut().push((entry.name, a, b));
        }
        use std::sync::atomic::Ordering::Relaxed;
        // `SHADOW_FLUSH_EVERY=N`: nudge the driver to submit its batched launches every N launches
        static FLUSH: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
        let every = *FLUSH.get_or_init(|| std::env::var("SHADOW_FLUSH_EVERY").ok().and_then(|v| v.parse::<u64>().ok()).unwrap_or(0));
        if every > 0 {
            if let Some(s) = stream {
                if (LAUNCHES.0.load(Relaxed) + 1) % every == 0 {
                    s.flush();
                }
            }
        }
        LAUNCHES.0.fetch_add(1, Relaxed);
        LAUNCHES.1.fetch_add(t0.elapsed().as_nanos() as u64, Relaxed);
        r
    }
}

impl Program<'_> {
    /// With `SHADOW_KTIME=1`: GPU milliseconds and launches per kernel since the start, largest first (waits for the
    /// device).
    pub fn kernel_times(&self) -> Vec<(&'static str, f64, usize)> {
        let Some(t) = self.timed.as_ref() else { return vec![] };
        let _ = self.module.ctx().synchronize();
        let mut sum: std::collections::HashMap<&'static str, (f64, usize)> = Default::default();
        for (name, a, b) in t.borrow().iter() {
            let e = sum.entry(*name).or_default();
            e.0 += b.since(a).unwrap_or(0.0) as f64;
            e.1 += 1;
        }
        let mut v: Vec<_> = sum.into_iter().map(|(n, (ms, c))| (n, ms, c)).collect();
        v.sort_by(|a, b| b.1.total_cmp(&a.1));
        v
    }
}

/// The parameter type a C declaration names, for the drift test: `const float* a` -> ConstPtr, `unsigned int n` -> U32.
pub fn parse_param(decl: &str) -> Option<P> {
    let d = decl.trim();
    let ty = match d.rfind(|c: char| c == '*' || c.is_whitespace()) {
        Some(i) if d[i..].starts_with('*') => &d[..=i],
        Some(i) => &d[..i],
        None => d,
    };
    let ty = ty.split_whitespace().filter(|w| *w != "__restrict__").collect::<Vec<_>>().join(" ");
    if ty.ends_with('*') {
        return Some(if ty.starts_with("const ") { P::ConstPtr } else { P::Ptr });
    }
    match ty.as_str() {
        "unsigned int" | "u32" | "uint32_t" => Some(P::U32),
        "int" | "i32" | "int32_t" => Some(P::I32),
        "unsigned long long" | "u64" | "uint64_t" => Some(P::U64),
        "float" => Some(P::F32),
        _ => None,
    }
}

/// Every `extern "C" __global__ void name(...)` in a source, with its parameter types (None for a type the parser
/// does not know - the drift test fails on those rather than skipping them).
pub fn parse_entries(source: &str) -> Vec<(String, Vec<Option<P>>)> {
    let mut out = Vec::new();
    let mut rest = source;
    while let Some(i) = rest.find("extern \"C\" __global__") {
        rest = &rest[i..];
        // a `__launch_bounds__(N)` before the name has its own parentheses: skip it, the name follows it
        let mut from = 0usize;
        let open = loop {
            let o = match rest[from..].find('(') { Some(o) => from + o, None => return out };
            if rest[from..o].trim_end().ends_with("__launch_bounds__") {
                match rest[o..].find(')') { Some(c) => from = o + c + 1, None => return out }
            } else {
                break o;
            }
        };
        let close = match rest[open..].find(')') { Some(c) => open + c, None => break };
        let head = &rest[from..open];
        let name = head.split_whitespace().last().unwrap_or("").to_string();
        let params = rest[open + 1..close]
            .split(',')
            .filter(|p| !p.trim().is_empty() && p.trim() != "void")
            .map(parse_param)
            .collect();
        out.push((name, params));
        rest = &rest[close..];
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_declarations() {
        assert_eq!(parse_param("const float* a"), Some(P::ConstPtr));
        assert_eq!(parse_param("float* __restrict__ out"), Some(P::Ptr));
        assert_eq!(parse_param("const u8 *q"), Some(P::ConstPtr));
        assert_eq!(parse_param("unsigned int n"), Some(P::U32));
        assert_eq!(parse_param("float scale"), Some(P::F32));
        assert_eq!(parse_param("double x"), None);
        let e = parse_entries("extern \"C\" __global__ void k(const float* a, int n) {}\n");
        assert_eq!(e, vec![("k".to_string(), vec![Some(P::ConstPtr), Some(P::I32)])]);
        // launch bounds before the name (one number or two) do not hide the name or the parameters
        let e = parse_entries("extern \"C\" __global__ void __launch_bounds__(256) k1(const float* a) {}\nextern \"C\" __global__ void __launch_bounds__(128, 2)\nk2(float* o, unsigned int n) {}\n");
        assert_eq!(e, vec![("k1".to_string(), vec![Some(P::ConstPtr)]), ("k2".to_string(), vec![Some(P::Ptr), Some(P::U32)])]);
    }
}
