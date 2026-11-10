//! Strata 0.1.44's fused int8 tensor-core MoE kernels (MIT) behind `shadow_fused.dll` (native/fused, built by `build.bat` with nvcc), loaded at run time like `shadow_mmq.dll`.
//! For the grouped experts of a long prompt: gate/up + SwiGLU + int8 H + down in a few launches, 38-43% less time a layer than Shadow's own grouped kernels in the standalone
//! harness (docs/STATUS.md 86).  Opt-in: `SHADOW_FUSED_MOE=1`; `SHADOW_FUSED_DLL=path` names the DLL.  Inexact (int8 activations and H), so never with `--exact`.

use crate::dylib::Dylib;
use std::path::PathBuf;

type InitFn = unsafe extern "C" fn() -> i32;
type SupportedFn = unsafe extern "C" fn(gu: i32, d: i32) -> i32;
type BytesFn = unsafe extern "C" fn(a: i64, b: i64) -> usize;
type QuantizeFn = unsafe extern "C" fn(x: u64, rows: i64, cols: i64, xa: u64, stream: u64);
type MaskFn = unsafe extern "C" fn(ids: u64, n: i64, copied: u64, phase: i32, out: u64, stream: u64);
type GroupFn = unsafe extern "C" fn(ids: u64, n: i64, k: i32, ne: i32, scratch: u64, slot: u64, src: u64, stream: u64);
type ExpertsFn = unsafe extern "C" fn(tab: u64, gu: i32, d: i32, gu_row: u64, d_row: u64, up_off: u64, down_off: u64, ne: i32, n: i64, scratch: u64, xa: u64, src: u64, ha: u64, dm: u64, stream: u64);

pub struct Fused {
    supported: SupportedFn,
    act_bytes: BytesFn,
    group_bytes: unsafe extern "C" fn(n: i64, ne: i32) -> usize,
    quantize: QuantizeFn,
    mask: MaskFn,
    group: GroupFn,
    experts: ExpertsFn,
    _lib: Dylib,
}

/// One layer's formats, as the kernels take them: ggml type numbers and the byte sizes of a gate/up row (`hidden` values) and a down row (`expert_ff` values).
pub struct Geom {
    pub gu_type: i32,
    pub d_type: i32,
    pub gu_row: u64,
    pub d_row: u64,
    pub up_off: u64,
    pub down_off: u64,
}

impl Fused {
    /// The DLL beside the executable, in `native/fused` of the checkout, or named by `SHADOW_FUSED_DLL`; None unless `SHADOW_FUSED_MOE=1` and it loads.
    pub fn load() -> Option<Fused> {
        if !std::env::var("SHADOW_FUSED_MOE").is_ok_and(|v| v == "1") {
            return None;
        }
        let mut cands: Vec<PathBuf> = Vec::new();
        if let Ok(p) = std::env::var("SHADOW_FUSED_DLL") {
            cands.push(PathBuf::from(p));
        }
        if let Ok(exe) = std::env::current_exe() {
            if let Some(dir) = exe.parent() {
                cands.push(dir.join("shadow_fused.dll"));
                if let Some(root) = dir.parent().and_then(|d| d.parent()) {
                    cands.push(root.join("native").join("fused").join("shadow_fused.dll"));
                }
            }
        }
        let path = cands.into_iter().find(|p| p.is_file());
        let Some(path) = path else {
            eprintln!("fused: SHADOW_FUSED_MOE=1 but shadow_fused.dll was not found (native/fused/build.bat builds it)");
            return None;
        };
        let lib = match Dylib::open(&path) {
            Ok(l) => l,
            Err(e) => {
                eprintln!("fused: {e}");
                return None;
            }
        };
        // SAFETY: the exported functions have exactly these C signatures (native/fused/shadow_fused.cu).
        let f = unsafe {
            macro_rules! sym {
                ($name:literal, $t:ty) => {
                    std::mem::transmute::<*mut std::ffi::c_void, $t>(lib.sym($name)?)
                };
            }
            let init = sym!("sf_init", InitFn);
            init();
            Fused {
                supported: sym!("sf_supported", SupportedFn),
                act_bytes: sym!("sf_act_bytes", BytesFn),
                group_bytes: sym!("sf_group_bytes", unsafe extern "C" fn(i64, i32) -> usize),
                quantize: sym!("sf_quantize", QuantizeFn),
                mask: sym!("sf_mask", MaskFn),
                group: sym!("sf_group", GroupFn),
                experts: sym!("sf_experts", ExpertsFn),
                _lib: lib,
            }
        };
        eprintln!("fused: loaded {} (Strata's int8 fused experts for the prompt path)", path.display());
        Some(f)
    }

    /// The ggml type number of a gate/up quantization the kernels read (by `TensorInfo::ty.name()`): IQ2_S, IQ3_XXS, IQ3_S, IQ4_XS.
    pub fn gu_type(name: &str) -> Option<i32> {
        match name {
            "IQ3_XXS" => Some(18),
            "IQ3_S" => Some(21),
            "IQ2_S" => Some(22),
            "IQ4_XS" => Some(23),
            _ => None,
        }
    }

    /// The same for the down projection: IQ4_NL, Q2_0 (upstream ggml's type 42, the one this model's GGUF holds: an f16 scale and 64 two-bit codes in 18 bytes).
    pub fn d_type(name: &str) -> Option<i32> {
        match name {
            "IQ4_NL" => Some(20),
            "Q2_0" => Some(42),
            _ => None,
        }
    }

    pub fn supported(&self, gu: i32, d: i32) -> bool {
        // SAFETY: a pure function of its arguments.
        unsafe { (self.supported)(gu, d) != 0 }
    }

    /// Bytes of `rows` quantized activation rows of `cols` values.
    pub fn act_bytes(&self, rows: usize, cols: usize) -> usize {
        // SAFETY: a pure function of its arguments.
        unsafe { (self.act_bytes)(rows as i64, cols as i64) }
    }

    /// Bytes of the grouping tables for `n` routed rows over `ne` experts (with the row -> pair table).
    pub fn group_bytes(&self, n: usize, ne: usize) -> usize {
        // SAFETY: a pure function of its arguments.
        unsafe { (self.group_bytes)(n as i64, ne as i32) }
    }

    /// # Safety
    /// All addresses are device addresses of this context holding what the arguments say, with the sizes the `*_bytes` functions give; `stream` a CUstream of it (0 = default).
    pub unsafe fn quantize(&self, x: u64, rows: usize, cols: usize, xa: u64, stream: u64) {
        unsafe { (self.quantize)(x, rows as i64, cols as i64, xa, stream) }
    }

    /// # Safety
    /// As `quantize`: `ids` holds `n` expert ids, `copied` one byte per expert, `out` room for `n` ids.
    pub unsafe fn mask(&self, ids: u64, n: usize, copied: u64, phase: i32, out: u64, stream: u64) {
        unsafe { (self.mask)(ids, n as i64, copied, phase, out, stream) }
    }

    /// # Safety
    /// As `quantize`: `scratch` has `group_bytes(n, ne)` bytes, `slot` and `src` room for `n` ints each.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn group(&self, ids: u64, n: usize, k: usize, ne: usize, scratch: u64, slot: u64, src: u64, stream: u64) {
        unsafe { (self.group)(ids, n as i64, k as i32, ne as i32, scratch, slot, src, stream) }
    }

    /// # Safety
    /// As `quantize`: `tab` is a device table of `ne` blob pointers (u64; expert e: gate rows, up rows, down rows in GGUF blocks), `xa` was made by `quantize`, `scratch` and `src` by
    /// `group`, `ha` has `act_bytes(n, expert_ff)` bytes, `dm` is 16-byte aligned with `n` rows of 2560 floats (the down products, by routed pair).
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn experts(&self, tab: u64, g: &Geom, ne: usize, n: usize, scratch: u64, xa: u64, src: u64, ha: u64, dm: u64, stream: u64) {
        unsafe { (self.experts)(tab, g.gu_type, g.d_type, g.gu_row, g.d_row, g.up_off, g.down_off, ne as i32, n as i64, scratch, xa, src, ha, dm, stream) }
    }
}
