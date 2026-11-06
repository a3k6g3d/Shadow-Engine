//! llama.cpp's MMQ (int8 tensor-core matrix products, MIT) behind `shadow_mmq.dll` (native/mmq, built by `build.bat` with nvcc), loaded at run time like the driver: a
//! machine without the DLL runs everything on Shadow's own kernels.  Used by the prompt path (`--tensor-prompt`, fast mode) for the quantized dense projections: the
//! activations are quantized to int8 once (`quantize`), then one `gemm` multiplies them with the weights in their GGUF layout, 1.7-2.4x faster than the fp16 kernels
//! (docs/STATUS.md 2026-11-05 (42)).  `SHADOW_MMQ=0` turns it off, `SHADOW_MMQ_DLL=path` names the DLL.

use crate::dylib::Dylib;
use std::path::PathBuf;

type QuantizeFn = unsafe extern "C" fn(x: u64, yq: u64, k: i64, ntok: i64, xs: i64, ty: i32, stream: u64) -> i32;
type GemmFn = unsafe extern "C" fn(ty: i32, w: u64, yq: u64, dst: u64, rows: i64, k: i64, ntok: i64, dst_stride: i64, stream: u64) -> i32;
type BytesFn = unsafe extern "C" fn(k: i64, ntok: i64) -> usize;

pub struct Mmq {
    quantize: QuantizeFn,
    gemm: GemmFn,
    y_bytes: BytesFn,
    _lib: Dylib,
}

impl Mmq {
    /// The DLL beside the executable, in `native/mmq` of the checkout (two folders above `target/release`), or named by `SHADOW_MMQ_DLL`; None when off, absent or unloadable.
    pub fn load() -> Option<Mmq> {
        if std::env::var("SHADOW_MMQ").is_ok_and(|v| v == "0") {
            return None;
        }
        let mut cands: Vec<PathBuf> = Vec::new();
        if let Ok(p) = std::env::var("SHADOW_MMQ_DLL") {
            cands.push(PathBuf::from(p));
        }
        if let Ok(exe) = std::env::current_exe() {
            if let Some(dir) = exe.parent() {
                cands.push(dir.join("shadow_mmq.dll"));
                if let Some(root) = dir.parent().and_then(|d| d.parent()) {
                    cands.push(root.join("native").join("mmq").join("shadow_mmq.dll"));
                }
            }
        }
        let path = cands.into_iter().find(|p| p.is_file())?;
        let lib = match Dylib::open(&path) {
            Ok(l) => l,
            Err(e) => {
                eprintln!("mmq: {e}");
                return None;
            }
        };
        // SAFETY: the exported functions have exactly these C signatures (native/mmq/shadow_mmq.cu).
        let (quantize, gemm, y_bytes) = unsafe {
            (
                std::mem::transmute::<*mut std::ffi::c_void, QuantizeFn>(lib.sym("shadow_mmq_quantize")?),
                std::mem::transmute::<*mut std::ffi::c_void, GemmFn>(lib.sym("shadow_mmq_gemm")?),
                std::mem::transmute::<*mut std::ffi::c_void, BytesFn>(lib.sym("shadow_mmq_y_bytes")?),
            )
        };
        eprintln!("mmq: loaded {} (int8 tensor-core projections for the prompt path)", path.display());
        Some(Mmq { quantize, gemm, y_bytes, _lib: lib })
    }

    /// The ggml type number of a quantization the DLL can multiply, by the name `TensorInfo::ty.name()` gives.
    pub fn ggml_type(name: &str) -> Option<i32> {
        match name {
            "Q4_K" => Some(12),
            "Q5_K" => Some(13),
            "Q6_K" => Some(14),
            "IQ4_XS" => Some(23),
            _ => None,
        }
    }

    /// Bytes of the quantized-activation buffer for `ntok` rows of `k` floats.
    pub fn y_bytes(&self, k: usize, ntok: usize) -> usize {
        // SAFETY: a pure function of its arguments.
        unsafe { (self.y_bytes)(k as i64, ntok as i64) }
    }

    /// Quantizes `ntok` rows of `k` floats (rows `xs` floats apart at device address `x`) into `yq` (`y_bytes(k, ntok)` bytes).
    ///
    /// # Safety
    /// `x` and `yq` are device addresses of this context with the sizes above; `stream` a CUstream of it (0 = the default stream).
    pub unsafe fn quantize(&self, x: u64, yq: u64, k: usize, ntok: usize, xs: usize, ty: i32, stream: u64) -> Result<(), String> {
        // SAFETY: the caller's contract.
        let r = unsafe { (self.quantize)(x, yq, k as i64, ntok as i64, xs as i64, ty, stream) };
        if r == 0 { Ok(()) } else { Err(format!("shadow_mmq_quantize: {r}")) }
    }

    /// `dst[t * dst_stride + r] = sum over k of w[r][k] * x[t][k]` for `rows` rows of the weights `w` (GGUF layout, type `ty`) and the `ntok` quantized rows in `yq`.
    ///
    /// # Safety
    /// `w` holds `rows` rows of `k` values of type `ty`, `yq` was made by `quantize` for (`k`, `ntok`), `dst` holds `ntok` rows `dst_stride` floats apart (>= `rows`).
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn gemm(&self, ty: i32, w: u64, yq: u64, dst: u64, rows: usize, k: usize, ntok: usize, dst_stride: usize, stream: u64) -> Result<(), String> {
        // SAFETY: the caller's contract.
        let r = unsafe { (self.gemm)(ty, w, yq, dst, rows as i64, k as i64, ntok as i64, dst_stride as i64, stream) };
        if r == 0 { Ok(()) } else { Err(format!("shadow_mmq_gemm: {r}")) }
    }
}
