//! Shadow's GPU interface (docs/SHADOW_PLAN.md, "The stack"): the CUDA driver API and NVRTC, bound by hand and
//! loaded at run time, and the registry of our own kernels, which ship as source and are compiled for whatever GPU is
//! present.  No bindgen, no bindings crate, no nvcc, no CUDA toolkit at build time.

pub mod backend;
pub mod dflash;
pub mod driver;
pub mod dylib;
pub mod fileview;
pub mod kernels;
pub mod kvvmm;
pub mod mmq;
pub mod mtp;
pub mod nvrtc;
pub mod pinned;
pub mod rambox;
pub mod session;
pub mod sha256;

pub use driver::{Context, CudaError, DeviceBuffer};
