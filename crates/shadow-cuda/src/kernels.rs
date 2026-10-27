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

/// One `.cu` file: its name under `kernels/`, its source, and its entry points.
pub struct File {
    pub file: &'static str,
    pub source: &'static str,
    pub entries: &'static [Entry],
}

pub const PRELUDE: &str = include_str!("../../../kernels/prelude.cuh");

const UNARY: &[P] = &[P::ConstPtr, P::Ptr, P::U32];

pub const FILES: &[File] = &[
    File {
        file: "vector_add.cu",
        source: include_str!("../../../kernels/vector_add.cu"),
        entries: &[Entry { name: "vector_add", params: &[P::ConstPtr, P::ConstPtr, P::Ptr, P::U32] }],
    },
    File {
        file: "convert.cu",
        source: include_str!("../../../kernels/convert.cu"),
        entries: &[
            Entry { name: "f16_to_f32_all", params: UNARY },
            Entry { name: "bf16_to_f32_all", params: UNARY },
            Entry { name: "f32_to_f16_all", params: UNARY },
            Entry { name: "f32_to_bf16_all", params: UNARY },
        ],
    },
];

pub fn file(name: &str) -> Option<&'static File> {
    FILES.iter().find(|f| f.file == name)
}

/// The whole program NVRTC compiles for one file: the prelude, then the file.
pub fn program_source(f: &File) -> String {
    format!("{PRELUDE}\n#line 1 \"{}\"\n{}", f.file, f.source)
}

/// One file compiled and loaded on one GPU.
pub struct Program<'c> {
    pub file: &'static File,
    pub module: Module<'c>,
    pub compiled: nvrtc::Compiled,
}

impl<'c> Program<'c> {
    pub fn load(ctx: &'c Context, file: &'static File) -> Result<Program<'c>, String> {
        let compiled = nvrtc::compile(file.file, &program_source(file), ctx.cc, &nvrtc::cache_dir())?;
        let module = ctx.load_module(&compiled.image).map_err(|e| e.to_string())?;
        Ok(Program { file, module, compiled })
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
        unsafe { f.launch(grid, block, shared, stream, &mut ptrs) }.map_err(|e| e.to_string())
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
        let open = match rest.find('(') { Some(o) => o, None => break };
        let close = match rest[open..].find(')') { Some(c) => open + c, None => break };
        let head = &rest[..open];
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
    }
}
