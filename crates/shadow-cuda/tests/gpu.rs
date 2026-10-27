//! The stage 0 GPU gate: `vector_add` compiled by NVRTC from a cold cache, then loaded from the warm one, launched on
//! the device, and checked element by element.  On a machine with no NVIDIA driver or no NVRTC the test says so out
//! loud and passes - that is the skip path, and it is not the same as green.

use shadow_cuda::kernels::{file, Arg, Program};
use shadow_cuda::{nvrtc, Context};

fn context() -> Option<Context> {
    match Context::new(0) {
        Ok(c) => Some(c),
        Err(e) => {
            eprintln!("SKIPPED, not passed: no CUDA device ({e})");
            None
        }
    }
}

#[test]
fn vector_add_cold_then_warm() {
    let Some(ctx) = context() else { return };
    let (dll, ver) = match nvrtc::version() {
        Ok(v) => v,
        Err(e) => {
            eprintln!("SKIPPED, not passed: no NVRTC ({e})");
            return;
        }
    };
    eprintln!("device: {} (cc {}.{}, {} SMs, {} MiB); NVRTC {}.{} from {}", ctx.name, ctx.cc.0, ctx.cc.1, ctx.sms,
              ctx.total_mem >> 20, ver.0, ver.1, dll.display());

    let cache = std::env::temp_dir().join(format!("shadow-kernel-cache-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&cache);
    std::env::set_var("SHADOW_KERNEL_CACHE", &cache);
    let f = file("vector_add.cu").unwrap();

    let cold = Program::load(&ctx, f).expect("cold compile");
    assert!(!cold.compiled.from_cache, "the first compile must not come from a fresh cache");
    eprintln!("cold: {:.1} ms, {} bytes of {}", cold.compiled.compile_ms, cold.compiled.image.len(),
              if cold.compiled.is_ptx { "PTX" } else { "CUBIN" });
    let warm = Program::load(&ctx, f).expect("warm load");
    assert!(warm.compiled.from_cache, "the second load must come from the cache");
    assert_eq!(warm.compiled.key, cold.compiled.key);
    assert_eq!(warm.compiled.image, cold.compiled.image);

    for prog in [&cold, &warm] {
        let n: u32 = 1 << 20;
        let a: Vec<f32> = (0..n).map(|i| i as f32 * 0.5).collect();
        let b: Vec<f32> = (0..n).map(|i| 3.0 - i as f32).collect();
        let bytes = |v: &[f32]| v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>();
        let mut da = ctx.alloc(n as usize * 4).unwrap();
        let mut db = ctx.alloc(n as usize * 4).unwrap();
        let mut dout = ctx.alloc(n as usize * 4).unwrap();
        da.upload(&bytes(&a)).unwrap();
        db.upload(&bytes(&b)).unwrap();
        dout.zero().unwrap();
        let block = 256u32;
        // SAFETY: three buffers of n f32 each on this context, and the kernel bounds-checks i < n.
        unsafe {
            prog.launch("vector_add", (n.div_ceil(block), 1, 1), (block, 1, 1), 0, None,
                        &[Arg::Ptr(da.ptr), Arg::Ptr(db.ptr), Arg::Ptr(dout.ptr), Arg::U32(n)])
                .unwrap();
        }
        ctx.synchronize().unwrap();
        let mut out = vec![0u8; n as usize * 4];
        dout.download(&mut out).unwrap();
        for i in 0..n as usize {
            let got = f32::from_le_bytes(out[4 * i..4 * i + 4].try_into().unwrap());
            assert_eq!(got, a[i] + b[i], "element {i}");
        }
    }
    let _ = std::fs::remove_dir_all(&cache);
}

#[test]
fn a_wrong_argument_list_is_refused_before_the_driver_sees_it() {
    let Some(ctx) = context() else { return };
    if nvrtc::version().is_err() {
        eprintln!("SKIPPED, not passed: no NVRTC");
        return;
    }
    let prog = Program::load(&ctx, file("vector_add.cu").unwrap()).unwrap();
    // SAFETY: the launch is refused before any pointer is used.
    let too_few = unsafe { prog.launch("vector_add", (1, 1, 1), (1, 1, 1), 0, None, &[Arg::U32(1)]) };
    assert!(too_few.unwrap_err().contains("takes 4 arguments"));
    // SAFETY: as above.
    let wrong_type = unsafe {
        prog.launch("vector_add", (1, 1, 1), (1, 1, 1), 0, None, &[Arg::Ptr(0), Arg::Ptr(0), Arg::Ptr(0), Arg::F32(1.0)])
    };
    assert!(wrong_type.unwrap_err().contains("does not fit"));
}

#[test]
fn a_compile_error_returns_the_log() {
    if Context::new(0).is_err() || nvrtc::version().is_err() {
        eprintln!("SKIPPED, not passed: no CUDA device or no NVRTC");
        return;
    }
    let dir = std::env::temp_dir().join(format!("shadow-kernel-bad-{}", std::process::id()));
    let e = nvrtc::compile("bad.cu", "extern \"C\" __global__ void k() { this is not C++; }", (12, 0), &dir).unwrap_err();
    assert!(e.contains("NVRTC log for bad.cu"), "{e}");
    let _ = std::fs::remove_dir_all(&dir);
}
