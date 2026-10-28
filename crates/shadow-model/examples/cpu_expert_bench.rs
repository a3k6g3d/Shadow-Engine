//! How fast does the CPU run real routed experts (cpu_expert), per layer type mix?
//!
//!   cargo run --release -p shadow-model --example cpu_expert_bench -- MODEL.gguf [threads] [experts]

use shadow_model::cpu_expert::{ExpertPool, ExpertRef, ExpertShape};
use shadow_model::gguf::Gguf;
use std::os::windows::fs::FileExt;

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let g = Gguf::open(std::path::Path::new(&a[1])).unwrap();
    let threads: usize = a.get(2).and_then(|v| v.parse().ok()).unwrap_or(8);
    let n: usize = a.get(3).and_then(|v| v.parse().ok()).unwrap_or(4);
    let files: Vec<std::fs::File> = g.shards.iter().map(|s| std::fs::File::open(&s.path).unwrap()).collect();
    let mut pool = ExpertPool::new(threads);
    for l in [5usize, 20, 40] {
        let names = ["gate", "up", "down"].map(|p| format!("blk.{l}.ffn_{p}_exps.weight"));
        let ts: Vec<_> = names.iter().map(|nm| g.tensor(nm).unwrap().clone()).collect();
        let per: Vec<usize> = ts.iter().map(|t| (t.nbytes() / t.dims[2]) as usize).collect();
        let bufs: Vec<Vec<Vec<u8>>> = (0..n).map(|e| ts.iter().zip(&per).map(|(t, &p)| {
            let mut b = vec![0u8; p];
            let mut done = 0;
            while done < p {
                done += files[t.shard].seek_read(&mut b[done..], t.offset + (e * 7 * p + done) as u64).unwrap();
            }
            b
        }).collect()).collect();
        let shape = ExpertShape { hidden: ts[0].dims[0] as usize, ff: ts[0].dims[1] as usize, gate_ty: ts[0].ty.name(),
                                  up_ty: ts[1].ty.name(), down_ty: ts[2].ty.name() };
        let x: Vec<f32> = (0..shape.hidden).map(|i| ((i * 37 % 101) as f32 - 50.0) / 500.0).collect();
        let refs: Vec<ExpertRef> = bufs.iter().map(|b| ExpertRef { gate: b[0].as_ptr(), up: b[1].as_ptr(), down: b[2].as_ptr() }).collect();
        let bytes: usize = per.iter().sum::<usize>() * n;
        let mut best = f64::MAX;
        for _ in 0..20 {
            let t0 = std::time::Instant::now();
            // SAFETY: bufs outlives wait().
            unsafe { pool.submit(&shape, refs.clone(), &x) };
            let out = pool.wait();
            std::hint::black_box(out);
            best = best.min(t0.elapsed().as_secs_f64());
        }
        println!("layer {l} ({}/{}/{}): {n} experts, {:.2} MB, best {:.3} ms, {:.1} GB/s, {:.3} ms/expert", shape.gate_ty,
                 shape.up_ty, shape.down_ty, bytes as f64 / 1e6, best * 1e3, bytes as f64 / best / 1e9, best * 1e3 / n as f64);
    }
}
