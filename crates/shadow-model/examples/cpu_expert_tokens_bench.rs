//! What does the CPU pay for ONE routed expert when T tokens of a prompt chunk route to it (T = 1 is decode; a 2,048-token chunk sends ~32 tokens to each
//! of 512 experts)?  The pool's job is a list of (expert, input) pairs, so T tokens through one expert is T entries over the same weights.
//!
//!   cargo run --release -p shadow-model --example cpu_expert_tokens_bench -- MODEL.gguf [threads]

use shadow_model::cpu_expert::{ExpertPool, ExpertRef, ExpertShape};
use shadow_model::gguf::Gguf;
use std::os::windows::fs::FileExt;

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let g = Gguf::open(std::path::Path::new(&a[1])).unwrap();
    let threads: usize = a.get(2).and_then(|v| v.parse().ok()).unwrap_or(14);
    let files: Vec<std::fs::File> = g.shards.iter().map(|s| std::fs::File::open(&s.path).unwrap()).collect();
    let mut pool = ExpertPool::new(threads);
    for l in [5usize, 20, 40] {
        let names = ["gate", "up", "down"].map(|p| format!("blk.{l}.ffn_{p}_exps.weight"));
        let ts: Vec<_> = names.iter().map(|nm| g.tensor(nm).unwrap().clone()).collect();
        let per: Vec<usize> = ts.iter().map(|t| (t.nbytes() / t.dims[2]) as usize).collect();
        let bufs: Vec<Vec<u8>> = ts.iter().zip(&per).map(|(t, &p)| {
            let mut b = vec![0u8; p];
            let mut done = 0;
            while done < p {
                done += files[t.shard].seek_read(&mut b[done..], t.offset + (7 * p + done) as u64).unwrap();
            }
            b
        }).collect();
        let shape = ExpertShape { hidden: ts[0].dims[0] as usize, ff: ts[0].dims[1] as usize, gate_ty: ts[0].ty.name(),
                                  up_ty: ts[1].ty.name(), down_ty: ts[2].ty.name() };
        let bytes: usize = per.iter().sum();
        let r = ExpertRef { gate: bufs[0].as_ptr(), up: bufs[1].as_ptr(), down: bufs[2].as_ptr() };
        for t in [1usize, 8, 32, 64] {
            let x: Vec<f32> = (0..shape.hidden * t).map(|i| (((i * 37) % 101) as f32 - 50.0) / 500.0).collect();
            let xi: Vec<usize> = (0..t).collect();
            let mut best = f64::MAX;
            for _ in 0..12 {
                let t0 = std::time::Instant::now();
                // SAFETY: bufs outlives wait().
                unsafe { pool.submit_multi(&shape, vec![r; t], xi.clone(), &x) };
                std::hint::black_box(pool.wait());
                best = best.min(t0.elapsed().as_secs_f64());
            }
            println!("layer {l} ({}/{}/{}), {:.2} MB an expert: {t:3} tokens -> {:7.3} ms an expert ({:.3} ms a token)", shape.gate_ty, shape.up_ty, shape.down_ty,
                     bytes as f64 / 1e6, best * 1e3, best * 1e3 / t as f64);
        }
    }
}
