//! The GPU dequant kernels against the Rust reference (shadow_model::quant), bit for bit: every block of
//! eval/fixtures/dequant.txt of the types the kernels cover (themselves checked bit for bit against ggml's reference),
//! plus Q2_0 on real blocks of the model when it is on this machine.

use shadow_cuda::kernels::{file, Arg, Program, DEQUANT};
use shadow_cuda::{nvrtc, Context};
use shadow_model::gguf::{GgmlType, Gguf};
use shadow_model::quant::dequantize;

fn hex(s: &str) -> Vec<u8> {
    (0..s.len()).step_by(2).map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap()).collect()
}

fn type_id(name: &str) -> u32 {
    (0..64).find(|&i| GgmlType::from_id(i).name() == name).unwrap()
}

fn gpu(ctx: &Context, prog: &Program, entry: &str, per_block: u32, data: &[u8], n_blocks: u32) -> Vec<f32> {
    let mut din = ctx.alloc(data.len()).unwrap();
    let n = (n_blocks * per_block) as usize;
    let dout = ctx.alloc(n * 4).unwrap();
    din.upload(data).unwrap();
    // SAFETY: n_blocks whole blocks in, n_blocks * per_block floats out; one thread block per quant block.
    unsafe {
        prog.launch(entry, (n_blocks, 1, 1), (per_block, 1, 1), 0, None,
                    &[Arg::Ptr(din.ptr), Arg::Ptr(dout.ptr), Arg::U32(n_blocks)]).unwrap();
    }
    ctx.synchronize().unwrap();
    let mut b = vec![0u8; n * 4];
    dout.download(&mut b).unwrap();
    b.chunks(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect()
}

fn same(gpu: &[f32], cpu: &[f32], what: &str) {
    for (i, (g, c)) in gpu.iter().zip(cpu).enumerate() {
        assert_eq!(g.to_bits(), c.to_bits(), "{what} element {i}: gpu {g} cpu {c}");
    }
}

#[test]
fn gpu_dequant_matches_the_reference_bit_for_bit() {
    let ctx = match Context::new(0) {
        Ok(c) => c,
        Err(e) => return eprintln!("SKIPPED, not passed: no CUDA device ({e})"),
    };
    if let Err(e) = nvrtc::version() {
        return eprintln!("SKIPPED, not passed: no NVRTC ({e})");
    }
    let prog = Program::load(&ctx, file("dequant.cu").unwrap()).unwrap();
    let text = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/../../eval/fixtures/dequant.txt")).unwrap();
    let lines: Vec<&str> = text.lines().filter(|l| !l.starts_with('#') && !l.is_empty()).collect();
    let mut checked = std::collections::BTreeMap::new();
    for rec in lines.chunks(3) {
        let head: Vec<&str> = rec[0].split_whitespace().collect();
        let ty = head[0];
        let bytes = hex(rec[1]);
        let mut cpu = vec![0f32; if ty == "BF16" { 1 } else { DEQUANT.iter().find(|d| d.1 == ty).map_or(0, |d| d.2 as usize) }];
        if cpu.is_empty() {
            continue;
        }
        dequantize(GgmlType::from_id(type_id(ty)), &bytes, &mut cpu).unwrap();
        let got = if ty == "BF16" {
            gpu(&ctx, &prog, "dequant_bf16", 1, &bytes, 1)
        } else {
            let d = DEQUANT.iter().find(|d| d.1 == ty).unwrap();
            gpu(&ctx, &prog, d.0, d.2, &bytes, 1)
        };
        same(&got, &cpu, &format!("{ty} {} block {}", head[1], head[2]));
        *checked.entry(ty).or_insert(0) += 1;
    }
    eprintln!("fixture blocks checked on the GPU: {checked:?}");
    assert_eq!(checked.len(), 6, "Q4_K, Q5_K, Q6_K, IQ4_NL, IQ4_XS, BF16");

    // Q2_0: no fixture exists, so against the Rust implementation on real blocks - a whole expert of blk.1's down
    // projection - when the model is here.
    let p = std::env::var("SHADOW_MODEL").unwrap_or_else(|_| {
        "J:/Strata-work/models/swift15/IQ3_S/Swift-Qwen3.8-Flash-Next-GSQ-RCO-abliterated-IQ3_S-00001-of-00002.gguf".into()
    });
    if !std::path::Path::new(&p).is_file() {
        return eprintln!("Q2_0 on real blocks SKIPPED, not passed: no model at {p}");
    }
    let g = Gguf::open(std::path::Path::new(&p)).unwrap();
    let t = g.tensor("blk.1.ffn_down_exps.weight").unwrap();
    assert_eq!(t.ty.name(), "Q2_0");
    let n_blocks: u32 = 640 * 2560 / 64;
    use std::io::{Read, Seek, SeekFrom};
    let mut f = std::fs::File::open(&g.shards[t.shard].path).unwrap();
    let mut data = vec![0u8; n_blocks as usize * 18];
    f.seek(SeekFrom::Start(t.offset)).unwrap();
    f.read_exact(&mut data).unwrap();
    let mut cpu = vec![0f32; n_blocks as usize * 64];
    dequantize(t.ty, &data, &mut cpu).unwrap();
    same(&gpu(&ctx, &prog, "dequant_q2_0", 64, &data, n_blocks), &cpu, "Q2_0 blk.1 expert 0");
    eprintln!("Q2_0: {} real blocks checked", n_blocks);
}
