//! `shadow`: the engine's command line.  Stage 0 has one command.
//!
//!     shadow inspect MODEL.gguf [--tensors] [--kv]
//!
//! `inspect` reads only the headers and prints what the plan's stage 0 gate asks for: every tensor accounted for,
//! the bytes by type, by layer kind and by role, and the always-resident share that decides the VRAM budget.

use shadow_model::gguf::{block_index, Gguf, TensorInfo, Value};
use std::collections::BTreeMap;
use std::path::PathBuf;
use std::process::ExitCode;

const GIB: f64 = 1024.0 * 1024.0 * 1024.0;

fn gib(b: u64) -> String {
    format!("{:.3} GiB", b as f64 / GIB)
}

/// The tensor's role inside its block: the name with `blk.N.` and the trailing `.weight`/`.bias` taken off.
fn role(name: &str) -> String {
    let rest = name.strip_prefix("blk.").and_then(|r| r.split_once('.')).map(|(_, r)| r).unwrap_or(name);
    rest.to_string()
}

fn is_routed_expert(name: &str) -> bool {
    name.contains("_exps")
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.first().map(String::as_str) == Some("kv") && args.len() == 3 {
        return kv(&PathBuf::from(&args[1]), &args[2]);
    }
    if args.first().map(String::as_str) == Some("stats") && args.len() >= 3 {
        return stats(&PathBuf::from(&args[1]), &args[2..]);
    }
    if args.first().map(String::as_str) == Some("generate") && args.len() >= 3 {
        return generate(&PathBuf::from(&args[1]), &args[2], &args[3..]);
    }
    if args.first().map(String::as_str) != Some("inspect") || args.len() < 2 {
        eprintln!("usage: shadow inspect MODEL.gguf [--tensors] [--kv]\n       shadow kv MODEL.gguf KEY");
        return ExitCode::from(2);
    }
    let path = PathBuf::from(&args[1]);
    let all_tensors = args.iter().any(|a| a == "--tensors");
    let all_kv = args.iter().any(|a| a == "--kv");
    let g = match Gguf::open(&path) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("shadow inspect: {e}");
            return ExitCode::FAILURE;
        }
    };
    inspect(&g, all_tensors, all_kv);
    ExitCode::SUCCESS
}

/// Greedy decoding.  `-n N` tokens (default 8); `--chat` wraps the prompt in the chat template (thinking off);
/// `--gpu` runs every matrix-vector product on GPU 0 (the dense weights resident in VRAM), the rest on the CPU.
/// Without it: the stage 1 CPU reference, slow by design.
fn generate(path: &std::path::Path, prompt: &str, rest: &[String]) -> ExitCode {
    use shadow_model::chat::{render, Message, Options, Role};
    use shadow_model::forward::{check_layout, CpuBackend, Model, Session};
    use shadow_model::tokenizer::Tokenizer;
    use std::io::Write;
    let n_new: usize = rest.iter().position(|a| a == "-n").and_then(|i| rest.get(i + 1)).and_then(|v| v.parse().ok())
        .unwrap_or(8);
    let chat = rest.iter().any(|a| a == "--chat");
    let use_gpu = rest.iter().any(|a| a == "--gpu");
    let g = match Gguf::open(path) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("shadow generate: {e}");
            return ExitCode::FAILURE;
        }
    };
    let tok = Tokenizer::from_gguf(&g).expect("tokenizer");
    let t0 = std::time::Instant::now();
    let model = match Model::load(g) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("shadow generate: {e}");
            return ExitCode::FAILURE;
        }
    };
    if let Err(e) = check_layout(&model.g, &model.cfg) {
        eprintln!("shadow generate: {e}");
        return ExitCode::FAILURE;
    }
    eprintln!("loaded {:.2} GiB of dense weights in {:.1} s", model.dense_bytes() as f64 / GIB, t0.elapsed().as_secs_f64());
    let text = if chat {
        let o = Options { thinking: Some(false), ..Options::default() };
        render(&[Message::new(Role::User, prompt)], &o).expect("template")
    } else {
        prompt.to_string()
    };
    let ids = tok.encode(&text, chat);
    eprintln!("prompt: {} tokens {:?}", ids.len(), &ids[..ids.len().min(16)]);
    let ctx = if use_gpu {
        match shadow_cuda::Context::new(0) {
            Ok(c) => Some(c),
            Err(e) => {
                eprintln!("shadow generate: --gpu: {e}");
                return ExitCode::FAILURE;
            }
        }
    } else {
        None
    };
    let gpu = match &ctx {
        Some(c) => {
            let (free, _) = c.mem_info().expect("VRAM");
            let budget = free.saturating_sub(1 << 30); // leave 1 GiB for scratch, the desktop and the driver
            eprintln!("GPU: {} - {:.2} GiB free, resident budget {:.2} GiB", c.name, free as f64 / GIB, budget as f64 / GIB);
            match shadow_cuda::backend::GpuBackend::new(c, budget) {
                Ok(b) => Some(b),
                Err(e) => {
                    eprintln!("shadow generate: --gpu: {e}");
                    return ExitCode::FAILURE;
                }
            }
        }
        None => None,
    };
    let be: &dyn shadow_model::forward::Backend = match &gpu {
        Some(g) => g,
        None => &CpuBackend,
    };
    let mut s = Session::new(&model, be);
    let t1 = std::time::Instant::now();
    let mut logits = Vec::new();
    for &id in &ids {
        logits = s.step(id);
    }
    eprintln!("prompt read in {:.1} s ({:.2} s/token)", t1.elapsed().as_secs_f64(),
              t1.elapsed().as_secs_f64() / ids.len() as f64);
    print!("{text}");
    let _ = std::io::stdout().flush();
    let mut out = Vec::new();
    let t2 = std::time::Instant::now();
    for _ in 0..n_new {
        let next = logits.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).map(|(i, _)| i as u32).unwrap();
        out.push(next);
        print!("{}", String::from_utf8_lossy(&tok.decode_bytes(&[next])));
        let _ = std::io::stdout().flush();
        if Some(next) == tok.eos {
            break;
        }
        logits = s.step(next);
    }
    println!();
    eprintln!("generated {} tokens {:?} in {:.1} s ({:.2} s/token)", out.len(), out, t2.elapsed().as_secs_f64(),
              t2.elapsed().as_secs_f64() / out.len().max(1) as f64);
    if let Some(g) = &gpu {
        eprintln!("GPU: {:.2} GiB of weights resident, {} uploads", g.resident_bytes.get() as f64 / GIB, g.uploads.get());
    }
    ExitCode::SUCCESS
}

/// One key's whole value, raw: a string as it is, an array one element per line.  For reading the chat template and
/// the vocabulary without a script.
fn kv(path: &std::path::Path, key: &str) -> ExitCode {
    let g = match Gguf::open(path) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("shadow kv: {e}");
            return ExitCode::FAILURE;
        }
    };
    match g.get(key) {
        Some(Value::Str(s)) => print!("{s}"),
        Some(Value::Array(a)) => {
            for v in a {
                match v {
                    Value::Str(s) => println!("{s}"),
                    v => println!("{}", v.brief()),
                }
            }
        }
        Some(v) => println!("{}", v.brief()),
        None => {
            eprintln!("shadow kv: no key {key}");
            return ExitCode::FAILURE;
        }
    }
    ExitCode::SUCCESS
}

/// min / mean / max and the first values of float tensors (F32, F16, BF16): for questions like "are the norm weights
/// stored zero-centred?" that only the numbers answer.
fn stats(path: &std::path::Path, names: &[String]) -> ExitCode {
    use shadow_model::half::{bf16_to_f32, f16_to_f32};
    let g = match Gguf::open(path) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("shadow stats: {e}");
            return ExitCode::FAILURE;
        }
    };
    for name in names {
        let Some(t) = g.tensor(name) else {
            println!("{name}: no such tensor");
            continue;
        };
        if t.nbytes() > 256 << 20 {
            println!("{name}: {} bytes, too big to summarise here", t.nbytes());
            continue;
        }
        let b = match g.read_tensor(t) {
            Ok(b) => b,
            Err(e) => {
                println!("{name}: {e}");
                continue;
            }
        };
        let v: Vec<f32> = match t.ty.name().as_str() {
            "F32" => b.chunks(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect(),
            "F16" => b.chunks(2).map(|c| f16_to_f32(u16::from_le_bytes(c.try_into().unwrap()))).collect(),
            "BF16" => b.chunks(2).map(|c| bf16_to_f32(u16::from_le_bytes(c.try_into().unwrap()))).collect(),
            ty => {
                println!("{name}: {ty}, not a float type");
                continue;
            }
        };
        let (mn, mx) = v.iter().fold((f32::MAX, f32::MIN), |(a, b), &x| (a.min(x), b.max(x)));
        let mean = v.iter().map(|&x| x as f64).sum::<f64>() / v.len() as f64;
        let head: Vec<String> = v.iter().take(6).map(|x| format!("{x:.5}")).collect();
        println!("{name} {} {:?}: min {mn:.5} mean {mean:.5} max {mx:.5}  [{}]", t.ty, t.dims, head.join(", "));
    }
    ExitCode::SUCCESS
}

fn inspect(g: &Gguf, all_tensors: bool, all_kv: bool) {
    println!("== files");
    for s in &g.shards {
        println!("  {}  v{}  {} bytes, data at {}, alignment {}", s.path.display(), s.version, s.file_len, s.data_start,
                 s.alignment);
    }
    let arch = g.get("general.architecture").and_then(Value::as_str).unwrap_or("?").to_string();

    // Every key, each shortened by `brief` (arrays print their length and first items, long strings their start).
    // `--kv` prints the arrays whole.
    println!("\n== key-values ({} total; arch {arch})", g.kv.len());
    for (k, v) in &g.kv {
        match (all_kv, v.as_array()) {
            (true, Some(a)) => println!("  {k} = {:?}", a.iter().map(Value::brief).collect::<Vec<_>>()),
            _ => println!("  {k} = {}", v.brief()),
        }
    }

    let total = g.tensor_bytes();
    let unknown: Vec<&TensorInfo> = g.tensors.iter().filter(|t| t.nbytes_layout.is_none()).collect();
    let padding: u64 = g.tensors.iter().map(|t| t.nbytes_span - t.nbytes()).sum();
    let file_total: u64 = g.shards.iter().map(|s| s.file_len).sum();
    println!("\n== accounting");
    println!("  tensors: {}", g.tensors.len());
    println!("  tensor bytes: {} ({} bytes)", gib(total), total);
    println!("  files: {} ({} bytes); headers + padding: {} bytes", gib(file_total), file_total,
             file_total - total);
    println!("  alignment padding inside the data sections: {padding} bytes");
    println!("  tensors whose size came from the offsets, not the type: {}", unknown.len());
    for t in unknown.iter().take(10) {
        println!("    {} {:?} {} -> {} bytes", t.name, t.dims, t.ty, t.nbytes_span);
    }

    let mut by_type: BTreeMap<String, (usize, u64)> = BTreeMap::new();
    for t in &g.tensors {
        let e = by_type.entry(t.ty.name()).or_default();
        e.0 += 1;
        e.1 += t.nbytes();
    }
    println!("\n== by type");
    for (ty, (n, b)) in &by_type {
        println!("  {ty:<8} {n:>5} tensors  {}", gib(*b));
    }

    // Layers: the K (full attention) layers are the ones that carry attention Q/K/V tensors.  Taken from the file,
    // not from `full_attention_interval`, and the two are checked against each other below.
    let n_layers = g.arch_key("block_count").and_then(Value::as_u64).unwrap_or(0) as usize;
    let mut layer_bytes = vec![0u64; n_layers];
    let mut layer_expert = vec![0u64; n_layers];
    let mut is_k = vec![false; n_layers];
    let mut non_layer = Vec::new();
    let mut by_role: BTreeMap<String, (usize, u64, String)> = BTreeMap::new();
    for t in &g.tensors {
        match block_index(&t.name) {
            Some(l) if l < n_layers => {
                layer_bytes[l] += t.nbytes();
                if is_routed_expert(&t.name) {
                    layer_expert[l] += t.nbytes();
                }
                if t.name.contains("attn_q.") || t.name.contains("attn_k.") || t.name.contains("attn_v.") {
                    is_k[l] = true;
                }
                let e = by_role.entry(role(&t.name)).or_insert((0, 0, String::new()));
                e.0 += 1;
                e.1 += t.nbytes();
                if e.2.is_empty() {
                    e.2 = format!("{} {:?}", t.ty, t.dims);
                }
            }
            _ => non_layer.push(t),
        }
    }
    let interval = g.arch_key("full_attention_interval").and_then(Value::as_u64);
    let k_layers: Vec<usize> = (0..n_layers).filter(|&l| is_k[l]).collect();
    println!("\n== layers ({n_layers})");
    println!("  K (attention tensors present): {} {:?}", k_layers.len(), k_layers);
    if let Some(i) = interval {
        let rule: Vec<usize> = (0..n_layers).filter(|&l| (l as u64) % i == i - 1).collect();
        println!("  full_attention_interval {i}: layer % {i} == {} {}", i - 1,
                 if rule == k_layers { "matches the tensors" } else { "DOES NOT match the tensors" });
    }
    let sum = |sel: &dyn Fn(usize) -> bool, v: &[u64]| -> u64 { (0..n_layers).filter(|&l| sel(l)).map(|l| v[l]).sum() };
    let (nj, nk) = (n_layers - k_layers.len(), k_layers.len());
    let (jb, kb) = (sum(&|l| !is_k[l], &layer_bytes), sum(&|l| is_k[l], &layer_bytes));
    let (je, ke) = (sum(&|l| !is_k[l], &layer_expert), sum(&|l| is_k[l], &layer_expert));
    if nj > 0 {
        println!("  J: {nj} layers, {} ({} per layer), routed experts {} ({} per layer)", gib(jb), gib(jb / nj as u64),
                 gib(je), gib(je / nj as u64));
    }
    if nk > 0 {
        println!("  K: {nk} layers, {} ({} per layer), routed experts {} ({} per layer)", gib(kb), gib(kb / nk as u64),
                 gib(ke), gib(ke / nk as u64));
    }

    println!("\n== roles inside a block (count, total, first seen)");
    for (r, (n, b, first)) in &by_role {
        println!("  {r:<36} {n:>3}  {:>12}  {first}", gib(*b));
    }

    let nl_bytes: u64 = non_layer.iter().map(|t| t.nbytes()).sum();
    println!("\n== outside the layers: {} in {} tensors", gib(nl_bytes), non_layer.len());
    for t in &non_layer {
        println!("  {:<36} {:<6} {:?}  {}", t.name, t.ty.name(), t.dims, gib(t.nbytes()));
    }

    let experts: u64 = je + ke;
    let dense_layers: u64 = jb + kb - experts;
    println!("\n== the budget that decides placement");
    println!("  routed experts (tiered: VRAM / pinned RAM / drive): {}", gib(experts));
    println!("  everything else inside the layers (always resident on the GPU): {}", gib(dense_layers));
    println!("  outside the layers (embeddings, head, norms): {}", gib(nl_bytes));
    println!("  total: {}", gib(total));

    if all_tensors {
        println!("\n== tensors");
        for t in &g.tensors {
            println!("  {:<44} {:<7} {:<24} shard {} @ {:>12}  {} bytes", t.name, t.ty.name(), format!("{:?}", t.dims),
                     t.shard, t.offset, t.nbytes());
        }
    }
}
