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
    if args.first().map(String::as_str) == Some("dump") && args.len() >= 4 {
        return dump(&PathBuf::from(&args[1]), &args[2], &PathBuf::from(&args[3]), &args[4..]);
    }
    if args.first().map(String::as_str) == Some("compare") && args.len() >= 3 {
        return compare(&PathBuf::from(&args[1]), &PathBuf::from(&args[2]), mode(&args), &args[3..]);
    }
    if args.first().map(String::as_str) == Some("generate") && args.len() >= 3 {
        return generate(&PathBuf::from(&args[1]), &args[2], &args[3..]);
    }
    if args.first().map(String::as_str) == Some("serve") && args.len() >= 2 {
        return serve(&PathBuf::from(&args[1]), &args[2..]);
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

/// `shadow dump MODEL NAME OUT [--rows a,b,c | --expert E]`: a tensor (or some rows of it, or one expert's slice of a
/// fused expert tensor) dequantized to little-endian f32 in OUT, with "ROWS COLS" printed.  The row length is the
/// tensor's first ggml dimension, so a `[in, out]` matrix comes out as `out` rows of `in` weights: the layout PyTorch's
/// `nn.Linear.weight` has.  This is how the reference is run on Shadow's own (dequantized) weights.
fn dump(path: &std::path::Path, name: &str, out: &std::path::Path, rest: &[String]) -> ExitCode {
    use shadow_model::quant::dequantize;
    use std::io::{Read, Seek, SeekFrom, Write};
    let g = match Gguf::open(path) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("shadow dump: {e}");
            return ExitCode::FAILURE;
        }
    };
    let Some(t) = g.tensor(name) else {
        eprintln!("shadow dump: no tensor {name}");
        return ExitCode::FAILURE;
    };
    let Some((qk, qb)) = t.ty.block() else {
        eprintln!("shadow dump: {name}: unknown type {}", t.ty);
        return ExitCode::FAILURE;
    };
    let cols = t.dims[0];
    let row_bytes = cols / qk * qb;
    // (first byte, byte count) of each chunk to read, in order
    let flag = |f: &str| rest.iter().position(|a| a == f).and_then(|i| rest.get(i + 1));
    let expert_idx: Option<usize> = flag("--expert").map(|e| e.parse::<usize>().expect("expert index"));
    let chunks: Vec<(u64, u64)> = if let Some(r) = flag("--rows") {
        r.split(',').map(|x| (x.parse::<u64>().expect("row index") * row_bytes, row_bytes)).collect()
    } else if let Some(e) = expert_idx {
        let per = t.nbytes() / t.dims[2];
        vec![(e as u64 * per, per)]
    } else {
        vec![(0, t.nbytes())]
    };
    let scale: f32 = name.strip_suffix(".weight")
        .and_then(|p| g.tensor(&format!("{p}.scale")))
        .map(|st| {
            let b = g.read_tensor(st).expect("read .scale");
            let idx = expert_idx.unwrap_or(0);
            f32::from_le_bytes(b[idx * 4..(idx + 1) * 4].try_into().unwrap())
        })
        .unwrap_or(1.0);
    let mut f = std::fs::File::open(&g.shards[t.shard].path).expect("open shard");
    let mut o = std::io::BufWriter::new(std::fs::File::create(out).expect("create output"));
    let mut rows = 0u64;
    for (start, len) in chunks {
        let mut buf = vec![0u8; len as usize];
        f.seek(SeekFrom::Start(t.offset + start)).unwrap();
        f.read_exact(&mut buf).unwrap();
        let mut y = vec![0f32; (len / qb * qk) as usize];
        dequantize(t.ty, &buf, &mut y).expect("dequantize");
        for v in &y {
            let sv = *v * scale;
            o.write_all(&sv.to_le_bytes()).unwrap();
        }
        rows += len / row_bytes;
    }
    o.flush().unwrap();
    println!("{rows} {cols}");
    ExitCode::SUCCESS
}

/// Where a forward pass runs: the CPU reference, the CPU with its matrix-vector products on the GPU (`--gpu-matvec`),
/// or the whole forward pass on the GPU (`--gpu`).
#[derive(Clone, Copy, PartialEq)]
enum Mode {
    Cpu,
    GpuMatvec,
    Gpu,
}

fn mode(args: &[String]) -> Mode {
    if args.iter().any(|a| a == "--gpu-matvec") {
        Mode::GpuMatvec
    } else if args.iter().any(|a| a == "--gpu") {
        Mode::Gpu
    } else {
        Mode::Cpu
    }
}

/// The positions a GPU session is made for: the prompt plus what is generated, rounded up, at least 4,096.
fn gpu_capacity(n: usize) -> usize {
    (n + 64).max(4096).next_multiple_of(1024)
}

/// The stage 1 gate: the reference fixture's prompt through Shadow's forward pass, stream by stream.
fn compare(path: &std::path::Path, fixture: &std::path::Path, mode: Mode, rest: &[String]) -> ExitCode {
    use shadow::reference::{compare, compare_stepper, print, Fixture};
    use shadow_model::forward::{CpuBackend, Model};
    let fx = match Fixture::load(fixture) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("shadow compare: {fixture:?}: {e}");
            return ExitCode::FAILURE;
        }
    };
    let g = match Gguf::open(path) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("shadow compare: {e}");
            return ExitCode::FAILURE;
        }
    };
    let model = Model::load(g).expect("model");
    println!("fixture: {:?} ({} tokens, {} layers)", fx.prompt.chars().take(120).collect::<String>(), fx.ids.len(), fx.layers);
    let t0 = std::time::Instant::now();
    let report = match mode {
        Mode::Cpu => compare(&model, &CpuBackend, &fx),
        Mode::GpuMatvec => {
            let ctx = shadow_cuda::Context::new(0).expect("--gpu-matvec");
            let gpu = shadow_cuda::backend::GpuBackend::new(&ctx, 8 << 30).expect("gpu backend");
            compare(&model, &gpu, &fx)
        }
        Mode::Gpu => {
            let ctx = shadow_cuda::Context::new(0).expect("--gpu");
            // tier flags (--ram-tier-gib, --placement, --cpu-experts ...) put the fixture through those paths too
            let tiers = shadow_cuda::session::Tiers::from_args(rest).expect("tier flags");
            let mut s = shadow_cuda::session::GpuSession::with_tiers(&ctx, &model, gpu_capacity(fx.ids.len()), tiers)
                .expect("gpu session");
            compare_stepper(&mut s, &fx)
        }
    };
    print(&report);
    println!("({:.1} s)", t0.elapsed().as_secs_f64());
    ExitCode::SUCCESS
}

/// `shadow serve MODEL [--host H] [--port P] [--ctx N] [tier flags]`: the HTTP server (shadow-serve) on the GPU
/// session.  Port 8081 by default - 8080 is where the user's other server lives.
fn serve(path: &std::path::Path, rest: &[String]) -> ExitCode {
    let flag = |f: &str| rest.iter().position(|a| a == f).and_then(|i| rest.get(i + 1));
    let host = flag("--host").map(String::as_str).unwrap_or("127.0.0.1");
    let port: u16 = flag("--port").and_then(|v| v.parse().ok()).unwrap_or(8081);
    let n_ctx: usize = flag("--ctx").and_then(|v| v.parse().ok()).unwrap_or(16384);
    let addr = match format!("{host}:{port}").parse() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("shadow serve: {host}:{port}: {e}");
            return ExitCode::FAILURE;
        }
    };
    let tiers = match shadow_cuda::session::Tiers::from_args(rest) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("shadow serve: {e}");
            return ExitCode::FAILURE;
        }
    };
    let cfg = shadow_serve::ServeConfig {
        engine: shadow_serve::engine::Config {
            model: path.to_path_buf(), n_ctx: gpu_capacity(n_ctx), tiers,
            spec: flag("--spec").and_then(|v| v.parse().ok()).unwrap_or(0),
        },
        addr,
    };
    match shadow_serve::run(cfg) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("shadow serve: {e}");
            ExitCode::FAILURE
        }
    }
}

/// Greedy decoding.  `-n N` tokens (default 8); `--chat` wraps the prompt in the chat template (thinking off).
/// `--gpu` runs the whole forward pass on GPU 0; `--gpu-matvec` only its matrix-vector products; neither: the stage 1
/// CPU reference, slow by design.
fn generate(path: &std::path::Path, prompt: &str, rest: &[String]) -> ExitCode {
    use shadow_model::chat::{render_for_arch, Message, Options, Role};
    // "@FILE": the prompt is that file's text (long prompts, without the command line's quoting)
    let from_file = prompt.strip_prefix('@').map(std::fs::read_to_string);
    let prompt = match &from_file {
        Some(Ok(t)) => t.as_str(),
        Some(Err(e)) => {
            eprintln!("shadow generate: {prompt}: {e}");
            return ExitCode::FAILURE;
        }
        None => prompt,
    };
    use shadow_model::forward::{check_layout, CpuBackend, Model, Session, Stepper};
    use shadow_model::tokenizer::Tokenizer;
    use std::io::Write;
    let n_new: usize = rest.iter().position(|a| a == "-n").and_then(|i| rest.get(i + 1)).and_then(|v| v.parse().ok())
        .unwrap_or(8);
    let chat = rest.iter().any(|a| a == "--chat");
    let mode = mode(rest);
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
        render_for_arch(&model.cfg.arch, &[Message::new(Role::User, prompt)], &o).expect("template")
    } else {
        prompt.to_string()
    };
    let ids = tok.encode(&text, chat);
    eprintln!("prompt: {} tokens {:?}", ids.len(), &ids[..ids.len().min(16)]);
    let ctx = if mode == Mode::Cpu {
        None
    } else {
        match shadow_cuda::Context::new(0) {
            Ok(c) => Some(c),
            Err(e) => {
                eprintln!("shadow generate: GPU: {e}");
                return ExitCode::FAILURE;
            }
        }
    };
    if let Some(c) = &ctx {
        let (free, _) = c.mem_info().expect("VRAM");
        eprintln!("GPU: {} - {:.2} GiB free", c.name, free as f64 / GIB);
    }
    let backend = match (&ctx, mode) {
        (Some(c), Mode::GpuMatvec) => Some(shadow_cuda::backend::GpuBackend::new(c, 8 << 30).expect("gpu backend")),
        _ => None,
    };
    let t_up = std::time::Instant::now();
    let mut s: Box<dyn Stepper + '_> = match (&ctx, &backend) {
        (Some(c), None) => {
            let tiers = match shadow_cuda::session::Tiers::from_args(rest) {
                Ok(t) => t,
                Err(e) => {
                    eprintln!("shadow generate: {e}");
                    return ExitCode::FAILURE;
                }
            };
            let gs = match shadow_cuda::session::GpuSession::with_tiers(c, &model, gpu_capacity(ids.len() + n_new), tiers) {
                Ok(s) => s,
                Err(e) => {
                    eprintln!("shadow generate: --gpu: {e}");
                    return ExitCode::FAILURE;
                }
            };
            eprintln!("GPU session: {:.2} GiB of weights resident in {:.1} s; {}", gs.resident_bytes as f64 / GIB,
                      t_up.elapsed().as_secs_f64(), gs.report());
            Box::new(gs)
        }
        (_, Some(b)) => Box::new(Session::new(&model, b)),
        _ => Box::new(Session::new(&model, &CpuBackend)),
    };
    let t1 = std::time::Instant::now();
    let mut logits = s.prefill(&ids);
    eprintln!("prompt read in {:.1} s ({:.3} s/token)", t1.elapsed().as_secs_f64(),
              t1.elapsed().as_secs_f64() / ids.len() as f64);
    if !s.report().is_empty() {
        eprintln!("after the prompt: {}", s.report());
    }
    print!("{text}");
    let _ = std::io::stdout().flush();
    let mut out = Vec::new();
    let t2 = std::time::Instant::now();
    for _ in 0..n_new {
        let next = logits.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).map(|(i, _)| i as u32).unwrap();
        out.push(next);
        print!("{}", String::from_utf8_lossy(&tok.decode_bytes(&[next])));
        let _ = std::io::stdout().flush();
        if tok.is_eog(next) {
            break;
        }
        logits = s.step(next);
    }
    println!();
    eprintln!("generated {} tokens {:?} in {:.1} s ({:.3} s/token)", out.len(), out, t2.elapsed().as_secs_f64(),
              t2.elapsed().as_secs_f64() / out.len().max(1) as f64);
    if !s.report().is_empty() {
        eprintln!("at the end: {}", s.report());
    }
    if let Some(p) = rest.iter().position(|a| a == "--record-routing").and_then(|i| rest.get(i + 1)) {
        let Some(rc) = s.routing() else {
            eprintln!("shadow generate: --record-routing needs --gpu");
            return ExitCode::FAILURE;
        };
        let path = std::path::Path::new(p);
        let mut all = rc.clone();
        if path.is_file() {
            match shadow_model::placement::RoutingCounts::load(path).and_then(|old| all.merge(&old).map(|_| ())) {
                Ok(()) => {}
                Err(e) => {
                    eprintln!("shadow generate: --record-routing: {e}");
                    return ExitCode::FAILURE;
                }
            }
        }
        if let Err(e) = all.save(path) {
            eprintln!("shadow generate: --record-routing: {e}");
            return ExitCode::FAILURE;
        }
        eprintln!("routing: {} tokens added to {p} ({} in all)", rc.tokens, all.tokens);
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
