//! shadow-bench: arms of `shadow serve` configurations, measured the way docs/BENCH_NOISE.md says a measurement has to
//! be made on this machine:
//!
//! * **interleaved**: run i of every arm before run i+1 of any, the order rotated each round, so no arm owns the
//!   warm (or cold) end of the session;
//! * **one fresh launch per run**: `shadow serve` under tools/run_capped.ps1, the full launch line recorded;
//! * **a fixed conversation**: the same user turns in the same order, so turn t of one run faces the same prompt as
//!   turn t of every other run of every arm;
//! * **decode_warm**: turn 0 (the cold one) is dropped from every figure;
//! * **paired, turn by turn**: arms are compared at the same run and turn, not median to median;
//! * **the floor**: within one arm, every pair of runs at the same turn - |a - b| / min(a, b) - and its median, p90
//!   and worst.  A difference between arms smaller than the floor is a tie.
//!
//! ```text
//! shadow-bench run --model M --out DIR --arm NAME=FLAGS [--arm NAME=FLAGS ...] [--runs 8] [--turns 6]
//!                  [--tokens 96] [--port 8091]
//! shadow-bench summary DIR
//! ```

use serde_json::{json, Value};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

const TURNS: &[&str] = &[
    "In three sentences, what is a hash table and why is lookup usually fast?",
    "Now give a short Python example of using a dictionary to count words in a sentence.",
    "What happens to lookup speed when many keys collide? Answer briefly.",
    "Name two collision-resolution strategies and one trade-off of each.",
    "Summarise our conversation so far in two sentences.",
    "Suggest a good name for a blog post about this topic.",
    "Write a haiku about hash tables.",
    "Thank you. One last tip for a beginner?",
];

fn usage() -> ! {
    eprintln!("usage: shadow-bench run --model M --out DIR --arm NAME=FLAGS [--arm ...] [--runs 8] [--turns 6] \
               [--tokens 96] [--port 8091]\n       shadow-bench summary DIR");
    std::process::exit(2)
}

fn main() {
    let a: Vec<String> = std::env::args().skip(1).collect();
    match a.first().map(String::as_str) {
        Some("run") => run(&a[1..]),
        Some("summary") if a.len() == 2 => summary(Path::new(&a[1])),
        _ => usage(),
    }
}

// ------------------------------------------------------------------------------------------------------------ HTTP
/// One request to the local server: (status, body).  Connection: close, so the body is whatever arrives until EOF.
fn http(port: u16, method: &str, path: &str, body: Option<&Value>, timeout: Duration) -> Result<(u16, String), String> {
    let mut s = std::net::TcpStream::connect(("127.0.0.1", port)).map_err(|e| e.to_string())?;
    s.set_read_timeout(Some(timeout)).ok();
    let b = body.map(|v| v.to_string()).unwrap_or_default();
    write!(s, "{method} {path} HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\
               Connection: close\r\n\r\n{b}", b.len()).map_err(|e| e.to_string())?;
    let mut out = Vec::new();
    s.read_to_end(&mut out).map_err(|e| e.to_string())?;
    let text = String::from_utf8_lossy(&out).into_owned();
    let (head, body) = text.split_once("\r\n\r\n").ok_or("no HTTP header")?;
    let status = head.split_whitespace().nth(1).and_then(|c| c.parse().ok()).ok_or("no status")?;
    Ok((status, body.to_string()))
}

/// GPU memory in use (MiB), from nvidia-smi; None if it cannot be asked.
fn gpu_used_mib() -> Option<u64> {
    let o = std::process::Command::new("nvidia-smi").args(["--query-gpu=memory.used", "--format=csv,noheader,nounits"])
        .output().ok()?;
    String::from_utf8_lossy(&o.stdout).lines().next()?.trim().parse().ok()
}

fn gib(b: u64) -> f64 {
    b as f64 / (1u64 << 30) as f64
}

// ------------------------------------------------------------------------------------------------------------- run
fn run(a: &[String]) {
    let flag = |f: &str| a.iter().position(|x| x == f).and_then(|i| a.get(i + 1)).cloned();
    let model = flag("--model").unwrap_or_else(|| usage());
    let out = PathBuf::from(flag("--out").unwrap_or_else(|| usage()));
    let runs: usize = flag("--runs").and_then(|v| v.parse().ok()).unwrap_or(8);
    let turns: usize = flag("--turns").and_then(|v| v.parse().ok()).unwrap_or(6).min(TURNS.len());
    let tokens: usize = flag("--tokens").and_then(|v| v.parse().ok()).unwrap_or(96);
    let port: u16 = flag("--port").and_then(|v| v.parse().ok()).unwrap_or(8091);
    let arms: Vec<(String, String)> = a.iter().enumerate().filter(|(_, x)| *x == "--arm")
        .filter_map(|(i, _)| a.get(i + 1)?.split_once('=').map(|(n, f)| (n.to_string(), f.to_string())))
        .collect();
    if arms.is_empty() {
        usage();
    }
    if runs < 2 {
        eprintln!("shadow-bench: --runs must be at least 2 (the floor needs pairs; the protocol says 8)");
        std::process::exit(2);
    }
    let root = std::env::current_dir().unwrap();
    let exe = root.join("target/release/shadow.exe");
    let capped = root.join("tools/run_capped.ps1");
    std::fs::create_dir_all(&out).unwrap();
    let results = out.join("results.jsonl");
    std::fs::write(out.join("plan.json"), json!({
        "model": model, "runs": runs, "turns": turns, "tokens": tokens, "port": port,
        "arms": arms.iter().map(|(n, f)| json!({"name": n, "flags": f})).collect::<Vec<_>>(),
        "prompts": &TURNS[..turns], "started": now(),
    }).to_string()).unwrap();
    println!("{} arms x {runs} runs x {turns} turns, {tokens} tokens per turn; results in {}", arms.len(), results.display());

    for i in 0..runs {
        for k in 0..arms.len() {
            let (name, flags) = &arms[(i + k) % arms.len()]; // rotate who goes first
            // the cap covers a RAM tier: its size plus the process (dense weights, buffers) plus headroom
            let cap = if flags.contains("--ram-tier-gib auto") { 44 } else {
                flags.split_whitespace().skip_while(|w| *w != "--ram-tier-gib").nth(1)
                    .and_then(|g| g.parse::<f64>().ok()).map_or(16, |g| g.ceil() as u64 + 12)
            };
            let log = out.join(format!("{name}-run{i}.log"));
            let args = format!("serve {model} --port {port} {flags}");
            let launch = format!("& '{}' -CapGiB {cap} -Out '{}' -Exe '{}' -Arguments '{}'", capped.display(),
                                 log.display(), exe.display(), args);
            let rec = one_run(name, i, port, &launch, turns, tokens);
            println!("run {i} {name}: {}", rec.get("line").and_then(Value::as_str).unwrap_or(""));
            let mut f = std::fs::OpenOptions::new().create(true).append(true).open(&results).unwrap();
            writeln!(f, "{rec}").unwrap();
        }
    }
    summary(&out);
}

fn now() -> String {
    format!("{}", std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs())
}

/// Waits (up to two minutes) until available RAM stops moving - the previous run's pinned tier being handed back to
/// the system - so every launch starts from the same place.  Returns the settled figure in GiB.
fn settle_ram() -> f64 {
    let t0 = Instant::now();
    let mut last = shadow_cuda::pinned::available_ram();
    loop {
        std::thread::sleep(Duration::from_secs(3));
        let now = shadow_cuda::pinned::available_ram();
        if now.abs_diff(last) < (256 << 20) || t0.elapsed() > Duration::from_secs(120) {
            return gib(now);
        }
        last = now;
    }
}

fn one_run(arm: &str, run: usize, port: u16, launch: &str, turns: usize, tokens: usize) -> Value {
    let ram_at_launch = settle_ram();
    let t0 = Instant::now();
    let mut child = std::process::Command::new("powershell")
        .args(["-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", launch])
        .stdout(std::process::Stdio::piped())
        .spawn()
        .expect("powershell");
    let mut min_ram = u64::MAX;
    let mut max_gpu = 0u64;
    let sample = |min_ram: &mut u64, max_gpu: &mut u64| {
        *min_ram = (*min_ram).min(shadow_cuda::pinned::available_ram());
        if let Some(g) = gpu_used_mib() {
            *max_gpu = (*max_gpu).max(g);
        }
    };
    // wait for /health (or the launcher to give up)
    let ready = loop {
        if let Ok(Some(_)) = child.try_wait() {
            break false;
        }
        if let Ok((200, _)) = http(port, "GET", "/health", None, Duration::from_secs(5)) {
            break true;
        }
        if t0.elapsed() > Duration::from_secs(900) {
            break false;
        }
        sample(&mut min_ram, &mut max_gpu);
        std::thread::sleep(Duration::from_secs(2));
    };
    let load_s = t0.elapsed().as_secs_f64();
    let mut turns_out = Vec::new();
    let mut error = None;
    if ready {
        let mut history: Vec<Value> = Vec::new();
        for (t, prompt) in TURNS.iter().take(turns).enumerate() {
            history.push(json!({"role": "user", "content": prompt}));
            let body = json!({"messages": history, "max_tokens": tokens, "chat_template_kwargs": {"enable_thinking": false}});
            sample(&mut min_ram, &mut max_gpu);
            match http(port, "POST", "/v1/chat/completions", Some(&body), Duration::from_secs(1800)) {
                Ok((200, b)) => {
                    let v: Value = serde_json::from_str(&b).unwrap_or(Value::Null);
                    let content = v["choices"][0]["message"]["content"].as_str().unwrap_or("").to_string();
                    let tm = &v["timings"];
                    turns_out.push(json!({
                        "turn": t, "decode_tps": tm["predicted_per_second"], "predicted_n": tm["predicted_n"],
                        "predicted_ms": tm["predicted_ms"], "prompt_n": tm["prompt_n"], "prompt_ms": tm["prompt_ms"],
                        "cache_n": tm["cache_n"], "content": content,
                    }));
                    history.push(json!({"role": "assistant", "content": content}));
                }
                Ok((s, b)) => {
                    error = Some(format!("turn {t}: HTTP {s}: {}", b.chars().take(200).collect::<String>()));
                    break;
                }
                Err(e) => {
                    error = Some(format!("turn {t}: {e}"));
                    break;
                }
            }
        }
        sample(&mut min_ram, &mut max_gpu);
        let _ = http(port, "POST", "/shutdown", None, Duration::from_secs(5));
    } else {
        error = Some("the server never became ready".into());
        let _ = child.kill();
    }
    let mut capped_out = String::new();
    if let Some(mut so) = child.stdout.take() {
        let _ = so.read_to_string(&mut capped_out);
    }
    let _ = child.wait();
    let warm: Vec<f64> = turns_out.iter().skip(1).filter_map(|t| t["decode_tps"].as_f64()).collect();
    let line = format!("load {load_s:.0} s, decode_warm median {:.2} tok/s over {} turns, min available RAM {:.1} GiB, \
                        peak GPU {max_gpu} MiB{}", median(&warm), warm.len(), gib(min_ram),
                       error.as_ref().map(|e| format!(" - ERROR {e}")).unwrap_or_default());
    json!({
        "arm": arm, "run": run, "launch": launch, "load_s": load_s, "turns": turns_out, "error": error,
        "available_ram_at_launch_gib": ram_at_launch,
        "min_available_ram_gib": gib(min_ram), "peak_gpu_mib": max_gpu,
        "run_capped": capped_out.trim().lines().last().unwrap_or(""), "line": line, "ended": now(),
    })
}

// ------------------------------------------------------------------------------------------------------- summary
fn sorted(mut v: Vec<f64>) -> Vec<f64> {
    v.retain(|x| x.is_finite());
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v
}

fn median(v: &[f64]) -> f64 {
    let s = sorted(v.to_vec());
    if s.is_empty() {
        return f64::NAN;
    }
    let n = s.len();
    if n % 2 == 1 { s[n / 2] } else { (s[n / 2 - 1] + s[n / 2]) / 2.0 }
}

/// The value at quantile q (nearest rank).
fn quantile(v: &[f64], q: f64) -> f64 {
    let s = sorted(v.to_vec());
    if s.is_empty() {
        return f64::NAN;
    }
    s[((q * s.len() as f64).ceil() as usize).clamp(1, s.len()) - 1]
}

fn summary(dir: &Path) {
    let text = std::fs::read_to_string(dir.join("results.jsonl")).unwrap_or_default();
    let recs: Vec<Value> = text.lines().filter_map(|l| serde_json::from_str(l).ok()).collect();
    let mut arms: Vec<String> = Vec::new();
    for r in &recs {
        let a = r["arm"].as_str().unwrap_or("").to_string();
        if !arms.contains(&a) {
            arms.push(a);
        }
    }
    // decode tok/s by (arm, run, turn)
    let tps = |arm: &str, run: u64, turn: u64| -> Option<f64> {
        recs.iter().find(|r| r["arm"] == arm && r["run"].as_u64() == Some(run))?["turns"].as_array()?
            .iter().find(|t| t["turn"].as_u64() == Some(turn))?["decode_tps"].as_f64()
    };
    let runs_of = |arm: &str| -> Vec<u64> {
        recs.iter().filter(|r| r["arm"] == arm && r["error"].is_null()).filter_map(|r| r["run"].as_u64()).collect()
    };
    let turns_n = recs.iter().filter_map(|r| r["turns"].as_array().map(Vec::len)).max().unwrap_or(0) as u64;
    let mut out = String::new();
    out += &format!("# shadow-bench: {}\n\n{} records, {} arms, decode_warm (turn 0 dropped), tok/s\n\n", dir.display(),
                    recs.len(), arms.len());
    let failed: Vec<String> = recs.iter().filter(|r| !r["error"].is_null())
        .map(|r| format!("{} run {}: {}", r["arm"].as_str().unwrap_or(""), r["run"], r["error"])).collect();
    if !failed.is_empty() {
        out += &format!("**{} runs failed** and are left out:\n\n", failed.len());
        for f in &failed {
            out += &format!("- {f}\n");
        }
        out += "\n";
    }
    out += "| arm | runs | median | p90 (slow end) | worst | best | floor median | floor p90 | floor worst | lowest RAM GiB |\n";
    out += "|---|---|---|---|---|---|---|---|---|---|\n";
    for arm in &arms {
        let runs = runs_of(arm);
        let per_run: Vec<f64> = runs.iter().map(|&r| median(&(1..turns_n).filter_map(|t| tps(arm, r, t)).collect::<Vec<_>>()))
            .collect();
        let mut pairs = Vec::new();
        for t in 1..turns_n {
            for (x, &i) in runs.iter().enumerate() {
                for &j in &runs[x + 1..] {
                    if let (Some(a), Some(b)) = (tps(arm, i, t), tps(arm, j, t)) {
                        pairs.push((a - b).abs() / a.min(b));
                    }
                }
            }
        }
        let low_ram = recs.iter().filter(|r| r["arm"] == arm.as_str()).filter_map(|r| r["min_available_ram_gib"].as_f64())
            .fold(f64::INFINITY, f64::min);
        out += &format!("| {arm} | {} | {:.2} | {:.2} | {:.2} | {:.2} | {:.1}% | {:.1}% | {:.1}% | {:.1} |\n", runs.len(),
                        median(&per_run), quantile(&per_run, 0.1), sorted(per_run.clone()).first().copied().unwrap_or(f64::NAN),
                        sorted(per_run.clone()).last().copied().unwrap_or(f64::NAN), 100.0 * median(&pairs),
                        100.0 * quantile(&pairs, 0.9), 100.0 * sorted(pairs.clone()).last().copied().unwrap_or(f64::NAN),
                        low_ram);
    }
    if arms.len() > 1 {
        out += "\nPaired against the first arm, same run and turn (positive = faster than it):\n\n";
        out += "| arm | pairs | median | p10 | p90 | worst |\n|---|---|---|---|---|---|\n";
        let base = &arms[0];
        for arm in &arms[1..] {
            let mut d = Vec::new();
            for r in runs_of(base) {
                for t in 1..turns_n {
                    if let (Some(a), Some(b)) = (tps(base, r, t), tps(arm, r, t)) {
                        d.push(b / a - 1.0);
                    }
                }
            }
            out += &format!("| {arm} | {} | {:+.1}% | {:+.1}% | {:+.1}% | {:+.1}% |\n", d.len(), 100.0 * median(&d),
                            100.0 * quantile(&d, 0.1), 100.0 * quantile(&d, 0.9),
                            100.0 * sorted(d.clone()).first().copied().unwrap_or(f64::NAN));
        }
        out += "\nA paired difference inside the first arm's floor is a tie.\n";
    }
    // same answers? greedy decoding should give the same text for the same turn in every run of every arm
    let mut distinct = 0;
    for t in 0..turns_n {
        let mut seen: Vec<&str> = Vec::new();
        for r in &recs {
            if let Some(c) = r["turns"].as_array().and_then(|ts| ts.iter().find(|x| x["turn"].as_u64() == Some(t)))
                .and_then(|x| x["content"].as_str()) {
                if !seen.contains(&c) {
                    seen.push(c);
                }
            }
        }
        distinct += seen.len().saturating_sub(1);
    }
    out += &format!("\nAnswers that differ from the first one seen at the same turn: {distinct} (greedy: expected 0).\n");
    std::fs::write(dir.join("summary.md"), &out).unwrap();
    print!("{out}");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn median_and_quantiles() {
        assert_eq!(median(&[3.0, 1.0, 2.0]), 2.0);
        assert_eq!(median(&[4.0, 1.0, 2.0, 3.0]), 2.5);
        assert_eq!(quantile(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0], 0.9), 9.0);
        assert_eq!(quantile(&[5.0], 0.1), 5.0);
        assert!(median(&[]).is_nan());
    }
}
