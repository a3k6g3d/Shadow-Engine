//! The command line of the yardstick:
//!
//!     shadow-eval rank                 # retrieval only, no model session, runs anywhere
//!     shadow-eval prompts              # write eval/prompts/<id>.txt for the answer arm
//!     shadow-eval answers [dir]        # grade the answers a fresh session wrote
//!     shadow-eval full                 # rank + prompts, in one go
//!     shadow-eval verify               # rank, and fail if it moved off eval/baseline.jsonl
//!     shadow-eval run [ids...]         # the answer arm: one fresh session per case (needs the harness)

use std::path::Path;
use shadow_eval::json::Value;
use shadow_eval::score;
use shadow_eval::{BATCH, SEP};

const TOP: usize = score::TOP;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let cmd = args.first().map(|s| s.as_str()).unwrap_or("");
    if cmd == "--help" || cmd == "-h" || cmd.is_empty() {
        print!("{}", include_str!("usage.txt"));
        std::process::exit(2);
    }
    if let Err(e) = dispatch(cmd, &args[1..]) {
        eprintln!("{e}");
        std::process::exit(1);
    }
}

fn dispatch(cmd: &str, rest: &[String]) -> Result<(), String> {
    let cases = cases()?;
    let store = store()?;
    match cmd {
        "rank" => arm_rank(&cases, &store, false),
        "verify" => arm_rank(&cases, &store, true),
        "prompts" => arm_prompts(&cases, &store),
        "full" => {
            arm_rank(&cases, &store, false)?;
            println!();
            arm_prompts(&cases, &store)
        }
        "answers" => arm_answers(&cases, rest.first().map(|s| s.as_str()).unwrap_or("answers")),
        "run" => arm_run(&cases, rest),
        other => Err(format!("unknown command: {other}")),
    }
}

fn cases() -> Result<Vec<score::Case>, String> {
    let path = shadow_eval::repo_file("eval/cases.jsonl").ok_or("no eval/cases.jsonl at the repo root")?;
    let rows = shadow_eval::read_jsonl(&path)?;
    rows.iter().map(score::Case::from).collect()
}

fn store() -> Result<Vec<score::Record>, String> {
    let path = shadow_eval::repo_file("memory/memories.jsonl").ok_or("no memory/memories.jsonl at the repo root")?;
    shadow_eval::read_jsonl(&path)?.iter().map(score::Record::from).collect()
}

/// The retrieval arm.  One embedder call for the whole set, then rank — no model session, so it is deterministic.
fn arm_rank(cases: &[score::Case], store: &[score::Record], against_baseline: bool) -> Result<(), String> {
    if store.is_empty() {
        return Err("the store is empty - nothing to rank".to_string());
    }
    let texts: Vec<String> = cases.iter().map(|c| c.ask.clone()).collect();
    let qvecs = shadow_eval::embed(&texts)?;
    println!("{:<6}{:>5}{:>8}  question", "case", "rank", "score");
    let mut rr = 0.0;
    let mut top1 = 0;
    let mut got: Vec<(String, u32)> = Vec::new();
    for (case, qv) in cases.iter().zip(qvecs.iter()) {
        let (rank, s) = score::rank_of(&case.gold_sub, qv, store);
        if rank != u32::MAX {
            rr += 1.0 / rank as f64;
        }
        if rank == 1 {
            top1 += 1;
        }
        got.push((case.id.clone(), rank));
        let flag = if rank == 1 {
            String::new()
        } else if rank <= TOP as u32 {
            "  <-- not first".to_string()
        } else {
            format!("  <-- outside top-{TOP}")
        };
        println!("{:<6}{:>5}{:>8.4}  {}{flag}", case.id, rank, s, truncate(&case.ask, 58));
    }
    let n = cases.len() as f64;
    println!("\n{} cases, top-1 {}/{}", cases.len(), top1, cases.len());
    println!("MRR {:.3}   (store: {} records)", rr / n, store.len());
    println!("this arm measures retrieval only; it says nothing about whether the answer was written correctly");

    if against_baseline {
        let path = shadow_eval::repo_file("eval/baseline.jsonl").ok_or("no eval/baseline.jsonl to verify against")?;
        let rows = shadow_eval::read_jsonl(&path)?;
        let mut moved = Vec::new();
        for row in &rows {
            let id = row.get("id").and_then(Value::as_str).unwrap_or("");
            let want = row.get("rank").and_then(Value::as_f64).unwrap_or(0.0) as u32;
            if let Some((_, got)) = got.iter().find(|(g, _)| g == id) {
                if *got != want {
                    moved.push(format!("{id}: baseline {want}, now {got}"));
                }
            }
        }
        if moved.is_empty() {
            println!("\nverify: unchanged against eval/baseline.jsonl — the yardstick has not moved.");
        } else {
            println!("\nverify: the retrieval arm moved.  A changed rank is only interesting if a case got WORSE:");
            for m in &moved {
                println!("  {m}");
            }
            println!("if the change is a fix to a case (not the store), re-freeze the baseline and say so in the commit;");
            println!("if anything got worse, this is the loop grading itself — do not ship it.");
        }
    }
    Ok(())
}

fn arm_prompts(cases: &[score::Case], store: &[score::Record]) -> Result<(), String> {
    if store.is_empty() {
        return Err("the store is empty - nothing to retrieve".to_string());
    }
    let dir = shadow_eval::repo_root().join("eval").join("prompts");
    std::fs::create_dir_all(&dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    let texts: Vec<String> = cases.iter().map(|c| c.ask.clone()).collect();
    let qvecs = shadow_eval::embed(&texts)?;
    for (case, qv) in cases.iter().zip(qvecs.iter()) {
        let path = dir.join(format!("{}.txt", case.id));
        std::fs::write(&path, score::prompt(case, &score::top_hits(qv, store))).map_err(|e| format!("{}: {e}", path.display()))?;
    }
    println!("wrote {} prompts to {}", cases.len(), dir.display());
    println!("each must be answered by a FRESH session - the one writing the prompt cannot grade its own answer");
    Ok(())
}

fn arm_answers(cases: &[score::Case], dir: &str) -> Result<(), String> {
    let d = Path::new(dir);
    let d = if d.is_absolute() { d.to_path_buf() } else { shadow_eval::repo_root().join("eval").join(dir) };
    if !d.is_dir() {
        return Err(format!("no such answers folder: {}", d.display()));
    }
    let (mut ok, mut missing) = (0, 0);
    for case in cases {
        let p = d.join(format!("{}.txt", case.id));
        if !p.is_file() {
            println!("{:<6}  --  no answer file", case.id);
            missing += 1;
            continue;
        }
        let got = std::fs::read_to_string(&p).map_err(|e| format!("{}: {e}", p.display()))?;
        let got = got.trim_start_matches('\u{feff}').to_lowercase();
        let need: Vec<String> = case.must_contain.iter().filter(|w| !got.contains(&w.to_lowercase())).cloned().collect();
        if need.is_empty() {
            println!("{:<6}  yes  {}", case.id, truncate(&case.ask, 50));
            ok += 1;
        } else {
            println!("{:<6}  no   missing {:?}  <- {}", case.id, need, p.file_name().unwrap_or_default().to_string_lossy());
        }
    }
    let graded = cases.len() - missing;
    println!("\n{graded} graded, {ok}/{graded} correct, {missing} missing");
    Ok(())
}

/// The answer arm: one fresh harness session per case, so the answer comes only from the notes in front of it.
/// The session that wrote the yardstick must not be the session answering off it.
fn arm_run(cases: &[score::Case], wanted: &[String]) -> Result<(), String> {
    let cases: Vec<score::Case> =
        if wanted.is_empty() { cases.to_vec() } else { cases.iter().filter(|c| wanted.contains(&c.id)).cloned().collect() };
    if cases.is_empty() {
        return Err("no matching cases".to_string());
    }
    let root = shadow_eval::repo_root();
    let harness = root.join("tools").join("shadow_harness.py");
    if !harness.is_file() {
        return Err(format!("missing {}", harness.display()));
    }
    let prompts = root.join("eval").join("prompts");
    let answers = root.join("eval").join("answers");
    std::fs::create_dir_all(&answers).map_err(|e| format!("{}: {e}", answers.display()))?;
    println!("{} cases, one fresh harness session each", cases.len());
    for case in cases {
        let pf = prompts.join(format!("{}.txt", case.id));
        if !pf.is_file() {
            return Err(format!("no prompt for {} - run `shadow-eval prompts` first", case.id));
        }
        let prompt = std::fs::read_to_string(&pf).map_err(|e| format!("{}: {e}", pf.display()))?;
        let t0 = std::time::Instant::now();
        let out = std::process::Command::new("python")
            .arg(&harness)
            .arg(prompt.trim())
            .args(["--memory", &answers.to_string_lossy(), "--rounds", "1", "--max-tokens", "220", "--quiet"])
            .current_dir(&root)
            .output()
            .map_err(|e| format!("{}: {e}", harness.display()))?;
        if !out.status.success() {
            println!("{:<6}  harness rc {}: {}", case.id, out.status.code().unwrap_or(-1), tail(&out.stderr, 120));
            continue;
        }
        let log = answers.join("sessions.jsonl");
        let answer = last_answer(&log).ok_or("the harness logged no answer")?;
        if answer.trim().is_empty() {
            println!("{:<6}  empty answer ({} s)", case.id, t0.elapsed().as_secs());
            continue;
        }
        let p = answers.join(format!("{}.txt", case.id));
        std::fs::write(&p, format!("{}\n", answer.trim())).map_err(|e| format!("{}: {e}", p.display()))?;
        println!("{:<6}  {:>4} s  {}", case.id, t0.elapsed().as_secs(), truncate(answer.trim(), 70));
    }
    println!("\nanswers in {}\ngrade with:  shadow-eval answers answers", answers.display());
    Ok(())
}

/// The last line the harness appended — its own log is the record of what it answered.
fn last_answer(path: &Path) -> Option<String> {
    let rows = shadow_eval::read_jsonl(path).ok()?;
    rows.last()?.get("answer").and_then(Value::as_str).map(|s| s.to_string())
}

fn truncate(s: &str, n: usize) -> String {
    let mut out = s.chars().take(n).collect::<String>();
    if s.chars().count() > n {
        out.push_str("...");
    }
    out
}

fn tail(bytes: &[u8], n: usize) -> String {
    let s = String::from_utf8_lossy(bytes);
    let t = s.trim();
    t.chars().rev().take(n).collect::<Vec<_>>().into_iter().rev().collect()
}

// kept so the constants the Python protocol pinned stay referenced here too
#[allow(dead_code)]
const _CHECK: (usize, &str) = (BATCH, SEP);
