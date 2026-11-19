//! The picture encoder: Strata's `strata-vision` (llama.cpp's mtmd library on the model's mmproj file), kept resident as a child
//! process.  It turns a picture into the rows the text model reads in place of its `<|image_pad|>` tokens.
//!
//! Protocol: after `READY <n_embd>`, `ENC <image> <out>` answers `OK <n_tokens> <nx> <ny> <ms>` or `ERR <why>`; the output file is
//! five little-endian i32s (`SVE1`, n_tokens, nx, ny, n_embd), then n_tokens x n_embd f32, row i at grid position
//! (x = i % nx, y = i / nx).  CPU encoding: a 260-token picture takes about 2.4 s on 8 threads.

use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};

/// Where the helper and the projector are looked for (after `SHADOWZERO_VISION_EXE` / `SHADOWZERO_MMPROJ`): ShadowZero's own folders, then Strata's build.
const EXE_CANDIDATES: &[&str] = &[
    "bin/strata-vision.exe",
    r"J:\Strata-work\build-0145\strata-vision.exe",
];
const MMPROJ_CANDIDATES: &[&str] = &[
    "models/Qwen3.8-Flash-Next.mmproj-f16.gguf",
    r"J:\Strata-work\models\sc117\Qwen3.8-Flash-Next.mmproj-f16.gguf",
];

/// `var` if set, else the first candidate that exists (relative ones in the working directory, then beside the ShadowZero checkout).
fn locate(var: &str, candidates: &[&str]) -> String {
    let abs = |p: PathBuf| std::path::absolute(&p).unwrap_or(p).display().to_string();
    if let Ok(p) = std::env::var(var) {
        return abs(PathBuf::from(p));
    }
    let root = Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/../.."));
    candidates
        .iter()
        .flat_map(|c| [PathBuf::from(c), root.join(c)])
        .find(|p| p.is_file())
        .map_or_else(|| candidates[0].to_string(), abs)
}

/// One encoded picture: `n` rows of `n_embd` floats on an `nx` x `ny` grid.
pub struct Picture {
    pub n: usize,
    pub nx: usize,
    pub ny: usize,
    pub rows: Vec<f32>,
}

pub struct Encoder {
    child: Child,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
    pub n_embd: usize,
    dir: PathBuf,
    seq: u64,
}

fn is_non_text_gguf(name_lower: &str) -> bool {
    name_lower.contains("mmproj")
        || name_lower.contains("bge-")
        || name_lower.contains("embed")
        || name_lower.contains("draft")
        || name_lower.contains("mtp")
}

fn is_secondary_split(name_lower: &str) -> bool {
    // e.g. model-00002-of-00004.gguf: only open shard 00001
    let Some(pos) = name_lower.find("-of-") else {
        return false;
    };
    let prefix = &name_lower[..pos];
    let digits: String = prefix.chars().rev().take_while(|c| c.is_ascii_digit()).collect();
    !digits.is_empty() && !digits.chars().rev().collect::<String>().ends_with("0001")
}

fn collect_ggufs(dir: &Path, depth: usize, out: &mut Vec<PathBuf>) {
    let Ok(rd) = std::fs::read_dir(dir) else {
        return;
    };
    let mut subdirs = Vec::new();
    for entry in rd.flatten() {
        let p = entry.path();
        let name = entry.file_name();
        let name_s = name.to_string_lossy();
        if name_s.starts_with('.') {
            continue;
        }
        if p.is_dir() {
            if depth > 0 {
                subdirs.push(p);
            }
        } else if p.is_file() && p.extension().is_some_and(|e| e.eq_ignore_ascii_case("gguf")) {
            let low = name_s.to_ascii_lowercase();
            if !is_non_text_gguf(&low) && !is_secondary_split(&low) {
                out.push(p);
            }
        }
    }
    for sub in subdirs {
        collect_ggufs(&sub, depth - 1, out);
    }
}

fn rank_text_gguf(p: &Path, model_name_lower: &str) -> (i32, String) {
    let low = p.file_name().unwrap_or_default().to_string_lossy().to_ascii_lowercase();
    let mut score = 0;
    if low.contains("qwen3.8") || low.contains("qwen35") || low.contains("qwen3") {
        score += 40;
    }
    if low.contains("swift") || low.contains("iq3_xxs") {
        score += 20;
    }
    if low.contains("00001-of-") {
        score += 10;
    }
    if model_name_lower.contains("qwen3.8") && low.contains("qwen3.8") {
        score += 30;
    }
    (-score, low)
}

/// Resolves the GGUF file that `strata-vision` opens vocab-only for model architecture and RoPE metadata.
/// Single-file or split `.gguf` models are used directly; safetensors/EXL3 directories resolve via `SHADOWZERO_VISION_MODEL`
/// or a companion Qwen3.8 `.gguf` inside the directory, its sibling folders, or the repo `models/` folders.
pub fn resolve_vision_model(model: &Path) -> Result<PathBuf, String> {
    if let Ok(v) = std::env::var("SHADOWZERO_VISION_MODEL") {
        let p = PathBuf::from(&v);
        if p.is_file() {
            return Ok(std::path::absolute(&p).unwrap_or(p));
        }
        return Err(format!("SHADOWZERO_VISION_MODEL does not point to a file: {v}"));
    }
    if model.is_file() && model.extension().is_some_and(|e| e.eq_ignore_ascii_case("gguf")) {
        return Ok(model.to_path_buf());
    }
    let model_name_low = model
        .file_name()
        .unwrap_or_default()
        .to_string_lossy()
        .to_ascii_lowercase();
    let root = Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/../.."));
    let mut roots: Vec<(PathBuf, usize)> = Vec::new();
    if model.is_dir() {
        roots.push((model.to_path_buf(), 2));
    }
    if let Some(parent) = model.parent() {
        roots.push((parent.to_path_buf(), 3));
    }
    roots.push((PathBuf::from("models"), 2));
    roots.push((root.join("models"), 2));
    roots.push((PathBuf::from(r"J:\Models"), 3));
    roots.push((PathBuf::from(r"J:\Strata-work\models"), 3));

    for (dir, depth) in roots {
        if !dir.is_dir() {
            continue;
        }
        let mut found = Vec::new();
        collect_ggufs(&dir, depth, &mut found);
        if !found.is_empty() {
            found.sort_by_key(|p| rank_text_gguf(p, &model_name_low));
            let chosen = found.into_iter().next().unwrap();
            return Ok(std::path::absolute(&chosen).unwrap_or(chosen));
        }
    }
    Err(format!(
        "cannot encode pictures for safetensors/EXL3 model {} without a companion Qwen3.8 .gguf for vision metadata (set SHADOWZERO_VISION_MODEL=path\\to\\model.gguf)",
        model.display()
    ))
}

impl Encoder {
    /// Starts the helper (`SHADOWZERO_VISION_EXE`, else Strata's build) on the projector (`SHADOWZERO_MMPROJ`) and the text model's GGUF metadata.
    pub fn start(model: &Path) -> Result<Encoder, String> {
        let exe = locate("SHADOWZERO_VISION_EXE", EXE_CANDIDATES);
        let mmproj = locate("SHADOWZERO_MMPROJ", MMPROJ_CANDIDATES);
        for (what, p) in [("picture encoder", &exe), ("vision projector", &mmproj)] {
            if !Path::new(p).is_file() {
                return Err(format!(
                    "no {what} at {p} (set SHADOWZERO_VISION_EXE / SHADOWZERO_MMPROJ)"
                ));
            }
        }
        let vision_model = resolve_vision_model(model)?;
        let threads =
            std::thread::available_parallelism().map_or(8, |n| (n.get() / 2).clamp(2, 16));
        let mut cmd = Command::new(&exe);
        cmd.arg("--mmproj")
            .arg(&mmproj)
            .arg("--model")
            .arg(&vision_model)
            .arg("--threads")
            .arg(threads.to_string())
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null());
        if let Some(d) = Path::new(&exe).parent() {
            cmd.current_dir(d);
        }
        let mut child = cmd.spawn().map_err(|e| format!("starting {exe}: {e}"))?;
        let stdin = child.stdin.take().ok_or("no stdin")?;
        let mut stdout = BufReader::new(child.stdout.take().ok_or("no stdout")?);
        let mut line = String::new();
        stdout.read_line(&mut line).map_err(|e| e.to_string())?;
        let n_embd = line
            .trim()
            .strip_prefix("READY ")
            .and_then(|v| v.parse().ok())
            .ok_or_else(|| format!("the picture encoder did not start: {:?}", line.trim()))?;
        let dir = std::env::temp_dir().join(format!("shadow-pictures-{}", std::process::id()));
        std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
        Ok(Encoder {
            child,
            stdin,
            stdout,
            n_embd,
            dir,
            seq: 0,
        })
    }

    /// The rows for a picture file's bytes (PNG, JPEG...).
    pub fn encode(&mut self, bytes: &[u8]) -> Result<Picture, String> {
        self.seq += 1;
        let (img, out) = (
            self.dir.join(format!("{}.img", self.seq)),
            self.dir.join(format!("{}.sve", self.seq)),
        );
        std::fs::write(&img, bytes).map_err(|e| e.to_string())?;
        let r = self.ask(&img, &out);
        let _ = std::fs::remove_file(&img);
        let _ = std::fs::remove_file(&out);
        r
    }

    fn ask(&mut self, img: &Path, out: &Path) -> Result<Picture, String> {
        // the helper splits the line at the last space, so the paths must have none (the temp dir's does not)
        writeln!(self.stdin, "ENC {} {}", img.display(), out.display())
            .and_then(|_| self.stdin.flush())
            .map_err(|e| e.to_string())?;
        let mut line = String::new();
        self.stdout
            .read_line(&mut line)
            .map_err(|e| e.to_string())?;
        let line = line.trim();
        if !line.starts_with("OK ") {
            return Err(format!("the picture encoder said: {line}"));
        }
        let d = std::fs::read(out).map_err(|e| e.to_string())?;
        let i32_at = |k: usize| i32::from_le_bytes(d[4 * k..4 * k + 4].try_into().unwrap());
        if d.len() < 20 || i32_at(0) != 0x3145_5653 {
            return Err("the picture encoder wrote a file that is not SVE1".into());
        }
        let (n, nx, ny, ne) = (
            i32_at(1) as usize,
            i32_at(2) as usize,
            i32_at(3) as usize,
            i32_at(4) as usize,
        );
        if ne != self.n_embd || d.len() != 20 + n * ne * 4 {
            return Err("the picture encoder's file has the wrong size".into());
        }
        let rows = d[20..]
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        Ok(Picture { n, nx, ny, rows })
    }
}

impl Drop for Encoder {
    fn drop(&mut self) {
        let _ = writeln!(self.stdin, "QUIT");
        let _ = self.child.wait();
        let _ = std::fs::remove_dir_all(&self.dir);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolve_vision_model_accepts_direct_gguf_and_finds_sibling_for_exl3_dir() {
        let base = std::env::temp_dir().join(format!("shadowzero-vision-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&base);
        let exl3_dir = base.join("Lygodactylus-Qwen3.8-Flash-Next-Uncensored-exl3-mixD18");
        let gguf_dir = base.join("Huihui-Qwen3.8-Flash-Next-abliterated-v2-GGUF").join("swift-IQ3_XXS");
        std::fs::create_dir_all(&exl3_dir).unwrap();
        std::fs::create_dir_all(&gguf_dir).unwrap();
        std::fs::write(exl3_dir.join("config.json"), "{}").unwrap();
        let mmproj = base.join("Huihui-Qwen3.8-Flash-Next-abliterated-v2-GGUF").join("mmproj-Swift-Qwen3.8.gguf");
        let shard2 = gguf_dir.join("Swift-Qwen3.8-00002-of-00002.gguf");
        let shard1 = gguf_dir.join("Swift-Qwen3.8-00001-of-00002.gguf");
        std::fs::write(&mmproj, b"gguf").unwrap();
        std::fs::write(&shard2, b"gguf").unwrap();
        std::fs::write(&shard1, b"gguf").unwrap();

        assert_eq!(resolve_vision_model(&shard1).unwrap(), shard1);
        let resolved = resolve_vision_model(&exl3_dir).unwrap();
        assert_eq!(resolved.file_name().unwrap(), "Swift-Qwen3.8-00001-of-00002.gguf");

        let _ = std::fs::remove_dir_all(&base);
    }
}
