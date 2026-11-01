//! The picture encoder: Strata's `strata-vision` (llama.cpp's mtmd library on the model's mmproj file), kept resident as a child
//! process.  It turns a picture into the rows the text model reads in place of its `<|image_pad|>` tokens.
//!
//! Protocol: after `READY <n_embd>`, `ENC <image> <out>` answers `OK <n_tokens> <nx> <ny> <ms>` or `ERR <why>`; the output file is
//! five little-endian i32s (`SVE1`, n_tokens, nx, ny, n_embd), then n_tokens x n_embd f32, row i at grid position
//! (x = i % nx, y = i / nx).  CPU encoding: a 260-token picture takes about 2.4 s on 8 threads.

use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};

/// Where the helper and the projector are looked for (after `SHADOW_VISION_EXE` / `SHADOW_MMPROJ`): Shadow's own folders, then Strata's build.
const EXE_CANDIDATES: &[&str] = &["bin/strata-vision.exe", r"J:\Strata-work\build-0145\strata-vision.exe"];
const MMPROJ_CANDIDATES: &[&str] = &["models/Qwen3.8-Flash-Next.mmproj-f16.gguf", r"J:\Strata-work\models\sc117\Qwen3.8-Flash-Next.mmproj-f16.gguf"];

/// `var` if set, else the first candidate that exists (relative ones in the working directory, then beside the Shadow checkout).
fn locate(var: &str, candidates: &[&str]) -> String {
    let abs = |p: PathBuf| std::path::absolute(&p).unwrap_or(p).display().to_string();
    if let Ok(p) = std::env::var(var) {
        return abs(PathBuf::from(p));
    }
    let root = Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/../.."));
    candidates.iter().flat_map(|c| [PathBuf::from(c), root.join(c)]).find(|p| p.is_file())
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

impl Encoder {
    /// Starts the helper (`SHADOW_VISION_EXE`, else Strata's build) on the projector (`SHADOW_MMPROJ`) and the text model's first GGUF.
    pub fn start(model: &Path) -> Result<Encoder, String> {
        let exe = locate("SHADOW_VISION_EXE", EXE_CANDIDATES);
        let mmproj = locate("SHADOW_MMPROJ", MMPROJ_CANDIDATES);
        for (what, p) in [("picture encoder", &exe), ("vision projector", &mmproj)] {
            if !Path::new(p).is_file() {
                return Err(format!("no {what} at {p} (set SHADOW_VISION_EXE / SHADOW_MMPROJ)"));
            }
        }
        let threads = std::thread::available_parallelism().map_or(8, |n| (n.get() / 2).clamp(2, 16));
        let mut cmd = Command::new(&exe);
        cmd.arg("--mmproj").arg(&mmproj).arg("--model").arg(model).arg("--threads").arg(threads.to_string())
            .stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::null());
        if let Some(d) = Path::new(&exe).parent() {
            cmd.current_dir(d);
        }
        let mut child = cmd.spawn().map_err(|e| format!("starting {exe}: {e}"))?;
        let stdin = child.stdin.take().ok_or("no stdin")?;
        let mut stdout = BufReader::new(child.stdout.take().ok_or("no stdout")?);
        let mut line = String::new();
        stdout.read_line(&mut line).map_err(|e| e.to_string())?;
        let n_embd = line.trim().strip_prefix("READY ").and_then(|v| v.parse().ok())
            .ok_or_else(|| format!("the picture encoder did not start: {:?}", line.trim()))?;
        let dir = std::env::temp_dir().join(format!("shadow-pictures-{}", std::process::id()));
        std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
        Ok(Encoder { child, stdin, stdout, n_embd, dir, seq: 0 })
    }

    /// The rows for a picture file's bytes (PNG, JPEG...).
    pub fn encode(&mut self, bytes: &[u8]) -> Result<Picture, String> {
        self.seq += 1;
        let (img, out) = (self.dir.join(format!("{}.img", self.seq)), self.dir.join(format!("{}.sve", self.seq)));
        std::fs::write(&img, bytes).map_err(|e| e.to_string())?;
        let r = self.ask(&img, &out);
        let _ = std::fs::remove_file(&img);
        let _ = std::fs::remove_file(&out);
        r
    }

    fn ask(&mut self, img: &Path, out: &Path) -> Result<Picture, String> {
        // the helper splits the line at the last space, so the paths must have none (the temp dir's does not)
        writeln!(self.stdin, "ENC {} {}", img.display(), out.display()).and_then(|_| self.stdin.flush()).map_err(|e| e.to_string())?;
        let mut line = String::new();
        self.stdout.read_line(&mut line).map_err(|e| e.to_string())?;
        let line = line.trim();
        if !line.starts_with("OK ") {
            return Err(format!("the picture encoder said: {line}"));
        }
        let d = std::fs::read(out).map_err(|e| e.to_string())?;
        let i32_at = |k: usize| i32::from_le_bytes(d[4 * k..4 * k + 4].try_into().unwrap());
        if d.len() < 20 || i32_at(0) != 0x3145_5653 {
            return Err("the picture encoder wrote a file that is not SVE1".into());
        }
        let (n, nx, ny, ne) = (i32_at(1) as usize, i32_at(2) as usize, i32_at(3) as usize, i32_at(4) as usize);
        if ne != self.n_embd || d.len() != 20 + n * ne * 4 {
            return Err("the picture encoder's file has the wrong size".into());
        }
        let rows = d[20..].chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
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
