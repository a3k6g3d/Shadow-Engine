//! A RAM black box (docs/PLAN_RAM_TIER_AND_LEVERS_2026-11-09.md, A0): while the server runs, available RAM is sampled every second, and when it falls under a threshold (the
//! RAM floor, 4 GiB by default) the moment is written to `rambox.log` with the trend of the last 20 s, how much Shadow itself has pinned, and the biggest programs by memory
//! (from `tasklist`).  Why: the RAM rule that sizes the tier comes from one freeze at ~0.4 GiB free whose cause nobody recorded; the next low-RAM event now has a record of who
//! held the memory (an outside program, Windows, or Shadow), so the floor and the tier can be chosen from evidence.  Every 10 minutes a one-line summary (min / mean / max
//! available) shows how RAM moves on a quiet day.  The log is `%LOCALAPPDATA%\Shadow\rambox.log`, rotated at 1 MiB; `SHADOW_RAM_BOX=0` turns it off,
//! `SHADOW_RAM_BOX_GIB` changes the threshold.  Nothing here allocates a page of pinned memory or touches the GPU.
use std::collections::VecDeque;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

const GIB: f64 = (1u64 << 30) as f64;
const RING_SECS: u64 = 60;
const COOLDOWN_SECS: u64 = 10;
const SUMMARY_SECS: u64 = 600;
const ROTATE_BYTES: u64 = 1 << 20;

/// What the sampler wants written.
#[derive(Debug, PartialEq)]
pub enum Event {
    /// Available RAM is under the threshold (the text is the log line); the caller adds the biggest programs.
    Low(String),
    /// Available RAM came back above the threshold.
    Recovered(String),
    /// The 10-minute summary line.
    Summary(String),
}

/// The pure part: feed it one sample a second and it says what to log.
pub struct RamBox {
    threshold: u64,
    ring: VecDeque<(u64, u64)>,
    low_since: Option<u64>,
    low_min: u64,
    last_low: Option<u64>,
    win_start: u64,
    win: (u64, u64, u64, u64), // min, max, sum, count
}

impl RamBox {
    pub fn new(threshold: u64) -> RamBox {
        RamBox { threshold, ring: VecDeque::new(), low_since: None, low_min: u64::MAX, last_low: None, win_start: 0, win: (u64::MAX, 0, 0, 0) }
    }

    /// One sample: `t` seconds since the epoch, `avail` bytes available, `pinned` bytes Shadow has pinned.
    pub fn observe(&mut self, t: u64, avail: u64, pinned: u64) -> Vec<Event> {
        let mut out = Vec::new();
        if self.win_start == 0 {
            self.win_start = t;
        }
        self.ring.push_back((t, avail));
        while self.ring.front().is_some_and(|(s, _)| t.saturating_sub(*s) > RING_SECS) {
            self.ring.pop_front();
        }
        self.win = (self.win.0.min(avail), self.win.1.max(avail), self.win.2 + avail, self.win.3 + 1);
        if avail < self.threshold {
            self.low_min = self.low_min.min(avail);
            let first = self.low_since.is_none();
            self.low_since.get_or_insert(t);
            if first || self.last_low.is_none_or(|l| t.saturating_sub(l) >= COOLDOWN_SECS) {
                self.last_low = Some(t);
                let trend: Vec<String> = [20u64, 15, 10, 5].iter()
                    .filter_map(|ago| self.ring.iter().rev().find(|(s, _)| t.saturating_sub(*s) >= *ago).map(|(_, a)| format!("{:.2} ({}s ago)", *a as f64 / GIB, ago)))
                    .collect();
                out.push(Event::Low(format!("{} LOW: {:.2} GiB available (threshold {:.1}); Shadow has {:.1} GiB pinned; before: {}; low for {} s so far",
                    utc(t), avail as f64 / GIB, self.threshold as f64 / GIB, pinned as f64 / GIB, if trend.is_empty() { "-".into() } else { trend.join(", ") }, t - self.low_since.unwrap_or(t))));
            }
        } else if let Some(since) = self.low_since {
            // back above the threshold with a little room, so a value wobbling around it is one episode
            if avail >= self.threshold + (1 << 29) {
                out.push(Event::Recovered(format!("{} RECOVERED: {:.2} GiB available after {} s under {:.1} GiB, lowest {:.2} GiB",
                    utc(t), avail as f64 / GIB, t - since, self.threshold as f64 / GIB, self.low_min as f64 / GIB)));
                self.low_since = None;
                self.low_min = u64::MAX;
                self.last_low = None;
            }
        }
        if t.saturating_sub(self.win_start) >= SUMMARY_SECS {
            out.push(Event::Summary(format!("{} SUMMARY {} min: available min {:.2} / mean {:.2} / max {:.2} GiB; Shadow {:.1} GiB pinned",
                utc(t), (t - self.win_start) / 60, self.win.0 as f64 / GIB, self.win.2 as f64 / self.win.3.max(1) as f64 / GIB, self.win.1 as f64 / GIB, pinned as f64 / GIB)));
            self.win_start = t;
            self.win = (u64::MAX, 0, 0, 0);
        }
        out
    }
}

/// `1700000000` -> `2023-11-14 22:13:20 UTC` (no date library: days since 1970 to a calendar date).
pub fn utc(t: u64) -> String {
    let (days, secs) = ((t / 86_400) as i64, t % 86_400);
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = yoe + era * 400 + (m <= 2) as i64;
    format!("{y:04}-{m:02}-{d:02} {:02}:{:02}:{:02} UTC", secs / 3600, secs % 3600 / 60, secs % 60)
}

/// The biggest programs by memory from `tasklist /FO CSV /NH` text: (name, KiB), largest first, at most `n`.
pub fn top_processes(csv: &str, n: usize) -> Vec<(String, u64)> {
    let mut rows: Vec<(String, u64)> = csv.lines().filter_map(|l| {
        let cells: Vec<&str> = l.trim().trim_matches('"').split("\",\"").collect();
        if cells.len() < 5 {
            return None;
        }
        let kib: u64 = cells[4].chars().filter(|c| c.is_ascii_digit()).collect::<String>().parse().ok()?;
        Some((cells[0].to_string(), kib))
    }).collect();
    // one name may run many times (a browser): add them up
    let mut sum: Vec<(String, u64)> = Vec::new();
    for (name, kib) in rows.drain(..) {
        match sum.iter_mut().find(|(n, _)| *n == name) {
            Some(e) => e.1 += kib,
            None => sum.push((name, kib)),
        }
    }
    sum.sort_by(|a, b| b.1.cmp(&a.1));
    sum.truncate(n);
    sum
}

/// The biggest programs right now (Windows `tasklist`); empty when it cannot be run.
fn top_processes_now(n: usize) -> Vec<(String, u64)> {
    let mut cmd = std::process::Command::new("tasklist");
    cmd.args(["/FO", "CSV", "/NH"]).stderr(std::process::Stdio::null());
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        cmd.creation_flags(0x0800_0000); // no console window
    }
    cmd.output().map(|o| top_processes(&String::from_utf8_lossy(&o.stdout), n)).unwrap_or_default()
}

fn append(path: &Path, line: &str) {
    if std::fs::metadata(path).is_ok_and(|m| m.len() > ROTATE_BYTES) {
        let _ = std::fs::rename(path, path.with_file_name("rambox.old.log"));
    }
    if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(path) {
        let _ = writeln!(f, "{line}");
    }
}

fn now() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map_or(0, |d| d.as_secs())
}

/// Starts the sampler thread once (a second call does nothing).  `dir` is where `rambox.log` goes.
pub fn start(dir: PathBuf) {
    static STARTED: OnceLock<()> = OnceLock::new();
    if std::env::var("SHADOW_RAM_BOX").is_ok_and(|v| v == "0") || STARTED.set(()).is_err() {
        return;
    }
    let threshold = std::env::var("SHADOW_RAM_BOX_GIB").ok().and_then(|v| v.parse::<f64>().ok()).map_or(4u64 << 30, |g| (g * GIB) as u64);
    let _ = std::fs::create_dir_all(&dir);
    let log = dir.join("rambox.log");
    let _ = std::thread::Builder::new().name("rambox".into()).spawn(move || {
        let mut b = RamBox::new(threshold);
        append(&log, &format!("{} STARTED: {:.1} GiB available, threshold {:.1} GiB", utc(now()), crate::pinned::available_ram() as f64 / GIB, threshold as f64 / GIB));
        loop {
            std::thread::sleep(Duration::from_secs(1));
            for e in b.observe(now(), crate::pinned::available_ram(), crate::pinned::pinned_bytes()) {
                match e {
                    Event::Low(line) => {
                        append(&log, &line);
                        // who holds the memory: asked on a side thread (a starved machine can be slow to answer), appended when it comes
                        let log2 = log.clone();
                        let _ = std::thread::Builder::new().name("rambox-top".into()).spawn(move || {
                            let top = top_processes_now(8);
                            if !top.is_empty() {
                                append(&log2, &format!("    biggest programs: {}", top.iter().map(|(n, k)| format!("{n} {:.1} GiB", *k as f64 / 1_048_576.0)).collect::<Vec<_>>().join(", ")));
                            }
                        });
                    }
                    Event::Recovered(line) | Event::Summary(line) => append(&log, &line),
                }
            }
        }
    });
}

/// `%LOCALAPPDATA%\Shadow` (the temp folder when that is not set).
pub fn default_dir() -> PathBuf {
    std::env::var_os("LOCALAPPDATA").map_or_else(|| std::env::temp_dir().join("Shadow"), |a| PathBuf::from(a).join("Shadow"))
}

#[cfg(test)]
mod tests {
    use super::*;
    const G: u64 = 1 << 30;

    #[test]
    fn it_is_quiet_above_the_threshold_and_says_so_once_below_with_the_trend() {
        let mut b = RamBox::new(4 * G);
        for t in 1000..1030 {
            assert!(b.observe(t, 8 * G, 36 * G).is_empty(), "nothing above the threshold");
        }
        // the free RAM falls: 8 -> 3 GiB over five seconds
        let mut lows = vec![];
        for (i, t) in (1030..1036).enumerate() {
            lows.extend(b.observe(t, (8 - i as u64) * G, 36 * G));
        }
        assert_eq!(lows.len(), 1, "one line when it first goes under: {lows:?}");
        let Event::Low(l) = &lows[0] else { panic!("{lows:?}") };
        assert!(l.contains("LOW: 3.00 GiB available") && l.contains("Shadow has 36.0 GiB pinned") && l.contains("(5s ago)") && l.contains("low for 0 s"), "{l}");
    }

    #[test]
    fn a_long_low_period_is_logged_every_ten_seconds_and_ends_with_a_recovery_line() {
        let mut b = RamBox::new(4 * G);
        let mut events = vec![];
        for t in 0..35u64 {
            events.extend(b.observe(100 + t, 3 * G + (t % 3) * (G / 10), 0));
        }
        let lows = events.iter().filter(|e| matches!(e, Event::Low(_))).count();
        assert_eq!(lows, 4, "at 0, 10, 20 and 30 s: {events:?}");
        // a value wobbling just above the threshold is not a recovery
        assert!(b.observe(140, 4 * G + 1, 0).is_empty());
        // 0.5 GiB above it is
        let r = b.observe(141, 5 * G, 0);
        assert!(matches!(&r[0], Event::Recovered(s) if s.contains("RECOVERED") && s.contains("lowest 3.00 GiB") && s.contains("after 41 s")), "{r:?}");
        // and the next dip is a new episode
        assert!(matches!(&b.observe(150, 2 * G, 0)[0], Event::Low(s) if s.contains("low for 0 s")));
    }

    #[test]
    fn every_ten_minutes_a_summary_of_min_mean_and_max() {
        let mut b = RamBox::new(2 * G);
        let mut summaries = vec![];
        for t in 0..=1200u64 {
            let avail = if t % 2 == 0 { 6 * G } else { 10 * G };
            summaries.extend(b.observe(5000 + t, avail, 40 * G).into_iter().filter(|e| matches!(e, Event::Summary(_))));
        }
        assert_eq!(summaries.len(), 2, "{summaries:?}");
        let Event::Summary(s) = &summaries[0] else { panic!() };
        assert!(s.contains("10 min") && s.contains("min 6.00 / mean 8.00 / max 10.00 GiB") && s.contains("40.0 GiB pinned"), "{s}");
    }

    #[test]
    fn the_date_is_right_and_tasklist_text_is_read_by_programs_summed() {
        assert_eq!(utc(0), "1970-01-01 00:00:00 UTC");
        assert_eq!(utc(1_700_000_000), "2023-11-14 22:13:20 UTC");
        assert_eq!(utc(1_782_864_000), "2026-07-01 00:00:00 UTC");
        let csv = "\"shadow.exe\",\"19736\",\"Console\",\"1\",\"41,689,492 K\"\r\n\"chrome.exe\",\"10\",\"Console\",\"1\",\"500,000 K\"\r\n\"chrome.exe\",\"11\",\"Console\",\"1\",\"700,000 K\"\r\n\"Weird, name.exe\",\"12\",\"Console\",\"1\",\"9 K\"\r\nINFO: nothing\r\n";
        let top = top_processes(csv, 3);
        assert_eq!(top, vec![("shadow.exe".to_string(), 41_689_492), ("chrome.exe".to_string(), 1_200_000), ("Weird, name.exe".to_string(), 9)]);
    }

    #[cfg(windows)]
    #[test]
    fn tasklist_really_answers_here() {
        let top = top_processes_now(5);
        assert!(!top.is_empty() && top.iter().all(|(n, k)| n.to_lowercase().ends_with(".exe") && *k > 0), "{top:?}");
        assert!(top.windows(2).all(|w| w[0].1 >= w[1].1), "largest first");
    }
}
