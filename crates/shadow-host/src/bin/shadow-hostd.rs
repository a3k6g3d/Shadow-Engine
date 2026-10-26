//! shadow-hostd: M8 option A of docs/RUST_ENGINE.md - a supervising proxy between `serve/server.py` and the engine.
//!
//! Point the config's `exe` at this binary and set `STRATA_HOSTD_ENGINE` to the real `strata.exe` (absolute path; the
//! config's `env` block carries it).  The server starts `shadow-hostd --serve <args>`; this starts
//! `<engine> --serve <args>` and sits between the two pipes:
//!
//!   * **Every byte forwarded as it came.**  Lines are read and written as raw bytes, `\r` and all, and flushed one by
//!     one (a token line is not held back for the next).  The engine's stderr is inherited: it lands in the same log.
//!   * **The request state machine checked** (`shadow_host::protocol`): a rule broken is a `shadow-hostd:` line on
//!     stderr - never on stdout, which the server parses - and the stream goes on unchanged.  `STRATA_HOSTD_STRICT=1`
//!     (tests) ends the engine and exits 3 instead.
//!   * **The engine never outlives the proxy.**  It runs in a job object that is killed when the last handle to it -
//!     this process's - closes, however this process ends.  (Without it, a proxy killed outright would leave an engine
//!     holding its pinned RAM; the engine does end at stdin EOF, but only after the request it is in.)
//!   * **The engine's exit code is this one's**, so the server's crash reading (`engine_crashed`) sees the engine.

use std::io::{BufRead, BufReader, Read, Write};
use std::process::{Command, Stdio};
use std::sync::{Arc, Mutex};
use shadow_host::protocol::Checker;

const STRICT_EXIT: i32 = 3;
const SAID_MAX: u64 = 50;

fn say(msg: &str) {
    eprintln!("shadow-hostd: {msg}");
}

/// Read one line as raw bytes, newline included.  Empty at end of stream.
fn read_raw(r: &mut impl BufRead, buf: &mut Vec<u8>) -> std::io::Result<usize> {
    buf.clear();
    r.read_until(b'\n', buf)
}

fn text(raw: &[u8]) -> String {
    String::from_utf8_lossy(raw).trim_end_matches(['\n', '\r']).to_string()
}

#[cfg(windows)]
mod job {
    use std::os::raw::c_void;
    type Handle = *mut c_void;

    #[repr(C)]
    #[derive(Default)]
    struct BasicLimit {
        per_process_user_time: i64,
        per_job_user_time: i64,
        limit_flags: u32,
        minimum_working_set: usize,
        maximum_working_set: usize,
        active_process_limit: u32,
        affinity: usize,
        priority_class: u32,
        scheduling_class: u32,
    }
    #[repr(C)]
    #[derive(Default)]
    struct IoCounters {
        _c: [u64; 6],
    }
    #[repr(C)]
    #[derive(Default)]
    struct ExtendedLimit {
        basic: BasicLimit,
        io: IoCounters,
        process_memory_limit: usize,
        job_memory_limit: usize,
        peak_process_memory_used: usize,
        peak_job_memory_used: usize,
    }
    const JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE: u32 = 0x2000;
    const JOB_OBJECT_EXTENDED_LIMIT_INFORMATION: i32 = 9;
    const PROCESS_SET_QUOTA_TERMINATE: u32 = 0x0100 | 0x0001;

    extern "system" {
        fn CreateJobObjectW(attrs: *mut c_void, name: *const u16) -> Handle;
        fn SetInformationJobObject(job: Handle, class: i32, info: *mut c_void, len: u32) -> i32;
        fn AssignProcessToJobObject(job: Handle, process: Handle) -> i32;
        fn OpenProcess(access: u32, inherit: i32, pid: u32) -> Handle;
        fn CloseHandle(h: Handle) -> i32;
    }

    /// Put `pid` in a fresh kill-on-close job.  The job handle is leaked on purpose: it closes when this process
    /// ends, and that close is what kills the engine.  Err with the reason when the OS refuses.
    pub fn bind(pid: u32) -> Result<(), String> {
        unsafe {
            let job = CreateJobObjectW(std::ptr::null_mut(), std::ptr::null());
            if job.is_null() {
                return Err("CreateJobObjectW failed".into());
            }
            let mut info = ExtendedLimit::default();
            info.basic.limit_flags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
            if SetInformationJobObject(job, JOB_OBJECT_EXTENDED_LIMIT_INFORMATION, &mut info as *mut _ as *mut c_void,
                                       std::mem::size_of::<ExtendedLimit>() as u32) == 0 {
                return Err("SetInformationJobObject failed".into());
            }
            let p = OpenProcess(PROCESS_SET_QUOTA_TERMINATE, 0, pid);
            if p.is_null() {
                return Err("OpenProcess failed".into());
            }
            let ok = AssignProcessToJobObject(job, p);
            CloseHandle(p);
            if ok == 0 {
                return Err("AssignProcessToJobObject failed".into());
            }
            Ok(())
        }
    }
}

fn main() {
    let engine = match std::env::var("STRATA_HOSTD_ENGINE") {
        Ok(e) if std::path::Path::new(&e).is_absolute() => e,
        Ok(e) => {
            say(&format!("STRATA_HOSTD_ENGINE must be an absolute path (got {e})"));
            std::process::exit(2);
        }
        Err(_) => {
            say("STRATA_HOSTD_ENGINE is not set: it names the engine this proxy starts");
            std::process::exit(2);
        }
    };
    let strict = std::env::var("STRATA_HOSTD_STRICT").is_ok_and(|v| v == "1");
    let args: Vec<String> = std::env::args().skip(1).collect();

    let mut child = match Command::new(&engine).args(&args).stdin(Stdio::piped()).stdout(Stdio::piped())
        .stderr(Stdio::inherit()).spawn()
    {
        Ok(c) => c,
        Err(e) => {
            say(&format!("cannot start {engine}: {e}"));
            std::process::exit(2);
        }
    };
    #[cfg(windows)]
    if let Err(why) = job::bind(child.id()) {
        say(&format!("the engine is not bound to this process ({why}): it can outlive a killed proxy"));
    }
    say(&format!("forwarding to {engine} (pid {}){}", child.id(), if strict { ", strict" } else { "" }));

    let checker = Arc::new(Mutex::new(Checker::new()));
    let said = Arc::new(Mutex::new(0u64));
    let report = {
        let said = said.clone();
        move |dir: &str, line: &str, why: String| {
            let mut n = said.lock().unwrap();
            *n += 1;
            if *n <= SAID_MAX {
                let shown: String = line.chars().take(80).collect();
                say(&format!("rule broken ({dir}): {why} - line {shown:?}"));
            }
        }
    };

    // server -> engine.  Checked *before* it is forwarded, so the checker has seen a request before any answer to it.
    let mut to_engine = child.stdin.take().expect("piped stdin");
    let c_in = checker.clone();
    let report_in = report.clone();
    let pid = child.id();
    std::thread::spawn(move || {
        let mut input = BufReader::new(std::io::stdin().lock());
        let mut buf = Vec::with_capacity(4096);
        loop {
            match read_raw(&mut input, &mut buf) {
                Ok(0) | Err(_) => break,
                Ok(_) => {}
            }
            let line = text(&buf);
            if let Some(why) = c_in.lock().unwrap().to_engine(&line) {
                report_in("to engine", &line, why);
                if strict {
                    kill(pid);
                    std::process::exit(STRICT_EXIT);
                }
            }
            if to_engine.write_all(&buf).and_then(|_| to_engine.flush()).is_err() {
                break; // the engine is gone; its stdout's end tells the main thread
            }
        }
        // stdin closed: close the engine's, which is how it learns to finish (it ends after the request it is in)
        drop(to_engine);
    });

    // engine -> server, on this thread: checked, then forwarded and flushed line by line.
    let mut from_engine = BufReader::new(child.stdout.take().expect("piped stdout"));
    let stdout = std::io::stdout();
    let mut out = stdout.lock();
    let mut buf = Vec::with_capacity(4096);
    loop {
        match read_raw(&mut from_engine, &mut buf) {
            Ok(0) | Err(_) => break,
            Ok(_) => {}
        }
        let line = text(&buf);
        if let Some(why) = checker.lock().unwrap().from_engine(&line) {
            report("from engine", &line, why);
            if strict {
                let _ = child.kill();
                std::process::exit(STRICT_EXIT);
            }
        }
        if out.write_all(&buf).and_then(|_| out.flush()).is_err() {
            break; // the server is gone; the job object takes the engine with this process
        }
    }
    // drain anything left without a newline at the very end, byte for byte
    let mut tail = Vec::new();
    let _ = from_engine.read_to_end(&mut tail);
    if !tail.is_empty() {
        let _ = out.write_all(&tail).and_then(|_| out.flush());
    }
    let status = child.wait();
    let summary = checker.lock().unwrap().summary();
    let code = status.ok().and_then(|s| s.code()).unwrap_or(1);
    say(&format!("engine exited {code}: {summary}"));
    std::process::exit(code);
}

fn kill(pid: u32) {
    // the strict path from the input thread, which no longer has the Child: the job would take the engine when this
    // process exits anyway, so this is only to make it immediate on platforms without the job
    #[cfg(not(windows))]
    {
        let _ = Command::new("kill").arg("-9").arg(pid.to_string()).status();
    }
    #[cfg(windows)]
    let _ = pid;
}
