//! M8 option A: shadow-hostd between a client and an engine (the fake one, `shadow-fake-engine`).
//!
//! The claims, each a test: the proxy changes no byte of either stream; it reports a broken rule on stderr and leaves
//! stdout as the engine wrote it; strict mode ends the engine; the engine's exit code is the proxy's; and a client
//! that closes stdin ends both.

use std::io::Write;
use std::process::{Command, Output, Stdio};

const HOSTD: &str = env!("CARGO_BIN_EXE_shadow-hostd");
const FAKE: &str = env!("CARGO_BIN_EXE_shadow-fake-engine");

const SESSION: &str = "GEN 3 temperature=0.6 1 2 3\nENC 0\nBGEN 1 4 1 2\nGEN 1 5\nQUIT\n";

/// Like the server: read up to READY before the first request (server.py reads READY in StrataEngine's start, before
/// any request can reach the pipe), then send the session and collect everything.  A client that sent first would be
/// breaking the protocol itself - which is a rule the proxy checks.
fn run(cmd: &mut Command, input: &str) -> Output {
    use std::io::{BufRead, BufReader, Read};
    let mut child = cmd.stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::piped()).spawn().expect("spawn");
    let mut stdin = child.stdin.take().unwrap();
    let mut out = BufReader::new(child.stdout.take().unwrap());
    let mut seen = Vec::new();
    loop {
        let mut line = Vec::new();
        if out.read_until(b'\n', &mut line).unwrap_or(0) == 0 {
            break; // the engine never got ready (a refused start): nothing to wait for
        }
        let ready = line.starts_with(b"READY");
        seen.extend_from_slice(&line);
        if ready {
            break;
        }
    }
    // one request at a time, read to its end line - "requests are serialized by the service's FIFO" (server.py)
    for req in input.lines() {
        // a proxy that ended (strict mode) has closed the pipe: that is its answer, not the test's failure
        if stdin.write_all(format!("{req}\n").as_bytes()).and_then(|_| stdin.flush()).is_err() {
            break;
        }
        let end: &[u8] = match req.split_whitespace().next() {
            Some("GEN") | Some("GENI") => b"DONE",
            Some("BGEN") | Some("BGENI") => b"BADM",
            Some("ENC") => b"OK",
            _ => continue,
        };
        loop {
            let mut line = Vec::new();
            if out.read_until(b'\n', &mut line).unwrap_or(0) == 0 {
                break;
            }
            let done = line.starts_with(end) || line.starts_with(b"ERR");
            seen.extend_from_slice(&line);
            if done {
                break;
            }
        }
    }
    drop(stdin);
    out.read_to_end(&mut seen).unwrap();
    let mut o = child.wait_with_output().expect("wait");
    o.stdout = seen;
    o
}

fn direct(env: &[(&str, &str)], input: &str) -> Output {
    let mut c = Command::new(FAKE);
    c.arg("--serve");
    for (k, v) in env {
        c.env(k, v);
    }
    run(&mut c, input)
}

fn proxied(env: &[(&str, &str)], input: &str) -> Output {
    let mut c = Command::new(HOSTD);
    c.arg("--serve").env("STRATA_HOSTD_ENGINE", FAKE);
    for (k, v) in env {
        c.env(k, v);
    }
    run(&mut c, input)
}

#[test]
fn the_proxy_changes_no_byte() {
    let a = direct(&[], SESSION);
    let b = proxied(&[], SESSION);
    assert!(a.stdout.windows(2).any(|w| w == b"\r\n"), "the fake engine's \\r line is missing: the test proves less");
    assert_eq!(b.stdout, a.stdout, "the stream through the proxy differs from the engine's own");
    let err = String::from_utf8_lossy(&b.stderr);
    assert!(err.contains("0 rule breaks"), "a clean session was judged: {err}");
    assert!(err.contains("3 requests"), "the proxy did not see the session's requests: {err}");
}

#[test]
fn a_broken_rule_is_reported_on_stderr_and_stdout_is_untouched() {
    for brk in ["stray_t", "bt_unknown_slot", "double_done"] {
        let env = [("FAKE_BREAK", brk)];
        let a = direct(&env, SESSION);
        let b = proxied(&env, SESSION);
        assert_eq!(b.stdout, a.stdout, "{brk}: observe mode changed the stream");
        let err = String::from_utf8_lossy(&b.stderr);
        assert!(err.contains("rule broken"), "{brk}: not reported: {err}");
        assert!(!String::from_utf8_lossy(&b.stdout).contains("shadow-hostd"), "{brk}: a report reached stdout");
    }
}

#[test]
fn strict_mode_ends_the_engine() {
    let b = proxied(&[("FAKE_BREAK", "stray_t"), ("STRATA_HOSTD_STRICT", "1")], SESSION);
    assert_eq!(b.status.code(), Some(3), "strict mode did not exit 3: {}", String::from_utf8_lossy(&b.stderr));
}

#[test]
fn the_engines_exit_code_is_the_proxys() {
    for code in ["0", "7"] {
        let b = proxied(&[("FAKE_EXIT", code)], "QUIT\n");
        assert_eq!(b.status.code(), Some(code.parse().unwrap()));
    }
}

#[test]
fn closing_stdin_ends_both() {
    // no QUIT: the client just goes away.  The engine sees its stdin end, exits, and so does the proxy.
    let b = proxied(&[], "GEN 1 1\n");
    assert_eq!(b.status.code(), Some(0));
    assert!(String::from_utf8_lossy(&b.stdout).contains("DONE"), "the request before the close was not answered");
}

/// The guarantee that matters on a machine where an engine pins 36 GiB: kill the proxy outright (TerminateProcess -
/// no stdin EOF, no cleanup of its own) and the engine goes with it, through the kill-on-close job.
#[cfg(windows)]
#[test]
fn a_killed_proxy_takes_the_engine_with_it() {
    use std::io::{BufRead, BufReader};
    let mut child = Command::new(HOSTD).arg("--serve").env("STRATA_HOSTD_ENGINE", FAKE).env("FAKE_LINGER", "1")
        .stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::piped()).spawn().expect("spawn");
    let mut err = BufReader::new(child.stderr.take().unwrap());
    // the line that names the engine's pid (a warning may come before it)
    let mut first = String::new();
    let pid: u32 = loop {
        first.clear();
        assert!(err.read_line(&mut first).unwrap() > 0, "the proxy never named the engine's pid");
        if let Some(p) = first.split("(pid ").nth(1).and_then(|s| s.split(')').next()).and_then(|s| s.parse().ok()) {
            break p;
        }
    };
    let alive = |pid: u32| {
        let o = Command::new("tasklist").args(["/FI", &format!("PID eq {pid}"), "/NH"]).output().unwrap();
        String::from_utf8_lossy(&o.stdout).contains(&pid.to_string())
    };
    assert!(alive(pid), "the engine was not running before the kill - the test proves nothing");
    child.kill().unwrap();
    let _ = child.wait();
    let gone = (0..50).any(|_| {
        std::thread::sleep(std::time::Duration::from_millis(100));
        !alive(pid)
    });
    assert!(gone, "the engine (pid {pid}) outlived its killed proxy");
}

#[test]
fn a_relative_or_missing_engine_is_refused() {
    let mut c = Command::new(HOSTD);
    c.env_remove("STRATA_HOSTD_ENGINE");
    assert_eq!(run(&mut c, "").status.code(), Some(2));
    let mut c = Command::new(HOSTD);
    c.env("STRATA_HOSTD_ENGINE", "strata.exe");
    let o = run(&mut c, "");
    assert_eq!(o.status.code(), Some(2));
    assert!(String::from_utf8_lossy(&o.stderr).contains("absolute"));
}
