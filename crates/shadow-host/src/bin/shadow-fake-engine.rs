//! A stand-in for `strata --serve`, for shadow-hostd's tests only: the line protocol's shape with no model behind it.
//!
//!   INFO + READY at start; `GEN <n> ...` -> `RESUME 0`, `PP`, n x `T`, `DONE`; `BGEN <slot> <n> ...` -> `T`, `DONE`,
//!   `BADM <slot> 1`, n-1 x `BT`, `BDONE`; `ENC` -> `OK`; `QUIT` (or stdin's end) -> exit with FAKE_EXIT (default 0).
//!   FAKE_BREAK=stray_t | bt_unknown_slot | double_done makes it break one rule after READY, so a test can see the
//!   proxy catch it.  One output line carries a `\r` before its newline, so byte-for-byte forwarding is tested too.

use std::io::{BufRead, Write};

fn main() {
    let exit: i32 = std::env::var("FAKE_EXIT").ok().and_then(|v| v.parse().ok()).unwrap_or(0);
    let brk = std::env::var("FAKE_BREAK").unwrap_or_default();
    let out = std::io::stdout();
    let mut o = out.lock();
    let mut put = |s: &str| {
        o.write_all(s.as_bytes()).unwrap();
        o.flush().unwrap();
    };
    put("INFO kv=int8 slots=2\n");
    put("READY 4096 stop\r\n");
    match brk.as_str() {
        "stray_t" => put("T 99\n"),
        "bt_unknown_slot" => put("BT 7 1\n"),
        _ => {}
    }
    for line in std::io::stdin().lock().lines() {
        let Ok(line) = line else { break };
        let f: Vec<&str> = line.split_whitespace().collect();
        match f.first().copied() {
            Some("GEN") => {
                let n: i64 = f.get(1).and_then(|v| v.parse().ok()).unwrap_or(1);
                put("RESUME 0\n");
                put("PP 3 3 1 100.0\n");
                for i in 0..n {
                    put(&format!("T {}\n", 100 + i));
                }
                put(&format!("DONE 3 {n} 1.0 2.0 len\n"));
                if brk == "double_done" {
                    put(&format!("DONE 3 {n} 1.0 2.0 len\n"));
                }
            }
            Some("BGEN") => {
                let slot: i64 = f.get(1).and_then(|v| v.parse().ok()).unwrap_or(0);
                let n: i64 = f.get(2).and_then(|v| v.parse().ok()).unwrap_or(1);
                put("T 200\n");
                put("DONE 3 1 1.0 1.0 len\n");
                put(&format!("BADM {slot} {}\n", if n > 1 { 1 } else { 0 }));
                if n > 1 {
                    for i in 1..n {
                        put(&format!("BT {slot} {}\n", 200 + i));
                    }
                    put(&format!("BDONE {slot} {n} len 3.0\n"));
                }
            }
            Some("ENC") => put("OK 4\n"),
            Some("QUIT") => break,
            _ => {}
        }
    }
    // FAKE_LINGER=1: stay alive 30 s past stdin's end, as a real engine does while it finishes a long request - so a
    // test can tell the proxy's job object from the engine simply seeing EOF
    if std::env::var("FAKE_LINGER").is_ok_and(|v| v == "1") {
        std::thread::sleep(std::time::Duration::from_secs(30));
    }
    std::process::exit(exit);
}
