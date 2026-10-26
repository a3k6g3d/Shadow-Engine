//! M8 of docs/RUST_ENGINE.md, option A: the request state machine of the engine's line protocol, checked.
//!
//! `serve/server.py` drives `strata --serve` over stdin/stdout.  The server writes requests (`GEN`, `GENI`, `BGEN`,
//! `BGENI`) and controls (`STOP`, `BSTOP`, `BYIELD`, `SAVE`, `RESTORE`, `VRAM`, `ENC`, `QUIT`); the engine answers with
//! `INFO`/`READY` at start, `PP`/`RESUME`/`T` while a request runs, `DONE` (solo) or `DONE` + `BADM <slot> <cont>` (a
//! batch admission), then `BT <slot> <id>` ... `BDONE <slot> ...` for a continuing slot, `ERR` for a refusal, and `OK`,
//! `VRAM`, `YIELDED` beside them.  The rules nobody checks today, all read off both ends (server.py `_control`,
//! generate.cpp's serve loop):
//!
//!   * nothing but `INFO`/`ERR` before `READY`, and no request before it;
//!   * one control request at a time (the server serializes them): a request ends at `DONE` (solo), at `BADM`
//!     (admission), or at an `ERR` (refused - "no DONE follows");
//!   * `T`, `PP`, `RESUME`, `YIELDED` only while a request is open;
//!   * `BT`/`BDONE` only for a slot an admission left running (`BADM <slot> 1`), and `BDONE` closes it.
//!
//! The checker only reports.  It never decides what is forwarded - the proxy forwards every byte as it came - so a
//! rule written here too strictly costs a stderr line, not a request.  Lines it does not know are counted, not judged.

use std::collections::BTreeSet;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Open {
    Solo,
    Admit(i64),
}

#[derive(Debug, Default)]
pub struct Checker {
    ready: bool,
    open: Option<Open>,
    slots: BTreeSet<i64>,
    pub requests: u64,
    pub tokens: u64,
    pub unknown: u64,
    pub violations: u64,
}

fn word(line: &str) -> (&str, Vec<&str>) {
    let mut it = line.split_whitespace();
    let w = it.next().unwrap_or("");
    (w, it.collect())
}

fn num(s: Option<&&str>) -> Option<i64> {
    s.and_then(|v| v.parse::<i64>().ok())
}

impl Checker {
    pub fn new() -> Checker {
        Checker::default()
    }

    fn bad(&mut self, why: String) -> Option<String> {
        self.violations += 1;
        Some(why)
    }

    /// A line the server sent the engine.  Some(reason) when it breaks a rule.
    pub fn to_engine(&mut self, line: &str) -> Option<String> {
        let (w, rest) = word(line);
        match w {
            "GEN" | "GENI" | "BGEN" | "BGENI" => {
                self.requests += 1;
                let refused = if !self.ready {
                    Some(format!("{w} before READY"))
                } else if let Some(o) = self.open {
                    Some(format!("{w} while a request is still open ({o:?})"))
                } else {
                    None
                };
                self.open = Some(if w.starts_with('B') { Open::Admit(num(rest.first()).unwrap_or(-1)) } else { Open::Solo });
                match refused {
                    Some(r) => self.bad(r),
                    None => None,
                }
            }
            _ => None, // STOP, BSTOP, BYIELD, SAVE, RESTORE, VRAM, ENC, QUIT: no request state of their own
        }
    }

    /// A line the engine sent the server.  Some(reason) when it breaks a rule.
    pub fn from_engine(&mut self, line: &str) -> Option<String> {
        let (w, rest) = word(line);
        match w {
            "INFO" => None,
            "READY" => {
                if self.ready {
                    return self.bad("a second READY from the same engine".into());
                }
                self.ready = true;
                None
            }
            "ERR" => {
                self.open = None; // a refusal: the engine is idle again, and no DONE follows
                None
            }
            _ if !self.ready => self.bad(format!("{w} before READY")),
            "T" | "PP" | "RESUME" | "YIELDED" => {
                if w == "T" {
                    self.tokens += 1;
                }
                if self.open.is_none() {
                    return self.bad(format!("{w} outside a request"));
                }
                None
            }
            "DONE" => match self.open {
                None => self.bad("DONE with no request open".into()),
                Some(Open::Solo) => {
                    self.open = None;
                    None
                }
                Some(Open::Admit(_)) => None, // an admission's DONE: its BADM ends it
            },
            "BADM" => {
                let slot = num(rest.first()).unwrap_or(-1);
                let cont = rest.get(1) == Some(&"1");
                match self.open {
                    Some(Open::Admit(s)) if s == slot => {
                        self.open = None;
                        if cont {
                            self.slots.insert(slot);
                        }
                        None
                    }
                    other => {
                        self.open = None;
                        self.bad(format!("BADM {slot} while {other:?} was open"))
                    }
                }
            }
            "BT" => {
                self.tokens += 1;
                let slot = num(rest.first()).unwrap_or(-1);
                if self.slots.contains(&slot) {
                    None
                } else {
                    self.bad(format!("BT for slot {slot}, which no admission left running"))
                }
            }
            "BDONE" => {
                let slot = num(rest.first()).unwrap_or(-1);
                if self.slots.remove(&slot) {
                    None
                } else {
                    self.bad(format!("BDONE for slot {slot}, which no admission left running"))
                }
            }
            "OK" | "VRAM" => None,
            _ => {
                self.unknown += 1;
                None
            }
        }
    }

    pub fn summary(&self) -> String {
        format!("{} requests, {} tokens, {} lines not known, {} rule breaks", self.requests, self.tokens, self.unknown,
                self.violations)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ready() -> Checker {
        let mut c = Checker::new();
        assert_eq!(c.from_engine("INFO kv=int8"), None);
        assert_eq!(c.from_engine("READY 65536 stop"), None);
        c
    }

    #[test]
    fn a_solo_request_and_a_batch_admission_pass_clean() {
        let mut c = ready();
        for (dir, line) in [("s", "GEN 4 temperature=0.6 1 2 3"), ("e", "RESUME 0"), ("e", "PP 3 3 12 250.0"),
                            ("e", "T 5"), ("e", "T 6"), ("e", "DONE 3 2 1.0 2.0 eos"),
                            ("s", "BGEN 1 8 1 2 3"), ("e", "T 9"), ("e", "DONE 3 1 1.0 1.0 len"), ("e", "BADM 1 1"),
                            ("e", "BT 1 10"), ("s", "GEN 2 4 5"), ("e", "BT 1 11"), ("e", "T 3"), ("e", "BDONE 1 3 eos 9.0"),
                            ("e", "DONE 2 1 1.0 1.0 eos"), ("s", "STOP"), ("s", "QUIT")] {
            let r = if dir == "s" { c.to_engine(line) } else { c.from_engine(line) };
            assert_eq!(r, None, "{line:?} was judged a break");
        }
        assert_eq!((c.requests, c.tokens, c.violations), (3, 6, 0));   // T 5, 6, 9, 3 and BT 10, 11
    }

    #[test]
    fn a_refusal_closes_the_request_without_a_done() {
        let mut c = ready();
        c.to_engine("BGEN 3 8 1 2");
        assert_eq!(c.from_engine("ERR BGEN: no such free slot"), None);
        assert_eq!(c.to_engine("GEN 2 1"), None, "after an ERR the engine is idle: a new request is fine");
    }

    #[test]
    fn each_rule_is_caught() {
        let mut c = Checker::new();
        assert!(c.from_engine("T 1").is_some(), "T before READY");
        let mut c = ready();
        assert!(c.from_engine("T 1").is_some(), "T outside a request");
        assert!(c.from_engine("DONE 1 1 1 1 eos").is_some(), "DONE with nothing open");
        c.to_engine("GEN 2 1");
        assert!(c.to_engine("GEN 2 1").is_some(), "a second request while one is open");
        let mut c = ready();
        assert!(c.from_engine("BT 2 5").is_some(), "BT for a slot never admitted");
        assert!(c.from_engine("BDONE 2 1 eos 1.0").is_some(), "BDONE for a slot never admitted");
        c.to_engine("BGEN 1 4 1");
        c.from_engine("DONE 1 1 1 1 len");
        assert!(c.from_engine("BADM 2 1").is_some(), "BADM for another slot than the one admitted");
        let mut c = ready();
        assert!(c.from_engine("READY 4096 stop").is_some(), "a second READY");
        assert_eq!(c.violations, 1);
    }

    #[test]
    fn a_slot_that_finished_in_its_admission_takes_no_bt() {
        let mut c = ready();
        c.to_engine("BGEN 0 1 7");
        c.from_engine("T 4");
        c.from_engine("DONE 1 1 1 1 len");
        assert_eq!(c.from_engine("BADM 0 0"), None);
        assert!(c.from_engine("BT 0 5").is_some(), "BADM 0 0 left nothing running");
    }

    #[test]
    fn unknown_lines_are_counted_not_judged() {
        let mut c = ready();
        assert_eq!(c.from_engine("SOMETHING new 1 2"), None);
        assert_eq!((c.unknown, c.violations), (1, 0));
    }
}
