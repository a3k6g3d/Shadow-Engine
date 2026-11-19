//! Remote pairing (docs/PLAN_APK_2026-11-13.md, Phase 1): a 4-digit pin typed on the desktop
//! (shown by GET /pair/pin, loopback only) exchanged over POST /pair for a 32-byte bearer
//! token the phone keeps; every later /v1/* request from a non-loopback peer must present
//! it, as `Authorization: Bearer` or, for EventSource which cannot set headers, `?token=`.
//! Loopback peers are always allowed, so the embedded UI and shadowzero-bench keep working
//! exactly as before, and a server bound to 127.0.0.1 behaves as if auth did not exist.
//!
//! The pin lives in memory only and dies with the server or after ten minutes (five wrong
//! tries burn it); the token's SHA-256 lives in %LOCALAPPDATA%\ShadowZero\pair.json so a
//! paired phone survives restarts.  Hashes, never the secret, on disk.

use sha2::{Digest, Sha256};
use std::path::PathBuf;
use std::sync::Mutex;

const PIN_TTL: std::time::Duration = std::time::Duration::from_secs(600);
const MAX_TRIES: u32 = 5;

fn store_path() -> PathBuf {
    std::env::var_os("LOCALAPPDATA").map_or_else(
        || std::env::temp_dir().join("shadowzero-pair.json"),
        |a| PathBuf::from(a).join("ShadowZero").join("pair.json"),
    )
}

/// True for 127.0.0.0/8 and ::1 (and ::ffff:127.x): these peers skip the token entirely.
pub fn is_local(addr: std::net::SocketAddr) -> bool {
    match addr.ip() {
        std::net::IpAddr::V4(v4) => v4.is_loopback(),
        std::net::IpAddr::V6(v6) => {
            v6.is_loopback()
                || v6
                    .to_ipv4_mapped()
                    .is_some_and(|v4| std::net::Ipv4Addr::is_loopback(&v4))
        }
    }
}

/// Equal-length constant-time compare of two hex strings; leaks only the lengths, which
/// here are fixed (both SHA-256 digests, 64 chars).
fn ct_eq(a: &str, b: &str) -> bool {
    let (a, b) = (a.as_bytes(), b.as_bytes());
    if a.len() != b.len() {
        return false;
    }
    let mut diff = 0u8;
    for (x, y) in a.iter().zip(b) {
        diff |= x ^ y;
    }
    diff == 0
}

fn hex_digest(secret: &str) -> String {
    let mut h = Sha256::new();
    h.update(secret.as_bytes());
    h.finalize()
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

fn random_bytes(n: usize) -> Vec<u8> {
    let mut buf = vec![0u8; n];
    // The OS CSPRNG; it filling 32 bytes cannot realistically fail, and a failure here
    // would leave the server unable to pair at all, so unwrapping is the honest answer.
    getrandom::getrandom(&mut buf).expect("no OS entropy for pairing");
    buf
}

/// A fresh 4-digit pin, zero-padded (uniform enough: 10000 has no bias worth speaking of
/// against a 32-bit draw).
fn random_pin() -> String {
    let n = u32::from_be_bytes(random_bytes(4).try_into().unwrap()) % 10_000;
    format!("{n:04}")
}

struct Pin {
    /// Kept in plain text as long as it lives: GET /pair/pin re-shows the same pin while
    /// it is live (a desktop that closed the dialog and reopens it must not have to
    /// re-pair), and it is never written to disk.
    value: String,
    hash: String,
    expires: std::time::Instant,
    tries: u32,
}

#[derive(Default)]
struct Inner {
    /// SHA-256 of the bearer token; persisted; None until someone pairs.
    token_hash: Option<String>,
    /// The live pin, if one is out; never persisted.
    pin: Option<Pin>,
}

pub struct Pairing {
    inner: Mutex<Inner>,
    ttl: std::time::Duration,
    /// Where the token hash is read from and written to; the real store for the
    /// server, a private temp file per test (so tests neither race each other nor
    /// touch the user's actual pairing).
    path: PathBuf,
}

/// Why a pairing attempt failed, as the message the phone shows.
#[derive(Debug)]
pub enum PairError {
    NoPin,
    Expired,
    Wrong(u32),
}

impl PairError {
    pub fn message(&self) -> String {
        match self {
            PairError::NoPin => "no pin is active on the desktop".into(),
            PairError::Expired => "the pin has expired".into(),
            PairError::Wrong(left) => format!("wrong pin ({left} tries left)"),
        }
    }
}

impl Pairing {
    /// Loads the stored token hash if there is one (a phone paired before this restart).
    pub fn load() -> Pairing {
        let token_hash = std::fs::read(store_path())
            .ok()
            .and_then(|b| serde_json::from_slice::<serde_json::Value>(&b).ok())
            .and_then(|v| {
                v.get("token_sha256")
                    .and_then(|t| t.as_str())
                    .map(|t| t.to_string())
            })
            .filter(|t| t.len() == 64);
        Pairing {
            inner: Mutex::new(Inner { token_hash, pin: None }),
            ttl: PIN_TTL,
            path: store_path(),
        }
    }

    #[cfg(test)]
    fn with_ttl(ttl: std::time::Duration) -> Pairing {
        // A private store file per test: neither the real pairing nor a sibling test's.
        let path = std::env::temp_dir().join(format!(
            "shadowzero-pair-test-{}-{:x}.json",
            std::process::id(),
            std::time::Instant::now().elapsed().as_nanos()
        ));
        let _ = std::fs::remove_file(&path);
        Pairing {
            inner: Mutex::new(Inner::default()),
            ttl,
            path,
        }
    }

    /// Show on the desktop (loopback only): hand out a pin, reusing the live one so the
    /// dialog can be reopened.  Answers {pin, expires_in_s} for the UI to print and tick down.
    pub fn pin(&self) -> serde_json::Value {
        let mut inner = self.inner.lock().unwrap();
        let now = std::time::Instant::now();
        let expired = inner.pin.as_ref().is_some_and(|p| p.expires <= now);
        if expired {
            inner.pin = None;
        }
        let (pin, left) = match &inner.pin {
            Some(p) => (p.value.clone(), p.expires.saturating_duration_since(now).as_secs()),
            None => {
                let pin = random_pin();
                inner.pin = Some(Pin {
                    value: pin.clone(),
                    hash: hex_digest(&pin),
                    expires: now + self.ttl,
                    tries: 0,
                });
                (pin, self.ttl.as_secs())
            }
        };
        serde_json::json!({"pin": pin, "expires_in_s": left})
    }

    /// What the phone asks before typing: is there a pin waiting, and is a token already stored?
    pub fn status(&self) -> serde_json::Value {
        let inner = self.inner.lock().unwrap();
        let live = inner.pin.as_ref().is_some_and(|p| p.expires > std::time::Instant::now());
        serde_json::json!({
            "paired": inner.token_hash.is_some(),
            "pin_active": live,
        })
    }

    /// POST /pair: the typed pin in, the bearer token out (its only showing, ever).
    /// The pin burns on success and on the fifth wrong try.
    pub fn pair(&self, typed: &str) -> Result<String, PairError> {
        let mut inner = self.inner.lock().unwrap();
        let now = std::time::Instant::now();
        match &mut inner.pin {
            None => return Err(PairError::NoPin),
            Some(p) if p.expires <= now => {
                inner.pin = None;
                return Err(PairError::Expired);
            }
            Some(p) => {
                if !ct_eq(&p.hash, &hex_digest(typed.trim())) {
                    p.tries += 1;
                    if p.tries >= MAX_TRIES {
                        inner.pin = None;
                        return Err(PairError::Wrong(0));
                    }
                    return Err(PairError::Wrong(MAX_TRIES - p.tries));
                }
                inner.pin = None;
            }
        }
        let token = hex_of(&random_bytes(32));
        inner.token_hash = Some(hex_digest(&token));
        self.save(&inner);
        Ok(token)
    }

    /// A presented bearer token against the stored hash.
    pub fn check(&self, presented: &str) -> bool {
        let inner = self.inner.lock().unwrap();
        match &inner.token_hash {
            Some(h) => ct_eq(h, &hex_digest(presented.trim())),
            None => false,
        }
    }

    /// DELETE /pair: forget this phone (the caller's own token must match, so a stray
    /// request cannot unpair everyone).
    pub fn unpair(&self, presented: &str) -> Result<(), String> {
        let mut inner = self.inner.lock().unwrap();
        // An empty presented token means the caller is this machine (the desktop UI
        // has no token of its own): the pin dialog may forget a device from here.
        // Any token that *is* presented must be the paired one.
        if !presented.trim().is_empty()
            && !inner
                .token_hash
                .as_ref()
                .is_some_and(|h| ct_eq(h, &hex_digest(presented.trim())))
        {
            return Err("that token is not the paired one".into());
        }
        inner.token_hash = None;
        inner.pin = None;
        self.save(&inner);
        Ok(())
    }

    fn save(&self, inner: &Inner) {
        let v = serde_json::json!({"token_sha256": inner.token_hash});
        if let Some(p) = self.path.parent() {
            let _ = std::fs::create_dir_all(p);
        }
        let _ = std::fs::write(&self.path, serde_json::to_vec_pretty(&v).unwrap_or_default());
    }
}

fn hex_of(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// The CORS headers a browser (or the .apk's WebView fallback) needs: the server answers
/// from its own origin, and a native client sends no Origin at all and is unaffected.
/// Reflect whatever Origin asked, allow the verbs and the two headers the client uses.
pub fn cors<B>(origin: Option<&str>, resp: &mut hyper::Response<B>) {
    let Some(o) = origin.filter(|o| !o.is_empty()) else {
        return;
    };
    let Ok(o) = o.parse() else { return };
    let h = resp.headers_mut();
    h.insert("access-control-allow-origin", o);
    h.insert(
        "access-control-allow-methods",
        hyper::header::HeaderValue::from_static("GET, POST, PUT, DELETE, OPTIONS"),
    );
    h.insert(
        "access-control-allow-headers",
        hyper::header::HeaderValue::from_static("Authorization, Content-Type"),
    );
    h.insert(
        "access-control-max-age",
        hyper::header::HeaderValue::from_static("86400"),
    );
}

/// The bearer token: `Authorization: Bearer x` first, `?token=x` second (EventSource
/// cannot set headers, so /v1/events carries it in the query).
pub fn presented<B>(req: &hyper::Request<B>) -> String {
    let from_header = req
        .headers()
        .get("authorization")
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.strip_prefix("Bearer "))
        .map(|v| v.to_string());
    match from_header {
        Some(t) => t,
        None => req
            .uri()
            .query()
            .map(|q| crate::query_param(Some(q), "token"))
            .unwrap_or_default(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pin_flow() {
        let p = Pairing::with_ttl(std::time::Duration::from_secs(3600));
        // A fresh server has no token and no pin out.
        assert_eq!(p.status()["paired"], false);
        assert_eq!(p.status()["pin_active"], false);
        let v = p.pin();
        let pin = v["pin"].as_str().unwrap().to_string();
        assert_eq!(pin.len(), 4);
        assert!(pin.chars().all(|c| c.is_ascii_digit()));
        assert_eq!(p.status()["pin_active"], true);
        // Reopening the dialog re-shows the same live pin, not a fresh one.
        assert_eq!(p.pin()["pin"].as_str().unwrap(), pin.as_str());
        // Wrong pins: four tries say "n left", the fifth burns the pin.
        for left in [4, 3, 2, 1] {
            let e = p.pair("0000").unwrap_err();
            match e {
                PairError::Wrong(l) => assert_eq!(l, left),
                _ => panic!("expected Wrong({left})"),
            }
        }
        assert!(matches!(p.pair("0000").unwrap_err(), PairError::Wrong(0)));
        assert_eq!(p.status()["pin_active"], false);
        // A burned pin says so, and the desktop hands out a fresh one.
        assert!(matches!(p.pair(&pin).unwrap_err(), PairError::NoPin));
        let pin2 = p.pin()["pin"].as_str().unwrap().to_string();
        let token = p.pair(&pin2).unwrap();
        assert_eq!(token.len(), 64);
        assert!(token.chars().all(|c| c.is_ascii_digit() || ('a'..='f').contains(&c)));
        assert!(p.check(&token));
        assert!(!p.check(&token[..63]));
        assert!(!p.check(""));
        // The pin is spent: pairing again without a new pin fails.
        assert!(matches!(p.pair(&pin2).unwrap_err(), PairError::NoPin));
        assert_eq!(p.status()["paired"], true);
        // Unpairing with the wrong token refuses, with the right one clears.
        assert!(p.unpair("deadbeef").is_err());
        p.unpair(&token).unwrap();
        assert_eq!(p.status()["paired"], false);
        assert!(!p.check(&token));
        // The desktop path: an empty presented token forgets the device (the pin
        // dialog on this machine has no token of its own).  Re-pair, then forget
        // with the empty token.
        let pin3 = p.pin()["pin"].as_str().unwrap().to_string();
        let token2 = p.pair(&pin3).unwrap();
        assert!(p.unpair("").is_ok());
        assert_eq!(p.status()["paired"], false);
        assert!(!p.check(&token2));

        let _ = std::fs::remove_file(&p.path);
    }
    #[test]
    fn pin_expires() {
        let p = Pairing::with_ttl(std::time::Duration::from_millis(10));
        let pin = p.pin()["pin"].as_str().unwrap().to_string();
        std::thread::sleep(std::time::Duration::from_millis(30));
        assert_eq!(p.status()["pin_active"], false);
        assert!(matches!(p.pair(&pin).unwrap_err(), PairError::Expired));
        // status() only reports: the expired pin is gone and nothing was minted.
        assert_eq!(p.status()["pin_active"], false);
        // Asking the desktop again hands out a fresh pin, which pairs fine.
        let fresh = p.pin()["pin"].as_str().unwrap().to_string();
        assert_eq!(p.status()["pin_active"], true);
        assert!(p.pair(&fresh).is_ok());
        let _ = std::fs::remove_file(&p.path);
    }

    #[test]
    fn loopback_detection() {
        assert!(is_local("127.0.0.1:8081".parse().unwrap()));
        assert!(is_local("[::1]:8081".parse().unwrap()));
        assert!(is_local("[::ffff:127.0.0.1]:8081".parse().unwrap()));
        assert!(!is_local("192.168.1.20:52234".parse().unwrap()));
        assert!(!is_local("[fe80::1]:8081".parse().unwrap()));
    }

    #[test]
    fn cors_reflects_origin() {
        use hyper::Response;
        let mut resp: Response<()> = Response::new(());
        cors(Some("http://192.168.1.5:8081"), &mut resp);
        assert_eq!(
            resp.headers()["access-control-allow-origin"],
            "http://192.168.1.5:8081"
        );
        assert_eq!(resp.headers()["access-control-allow-methods"], "GET, POST, PUT, DELETE, OPTIONS");
        let mut resp: Response<()> = Response::new(());
        cors(None, &mut resp);
        assert!(!resp.headers().contains_key("access-control-allow-origin"));
        let mut resp: Response<()> = Response::new(());
        cors(Some(""), &mut resp);
        assert!(!resp.headers().contains_key("access-control-allow-origin"));
    }

    #[test]
    fn presented_prefers_header_falls_back_to_query() {
        use hyper::Request;
        let mut req = Request::new(());
        req.headers_mut().insert(
            "authorization",
            hyper::header::HeaderValue::from_static("Bearer deadbeef"),
        );
        assert_eq!(presented(&req), "deadbeef");
        let req: Request<()> = Request::builder().uri("/v1/events?token=cafe").body(()).unwrap();
        assert_eq!(presented(&req), "cafe");
        let req: Request<()> = Request::builder().uri("/v1/events").body(()).unwrap();
        assert_eq!(presented(&req), "");
    }
}
