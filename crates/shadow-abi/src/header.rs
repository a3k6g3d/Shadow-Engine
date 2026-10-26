//! Reading include/shadow/c/abi.h as text, for the drift tests on both sides of the seam.
//!
//! It parses text rather than including the header, so the tests need no C toolchain.  It lives in the crate
//! rather than in one test file because two crates' tests use it - shadow-abi's for the calls the facade exports,
//! shadow-host's for the calls Rust exports - and a guard copied into two places is two guards that can drift.

use std::path::PathBuf;

/// The header's text.  `manifest_dir` is the calling crate's `CARGO_MANIFEST_DIR` (crates/<crate>).
pub fn read(manifest_dir: &str) -> String {
    let mut p = PathBuf::from(manifest_dir);
    p.pop(); p.pop(); // crates/<crate> -> crates -> repo root
    p.push("include"); p.push("shadow"); p.push("c"); p.push("abi.h");
    std::fs::read_to_string(&p).unwrap_or_else(|e| panic!("cannot read {}: {e}", p.display()))
}

/// The header with every `/* ... */` comment removed, so a prototype quoted in prose is never read as one.
fn strip_comments(src: &str) -> String {
    let mut out = String::with_capacity(src.len());
    let mut rest = src;
    while let Some(i) = rest.find("/*") {
        out.push_str(&rest[..i]);
        match rest[i..].find("*/") {
            Some(j) => rest = &rest[i + j + 2..],
            None => return out,
        }
    }
    out.push_str(rest);
    out
}

fn squash(s: &str) -> String {
    s.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// One declared function, normalised: return type, name, and each parameter as written, whitespace collapsed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Prototype {
    pub ret: String,
    pub name: String,
    pub params: Vec<String>,
}

impl Prototype {
    /// The declaration as one line, e.g. `int strata_host_file_wake(strata_host_file *file)`.
    pub fn line(&self) -> String {
        let sep = if self.ret.ends_with('*') { "" } else { " " };
        format!("{}{sep}{}({})", self.ret, self.name, self.params.join(", "))
    }
}

/// Every function declared with `macro_name` in front of it (`STRATA_API`, `STRATA_HOST_API`), in header order.
/// A declaration may span lines; it ends at its `;`.
pub fn prototypes(src: &str, macro_name: &str) -> Vec<Prototype> {
    let src = strip_comments(src);
    let lead = format!("{macro_name} ");
    let mut out = Vec::new();
    let mut rest = src.as_str();
    while let Some(i) = rest.find(&lead) {
        // Only a declaration at the start of a line counts: `#define STRATA_API ...` does not.
        let line_start = rest[..i].rfind('\n').map(|n| n + 1).unwrap_or(0);
        let before = rest[line_start..i].trim();
        let after = &rest[i + lead.len()..];
        if !before.is_empty() {
            rest = after;
            continue;
        }
        let end = after.find(';').unwrap_or_else(|| panic!("{macro_name} declaration with no ';'"));
        let decl = squash(&after[..end]);
        rest = &after[end + 1..];

        let open = decl.find('(').unwrap_or_else(|| panic!("not a function: {decl:?}"));
        let close = decl.rfind(')').unwrap_or_else(|| panic!("not a function: {decl:?}"));
        let head = decl[..open].trim();
        let (mut ret, mut name) = match head.rsplit_once(' ') {
            Some((r, n)) => (r.to_string(), n.to_string()),
            None => panic!("no return type: {decl:?}"),
        };
        while let Some(n) = name.strip_prefix('*') {
            ret.push_str(" *");
            name = n.to_string();
        }
        let ret = ret.replace(" * *", " **");
        let params: Vec<String> = decl[open + 1..close]
            .split(',')
            .map(squash)
            .filter(|p| !p.is_empty() && p != "void")
            .collect();
        out.push(Prototype { ret, name, params });
    }
    out
}

/// The fields of `typedef struct <name> { ... }`, as (type, name) pairs, in header order.
pub fn struct_fields(src: &str, name: &str) -> Vec<(String, String)> {
    let src = strip_comments(src);
    let mut out = Vec::new();
    let mut inside = false;
    for line in src.lines() {
        let l = line.trim();
        if !inside {
            if l.starts_with("typedef struct") && l.contains(name) && l.ends_with('{') {
                inside = true;
            }
            continue;
        }
        if l.starts_with('}') {
            break;
        }
        let body = l.trim_end_matches(';').trim();
        if body.is_empty() {
            continue;
        }
        // "const unsigned char *bytes;" -> type "const unsigned char *", name "bytes"
        let (ty, field) = match body.rsplit_once(char::is_whitespace) {
            Some(pair) => pair,
            None => panic!("{name}: field line {l:?} has no name"),
        };
        let ty = if field.starts_with('*') { format!("{ty} *") } else { ty.to_string() };
        out.push((ty, field.trim_start_matches('*').to_string()));
    }
    assert!(!out.is_empty(), "struct {name} not found in the header, or has no fields");
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_prototype_in_a_comment_or_a_define_is_not_one() {
        let src = "#define STRATA_API __declspec(dllexport)\n/* STRATA_API int ghost(void); */\nSTRATA_API int real(void);\n";
        let p = prototypes(src, "STRATA_API");
        assert_eq!(p.len(), 1);
        assert_eq!(p[0].line(), "int real()");
    }

    #[test]
    fn multi_line_declarations_and_pointer_returns_normalise() {
        let src = "STRATA_API const char *last(void);\nSTRATA_API int f(int a,\n                  float *b);\n";
        let p = prototypes(src, "STRATA_API");
        assert_eq!(p[0].line(), "const char *last()");
        assert_eq!(p[1].params, vec!["int a".to_string(), "float *b".to_string()]);
    }
}
