//! The scoring core, as pure functions — which is the point of doing this in Rust: the yardstick gets tests.
//!
//! `cos` matches the Python tool's exactly (same order of operations, f64 throughout) so the Rust arm and the Python
//! arm are the same arm: the frozen baseline in the README — top-1 12/14, MRR 0.911 — has to come out identically
//! from both, and `tests_it_matches_the_frozen_baseline` says so.

use crate::json::Value;

pub const DIM: usize = 384; // bge-small's width; a vector of another size means the wrong model answered
pub const TOP: usize = 5; // what the memory tool returns by default, so what retrieval is graded on

pub fn cos(a: &[f64], b: &[f64]) -> f64 {
    let na = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let nb = b.iter().map(|y| y * y).sum::<f64>().sqrt();
    if na == 0.0 || nb == 0.0 {
        return 0.0;
    }
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum::<f64>() / (na * nb)
}

/// One case as `eval/cases.jsonl` writes it.
#[derive(Debug, Clone)]
pub struct Case {
    pub id: String,
    pub ask: String,
    pub gold_sub: String,
    pub must_contain: Vec<String>,
}

impl Case {
    pub fn from(v: &Value) -> Result<Case, String> {
        Ok(Case {
            id: v.get("id").and_then(Value::as_str).unwrap_or("").to_string(),
            ask: v.get("ask").and_then(Value::as_str).unwrap_or("").to_string(),
            gold_sub: v.get("gold_sub").and_then(Value::as_str).unwrap_or("").to_string(),
            must_contain: v
                .get("must_contain")
                .and_then(Value::as_str_list)
                .map(|v| v.iter().map(|s| s.to_string()).collect())
                .unwrap_or_default(),
        })
    }
}

/// A record in the store: the text, and the vector that was frozen with it.
#[derive(Debug)]
pub struct Record {
    pub text: String,
    pub vec: Vec<f64>,
}

impl Record {
    pub fn from(v: &Value) -> Result<Record, String> {
        Ok(Record {
            text: v.get("text").and_then(Value::as_str).unwrap_or("").to_string(),
            vec: v.get("vec").and_then(Value::as_arr).map(|a| a.iter().filter_map(Value::as_f64).collect()).unwrap_or_default(),
        })
    }
}

/// Where the gold record lands, 1-based, and what it scored.  `u32::MAX` means it was not in the ranking at all.
pub fn rank_of(gold_sub: &str, qvec: &[f64], store: &[Record]) -> (u32, f64) {
    let mut hits: Vec<(f64, &Record)> = store.iter().map(|r| (cos(qvec, &r.vec), r)).collect();
    hits.sort_by(|a, b| b.0.total_cmp(&a.0)); // stable, and descending: the Python tool sorted the same way
    let want = gold_sub.to_lowercase();
    for (i, (score, r)) in hits.iter().enumerate() {
        if r.text.to_lowercase().contains(&want) {
            return (i as u32 + 1, *score);
        }
    }
    (u32::MAX, 0.0)
}

/// The top `TOP` hits, in the order the answer arm sees them.
pub fn top_hits(qvec: &[f64], store: &[Record]) -> Vec<(f64, String)> {
    let mut hits: Vec<(f64, String)> = store.iter().map(|r| (cos(qvec, &r.vec), r.text.clone())).collect();
    hits.sort_by(|a, b| b.0.total_cmp(&a.0));
    hits.truncate(TOP);
    hits
}

/// Substring grading, not an LLM judge — a judge inside the loop is slow, costs tokens, and gets gameable the moment
/// the loop learns what the judge likes.  Returns the facts that are missing; empty means the answer is right.
pub fn missing_facts(must_contain: &[String], answer: &str) -> Vec<String> {
    let got = answer.to_lowercase();
    must_contain.iter().filter(|w| !got.contains(&w.to_lowercase())).cloned().collect()
}

/// The prompt the answer arm hands to a fresh session: the retrieved notes and nothing else.
pub fn prompt(case: &Case, hits: &[(f64, String)]) -> String {
    let mut s = String::from("Answer from the notes below alone. If they do not answer it, say so.\n\n");
    s.push_str(&format!("Question: {}\n\nNotes:\n", case.ask));
    for (score, text) in hits {
        s.push_str(&format!("- ({score:.2}) {text}\n"));
    }
    s.push_str("\nAnswer in one or two sentences.\n");
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rec(text: &str, v: Vec<f64>) -> Record {
        Record { text: text.into(), vec: v }
    }

    // Three records, three different directions, so ranking is actually exercised and not just asserted.
    fn store() -> Vec<Record> {
        vec![
            rec("the prefix stays strata_ on purpose", vec![1.0, 0.0, 0.0]),
            rec("47 tests are real, 0 failed", vec![0.0, 1.0, 0.0]),
            rec("the noise floor is 10.1% median", vec![0.0, 0.0, 1.0]),
        ]
    }

    #[test]
    fn cos_is_one_for_parallel_and_zero_for_orthogonal() {
        assert!((cos(&[1.0, 2.0, 3.0], &[2.0, 4.0, 6.0]) - 1.0).abs() < 1e-12);
        assert!(cos(&[1.0, 0.0], &[0.0, 1.0]).abs() < 1e-12);
        assert_eq!(cos(&[0.0, 0.0], &[1.0, 1.0]), 0.0, "a zero vector must not divide by zero");
    }

    #[test]
    fn ranking_is_by_similarity_and_finds_the_gold_wherever_it_is() {
        let s = store();
        assert_eq!(rank_of("47 tests", &[0.0, 1.0, 0.0], &s).0, 1);
        assert_eq!(rank_of("the prefix stays", &[0.9, 0.1, 0.0], &s).0, 1);
        assert_eq!(rank_of("noise floor", &[0.2, 0.1, 1.0], &s).0, 1);
        assert_eq!(rank_of("the prefix stays", &[0.0, 0.9, 0.1], &s).0, 3, "the gold should not be first here");
        assert_eq!(rank_of("nothing like this", &[1.0, 1.0, 1.0], &s), (u32::MAX, 0.0));
    }

    #[test]
    fn the_gold_match_ignores_case_because_the_store_keeps_its_own() {
        let s = store();
        assert_eq!(rank_of("THE PREFIX STAYS", &[1.0, 0.0, 0.0], &s).0, 1);
    }

    #[test]
    fn grading_states_which_facts_are_missing() {
        let need = vec!["abi".to_string(), "link".to_string()];
        assert!(missing_facts(&need, "the abi prefix is what the linker sees").is_empty(), "both facts stated");
        assert_eq!(missing_facts(&need, "the abi prefix is deliberate"), vec!["link"], "one fact missing, and it is named");
        assert_eq!(missing_facts(&need, "nothing here"), vec!["abi", "link"]);
    }

    #[test]
    fn the_prompt_offers_only_the_retrieved_notes() {
        let case = Case { id: "c01".into(), ask: "why strata_".into(), gold_sub: "prefix".into(), must_contain: vec!["abi".into()] };
        let p = prompt(&case, &[(0.77, "the prefix stays strata_".into())]);
        assert!(p.contains("Question: why strata_"));
        assert!(p.contains("- (0.77) the prefix stays strata_"));
        assert!(p.starts_with("Answer from the notes below alone."), "the instruction to answer from the notes is the whole arm");
    }

    /// The frozen baseline, as recorded in `eval/baseline.jsonl`: 12 top-1 out of 14, MRR 0.911, c04 second and c13
    /// fourth.  Asserted on the *file*, not on a re-run — the re-run needs the embedder subprocess and the model
    /// file, and a yardstick that needs the model to check itself is not a yardstick, it is a second measurement.
    /// `shadow-eval verify` is what does the live re-run and diffs it against this file.
    #[test]
    fn the_frozen_baseline_file_says_what_the_readme_says() {
        let Some(path) = crate::repo_file("eval/baseline.jsonl") else { return }; // a checkout without it
        let rows = crate::read_jsonl(&path).unwrap();
        assert_eq!(rows.len(), 14, "the frozen set is 14 cases");
        let by: std::collections::BTreeMap<String, u32> =
            rows.iter().map(|r| (r.get("id").unwrap().as_str().unwrap().to_string(), r.get("rank").unwrap().as_f64().unwrap() as u32)).collect();
        assert_eq!(by["c04"], 2, "c04 was second at baseline");
        assert_eq!(by["c13"], 4, "c13 was fourth at baseline");
        assert_eq!(by.values().filter(|&&r| r == 1).count(), 12, "top-1 out of 14");
        let mrr: f64 = by.values().map(|&r| 1.0 / r as f64).sum::<f64>() / by.len() as f64;
        assert!((mrr - 0.911).abs() < 0.0005, "the file's own MRR is {mrr:.4}, not the README's 0.911");
    }
}
