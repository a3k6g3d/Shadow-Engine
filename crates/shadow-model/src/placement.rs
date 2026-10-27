//! Routing counts: how often each (layer, expert) was routed to, recorded from Shadow's own runs, and the placement
//! that follows from them (docs/SHADOW_PLAN.md, stage 3: "a placement profile in our own format, learned from Shadow's
//! own routing traces").  Nothing here reads another engine's profile.
//!
//! The file, little-endian throughout:
//!
//! ```text
//! b"SHRC"            magic
//! u32 version        1
//! u32 n_layer
//! u32 n_expert
//! u64 tokens         steps recorded (each step routes every layer once)
//! u64 count[n_layer][n_expert]
//! ```

use std::path::Path;

const MAGIC: &[u8; 4] = b"SHRC";
const VERSION: u32 = 1;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RoutingCounts {
    pub n_layer: usize,
    pub n_expert: usize,
    pub tokens: u64,
    /// `count[l * n_expert + e]`
    pub count: Vec<u64>,
}

impl RoutingCounts {
    pub fn new(n_layer: usize, n_expert: usize) -> RoutingCounts {
        RoutingCounts { n_layer, n_expert, tokens: 0, count: vec![0; n_layer * n_expert] }
    }

    pub fn add(&mut self, l: usize, e: usize) {
        self.count[l * self.n_expert + e] += 1;
    }

    /// Adds another recording of the same model shape.
    pub fn merge(&mut self, o: &RoutingCounts) -> Result<(), String> {
        if (o.n_layer, o.n_expert) != (self.n_layer, self.n_expert) {
            return Err(format!("routing counts for {}x{} cannot merge into {}x{}", o.n_layer, o.n_expert,
                               self.n_layer, self.n_expert));
        }
        self.tokens += o.tokens;
        self.count.iter_mut().zip(&o.count).for_each(|(a, b)| *a += b);
        Ok(())
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut b = Vec::with_capacity(24 + 8 * self.count.len());
        b.extend_from_slice(MAGIC);
        b.extend_from_slice(&VERSION.to_le_bytes());
        b.extend_from_slice(&(self.n_layer as u32).to_le_bytes());
        b.extend_from_slice(&(self.n_expert as u32).to_le_bytes());
        b.extend_from_slice(&self.tokens.to_le_bytes());
        for c in &self.count {
            b.extend_from_slice(&c.to_le_bytes());
        }
        b
    }

    pub fn from_bytes(b: &[u8]) -> Result<RoutingCounts, String> {
        if b.len() < 24 || &b[..4] != MAGIC {
            return Err("not a Shadow routing-counts file (SHRC)".into());
        }
        let u32_at = |i: usize| u32::from_le_bytes(b[i..i + 4].try_into().unwrap());
        let version = u32_at(4);
        if version != VERSION {
            return Err(format!("routing counts version {version}, this build reads {VERSION}"));
        }
        let (n_layer, n_expert) = (u32_at(8) as usize, u32_at(12) as usize);
        let tokens = u64::from_le_bytes(b[16..24].try_into().unwrap());
        let n = n_layer * n_expert;
        if b.len() != 24 + 8 * n {
            return Err(format!("routing counts: {} bytes, expected {}", b.len(), 24 + 8 * n));
        }
        let count = b[24..].chunks(8).map(|c| u64::from_le_bytes(c.try_into().unwrap())).collect();
        Ok(RoutingCounts { n_layer, n_expert, tokens, count })
    }

    pub fn load(path: &Path) -> Result<RoutingCounts, String> {
        Self::from_bytes(&std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?)
    }

    pub fn save(&self, path: &Path) -> Result<(), String> {
        std::fs::write(path, self.to_bytes()).map_err(|e| format!("{}: {e}", path.display()))
    }

    /// Every (layer, expert), most routed first; ties (and every expert of a file with no counts) in layer-major,
    /// expert-minor order, so the ranking is deterministic.
    pub fn ranked(&self) -> Vec<(usize, usize)> {
        let mut v: Vec<(usize, usize)> = (0..self.n_layer).flat_map(|l| (0..self.n_expert).map(move |e| (l, e))).collect();
        v.sort_by_key(|&(l, e)| (std::cmp::Reverse(self.count[l * self.n_expert + e]), l, e));
        v
    }
}

/// With no recording: every layer gets the same share, expert 0 of every layer first, then expert 1, and so on.  An
/// arbitrary choice (no expert is known to be hotter than another), deterministic so runs are comparable.
pub fn uniform_order(n_layer: usize, n_expert: usize) -> Vec<(usize, usize)> {
    (0..n_expert).flat_map(|e| (0..n_layer).map(move |l| (l, e))).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trips_and_merges() {
        let mut a = RoutingCounts::new(3, 4);
        a.tokens = 2;
        a.add(1, 2);
        a.add(1, 2);
        a.add(0, 3);
        let b = RoutingCounts::from_bytes(&a.to_bytes()).unwrap();
        assert_eq!(a, b);
        let mut c = a.clone();
        c.merge(&b).unwrap();
        assert_eq!(c.tokens, 4);
        assert_eq!(c.count[6], 4);
        assert!(c.merge(&RoutingCounts::new(2, 4)).is_err());
    }

    #[test]
    fn rejects_bad_files() {
        assert!(RoutingCounts::from_bytes(b"GGUF\0\0\0\0").is_err());
        let mut b = RoutingCounts::new(2, 2).to_bytes();
        b.pop();
        assert!(RoutingCounts::from_bytes(&b).is_err());
    }

    #[test]
    fn ranks_hottest_first_then_by_position() {
        let mut a = RoutingCounts::new(2, 3);
        a.add(1, 1);
        a.add(1, 1);
        a.add(0, 2);
        assert_eq!(a.ranked(), vec![(1, 1), (0, 2), (0, 0), (0, 1), (1, 0), (1, 2)]);
        assert_eq!(uniform_order(2, 2), vec![(0, 0), (1, 0), (0, 1), (1, 1)]);
    }
}
