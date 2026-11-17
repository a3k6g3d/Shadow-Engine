//! M7 of docs/history/RUST_ENGINE.md: the prefetch window as a type, and "a prediction only ever warms" as a test.
//!
//! The C++ (`RouterLookahead`, `include/strata/core/expert_source.hpp`) predicts layers l+1..l+depth (depth 1..4,
//! STRATA_IO_PREFETCH_DEPTH) from layer l's input and asks the source to `warm` the experts no cache holds.  Its rule -
//! "a prediction only warms pages: it never changes which experts are computed or how" - is a comment.  Here:
//!
//!   * **The window is a value with a capacity.**  `Window::new(depth)` holds at most `depth` layers of predictions;
//!     a prediction for a layer the engine has reached, or one pushed out by a newer one, is dropped - and dropping a
//!     `Prediction` cancels its pending warms (its `Drop`), so nothing it asked for outlives it unaccounted.
//!   * **Predictions flow one way.**  The window reaches a source only through `Warm::warm`, which returns nothing;
//!     the compute path reads blobs through `Blobs::blob`, which never sees the window.  So a prediction can change
//!     *when* bytes arrive (warm or cold) and has no channel to change *what* is computed.
//!   * **The test that holds it to that** (below): the same computation with correct, wrong, thrashing and absent
//!     predictions gives byte-identical output, while only the cold-read count moves - and a deliberately broken warm
//!     path that serves the wrong bytes is caught, so the test cannot pass by never looking.

use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicI64, Ordering};
use std::sync::Arc;
use crate::Status;

/// Where predictions go.  Returns nothing on purpose: a prediction has no way to report back into the compute path.
pub trait Warm {
    fn warm(&mut self, layer: u32, experts: &[u32]);
    /// The warms for `layer` will not be needed (the prediction was dropped); a source may stop them.
    fn cancel(&mut self, layer: u32);
}

/// Where the computation reads expert bytes.  It never takes a `Window`.
pub trait Blobs {
    fn blob(&mut self, layer: u32, expert: u32) -> Vec<u8>;
}

/// One layer's predicted experts, while they are being warmed.  Counts itself in `live` so a test can see that every
/// prediction is dropped exactly once.
pub struct Prediction {
    pub layer: u32,
    pub experts: Vec<u32>,
    live: Arc<AtomicI64>,
}

impl Drop for Prediction {
    fn drop(&mut self) {
        self.live.fetch_sub(1, Ordering::Relaxed);
    }
}

/// The prefetch window: at most `depth` layers of predictions ahead of the layer the engine is on.
pub struct Window {
    depth: u32,
    at: Option<u32>,
    held: VecDeque<Prediction>,
    live: Arc<AtomicI64>,
}

impl Window {
    /// `depth` 1..=4, as the C++ clamps it; anything else is refused, not clamped, so a caller learns it asked for
    /// something the engine does not do.
    pub fn new(depth: u32) -> Result<Window, Status> {
        if !(1..=4).contains(&depth) {
            return Err(Status::BadSlot);
        }
        Ok(Window { depth, at: None, held: VecDeque::new(), live: Arc::new(AtomicI64::new(0)) })
    }

    pub fn depth(&self) -> u32 {
        self.depth
    }

    /// Predictions alive anywhere (held here or not yet dropped).  Zero once the window and its warms are gone.
    pub fn live(&self) -> i64 {
        self.live.load(Ordering::Relaxed)
    }

    pub fn held(&self) -> usize {
        self.held.len()
    }

    /// The engine has reached `layer`: every prediction for it or an earlier layer is done with - dropped, and its
    /// warms cancelled at the source.
    pub fn advance(&mut self, layer: u32, w: &mut impl Warm) {
        // The layer clock wraps: a layer below the current one is the next forward pass (the next token, or the next
        // window of tokens), and nothing predicted for the last pass's later layers applies to it.  Found by this
        // type's own capacity assertion - the first version kept them, and the new pass's predictions piled on top.
        if self.at.is_some_and(|at| layer < at) {
            while let Some(p) = self.held.pop_front() {
                w.cancel(p.layer);
            }
        }
        self.at = Some(layer);
        while self.held.front().is_some_and(|p| p.layer <= layer) {
            let p = self.held.pop_front().unwrap();
            w.cancel(p.layer);
        }
    }

    /// Predictions made at the current layer for the layers after it.  Only layers inside the window (at+1 ..=
    /// at+depth) are taken; a newer prediction for a layer replaces the older one (which is dropped and cancelled).
    /// Each kept prediction is handed to `w` to warm - and that is the only place a prediction goes.
    pub fn submit(&mut self, predicted: Vec<(u32, Vec<u32>)>, w: &mut impl Warm) {
        let at = self.at.unwrap_or(0);
        for (layer, experts) in predicted {
            if layer <= at || layer > at + self.depth {
                continue;
            }
            if let Some(i) = self.held.iter().position(|p| p.layer == layer) {
                let old = self.held.remove(i).unwrap();
                w.cancel(old.layer);
            }
            self.live.fetch_add(1, Ordering::Relaxed);
            let p = Prediction { layer, experts, live: self.live.clone() };
            w.warm(p.layer, &p.experts);
            let at_pos = self.held.iter().position(|q| q.layer > layer).unwrap_or(self.held.len());
            self.held.insert(at_pos, p);
        }
        debug_assert!(self.held.len() as u32 <= self.depth, "the window holds more layers than its depth");
    }
}

/// A file tier with a warm cache: the reference shape of what the window feeds.  `warm` copies an expert's bytes
/// into a bounded cache (evicting the oldest); `blob` serves from the cache when it can (a warm read) and from the
/// backing store when it cannot (a cold read).  Generic over where `warm` reads from, so a test can plant a broken
/// one.
pub struct WarmTier<F: Fn(u32, u32) -> Vec<u8>> {
    backing: F,
    warm_from: Box<dyn Fn(u32, u32) -> Vec<u8>>,
    cache: HashMap<(u32, u32), Vec<u8>>,
    order: VecDeque<(u32, u32)>,
    capacity: usize,
    pub cold: u64,
    pub warm_hits: u64,
    pub cancelled: u64,
}

impl<F: Fn(u32, u32) -> Vec<u8>> WarmTier<F> {
    pub fn new(backing: F, warm_from: Box<dyn Fn(u32, u32) -> Vec<u8>>, capacity: usize) -> Self {
        WarmTier { backing, warm_from, cache: HashMap::new(), order: VecDeque::new(), capacity, cold: 0, warm_hits: 0, cancelled: 0 }
    }
}

impl<F: Fn(u32, u32) -> Vec<u8>> Warm for WarmTier<F> {
    fn warm(&mut self, layer: u32, experts: &[u32]) {
        for &e in experts {
            if self.cache.contains_key(&(layer, e)) {
                continue;
            }
            if self.order.len() == self.capacity {
                if let Some(old) = self.order.pop_front() {
                    self.cache.remove(&old);
                }
            }
            self.cache.insert((layer, e), (self.warm_from)(layer, e));
            self.order.push_back((layer, e));
        }
    }

    fn cancel(&mut self, _layer: u32) {
        self.cancelled += 1;
    }
}

impl<F: Fn(u32, u32) -> Vec<u8>> Blobs for WarmTier<F> {
    fn blob(&mut self, layer: u32, expert: u32) -> Vec<u8> {
        match self.cache.get(&(layer, expert)) {
            Some(b) => {
                self.warm_hits += 1;
                b.clone()
            }
            None => {
                self.cold += 1;
                (self.backing)(layer, expert)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const LAYERS: u32 = 12;
    const EXPERTS: u32 = 64;
    const K: usize = 6;

    /// An expert's bytes: a function of (layer, expert) only, so the right bytes are known without a file.
    fn bytes(layer: u32, expert: u32) -> Vec<u8> {
        let mut x = (layer as u64) << 32 | expert as u64 | 0x9E37_79B9_0000_0000;
        (0..64).map(|_| { x = x.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407); (x >> 56) as u8 }).collect()
    }

    /// Which experts token `t` routes at `layer`: deterministic, so every run computes the same thing.
    fn route(t: u32, layer: u32) -> Vec<u32> {
        // never zero: xorshift from 0 stays 0, and the loop below would wait for distinct experts forever
        let mut x = (t as u64).wrapping_mul(0x2545F4914F6CDD1D)
            .wrapping_add((layer as u64).wrapping_mul(0x9E3779B97F4A7C15))
            .wrapping_add(0x853C_49E6_748F_EA9B);
        let mut out = Vec::new();
        while out.len() < K {
            x ^= x >> 12; x ^= x << 25; x ^= x >> 27;
            let e = (x.wrapping_mul(0x2545F4914F6CDD1D) >> 40) as u32 % EXPERTS;
            if !out.contains(&e) { out.push(e); }
        }
        out
    }

    /// The stand-in for a forward pass: every layer reads its routed experts' bytes through `Blobs` and folds them
    /// into a running state.  The window is consulted only for what to `warm`.
    fn forward(tokens: u32, depth: u32, tier: &mut (impl Blobs + Warm), predict: &dyn Fn(u32, u32) -> Vec<(u32, Vec<u32>)>) -> u64 {
        let mut h: u64 = 0xcbf29ce484222325;
        let mut win = Window::new(depth).unwrap();
        for t in 0..tokens {
            for l in 0..LAYERS {
                win.advance(l, tier);
                win.submit(predict(t, l), tier);
                for e in route(t, l) {
                    for b in tier.blob(l, e) { h = (h ^ b as u64).wrapping_mul(0x100000001b3); }
                }
            }
        }
        drop(win);
        h
    }

    fn tier(capacity: usize) -> WarmTier<fn(u32, u32) -> Vec<u8>> {
        WarmTier::new(bytes as fn(u32, u32) -> Vec<u8>, Box::new(bytes), capacity)
    }

    /// M7's invariant: what is computed does not depend on what was predicted.  Correct, wrong, thrashing and absent
    /// predictions, at every depth - one answer.  And the predictions are not idle: correct ones make reads warm.
    #[test]
    fn a_prediction_only_ever_warms() {
        let none = |_: u32, _: u32| Vec::new();
        let exact = |t: u32, l: u32| (1..=4).filter(|j| l + j < LAYERS).map(|j| (l + j, route(t, l + j))).collect::<Vec<_>>();
        let wrong = |t: u32, l: u32| (1..=4).filter(|j| l + j < LAYERS).map(|j| (l + j, route(t + 7777, l + j))).collect::<Vec<_>>();
        let thrash = |_: u32, l: u32| (1..=4).filter(|j| l + j < LAYERS).map(|j| (l + j, (0..EXPERTS).collect())).collect::<Vec<_>>();
        type Pred = dyn Fn(u32, u32) -> Vec<(u32, Vec<u32>)>;
        let preds: [(&str, &Pred); 4] =
            [("none", &none), ("exact", &exact), ("wrong", &wrong), ("thrash", &thrash)];

        let mut reference = None;
        for depth in 1..=4 {
            for (name, p) in preds.iter() {
                let mut t = tier(48);
                let h = forward(8, depth, &mut t, *p);
                let r = *reference.get_or_insert(h);
                assert_eq!(h, r, "depth {depth}, {name} predictions changed what was computed");
                if *name == "none" {
                    assert_eq!(t.warm_hits, 0, "no prediction, yet a warm read");
                }
                if *name == "exact" {
                    assert!(t.warm_hits > t.cold, "depth {depth}: correct predictions warmed almost nothing ({} warm, {} cold)",
                            t.warm_hits, t.cold);
                }
            }
        }
    }

    /// The test is not vacuous: a warm path that fetches the wrong expert's bytes (an off-by-one, the classic wrong
    /// offset) is caught - the computed answer moves.  Without this, the invariant test could pass by never warming.
    #[test]
    fn a_broken_warm_path_is_caught() {
        let exact = |t: u32, l: u32| vec![(l + 1, route(t, l + 1))];
        let right = forward(4, 1, &mut tier(48), &exact);
        let mut broken = WarmTier::new(bytes as fn(u32, u32) -> Vec<u8>, Box::new(|l, e| bytes(l, e + 1)), 48);
        let wrong = forward(4, 1, &mut broken, &exact);
        assert!(broken.warm_hits > 0, "the broken path was never read, so this proves nothing");
        assert_ne!(right, wrong, "a warm path serving the wrong bytes went unnoticed");
    }

    /// The window is a value with a capacity: it never holds more than `depth` layers, takes nothing outside them,
    /// and every prediction it made is dropped exactly once - by the advance that reached its layer, by a newer
    /// prediction for the same layer, or by the window's own drop.
    #[test]
    fn the_window_holds_depth_layers_and_drops_every_prediction() {
        let mut t = tier(8);
        let mut w = Window::new(2).unwrap();
        w.advance(0, &mut t);
        w.submit(vec![(1, vec![1]), (2, vec![2]), (3, vec![3]), (0, vec![9])], &mut t);
        assert_eq!(w.held(), 2, "layers 3 (beyond depth 2) and 0 (already reached) must not be held");
        w.submit(vec![(2, vec![5])], &mut t);
        assert_eq!((w.held(), w.live()), (2, 2), "a newer prediction for layer 2 replaces the old, not adds");
        assert_eq!(t.cancelled, 1, "the replaced prediction's warms were not cancelled");
        w.advance(1, &mut t);
        assert_eq!((w.held(), w.live()), (1, 1));
        let live = w.live.clone();
        drop(w);
        assert_eq!(live.load(Ordering::Relaxed), 0, "a prediction outlived its window");
    }

    /// The next forward pass starts at a lower layer: every prediction for the last pass is dropped and cancelled,
    /// so the window never carries one pass's guesses into the next (the first version did, and overfilled).
    #[test]
    fn the_next_forward_pass_drops_the_last_ones_predictions() {
        let mut t = tier(8);
        let mut w = Window::new(3).unwrap();
        w.advance(9, &mut t);
        w.submit(vec![(10, vec![1]), (11, vec![2])], &mut t);
        assert_eq!(w.held(), 2);
        w.advance(0, &mut t);
        assert_eq!((w.held(), w.live(), t.cancelled), (0, 0, 2), "the last pass's predictions survived the wrap");
        w.submit(vec![(1, vec![3]), (2, vec![4]), (3, vec![5])], &mut t);
        assert_eq!(w.held(), 3);
    }

    #[test]
    fn a_depth_the_engine_does_not_do_is_refused() {
        assert_eq!(Window::new(0).err(), Some(Status::BadSlot));
        assert_eq!(Window::new(5).err(), Some(Status::BadSlot));
        assert!(Window::new(4).is_ok());
    }
}
