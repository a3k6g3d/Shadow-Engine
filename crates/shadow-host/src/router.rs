//! The router's bookkeeping: the counters, the hysteresis, the eviction - `docs/SHADOW_PLAN.md` Phase 1 item 3.
//! The score itself is not here, because the plan says it is a kernel, and it is right: the scores arrive as data,
//! from CUDA.  What is here is the half that is pure, and therefore the half that can be tested at all.
//!
//! Measured, with its source:
//!
//!   * **512 routed experts over 48 blocks** (the plan's table, measured off the file) - so 24,576 routed experts.
//!   * **8 active per token** (the plan's line on speculative decoding).  That line says "of 256", and 256 is the
//!     stale table's number - the model has 512 - but the 8 is the top-k, and the 8 is the part this needs.
//!   * **`expert_slots 1951`**, the engine's own probe (`docs/BENCH_NOISE.md`).  1951 of 24,576 is **7.9%**: the
//!     cache holds under a thirteenth of the experts, and the measured hit rate is ~50%.  Those two facts together
//!     are the reason this module exists - and the test that separates them is worth more than any test here.
//!   * **The ladders the machine measured**: swaps 0 -> 24.9%, 4 -> 40.6%, 16 -> 56.9%; decay 0.65 -> 47.6%,
//!     0.80 -> 47.1%.  The doc says the mechanism is live and the magnitudes are confounded by the cold region, so
//!     below the *direction* of each ladder is tested and its numbers are not.
//!
//! Assumed, and labelled everywhere it bites:
//!
//!   * decay is **per token**, not per layer-call - a token touches 48 layers, so per-call would make it 48x
//!     stronger, which is a large enough difference that it had better be the stated reading and not a habit;
//!   * the swap budget is **per token**, spent across all 48 layers of it;
//!   * the eviction pool is **the whole resident set**, since the slots are one global pool rather than a slice per
//!     layer - if the engine's 1951 are per-layer the answer changes and this file is wrong about it;
//!   * a selected expert's counter is bumped **before** the admission decision its own bump caused.
//!
//! What this does not claim: that it reproduces the machine's numbers.  It claims to reproduce their direction, on
//! a trace, with the assumptions above visible.  The numbers belong to the machine, and the machine's own doc says
//! they cannot be ranked against each other.

use std::collections::{HashMap, HashSet};

/// A routed expert, by (layer, index in that layer).  The key the counters and the resident set are kept under.
pub type Key = (u32, u32);

/// One expert's score, as the kernel handed it: (expert, score).  The router does not compute this and does not
/// know what it means, only that bigger wins.
pub type Score = (u32, f32);

/// The knobs.  The four geometry ones are measured; the three policy ones are at their assumption-free point in
/// `Default` - 1.0 meaning "this knob does nothing" - because the values the machine was benched at (0.65/0.80,
/// 0/4/16) are what it was *tested at*, and the config it actually runs on is not in the record I have.
#[derive(Debug, Clone, PartialEq)]
pub struct Config {
    pub experts_per_layer: usize,
    pub layers: u32,
    pub top_k: usize,
    /// The cache's capacity, in experts - the probe's `expert_slots`.
    pub slots: usize,
    /// Multiplier applied to every counter once per token.  1.0 is no decay: counters only ever grow.
    pub decay: f64,
    /// Evictions allowed per token.  0 freezes the resident set once it is full, which is the machine's own arm.
    pub swap_budget: usize,
    /// How much a newcomer must beat an incumbent's counter by to move it.  1.0 is no hysteresis at all.
    pub hysteresis: f64,
}

impl Default for Config {
    fn default() -> Self {
        Config {
            experts_per_layer: 512, // measured off the file
            layers: 48,             // measured off the file
            top_k: 8,               // the plan's line, above
            slots: 1951,            // the engine's probe
            decay: 1.0,             // assumed neutral
            swap_budget: 0,         // assumed neutral: fill the cache, never evict
            hysteresis: 1.0,        // assumed neutral
        }
    }
}

/// What one (token, layer) call decided.  `selected` is the top-k, in the order the ranking gives - which is the
/// answer the caller acts on; the rest is what it cost.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Decision {
    pub selected: Vec<u32>,
    pub hits: usize,
    pub misses: usize,
    pub swaps: usize,
}

/// The counters, the resident set, and the two knobs that decide when one is traded for the other.
#[derive(Debug, Clone, PartialEq)]
pub struct Router {
    cfg: Config,
    counters: HashMap<Key, f64>,
    resident: HashSet<Key>,
    free: usize,
    swaps_this_token: usize,
    /// The running tally, over every token so far - the number the machine reports as "hit rate".
    hits: usize,
    misses: usize,
    swaps: usize,
}

impl Router {
    pub fn new(cfg: Config) -> Self {
        let free = cfg.slots;
        Router {
            cfg,
            counters: HashMap::new(),
            resident: HashSet::new(),
            free,
            swaps_this_token: 0,
            hits: 0,
            misses: 0,
            swaps: 0,
        }
    }

    pub fn config(&self) -> &Config {
        &self.cfg
    }

    /// The token boundary: decay every counter, and hand back this token's swap budget.  Called once per token,
    /// before the 48 layer calls - which is the assumption the header names.  A token that selected nothing still
    /// decays, because that is what "per token" means.
    pub fn start_token(&mut self) {
        if self.cfg.decay != 1.0 {
            for c in self.counters.values_mut() {
                *c *= self.cfg.decay;
            }
        }
        self.swaps_this_token = 0;
    }

    /// One layer of one token.  `scores` is the kernel's answer for this layer; the router ranks it, updates the
    /// counters of everything it picked, and decides for each pick whether the bytes are already where they need
    /// to be.  It is deterministic by construction - every ordering here is explicit, so the answer cannot depend
    /// on which order the scores arrived in, which is what "a policy you can change at runtime" has to mean.
    pub fn route(&mut self, layer: u32, scores: &[Score]) -> Decision {
        let mut order: Vec<(u32, f32)> = scores.to_vec();
        order.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal).then(a.0.cmp(&b.0)));
        order.truncate(self.cfg.top_k);

        // Counters first, for every pick, before any of them is admitted - the order is the assumption named above,
        // and a test pins it because it changes which expert gets evicted.
        for (expert, _) in &order {
            let key = (layer, *expert);
            let c = self.counters.entry(key).or_insert(0.0);
            *c = *c * self.cfg.decay + 1.0;
        }

        let mut d = Decision::default();
        for (expert, _) in &order {
            d.selected.push(*expert);
            let key = (layer, *expert);
            if self.resident.contains(&key) {
                d.hits += 1;
                continue;
            }
            if self.free > 0 {
                // Filling a free slot is not a swap, and costs no budget: nothing was moved to make room.
                self.free -= 1;
                self.resident.insert(key);
                d.misses += 1;
                continue;
            }
            match self.evict_candidate(&key) {
                Some(worst) if self.swaps_this_token < self.cfg.swap_budget => {
                    self.resident.remove(&worst);
                    self.resident.insert(key);
                    self.swaps_this_token += 1;
                    self.swaps += 1;
                    d.swaps += 1;
                    d.misses += 1;
                }
                // No victim that qualifies, or no budget left: the bytes are streamed.  A miss that costs nothing
                // is still a miss, and it is still counted, because the machine counts it.
                _ => d.misses += 1,
            }
        }
        self.hits += d.hits;
        self.misses += d.misses;
        d
    }

    /// The resident expert that would be traded away for `incoming`: lowest counter, ties by key so the answer is
    /// the same every time.  The scan is over the whole resident set - O(slots) per miss, which at the machine's
    /// own numbers is ~1951 comparisons x ~half the picks x 8 picks x 48 layers: ~380k comparisons a token at
    /// ~25 tok/s.  A CPU core can do that, but it is not free, and it is worth saying before anyone benchmarks
    /// this against the C++ version's table.  `None` when the newcomer does not beat the incumbent by
    /// `hysteresis` - which is the whole of what hysteresis is for, and why it is a factor on the incumbent's
    /// counter rather than a threshold.
    fn evict_candidate(&self, incoming: &Key) -> Option<Key> {
        let incoming_score = self.counter(incoming);
        let worst = self
            .resident
            .iter()
            .filter(|k| **k != *incoming)
            .min_by(|a, b| {
                self.counter(a).partial_cmp(&self.counter(b)).unwrap_or(std::cmp::Ordering::Equal).then(a.cmp(b))
            })
            .copied()?;
        // The incumbent is damped, the newcomer is not: the standard shape, and the same one placement.rs uses.
        (incoming_score > self.counter(&worst) * self.cfg.hysteresis).then_some(worst)
    }

    /// A counter, as it stands right now.  An expert that has never been picked has none, and 0.0 is what it is
    /// worth - which is why a cold expert can be evicted by a cold newcomer the moment hysteresis is 1.0.
    pub fn counter(&self, key: &Key) -> f64 {
        self.counters.get(key).copied().unwrap_or(0.0)
    }

    pub fn is_resident(&self, key: &Key) -> bool {
        self.resident.contains(key)
    }

    pub fn resident_len(&self) -> usize {
        self.resident.len()
    }

    pub fn free_slots(&self) -> usize {
        self.free
    }

    /// The running hit rate, the number the machine's own probes report - hits over hits plus misses, over every
    /// token since the router was made.  It counts the fill, because the machine's does too: the first ~244 tokens
    /// of a 1951-slot cache at 8 picks a token are all misses, and that is part of the number, not a rounding.
    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            0.0
        } else {
            self.hits as f64 / total as f64
        }
    }

    pub fn tally(&self) -> (usize, usize, usize) {
        (self.hits, self.misses, self.swaps)
    }
}

/// The capacity the machine's probe implies, as arithmetic the tests below quote rather than restate: 1951 slots
/// over 24,576 routed experts.
pub fn capacity_fraction(cfg: &Config) -> f64 {
    cfg.slots as f64 / (cfg.experts_per_layer as f64 * cfg.layers as f64)
}
