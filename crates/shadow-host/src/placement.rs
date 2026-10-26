//! The placement policy: which parts of the model get RAM, and which are streamed.
//! See docs/SHADOW_PLAN.md, Phase 1 item 2.
//!
//! The plan's argument for putting this in Rust is one line long and it is the whole reason this file exists:
//! *"a policy is pure and testable; a policy in C++ is a policy you can't change at runtime."*  So this file is
//! pure in the literal sense - no I/O, no environment, no globals, no clock, no allocation of anything but a
//! `Vec` - and it takes the numbers it needs as arguments, so a test can hand it the numbers it wants.
//!
//! **What is measured and what is assumed, because both are in here.**  `Shape::per_gib` is measured: it is the
//! slope of the plan's own budget table (36/40/44/45.5 GiB -> 10.7/13.2/15.5/17.1 tok/s), which over that span
//! is 6.4 tok/s across 9.5 GiB, i.e. **~0.67 tok/s per GiB**, the plan's "~0.7 per GiB".  The *shape* - the
//! relative weight of one kind of region against another - is **assumed**, and `Shape::default` assumes as little
//! as it can: every kind weighs the same, which is what the one measured point actually supports.  The shape
//! knobs are where the intuition ("a recurrent layer's bytes are worth more than an attention layer's, because
//! its state is a fixed size regardless of context") enters, and a test says plainly what turning one does and
//! what it does not.
//!
//! The geometry is the plan's, measured off the file itself - 48 blocks, 36 recurrent + 12 full-attention, every
//! 4th a K; per layer 1.032 GiB (J) and 1.086 GiB (K); embeddings 26.82 GiB.  The two `Split::other` rows are
//! *derived* (measured layer total minus the three named parts) and are labelled as such, because the plan names
//! three parts and its per-layer totals do not close on them.

use std::collections::HashSet;

/// The unit the plan quotes its numbers in.
pub const GIB: f64 = 1024.0 * 1024.0 * 1024.0;

/// The four things the plan says a placement decision is made of: the two kinds of layer, the embedding, and
/// whatever the plan does not attribute to a layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Kind {
    /// A `qwen4exp` recurrent block (a "J"): the fixed-size state the plan calls the reason this model is fast.
    Recurrent,
    /// A full-attention block (a "K"): 12 of the 48, and at long context the thing that stops fitting.
    Attention,
    /// `per_layer_token_embd`: 26.82 GiB, 35% of the model, and not in any layer.
    Embedding,
}

/// One thing that could be resident: a layer, or the embedding.  `gib` is its whole self - this policy places a
/// candidate whole or not at all, because a layer whose expert weights are half resident still gets read whole on
/// the tokens that route to the other half.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Candidate {
    pub id: Id,
    pub kind: Kind,
    pub gib: f64,
    /// The named parts, as the plan measured them, so a test can check the sums close.
    pub split: Split,
}

/// The plan's per-layer split: routed experts, attention, shared expert - and `other`, which is derived, not
/// measured (see the module header).
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Split {
    pub experts_gib: f64,
    pub attention_gib: f64,
    pub shared_expert_gib: f64,
    pub other_gib: f64,
}

impl Split {
    pub fn total_gib(&self) -> f64 {
        self.experts_gib + self.attention_gib + self.shared_expert_gib + self.other_gib
    }
}

/// Which candidate this is.  Layer index for a layer, `NO_LAYER` for the embedding.
pub const NO_LAYER: u32 = u32::MAX;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Id(pub u32, pub Kind);

/// The ranking.  `per_gib` is the measured slope; the weights are the assumed shape, and they only ever
/// *redistribute* it - a candidate's score is `per_gib * weight(kind)`, so with every weight at 1.0 the ranking
/// is exactly the measurement and nothing has been assumed.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Shape {
    pub per_gib: f64,
    pub weight_recurrent: f64,
    pub weight_attention: f64,
    pub weight_embedding: f64,
}

impl Default for Shape {
    fn default() -> Self {
        Shape {
            per_gib: 0.67, // measured: the plan's budget table, above
            weight_recurrent: 1.0,
            weight_attention: 1.0,
            weight_embedding: 1.0,
        }
    }
}

impl Shape {
    fn weight(&self, kind: Kind) -> f64 {
        match kind {
            Kind::Recurrent => self.weight_recurrent,
            Kind::Attention => self.weight_attention,
            Kind::Embedding => self.weight_embedding,
        }
    }
}

/// The policy, as one call's worth of decisions.  Everything in it is data, so the same policy can be handed to
/// two calls and give the same answer, which is the "you can change it at runtime" the plan wants.
#[derive(Debug, Clone, PartialEq)]
pub struct Policy {
    pub budget_gib: f64,
    pub shape: Shape,
    /// Candidates that must be placed if they fit at all.  A pin that does not fit is *reported*, not dropped.
    pub pins: Vec<Id>,
    /// What the last decision left resident.  An incumbent is only moved by a newcomer that beats it by
    /// `hysteresis`, which is the plan's Phase 1 item 3 rule, stated where it can be tested.
    pub incumbent: HashSet<Id>,
    /// The factor an incumbent must be beaten by to be evicted: 1.0 is no hysteresis at all.  Damping is the
    /// standard shape of it - the incumbent's score is multiplied by the factor - and it is here because a policy
    /// that oscillates is a policy that reads the drive twice.
    pub hysteresis: f64,
}

impl Default for Policy {
    fn default() -> Self {
        Policy {
            budget_gib: 36.0, // the config this session runs on, per the plan
            shape: Shape::default(),
            pins: Vec::new(),
            incumbent: HashSet::new(),
            hysteresis: 1.0,
        }
    }
}

/// What a decision looked like: the set, the arithmetic, and the things worth saying out loud.
#[derive(Debug, Clone, PartialEq)]
pub struct Plan {
    pub resident: Vec<Candidate>,
    pub streamed: Vec<Candidate>,
    pub used_gib: f64,
    pub spare_gib: f64,
    pub notes: Vec<Note>,
}

/// The things that are true of a plan and that the caller should have to look at.
#[derive(Debug, Clone, PartialEq)]
pub enum Note {
    /// The budget's last fraction is smaller than the smallest candidate left, so it buys nothing.  Real at the
    /// plan's own arm A: 36 GiB of budget and 0.91 GiB of spare, with nothing that small to spend it on.
    SpareBuysNothing { spare_gib: f64, smallest_gib: f64 },
    /// A pin that did not fit.  Reported, because dropping it quietly is how a policy starts lying.
    PinDoesNotFit { id: Id, gib: f64 },
    /// The budget is smaller than the smallest candidate, so the answer is "stream everything".
    NothingFits,
}

/// The tolerance the sums are checked at, and the reason it is not zero: the plan's figures are quoted to three
/// decimals, so 36 x 1.032 is 37.152 where the plan writes 37.14.  A test that asserted equality would be
/// asserting a rounding convention, so it asserts the closeness and says so.
pub const TOLERANCE_GIB: f64 = 0.02;

/// The model, as the plan's table measured it: 36 J + 12 K + the embedding.
pub fn swift15() -> Vec<Candidate> {
    let j_split = Split {
        experts_gib: 0.961,
        attention_gib: 0.041,
        shared_expert_gib: 0.020,
        other_gib: 0.010, // derived: 1.032 - the three above
    };
    let k_split = Split {
        experts_gib: 1.020,
        attention_gib: 0.041,
        shared_expert_gib: 0.020,
        other_gib: 0.005, // derived: 1.086 - the three above
    };
    let mut out = Vec::new();
    // `full_attention_interval 4`: every 4th of the 48 blocks is a K, so 12 K and 36 J - the plan's own split.
    for layer in 0..48u32 {
        let k = (layer + 1) % 4 == 0;
        let (kind, split) = if k { (Kind::Attention, k_split) } else { (Kind::Recurrent, j_split) };
        out.push(Candidate { id: Id(layer, kind), kind, gib: split.total_gib(), split });
    }
    let emb = Split { experts_gib: 26.82, ..Default::default() };
    out.push(Candidate { id: Id(NO_LAYER, Kind::Embedding), kind: Kind::Embedding, gib: emb.total_gib(), split: emb });
    out
}

impl Policy {
    /// Rank the candidates and fill the budget, best score per GiB first, whole candidates only.
    ///
    /// Pins go first, in the order given (they are the caller's, so the caller's order means something), and a pin
    /// that cannot fit is reported rather than skipped.  After that it is a plain greedy fill: the ordering is by
    /// score-per-GiB descending, and ties - which the default shape produces between every pair of same-kind
    /// candidates, since its score does not depend on size - are broken by id, so the answer is deterministic
    /// without depending on the input order.
    pub fn place(&self, candidates: &[Candidate]) -> Plan {
        let score = |c: &Candidate| -> f64 {
            let w = self.shape.weight(c.kind);
            let s = self.shape.per_gib * w;
            // The incumbent's damping, applied to its score: the standard shape, and the only non-measured thing
            // in the ranking besides the weights themselves.
            if self.incumbent.contains(&c.id) { s * self.hysteresis } else { s }
        };
        let mut order: Vec<&Candidate> = candidates.iter().collect();
        order.sort_by(|a, b| {
            score(b).partial_cmp(&score(a)).unwrap().then(a.id.cmp(&b.id))
        });

        let mut pins: Vec<&Candidate> = Vec::new();
        let mut notes = Vec::new();
        let mut chosen: Vec<&Candidate> = Vec::new();
        let mut used = 0.0f64;

        for id in &self.pins {
            match order.iter().find(|c| &c.id == id) {
                Some(c) if fits(used, c.gib, self.budget_gib) => {
                    chosen.push(c);
                    used += c.gib;
                    pins.push(c);
                }
                Some(c) => notes.push(Note::PinDoesNotFit { id: c.id, gib: c.gib }),
                // A pin for something that is not on the table is the caller's bug, and it is worth a note too.
                None => notes.push(Note::PinDoesNotFit { id: *id, gib: 0.0 }),
            }
        }
        for c in order.iter() {
            if pins.contains(c) {
                continue;
            }
            // A candidate that scores nothing is not placed by the fill, even if it would fit: that is what a
            // score of zero means, and it is the only way a caller can say "this one is not worth the RAM" and
            // have the answer come out as a plan rather than as an argument.  Pins are exempt, because a pin is
            // the caller saying exactly that in the other direction.
            if score(c) <= 0.0 {
                continue;
            }
            if fits(used, c.gib, self.budget_gib) {
                chosen.push(c);
                used += c.gib;
            }
        }

        let chosen_ids: HashSet<Id> = chosen.iter().map(|c| c.id).collect();
        let resident: Vec<Candidate> = candidates.iter().filter(|c| chosen_ids.contains(&c.id)).cloned().collect();
        let streamed: Vec<Candidate> = candidates.iter().filter(|c| !chosen_ids.contains(&c.id)).cloned().collect();

        // The spare is the plan's most interesting number, so it gets said out loud: 0.91 GiB at arm A buys
        // nothing, and 2.85 GiB at arm B buys two K layers.
        let spare = self.budget_gib - used;
        let smallest = streamed.iter().map(|c| c.gib).fold(f64::INFINITY, f64::min);
        if chosen.is_empty() {
            notes.push(Note::NothingFits);
        } else if smallest.is_finite() && spare + 1e-9 < smallest {
            notes.push(Note::SpareBuysNothing { spare_gib: spare, smallest_gib: smallest });
        }

        Plan {
            resident,
            streamed,
            used_gib: used,
            spare_gib: spare,
            notes,
        }
    }
}

/// `used + need <= budget`, with a hair of slack so a budget spent exactly on the plan's own numbers counts as
/// spent.  The plan's arm A is exactly this question - 36 GiB against a 37.15 GiB chain - so it must not be
/// answered by a float rounding its way to the wrong answer.
fn fits(used: f64, need: f64, budget: f64) -> bool {
    used + need <= budget + 1e-9
}

impl Plan {
    pub fn has(&self, kind: Kind) -> bool {
        self.resident.iter().any(|c| c.kind == kind)
    }

    pub fn count(&self, kind: Kind) -> usize {
        self.resident.iter().filter(|c| c.kind == kind).count()
    }
}
