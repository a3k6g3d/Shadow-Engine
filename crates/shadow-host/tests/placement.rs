//! The placement policy's tests: the plan's own arms, the plan's own numbers, and what the policy does with them.
//!
//! The point of these is not "the code does what the code says".  It is the plan's Phase 0 table, expressed as
//! data: the arms A/B/C, and the one arithmetic fact the whole of the plan's recomputed answer turns on - that
//! the 36-layer J chain is 37.15 GiB and so **does not quite fit in 36**.

use std::collections::HashSet;

use shadow_host::placement::{Id, Kind, TOLERANCE_GIB};
use shadow_host::placement::{Note, Policy, Shape};

fn close(a: f64, b: f64) -> bool {
    (a - b).abs() <= TOLERANCE_GIB
}

fn geometry() -> Vec<shadow_host::placement::Candidate> {
    shadow_host::placement::swift15()
}

fn emb(geo: &[shadow_host::placement::Candidate]) -> shadow_host::placement::Candidate {
    *geo.iter().find(|c| c.kind == Kind::Embedding).unwrap()
}

fn layers(geo: &[shadow_host::placement::Candidate], kind: Kind) -> Vec<shadow_host::placement::Candidate> {
    geo.iter().copied().filter(|c| c.kind == kind).collect()
}

/// The geometry, checked against the plan's table.  The plan says every one of its numbers was measured off the
/// file, so this is where that claim is tested - and the tolerance is a rounding convention, not a fudge: the
/// plan quotes 37.14 for a chain whose parts sum to 37.152.
#[test]
fn geometry_matches_the_plan() {
    let geo = geometry();
    let j = layers(&geo, Kind::Recurrent);
    let k = layers(&geo, Kind::Attention);
    assert_eq!(j.len(), 36, "48 blocks, every 4th a K: 36 J");
    assert_eq!(k.len(), 12);
    assert_eq!(geo.len(), j.len() + k.len() + 1, "the 48 layers, plus the embedding, which is not in a layer");

    assert!(close(j.iter().map(|c| c.gib).sum::<f64>(), 37.14));
    assert!(close(k.iter().map(|c| c.gib).sum::<f64>(), 13.03));
    assert!(close(emb(&geo).gib, 26.82));
    for c in j.iter().chain(k.iter()) {
        assert!(close(c.gib, c.split.total_gib()), "a candidate's bytes are its parts: {:?}", c.id);
    }

    // The plan's per-layer split, named parts only - and the two derived `other` rows, which is why the sums
    // close at all.  J: 0.961 + 0.041 + 0.020 + 0.010 = 1.032.  K: 1.020 + 0.041 + 0.020 + 0.005 = 1.086.
    assert!(close(j[0].split.experts_gib, 0.961));
    assert!(close(k[0].split.experts_gib, 1.020));
    assert!(close(j[0].gib, 1.032));
    assert!(close(k[0].gib, 1.086));

    // And the gap the plan does not close: 37.14 + 13.03 + 26.82 is 76.99, and it says 77.80 of tensor data.
    // ~0.8 GiB is not attributed to any of the three.  Asserted as a gap, because a test that silently
    // absorbed it would be absorbing a fact.
    let attributed: f64 = geo.iter().map(|c| c.gib).sum();
    assert!(close(attributed, 76.99), "the three together: {attributed}");
    assert!(
        attributed + 0.5 < 77.80,
        "the plan's total is 77.80 - {attributed} is unaccounted, so the figures below cannot all be room"
    );
}

/// **The fact the plan's recomputed answer turns on**, tested as arithmetic and not as prose: the whole J chain
/// does not fit in the 36 GiB this session runs on, and it does fit in the next rung the plan's table has a
/// measured number for.
#[test]
fn the_j_chain_does_not_quite_fit_in_36() {
    let geo = geometry();
    let chain: f64 = layers(&geo, Kind::Recurrent).iter().map(|c| c.gib).sum();
    assert!(chain > 36.0, "37.15 GiB of J chain against a 36 GiB budget - the plan's whole section is this line");
    assert!(chain <= 40.0, "and it fits in 40, which is the plan's arm B and its 13.2 tok/s");

    let p = Policy {
        pins: layers(&geo, Kind::Recurrent).iter().map(|c| c.id).collect(),
        ..Policy::default()
    };
    let plan = p.place(&geo);
    assert_eq!(plan.count(Kind::Recurrent), 34, "so arm A is 34 of the 36, and the last two are streamed");
    assert!(close(plan.used_gib, 35.088), "34 x 1.032");
    assert!(close(plan.spare_gib, 0.912), "and 0.91 GiB left over");
    assert!(
        plan.notes.contains(&Note::SpareBuysNothing { spare_gib: plan.spare_gib, smallest_gib: 1.032 }),
        "which buys nothing: {plan:?}"
    );
}

/// Arm B as the plan writes it is "the whole J chain resident".  The policy is asked that question directly, by
/// ranking J above K - and then it goes further, and the test is about saying so out loud rather than quietly
/// moving the goalposts: 2.85 GiB of spare buys two K layers, and on the plan's own measured line that is worth
/// ~1.5 tok/s, which is a **prediction**, not a result.
#[test]
fn arm_b_the_chain_fits_and_the_policy_spends_the_change() {
    let geo = geometry();
    let mut p = Policy { budget_gib: 40.0, ..Policy::default() };
    p.shape.weight_recurrent = 2.0; // assumed shape - see Shape::default; the scale is the measured part
    let plan = p.place(&geo);

    assert_eq!(plan.count(Kind::Recurrent), 36, "the whole J chain, all 36 - the plan's arm B");
    assert_eq!(plan.count(Kind::Attention), 2, "plus two K layers, which the plan's arm B does not mention");
    assert_eq!(plan.count(Kind::Embedding), 0);
    assert!(close(plan.used_gib, 39.324), "37.152 + 2 x 1.086");
    assert!(close(plan.spare_gib, 0.676));
    assert!(close(2.172 * p.shape.per_gib, 1.4555), "the measured line, applied to the change it spends");
    assert!(plan.notes.iter().any(|n| matches!(n, Note::SpareBuysNothing { .. })));
}

/// Arm C: "40 GiB, embedding resident, all 48 layers streamed".  The policy agrees with the first half and not
/// the second - pin the embedding and 13.18 GiB of the budget is left, which is 12 layers of something.  A plan
/// that pins the 26.82 GiB embedding and then leaves a third of the budget unspent is a plan that wants to be
/// told so.
#[test]
fn arm_c_pins_the_embedding_and_then_owes_the_rest() {
    let geo = geometry();
    let mut p = Policy { budget_gib: 40.0, ..Policy::default() };
    p.pins = vec![emb(&geo).id];
    let plan = p.place(&geo);

    assert!(plan.has(Kind::Embedding), "the pin holds");
    assert_eq!(plan.count(Kind::Recurrent) + plan.count(Kind::Attention), 12, "and the spare is spent");
    assert!(close(plan.used_gib, 39.366));
    assert_eq!(plan.count(Kind::Recurrent), 9, "id order, since the default shape cannot rank one kind against \
another - which is the point of the next test");
    assert_eq!(plan.count(Kind::Attention), 3);

    // Arm C as written - embedding alone - is a legal answer, and this is what it costs.
    let mut as_written = Policy { budget_gib: 40.0, ..Policy::default() };
    as_written.pins = vec![emb(&geo).id];
    as_written.shape.per_gib = 0.0; // nothing else is worth anything: the plan's arm C, in the only way it
                                     // can be expressed as a ranking
    let plain = as_written.place(&geo);
    assert_eq!(plain.count(Kind::Embedding), 1);
    assert_eq!(plain.resident.len(), 1, "and nothing else resident, as the plan writes it");
    assert!(close(plain.spare_gib, 13.18), "13.18 GiB of budget, unspent");
    assert!(close(13.18 * 0.67, 8.83), "on the measured line: the difference between arm C and arm C-plus");
}

/// **What the measurement can and cannot do.**  One measured point - 0.67 tok/s per GiB, an aggregate slope - says
/// nothing about whether a GiB of J is worth more than a GiB of K, so `Shape::default` does not pretend to: every
/// weight is 1.0, every candidate scores the same, and the order that comes out is an order over ids.  That is
/// the honest baseline, and this test pins it, because the moment someone quotes a default placement as if it
/// were measured this test will fail and make them say which knob they turned.
#[test]
fn the_measured_line_alone_cannot_rank_one_kind_against_another() {
    let geo = geometry();
    let shape = Shape::default();
    assert_eq!(shape.per_gib, 0.67, "measured: 6.4 tok/s across 9.5 GiB of the plan's budget table");
    assert_eq!(shape.weight_recurrent, shape.weight_attention, "and equal weights are what that point supports");

    let p = Policy::default();
    let plan = p.place(&geo);
    // Equal scores, so the order is by id - which means the composition of the answer is the input order, not a
    // measurement.  36 GiB of budget, filled in id order: eight (3J+1K) groups, then two more J.
    assert_eq!(plan.count(Kind::Recurrent), 26);
    assert_eq!(plan.count(Kind::Attention), 8);
    assert!(close(plan.used_gib, 35.52));

    // And the same budget with the shape knob turned gives a different set, which is the whole reason the knob is
    // named in the code and the assumption named in the comment above it.
    let mut shaped = Policy::default();
    shaped.shape.weight_recurrent = 2.0;
    let other = shaped.place(&geo);
    assert_ne!(
        other.resident.iter().map(|c| c.id).collect::<Vec<_>>(),
        plan.resident.iter().map(|c| c.id).collect::<Vec<_>>()
    );
    // And the totals differ too, because the two answers are made of different-sized candidates: the shape put
    // 34 J layers (35.088) where the flat ranking put eight groups of three-plus-one (35.520).  Same budget,
    // different set - which is the whole reason the shape is a named knob in the code.
    assert_eq!(other.count(Kind::Recurrent), 34);
    assert_eq!(other.count(Kind::Attention), 0);
    assert!(close(other.used_gib, 35.088));
    assert!(other.used_gib < plan.used_gib, "and it is the shape, not the measurement, that chose which");
}

/// The plan's Phase 1 item 3 rule (hysteresis), stated where it can be tested.  Two candidates, same kind, so the
/// measurement says they are worth the same; the only difference is which one is already resident.  With no
/// damping the incumbent loses the tie to a deterministic id order; with any damping at all it keeps its place,
/// which is exactly what damping is for - a policy that oscillates reads the drive twice.
#[test]
fn an_incumbent_is_only_moved_by_a_clear_win() {
    let geo = geometry();
    let two = vec![geo[0], geo[1]]; // both J, both 1.032 GiB, ids 0 and 1
    assert_eq!(two[0].gib, two[1].gib);

    let mut p = Policy { budget_gib: 1.032, ..Policy::default() };
    p.incumbent = HashSet::from([two[1].id]); // the incumbent is the one that loses an undamped tie
    assert_eq!(p.place(&geo_only(&two)).resident[0].id, two[0].id, "no hysteresis: id order decides");

    p.hysteresis = 1.2;
    assert_eq!(p.place(&geo_only(&two)).resident[0].id, two[1].id, "1.2x: the incumbent keeps its place");
    p.hysteresis = 1.0000001;
    assert_eq!(p.place(&geo_only(&two)).resident[0].id, two[1].id, "the smallest damping the type can express");
}

/// A pin that does not fit is reported, not dropped - the plan's rule about honesty, applied to the policy's own
/// output.  Here the budget is 20 GiB and the pin is the 26.82 GiB embedding.
#[test]
fn a_pin_that_does_not_fit_says_so() {
    let geo = geometry();
    let mut p = Policy { budget_gib: 20.0, ..Policy::default() };
    p.pins = vec![emb(&geo).id];
    let plan = p.place(&geo);
    assert!(!plan.has(Kind::Embedding));
    assert!(plan.notes.iter().any(|n| matches!(n, Note::PinDoesNotFit { id, .. } if *id == emb(&geo).id)));
    assert!(plan.resident.iter().all(|c| c.kind != Kind::Embedding), "it is not quietly placed anyway");
    assert_eq!(plan.resident.len(), 19, "and the rest of the budget still gets spent: {plan:?}");

    // A pin for something not on the table is the caller's bug, and it gets the same treatment.
    let mut p = Policy { budget_gib: 20.0, ..Policy::default() };
    p.pins = vec![Id(0, Kind::Embedding)];
    assert!(p.place(&geo).notes.iter().any(|n| matches!(n, Note::PinDoesNotFit { gib, .. } if *gib == 0.0)));
}

/// The edges: no budget at all, and enough for everything.  Both are answers, and both say which one it was.
#[test]
fn the_edges_are_answers_too() {
    let geo = geometry();

    let p = Policy { budget_gib: 0.0, ..Policy::default() };
    let plan = p.place(&geo);
    assert!(plan.resident.is_empty());
    assert_eq!(plan.streamed.len(), geo.len());
    assert!(plan.notes.contains(&Note::NothingFits));

    let total: f64 = geo.iter().map(|c| c.gib).sum();
    let p = Policy { budget_gib: total + 1.0, ..Policy::default() };
    let plan = p.place(&geo);
    assert_eq!(plan.resident.len(), geo.len(), "the whole model, which is 78 GiB of RAM this PC does not have");
    assert!(plan.streamed.is_empty());
    assert!(plan.notes.is_empty(), "nothing was refused, so there is nothing to say");
    assert!(close(plan.used_gib, total));

    // A budget spent exactly on the plan's own numbers counts as spent - arm A is this question.
    let p = Policy { budget_gib: 37.152, ..Policy::default() };
    assert_eq!(p.place(&layers(&geo, Kind::Recurrent)).resident.len(), 36);
}

/// Determinism, which the plan needs the policy to have: the answer is a function of the data, so the same
/// policy over the same model in a different order must give the same set.
#[test]
fn same_policy_same_answer_whatever_order_the_data_arrives_in() {
    let geo = geometry();
    let p = Policy { budget_gib: 40.0, ..Policy::default() };
    let a = p.place(&geo);
    let mut shuffled = geo.clone();
    shuffled.reverse();
    let b = p.place(&shuffled);
    let (used_a, notes_a) = (a.used_gib, a.notes.clone());
    let (used_b, notes_b) = (b.used_gib, b.notes.clone());
    let mut a: Vec<Id> = a.resident.iter().map(|c| c.id).collect();
    let mut b: Vec<Id> = b.resident.iter().map(|c| c.id).collect();
    a.sort();
    b.sort();
    assert_eq!(a, b, "the same set, whichever order the data arrived in");
    assert!(close(used_a, used_b));
    assert_eq!(notes_a, notes_b);
}

/// The set is what the decision is; the lists come out in the order the data arrived, which is the caller's
/// order and not part of the answer - so the comparison above sorts, and this says so out loud.
#[test]
fn the_lists_follow_the_data_and_the_decision_does_not() {
    let geo = geometry();
    let p = Policy { budget_gib: 40.0, ..Policy::default() };
    let mut shuffled = geo.clone();
    shuffled.reverse();
    let (a, b) = (p.place(&geo), p.place(&shuffled));
    assert_eq!(a.resident.len(), b.resident.len());
    assert_eq!(a.resident[0].id, geo[0].id, "in the order it arrived");
    assert_eq!(b.resident[0].id, a.resident.iter().map(|c| c.id).max().unwrap(), "and so, the other way round");
}

fn geo_only(v: &[shadow_host::placement::Candidate]) -> Vec<shadow_host::placement::Candidate> {
    v.to_vec()
}
