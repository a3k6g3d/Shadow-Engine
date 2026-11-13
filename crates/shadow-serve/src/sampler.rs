//! Token sampling for the reply: temperature, top-k, top-p, min-p, repetition / presence / frequency penalties, a seed.
//!
//! **How it composes with speculative decoding.**  The token drawn at output position `pos` is a pure function of
//! (the logits at that position, the tokens generated so far, the seed, `pos`): the random number comes from a
//! counter-based generator keyed by `(seed, pos)`, not from a stream.  Checking a drafted token is therefore "draw the
//! token the target would have drawn at this position, and accept the draft if it is that token" - for a drafter that
//! proposes one token with no probabilities (every drafter here) that is exactly speculative sampling's acceptance, and
//! the reply is the one a non-speculative run would give with the same seed whenever the logits agree (they agree bit
//! for bit in `--exact`, and to the fast mode's usual noise otherwise).
//!
//! No sampling parameters in a request means greedy (`argmax`), as before: the benchmarks, `--exact` and every client that
//! sends nothing keep their old behaviour.  `SHADOW_DEFAULT_SAMPLING=model` makes the model's recommended values the default
//! for requests that carry no `temperature` (the chat page always sends its own).

/// What a request asked for.  `None` fields take the default of the mode (`Sampling::resolve`).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Sampling {
    pub temperature: Option<f32>,
    pub top_k: Option<usize>,
    pub top_p: Option<f32>,
    pub min_p: Option<f32>,
    /// llama.cpp's multiplicative repeat penalty (1.0 = off); `repetition_penalty` in the request.
    pub repeat_penalty: Option<f32>,
    /// OpenAI's additive penalties (0.0 = off).
    pub presence_penalty: Option<f32>,
    pub frequency_penalty: Option<f32>,
    pub seed: Option<u64>,
}

/// The values the Qwen3 family recommends (and the GGUF's `general.sampling.*` carries): temperature 1.0, top-k 20, top-p 0.95.
pub const MODEL_TEMP: f32 = 1.0;
pub const MODEL_TOP_K: usize = 20;
pub const MODEL_TOP_P: f32 = 0.95;

/// How many of the latest generated tokens the penalties look at.
const PENALTY_WINDOW: usize = 512;

/// A resolved sampler for one reply.
#[derive(Debug, Clone)]
pub struct Sampler {
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub min_p: f32,
    pub repeat_penalty: f32,
    pub presence_penalty: f32,
    pub frequency_penalty: f32,
    pub seed: u64,
    scratch: Vec<f32>,
}

impl Sampling {
    pub fn is_empty(&self) -> bool {
        *self == Sampling::default()
    }

    /// The sampler for a request, or `None` for greedy: `temperature` 0 (or anything that is not a finite positive number), or no
    /// sampling parameter at all and no `model_default`.
    pub fn resolve(&self, model_default: bool, fallback_seed: u64) -> Option<Sampler> {
        let temp = match self.temperature {
            Some(t) => t,
            None if model_default || !self.is_empty() => MODEL_TEMP,
            None => return None,
        };
        if !temp.is_finite() || temp <= 0.0 {
            return None;
        }
        let top_k = self.top_k.unwrap_or(MODEL_TOP_K);
        Some(Sampler {
            temperature: temp,
            top_k: if top_k == 0 { usize::MAX } else { top_k },
            top_p: self.top_p.filter(|p| p.is_finite()).unwrap_or(MODEL_TOP_P).clamp(0.0, 1.0),
            min_p: self.min_p.filter(|p| p.is_finite()).unwrap_or(0.0).clamp(0.0, 1.0),
            repeat_penalty: self.repeat_penalty.filter(|p| p.is_finite() && *p > 0.0).unwrap_or(1.0),
            presence_penalty: self.presence_penalty.filter(|p| p.is_finite()).unwrap_or(0.0),
            frequency_penalty: self.frequency_penalty.filter(|p| p.is_finite()).unwrap_or(0.0),
            seed: self.seed.unwrap_or(fallback_seed),
            scratch: Vec::new(),
        })
    }
}

fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

/// A uniform number in [0, 1) from (seed, position).
fn uniform(seed: u64, pos: usize) -> f64 {
    let bits = splitmix64(splitmix64(seed) ^ (pos as u64).wrapping_mul(0xD6E8_FEB8_6659_FD93));
    (bits >> 11) as f64 / (1u64 << 53) as f64
}

pub fn argmax(v: &[f32]) -> u32 {
    let mut best = 0usize;
    for (i, &x) in v.iter().enumerate() {
        if x > v[best] {
            best = i;
        }
    }
    best as u32
}

impl Sampler {
    /// Draws the token for output position `pos` from `logits`; `generated` is every token of the reply so far (the penalties
    /// look at its last `PENALTY_WINDOW`).  Deterministic in its arguments.
    pub fn sample(&mut self, logits: &[f32], pos: usize, generated: &[u32]) -> u32 {
        let penalised = self.repeat_penalty != 1.0 || self.presence_penalty != 0.0 || self.frequency_penalty != 0.0;
        let src: &[f32] = if penalised && !generated.is_empty() {
            self.scratch.clear();
            self.scratch.extend_from_slice(logits);
            let window = &generated[generated.len().saturating_sub(PENALTY_WINDOW)..];
            // a token's count in the window, once each (the window is short: sort a copy)
            let mut w: Vec<u32> = window.to_vec();
            w.sort_unstable();
            let mut i = 0;
            while i < w.len() {
                let t = w[i] as usize;
                let mut j = i;
                while j < w.len() && w[j] == w[i] {
                    j += 1;
                }
                let count = (j - i) as f32;
                if let Some(l) = self.scratch.get_mut(t) {
                    if self.repeat_penalty != 1.0 {
                        *l = if *l > 0.0 { *l / self.repeat_penalty } else { *l * self.repeat_penalty };
                    }
                    *l -= self.frequency_penalty * count + self.presence_penalty;
                }
                i = j;
            }
            &self.scratch
        } else {
            logits
        };
        let max = src.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        if !max.is_finite() {
            return argmax(src);
        }
        // candidates: everything within 18 temperatures of the best (a probability under 2e-8 of it); a vocabulary of 250k
        // is not sorted for that
        let cut = max - 18.0 * self.temperature;
        let mut cand: Vec<(f32, u32)> = Vec::with_capacity(512);
        for (i, &x) in src.iter().enumerate() {
            if x >= cut {
                cand.push((x, i as u32));
            }
        }
        cand.sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap().then(a.1.cmp(&b.1)));
        cand.truncate(self.top_k.max(1));
        // softmax at the temperature (max subtracted)
        let inv_t = 1.0 / self.temperature;
        let mut probs: Vec<f64> = cand.iter().map(|&(x, _)| (((x - max) * inv_t) as f64).exp()).collect();
        let total: f64 = probs.iter().sum();
        for p in probs.iter_mut() {
            *p /= total;
        }
        // min-p: drop what is under min_p of the best; top-p: the smallest head holding top_p of the mass
        let floor = self.min_p as f64 * probs[0];
        let mut keep = probs.iter().take_while(|&&p| p >= floor).count().max(1);
        if self.top_p < 1.0 {
            let mut acc = 0.0;
            let mut n = 0;
            for &p in probs.iter().take(keep) {
                acc += p;
                n += 1;
                if acc >= self.top_p as f64 {
                    break;
                }
            }
            keep = n.max(1);
        }
        let mass: f64 = probs[..keep].iter().sum();
        let mut u = uniform(self.seed, pos) * mass;
        for k in 0..keep {
            if u < probs[k] {
                return cand[k].1;
            }
            u -= probs[k];
        }
        cand[keep - 1].1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn logits(n: usize) -> Vec<f32> {
        // a fixed pseudo-random landscape with a few clear leaders
        (0..n).map(|i| ((splitmix64(i as u64) >> 40) as f32 / 16777216.0) * 6.0 - 3.0).collect()
    }

    fn sampler(s: Sampling) -> Sampler {
        s.resolve(false, 7).expect("not greedy")
    }

    #[test]
    fn nothing_asked_is_greedy_and_temperature_zero_is_greedy() {
        assert!(Sampling::default().resolve(false, 1).is_none());
        assert!(Sampling { temperature: Some(0.0), ..Default::default() }.resolve(true, 1).is_none());
        assert!(Sampling { temperature: Some(0.0), top_p: Some(0.9), ..Default::default() }.resolve(false, 1).is_none());
        // the model default applies only when asked for
        assert!(Sampling::default().resolve(true, 1).is_some());
        // a request with only top_p set samples at the model's temperature
        let s = Sampling { top_p: Some(0.5), ..Default::default() }.resolve(false, 1).unwrap();
        assert_eq!(s.temperature, MODEL_TEMP);
    }

    #[test]
    fn same_seed_and_position_same_token_and_it_does_not_depend_on_the_call_order() {
        let l = logits(5000);
        let mut a = sampler(Sampling { temperature: Some(1.0), seed: Some(42), ..Default::default() });
        let mut b = a.clone();
        let fwd: Vec<u32> = (0..50).map(|p| a.sample(&l, p, &[])).collect();
        let mut back: Vec<u32> = (0..50).rev().map(|p| b.sample(&l, p, &[])).collect();
        back.reverse();
        assert_eq!(fwd, back, "a position's draw must not depend on the draws before it (speculation verifies out of order)");
        let mut c = sampler(Sampling { temperature: Some(1.0), seed: Some(43), ..Default::default() });
        let other: Vec<u32> = (0..50).map(|p| c.sample(&l, p, &[])).collect();
        assert_ne!(fwd, other, "another seed must give another stream");
    }

    #[test]
    fn low_temperature_approaches_argmax_and_top_k_one_is_argmax() {
        let l = logits(5000);
        let best = argmax(&l);
        let mut cold = sampler(Sampling { temperature: Some(0.00001), seed: Some(1), ..Default::default() });
        assert!((0..40).all(|p| cold.sample(&l, p, &[]) == best));
        let mut k1 = sampler(Sampling { temperature: Some(5.0), top_k: Some(1), seed: Some(1), ..Default::default() });
        assert!((0..40).all(|p| k1.sample(&l, p, &[]) == best));
    }

    #[test]
    fn top_k_and_top_p_and_min_p_bound_the_choices() {
        let l = logits(5000);
        let mut order: Vec<usize> = (0..l.len()).collect();
        order.sort_by(|&a, &b| l[b].partial_cmp(&l[a]).unwrap());
        let top5: Vec<u32> = order[..5].iter().map(|&i| i as u32).collect();
        let mut s = sampler(Sampling { temperature: Some(3.0), top_k: Some(5), top_p: Some(1.0), seed: Some(9), ..Default::default() });
        let mut seen = std::collections::HashSet::new();
        for p in 0..400 {
            let t = s.sample(&l, p, &[]);
            assert!(top5.contains(&t), "top_k 5 drew outside the five best");
            seen.insert(t);
        }
        assert!(seen.len() >= 3, "a hot temperature over 5 candidates should spread");
        // min-p: logits 3.0, 2.9, 0.0 -> probabilities relative to the best 1, 0.905, 0.05
        let mut l3 = vec![-30.0f32; 20];
        l3[0] = 3.0;
        l3[1] = 2.9;
        l3[2] = 0.0;
        let mut m = sampler(Sampling { temperature: Some(1.0), min_p: Some(0.95), top_p: Some(1.0), top_k: Some(0), seed: Some(9), ..Default::default() });
        assert!((0..100).all(|p| m.sample(&l3, p, &[]) == 0));
        let mut m = sampler(Sampling { temperature: Some(1.0), min_p: Some(0.5), top_p: Some(1.0), top_k: Some(0), seed: Some(9), ..Default::default() });
        let drawn: std::collections::HashSet<u32> = (0..300).map(|p| m.sample(&l3, p, &[])).collect();
        assert_eq!(drawn, [0u32, 1].into_iter().collect());
        // top-p 0.5 of (0.51, 0.46, 0.03) is the best alone
        let mut tp = sampler(Sampling { temperature: Some(1.0), top_p: Some(0.5), top_k: Some(0), seed: Some(9), ..Default::default() });
        assert!((0..100).all(|p| tp.sample(&l3, p, &[]) == 0));
    }

    #[test]
    fn draws_follow_the_distribution() {
        // two tokens with logits ln(3) and 0: probabilities 0.75 / 0.25
        let mut l = vec![-30.0f32; 100];
        l[10] = (3.0f32).ln();
        l[20] = 0.0;
        let mut s = sampler(Sampling { temperature: Some(1.0), top_p: Some(1.0), top_k: Some(0), seed: Some(5), ..Default::default() });
        let n = 20000;
        let a = (0..n).filter(|&p| s.sample(&l, p, &[]) == 10).count() as f64 / n as f64;
        assert!((a - 0.75).abs() < 0.015, "got {a}");
    }

    #[test]
    fn penalties_push_repeated_tokens_down() {
        let mut l = vec![0.0f32; 50];
        l[3] = 5.0;
        l[4] = 4.9;
        let mut s = sampler(Sampling { temperature: Some(0.001), presence_penalty: Some(1.0), seed: Some(1), ..Default::default() });
        assert_eq!(s.sample(&l, 0, &[]), 3);
        assert_eq!(s.sample(&l, 1, &[3]), 4, "presence penalty must demote a token already generated");
        let mut r = sampler(Sampling { temperature: Some(0.001), repeat_penalty: Some(1.5), seed: Some(1), ..Default::default() });
        assert_eq!(r.sample(&l, 1, &[3]), 4);
        let mut f = sampler(Sampling { temperature: Some(0.001), frequency_penalty: Some(0.06), seed: Some(1), ..Default::default() });
        assert_eq!(f.sample(&l, 1, &[3]), 3, "one occurrence of a small frequency penalty is not enough");
        assert_eq!(f.sample(&l, 1, &[3, 3, 3]), 4);
    }
}
