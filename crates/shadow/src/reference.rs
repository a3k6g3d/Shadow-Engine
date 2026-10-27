//! The stage 1 gate: Shadow's forward pass against the reference fixture (eval/fixtures/reference, written by
//! tools/make_reference_fixture.py from the published implementation run on the BF16 checkpoint).
//!
//! The reference ran on the original BF16 weights; Shadow runs on the quantized file, so the two cannot agree bit for
//! bit.  What they must agree on is direction: each stream's cosine similarity with the reference, per token, per layer.
//! Quantization noise shows as a few percent and grows slowly with depth; a wrong operation shows as a cliff at the
//! layer that has it.  `compare` reports both, and the test reads the cliff.

use shadow_eval::json::{parse, Value as J};
use shadow_model::forward::{Backend, Model, Session, Stepper};
use std::collections::BTreeMap;
use std::path::Path;

pub struct Fixture {
    pub prompt: String,
    pub ids: Vec<u32>,
    /// With `Some`, the fixture holds only these positions' rows of every stream (long prompts); otherwise all of them.
    pub positions: Option<Vec<usize>>,
    pub layers: usize,
    /// name -> (shape, values)
    pub tensors: BTreeMap<String, (Vec<usize>, Vec<f32>)>,
    pub order: Vec<String>,
}

impl Fixture {
    pub fn load(dir: &Path) -> Result<Fixture, String> {
        let m = parse(&std::fs::read_to_string(dir.join("manifest.json")).map_err(|e| e.to_string())?)?;
        let blob = std::fs::read(dir.join("tensors.bin")).map_err(|e| e.to_string())?;
        let ids = m.get("ids").and_then(J::as_arr).ok_or("no ids")?.iter().map(|v| v.as_f64().unwrap() as u32).collect();
        let mut tensors = BTreeMap::new();
        let mut order = Vec::new();
        for t in m.get("tensors").and_then(J::as_arr).ok_or("no tensors")? {
            let name = t.get("name").and_then(J::as_str).ok_or("tensor without a name")?.to_string();
            let shape: Vec<usize> =
                t.get("shape").and_then(J::as_arr).ok_or("no shape")?.iter().map(|v| v.as_f64().unwrap() as usize).collect();
            let off = t.get("offset").and_then(J::as_f64).ok_or("no offset")? as usize;
            let n: usize = shape.iter().product();
            let vals = blob[off..off + 4 * n].chunks(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
            order.push(name.clone());
            tensors.insert(name, (shape, vals));
        }
        Ok(Fixture {
            prompt: m.get("prompt").and_then(J::as_str).unwrap_or("").to_string(),
            ids,
            positions: m.get("positions").and_then(J::as_arr).map(|a| a.iter().map(|v| v.as_f64().unwrap() as usize).collect()),
            layers: m.get("layers").and_then(J::as_f64).ok_or("no layers")? as usize,
            tensors,
            order,
        })
    }
}

#[derive(Debug, Clone)]
pub struct Row {
    pub name: String,
    pub dim: usize,
    /// per token: cosine similarity with the reference
    pub cos: Vec<f64>,
    /// per token: |shadow - reference| / |reference|
    pub rel: Vec<f64>,
    pub max_abs: f64,
}

impl Row {
    pub fn cos_min(&self) -> f64 {
        self.cos.iter().cloned().fold(f64::MAX, f64::min)
    }
    pub fn cos_mean(&self) -> f64 {
        self.cos.iter().sum::<f64>() / self.cos.len() as f64
    }
    pub fn rel_max(&self) -> f64 {
        self.rel.iter().cloned().fold(0.0, f64::max)
    }
}

pub struct Report {
    pub rows: Vec<Row>,
    /// (layer name, per kept position: how many attended positions differ between Shadow's sparse selection and the
    /// reference's, counted both ways)
    pub selections: Vec<(String, Vec<usize>)>,
    /// (layer name, k, per token: how many of the k routed experts Shadow shares with the reference)
    pub routes: Vec<(String, usize, Vec<usize>)>,
}

/// Runs the fixture's prompt through Shadow's forward pass (stopping after the fixture's last layer) and compares every
/// stream the fixture holds.
pub fn compare(model: &Model, be: &dyn Backend, fx: &Fixture) -> Report {
    compare_stepper(&mut Session::new(model, be), fx)
}

/// The same comparison for any `Stepper` (the CPU session, or the GPU session).
pub fn compare_stepper(s: &mut dyn Stepper, fx: &Fixture) -> Report {
    s.start_trace();
    s.set_stop_after(Some(fx.layers));
    for &id in &fx.ids {
        s.step(id);
    }
    let trace = s.take_trace().unwrap();
    let t = fx.ids.len();
    let pos: Vec<usize> = fx.positions.clone().unwrap_or_else(|| (0..t).collect());
    let np = pos.len();
    let mut rows = Vec::new();
    let mut routes = Vec::new();
    let mut selections = Vec::new();
    for name in &fx.order {
        let (shape, want) = &fx.tensors[name];
        let Some(all) = trace.get(name) else { continue };
        let dim = want.len() / np;
        assert_eq!(all.len(), t * dim, "{name}: shadow {} values, expected {} (reference shape {shape:?})", all.len(), t * dim);
        // Shadow's rows at the fixture's positions
        let got: Vec<f32> = pos.iter().flat_map(|&p| all[p * dim..(p + 1) * dim].iter().copied()).collect();
        if name.ends_with(".selected") {
            let diff = (0..np)
                .map(|i| {
                    let set = |v: &[f32]| -> std::collections::BTreeSet<i64> { v.iter().filter(|&&x| x >= 0.0).map(|&x| x as i64).collect() };
                    let (a, b) = (set(&got[i * dim..(i + 1) * dim]), set(&want[i * dim..(i + 1) * dim]));
                    a.symmetric_difference(&b).count()
                })
                .collect();
            selections.push((name.trim_end_matches(".selected").to_string(), diff));
            continue;
        }
        if name.ends_with(".route_idx") {
            let k = dim;
            let shared = (0..np)
                .map(|i| got[i * k..(i + 1) * k].iter().filter(|g| want[i * k..(i + 1) * k].contains(g)).count())
                .collect();
            routes.push((name.trim_end_matches(".route_idx").to_string(), k, shared));
            continue;
        }
        if name.ends_with(".route_w") {
            continue;
        }
        let (mut cos, mut rel, mut max_abs) = (Vec::new(), Vec::new(), 0f64);
        for i in 0..np {
            let (a, b) = (&got[i * dim..(i + 1) * dim], &want[i * dim..(i + 1) * dim]);
            let (mut ab, mut aa, mut bb, mut dd) = (0f64, 0f64, 0f64, 0f64);
            for (&x, &y) in a.iter().zip(b) {
                let (x, y) = (x as f64, y as f64);
                ab += x * y;
                aa += x * x;
                bb += y * y;
                dd += (x - y) * (x - y);
                max_abs = max_abs.max((x - y).abs());
            }
            cos.push(ab / (aa.sqrt() * bb.sqrt() + 1e-30));
            rel.push(dd.sqrt() / (bb.sqrt() + 1e-30));
        }
        rows.push(Row { name: name.clone(), dim, cos, rel, max_abs });
    }
    Report { rows, selections, routes }
}

pub fn print(r: &Report) {
    println!("{:<22} {:>6} {:>10} {:>10} {:>10} {:>10}", "stream", "dim", "cos min", "cos mean", "rel max", "max |diff|");
    for row in &r.rows {
        println!("{:<22} {:>6} {:>10.5} {:>10.5} {:>10.4} {:>10.4}", row.name, row.dim, row.cos_min(), row.cos_mean(),
                 row.rel_max(), row.max_abs);
    }
    for (name, diff) in &r.selections {
        println!("{name}: attended positions that differ from the reference's sparse selection, per kept position: {diff:?}");
    }
    for (name, k, shared) in &r.routes {
        println!("{name}: routed experts shared with the reference, per token: {shared:?} (of {k})");
        if let Some(row) = r.rows.iter().find(|x| x.name == format!("{name}.moe")) {
            let c: Vec<String> = row.cos.iter().map(|v| format!("{v:.3}")).collect();
            println!("{name}.moe cosine per token:                                     [{}]", c.join(", "));
        }
    }
}
