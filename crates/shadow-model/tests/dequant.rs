//! Shadow's dequantizer against ggml's reference (the gguf Python package), on real blocks of the model:
//! eval/fixtures/dequant.txt, written by tools/gen_dequant_fixtures.py.  Bit for bit - both sides do the same f32
//! operations in the same order, so any difference is a bug, not rounding.  Needs no model file.

use shadow_model::gguf::GgmlType;
use shadow_model::quant::dequantize;

const IDS: &[(&str, u32)] = &[("F32", 0), ("F16", 1), ("Q4_K", 12), ("Q5_K", 13), ("Q6_K", 14), ("IQ3_XXS", 18),
                              ("IQ4_NL", 20), ("IQ3_S", 21), ("IQ2_S", 22), ("IQ4_XS", 23), ("BF16", 30)];

fn hex(s: &str) -> Vec<u8> {
    (0..s.len()).step_by(2).map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap()).collect()
}

#[test]
fn every_fixture_block_matches_the_reference_bit_for_bit() {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../eval/fixtures/dequant.txt");
    let text = std::fs::read_to_string(path).expect("eval/fixtures/dequant.txt (tools/gen_dequant_fixtures.py)");
    let lines: Vec<&str> = text.lines().filter(|l| !l.starts_with('#') && !l.is_empty()).collect();
    assert_eq!(lines.len() % 3, 0);
    let mut per_type = std::collections::BTreeMap::new();
    for rec in lines.chunks(3) {
        let head: Vec<&str> = rec[0].split_whitespace().collect();
        let (ty_name, tensor, block) = (head[0], head[1], head[2]);
        let id = IDS.iter().find(|(n, _)| *n == ty_name).unwrap_or_else(|| panic!("type {ty_name}")).1;
        let ty = GgmlType::from_id(id);
        let bytes = hex(rec[1]);
        let want: Vec<u32> = hex(rec[2]).chunks(4).map(|c| u32::from_be_bytes(c.try_into().unwrap())).collect();
        let mut got = vec![0f32; want.len()];
        dequantize(ty, &bytes, &mut got).unwrap();
        for (i, (g, w)) in got.iter().zip(&want).enumerate() {
            assert_eq!(g.to_bits(), *w, "{ty_name} {tensor} block {block} element {i}: got {g}, reference {}",
                       f32::from_bits(*w));
        }
        *per_type.entry(ty_name).or_insert(0) += 1;
    }
    eprintln!("blocks checked per type: {per_type:?}");
    assert_eq!(per_type.len(), IDS.len(), "every type the model uses (but Q2_0) must have fixtures");
}

#[test]
fn q2_0_is_refused_by_name_not_guessed() {
    let e = dequantize(GgmlType::from_id(42), &[0u8; 18], &mut [0f32; 64]).unwrap_err();
    assert!(e.contains("Q2_0"), "{e}");
}
