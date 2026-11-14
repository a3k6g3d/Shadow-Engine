//! Shadow's dequantizer against ggml's reference (the gguf Python package), on real blocks of the model:
//! eval/fixtures/dequant.txt, written by tools/gen_dequant_fixtures.py.  Bit for bit - both sides do the same f32
//! operations in the same order, so any difference is a bug, not rounding.  Needs no model file.

use shadow_model::gguf::GgmlType;
use shadow_model::quant::dequantize;

const IDS: &[(&str, u32)] = &[
    ("F32", 0), ("F16", 1), ("Q4_K", 12), ("Q5_K", 13), ("Q6_K", 14), ("IQ3_XXS", 18),
    ("IQ4_NL", 20), ("IQ3_S", 21), ("IQ2_S", 22), ("IQ4_XS", 23), ("BF16", 30), ("NVFP4", 40),
    ("IQ2_XXS", 16), ("IQ2_XS", 17),
];

fn hex(s: &str) -> Vec<u8> {
    (0..s.len()).step_by(2).map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap()).collect()
}

#[test]
fn every_fixture_block_matches_the_reference_bit_for_bit() {
    let per_type = check_fixtures("dequant.txt");
    assert_eq!(per_type.len(), IDS.len() - 2, "every type the models use (but Q2_0, IQ2_XXS, IQ2_XS) must have fixtures");
}

/// IQ2_XXS and IQ2_XS only occur in the Swift file named IQ3_XXS (its expert gate/up matrices; STATUS 128), so their blocks
/// are in a file of their own.
#[test]
fn iq2_xxs_and_iq2_xs_blocks_match_the_reference_bit_for_bit() {
    let per_type = check_fixtures("dequant_iq2.txt");
    assert_eq!(per_type.keys().copied().collect::<Vec<_>>(), ["IQ2_XS", "IQ2_XXS"]);
}

fn check_fixtures(file: &str) -> std::collections::BTreeMap<&'static str, usize> {
    let path = format!("{}/../../eval/fixtures/{file}", env!("CARGO_MANIFEST_DIR"));
    let text = std::fs::read_to_string(&path).unwrap_or_else(|_| panic!("{path} (tools/gen_dequant_fixtures.py)"));
    let lines: Vec<&str> = text.lines().filter(|l| !l.starts_with('#') && !l.is_empty()).collect();
    assert_eq!(lines.len() % 3, 0);
    let mut per_type = std::collections::BTreeMap::<&'static str, usize>::new();
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
        *per_type.entry(IDS.iter().find(|(n, _)| *n == ty_name).unwrap().0).or_insert(0) += 1;
    }
    eprintln!("blocks checked per type: {per_type:?}");
    per_type
}

#[test]
fn q2_0_by_hand() {
    // d = 0.5 (f16 0x3800); codes 0,1,2,3 repeated: byte 0b11_10_01_00 = 0xe4.
    let mut block = vec![0x00, 0x38];
    block.extend([0xe4u8; 16]);
    let mut y = [0f32; 64];
    dequantize(GgmlType::from_id(42), &block, &mut y).unwrap();
    for (j, v) in y.iter().enumerate() {
        assert_eq!(*v, [-0.5, 0.0, 0.5, 1.0][j % 4], "element {j}");
    }
}

/// No reference exists for Q2_0 here, so this checks it is at least the right kind of number: a Q2_0 down
/// projection's weights must have about the spread of an IQ4_NL down projection's (the same role, a layer away).
#[test]
fn q2_0_real_blocks_look_like_their_neighbours() {
    let p = std::env::var("SHADOW_MODEL").unwrap_or_else(|_| {
        "J:/Strata-work/models/swift15/IQ3_S/Swift-Qwen3.8-Flash-Next-GSQ-RCO-abliterated-IQ3_S-00001-of-00002.gguf".into()
    });
    if !std::path::Path::new(&p).is_file() {
        return eprintln!("SKIPPED, not passed: no model at {p}");
    }
    let g = shadow_model::gguf::Gguf::open(std::path::Path::new(&p)).unwrap();
    let Some(t1) = g.tensor("blk.1.ffn_down_exps.weight") else { return };
    if t1.ty.name() != "Q2_0" {
        return;
    }
    let rms = |name: &str| -> f64 {
        let t = g.tensor(name).unwrap();
        let (block, bytes) = t.ty.block().unwrap();
        // the first expert's whole down projection: 2560 rows of 640
        let n = 640 * 2560;
        let nbytes = (n / block * bytes) as usize;
        use std::io::{Read, Seek, SeekFrom};
        let mut f = std::fs::File::open(&g.shards[t.shard].path).unwrap();
        let mut buf = vec![0u8; nbytes];
        f.seek(SeekFrom::Start(t.offset)).unwrap();
        f.read_exact(&mut buf).unwrap();
        let mut y = vec![0f32; n as usize];
        dequantize(t.ty, &buf, &mut y).unwrap();
        (y.iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>() / n as f64).sqrt()
    };
    let (q2, nl) = (rms("blk.1.ffn_down_exps.weight"), rms("blk.2.ffn_down_exps.weight"));
    eprintln!("expert 0 down projection RMS: blk.1 (Q2_0) {q2:.5}, blk.2 (IQ4_NL) {nl:.5}");
    assert!(q2 > nl / 3.0 && q2 < nl * 3.0, "Q2_0 {q2} vs IQ4_NL {nl}");
}
