//! Shadow's NFC against Python's unicodedata on 4,000 random strings over the code points where NFC does something
//! (eval/fixtures/nfc.txt, written by tools/gen_nfc_table.py).  Exact match required.

use shadow_model::nfc::nfc;

fn cps(s: &str) -> String {
    s.chars().map(|c| format!("{:X}", c as u32)).collect::<Vec<_>>().join(" ")
}

#[test]
fn matches_unicodedata_on_4000_strings() {
    let text = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/../../eval/fixtures/nfc.txt")).unwrap();
    let mut n = 0;
    let mut changed = 0;
    for line in text.lines().filter(|l| !l.starts_with('#') && !l.is_empty()) {
        let (i, o) = line.split_once(':').unwrap();
        let parse = |s: &str| -> String { s.split_whitespace().map(|h| char::from_u32(u32::from_str_radix(h, 16).unwrap()).unwrap()).collect() };
        let (input, want) = (parse(i), parse(o));
        let got = nfc(&input);
        assert_eq!(cps(&got), cps(&want), "input {}", cps(&input));
        n += 1;
        changed += (input != want) as usize;
    }
    eprintln!("{n} strings, {changed} of them changed by NFC");
    assert!(n >= 4000 && changed > 400, "the fixture must exercise NFC");
}
