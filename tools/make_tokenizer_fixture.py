"""Writes eval/fixtures/tokenizer.json and eval/fixtures/chat.json: what the checkpoint's own tokenizer
(tokenizer.json, via Hugging Face `tokenizers`) and chat template (chat_template.jinja, rendered by transformers) make
of a set of texts and conversations.  crates/shadow-model/tests/tokenizer_fixture.rs holds Shadow's tokenizer and
template to them exactly.  That is the outside reference for stage 0's "tokenizer reproduces the fixture ids".

    I:/shadow-ref/hf/ holds the files (tools/hf_range.py fetched them).
    .venv-ref/Scripts/python.exe tools/make_tokenizer_fixture.py
"""
import json
import pathlib
import unicodedata

from transformers import AutoTokenizer

HF = pathlib.Path("I:/shadow-ref/hf")
OUT = pathlib.Path(__file__).resolve().parent.parent / "eval" / "fixtures"

TEXTS = [
    "", " ", "\n", "Hello world", "Hello, world!", "The capital of France is Paris.", "don't I'll we've she'd he's I'm",
    "DON'T I'LL WE'VE", "12345 67890 3.14159", "x² + y³ = z", "a\n\nb\n\n\nc", "  leading and trailing  ",
    "tab\tseparated\tvalues", "line one\r\nline two\r\n", "fn main() {\n    println!(\"hi\");\n}\n",
    "def f(x):\n    return x**2  # square\n", "https://example.com/path?query=1&b=2", "snake_case camelCase PascalCase",
    "日本語のテキストです。", "中文文本，包括标点。", "한국어 텍스트입니다.", "Привет, мир! Это тест.", "مرحبا بالعالم",
    "naïve café Ünïcödé", "e\u0301 combining accent", "emoji 🙂👍🏽 and 🇫🇷 flag", "𝔘𝔫𝔦𝔠𝔬𝔡𝔢 math letters",
    "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n", "<think>\nreasoning\n</think>\n\nanswer",
    "<tool_call>\n<function=f>\n</function>\n</tool_call>", "a<|endoftext|>b", "<|im_start|>", "<|im_end|><|im_end|>",
    "Spaces    inside   text", "!!!???...,,,;;;:::", "$100.00 (approx.) [see #3] {x: 1}", "ALLCAPS lowercase MiXeD",
    "The quick brown fox jumps over the lazy dog. " * 6,
    "What is 17 times 23? Answer in one short sentence.",
    "He said: \"it's fine\" — and left.", "tabs\t\tand  spaces \t mix", "trailing newline\n", "\n\nstarts with newlines",
    "0xDEADBEEF 1e-6 -42", "email@example.com +1 (555) 010-9999", "naïve\u00a0nbsp and \u3000ideographic space",
]

CONVS = [
    ("user only", [{"role": "user", "content": "What is 17 times 23?"}], {}),
    ("system + user", [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Hi"}], {}),
    ("thinking off", [{"role": "user", "content": "Hi"}], {"enable_thinking": False}),
    ("thinking off + system", [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Hi"}],
     {"enable_thinking": False}),
    ("effort medium", [{"role": "user", "content": "Hi"}], {"reasoning_effort": "medium"}),
    ("effort low", [{"role": "system", "content": "S"}, {"role": "user", "content": "Hi"}], {"reasoning_effort": "low"}),
    ("whitespace trimmed", [{"role": "user", "content": "  padded \n"}], {}),
    ("multi-turn with reasoning",
     [{"role": "user", "content": "2+2?"}, {"role": "assistant", "content": "4", "reasoning_content": "two and two"},
      {"role": "user", "content": "and 3+3?"}], {}),
    ("history without reasoning",
     [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}, {"role": "user", "content": "c"}], {}),
    ("preserve_thinking off",
     [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b", "reasoning_content": "r"},
      {"role": "user", "content": "c"}], {"preserve_thinking": False}),
    ("tool results",
     [{"role": "user", "content": "go"}, {"role": "assistant", "content": "ok"}, {"role": "tool", "content": "r1"},
      {"role": "tool", "content": "r2"}], {"reasoning_effort": "medium"}),
    ("no generation prompt", [{"role": "user", "content": "Hi"}], {"add_generation_prompt": False}),
]


def main() -> None:
    tok = AutoTokenizer.from_pretrained(str(HF))
    OUT.mkdir(parents=True, exist_ok=True)
    cases = []
    for t in TEXTS:
        ids = tok(t, add_special_tokens=False)["input_ids"]
        # the checkpoint's tokenizer.json has an NFC normalizer, so the text comes back normalized
        assert tok.decode(ids) == unicodedata.normalize("NFC", t) or "\ufffd" in tok.decode(ids), repr(t)
        cases.append({"text": t, "ids": ids})
    (OUT / "tokenizer.json").write_text(
        json.dumps({"source": "Qwen/Qwen3.8-Flash-Next tokenizer.json via tokenizers/transformers " +
                    __import__("transformers").__version__, "special": True, "cases": cases}, ensure_ascii=False, indent=1),
        encoding="utf-8", newline="\n")
    chats = []
    for name, msgs, kw in CONVS:
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=kw.pop("add_generation_prompt", True), **kw)
        chats.append({"name": name, "messages": msgs, "options": kw, "text": text})
    (OUT / "chat.json").write_text(json.dumps({"cases": chats}, ensure_ascii=False, indent=1), encoding="utf-8", newline="\n")
    print(f"{len(cases)} tokenizer cases ({sum(len(c['ids']) for c in cases)} ids), {len(chats)} chat cases -> {OUT}")


if __name__ == "__main__":
    main()
