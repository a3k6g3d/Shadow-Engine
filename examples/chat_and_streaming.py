"""Basic OpenAI-compatible chat and SSE streaming examples against Shadow."""

import json
import os
import urllib.request

BASE = os.environ.get("SHADOW_URL", "http://127.0.0.1:8081").rstrip("/")


def check_health() -> None:
    with urllib.request.urlopen(f"{BASE}/health", timeout=5) as resp:
        data = json.load(resp)
        print("Server health:", data)


def chat_once() -> None:
    payload = {
        "messages": [
            {"role": "user", "content": "Give me a two-sentence summary of how speculative decoding works."}
        ],
        "max_tokens": 160,
        "temperature": 0,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    req = urllib.request.Request(
        f"{BASE}/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        out = json.load(resp)

    reply = out["choices"][0]["message"]["content"]
    timings = out.get("timings", {})
    print("\n--- Non-streaming reply ---")
    print(reply.strip())
    if timings:
        print(
            f"[timings] prompt={timings.get('prompt_n')} tok ({timings.get('prompt_per_second', 0):.1f} tok/s), "
            f"decode={timings.get('predicted_n')} tok ({timings.get('predicted_per_second', 0):.1f} tok/s)"
        )


def chat_stream() -> None:
    payload = {
        "messages": [
            {"role": "user", "content": "Write a tiny Rust function that checks if a number is prime."}
        ],
        "max_tokens": 256,
        "temperature": 0.2,
        "stream": True,
        "reasoning_effort": "low",
    }
    req = urllib.request.Request(
        f"{BASE}/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    print("\n--- Streaming reply (thinking + answer) ---")
    in_think = False
    with urllib.request.urlopen(req, timeout=180) as resp:
        for raw_line in resp:
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line.startswith("data: "):
                continue
            data = line[6:]
            if data == "[DONE]":
                break
            chunk = json.loads(data)
            delta = (chunk.get("choices") or [{}])[0].get("delta", {})
            reason = delta.get("reasoning_content")
            if reason:
                if not in_think:
                    print("[thinking] ", end="", flush=True)
                    in_think = True
                print(reason, end="", flush=True)
            text = delta.get("content")
            if text:
                if in_think:
                    print("\n[answer] ", end="", flush=True)
                    in_think = False
                print(text, end="", flush=True)
    print()


if __name__ == "__main__":
    check_health()
    chat_once()
    chat_stream()
