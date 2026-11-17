"""Minimal local agent loop using ShadowZero's built-in /v1/tools endpoints."""

import json
import os
import urllib.request

BASE = os.environ.get("SHADOWZERO_URL", "http://127.0.0.1:8081").rstrip("/")


def get_json(path: str):
    with urllib.request.urlopen(f"{BASE}{path}", timeout=15) as resp:
        return json.load(resp)


def post_json(path: str, payload: dict):
    req = urllib.request.Request(
        f"{BASE}{path}",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=180) as resp:
        return json.load(resp)


def main() -> None:
    tools = get_json("/v1/tools")
    print(f"Loaded {len(tools)} tools from {BASE}/v1/tools")

    messages = [
        {
            "role": "system",
            "content": "You have read-only workspace tools and a finish tool. Keep answers concise.",
        },
        {
            "role": "user",
            "content": "Check Cargo.toml in the repo root and tell me the current workspace version, then call finish.",
        },
    ]

    for step in range(1, 7):
        resp = post_json(
            "/v1/chat/completions",
            {
                "messages": messages,
                "tools": tools,
                "max_tokens": 300,
                "temperature": 0,
            },
        )
        msg = resp["choices"][0]["message"]
        messages.append(msg)

        calls = msg.get("tool_calls") or []
        if not calls:
            print(f"\n[step {step}] Final reply:\n{msg.get('content', '').strip()}")
            break

        for tc in calls:
            fn = tc["function"]["name"]
            raw_args = tc["function"].get("arguments", "{}")
            args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
            print(f"[step {step}] -> calling {fn}({args})")

            result = post_json("/v1/tools/execute", {"name": fn, "arguments": args})
            output = result.get("output", "")
            print(f"           <- ok={not result.get('is_error')} ({len(output)} chars)")

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "name": fn,
                    "content": output,
                }
            )
            if fn == "finish":
                print("\nTask closed via finish:", json.dumps(args, indent=2))
                return


if __name__ == "__main__":
    main()
