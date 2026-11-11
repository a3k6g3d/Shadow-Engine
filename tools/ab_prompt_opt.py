"""Interleaved A-B-B-A benchmark on fresh servers (tier 45, floor 3.5, port 8091) across 3k, 9.6k, and 15k prompts + 200-token decode.

  py tools/ab_prompt_opt.py [tier_gib] [floor_gib]
"""
import hashlib, json, os, subprocess, sys, time, urllib.request

MODEL = "J:/Strata-work/models/swift15/IQ3_S/Swift-Qwen3.8-Flash-Next-GSQ-RCO-abliterated-IQ3_S-00001-of-00002.gguf"
PORT = 8091
TIER = sys.argv[1] if len(sys.argv) > 1 else "45"
FLOOR = sys.argv[2] if len(sys.argv) > 2 else "3.5"

text = open("J:/Shadow/docs/STATUS.md", encoding="utf-8").read()
NL = chr(10)


def make_prompt(tag, chars):
    doc = text[:chars]
    return f"[{tag}]{NL}{doc}{NL}{NL}Question: summarize the key engineering findings above in three concise sentences."


PROMPTS = [
    ("3k_a", make_prompt("p3k-a-1001", 8800), 16),
    ("3k_b", make_prompt("p3k-b-2002", 8000), 16),
    ("9k6", make_prompt("p9k6-3003", 28000), 16),
    ("15k", make_prompt("p15k-4004", 44500), 16),
    ("decode200", "Write a concise technical explanation of how speculative decoding with a verify pass and rollback works in a transformer + recurrent hybrid model.", 200),
]


def wait_ready(port, proc, timeout=180):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc.poll() is not None:
            raise RuntimeError(f"server exited early with code {proc.returncode}")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as r:
                if r.status == 200:
                    return time.time() - t0
        except Exception:
            pass
        time.sleep(0.5)
    raise TimeoutError("server did not become ready")


def shutdown(port, proc):
    try:
        req = urllib.request.Request(f"http://127.0.0.1:{port}/shutdown", data=b"", method="POST")
        urllib.request.urlopen(req, timeout=5)
    except Exception:
        pass
    try:
        proc.wait(timeout=15)
    except Exception:
        proc.kill()


def run_arm(arm_name, extra_flags, run_idx, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    log_out = os.path.join(out_dir, f"{arm_name}_r{run_idx}.out")
    log_err = os.path.join(out_dir, f"{arm_name}_r{run_idx}.err")
    args = [
        "J:/Shadow/target/release/shadow.exe",
        "serve",
        MODEL,
        "--port", str(PORT),
        "--ctx", "24576",
        "--ram-tier-gib", str(TIER),
        "--ram-floor-gib", str(FLOOR),
    ] + extra_flags
    env = os.environ.copy()
    env["SHADOW_PHASES"] = "1"
    with open(log_out, "w") as fout, open(log_err, "w") as ferr:
        proc = subprocess.Popen(args, stdout=fout, stderr=ferr, env=env)
        try:
            load_s = wait_ready(PORT, proc)
            print(f"[{arm_name} r{run_idx}] ready in {load_s:.1f} s", flush=True)
            results = {}
            for pname, content, max_tok in PROMPTS:
                body = {
                    "messages": [{"role": "user", "content": content}],
                    "max_tokens": max_tok,
                    "temperature": 0,
                    "stream": False,
                    "cache_prompt": False,
                    "chat_template_kwargs": {"enable_thinking": False},
                }
                t0 = time.time()
                req = urllib.request.Request(
                    f"http://127.0.0.1:{PORT}/v1/chat/completions",
                    json.dumps(body).encode(),
                    {"Content-Type": "application/json"},
                )
                resp = json.load(urllib.request.urlopen(req, timeout=600))
                wall_s = time.time() - t0
                tim = resp.get("timings", {})
                reply = resp["choices"][0]["message"].get("content") or ""
                h = hashlib.sha256(reply.encode("utf-8")).hexdigest()[:8]
                p_n = tim.get("prompt_n", 0)
                p_ms = tim.get("prompt_ms", 0.0)
                c_n = tim.get("predicted_n", 0)
                c_ms = tim.get("predicted_ms", 0.0)
                p_tps = p_n / (p_ms / 1000.0) if p_ms > 0 else 0.0
                c_tps = c_n / (c_ms / 1000.0) if c_ms > 0 else 0.0
                results[pname] = {
                    "prompt_n": p_n,
                    "prompt_s": p_ms / 1000.0,
                    "prompt_tps": p_tps,
                    "predicted_n": c_n,
                    "predicted_s": c_ms / 1000.0,
                    "decode_tps": c_tps,
                    "wall_s": wall_s,
                    "hash": h,
                }
                print(
                    f"  {pname:10s}: {p_n:5d} tok in {p_ms/1000.0:6.2f} s ({p_tps:7.1f} tok/s) | decode {c_n:3d} tok in {c_ms/1000.0:5.2f} s ({c_tps:5.1f} tok/s) | hash {h}",
                    flush=True,
                )
            return results
        finally:
            shutdown(PORT, proc)
            time.sleep(2.0)


def main():
    out_dir = "J:/Shadow/bench/prompt-opt-2026-11-10"
    schedule = [
        ("r3", [], 1),
        ("r2", ["--no-prompt-opt-r3"], 1),
        ("r2", ["--no-prompt-opt-r3"], 2),
        ("r3", [], 2),
    ]
    all_runs = []
    for arm, flags, r in schedule:
        res = run_arm(arm, flags, r, out_dir)
        all_runs.append({"arm": arm, "run": r, "metrics": res})
        with open(os.path.join(out_dir, "results_r3.json"), "w") as f:
            json.dump(all_runs, f, indent=2)

    print("\n=== SUMMARY (mean over 2 runs per arm) ===")
    for pname, _, _ in PROMPTS:
        for arm in ["r2", "r3"]:
            vals = [x["metrics"][pname] for x in all_runs if x["arm"] == arm]
            mean_s = sum(v["prompt_s"] for v in vals) / len(vals)
            mean_tps = sum(v["prompt_tps"] for v in vals) / len(vals)
            mean_dec = sum(v["decode_tps"] for v in vals) / len(vals)
            hashes = ",".join(v["hash"] for v in vals)
            print(f"{pname:10s} [{arm:4s}]: prompt {mean_s:6.2f} s ({mean_tps:7.1f} tok/s), decode {mean_dec:5.1f} tok/s, hashes={hashes}")


if __name__ == "__main__":
    main()
