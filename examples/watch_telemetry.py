"""Poll ShadowZero's /v1/telemetry endpoint and print per-GPU + cache + spec stats."""

import json
import os
import sys
import time
import urllib.request

BASE = os.environ.get("SHADOWZERO_URL", "http://127.0.0.1:8081").rstrip("/")
GIB = 1024 ** 3


def fetch_telemetry() -> dict:
    with urllib.request.urlopen(f"{BASE}/v1/telemetry", timeout=3) as resp:
        return json.load(resp)


def format_line(t: dict) -> str:
    proc = t["process"]
    gpu = t["gpu"]
    ram = t["ram"]
    spec = t["spec"]

    devs = gpu.get("devices") or []
    gpu_str = " | ".join(
        f"GPU{d['index']} {d['utilization_percent']:4.0f}% "
        f"({d['vram_used_bytes']/GIB:.1f}/{d['vram_total_bytes']/GIB:.1f}G {d['temperature_c']}C)"
        for d in devs
    ) or f"{gpu['name']} {gpu['utilization_percent']:.0f}%"

    pinned_g = ram["pinned_tier_bytes"] / GIB
    avail_g = ram["system_available_bytes"] / GIB
    return (
        f"state={proc['state']} tok/s={proc['tok_per_sec']:5.1f} | "
        f"{gpu_str} | "
        f"VRAM cache={gpu['expert_cache_hit_rate']:5.1f}% | "
        f"RAM tier={pinned_g:.1f}G (free {avail_g:.1f}G, hits={ram['ram_served_hits']}, disk={ram['file_served_misses']}) | "
        f"DFlash accept={spec['acceptance_rate']:4.1f}% ({spec['mean_accepted_per_pass']:.2f}/pass)"
    )


def main() -> None:
    interval = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
    print(f"Watching {BASE}/v1/telemetry every {interval}s (Ctrl+C to stop)")
    while True:
        try:
            snap = fetch_telemetry()
            print(format_line(snap), flush=True)
        except Exception as err:
            print(f"waiting for {BASE}: {err}", flush=True)
        time.sleep(interval)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
