"""Charts and tables of the machine's resource use during a shadow-bench run (docs/BENCHMARKING.md).

`shadow-bench run` samples the PC once a second for every run (CPU, available RAM, disk, paging, GPU utilisation, GPU memory,
power, temperature, clocks, PCIe throughput) into DIR/ARM-runN.{sys,gpu,pcie}.csv, and records each turn's start and end in
results.jsonl.  This script lines the two up:

  DIR/charts/ARM-runN.png   a timeline per run: the panels share one time axis, the warm decode turns are shaded
  DIR/charts/compare.png    the arms side by side (mean over the warm decode turns of every run)
  DIR/charts/resources.csv  the same numbers as a table, and resources.md for pasting into a report

    python tools/plot_bench.py bench/DIR [bench/DIR2 ...]
"""
import csv
import datetime as dt
import json
import os
import sys
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

THROTTLE_MASK = 0x4 | 0x8 | 0x20 | 0x40 | 0x80  # sw power cap, hw slowdown, sw/hw thermal, power brake


def ts(s):
    s = s.strip().replace("/", "-")
    for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S", "%Y%m%d %H:%M:%S"):
        try:
            return dt.datetime.strptime(s, fmt).timestamp()
        except ValueError:
            pass
    return None


def num(x):
    try:
        return float(x)
    except ValueError:
        return float("nan")


def read_sys(path):
    rows = []
    if not os.path.exists(path):
        return rows
    for r in csv.DictReader(open(path, encoding="ascii", errors="replace")):
        t = ts(r.get("time", ""))
        if t is not None:
            rows.append((t, num(r["cpu_pct"]), num(r["avail_mb"]) / 1024, num(r["disk_read_bps"]) / 1e6, num(r["disk_write_bps"]) / 1e6,
                         num(r["pages_per_s"])))
    return rows


def read_gpu(path):
    rows = []
    if not os.path.exists(path):
        return rows
    lines = open(path, encoding="ascii", errors="replace").read().splitlines()
    for line in lines[1:]:
        c = [x.strip() for x in line.split(",")]
        t = ts(c[0]) if c else None
        if t is None or len(c) < 11:
            continue
        thr = int(c[10], 16) if c[10].startswith("0x") else 0
        rows.append((t, num(c[1]), num(c[2]), num(c[3]) / 1024, num(c[4]), num(c[5]), num(c[6]), num(c[7]), float(bool(thr & THROTTLE_MASK))))
    return rows


def read_pcie(path):
    rows = []
    if not os.path.exists(path):
        return rows
    for line in open(path, encoding="ascii", errors="replace"):
        if line.startswith("#") or not line.strip():
            continue
        c = line.split()
        if len(c) >= 5:
            t = ts(c[0] + " " + c[1])
            if t is not None:
                rows.append((t, num(c[3]), num(c[4])))
    return rows


def col(rows, i):
    return np.array([r[i] for r in rows]) if rows else np.array([])


def window_mask(t, spans):
    m = np.zeros(len(t), dtype=bool)
    for a, b in spans:
        m |= (t >= a) & (t <= b)
    return m


PANELS = [  # (title, series name, unit)
    ("GPU utilisation", "gpu_util", "%"), ("GPU power", "gpu_power", "W"), ("GPU memory used", "gpu_mem", "GiB"),
    ("PCIe host->GPU", "pcie_rx", "GB/s"), ("RAM available", "ram", "GiB"), ("CPU", "cpu", "%"), ("Disk read", "disk_read", "MB/s"),
]


def series_of(rec, prefix):
    sysr, gpu, pcie = read_sys(prefix + ".sys.csv"), read_gpu(prefix + ".gpu.csv"), read_pcie(prefix + ".pcie.csv")
    s = {}
    if gpu:
        t = col(gpu, 0)
        s["gpu_util"] = (t, col(gpu, 1))
        s["gpu_power"] = (t, col(gpu, 4))
        s["gpu_mem"] = (t, col(gpu, 3))
        s["gpu_temp"] = (t, col(gpu, 5))
        s["gpu_sm"] = (t, col(gpu, 6))
        s["gpu_throttle"] = (t, col(gpu, 8))
    if pcie:
        s["pcie_rx"] = (col(pcie, 0), col(pcie, 1) / 1000)
        s["pcie_tx"] = (col(pcie, 0), col(pcie, 2) / 1000)
    if sysr:
        t = col(sysr, 0)
        s["cpu"], s["ram"], s["disk_read"], s["disk_write"], s["paging"] = (t, col(sysr, 1)), (t, col(sysr, 2)), (t, col(sysr, 3)), (t, col(sysr, 4)), (t, col(sysr, 5))
    return s


def main():
    for d in sys.argv[1:]:
        out = os.path.join(d, "charts")
        os.makedirs(out, exist_ok=True)
        recs = [json.loads(l) for l in open(os.path.join(d, "results.jsonl"), encoding="utf-8")]
        stats = defaultdict(lambda: defaultdict(list))  # arm -> metric -> per-run values
        for rec in recs:
            if rec.get("error") or not rec.get("resources"):
                continue
            prefix = rec["resources"] if os.path.isabs(rec["resources"]) else os.path.join(os.getcwd(), rec["resources"])
            s = series_of(rec, prefix)
            if not s:
                continue
            turns = [x for x in rec["turns"] if x["turn"] > 0 and "t_start" in x]
            spans = [(x["t_start"], x["t_end"]) for x in turns]
            tokens = sum(x["predicted_n"] for x in turns)
            t0 = rec["t_launch"]
            fig, axes = plt.subplots(len(PANELS), 1, figsize=(11, 2.0 * len(PANELS)), sharex=True)
            for ax, (title, key, unit) in zip(axes, PANELS):
                if key in s:
                    t, v = s[key]
                    ax.plot(t - t0, v, lw=1)
                    if key == "pcie_rx" and "pcie_tx" in s:
                        ax.plot(s["pcie_tx"][0] - t0, s["pcie_tx"][1], lw=1, alpha=0.6)
                for a, b in spans:
                    ax.axvspan(a - t0, b - t0, color="tab:orange", alpha=0.12, lw=0)
                ax.axvline(rec["t_ready"] - t0, color="k", lw=0.8, ls="--")
                ax.set_ylabel(f"{title}\n[{unit}]", fontsize=8)
                ax.grid(alpha=0.3)
            axes[-1].set_xlabel("seconds since launch (dashed: server ready; shaded: warm decode turns)")
            fig.suptitle(f"{os.path.basename(d)}: {rec['arm']} run {rec['run']}   {rec['line']}", fontsize=9)
            fig.tight_layout()
            fig.savefig(os.path.join(out, f"{rec['arm']}-run{rec['run']}.png"), dpi=90)
            plt.close(fig)
            # figures over the warm decode turns
            decode_s = sum(b - a for a, b in spans)
            for key, (t, v) in s.items():
                m = window_mask(t, spans)
                if m.any():
                    stats[rec["arm"]][key + "_mean"].append(float(np.nanmean(v[m])))
                    if key in ("gpu_util", "gpu_power", "pcie_rx", "cpu"):
                        stats[rec["arm"]][key + "_p95"].append(float(np.nanpercentile(v[m], 95)))
                    if key in ("ram",):
                        stats[rec["arm"]]["ram_min"].append(float(np.nanmin(v[m])))
            if "gpu_power" in s and tokens:
                m = window_mask(s["gpu_power"][0], spans)
                if m.any():
                    stats[rec["arm"]]["joules_per_token"].append(float(np.nanmean(s["gpu_power"][1][m]) * decode_s / tokens))
            stats[rec["arm"]]["tok_s"].append(1000 * tokens / sum(x["predicted_ms"] for x in turns) if turns else float("nan"))
        if not stats:
            print(f"{d}: no resource files (run was made with --no-resources, or before they existed)")
            continue
        metrics = ["tok_s", "gpu_util_mean", "gpu_util_p95", "gpu_power_mean", "joules_per_token", "gpu_mem_mean", "gpu_temp_mean",
                   "gpu_sm_mean", "gpu_throttle_mean", "pcie_rx_mean", "pcie_rx_p95", "pcie_tx_mean", "cpu_mean", "cpu_p95", "ram_mean",
                   "ram_min", "disk_read_mean", "disk_write_mean", "paging_mean"]
        arms = list(stats)
        with open(os.path.join(out, "resources.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["arm", "runs"] + metrics)
            for a in arms:
                w.writerow([a, len(stats[a]["tok_s"])] + [f"{np.mean(stats[a][m]):.3f}" if stats[a][m] else "" for m in metrics])
        names = {"tok_s": "decode tok/s", "gpu_util_mean": "GPU util % (mean)", "gpu_util_p95": "GPU util % (p95)", "gpu_power_mean": "GPU power W",
                 "joules_per_token": "GPU J / token", "gpu_mem_mean": "GPU mem GiB", "gpu_temp_mean": "GPU temp C", "gpu_sm_mean": "SM clock MHz",
                 "gpu_throttle_mean": "power/thermal limited (fraction of time)", "pcie_rx_mean": "PCIe rx GB/s (mean)", "pcie_rx_p95": "PCIe rx GB/s (p95)",
                 "pcie_tx_mean": "PCIe tx GB/s (mean)", "cpu_mean": "CPU % (mean)", "cpu_p95": "CPU % (p95)", "ram_mean": "RAM available GiB (mean)",
                 "ram_min": "RAM available GiB (min)", "disk_read_mean": "disk read MB/s", "disk_write_mean": "disk write MB/s", "paging_mean": "pages/s"}
        with open(os.path.join(out, "resources.md"), "w", encoding="utf-8") as f:
            f.write(f"Resources during the warm decode turns, mean over runs ({os.path.basename(d)}):\n\n")
            f.write("| metric | " + " | ".join(arms) + " |\n|---|" + "---|" * len(arms) + "\n")
            for m in metrics:
                row = [f"{np.mean(stats[a][m]):.2f}" if stats[a][m] else "-" for a in arms]
                if any(x != "-" for x in row):
                    f.write(f"| {names[m]} | " + " | ".join(row) + " |\n")
        shown = [m for m in metrics if any(stats[a][m] for a in arms)]
        cols = 4
        rows = (len(shown) + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 2.6 * rows))
        for ax, m in zip(np.atleast_1d(axes).ravel(), shown):
            vals = [np.mean(stats[a][m]) if stats[a][m] else 0 for a in arms]
            ax.bar(range(len(arms)), vals, color=[f"C{i}" for i in range(len(arms))])
            ax.set_xticks(range(len(arms)))
            ax.set_xticklabels(arms, rotation=30, fontsize=7)
            ax.set_title(names[m], fontsize=8)
            ax.grid(axis="y", alpha=0.3)
        for ax in np.atleast_1d(axes).ravel()[len(shown):]:
            ax.axis("off")
        fig.suptitle(f"{os.path.basename(d)}: resources over the warm decode turns", fontsize=10)
        fig.tight_layout()
        fig.savefig(os.path.join(out, "compare.png"), dpi=90)
        plt.close(fig)
        print(f"{d}: charts in {out}")


main()
