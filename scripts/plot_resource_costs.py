"""Plot paired exact/checkpoint ratios from retained resource observations."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("--shape", default="1024x2048")
    args = parser.parse_args()
    rows = [json.loads(line) for line in (args.results / "observations.jsonl").read_text().splitlines()]
    rows = [r for r in rows if r["shape"] == args.shape and r["status"] == "ok"]
    checkpoints = {(r["fixture_id"], r["repeat"]): r for r in rows if r["method"] == "checkpoint"}
    precisions = ("float32", "float16", "bfloat16")
    metrics = (("Serialized record size", "build", "serialized_bytes"),
               ("Record construction + encoding time", "build", "total_seconds"),
               ("Decoding + restoration time", "undo", "total_seconds"),
               ("Undo process peak RSS", "undo", "process_peak_rss_bytes"))
    colors = ("#2274a5", "#c77727", "#7954a1")
    with plt.rc_context({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False}):
        fig, axes = plt.subplots(2, 2, figsize=(10, 7))
        for axis, (title, phase, metric) in zip(axes.flat, metrics):
            values = []
            for precision in precisions:
                sample = [r[phase][metric] / checkpoints[(r["fixture_id"], r["repeat"])][phase][metric]
                          for r in rows if r["precision"] == precision and r["method"] == "delta_exact"
                          and (r["fixture_id"], r["repeat"]) in checkpoints]
                if not sample:
                    raise ValueError(f"No paired observations for {precision}")
                values.append(sample)
            medians = np.array([np.median(v) for v in values])
            errors = np.array([[m - min(v) for m, v in zip(medians, values)],
                               [max(v) - m for m, v in zip(medians, values)]])
            axis.bar(range(3), medians, color=colors, width=0.55, alpha=0.8)
            axis.errorbar(range(3), medians, yerr=errors, fmt="none", color="#333333", capsize=4)
            axis.axhline(1, color="#666666", linestyle="--", linewidth=1)
            axis.set_xticks(range(3), ("FP32", "FP16", "BF16 (emulated)"))
            axis.set_title(title, loc="left", weight="bold", fontsize=11)
            axis.set_ylabel("Exact / checkpoint")
            axis.set_ylim(0, max(1.1, max(max(v) for v in values) * 1.25))
            axis.grid(axis="y", alpha=0.16)
            axis.set_axisbelow(True)
            for i, (median, sample) in enumerate(zip(medians, values)):
                axis.annotate(f"{median:.2f}×", (i, max(sample)), xytext=(0, 6),
                              textcoords="offset points", ha="center", fontsize=10)
        fig.suptitle(f"Exact rollback resource tradeoff · {args.shape}", fontsize=16, weight="bold", y=.99)
        fig.text(.5, .945, "Synthetic ROME · alpha 0.95 · 3 seeds × 3 repeats · fresh processes", ha="center", color="#555555")
        fig.text(.5, .02, "Bars: paired medians; whiskers: min–max (not confidence intervals). Dashed line: compressed checkpoint.\n"
                 "RSS includes imports and input preparation. CPU NumPy storage emulation; no real-model or GPU inference.",
                 ha="center", fontsize=9, color="#555555")
        fig.tight_layout(rect=(0, .075, 1, .93))
        fig.savefig(args.results / "resource_tradeoff.png", dpi=180)
        plt.close(fig)


if __name__ == "__main__":
    main()
