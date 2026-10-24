#!/usr/bin/env python3
"""Mechanistic census over every retained v6 seed and step row.

Read-only. This analyzes the six frozen v6 step logs; it does not rerun the
shield, solve a QP, or alter any confirmation artifact.
"""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analyze_v6_anchor_order_gap import ARMS, BLOCKS, read_log, sha256


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis" / "out" / "v6_all_seed_mechanism_20260729"


def finite(value: object) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def mean_or_nan(values: list[float]) -> float:
    return float(np.mean(values)) if values else math.nan


def quantile_or_nan(values: list[float], q: float) -> float:
    return float(np.quantile(values, q)) if values else math.nan


def correlation(rows: list[dict[str, object]], x_key: str, y_key: str) -> float:
    pairs = [
        (float(row[x_key]), float(row[y_key]))
        for row in rows
        if finite(row[x_key]) and finite(row[y_key])
    ]
    if len(pairs) < 2:
        return math.nan
    x, y = np.asarray(pairs, dtype=float).T
    if np.std(x) == 0 or np.std(y) == 0:
        return math.nan
    return float(np.corrcoef(x, y)[0, 1])


def episode_summary(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, str, int], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["block"]), str(row["arm"]), int(row["seed"]))].append(row)

    output: list[dict[str, object]] = []
    for (block, arm, seed), items in sorted(groups.items()):
        items.sort(key=lambda item: int(item["step"]))
        gaps = [float(item["anchor_gap_l2"]) for item in items]
        queues = [float(item["formal_queue_len_before"]) for item in items if finite(item["formal_queue_len_before"])]
        fallback_items = [item for item in items if item["formal_fallback_used"] is True]
        severity = [float(item["normalized_violation_severity"]) for item in items if finite(item["normalized_violation_severity"])]
        first_fallback = min((int(item["step"]) for item in fallback_items), default=None)
        last_fallback = max((int(item["step"]) for item in fallback_items), default=None)
        pre = [item for item in items if first_fallback is not None and int(item["step"]) < first_fallback]
        on_fallback = fallback_items
        post = [item for item in items if last_fallback is not None and int(item["step"]) > last_fallback]

        def gap_values(selected: list[dict[str, object]]) -> list[float]:
            return [float(item["anchor_gap_l2"]) for item in selected]

        q_counts = {str(q): sum(1 for item in items if float(item["formal_queue_len_before"]) == q) for q in (0.0, 1.0, 2.0)}
        output.append(
            {
                "block": block,
                "arm": arm,
                "seed": seed,
                "n_rows": len(items),
                "anchor_gap_positive_fraction": float(np.mean(np.asarray(gaps) > 1e-12)),
                "anchor_gap_mean": mean_or_nan(gaps),
                "anchor_gap_p95": quantile_or_nan(gaps, 0.95),
                "anchor_gap_max": float(np.max(gaps)),
                "queue_mean": mean_or_nan(queues),
                "queue_nonzero_fraction": float(np.mean(np.asarray(queues) > 0)) if queues else math.nan,
                "queue_max": float(np.max(queues)) if queues else math.nan,
                "queue0_rows": q_counts["0.0"],
                "queue1_rows": q_counts["1.0"],
                "queue2_rows": q_counts["2.0"],
                "fallback_count": len(fallback_items),
                "fallback_rate": len(fallback_items) / len(items),
                "severity_sum": float(np.sum(severity)) if severity else 0.0,
                "severity_max": float(np.max(severity)) if severity else 0.0,
                "first_fallback_step": first_fallback if first_fallback is not None else "",
                "last_fallback_step": last_fallback if last_fallback is not None else "",
                "pre_first_fallback_rows": len(pre),
                "pre_first_fallback_gap_mean": mean_or_nan(gap_values(pre)),
                "pre_first_fallback_positive_fraction": float(np.mean(np.asarray(gap_values(pre)) > 1e-12)) if pre else math.nan,
                "fallback_row_gap_mean": mean_or_nan(gap_values(on_fallback)),
                "post_last_fallback_gap_mean": mean_or_nan(gap_values(post)),
                "gap_positive_before_fallback": bool(pre and any(float(item["anchor_gap_l2"]) > 1e-12 for item in pre)),
            }
        )
    return output


def block_summary(episodes: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in episodes:
        groups[(str(row["block"]), str(row["arm"]))].append(row)
    output: list[dict[str, object]] = []
    for (block, arm), rows in sorted(groups.items()):
        with_fallback = [row for row in rows if int(row["fallback_count"]) > 0]
        pre_positive = [row for row in with_fallback if row["gap_positive_before_fallback"]]
        output.append(
            {
                "block": block,
                "arm": arm,
                "n_episodes": len(rows),
                "seed_min": min(int(row["seed"]) for row in rows),
                "seed_max": max(int(row["seed"]) for row in rows),
                "episodes_with_positive_gap": sum(float(row["anchor_gap_positive_fraction"]) > 0 for row in rows),
                "episodes_with_fallback": len(with_fallback),
                "episodes_positive_gap_without_fallback": sum(int(row["fallback_count"]) == 0 and float(row["anchor_gap_positive_fraction"]) > 0 for row in rows),
                "mean_episode_gap": float(np.mean([float(row["anchor_gap_mean"]) for row in rows])),
                "p95_episode_gap_mean": float(np.quantile([float(row["anchor_gap_mean"]) for row in rows], 0.95)),
                "max_episode_gap": float(np.max([float(row["anchor_gap_max"]) for row in rows])),
                "mean_fallback_rate": float(np.mean([float(row["fallback_rate"]) for row in rows])),
                "mean_severity_sum": float(np.mean([float(row["severity_sum"]) for row in rows])),
                "pre_fallback_positive_fraction": float(len(pre_positive) / len(with_fallback)) if with_fallback else math.nan,
                "corr_gap_mean_vs_fallback_rate": correlation(rows, "anchor_gap_mean", "fallback_rate"),
                "corr_gap_mean_vs_severity_sum": correlation(rows, "anchor_gap_mean", "severity_sum"),
                "corr_queue_mean_vs_fallback_rate": correlation(rows, "queue_mean", "fallback_rate"),
                "corr_queue_mean_vs_gap_mean": correlation(rows, "queue_mean", "anchor_gap_mean"),
            }
        )
    return output


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise AssertionError(f"No rows to write: {path}")
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot(episodes: list[dict[str, object]]) -> list[Path]:
    colors = {"F-comp": "#0072B2", "F-blind": "#D55E00"}
    outputs: list[Path] = []
    for block in BLOCKS:
        fig, axis = plt.subplots(figsize=(7.2, 5.0))
        for arm in ARMS:
            selected = [row for row in episodes if row["block"] == block and row["arm"] == arm]
            axis.scatter(
                [float(row["anchor_gap_mean"]) for row in selected],
                [float(row["fallback_rate"]) for row in selected],
                s=12,
                alpha=0.32,
                color=colors[arm],
                label=arm,
            )
        axis.set_xlabel("Episode mean anchor gap L2")
        axis.set_ylabel("Fallback rate")
        axis.set_title(f"All retained seeds: {block.replace('_', ' ')}")
        axis.grid(alpha=0.25)
        axis.legend()
        fig.text(0.01, 0.01, "6000 retained episodes across the three blocks; descriptive association only.", fontsize=8)
        fig.tight_layout(rect=[0, 0.04, 1, 1])
        path = OUT / f"{block}_anchor_gap_vs_fallback.png"
        fig.savefig(path, dpi=220, facecolor="white")
        fig.savefig(OUT / f"{block}_anchor_gap_vs_fallback.svg", facecolor="white")
        plt.close(fig)
        outputs.extend([path, OUT / f"{block}_anchor_gap_vs_fallback.svg"])
    return outputs


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    source_hashes: dict[str, str] = {}
    all_rows: list[dict[str, object]] = []
    for block, block_root in BLOCKS.items():
        manifest = block_root / "BLOCK_MANIFEST.sha256"
        source_hashes[str(manifest)] = sha256(manifest)
        for arm in ARMS:
            path = block_root / "full_steps" / f"{arm}_steps.csv"
            digest, rows = read_log(path, block, arm, manifest)
            source_hashes[str(path)] = digest
            all_rows.extend(rows)
    episodes = episode_summary(all_rows)
    blocks = block_summary(episodes)
    episode_path = OUT / "v6_all_seed_mechanism_episode.csv"
    block_path = OUT / "v6_all_seed_mechanism_summary.csv"
    write_csv(episode_path, episodes)
    write_csv(block_path, blocks)
    figure_paths = plot(episodes)

    report_lines = [
        "# v6 All-Seed Mechanism Census",
        "",
        "This read-only analysis covers every retained v6 episode: 6,000 episodes and 600,000 comparable step rows across F-comp and F-blind in the uniform, gradual, and persistent blocks. It measures whether the logged queue-anchor displacement is present before fallback and whether episode-level displacement tracks fallback or severity. These are descriptive associations, not causal or stability claims.",
        "",
        "## Definitions",
        "",
        "- Anchor gap: `||formal_barrier_anchor_state - formal_state_at_decision||_2`.",
        "- Pre-fallback gap: rows strictly before the first fallback row in an episode.",
        "- Fallback-row gap: rows where `formal_fallback_used` is true.",
        "- Episode correlations are Pearson correlations over the 1,000 retained seeds within each block/arm cell.",
        "",
        "## All-seed results",
        "",
        "| block | arm | episodes | positive-gap episodes | positive-gap episodes without fallback | fallback episodes | mean episode gap | mean fallback rate | mean severity sum | pre-fallback positive fraction | gap/fallback r | gap/severity r | queue/fallback r | queue/gap r |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in blocks:
        report_lines.append(
            f"| {row['block']} | {row['arm']} | {row['n_episodes']} | {row['episodes_with_positive_gap']} | {row['episodes_positive_gap_without_fallback']} | {row['episodes_with_fallback']} | {float(row['mean_episode_gap']):.17g} | {float(row['mean_fallback_rate']):.17g} | {float(row['mean_severity_sum']):.17g} | {float(row['pre_fallback_positive_fraction']):.17g} | {float(row['corr_gap_mean_vs_fallback_rate']):.17g} | {float(row['corr_gap_mean_vs_severity_sum']):.17g} | {float(row['corr_queue_mean_vs_fallback_rate']):.17g} | {float(row['corr_queue_mean_vs_gap_mean']):.17g} |"
        )
    report_lines.extend([
        "",
        "## Mechanistic reading boundary",
        "",
        "The decisive temporal check is whether a gap exists before fallback. A positive pre-fallback gap shows that the anchor/current-state discrepancy is not merely a by-product of entering the fallback branch. Correlations then quantify whether the discrepancy tracks response intensity within these frozen cells; they do not identify a causal direction. The multi-agent proxy remains a separate exploratory system: its worlds are not the v6 registered seeds, so its matrix results must not be merged with this single-vehicle census.",
        "",
        "## Input provenance",
        "",
    ])
    report_lines.extend(f"- `{path}`: `{digest}`" for path, digest in sorted(source_hashes.items()))
    report_path = OUT / "v6_all_seed_mechanism_report.md"
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    outputs = [episode_path, block_path, report_path, *figure_paths]
    manifest_lines = [f"{digest} *{path}" for path, digest in sorted(source_hashes.items())]
    manifest_lines.extend(f"{sha256(path)} *{path}" for path in outputs)
    manifest_path = OUT / "V6_ALL_SEED_MECHANISM_MANIFEST.sha256"
    manifest_path.write_text("\n".join(manifest_lines) + "\n", encoding="utf-8")
    print("v6 all-seed mechanism census complete")
    print(f"Compared rows: {len(all_rows)}")
    print(f"Episode rows: {len(episodes)}")
    print(f"Output: {OUT}")


if __name__ == "__main__":
    main()
