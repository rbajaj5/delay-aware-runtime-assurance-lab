"""Read-only tail audit for the delay random-matrix diagnostic.

The supplied excerpt separates a high-probability regular event from its
exceptional complement before taking expectations. This script applies the
same bookkeeping discipline descriptively to the retained finite-run CSV.
It does not claim an asymptotic tail bound.
"""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis" / "out" / "hex_delay_exploratory_20260730"
DETAIL = OUT / "delay_random_matrix_loop_games.csv"
SOURCE_MANIFEST = OUT / "DELAY_RANDOM_MATRIX_LOOP_MANIFEST.sha256"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_source() -> None:
    expected = None
    for line in SOURCE_MANIFEST.read_text(encoding="utf-8").splitlines():
        if line.endswith(f"*{DETAIL}"):
            expected = line.split(" *", 1)[0]
            break
    if expected is None or expected.lower() != sha256(DETAIL).lower():
        raise AssertionError(f"Input hash mismatch for {DETAIL}")


def read_rows() -> list[dict[str, float | int | str]]:
    with DETAIL.open("r", encoding="utf-8", newline="") as handle:
        return [
            {
                **row,
                "delay": int(row["delay"]),
                "mean_receipt_age": float(row["mean_receipt_age"]),
                "mean_loop_residual": float(row["mean_loop_residual"]),
                "max_disagreement": float(row["max_disagreement"]),
            }
            for row in csv.DictReader(handle)
        ]


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    verify_source()
    rows = read_rows()
    metrics = {
        "mean_receipt_age": "high_age_tail",
        "mean_loop_residual": "high_loop_tail",
        "max_disagreement": "high_disagreement_tail",
    }
    thresholds = {metric: float(np.percentile([row[metric] for row in rows], 95)) for metric in metrics}
    for row in rows:
        for metric, flag in metrics.items():
            row[flag] = int(row[metric] >= thresholds[metric])

    summary = []
    for delay in sorted({row["delay"] for row in rows}):
        for policy in sorted({row["policy"] for row in rows}):
            subset = [row for row in rows if row["delay"] == delay and row["policy"] == policy]
            output = {"delay": delay, "policy": policy, "n_games": len(subset)}
            for metric, flag in metrics.items():
                tail = [row[metric] for row in subset if row[flag]]
                output[f"{flag}_count"] = len(tail)
                output[f"{flag}_rate"] = len(tail) / len(subset)
                output[f"{flag}_conditional_mean"] = float(np.mean(tail)) if tail else 0.0
            summary.append(output)

    detail_path = OUT / "delay_tail_event_games.csv"
    summary_path = OUT / "delay_tail_event_summary.csv"
    write_csv(detail_path, rows)
    write_csv(summary_path, summary)

    colors = {"blind_delay": "#D55E00", "queue_aware": "#0072B2"}
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.8))
    for axis, (metric, flag) in zip(axes, metrics.items(), strict=True):
        for policy in ("blind_delay", "queue_aware"):
            selected = [row for row in summary if row["policy"] == policy]
            axis.plot([row["delay"] for row in selected], [row[f"{flag}_rate"] for row in selected], marker="o", linewidth=3, color=colors[policy], label=policy)
        axis.set_xlabel("Receipt delay (rounds)")
        axis.set_ylabel("95th-percentile tail rate")
        axis.set_title(flag.replace("_", " "))
        axis.set_ylim(-0.02, 1.02)
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    fig.suptitle("Finite-run exceptional-tail audit", fontsize=15, fontweight="bold")
    fig.text(0.01, 0.01, "Tail = observed global 95th percentile; descriptive only; input hash verified.", fontsize=8)
    fig.tight_layout(rect=[0, 0.05, 1, 0.94])
    figure = OUT / "delay_tail_event_audit.png"
    fig.savefig(figure, dpi=220, facecolor="white")
    plt.close(fig)

    report = OUT / "delay_tail_event_audit_report.md"
    threshold_lines = "\n".join(f"- `{metric}`: {thresholds[metric]:.17g}" for metric in metrics)
    report.write_text(
        "# Finite-Run Tail Event Audit\n\n"
        f"Input: `{DETAIL}`\n\n"
        "The supplied excerpt separates a regular event from its exceptional complement and shows that the complement can be controlled before taking expectations. This audit follows that bookkeeping pattern only: it defines empirical tail events from the retained finite-run data. It does not infer an exponential probability bound or claim the source theorem's hypotheses.\n\n"
        "## Data-defined thresholds\n\n"
        + threshold_lines
        + "\n\nTail membership is `value >= threshold`.\n\n"
        "| delay | policy | games | age tail rate | loop tail rate | disagreement tail rate |\n|---:|---|---:|---:|---:|---:|\n"
        + "\n".join(
            f"| {row['delay']} | {row['policy']} | {row['n_games']} | {row['high_age_tail_rate']:.17g} | {row['high_loop_tail_rate']:.17g} | {row['high_disagreement_tail_rate']:.17g} |"
            for row in summary
        )
        + "\n\nThe tail rates are descriptive diagnostics for separating ordinary and exceptional episodes. They are not hypothesis tests and should not be presented as asymptotic rare-event probabilities.\n",
        encoding="utf-8",
    )
    manifest = OUT / "DELAY_TAIL_EVENT_MANIFEST.sha256"
    manifest.write_text("\n".join(f"{sha256(path)} *{path}" for path in [DETAIL, detail_path, summary_path, figure, report, SOURCE_MANIFEST, Path(__file__)]) + "\n", encoding="utf-8")
    print(f"Rows: {len(rows)}")
    print(f"Summary: {summary_path}")
    print(f"Figure: {figure}")
    print(f"Report: {report}")
    print(f"Manifest: {manifest}")


if __name__ == "__main__":
    main()
