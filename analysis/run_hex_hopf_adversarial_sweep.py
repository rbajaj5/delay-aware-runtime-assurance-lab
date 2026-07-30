"""Sweep the strength of the deliberately selective Hopf-fiber message hold."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from run_hex_hopf_delay_simulation import DELAYS, N, OUT, POLICIES, sha256, simulate_game


HOLD_STRENGTHS = (0, 1, 2)
GAMES = 15
SEED_BASE = 2026073200


def write_csv(path: Path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for hold_strength in HOLD_STRENGTHS:
        for delay in DELAYS:
            for index in range(GAMES):
                seed = SEED_BASE + hold_strength * 100000 + delay * 10000 + index
                for policy in POLICIES:
                    result = simulate_game(delay, policy, seed, hold_strength=hold_strength)
                    rows.append({"hold_strength": hold_strength, **{key: value for key, value in result.items() if key != "trace"}})

    summary = []
    for hold_strength in HOLD_STRENGTHS:
        for delay in DELAYS:
            for policy in POLICIES:
                subset = [row for row in rows if row["hold_strength"] == hold_strength and row["delay"] == delay and row["policy"] == policy]
                summary.append({
                    "hold_strength": hold_strength,
                    "delay": delay,
                    "policy": policy,
                    "n_games": len(subset),
                    "mean_stale_conflicts": sum(float(row["stale_conflicts"]) for row in subset) / len(subset),
                    "any_conflict_rate": sum(int(row["stale_conflicts"]) > 0 for row in subset) / len(subset),
                    "mean_phase_error": sum(float(row["mean_phase_error"]) for row in subset) / len(subset),
                    "mean_extra_hold": sum(float(row["mean_extra_hold"]) for row in subset) / len(subset),
                })

    detail = OUT / "hex_hopf_adversarial_sweep_games.csv"
    summary_path = OUT / "hex_hopf_adversarial_sweep_summary.csv"
    write_csv(detail, rows)
    write_csv(summary_path, summary)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
    colors = {0: "#009E73", 1: "#D55E00", 2: "#CC79A7"}
    for hold_strength in HOLD_STRENGTHS:
        for policy in POLICIES:
            selected = [row for row in summary if row["hold_strength"] == hold_strength and row["policy"] == policy]
            label = f"{policy}, hold={hold_strength}"
            axes[0].plot([row["delay"] for row in selected], [row["mean_stale_conflicts"] for row in selected], marker="o", linewidth=2.5, color=colors[hold_strength], linestyle="-" if policy == "blind_delay" else "--", label=label)
            axes[1].plot([row["delay"] for row in selected], [row["mean_phase_error"] for row in selected], marker="o", linewidth=2.5, color=colors[hold_strength], linestyle="-" if policy == "blind_delay" else "--", label=label)
    axes[0].set_title("Stale conflicts")
    axes[0].set_ylabel("Mean conflicts per game")
    axes[1].set_title("Hidden fiber phase error")
    axes[1].set_ylabel("Mean circular phase error (radians)")
    for axis in axes:
        axis.set_xlabel("Base message delay (moves)")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    fig.suptitle("Adversarial selective-hold strength sweep", fontsize=15, fontweight="bold")
    fig.text(0.01, 0.01, f"{N}x{N} Hex; {GAMES} games per cell; exploratory toy Hopf-fiber transport; not v6 evidence.", fontsize=8)
    fig.tight_layout(rect=[0, 0.04, 1, 0.95])
    figure = OUT / "hex_hopf_adversarial_sweep.png"
    fig.savefig(figure, dpi=220, facecolor="white")
    plt.close(fig)

    report_lines = [
        "# Hopf-Fiber Adversarial Hold Sweep",
        "",
        f"The selective hold strength was swept over `{list(HOLD_STRENGTHS)}` extra steps. The hold predicate is deterministic in payload phase and cell coordinates. `blind_delay` ignores pending messages; `queue_aware` overlays them.",
        "",
        "| hold strength | base delay | policy | games | mean conflicts | any-conflict rate | mean phase error | mean extra hold |",
        "|---:|---:|---|---:|---:|---:|---:|---:|",
    ]
    report_lines.extend(f"| {row['hold_strength']} | {row['delay']} | {row['policy']} | {row['n_games']} | {row['mean_stale_conflicts']:.17g} | {row['any_conflict_rate']:.17g} | {row['mean_phase_error']:.17g} | {row['mean_extra_hold']:.17g} |" for row in summary)
    report_lines.extend(["", "This is a deliberately adversarial communication toy. The Hopf coordinates provide a hidden fiber-phase state; they do not establish the external paper's proposed physical theory. No v6 claim is updated."])
    report = OUT / "hex_hopf_adversarial_sweep_report.md"
    report.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    manifest = OUT / "HEX_HOPF_ADVERSARIAL_SWEEP_MANIFEST.sha256"
    manifest.write_text("\n".join(f"{sha256(path)} *{path}" for path in [detail, summary_path, figure, report, Path(__file__)]) + "\n", encoding="utf-8")
    print(f"Games: {len(rows)}")
    print(f"Summary: {summary_path}")
    print(f"Figure: {figure}")
    print(f"Manifest: {manifest}")


if __name__ == "__main__":
    main()
