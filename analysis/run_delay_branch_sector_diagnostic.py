"""Exploratory branch-sector diagnostic for delayed complex phase messages.

The construction is inspired by the branch-factor transport argument in the
supplied excerpt from Bourgade and Huang. It measures whether delayed message
updates approach the real-axis branch cut or collapse pairwise separation.
It is a toy diagnostic, not a proof of the proposition and not a controller.
"""

from __future__ import annotations

import csv
import hashlib
import math
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis" / "out" / "hex_delay_exploratory_20260730"
PAPER_URL = "https://arxiv.org/abs/2607.07617"
AGENTS = 24
ROUNDS = 120
DELAYS = (0, 1, 2, 3, 5)
POLICIES = ("blind_delay", "queue_aware")
GAMES = 30
SEED_BASE = 2026073700
ALPHA = 0.68
PATH_SAMPLES = 31
SECTOR_EPS = 0.02


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def neighbors(agent: int) -> tuple[int, int]:
    return ((agent - 1) % AGENTS, (agent + 1) % AGENTS)


def transport_metrics(old: np.ndarray, new: np.ndarray, signs: np.ndarray) -> tuple[float, float, float, int]:
    min_modulus = math.inf
    min_pair = math.inf
    min_sector_margin = math.inf
    crossings = 0
    for sample in np.linspace(0.0, 1.0, PATH_SAMPLES):
        point = (1.0 - sample) * old + sample * new
        min_modulus = min(min_modulus, float(np.min(np.abs(point))))
        min_sector_margin = min(min_sector_margin, float(np.min(signs * point.imag)))
        if np.any(signs * point.imag <= SECTOR_EPS):
            crossings += 1
        distances = np.abs(point[:, None] - point[None, :])
        distances[np.diag_indices_from(distances)] = math.inf
        min_pair = min(min_pair, float(np.min(distances)))
    return min_modulus, min_sector_margin, min_pair, crossings


def simulate(delay: int, policy: str, seed: int) -> dict[str, float | int | str]:
    rng = random.Random(seed)
    signs = np.array([1.0 if index < AGENTS // 2 else -1.0 for index in range(AGENTS)])
    phases = np.array([signs[index] * (0.25 + 0.45 * rng.random()) for index in range(AGENTS)])
    radii = np.array([1.0 + 0.15 * rng.random() for _ in range(AGENTS)])
    state = radii * np.exp(1j * phases)
    latest = {(receiver, sender): state[sender] for receiver in range(AGENTS) for sender in neighbors(receiver)}
    latest_sent = {(receiver, sender): 0 for receiver in range(AGENTS) for sender in neighbors(receiver)}
    pending: list[tuple[int, int, int, complex, int]] = []
    modulus_values: list[float] = []
    sector_values: list[float] = []
    pair_values: list[float] = []
    crossings = 0

    for round_index in range(ROUNDS):
        due = [item for item in pending if item[0] <= round_index]
        pending = [item for item in pending if item[0] > round_index]
        for _, receiver, sender, payload, sent_round in due:
            if sent_round >= latest_sent[(receiver, sender)]:
                latest[(receiver, sender)] = payload
                latest_sent[(receiver, sender)] = sent_round

        visible = {}
        for receiver in range(AGENTS):
            for sender in neighbors(receiver):
                choice = (latest[(receiver, sender)], latest_sent[(receiver, sender)])
                if policy == "queue_aware":
                    queued = [item for item in pending if item[1] == receiver and item[2] == sender]
                    if queued:
                        newest = max(queued, key=lambda item: item[4])
                        if newest[4] >= choice[1]:
                            choice = (newest[3], newest[4])
                visible[(receiver, sender)] = choice[0]

        target = np.zeros(AGENTS, dtype=complex)
        for receiver in range(AGENTS):
            left, right = neighbors(receiver)
            target[receiver] = 0.5 * (visible[(receiver, left)] + visible[(receiver, right)])
        drive = np.array(
            [0.04j * signs[index] * math.sin(2.0 * math.pi * round_index / 37.0 + index / 5.0) for index in range(AGENTS)],
            dtype=complex,
        )
        next_state = state + ALPHA * (target - state) + drive
        min_modulus, min_sector, min_pair, path_crossings = transport_metrics(state, next_state, signs)
        modulus_values.append(min_modulus)
        sector_values.append(min_sector)
        pair_values.append(min_pair)
        crossings += path_crossings
        for sender in range(AGENTS):
            for receiver in neighbors(sender):
                pending.append((round_index + delay, receiver, sender, complex(next_state[sender]), round_index + 1))
        state = next_state

    return {
        "delay": delay,
        "policy": policy,
        "seed": seed,
        "agents": AGENTS,
        "rounds": ROUNDS,
        "mean_min_modulus": float(np.mean(modulus_values)),
        "min_modulus": float(np.min(modulus_values)),
        "mean_sector_margin": float(np.mean(sector_values)),
        "min_sector_margin": float(np.min(sector_values)),
        "mean_pair_separation": float(np.mean(pair_values)),
        "min_pair_separation": float(np.min(pair_values)),
        "branch_crossing_samples": crossings,
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for delay in DELAYS:
        for index in range(GAMES):
            seed = SEED_BASE + delay * 10000 + index
            for policy in POLICIES:
                rows.append(simulate(delay, policy, seed))

    summary = []
    for delay in DELAYS:
        for policy in POLICIES:
            subset = [row for row in rows if row["delay"] == delay and row["policy"] == policy]
            summary.append({
                "delay": delay,
                "policy": policy,
                "n_games": len(subset),
                "mean_min_modulus": sum(row["mean_min_modulus"] for row in subset) / len(subset),
                "mean_sector_margin": sum(row["mean_sector_margin"] for row in subset) / len(subset),
                "mean_pair_separation": sum(row["mean_pair_separation"] for row in subset) / len(subset),
                "mean_branch_crossing_samples": sum(row["branch_crossing_samples"] for row in subset) / len(subset),
                "min_modulus_across_games": min(row["min_modulus"] for row in subset),
                "min_sector_margin_across_games": min(row["min_sector_margin"] for row in subset),
                "min_pair_separation_across_games": min(row["min_pair_separation"] for row in subset),
            })

    detail = OUT / "delay_branch_sector_games.csv"
    summary_path = OUT / "delay_branch_sector_summary.csv"
    write_csv(detail, rows)
    write_csv(summary_path, summary)

    colors = {"blind_delay": "#D55E00", "queue_aware": "#0072B2"}
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.8))
    for policy in POLICIES:
        selected = [row for row in summary if row["policy"] == policy]
        label = "blind delay" if policy == "blind_delay" else "queue-aware"
        axes[0].plot([row["delay"] for row in selected], [row["mean_min_modulus"] for row in selected], marker="o", linewidth=3, color=colors[policy], label=label)
        axes[1].plot([row["delay"] for row in selected], [row["mean_sector_margin"] for row in selected], marker="o", linewidth=3, color=colors[policy], label=label)
        axes[2].plot([row["delay"] for row in selected], [row["mean_pair_separation"] for row in selected], marker="o", linewidth=3, color=colors[policy], label=label)
    axes[0].set_title("Distance from branch point")
    axes[0].set_ylabel("Mean minimum |z|")
    axes[1].axhline(SECTOR_EPS, color="#333333", linestyle="--", linewidth=1.5, label="sector tolerance")
    axes[1].set_title("Half-plane sector margin")
    axes[1].set_ylabel("Mean min sigma Im(z)")
    axes[2].set_title("Pairwise separation")
    axes[2].set_ylabel("Mean minimum |z_i-z_j|")
    for axis in axes:
        axis.set_xlabel("Receipt delay (rounds)")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    fig.suptitle("Branch-sector transport under delayed phase messages", fontsize=15, fontweight="bold")
    fig.text(0.01, 0.01, f"24-agent complex phase surrogate; {GAMES} games/cell; exploratory only; source excerpt: arXiv:2607.07617.", fontsize=8)
    fig.tight_layout(rect=[0, 0.05, 1, 0.94])
    figure = OUT / "delay_branch_sector_diagnostic.png"
    fig.savefig(figure, dpi=220, facecolor="white")
    plt.close(fig)

    report = OUT / "delay_branch_sector_diagnostic_report.md"
    report.write_text(
        "# Delayed Branch-Sector Diagnostic\n\n"
        f"This experiment is inspired by the branch-factor transport argument in the supplied excerpt from [Bourgade and Huang, *Loop Equations Characterize Random Matrix Statistics*]({PAPER_URL}). The excerpt constructs paths that remain away from the real axis and keep pairwise variables separated.\n\n"
        "Each agent carries a complex phase surrogate with a fixed sign sector. A linear interpolation between consecutive delayed updates is sampled at 31 points. We record minimum modulus, signed half-plane margin, minimum pairwise separation, and samples that enter the sector tolerance. `blind_delay` uses the latest delivered message; `queue_aware` overlays the newest pending payload.\n\n"
        "This is not a reproduction of Proposition 8.4: no branch-factor solution is evaluated, and no theorem assumptions are asserted. The quantities are an engineering diagnostic for branch-cut proximity and phase collision under delayed receipts.\n\n"
        "| delay | policy | games | mean min modulus | mean sector margin | mean pair separation | mean crossing samples | worst modulus | worst sector margin | worst pair separation |\n|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|\n"
        + "\n".join(
            f"| {row['delay']} | {row['policy']} | {row['n_games']} | {row['mean_min_modulus']:.17g} | {row['mean_sector_margin']:.17g} | {row['mean_pair_separation']:.17g} | {row['mean_branch_crossing_samples']:.17g} | {row['min_modulus_across_games']:.17g} | {row['min_sector_margin_across_games']:.17g} | {row['min_pair_separation_across_games']:.17g} |"
            for row in summary
        )
        + "\n\nThe result should be read as a branch-geometry diagnostic only. A follow-up could add a sector-preserving projection and test whether it reduces branch crossings without hiding state error.\n",
        encoding="utf-8",
    )
    manifest = OUT / "DELAY_BRANCH_SECTOR_MANIFEST.sha256"
    manifest.write_text("\n".join(f"{sha256(path)} *{path}" for path in [detail, summary_path, figure, report, Path(__file__)]) + "\n", encoding="utf-8")
    print(f"Games: {len(rows)}")
    print(f"Summary: {summary_path}")
    print(f"Figure: {figure}")
    print(f"Report: {report}")
    print(f"Manifest: {manifest}")


if __name__ == "__main__":
    main()
