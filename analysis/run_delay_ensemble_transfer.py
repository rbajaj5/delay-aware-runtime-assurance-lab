"""Topology-transfer check for the delayed random-matrix diagnostic.

Inspired by the paper's universality program and its random-regular-graph
application. This is a finite descriptive comparison, not a universality
proof: ring, random 4-regular, and two-block graphs are simulated with the
same delayed-message policies and their age-matrix observables are compared.
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
TOPOLOGIES = ("ring", "random_4_regular", "two_block")
GAMES = 15
SEED_BASE = 2026073800
ALPHA = 0.82
RESOLVENT_Z = 2.0 + 0.5j


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ring_graph() -> dict[int, list[int]]:
    return {i: sorted({(i - 1) % AGENTS, (i + 1) % AGENTS}) for i in range(AGENTS)}


def random_regular_graph(rng: random.Random, degree: int = 4) -> dict[int, list[int]]:
    for _ in range(500):
        stubs = [node for node in range(AGENTS) for _ in range(degree)]
        rng.shuffle(stubs)
        edges: set[tuple[int, int]] = set()
        valid = True
        for index in range(0, len(stubs), 2):
            a, b = stubs[index], stubs[index + 1]
            edge = tuple(sorted((a, b)))
            if a == b or edge in edges:
                valid = False
                break
            edges.add(edge)
        if valid:
            graph = {i: set() for i in range(AGENTS)}
            for a, b in edges:
                graph[a].add(b)
                graph[b].add(a)
            if all(len(graph[i]) == degree for i in graph):
                return {i: sorted(graph[i]) for i in graph}
    raise RuntimeError("Could not generate a simple random regular graph")


def two_block_graph() -> dict[int, list[int]]:
    half = AGENTS // 2
    graph = {i: set() for i in range(AGENTS)}
    for block_start in (0, half):
        for offset in range(half):
            node = block_start + offset
            for shift in (-1, 1, -2, 2):
                graph[node].add(block_start + ((offset + shift) % half))
            graph[node].add(half + offset if block_start == 0 else offset)
    return {i: sorted(graph[i]) for i in graph}


def graph_for(name: str, rng: random.Random) -> dict[int, list[int]]:
    if name == "ring":
        return ring_graph()
    if name == "random_4_regular":
        return random_regular_graph(rng)
    return two_block_graph()


def matrix_from_ages(ages: dict[tuple[int, int], float]) -> np.ndarray:
    raw = np.zeros((AGENTS, AGENTS), dtype=float)
    for (receiver, sender), age in ages.items():
        raw[receiver, sender] = age
    raw = 0.5 * (raw + raw.T)
    values = raw[np.triu_indices(AGENTS, k=1)]
    std = float(np.std(values))
    if std <= 1e-12:
        return np.zeros_like(raw)
    centered = (raw - float(np.mean(values))) / std
    np.fill_diagonal(centered, 0.0)
    return centered / math.sqrt(AGENTS)


def matrix_metrics(matrix: np.ndarray) -> tuple[float, float]:
    eigenvalues = np.linalg.eigvalsh(matrix)
    z = RESOLVENT_Z
    resolvent = np.linalg.inv(matrix.astype(complex) - z * np.eye(AGENTS, dtype=complex))
    m = np.trace(resolvent) / AGENTS
    residual = abs(m * m + z * m + 1.0)
    return float(eigenvalues[-1]), float(residual)


def simulate(topology: str, delay: int, policy: str, seed: int) -> dict[str, float | int | str]:
    rng = random.Random(seed)
    graph = graph_for(topology, rng)
    phase = rng.random() * 2.0 * math.pi
    state = np.array([0.45 * math.sin(2.0 * math.pi * i / AGENTS + phase) + 0.08 * rng.gauss(0.0, 1.0) for i in range(AGENTS)], dtype=float)
    latest = {(receiver, sender): state[sender] for receiver in range(AGENTS) for sender in graph[receiver]}
    latest_sent = {(receiver, sender): 0 for receiver in range(AGENTS) for sender in graph[receiver]}
    pending: list[tuple[int, int, int, float, int]] = []
    edges: list[float] = []
    residuals: list[float] = []
    ages: list[float] = []
    disagreements: list[float] = []

    for round_index in range(ROUNDS):
        due = [item for item in pending if item[0] <= round_index]
        pending = [item for item in pending if item[0] > round_index]
        for _, receiver, sender, payload, sent_round in due:
            if sent_round >= latest_sent[(receiver, sender)]:
                latest[(receiver, sender)] = payload
                latest_sent[(receiver, sender)] = sent_round

        visible = {}
        visible_ages = {}
        for receiver in range(AGENTS):
            for sender in graph[receiver]:
                choice = (latest[(receiver, sender)], latest_sent[(receiver, sender)])
                if policy == "queue_aware":
                    queued = [item for item in pending if item[1] == receiver and item[2] == sender]
                    if queued:
                        newest = max(queued, key=lambda item: item[4])
                        if newest[4] >= choice[1]:
                            choice = (newest[3], newest[4])
                visible[(receiver, sender)] = choice[0]
                visible_ages[(receiver, sender)] = round_index - choice[1]

        edge, residual = matrix_metrics(matrix_from_ages(visible_ages))
        edges.append(edge)
        residuals.append(residual)
        ages.append(float(np.mean(list(visible_ages.values()))))
        disagreements.append(float(np.std(state)))

        next_state = np.zeros(AGENTS, dtype=float)
        for receiver in range(AGENTS):
            target = sum(visible[(receiver, sender)] for sender in graph[receiver]) / len(graph[receiver])
            drive = 0.10 * math.sin(2.0 * math.pi * round_index / 45.0 + 0.12 * receiver + phase)
            next_state[receiver] = state[receiver] + ALPHA * (target - state[receiver]) + drive
        for sender in range(AGENTS):
            for receiver in graph[sender]:
                pending.append((round_index + delay, receiver, sender, float(next_state[sender]), round_index + 1))
        state = next_state

    return {
        "topology": topology,
        "delay": delay,
        "policy": policy,
        "seed": seed,
        "agents": AGENTS,
        "rounds": ROUNDS,
        "mean_spectral_edge": float(np.mean(edges)),
        "mean_loop_residual": float(np.mean(residuals)),
        "mean_receipt_age": float(np.mean(ages)),
        "mean_disagreement": float(np.mean(disagreements)),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for topology in TOPOLOGIES:
        for delay in DELAYS:
            for index in range(GAMES):
                seed = SEED_BASE + TOPOLOGIES.index(topology) * 100000 + delay * 10000 + index
                for policy in POLICIES:
                    rows.append(simulate(topology, delay, policy, seed))

    summary = []
    for topology in TOPOLOGIES:
        for delay in DELAYS:
            for policy in POLICIES:
                subset = [row for row in rows if row["topology"] == topology and row["delay"] == delay and row["policy"] == policy]
                summary.append({
                    "topology": topology,
                    "delay": delay,
                    "policy": policy,
                    "n_games": len(subset),
                    "mean_spectral_edge": sum(row["mean_spectral_edge"] for row in subset) / len(subset),
                    "mean_loop_residual": sum(row["mean_loop_residual"] for row in subset) / len(subset),
                    "mean_receipt_age": sum(row["mean_receipt_age"] for row in subset) / len(subset),
                    "mean_disagreement": sum(row["mean_disagreement"] for row in subset) / len(subset),
                })

    detail = OUT / "delay_ensemble_transfer_games.csv"
    summary_path = OUT / "delay_ensemble_transfer_summary.csv"
    write_csv(detail, rows)
    write_csv(summary_path, summary)

    colors = {"blind_delay": "#D55E00", "queue_aware": "#0072B2"}
    fig, axes = plt.subplots(3, 3, figsize=(15.0, 12.0), sharex="col")
    metrics = (("mean_spectral_edge", "Spectral edge"), ("mean_loop_residual", "Loop residual"), ("mean_disagreement", "Disagreement"))
    for row_index, topology in enumerate(TOPOLOGIES):
        for col_index, (metric, title) in enumerate(metrics):
            axis = axes[row_index, col_index]
            for policy in POLICIES:
                selected = [item for item in summary if item["topology"] == topology and item["policy"] == policy]
                label = "blind" if policy == "blind_delay" else "queue-aware"
                axis.plot([item["delay"] for item in selected], [item[metric] for item in selected], marker="o", linewidth=2.5, color=colors[policy], label=label)
            if row_index == 0:
                axis.set_title(title)
            if col_index == 0:
                axis.set_ylabel(topology.replace("_", " ") + "\nvalue")
            axis.grid(alpha=0.25)
            if row_index == len(TOPOLOGIES) - 1:
                axis.set_xlabel("Receipt delay (rounds)")
            if row_index == 0 and col_index == 0:
                axis.legend(fontsize=8)
    fig.suptitle("Topology transfer of delayed loop diagnostics", fontsize=16, fontweight="bold")
    fig.text(0.01, 0.01, f"24 agents; {GAMES} games/cell; ring, random 4-regular, two-block; exploratory only.", fontsize=8)
    fig.tight_layout(rect=[0, 0.03, 1, 0.96])
    figure = OUT / "delay_ensemble_transfer.png"
    fig.savefig(figure, dpi=220, facecolor="white")
    plt.close(fig)

    report = OUT / "delay_ensemble_transfer_report.md"
    report.write_text(
        "# Delayed Loop-Diagnostic Topology Transfer\n\n"
        f"Conceptual source: [Bourgade and Huang, *Loop Equations Characterize Random Matrix Statistics*]({PAPER_URL}). The paper's universality program asks when local spectral observables persist across different underlying ensembles; its applications include random regular graphs.\n\n"
        "Here the same delayed-message dynamics are run on a ring, a deterministic two-block graph, and independently generated simple random 4-regular graphs. We compare the spectral-edge proxy, first loop residual, receipt age, and state disagreement. This is a finite transfer check only: no universal limit, local law, or theorem hypothesis is asserted.\n\n"
        "| topology | delay | policy | games | spectral edge | loop residual | receipt age | disagreement |\n|---|---:|---|---:|---:|---:|---:|---:|\n"
        + "\n".join(
            f"| {row['topology']} | {row['delay']} | {row['policy']} | {row['n_games']} | {row['mean_spectral_edge']:.17g} | {row['mean_loop_residual']:.17g} | {row['mean_receipt_age']:.17g} | {row['mean_disagreement']:.17g} |"
            for row in summary
        )
        + "\n\nThe intended reading is comparative: if a delay effect survives topology changes, it is less likely to be a ring-only artifact; if it changes substantially, topology is part of the mechanism and must be treated as a registered factor in any larger study.\n",
        encoding="utf-8",
    )
    manifest = OUT / "DELAY_ENSEMBLE_TRANSFER_MANIFEST.sha256"
    manifest.write_text("\n".join(f"{sha256(path)} *{path}" for path in [detail, summary_path, figure, report, Path(__file__)]) + "\n", encoding="utf-8")
    print(f"Games: {len(rows)}")
    print(f"Summary: {summary_path}")
    print(f"Figure: {figure}")
    print(f"Report: {report}")
    print(f"Manifest: {manifest}")


if __name__ == "__main__":
    main()
