"""Exploratory random-matrix diagnostic for delayed multi-agent coordination.

This is a finite-system diagnostic inspired by Bourgade and Huang,
"Loop Equations Characterize Random Matrix Statistics" (arXiv:2607.07617).
It does not assert that the fleet matrices form a Wigner ensemble or that a
random-matrix universality theorem applies.  The experiment asks whether
receipt-age structure changes a spectral-edge proxy and a resolvent loop
residual in a small delayed coordination model.
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
ROUNDS = 180
DELAYS = (0, 1, 2, 3, 5)
POLICIES = ("blind_delay", "queue_aware")
GAMES = 30
SEED_BASE = 2026073600
ALPHA = 0.82
FORCING = 0.10
RESOLVENT_Z = 2.0 + 0.5j


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ring_neighbors(agent: int) -> tuple[int, int]:
    return ((agent - 1) % AGENTS, (agent + 1) % AGENTS)


def centered_age_matrix(ages: dict[tuple[int, int], float]) -> np.ndarray:
    """Build a symmetric, centered 1/sqrt(N)-scaled age matrix."""
    raw = np.zeros((AGENTS, AGENTS), dtype=float)
    for (receiver, sender), age in ages.items():
        raw[receiver, sender] = age
    raw = 0.5 * (raw + raw.T)
    off_diagonal = raw[np.triu_indices(AGENTS, k=1)]
    mean = float(np.mean(off_diagonal))
    std = float(np.std(off_diagonal))
    if std <= 1e-12:
        return np.zeros_like(raw)
    centered = (raw - mean) / std
    np.fill_diagonal(centered, 0.0)
    return centered / math.sqrt(AGENTS)


def spectral_metrics(matrix: np.ndarray) -> tuple[float, float, float]:
    """Return spectral edge, resolvent norm, and loop-equation residual."""
    eigenvalues = np.linalg.eigvalsh(matrix)
    z = RESOLVENT_Z
    resolvent = np.linalg.inv(matrix.astype(complex) - z * np.eye(AGENTS, dtype=complex))
    m = np.trace(resolvent) / AGENTS
    residual = abs(m * m + z * m + 1.0)
    return float(eigenvalues[-1]), float(abs(m)), float(residual)


def single_entry_stability(matrix: np.ndarray) -> tuple[float, float, float, float]:
    """Measure one symmetric edge perturbation of the resolvent trace."""
    z = RESOLVENT_Z
    base_resolvent = np.linalg.inv(matrix.astype(complex) - z * np.eye(AGENTS, dtype=complex))
    base_m = np.trace(base_resolvent) / AGENTS
    lambda_bound = float(np.max(np.abs(base_resolvent)))
    nonzero = np.argwhere(np.triu(np.abs(matrix), k=1) > 1e-12)
    if len(nonzero) == 0:
        return 0.0, 0.0, 0.0, 0.0
    i, j = max(nonzero.tolist(), key=lambda pair: abs(matrix[pair[0], pair[1]]))
    perturbed = matrix.copy()
    epsilon = 0.05 / math.sqrt(AGENTS)
    perturbed[i, j] += epsilon
    perturbed[j, i] += epsilon
    perturbed_resolvent = np.linalg.inv(perturbed.astype(complex) - z * np.eye(AGENTS, dtype=complex))
    perturbed_m = np.trace(perturbed_resolvent) / AGENTS
    delta_m = abs(perturbed_m - base_m)
    minus = matrix.copy()
    minus[i, j] -= epsilon
    minus[j, i] -= epsilon
    minus_resolvent = np.linalg.inv(minus.astype(complex) - z * np.eye(AGENTS, dtype=complex))
    minus_m = np.trace(minus_resolvent) / AGENTS
    finite_difference = (perturbed_m - minus_m) / (2.0 * epsilon)
    exact_derivative = -(2.0 / AGENTS) * (base_resolvent @ base_resolvent)[i, j]
    derivative_error = abs(finite_difference - exact_derivative) / max(abs(exact_derivative), 1e-15)
    stability_proxy = (AGENTS ** -0.5) * (lambda_bound ** 3)
    ratio = delta_m / stability_proxy if stability_proxy > 0.0 else 0.0
    return float(delta_m), float(stability_proxy), float(ratio), float(derivative_error)


def simulate(delay: int, policy: str, seed: int) -> dict[str, float | int | str]:
    rng = random.Random(seed)
    phase = rng.random() * 2.0 * math.pi
    state = np.array(
        [0.45 * math.sin(2.0 * math.pi * i / AGENTS + phase) + 0.08 * rng.gauss(0.0, 1.0) for i in range(AGENTS)],
        dtype=float,
    )
    latest = {(receiver, sender): state[sender] for receiver in range(AGENTS) for sender in ring_neighbors(receiver)}
    latest_sent_round = {(receiver, sender): 0 for receiver in range(AGENTS) for sender in ring_neighbors(receiver)}
    pending: list[tuple[int, int, int, float, int]] = []
    edge_values: list[float] = []
    edge_abs: list[float] = []
    residuals: list[float] = []
    resolvent_norms: list[float] = []
    delta_ms: list[float] = []
    stability_proxies: list[float] = []
    stability_ratios: list[float] = []
    derivative_errors: list[float] = []
    disagreement: list[float] = []
    staleness: list[float] = []
    overshoot = 0

    for round_index in range(ROUNDS):
        due = [item for item in pending if item[0] <= round_index]
        pending = [item for item in pending if item[0] > round_index]
        for _, receiver, sender, payload, sent_round in due:
            if sent_round >= latest_sent_round[(receiver, sender)]:
                latest[(receiver, sender)] = payload
                latest_sent_round[(receiver, sender)] = sent_round

        visible = {}
        visible_ages = {}
        for receiver in range(AGENTS):
            for sender in ring_neighbors(receiver):
                choice = (latest[(receiver, sender)], latest_sent_round[(receiver, sender)])
                if policy == "queue_aware":
                    queued = [item for item in pending if item[1] == receiver and item[2] == sender]
                    if queued:
                        queued_choice = max(queued, key=lambda item: item[4])
                        if queued_choice[4] >= choice[1]:
                            choice = (queued_choice[3], queued_choice[4])
                visible[(receiver, sender)] = choice[0]
                visible_ages[(receiver, sender)] = round_index - choice[1]

        matrix = centered_age_matrix(visible_ages)
        edge, resolvent_norm, loop_residual = spectral_metrics(matrix)
        delta_m, stability_proxy, stability_ratio, derivative_error = single_entry_stability(matrix)
        spectral_edge = edge
        if spectral_edge > 2.0:
            overshoot += 1
        edge_values.append(spectral_edge)
        edge_abs.append(abs(spectral_edge))
        residuals.append(loop_residual)
        resolvent_norms.append(resolvent_norm)
        delta_ms.append(delta_m)
        stability_proxies.append(stability_proxy)
        stability_ratios.append(stability_ratio)
        derivative_errors.append(derivative_error)
        staleness.append(float(np.mean(list(visible_ages.values()))))

        target = np.zeros(AGENTS, dtype=float)
        for receiver in range(AGENTS):
            neighbor_values = [visible[(receiver, sender)] for sender in ring_neighbors(receiver)]
            target[receiver] = sum(neighbor_values) / len(neighbor_values)
        drive = np.array(
            [FORCING * math.sin(2.0 * math.pi * round_index / 45.0 + 0.12 * i + phase) for i in range(AGENTS)],
            dtype=float,
        )
        next_state = state + ALPHA * (target - state) + drive
        disagreement.append(float(np.std(state)))
        if float(np.max(np.abs(next_state))) > 2.0:
            overshoot += 1

        for sender in range(AGENTS):
            for receiver in ring_neighbors(sender):
                pending.append((round_index + delay, receiver, sender, float(next_state[sender]), round_index + 1))
        state = next_state

    return {
        "delay": delay,
        "policy": policy,
        "seed": seed,
        "agents": AGENTS,
        "rounds": ROUNDS,
        "mean_spectral_edge": float(np.mean(edge_values)),
        "p95_spectral_edge": float(np.percentile(edge_values, 95)),
        "mean_abs_spectral_edge": float(np.mean(edge_abs)),
        "mean_loop_residual": float(np.mean(residuals)),
        "p95_loop_residual": float(np.percentile(residuals, 95)),
        "mean_resolvent_norm": float(np.mean(resolvent_norms)),
        "mean_single_entry_delta_m": float(np.mean(delta_ms)),
        "mean_stability_proxy": float(np.mean(stability_proxies)),
        "mean_delta_to_proxy_ratio": float(np.mean(stability_ratios)),
        "mean_derivative_relative_error": float(np.mean(derivative_errors)),
        "mean_receipt_age": float(np.mean(staleness)),
        "mean_disagreement": float(np.mean(disagreement)),
        "max_disagreement": float(np.max(disagreement)),
        "spectral_edge_exceedance_rate": float(sum(value > 2.0 for value in edge_values) / ROUNDS),
        "state_overshoot_count": overshoot,
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
                "mean_spectral_edge": sum(row["mean_spectral_edge"] for row in subset) / len(subset),
                "mean_p95_spectral_edge": sum(row["p95_spectral_edge"] for row in subset) / len(subset),
                "mean_loop_residual": sum(row["mean_loop_residual"] for row in subset) / len(subset),
                "mean_p95_loop_residual": sum(row["p95_loop_residual"] for row in subset) / len(subset),
                "mean_receipt_age": sum(row["mean_receipt_age"] for row in subset) / len(subset),
                "mean_single_entry_delta_m": sum(row["mean_single_entry_delta_m"] for row in subset) / len(subset),
                "mean_stability_proxy": sum(row["mean_stability_proxy"] for row in subset) / len(subset),
                "mean_delta_to_proxy_ratio": sum(row["mean_delta_to_proxy_ratio"] for row in subset) / len(subset),
                "mean_derivative_relative_error": sum(row["mean_derivative_relative_error"] for row in subset) / len(subset),
                "mean_disagreement": sum(row["mean_disagreement"] for row in subset) / len(subset),
                "mean_max_disagreement": sum(row["max_disagreement"] for row in subset) / len(subset),
                "mean_spectral_edge_exceedance_rate": sum(row["spectral_edge_exceedance_rate"] for row in subset) / len(subset),
                "mean_state_overshoot_count": sum(row["state_overshoot_count"] for row in subset) / len(subset),
            })

    detail = OUT / "delay_random_matrix_loop_games.csv"
    summary_path = OUT / "delay_random_matrix_loop_summary.csv"
    write_csv(detail, rows)
    write_csv(summary_path, summary)

    colors = {"blind_delay": "#D55E00", "queue_aware": "#0072B2"}
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.8))
    for policy in POLICIES:
        selected = [row for row in summary if row["policy"] == policy]
        label = "blind delay" if policy == "blind_delay" else "queue-aware"
        axes[0].plot([row["delay"] for row in selected], [row["mean_spectral_edge"] for row in selected], marker="o", linewidth=3, color=colors[policy], label=label)
        axes[1].plot([row["delay"] for row in selected], [row["mean_loop_residual"] for row in selected], marker="o", linewidth=3, color=colors[policy], label=label)
        axes[2].plot([row["delay"] for row in selected], [row["mean_disagreement"] for row in selected], marker="o", linewidth=3, color=colors[policy], label=label)
    axes[0].axhline(2.0, color="#333333", linestyle="--", linewidth=1.5, label="edge reference 2")
    axes[0].set_title("Spectral-edge proxy")
    axes[0].set_ylabel("Mean largest eigenvalue")
    axes[1].set_title("First loop residual")
    axes[1].set_ylabel("Mean |m(z)^2 + z m(z) + 1|")
    axes[2].set_title("Coordination disagreement")
    axes[2].set_ylabel("Mean cross-agent state std")
    for axis in axes:
        axis.set_xlabel("Receipt delay (rounds)")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    fig.suptitle("Loop-equation-inspired diagnostic for delayed coordination", fontsize=15, fontweight="bold")
    fig.text(0.01, 0.01, f"24-agent ring; {GAMES} games/cell; z={RESOLVENT_Z}; exploratory proxy, not a universality test.", fontsize=8)
    fig.tight_layout(rect=[0, 0.05, 1, 0.94])
    figure = OUT / "delay_random_matrix_loop_diagnostic.png"
    fig.savefig(figure, dpi=220, facecolor="white")
    plt.close(fig)

    report = OUT / "delay_random_matrix_loop_diagnostic_report.md"
    report.write_text(
        "# Delay / Random-Matrix Loop Diagnostic\n\n"
        f"Conceptual source: [Bourgade and Huang, *Loop Equations Characterize Random Matrix Statistics*]({PAPER_URL}). "
        "The source proves uniqueness results for Sine_beta and Airy_beta point processes from loop-equation hierarchies and motivates checking approximate loop equations as a route to universality.\n\n"
        "## Operational translation\n\n"
        "At each round, the experiment forms a 24-by-24 receipt-age matrix on a ring. It is symmetrized, centered, and scaled by 1/sqrt(N). The spectral-edge proxy is the largest eigenvalue. For z=2+0.5i, the reported first residual is |m(z)^2 + z m(z) + 1|, using m(z)=N^-1 Tr((H-zI)^-1), the semicircle loop-equation form as a diagnostic reference.\n\n"
        "The blind policy uses the latest delivered receipt. The queue-aware policy overlays the newest pending payload before updating. This is a toy coordination model: the matrix is not assumed to be Wigner, no Sine_beta/Airy_beta limit is claimed, and the residual is not a safety certificate. The diagnostic asks whether receipt-age structure changes the spectral summaries that a later, larger fleet study could test more seriously.\n\n"
        "## Connection to the attached excerpt\n\n"
        "The supplied excerpt gives two useful controls for this interpretation: a Gronwall-type exponential envelope for a growth quantity, and a single-entry resolvent-stability estimate of the form C N^-1/2 Lambda^3 under explicit bounds on resolvent entries and matrix entries. We use those ideas to justify tracking both a growth/disagreement trace and the resolvent observable. We do not claim that the theorem hypotheses hold for this receipt-age matrix. The next rigorous extension would perturb one age edge at a time, measure the resulting change in m(z), and check the stated assumptions before comparing against a stability bound.\n\n"
        "This run performs that perturbation check descriptively. For each nondegenerate age matrix, the largest off-diagonal age edge is increased symmetrically by 0.05/sqrt(N). `single_entry_delta_m` is the observed change in m(z); `stability_proxy` is N^-1/2 Lambda^3 with the constant omitted. The ratio is not a pass/fail theorem test because the source bound has an unspecified constant and our matrices are not sampled from the source ensembles.\n\n"
        "The derivative identity is also checked numerically. For the selected off-diagonal edge, the central finite difference of m(z) is compared with `-(2/N)(G^2)ij`; the reported relative error is a calculus check, not a physical metric.\n\n"
        "| delay | policy | games | spectral edge | loop residual | receipt age | delta m | stability proxy | delta/proxy | derivative rel. error | disagreement | edge exceedance | state overshoot |\n|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n"
        + "\n".join(
            f"| {row['delay']} | {row['policy']} | {row['n_games']} | {row['mean_spectral_edge']:.17g} | {row['mean_loop_residual']:.17g} | {row['mean_receipt_age']:.17g} | {row['mean_single_entry_delta_m']:.17g} | {row['mean_stability_proxy']:.17g} | {row['mean_delta_to_proxy_ratio']:.17g} | {row['mean_derivative_relative_error']:.17g} | {row['mean_disagreement']:.17g} | {row['mean_spectral_edge_exceedance_rate']:.17g} | {row['mean_state_overshoot_count']:.17g} |"
            for row in summary
        )
        + "\n\nInterpretation is deliberately limited: a difference in the spectral proxy would motivate a larger ensemble study; it does not establish random-matrix universality or a causal safety mechanism.\n",
        encoding="utf-8",
    )
    manifest = OUT / "DELAY_RANDOM_MATRIX_LOOP_MANIFEST.sha256"
    manifest.write_text("\n".join(f"{sha256(path)} *{path}" for path in [detail, summary_path, figure, report, Path(__file__)]) + "\n", encoding="utf-8")
    print(f"Games: {len(rows)}")
    print(f"Summary: {summary_path}")
    print(f"Figure: {figure}")
    print(f"Report: {report}")
    print(f"Manifest: {manifest}")


if __name__ == "__main__":
    main()
