"""Derive temporal network metrics from the retained helicopter trace.

Read-only postprocessing. No simulation or new flight trajectory is created.
The graph is a conflict graph: helicopters are nodes and edges connect two
active helicopters assigned to the same pad while within the separation
threshold used by the replay.
"""

from __future__ import annotations

import csv
import hashlib
import itertools
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "artifacts" / "20260802" / "helicopter_3d"
SOURCE = ARTIFACT / "helicopter_3d_step_trace.csv"
OUT_TIMESERIES = ARTIFACT / "helicopter_network_timeseries.csv"
OUT_SUMMARY = ARTIFACT / "helicopter_network_summary.csv"
OUT_PLOT = ARTIFACT / "helicopter_network_metrics.png"
OUT_REPORT = ARTIFACT / "helicopter_network_report.md"
OUT_MANIFEST = ARTIFACT / "HELICOPTER_NETWORK_MANIFEST.sha256"
SEPARATION_THRESHOLD = 2.0


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def component_groups(nodes: list[int], edges: set[tuple[int, int]]) -> list[list[int]]:
    parent = {node: node for node in nodes}

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(a: int, b: int) -> None:
        root_a, root_b = find(a), find(b)
        if root_a != root_b:
            parent[root_b] = root_a

    for a, b in edges:
        union(a, b)
    groups: dict[int, list[int]] = defaultdict(list)
    for node in nodes:
        groups[find(node)].append(node)
    return list(groups.values())


def graph_metrics(rows: list[dict[str, str]], policy: str, step: int) -> dict[str, object]:
    current = [row for row in rows if row["policy"] == policy and int(row["step"]) == step]
    positions = {
        int(row["helicopter"]): np.array([float(row["x"]), float(row["y"]), float(row["z"])])
        for row in current
    }
    pads = {int(row["helicopter"]): int(row["pad"]) for row in current}
    active = [int(row["helicopter"]) for row in current if row["phase"] in {"approach", "landed"}]
    edges: set[tuple[int, int]] = set()
    for a, b in itertools.combinations(active, 2):
        if pads[a] != pads[b]:
            continue
        if float(np.linalg.norm(positions[a] - positions[b])) < SEPARATION_THRESHOLD:
            edges.add((a, b))
    degrees = {node: 0 for node in active}
    for a, b in edges:
        degrees[a] += 1
        degrees[b] += 1
    groups = component_groups(active, edges)
    n_components = len(groups)
    largest_group = max(groups, key=len, default=[])
    largest_component = len(largest_group)
    if len(largest_group) > 1:
        laplacian = np.zeros((len(largest_group), len(largest_group)), dtype=float)
        index = {node: i for i, node in enumerate(largest_group)}
        for a, b in edges:
            if a not in index or b not in index:
                continue
            ia, ib = index[a], index[b]
            laplacian[ia, ia] += 1
            laplacian[ib, ib] += 1
            laplacian[ia, ib] -= 1
            laplacian[ib, ia] -= 1
        eigenvalues = np.linalg.eigvalsh(laplacian)
        algebraic_connectivity = float(eigenvalues[1]) if len(eigenvalues) > 1 else 0.0
    else:
        algebraic_connectivity = 0.0
    triangles = sum(1 for a, b, c in itertools.combinations(active, 3) if {(a, b), (a, c), (b, c)} <= edges)
    possible_edges = len(active) * (len(active) - 1) / 2
    return {
        "policy": policy,
        "step": step,
        "active_nodes": len(active),
        "edge_count": len(edges),
        "mean_degree": (sum(degrees.values()) / len(active)) if active else 0.0,
        "max_degree": max(degrees.values(), default=0),
        "network_density": len(edges) / possible_edges if possible_edges else 0.0,
        "components": n_components,
        "largest_component": largest_component,
        "algebraic_connectivity": algebraic_connectivity,
        "triangles": triangles,
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    with SOURCE.open(newline="", encoding="utf-8") as handle:
        source_rows = list(csv.DictReader(handle))
    policies = sorted({row["policy"] for row in source_rows})
    steps = sorted({int(row["step"]) for row in source_rows})
    metrics = [graph_metrics(source_rows, policy, step) for policy in policies for step in steps]
    write_csv(OUT_TIMESERIES, metrics)

    summary_rows: list[dict[str, object]] = []
    exposure: dict[str, dict[int, int]] = {policy: defaultdict(int) for policy in policies}
    for row in metrics:
        policy = str(row["policy"])
        summary_rows.append(row)
        if int(row["edge_count"]) > 0:
            for source_row in source_rows:
                if source_row["policy"] == policy and int(source_row["step"]) == int(row["step"]):
                    exposure[policy][int(source_row["helicopter"])] += 1
    summaries: list[dict[str, object]] = []
    for policy in policies:
        selected = [row for row in metrics if row["policy"] == policy]
        most_exposed = max(exposure[policy], key=exposure[policy].get, default=-1)
        summaries.append(
            {
                "policy": policy,
                "conflict_steps": sum(int(row["edge_count"]) > 0 for row in selected),
                "total_conflict_edges": sum(int(row["edge_count"]) for row in selected),
                "peak_edge_count": max(int(row["edge_count"]) for row in selected),
                "mean_degree_active": sum(float(row["mean_degree"]) for row in selected) / len(selected),
                "peak_largest_component": max(int(row["largest_component"]) for row in selected),
                "peak_algebraic_connectivity": max(float(row["algebraic_connectivity"]) for row in selected),
                "triangle_steps": sum(int(row["triangles"]) > 0 for row in selected),
                "most_exposed_helicopter": f"H{most_exposed + 1}" if most_exposed >= 0 else "none",
                "most_exposed_steps": exposure[policy].get(most_exposed, 0),
            }
        )
    write_csv(OUT_SUMMARY, summaries)

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.7), constrained_layout=True)
    for policy, color in zip(policies, ["#b2182b", "#2166ac"]):
        selected = [row for row in metrics if row["policy"] == policy]
        axes[0].plot([row["step"] for row in selected], [row["edge_count"] for row in selected], color=color, label=policy.replace("_", " "))
        axes[1].plot([row["step"] for row in selected], [row["largest_component"] for row in selected], color=color, label=policy.replace("_", " "))
    axes[0].set_title("Temporal conflict edges")
    axes[0].set_xlabel("step")
    axes[0].set_ylabel("active conflict edges")
    axes[1].set_title("Largest connected conflict component")
    axes[1].set_xlabel("step")
    axes[1].set_ylabel("helicopters")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend()
    figure.suptitle("Network-science view of delayed helicopter coordination")
    figure.savefig(OUT_PLOT, dpi=180)
    plt.close(figure)

    OUT_REPORT.write_text(
        "\n".join(
            [
                "# Network-Science Diagnostic for the Altara Helicopter Trace",
                "",
                f"Source: `{SOURCE.relative_to(ROOT).as_posix()}`",
                f"Source SHA-256: `{sha256(SOURCE)}`",
                "",
                "This is read-only postprocessing of the retained kinematic trace. Each frame is a temporal conflict graph: helicopters are nodes, and an edge joins two active helicopters assigned to the same pad within the replay's 2.0-unit separation threshold.",
                "",
                "The metrics are descriptive network diagnostics, not a claim that a centrality or spectral quantity is a safety certificate. The network lens makes the cascade visible: a small number of stale clearance decisions can create connected conflict components that persist over several steps.",
                "",
                "| policy | conflict steps | total conflict edges | peak component | triangle steps | most exposed |",
                "|---|---:|---:|---:|---:|---|",
                *[
                    f"| {row['policy']} | {row['conflict_steps']} | {row['total_conflict_edges']} | {row['peak_largest_component']} | {row['triangle_steps']} | {row['most_exposed_helicopter']} ({row['most_exposed_steps']} steps) |"
                    for row in summaries
                ],
                "",
            ]
        ),
        encoding="utf-8",
    )
    outputs = [SOURCE, OUT_TIMESERIES, OUT_SUMMARY, OUT_PLOT, OUT_REPORT, Path(__file__)]
    OUT_MANIFEST.write_text(
        "".join(f"{sha256(path)}  {path.relative_to(ROOT).as_posix()}\n" for path in outputs),
        encoding="ascii",
    )
    print(f"Network summary: {OUT_SUMMARY}")
    print(f"Network plot: {OUT_PLOT}")
    print(f"Manifest: {OUT_MANIFEST}")
    for row in summaries:
        print(row)


if __name__ == "__main__":
    main()
