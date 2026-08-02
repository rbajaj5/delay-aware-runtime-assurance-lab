"""Derive an optional prime-labeled fingerprint from the retained graph trace.

This is read-only postprocessing. The arithmetic labels are a deterministic
serialization of the conflict graph, not a new safety metric or theorem.
"""

from __future__ import annotations

import csv
import hashlib
import itertools
import math
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "artifacts" / "20260802" / "helicopter_3d"
SOURCE = ARTIFACT / "helicopter_3d_step_trace.csv"
OUT_CSV = ARTIFACT / "helicopter_prime_graph_signature.csv"
OUT_REPORT = ARTIFACT / "helicopter_prime_graph_signature_report.md"
OUT_MANIFEST = ARTIFACT / "HELICOPTER_PRIME_GRAPH_MANIFEST.sha256"
SEPARATION_THRESHOLD = 2.0
AGENT_PRIMES = [2, 3, 5, 7, 11, 13, 17, 19]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def graph_edges(rows: list[dict[str, str]], policy: str, step: int) -> set[tuple[int, int]]:
    current = [row for row in rows if row["policy"] == policy and int(row["step"]) == step]
    positions = {
        int(row["helicopter"]): np.array([float(row["x"]), float(row["y"]), float(row["z"])])
        for row in current
    }
    pads = {int(row["helicopter"]): int(row["pad"]) for row in current}
    active = [
        int(row["helicopter"])
        for row in current
        if row["phase"] in {"approach", "landed"}
    ]
    return {
        (a, b)
        for a, b in itertools.combinations(active, 2)
        if pads[a] == pads[b]
        and float(np.linalg.norm(positions[a] - positions[b])) < SEPARATION_THRESHOLD
    }


def main() -> None:
    with SOURCE.open(newline="", encoding="utf-8") as handle:
        source_rows = list(csv.DictReader(handle))
    policies = sorted({row["policy"] for row in source_rows})
    steps = sorted({int(row["step"]) for row in source_rows})
    signatures: list[dict[str, object]] = []
    previous_edges: dict[str, set[tuple[int, int]]] = {policy: set() for policy in policies}
    edge_runs: dict[str, dict[tuple[int, int], int]] = {policy: {} for policy in policies}

    for policy in policies:
        for step in steps:
            edges = graph_edges(source_rows, policy, step)
            current_runs: dict[tuple[int, int], int] = {}
            for edge in edges:
                current_runs[edge] = edge_runs[policy].get(edge, 0) + 1 if edge in previous_edges[policy] else 1
            edge_runs[policy] = current_runs
            previous_edges[policy] = edges
            current = [
                row
                for row in source_rows
                if row["policy"] == policy and int(row["step"]) == step
            ]
            active_nodes = sum(row["phase"] in {"approach", "landed"} for row in current)
            edge_codes = [AGENT_PRIMES[a] * AGENT_PRIMES[b] for a, b in sorted(edges)]
            prime_sum = sum(edge_codes)
            possible_edges = max(1, active_nodes * (active_nodes - 1) // 2)
            persistence_log10 = sum(
                run * math.log10(code)
                for (a, b), run in current_runs.items()
                for code in [AGENT_PRIMES[a] * AGENT_PRIMES[b]]
            )
            signatures.append(
                {
                    "policy": policy,
                    "step": step,
                    "active_nodes": active_nodes,
                    "edge_count": len(edges),
                    "edge_codes": ";".join(map(str, edge_codes)),
                    "prime_sum_addition": prime_sum,
                    "prime_normalized_division": prime_sum / possible_edges,
                    "persistence_log10_exponentiation": persistence_log10,
                }
            )

    write_csv(OUT_CSV, signatures)
    by_policy = {
        policy: {int(row["step"]): row for row in signatures if row["policy"] == policy}
        for policy in policies
    }
    contrast_rows = []
    for step in steps:
        blind = float(by_policy["delay_blind"][step]["prime_normalized_division"])
        aware = float(by_policy["queue_aware"][step]["prime_normalized_division"])
        contrast_rows.append((step, blind, aware, blind - aware))
    peak_step, _, _, peak_delta = max(contrast_rows, key=lambda item: item[3])

    OUT_REPORT.write_text(
        "\n".join(
            [
                "# Prime-Labeled Conflict-Graph Fingerprint",
                "",
                f"Source: `{SOURCE.relative_to(ROOT).as_posix()}`",
                f"Source SHA-256: `{sha256(SOURCE)}`",
                "",
                "This is deterministic read-only postprocessing of the retained trace. It is an optional serialization aid, not a safety certificate, causal model, or complexity result.",
                "",
                "Assign agent H1-H8 the first eight primes `2, 3, 5, 7, 11, 13, 17, 19`. For a conflict edge (i,j), define the multiplicative edge code `q_ij = p_i * p_j`. At step t:",
                "",
                "`A_t = sum_(i,j in E_t) q_ij` is the additive conflict code; `N_t = A_t / max(1, C(|V_t|, 2))` is the normalized division form; and `P_t = product_(i,j in E_t) q_ij^(r_ij(t))` is the persistence form, where r_ij(t) is the current consecutive edge run length. The CSV stores log10(P_t) to avoid overflow.",
                "",
                "The policy contrast is the subtraction `Delta_t = N_t(delay_blind) - N_t(queue_aware)`. In this trace its largest value occurs at step "
                + str(peak_step)
                + " and equals "
                + f"{peak_delta:.6f}. This arithmetic is a reproducible fingerprint of graph structure; the substantive findings remain the conflict counts and throughput reported in the main paper.",
                "",
                "The prime labels are intentionally kept out of the paper's primary claim. Replacing them with another injective node-label scheme would preserve the underlying graph comparison.",
                "",
            ]
        ),
        encoding="ascii",
    )
    outputs = [SOURCE, OUT_CSV, OUT_REPORT, Path(__file__)]
    OUT_MANIFEST.write_text(
        "".join(f"{sha256(path)}  {path.relative_to(ROOT).as_posix()}\n" for path in outputs),
        encoding="ascii",
    )
    print(f"Signature: {OUT_CSV}")
    print(f"Report: {OUT_REPORT}")
    print(f"Manifest: {OUT_MANIFEST}")


if __name__ == "__main__":
    main()
