"""Johnson-Lindenstrauss projection versus coordinatewise safety detection.

The experiment asks whether a low-dimensional random projection can preserve
geometry while a simple projected-norm monitor misses a violation in one
coordinate. This is an exploratory diagnostic for compressed multi-agent
state/communication features, not a JL proof and not a safety certificate.
"""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "20260730" / "jl_projection"
D = 64
SAFETY_DIMS = 8
PROJECTION_DIMS = (2, 4, 8, 16, 32)
REPEATS = 20
NORMAL_SAMPLES = 2000
VIOLATION_SAMPLES = 1000


def run_repeat(k: int, seed: int) -> dict[str, float | int]:
    rng = np.random.default_rng(seed)
    projection = rng.normal(0.0, 1.0, size=(D, k)) / np.sqrt(k)
    normal = rng.normal(0.0, 0.25, size=(NORMAL_SAMPLES, D))
    violations = rng.normal(0.0, 0.25, size=(VIOLATION_SAMPLES, D))
    violated_coordinates = rng.integers(0, SAFETY_DIMS, size=VIOLATION_SAMPLES)
    signs = rng.choice((-1.0, 1.0), size=VIOLATION_SAMPLES)
    for index, coordinate in enumerate(violated_coordinates):
        violations[index, coordinate] = signs[index] * (1.2 + abs(violations[index, coordinate]))

    normal_projected = normal @ projection
    violation_projected = violations @ projection
    normal_norms = np.linalg.norm(normal_projected, axis=1)
    violation_norms = np.linalg.norm(violation_projected, axis=1)
    threshold = float(np.quantile(normal_norms, 0.995))

    pair_indices = rng.integers(0, NORMAL_SAMPLES, size=(400, 2))
    original_distances = np.linalg.norm(normal[pair_indices[:, 0]] - normal[pair_indices[:, 1]], axis=1)
    projected_distances = np.linalg.norm(
        normal_projected[pair_indices[:, 0]] - normal_projected[pair_indices[:, 1]], axis=1
    )
    nonzero = original_distances > 1e-12
    relative_distortion = np.abs(projected_distances[nonzero] / original_distances[nonzero] - 1.0)

    return {
        "projection_dim": k,
        "seed": seed,
        "mean_abs_relative_distance_distortion": float(np.mean(relative_distortion)),
        "p95_abs_relative_distance_distortion": float(np.quantile(relative_distortion, 0.95)),
        "projected_norm_threshold": threshold,
        "projected_violation_recall": float(np.mean(violation_norms > threshold)),
        "projected_normal_false_positive_rate": float(np.mean(normal_norms > threshold)),
        "coordinate_oracle_recall": 1.0,
        "coordinate_oracle_false_positive_rate": 0.0,
    }


def aggregate(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    summary: list[dict[str, object]] = []
    for k in PROJECTION_DIMS:
        group = [row for row in rows if int(row["projection_dim"]) == k]
        summary.append(
            {
                "projection_dim": k,
                "n_repeats": len(group),
                "mean_abs_relative_distance_distortion": sum(float(row["mean_abs_relative_distance_distortion"]) for row in group) / len(group),
                "p95_abs_relative_distance_distortion": sum(float(row["p95_abs_relative_distance_distortion"]) for row in group) / len(group),
                "mean_projected_violation_recall": sum(float(row["projected_violation_recall"]) for row in group) / len(group),
                "mean_projected_false_positive_rate": sum(float(row["projected_normal_false_positive_rate"]) for row in group) / len(group),
                "coordinate_oracle_recall": 1.0,
                "coordinate_oracle_false_positive_rate": 0.0,
            }
        )
    return summary


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def render(summary: list[dict[str, object]], path: Path) -> None:
    dimensions = [int(row["projection_dim"]) for row in summary]
    distortion = [float(row["mean_abs_relative_distance_distortion"]) for row in summary]
    p95_distortion = [float(row["p95_abs_relative_distance_distortion"]) for row in summary]
    recall = [float(row["mean_projected_violation_recall"]) for row in summary]
    false_positive = [float(row["mean_projected_false_positive_rate"]) for row in summary]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=False)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.82, bottom=0.18, wspace=0.28)
    axes[0].plot(dimensions, distortion, marker="o", linewidth=2.5, color="#0072b2", label="mean absolute distortion")
    axes[0].plot(dimensions, p95_distortion, marker="s", linestyle="--", linewidth=2, color="#56b4e9", label="p95 absolute distortion")
    axes[1].plot(dimensions, recall, marker="o", linewidth=2.5, color="#d55e00", label="projected-norm recall")
    axes[1].plot(dimensions, false_positive, marker="s", linestyle="--", linewidth=2, color="#e69f00", label="projected false-positive rate")
    axes[1].axhline(1.0, linestyle=":", color="#009e73", label="coordinate oracle recall")
    for axis in axes:
        axis.set_xlabel("projection dimension k")
        axis.grid(alpha=0.25)
        axis.legend()
    axes[0].set_title("Pairwise geometry")
    axes[0].set_ylabel("absolute relative distance distortion")
    axes[1].set_title("Single-coordinate violation detection")
    axes[1].set_ylabel("rate")
    fig.suptitle("JL-style compression: geometry versus coordinate safety", fontsize=16, fontweight="bold")
    fig.text(0.01, 0.025, f"D={D}; {SAFETY_DIMS} safety coordinates; {REPEATS} projection seeds/dimension; exploratory only.", fontsize=9)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for k in PROJECTION_DIMS:
        for repeat in range(REPEATS):
            rows.append(run_repeat(k, 20260730 + 1000 * k + repeat))
    summary = aggregate(rows)
    raw_path = OUT / "jl_projection_safety_games.csv"
    summary_path = OUT / "jl_projection_safety_summary.csv"
    figure_path = OUT / "jl_projection_safety.png"
    report_path = OUT / "jl_projection_safety_report.md"
    manifest_path = OUT / "JL_PROJECTION_SAFETY_MANIFEST.sha256"
    write_csv(raw_path, rows)
    write_csv(summary_path, summary)
    render(summary, figure_path)
    report_path.write_text(
        """# JL-Style Projection: Geometry versus Coordinate Safety

This diagnostic is inspired by Yingru Li's *Simple, unified analysis of Johnson-Lindenstrauss with applications* (arXiv:2402.10232). It tests a narrow engineering question: a random projection may preserve pairwise geometry while a projected-norm monitor fails to preserve a coordinatewise safety condition.

Each synthetic multi-agent feature vector has 64 coordinates, eight of which are treated as safety channels. Violations are single-coordinate spikes. The coordinate oracle checks those channels directly. The compressed monitor projects the vector to dimension `k` and thresholds the projected norm using the 99.5th percentile of normal projected samples.

This is not a JL theorem, and the projected-norm detector is only one possible compressed monitor. The result should be read as a warning against treating distance preservation as automatic preservation of axis-aligned safety constraints. Any real projection-based monitor would need a task-specific proof or empirical certification of the safety predicate.
""",
        encoding="utf-8",
    )
    outputs = [raw_path, summary_path, figure_path, report_path]
    manifest_path.write_text("".join(f"{sha256(path)}  *{path}\n" for path in outputs), encoding="utf-8")
    print(f"Rows: {len(rows)}")
    print(f"Summary: {summary_path}")
    print(f"Figure: {figure_path}")
    print(f"Report: {report_path}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
