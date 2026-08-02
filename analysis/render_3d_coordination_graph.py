"""Render the retained helicopter trace as an abstract 3D temporal graph.

The figure is about graph structure, not an operational scenario. Helicopter
nodes use logged 3D positions; blue edges represent logged pad assignments;
red edges represent logged conflict-graph edges under the replay threshold.
"""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter, FuncAnimation


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "artifacts" / "20260802" / "helicopter_3d"
SOURCE = ARTIFACT / "helicopter_3d_step_trace.csv"
OUT_VIDEO = ARTIFACT / "coordination_graph_3d.mp4"
OUT_POSTER = ARTIFACT / "coordination_graph_3d_poster.png"
OUT_SUMMARY = ARTIFACT / "coordination_graph_3d_summary.csv"
OUT_REPORT = ARTIFACT / "coordination_graph_3d_report.md"
OUT_MANIFEST = ARTIFACT / "COORDINATION_GRAPH_3D_MANIFEST.sha256"
SEPARATION_THRESHOLD = 2.0
FPS = 15


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_pad_positions() -> np.ndarray:
    module_path = Path(__file__).with_name("run_helicopter_3d_landing_simulation.py")
    spec = importlib.util.spec_from_file_location("helicopter_sim", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["helicopter_sim"] = module
    spec.loader.exec_module(module)
    return module.PADS.copy()


def edge_sets(rows: list[dict[str, str]], step: int) -> tuple[set[tuple[int, int]], set[tuple[int, int]]]:
    current = [row for row in rows if int(row["step"]) == step]
    positions = {
        int(row["helicopter"]): np.array([float(row["x"]), float(row["y"]), float(row["z"])])
        for row in current
    }
    pads = {int(row["helicopter"]): int(row["pad"]) for row in current}
    assignment_edges = {(node, 8 + pad) for node, pad in pads.items()}
    conflict_edges: set[tuple[int, int]] = set()
    for a in positions:
        for b in positions:
            if a >= b or pads[a] != pads[b]:
                continue
            active_a = next(row["phase"] in {"approach", "landed"} for row in current if int(row["helicopter"]) == a)
            active_b = next(row["phase"] in {"approach", "landed"} for row in current if int(row["helicopter"]) == b)
            if active_a and active_b and np.linalg.norm(positions[a] - positions[b]) < SEPARATION_THRESHOLD:
                conflict_edges.add((a, b))
    return assignment_edges, conflict_edges


def draw_panel(ax, rows: list[dict[str, str]], pads: np.ndarray, policy: str, step: int, colors) -> tuple[int, int]:
    ax.clear()
    current = [row for row in rows if row["policy"] == policy and int(row["step"]) == step]
    positions = {
        int(row["helicopter"]): np.array([float(row["x"]), float(row["y"]), float(row["z"])])
        for row in current
    }
    assignment_edges, conflict_edges = edge_sets(current, step)
    for pad_id, pad in enumerate(pads):
        ax.scatter([pad[0]], [pad[1]], [pad[2]], marker="s", s=120, color="#2166ac", depthshade=False)
        ax.text(pad[0], pad[1], pad[2] + 0.3, f"P{pad_id + 1}", fontsize=8)
    for node, pad_node in assignment_edges:
        pad = pads[pad_node - 8]
        point = positions[node]
        ax.plot([point[0], pad[0]], [point[1], pad[1]], [point[2], pad[2]], color="#4c78a8", alpha=0.45, linewidth=1.0, linestyle="--")
    for a, b in conflict_edges:
        a_pos, b_pos = positions[a], positions[b]
        ax.plot([a_pos[0], b_pos[0]], [a_pos[1], b_pos[1]], [a_pos[2], b_pos[2]], color="#d62728", linewidth=2.8)
    for node, position in positions.items():
        ax.scatter([position[0]], [position[1]], [position[2]], color=colors[node], s=48, depthshade=False)
        ax.text(position[0], position[1], position[2] + 0.28, f"H{node + 1}", fontsize=7)
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 17)
    ax.set_zlim(0, 10)
    ax.set_xlabel("logged x")
    ax.set_ylabel("logged y")
    ax.set_zlabel("logged z")
    ax.view_init(elev=26, azim=-62)
    ax.set_title(f"{policy.replace('_', ' ').title()} | step {step:03d}")
    ax.text2D(0.02, 0.97, f"blue assignment edges: {len(assignment_edges)} | red conflict edges: {len(conflict_edges)}", transform=ax.transAxes, fontsize=8, va="top")
    return len(assignment_edges), len(conflict_edges)


def main() -> None:
    with SOURCE.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    pads = load_pad_positions()
    policies = ["delay_blind", "queue_aware"]
    steps = sorted({int(row["step"]) for row in rows})
    colors = plt.cm.tab10(np.linspace(0, 1, 8))
    summary: list[dict[str, object]] = []
    metric_figure = plt.figure()
    metric_axis = metric_figure.add_subplot(111, projection="3d")
    for policy in policies:
        assignment_count = []
        conflict_count = []
        for step in steps:
            a, c = draw_panel(metric_axis, [row for row in rows if row["policy"] == policy], pads, policy, step, colors)
            assignment_count.append(a)
            conflict_count.append(c)
        summary.append(
            {
                "policy": policy,
                "frames": len(steps),
                "mean_assignment_edges": float(np.mean(assignment_count)),
                "total_conflict_edges": int(sum(conflict_count)),
                "peak_conflict_edges": int(max(conflict_count)),
            }
        )
    plt.close(metric_figure)

    with OUT_SUMMARY.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)

    figure = plt.figure(figsize=(15, 8), constrained_layout=True)
    axes = [figure.add_subplot(1, 2, 1, projection="3d"), figure.add_subplot(1, 2, 2, projection="3d")]

    def update(step: int) -> None:
        for axis, policy in zip(axes, policies):
            draw_panel(axis, rows, pads, policy, step, colors)
        figure.suptitle("3D temporal coordination graph: assignments and conflicts", fontsize=15)

    update(0)
    figure.savefig(OUT_POSTER, dpi=180)
    animation = FuncAnimation(figure, update, frames=steps, interval=1000 / FPS, blit=False)
    animation.save(OUT_VIDEO, writer=FFMpegWriter(fps=FPS, bitrate=1800), dpi=120)
    plt.close(figure)

    OUT_REPORT.write_text(
        "\n".join(
            [
                "# 3D Temporal Coordination Graph",
                "",
                f"Source trace: `{SOURCE.relative_to(ROOT).as_posix()}`",
                f"Source SHA-256: `{sha256(SOURCE)}`",
                "",
                "This artifact deliberately removes the operational scenario. Helicopters are nodes at their logged 3D positions; blue dashed edges represent logged pad assignments; red edges represent the retained conflict rule. The animation is a graph visualization, not a military or aviation planning tool.",
                "",
                "Network centrality, connected components, motifs, and spectral summaries remain descriptive diagnostics. They do not constitute safety certificates.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    outputs = [SOURCE, OUT_SUMMARY, OUT_POSTER, OUT_VIDEO, OUT_REPORT, Path(__file__)]
    OUT_MANIFEST.write_text(
        "".join(f"{sha256(path)}  {path.relative_to(ROOT).as_posix()}\n" for path in outputs),
        encoding="ascii",
    )
    print(f"Graph video: {OUT_VIDEO}")
    print(f"Graph poster: {OUT_POSTER}")
    print(f"Graph manifest: {OUT_MANIFEST}")
    for row in summary:
        print(row)


if __name__ == "__main__":
    main()
