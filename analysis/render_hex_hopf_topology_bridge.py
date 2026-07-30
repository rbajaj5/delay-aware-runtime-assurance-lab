"""Create a topology-only bridge between Hex and the Hopf-fibration paper.

This does not change the Hex policy or add a physics claim. It visualizes the
standard Hopf map S^3 -> S^2 beside a completed Hex board, as a teaching aid
for the distinction between global invariants and local coordinates.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import deque
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis" / "out" / "hex_delay_exploratory_20260730"
TRACE = OUT / "hex_delay_trace_delay3_seed0.json"
PAPER_URL = "https://philpapers.org/archive/NIETTU.pdf"
N = 7


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def neighbors(r: int, c: int):
    return [(r, c - 1), (r, c + 1), (r - 1, c), (r + 1, c), (r - 1, c + 1), (r + 1, c - 1)]


def crossing(board, color: int):
    starts = [(r, 0) for r in range(N)] if color == 1 else [(0, c) for c in range(N)]
    target = (lambda r, c: c == N - 1) if color == 1 else (lambda r, c: r == N - 1)
    queue = deque(cell for cell in starts if board[cell[0]][cell[1]] == color)
    parent = {cell: None for cell in queue}
    goal = None
    while queue:
        cell = queue.popleft()
        r, c = cell
        if target(r, c):
            goal = cell
            break
        for nr, nc in neighbors(r, c):
            if 0 <= nr < N and 0 <= nc < N and board[nr][nc] == color and (nr, nc) not in parent:
                parent[(nr, nc)] = cell
                queue.append((nr, nc))
    if goal is None:
        return []
    path = []
    while goal is not None:
        path.append(goal)
        goal = parent[goal]
    return list(reversed(path))


def hopf(eta, xi1, xi2):
    """Standard Hopf map for z1=cos(eta)e^{i xi1}, z2=sin(eta)e^{i xi2}."""
    z1 = np.cos(eta) * np.exp(1j * xi1)
    z2 = np.sin(eta) * np.exp(1j * xi2)
    return np.stack((2 * np.real(z1 * np.conj(z2)), 2 * np.imag(z1 * np.conj(z2)), np.abs(z1) ** 2 - np.abs(z2) ** 2), axis=-1)


def main() -> None:
    trace = json.loads(TRACE.read_text(encoding="utf-8"))
    final = trace["queue_aware"][-1]["true_board"]
    path = crossing(final, 1) or crossing(final, 2)

    fig = plt.figure(figsize=(13.4, 6.9), facecolor="white")
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    eta = np.linspace(0.03, np.pi / 2 - 0.03, 40)
    delta = np.linspace(0, 2 * np.pi, 120)
    E, D = np.meshgrid(eta, delta, indexing="ij")
    sphere = hopf(E, D, np.zeros_like(D))
    ax.plot_surface(sphere[..., 0], sphere[..., 1], sphere[..., 2], cmap="viridis", alpha=0.18, linewidth=0)
    for phase in np.linspace(0, 2 * np.pi, 5, endpoint=False):
        circle = hopf(np.full_like(delta, np.pi / 4), delta, delta - phase)
        ax.plot(circle[:, 0], circle[:, 1], circle[:, 2], linewidth=2, label=f"fiber phase {phase / np.pi:.1f}π")
    ax.set_title("Hopf map: local fiber phase over a global base", pad=15, fontsize=13, fontweight="bold")
    ax.set_xlabel("base x")
    ax.set_ylabel("base y")
    ax.set_zlabel("base z")
    ax.set_box_aspect((1, 1, 1))
    ax.legend(fontsize=8, loc="upper left")

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.set_aspect("equal")
    for r in range(N):
        for c in range(N):
            x = c + 0.5 * r
            y = r * 0.86
            angles = np.arange(6) * np.pi / 3 + np.pi / 6
            phase = (c - r) * np.pi / 7
            # Phase is a visual tag only; it does not modify the trace or policy.
            edge = plt.cm.twilight((phase / (2 * np.pi)) % 1.0)
            value = final[r][c]
            face = "#2369be" if value == 1 else "#eaa723" if value == 2 else "#fafbfc"
            polygon = np.column_stack((x + 0.42 * np.cos(angles), y + 0.42 * np.sin(angles)))
            ax2.fill(polygon[:, 0], polygon[:, 1], facecolor=face, edgecolor=edge, linewidth=2.2)
            if (r, c) in path:
                ax2.scatter([x], [y], s=28, color="#00965a", edgecolor="white", zorder=4)
    ax2.set_xlim(-0.8, N - 1 + 0.5 * (N - 1) + 0.8)
    ax2.set_ylim(-0.8, (N - 1) * 0.86 + 0.8)
    ax2.axis("off")
    ax2.set_title("Hex: global crossing with local phase labels", fontsize=13, fontweight="bold", pad=15)
    ax2.text(0.02, -0.08, "green dots = queue-aware trace crossing; edge color = illustrative fiber tag", transform=ax2.transAxes, fontsize=9, color="#4d555c")

    fig.suptitle("Topology bridge for delayed information", fontsize=20, fontweight="bold")
    fig.text(0.01, 0.01, "Hopf map is standard mathematics; the right panel is a visual analogy, not a physical or v6 result.", fontsize=9, color="#4d555c")
    fig.tight_layout(rect=[0, 0.04, 1, 0.95])
    figure = OUT / "hex_hopf_topology_bridge.png"
    fig.savefig(figure, dpi=220, facecolor="white")
    plt.close(fig)

    report = OUT / "hex_hopf_topology_bridge_report.md"
    report.write_text(
        "# Hex/Hopf Topology Bridge\n\n"
        f"The bridge uses the standard Hopf parametrization `z1=cos(eta)e^(i xi1)`, `z2=sin(eta)e^(i xi2)` and the map `S^3 -> S^2`. The completed Hex board comes from the retained queue-aware delay-3 trace `{TRACE}` (SHA-256 `{sha256(TRACE)}`).\n\n"
        f"The external paper supplied for context is `{PAPER_URL}`. Its proposed unified-field interpretation is not used as a validated physical premise. The figure uses only the mathematical Hopf map and a separately labeled Hex crossing visualization. No policy, score, delay, or v6 conclusion is changed.\n",
        encoding="utf-8",
    )
    manifest = OUT / "HEX_HOPF_TOPOLOGY_BRIDGE_MANIFEST.sha256"
    manifest.write_text("\n".join(f"{sha256(path)} *{path}" for path in [TRACE, figure, report, Path(__file__)]) + "\n", encoding="utf-8")
    print(f"Figure: {figure}")
    print(f"Report: {report}")
    print(f"Manifest: {manifest}")


if __name__ == "__main__":
    main()
