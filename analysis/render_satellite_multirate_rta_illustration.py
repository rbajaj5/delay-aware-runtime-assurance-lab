"""Render an illustrative multirate RTA diagram for satellite robotics.

The paired traces use a planar Clohessy-Wiltshire relative-motion model. They
are teaching illustrations, not mission data, flight software, or a formal
spacecraft-safety result. The point is architectural: a slow planner can be
wrapped by a fast local safety layer only when timing assumptions are explicit.
"""

from __future__ import annotations

import csv
import hashlib
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "20260809" / "satellite_multirate_rta"
FIGURE = OUT / "satellite_multirate_rta_illustration.png"
TRACE = OUT / "satellite_multirate_rta_trace.csv"
REPORT = OUT / "satellite_multirate_rta_report.md"
MANIFEST = OUT / "SATELLITE_MULTIRATE_RTA_MANIFEST.sha256"

DT = 0.5
STEPS = 440
MEAN_MOTION = 1.1e-3
KEEP_OUT_RADIUS = 18.0
BUFFER_RADIUS = 23.0
MAX_ACCELERATION = 0.028
REMOTE_SAMPLE_STEPS = 20
ONE_WAY_DELAY_STEPS = 24


def clamp_norm(vector: np.ndarray, limit: float) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= limit or norm == 0.0:
        return vector
    return vector * (limit / norm)


def nominal_command(state: np.ndarray) -> np.ndarray:
    position = state[:2]
    velocity = state[2:]
    return clamp_norm(-0.0014 * position - 0.055 * velocity, MAX_ACCELERATION)


def local_safety_filter(state: np.ndarray, command: np.ndarray) -> tuple[np.ndarray, bool]:
    position = state[:2]
    velocity = state[2:]
    radius = max(float(np.linalg.norm(position)), 1.0e-9)
    radial = position / radius
    closing_speed = min(float(np.dot(radial, velocity)), 0.0)
    stopping_distance = closing_speed**2 / (2.0 * MAX_ACCELERATION)
    predicted_floor = radius - stopping_distance - 2.0
    if predicted_floor >= BUFFER_RADIUS:
        return command, False

    outward = MAX_ACCELERATION * radial
    tangential = command - np.dot(command, radial) * radial
    return clamp_norm(outward + 0.30 * tangential, MAX_ACCELERATION), True


def cw_step(state: np.ndarray, command: np.ndarray) -> np.ndarray:
    x, y, vx, vy = state
    ux, uy = command
    xdd = 3.0 * MEAN_MOTION**2 * x + 2.0 * MEAN_MOTION * vy + ux
    ydd = -2.0 * MEAN_MOTION * vx + uy
    next_velocity = np.array([vx + DT * xdd, vy + DT * ydd])
    next_position = np.array([x, y]) + DT * next_velocity
    return np.concatenate([next_position, next_velocity])


def simulate(arm: str) -> list[dict[str, float | int | str]]:
    state = np.array([-72.0, 31.0, 0.38, -0.20], dtype=float)
    rows: list[dict[str, float | int | str]] = []
    state_history: list[np.ndarray] = []
    remote_commands: dict[int, np.ndarray] = {}
    remote_retreat = False

    for step in range(STEPS):
        state_history.append(state.copy())
        proposed = nominal_command(state)
        intervention = False
        decision_age = 0.0

        if arm == "remote_slow":
            if step % REMOTE_SAMPLE_STEPS == 0:
                observed_step = max(0, step - ONE_WAY_DELAY_STEPS)
                observed_state = state_history[observed_step]
                observed_radius = float(np.linalg.norm(observed_state[:2]))
                observed_radial_speed = float(
                    np.dot(observed_state[:2], observed_state[2:])
                    / max(observed_radius, 1.0e-9)
                )
                projected_radius = observed_radius + 22.0 * min(observed_radial_speed, 0.0)
                activate = projected_radius < BUFFER_RADIUS
                arrival = step + ONE_WAY_DELAY_STEPS
                radial = observed_state[:2] / max(observed_radius, 1.0e-9)
                remote_commands[arrival] = MAX_ACCELERATION * radial if activate else np.array([np.nan, np.nan])

            if step in remote_commands:
                received = remote_commands[step]
                remote_retreat = bool(np.isfinite(received).all())
            if remote_retreat:
                radius = max(float(np.linalg.norm(state[:2])), 1.0e-9)
                proposed = MAX_ACCELERATION * state[:2] / radius
                intervention = True
            decision_age = min(step, 2 * ONE_WAY_DELAY_STEPS) * DT
        else:
            proposed, intervention = local_safety_filter(state, proposed)
            decision_age = DT

        state = cw_step(state, proposed)
        radius = float(np.linalg.norm(state[:2]))
        rows.append(
            {
                "arm": arm,
                "step": step,
                "time_s": step * DT,
                "x_m": state[0],
                "y_m": state[1],
                "vx_mps": state[2],
                "vy_mps": state[3],
                "ux_mps2": proposed[0],
                "uy_mps2": proposed[1],
                "distance_m": radius,
                "inside_keep_out": int(radius < KEEP_OUT_RADIUS),
                "intervention": int(intervention),
                "decision_age_s": decision_age,
            }
        )
    return rows


def draw_satellite(ax: plt.Axes, x: float, y: float, scale: float, color: str) -> None:
    ax.add_patch(Rectangle((x - 0.65 * scale, y - 0.38 * scale), 1.3 * scale, 0.76 * scale,
                           facecolor=color, edgecolor="white", linewidth=1.0, zorder=7))
    ax.add_patch(Rectangle((x - 2.2 * scale, y - 0.27 * scale), 1.35 * scale, 0.54 * scale,
                           facecolor="#173f5f", edgecolor="white", linewidth=0.7, zorder=6))
    ax.add_patch(Rectangle((x + 0.85 * scale, y - 0.27 * scale), 1.35 * scale, 0.54 * scale,
                           facecolor="#173f5f", edgecolor="white", linewidth=0.7, zorder=6))


def panel(
    ax: plt.Axes,
    rows: list[dict[str, float | int | str]],
    title: str,
    color: str,
    action_label: str,
) -> None:
    x = np.array([float(row["x_m"]) for row in rows])
    y = np.array([float(row["y_m"]) for row in rows])
    unsafe = np.array([int(row["inside_keep_out"]) for row in rows], dtype=bool)
    intervention = np.array([int(row["intervention"]) for row in rows], dtype=bool)

    ax.add_patch(Circle((0, 0), KEEP_OUT_RADIUS, facecolor="#fee2e2", edgecolor="#c43d2f",
                        linewidth=2.0, linestyle="--", zorder=0))
    ax.add_patch(Circle((0, 0), BUFFER_RADIUS, facecolor="none", edgecolor="#d6a34a",
                        linewidth=1.4, linestyle=":", zorder=0))
    ax.plot(x, y, color=color, linewidth=3.0, zorder=3)
    ax.scatter(x[::20], y[::20], s=17, color=color, edgecolor="white", linewidth=0.5, zorder=4)
    if unsafe.any():
        ax.scatter(x[unsafe], y[unsafe], s=28, color="#c43d2f", zorder=5)
    if intervention.any():
        first = int(np.flatnonzero(intervention)[0])
        ax.scatter([x[first]], [y[first]], marker="D", s=70, color="#7c3aed", edgecolor="white", zorder=8)
        ax.annotate(action_label, (x[first], y[first]), xytext=(10, 13), textcoords="offset points",
                    fontsize=9, color="#6d28d9", weight="bold")

    draw_satellite(ax, 0, 0, 2.1, "#667085")
    draw_satellite(ax, x[0], y[0], 1.45, color)
    ax.scatter([x[0]], [y[0]], marker="o", s=30, color=color, zorder=8)
    ax.text(0, -2.0, "target", ha="center", va="top", fontsize=9, color="#344054")
    ax.text(-15.8, 13.5, "KEEP-OUT", color="#a93226", fontsize=9, weight="bold")
    ax.text(-19.5, 20.0, "buffer", color="#9a6700", fontsize=8)
    ax.set_title(title, fontsize=15, weight="bold", color="#182230", pad=12)
    ax.set_xlim(-82, 40)
    ax.set_ylim(-47, 50)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("radial relative position (m)", fontsize=9)
    ax.set_ylabel("along-track position (m)", fontsize=9)
    ax.grid(True, color="#d0d5dd", linewidth=0.7, alpha=0.8)
    ax.set_facecolor("#ffffff")


def timing_lane(ax: plt.Axes, y: float, labels: list[str], colors: list[str], widths: list[float]) -> None:
    x = 0.02
    for index, (label, color, width) in enumerate(zip(labels, colors, widths, strict=True)):
        box = Rectangle((x, y), width, 0.25, transform=ax.transAxes, facecolor=color,
                        edgecolor="none", zorder=2)
        ax.add_patch(box)
        ax.text(x + width / 2, y + 0.125, label, transform=ax.transAxes, ha="center", va="center",
                fontsize=9, color="white" if color != "#e4e7ec" else "#344054", weight="bold")
        x += width + 0.012
        if index < len(labels) - 1:
            ax.add_patch(FancyArrowPatch((x - 0.010, y + 0.125), (x + 0.004, y + 0.125),
                                         transform=ax.transAxes, arrowstyle="-|>", mutation_scale=10,
                                         linewidth=1.0, color="#667085"))


def render(remote: list[dict[str, float | int | str]], local: list[dict[str, float | int | str]]) -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.labelcolor": "#475467", "xtick.color": "#667085", "ytick.color": "#667085"})
    fig = plt.figure(figsize=(16, 9), facecolor="#f6f8fb")
    grid = fig.add_gridspec(2, 2, height_ratios=[3.3, 1.0], hspace=0.24, wspace=0.16,
                           left=0.055, right=0.965, top=0.84, bottom=0.075)
    left = fig.add_subplot(grid[0, 0])
    right = fig.add_subplot(grid[0, 1])
    timing = fig.add_subplot(grid[1, :])
    timing.axis("off")

    panel(left, remote, "Slow remote assurance loop", "#d97706", "late retreat arrives")
    panel(right, local, "Fast onboard modular shield", "#147d92", "local shield acts")

    remote_min = min(float(row["distance_m"]) for row in remote)
    local_min = min(float(row["distance_m"]) for row in local)
    remote_unsafe = sum(int(row["inside_keep_out"]) for row in remote)
    local_unsafe = sum(int(row["inside_keep_out"]) for row in local)
    left.text(0.02, 0.02, f"minimum distance {remote_min:.1f} m  |  keep-out steps {remote_unsafe}",
              transform=left.transAxes, fontsize=10, color="#a93226", weight="bold")
    right.text(0.02, 0.02, f"minimum distance {local_min:.1f} m  |  keep-out steps {local_unsafe}",
               transform=right.transAxes, fontsize=10, color="#0f766e", weight="bold")

    timing.text(0.01, 0.84, "Timing path", transform=timing.transAxes, fontsize=11,
                color="#344054", weight="bold")
    timing_lane(timing, 0.50, ["REMOTE\nagentic planner", "uplink / queue", "remote monitor", "downlink / queue", "actuator"],
                ["#b45309", "#c2410c", "#9a3412", "#c2410c", "#667085"],
                [0.15, 0.15, 0.16, 0.15, 0.12])
    timing_lane(timing, 0.15, ["ONBOARD\nagentic planner", "command queue", "local sampled-data shield", "actuator"],
                ["#147d92", "#e4e7ec", "#0f766e", "#667085"],
                [0.15, 0.14, 0.25, 0.12])
    timing.text(0.80, 0.52, "decision arrives after\nthe recoverability window", transform=timing.transAxes,
                fontsize=10, color="#b42318", weight="bold", va="center")
    timing.text(0.80, 0.17, "local shield closes the\nsafety loop onboard", transform=timing.transAxes,
                fontsize=10, color="#0f766e", weight="bold", va="center")

    fig.text(0.055, 0.935, "Why modular RTA helps fast satellite robots", fontsize=27,
             weight="bold", color="#182230")
    fig.text(0.055, 0.892, "The planner may be slow. The safety loop cannot be slower than the recoverability window.",
             fontsize=14, color="#526273")
    fig.text(0.955, 0.935, "ARCHITECTURE TRANSFERS. PROOF MUST BE REBUILT.", ha="right",
             fontsize=10, color="#7c3aed", weight="bold",
             bbox={"boxstyle": "round,pad=0.45", "facecolor": "#f3e8ff", "edgecolor": "#d8b4fe"})
    fig.text(0.055, 0.025,
             "Illustrative planar Hill-frame trace, not mission data or certification evidence. A spacecraft shield must include sampling, delay, orbital dynamics, navigation uncertainty, fuel, and discrete actuation.",
             fontsize=9.5, color="#667085")
    fig.savefig(FIGURE, dpi=180, facecolor=fig.get_facecolor())
    plt.close(fig)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    remote = simulate("remote_slow")
    local = simulate("local_modular")
    fieldnames = list(remote[0].keys())
    with TRACE.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(remote + local)
    render(remote, local)

    remote_min = min(float(row["distance_m"]) for row in remote)
    local_min = min(float(row["distance_m"]) for row in local)
    remote_unsafe = sum(int(row["inside_keep_out"]) for row in remote)
    local_unsafe = sum(int(row["inside_keep_out"]) for row in local)
    REPORT.write_text(
        "\n".join(
            [
                "# Multirate Satellite-RTA Illustration",
                "",
                "This artifact is a teaching illustration using a planar Clohessy-Wiltshire relative-motion model. It is not mission data, flight software, or certification evidence.",
                "",
                f"- Slow remote loop: minimum distance {remote_min:.3f} m; {remote_unsafe} steps inside the illustrative {KEEP_OUT_RADIUS:.1f} m keep-out zone.",
                f"- Fast onboard modular shield: minimum distance {local_min:.3f} m; {local_unsafe} steps inside the keep-out zone.",
                "",
                "The comparison is architectural. It illustrates why planner, communication, monitoring, and actuator latencies belong in each module's assurance contract. It does not establish a general performance or safety result for satellite systems.",
            ]
        ),
        encoding="utf-8",
    )
    entries = [
        (sha256(FIGURE), FIGURE.name),
        (sha256(TRACE), TRACE.name),
        (sha256(REPORT), REPORT.name),
        (sha256(Path(__file__)), f"analysis/{Path(__file__).name}"),
    ]
    MANIFEST.write_text("".join(f"{digest}  {label}\n" for digest, label in entries), encoding="ascii")
    print(f"figure={FIGURE}")
    print(f"remote_min={remote_min:.3f}, remote_keep_out_steps={remote_unsafe}")
    print(f"local_min={local_min:.3f}, local_keep_out_steps={local_unsafe}")


if __name__ == "__main__":
    main()
