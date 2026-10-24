"""Render a captioned MP4 from the retained satellite multirate trace."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter
from matplotlib.patches import Circle, Rectangle


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "20260809" / "satellite_multirate_rta"
TRACE = OUT / "satellite_multirate_rta_trace.csv"
MOVIE = OUT / "satellite_multirate_rta_animation.mp4"
MANIFEST = OUT / "SATELLITE_MULTIRATE_RTA_VIDEO_MANIFEST.sha256"

FPS = 12
KEEP_OUT_RADIUS = 18.0
BUFFER_RADIUS = 23.0
X_LIMITS = (-82.0, 40.0)
Y_LIMITS = (-47.0, 50.0)


def read_trace() -> dict[str, list[dict[str, float | int | str]]]:
    arms: dict[str, list[dict[str, float | int | str]]] = {
        "remote_slow": [],
        "local_modular": [],
    }
    with TRACE.open(newline="", encoding="utf-8") as handle:
        for raw in csv.DictReader(handle):
            row: dict[str, float | int | str] = {
                "arm": raw["arm"],
                "step": int(raw["step"]),
                "time_s": float(raw["time_s"]),
                "x_m": float(raw["x_m"]),
                "y_m": float(raw["y_m"]),
                "distance_m": float(raw["distance_m"]),
                "inside_keep_out": int(raw["inside_keep_out"]),
                "intervention": int(raw["intervention"]),
                "decision_age_s": float(raw["decision_age_s"]),
            }
            arms[str(row["arm"])].append(row)
    expected = {len(rows) for rows in arms.values()}
    if expected != {440}:
        raise RuntimeError(f"Unexpected retained trace lengths: {expected}")
    return arms


def draw_target(ax: plt.Axes) -> None:
    ax.add_patch(Rectangle((-1.3, -0.8), 2.6, 1.6, facecolor="#667085", edgecolor="white", zorder=6))
    ax.add_patch(Rectangle((-6.0, -0.45), 4.0, 0.9, facecolor="#173f5f", edgecolor="white", zorder=5))
    ax.add_patch(Rectangle((2.0, -0.45), 4.0, 0.9, facecolor="#173f5f", edgecolor="white", zorder=5))
    ax.text(0, -2.6, "target", ha="center", va="top", fontsize=9, color="#344054")


def draw_panel(ax: plt.Axes, title: str) -> None:
    ax.add_patch(Circle((0, 0), KEEP_OUT_RADIUS, facecolor="#fee2e2", edgecolor="#c43d2f",
                        linewidth=2.0, linestyle="--", zorder=0))
    ax.add_patch(Circle((0, 0), BUFFER_RADIUS, facecolor="none", edgecolor="#d69e2e",
                        linewidth=1.4, linestyle=":", zorder=0))
    draw_target(ax)
    ax.text(-16.3, 13.7, "KEEP-OUT", color="#b42318", fontsize=9, weight="bold")
    ax.text(-19.6, 20.4, "buffer", color="#9a6700", fontsize=8)
    ax.set_title(title, fontsize=15, color="#182230", weight="bold", pad=10)
    ax.set_xlim(*X_LIMITS)
    ax.set_ylim(*Y_LIMITS)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, color="#d0d5dd", linewidth=0.7, alpha=0.8)
    ax.set_xlabel("radial relative position (m)", fontsize=9)
    ax.set_ylabel("along-track position (m)", fontsize=9)
    ax.tick_params(labelsize=8, colors="#667085")
    ax.set_facecolor("white")


def add_chaser(ax: plt.Axes, color: str) -> tuple[Rectangle, Rectangle, Rectangle]:
    body = Rectangle((0, 0), 2.4, 1.4, facecolor=color, edgecolor="white", linewidth=0.8, zorder=8)
    left = Rectangle((0, 0), 3.2, 0.8, facecolor="#173f5f", edgecolor="white", linewidth=0.6, zorder=7)
    right = Rectangle((0, 0), 3.2, 0.8, facecolor="#173f5f", edgecolor="white", linewidth=0.6, zorder=7)
    for patch in (body, left, right):
        ax.add_patch(patch)
    return body, left, right


def move_chaser(parts: tuple[Rectangle, Rectangle, Rectangle], x: float, y: float) -> None:
    body, left, right = parts
    body.set_xy((x - 1.2, y - 0.7))
    left.set_xy((x - 4.8, y - 0.4))
    right.set_xy((x + 1.6, y - 0.4))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def render(arms: dict[str, list[dict[str, float | int | str]]]) -> None:
    remote = arms["remote_slow"]
    local = arms["local_modular"]
    remote_x = np.array([float(row["x_m"]) for row in remote])
    remote_y = np.array([float(row["y_m"]) for row in remote])
    local_x = np.array([float(row["x_m"]) for row in local])
    local_y = np.array([float(row["y_m"]) for row in local])

    plt.rcParams.update({"font.family": "DejaVu Sans"})
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 7.2), dpi=100, facecolor="#f6f8fb")
    fig.subplots_adjust(left=0.06, right=0.96, top=0.78, bottom=0.19, wspace=0.22)
    draw_panel(axes[0], "Slow remote assurance loop")
    draw_panel(axes[1], "Fast onboard modular shield")

    remote_line, = axes[0].plot([], [], color="#d97706", linewidth=2.8, zorder=3)
    local_line, = axes[1].plot([], [], color="#147d92", linewidth=2.8, zorder=3)
    remote_unsafe = axes[0].scatter([], [], s=22, color="#c43d2f", zorder=5)
    local_unsafe = axes[1].scatter([], [], s=22, color="#c43d2f", zorder=5)
    remote_parts = add_chaser(axes[0], "#d97706")
    local_parts = add_chaser(axes[1], "#147d92")

    remote_status = axes[0].text(0.02, 0.94, "", transform=axes[0].transAxes, fontsize=10,
                                 weight="bold", va="top")
    local_status = axes[1].text(0.02, 0.94, "", transform=axes[1].transAxes, fontsize=10,
                                weight="bold", va="top")
    remote_metrics = axes[0].text(0.02, 0.02, "", transform=axes[0].transAxes, fontsize=9,
                                  color="#9a3412", weight="bold")
    local_metrics = axes[1].text(0.02, 0.02, "", transform=axes[1].transAxes, fontsize=9,
                                 color="#0f766e", weight="bold")

    title = fig.text(0.055, 0.93, "Fast satellite robots need a faster local safety loop",
                     fontsize=23, color="#182230", weight="bold")
    subtitle = fig.text(0.055, 0.878,
                        "Same nominal planner and Hill-frame dynamics; only the assurance-loop placement changes.",
                        fontsize=12, color="#526273")
    clock = fig.text(0.945, 0.93, "", ha="right", fontsize=11, color="#475467", weight="bold")
    caption = fig.text(0.055, 0.105, "", fontsize=11, color="#344054")
    footer = fig.text(0.055, 0.045,
                      "Illustrative retained trace, not mission data or certification evidence. Keep-out radius: 18 m; buffer: 23 m.",
                      fontsize=8.5, color="#667085")

    end_heading = fig.text(0.5, 0.70, "Same planner. Different safety-loop placement.", ha="center",
                           fontsize=24, color="#182230", weight="bold", visible=False)
    end_left = fig.text(0.23, 0.43, "", ha="center", fontsize=16, color="#b42318", visible=False)
    end_right = fig.text(0.77, 0.43, "", ha="center", fontsize=16, color="#0f766e", visible=False)
    end_note = fig.text(0.5, 0.20,
                        "Architecture can be modular. Its sampling, delay, orbital, and actuation assumptions still require a new proof.",
                        ha="center", fontsize=11, color="#475467", visible=False)

    frame_steps = list(range(0, 440, 2))
    hold_frames = 24
    writer = FFMpegWriter(fps=FPS, codec="libx264", bitrate=2800,
                          metadata={"title": "Multirate RTA for satellite robots"},
                          extra_args=["-pix_fmt", "yuv420p", "-movflags", "+faststart"])
    with writer.saving(fig, str(MOVIE), dpi=100):
        for frame_index, step in enumerate(frame_steps):
            rrow = remote[step]
            lrow = local[step]
            remote_line.set_data(remote_x[: step + 1], remote_y[: step + 1])
            local_line.set_data(local_x[: step + 1], local_y[: step + 1])
            remote_mask = np.array([int(row["inside_keep_out"]) for row in remote[: step + 1]], dtype=bool)
            local_mask = np.array([int(row["inside_keep_out"]) for row in local[: step + 1]], dtype=bool)
            remote_unsafe.set_offsets(np.c_[remote_x[: step + 1][remote_mask], remote_y[: step + 1][remote_mask]])
            local_unsafe.set_offsets(np.c_[local_x[: step + 1][local_mask], local_y[: step + 1][local_mask]])
            move_chaser(remote_parts, float(rrow["x_m"]), float(rrow["y_m"]))
            move_chaser(local_parts, float(lrow["x_m"]), float(lrow["y_m"]))

            remote_count = int(remote_mask.sum())
            local_count = int(local_mask.sum())
            remote_min = min(float(row["distance_m"]) for row in remote[: step + 1])
            local_min = min(float(row["distance_m"]) for row in local[: step + 1])
            remote_action = bool(int(rrow["intervention"]))
            local_action = bool(int(lrow["intervention"]))
            remote_status.set_text("LATE RETREAT ARRIVED" if remote_action else "REMOTE DECISION IN TRANSIT")
            remote_status.set_color("#7c3aed" if remote_action else "#b45309")
            local_status.set_text("LOCAL SHIELD ACTIVE" if local_action else "NOMINAL COMMAND")
            local_status.set_color("#7c3aed" if local_action else "#147d92")
            remote_metrics.set_text(
                f"minimum {remote_min:4.1f} m  |  keep-out steps {remote_count:3d}  |  data age {float(rrow['decision_age_s']):4.1f} s"
            )
            local_metrics.set_text(
                f"minimum {local_min:4.1f} m  |  keep-out steps {local_count:3d}  |  data age {float(lrow['decision_age_s']):3.1f} s"
            )
            clock.set_text(f"t = {float(rrow['time_s']):5.1f} s")

            progress = frame_index / max(len(frame_steps) - 1, 1)
            if progress < 0.34:
                caption.set_text("The high-level planner is identical in both arms. The assurance path is not.")
            elif progress < 0.70:
                caption.set_text("Left: the decision crosses an uplink and downlink queue. Right: the shield acts onboard at the current sample.")
            else:
                caption.set_text("A continuous-time safety idea must be rebuilt for sampling, delay, orbital dynamics, and discrete thrust.")
            writer.grab_frame()

        for ax in axes:
            ax.set_visible(False)
        for artist in (title, subtitle, clock, caption):
            artist.set_visible(False)
        end_heading.set_visible(True)
        end_left.set_text("Slow remote loop\n99 keep-out steps\nminimum distance 7.5 m")
        end_right.set_text("Fast onboard shield\n0 keep-out steps\nminimum distance 24.9 m")
        for artist in (end_left, end_right, end_note):
            artist.set_visible(True)
        footer.set_text("Illustrative planar Hill-frame trace. The comparison demonstrates timing placement; it does not certify a satellite system.")
        for _ in range(hold_frames):
            writer.grab_frame()
    plt.close(fig)


def main() -> None:
    if not TRACE.exists():
        raise FileNotFoundError(f"Retained trace not found: {TRACE}")
    arms = read_trace()
    render(arms)
    entries = [
        (sha256(MOVIE), MOVIE.name),
        (sha256(TRACE), TRACE.name),
        (sha256(Path(__file__)), f"analysis/{Path(__file__).name}"),
    ]
    MANIFEST.write_text("".join(f"{digest}  {label}\n" for digest, label in entries), encoding="ascii")
    print(f"movie={MOVIE}")


if __name__ == "__main__":
    main()
