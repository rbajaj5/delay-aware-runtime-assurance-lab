"""Exploratory 3D helicopter landing coordination under delayed clearances.

This is a kinematic communication experiment, not a validated helicopter
flight-dynamics model. It compares a delay-blind clearance consumer with a
queue-aware consumer that rejects stale or expired clearances.
"""

from __future__ import annotations

import csv
import hashlib
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, FFMpegWriter


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "20260802" / "helicopter_3d"
STEPS = 180
N_HELICOPTERS = 8
N_PADS = 3
FPS = 15
MAX_FRESHNESS = 3


@dataclass(frozen=True)
class Clearance:
    helicopter: int
    pad: int
    issued: int
    delivered: int
    slot_start: int
    slot_end: int
    version: int


PADS = np.array(
    [
        [4.0, 6.0, 0.5],
        [10.0, 6.0, 0.5],
        [7.0, 12.0, 0.5],
    ],
    dtype=float,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make_clearances() -> list[Clearance]:
    # The same nominal pad schedule is replayed under both policies. Delays
    # are intentionally heterogeneous so late arrivals can be observed.
    delays = [1, 5, 0, 4, 1, 6, 2, 5]
    clearances: list[Clearance] = []
    for heli in range(N_HELICOPTERS):
        group = heli // N_PADS
        pad = heli % N_PADS
        slot_start = 12 + group * 14 + pad * 2
        issued = slot_start - 2
        clearances.append(
            Clearance(
                helicopter=heli,
                pad=pad,
                issued=issued,
                delivered=issued + delays[heli],
                slot_start=slot_start,
                slot_end=slot_start + 12,
                version=heli + 1,
            )
        )
    return clearances


def initial_positions() -> np.ndarray:
    positions = np.zeros((N_HELICOPTERS, 3), dtype=float)
    for heli in range(N_HELICOPTERS):
        pad = PADS[heli % N_PADS]
        angle = (heli * 2.0 * math.pi / N_HELICOPTERS) + 0.2
        positions[heli] = pad + np.array([math.cos(angle) * 5.0, math.sin(angle) * 5.0, 7.5])
    return positions


def simulate(queue_aware: bool) -> tuple[list[dict[str, object]], dict[str, object]]:
    clearances = make_clearances()
    positions = initial_positions()
    phases = ["hold"] * N_HELICOPTERS
    accepted = [False] * N_HELICOPTERS
    rejected = [False] * N_HELICOPTERS
    decisions = ["waiting"] * N_HELICOPTERS
    rows: list[dict[str, object]] = []
    conflict_pair_steps = 0
    conflict_step_indices: set[int] = set()
    conflict_pairs: set[tuple[int, int, int]] = set()

    for step in range(STEPS):
        for heli, clearance in enumerate(clearances):
            if not accepted[heli] and not rejected[heli] and step >= clearance.delivered:
                age = step - clearance.issued
                if queue_aware and (age > MAX_FRESHNESS or step > clearance.slot_end):
                    rejected[heli] = True
                    decisions[heli] = "reject_stale"
                else:
                    accepted[heli] = True
                    phases[heli] = "approach"
                    decisions[heli] = "accept_clearance"

        for heli, clearance in enumerate(clearances):
            if phases[heli] != "approach":
                continue
            target = PADS[clearance.pad].copy()
            target[2] = 0.8
            delta = target - positions[heli]
            distance = float(np.linalg.norm(delta))
            if distance <= 0.24:
                positions[heli] = target
                phases[heli] = "landed"
                decisions[heli] = "landed"
            else:
                positions[heli] += delta * min(0.18, 0.24 / max(distance, 1e-9))

        for a in range(N_HELICOPTERS):
            for b in range(a + 1, N_HELICOPTERS):
                if clearances[a].pad != clearances[b].pad:
                    continue
                active_a = phases[a] in {"approach", "landed"}
                active_b = phases[b] in {"approach", "landed"}
                if active_a and active_b:
                    separation = float(np.linalg.norm(positions[a] - positions[b]))
                    if separation < 2.0:
                        conflict_pair_steps += 1
                        conflict_step_indices.add(step)
                        conflict_pairs.add((clearances[a].pad, a, b))

        for heli, clearance in enumerate(clearances):
            rows.append(
                {
                    "policy": "queue_aware" if queue_aware else "delay_blind",
                    "step": step,
                    "helicopter": heli,
                    "pad": clearance.pad,
                    "x": positions[heli, 0],
                    "y": positions[heli, 1],
                    "z": positions[heli, 2],
                    "phase": phases[heli],
                    "clearance_issued": clearance.issued,
                    "clearance_delivered": clearance.delivered,
                    "message_age": max(0, step - clearance.issued),
                    "slot_start": clearance.slot_start,
                    "slot_end": clearance.slot_end,
                    "clearance_version": clearance.version,
                    "decision": decisions[heli],
                    "accepted": int(accepted[heli]),
                    "stale_rejected": int(rejected[heli]),
                    "conflict_step": int(step in conflict_step_indices),
                }
            )

    summary = {
        "policy": "queue_aware" if queue_aware else "delay_blind",
        "helicopters": N_HELICOPTERS,
        "landing_pads": N_PADS,
        "clearance_messages": N_HELICOPTERS,
        "accepted_clearances": sum(accepted),
        "stale_rejections": sum(rejected),
        "landed_helicopters": sum(phase == "landed" for phase in phases),
        "conflict_steps": len(conflict_step_indices),
        "conflict_pair_steps": conflict_pair_steps,
        "conflict_pairs": len(conflict_pairs),
        "max_clearance_age": max(row["message_age"] for row in rows),
        "freshness_threshold": MAX_FRESHNESS,
    }
    return rows, summary


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def frame_rows(rows: list[dict[str, object]], step: int) -> np.ndarray:
    selected = [row for row in rows if int(row["step"]) == step]
    return np.array([[float(row["x"]), float(row["y"]), float(row["z"])] for row in selected])


def draw_helicopter(ax, position: np.ndarray, color: str, phase: str) -> None:
    x, y, z = position
    rotor = 0.65
    body = 0.42
    ax.plot([x - body, x + body], [y, y], [z, z], color=color, linewidth=2.2)
    ax.plot([x, x], [y - rotor, y + rotor], [z + 0.12, z + 0.12], color=color, linewidth=1.5)
    ax.scatter([x], [y], [z], color=color, s=32 if phase != "landed" else 44, depthshade=False)


def render(rows_blind: list[dict[str, object]], rows_aware: list[dict[str, object]], summary: dict[str, dict[str, object]]) -> tuple[Path, Path]:
    video = OUT / "helicopter_3d_delay_comparison.mp4"
    poster = OUT / "helicopter_3d_delay_comparison_poster.png"
    colors = plt.cm.tab10(np.linspace(0, 1, N_HELICOPTERS))
    fig = plt.figure(figsize=(15, 8), constrained_layout=True)
    axes = [fig.add_subplot(1, 2, 1, projection="3d"), fig.add_subplot(1, 2, 2, projection="3d")]

    def draw_panel(ax, rows, policy, step):
        ax.clear()
        ax.set_xlim(0, 14)
        ax.set_ylim(0, 17)
        ax.set_zlim(0, 9)
        ax.set_xlabel("east")
        ax.set_ylabel("north")
        ax.set_zlabel("altitude")
        ax.view_init(elev=26, azim=-62)
        ax.set_title(policy.replace("_", " ").title())
        for pad_id, pad in enumerate(PADS):
            ax.scatter([pad[0]], [pad[1]], [pad[2]], marker="s", s=130, color="black", depthshade=False)
            ax.text(pad[0], pad[1], pad[2] + 0.35, f"P{pad_id + 1}", fontsize=8)
        current = [row for row in rows if int(row["step"]) == step]
        for heli, row in enumerate(current):
            pos = np.array([float(row["x"]), float(row["y"]), float(row["z"])])
            if step > 0:
                trail = [
                    [float(prev["x"]), float(prev["y"]), float(prev["z"])]
                    for prev in rows
                    if int(prev["helicopter"]) == heli and int(prev["step"]) <= step
                ]
                trail = np.array(trail)
                ax.plot(trail[:, 0], trail[:, 1], trail[:, 2], color=colors[heli], alpha=0.28, linewidth=0.9)
            draw_helicopter(ax, pos, colors[heli], str(row["phase"]))
            ax.text(pos[0], pos[1], pos[2] + 0.3, f"H{heli + 1}", fontsize=7)
        conflicts_so_far = len(
            {
                int(row["step"])
                for row in rows
                if int(row["step"]) <= step and int(row["conflict_step"])
            }
        )
        rejected_so_far = sum(int(row["stale_rejected"]) for row in current)
        ax.text2D(
            0.02,
            0.97,
            f"step {step:03d} | conflict steps {conflicts_so_far} | rejected {rejected_so_far}",
            transform=ax.transAxes,
            fontsize=9,
            va="top",
        )

    def update(step):
        draw_panel(axes[0], rows_blind, "delay_blind", step)
        draw_panel(axes[1], rows_aware, "queue_aware", step)
        fig.suptitle("3D helicopter landing coordination under delayed clearances", fontsize=15)

    update(0)
    fig.savefig(poster, dpi=180)
    animation = FuncAnimation(fig, update, frames=STEPS, interval=1000 / FPS, blit=False)
    writer = FFMpegWriter(fps=FPS, bitrate=1800, metadata={"title": "Logged kinematic helicopter coordination replay"})
    animation.save(video, writer=writer, dpi=120)
    plt.close(fig)
    return video, poster


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows_blind, summary_blind = simulate(queue_aware=False)
    rows_aware, summary_aware = simulate(queue_aware=True)
    trace = OUT / "helicopter_3d_step_trace.csv"
    summary_csv = OUT / "helicopter_3d_summary.csv"
    report = OUT / "helicopter_3d_report.md"
    write_csv(trace, rows_blind + rows_aware)
    write_csv(summary_csv, [summary_blind, summary_aware])
    video, poster = render(rows_blind, rows_aware, {"delay_blind": summary_blind, "queue_aware": summary_aware})

    report.write_text(
        "\n".join(
            [
                "# 3D Helicopter Landing Coordination",
                "",
                "This is an exploratory kinematic communication experiment, not a validated helicopter flight-dynamics model.",
                "Both policies use the same eight helicopters, three pads, clearance schedule, and heterogeneous message delays.",
                "The queue-aware policy rejects clearances older than the three-step freshness threshold or past their slot expiry.",
                "",
                "| policy | accepted | stale rejected | landed | conflict steps | conflict pairs | pair-conflict count |",
                "|---|---:|---:|---:|---:|---:|---:|",
                *[
                    f"| {s['policy']} | {s['accepted_clearances']} | {s['stale_rejections']} | {s['landed_helicopters']} | {s['conflict_steps']} | {s['conflict_pairs']} | {s['conflict_pair_steps']} |"
                    for s in (summary_blind, summary_aware)
                ],
                "",
                "The comparison is a mechanism diagnostic: freshness checks trade some late clearances for fewer pad conflicts in this toy schedule.",
                "It should not be interpreted as evidence about real helicopter handling qualities or aviation operations.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    manifest = OUT / "HELICOPTER_3D_MANIFEST.sha256"
    outputs = [trace, summary_csv, video, poster, report, Path(__file__)]
    manifest.write_text(
        "".join(
            f"{sha256(path)}  {str(path.relative_to(ROOT)).replace(chr(92), '/') }\n"
            for path in outputs
        ),
        encoding="ascii",
    )
    print(f"Video: {video}")
    print(f"Poster: {poster}")
    print(f"Summary: {summary_csv}")
    print(f"Manifest: {manifest}")
    print(summary_blind)
    print(summary_aware)


if __name__ == "__main__":
    main()
