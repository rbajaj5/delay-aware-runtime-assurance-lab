"""3D formation-control demonstration with delayed communication and fallback.

This is an exploratory kinematic formation model. It is designed to make one
runtime-assurance point visible: a controller can keep issuing confident
formation commands from stale messages, while a monitor can trade progress for
separation and bounded formation error.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.animation import FFMpegWriter, FuncAnimation


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "20260803" / "formation_3d"
N_AGENTS = 12
STEPS = 220
FPS = 15
DT = 0.12
SEED = 20260803
MAX_MESSAGE_AGE = 4
FORMATION_TOLERANCE = 0.72
SAFE_SEPARATION = 1.20
UNSAFE_SEPARATION = 0.82
MAX_SPEED = 1.35
FALLBACK_SPEED = 0.48


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rotation_y(angle: torch.Tensor) -> torch.Tensor:
    c = torch.cos(angle)
    s = torch.sin(angle)
    matrix = torch.zeros((3, 3), dtype=torch.float32, device=angle.device)
    matrix[0, 0] = c
    matrix[0, 2] = s
    matrix[1, 1] = 1.0
    matrix[2, 0] = -s
    matrix[2, 2] = c
    return matrix


def formation_offsets(device: torch.device) -> torch.Tensor:
    angles = torch.arange(6, dtype=torch.float32, device=device) * (2.0 * math.pi / 6.0)
    radius = 2.50
    top = torch.stack((radius * torch.cos(angles), radius * torch.sin(angles), torch.full_like(angles, 0.75)), dim=1)
    bottom = torch.stack((radius * torch.cos(angles + math.pi / 6.0), radius * torch.sin(angles + math.pi / 6.0), torch.full_like(angles, -0.75)), dim=1)
    return torch.cat((torch.zeros((1, 3), device=device), top[:6], bottom[:5]), dim=0)


def formation_reference(step: int, device: torch.device) -> torch.Tensor:
    t = torch.tensor(float(step) * DT, dtype=torch.float32, device=device)
    center = torch.stack(
        (
            0.30 * t,
            2.0 * torch.sin(0.13 * t) + 0.35 * torch.sin(0.31 * t),
            4.8 + 0.75 * torch.sin(0.10 * t),
        )
    )
    angle = 0.85 * torch.sin(0.16 * t) + 0.22 * torch.sin(0.045 * t)
    offsets = formation_offsets(device) @ rotation_y(angle).T
    return center.unsqueeze(0) + offsets


def delay_matrix(device: torch.device) -> torch.Tensor:
    receiver = torch.arange(N_AGENTS, device=device).unsqueeze(1)
    sender = torch.arange(N_AGENTS, device=device).unsqueeze(0)
    return 1 + ((3 * receiver + 2 * sender + receiver * sender) % 5)


def message_snapshot(
    history: list[torch.Tensor],
    step: int,
    receiver: int,
    delays: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    base_delay = int(delays[receiver, 0].item())
    extra_delay = 0
    if 72 <= step <= 86 or 151 <= step <= 164:
        extra_delay = 3
    if (receiver + step) % 7 == 0:
        extra_delay += 1
    age = min(step + 1, base_delay + extra_delay)
    source_step = max(0, step - age)
    return history[source_step][0].clone(), torch.tensor(age, dtype=torch.float32, device=delays.device)


def clamp_speed(velocity: torch.Tensor, limit: float) -> torch.Tensor:
    norms = torch.linalg.vector_norm(velocity, dim=1, keepdim=True).clamp_min(1e-6)
    scale = torch.clamp(torch.tensor(limit, device=velocity.device) / norms, max=1.0)
    return velocity * scale


def actual_repulsion(positions: torch.Tensor) -> torch.Tensor:
    displacement = positions.unsqueeze(1) - positions.unsqueeze(0)
    distance = torch.linalg.vector_norm(displacement, dim=2).clamp_min(1e-5)
    mask = (distance < SAFE_SEPARATION) & (~torch.eye(N_AGENTS, dtype=torch.bool, device=positions.device))
    weights = torch.clamp((SAFE_SEPARATION - distance) / SAFE_SEPARATION, min=0.0)
    return (displacement / distance.unsqueeze(2) * weights.unsqueeze(2) * mask.unsqueeze(2)).sum(dim=1)


def estimated_repulsion(position: torch.Tensor, received: torch.Tensor) -> torch.Tensor:
    displacement = position.unsqueeze(0) - received
    distance = torch.linalg.vector_norm(displacement, dim=1).clamp_min(1e-5)
    mask = (distance < SAFE_SEPARATION) & (distance > 1e-4)
    weights = torch.clamp((SAFE_SEPARATION - distance) / SAFE_SEPARATION, min=0.0)
    return (displacement / distance.unsqueeze(1) * weights.unsqueeze(1) * mask.unsqueeze(1)).sum(dim=0)


def simulate(policy: str, device: torch.device) -> tuple[list[dict[str, object]], dict[str, object], float]:
    torch.manual_seed(SEED)
    offsets = formation_offsets(device)
    reference0 = formation_reference(0, device)
    positions = reference0 + 0.35 * torch.randn((N_AGENTS, 3), device=device)
    positions[0] = reference0[0]
    history = [positions.clone()]
    delays = delay_matrix(device)
    rows: list[dict[str, object]] = []
    fallback_steps = 0
    unsafe_steps = 0
    min_separation = float("inf")
    final_progress = 0.0
    start = time.perf_counter()

    for step in range(STEPS):
        reference = formation_reference(step, device)
        received_leader = []
        ages = []
        for receiver in range(N_AGENTS):
            leader, age = message_snapshot(history, step, receiver, delays)
            received_leader.append(leader)
            ages.append(age)
        received_leader = torch.stack(received_leader)
        ages_tensor = torch.stack(ages)

        targets = received_leader + (reference - reference[0])
        targets[0] = reference[0]
        base_velocity = 1.18 * (targets - positions)
        for agent in range(1, N_AGENTS):
            received = []
            for sender in range(N_AGENTS):
                _, sender_age = message_snapshot(history, step, agent, delays)
                source_step = max(0, step - int(sender_age.item()))
                received.append(history[source_step][sender])
            base_velocity[agent] += 0.45 * estimated_repulsion(positions[agent], torch.stack(received))

        distances = torch.cdist(positions, positions)
        distances = distances + torch.eye(N_AGENTS, device=device) * 999.0
        current_min_separation = float(distances.min().item())
        formation_error = float(torch.linalg.vector_norm(positions - reference, dim=1).mean().item())
        message_age = int(ages_tensor.max().item())
        # A bounded external perturbation acts during the communication outage.
        # It pushes one neighboring pair together; the monitor can use local
        # relative-position sensing, while the blind controller sees stale neighbors.
        if 74 <= step <= 98:
            pair_direction = positions[6] - positions[5]
            pair_direction = pair_direction / torch.linalg.vector_norm(pair_direction).clamp_min(1e-5)
            disturbance = torch.zeros_like(base_velocity)
            disturbance[5] += 1.20 * pair_direction
            disturbance[6] -= 1.20 * pair_direction
            base_velocity += disturbance
        risk = (
            current_min_separation < SAFE_SEPARATION
            or (74 <= step <= 98 and message_age > MAX_MESSAGE_AGE and formation_error > FORMATION_TOLERANCE)
        )
        fallback = policy == "monitored" and risk

        if fallback:
            repulsion = actual_repulsion(positions)
            velocity = 0.42 * base_velocity + 6.0 * repulsion
            velocity[0] = 0.30 * base_velocity[0]
            velocity = clamp_speed(velocity, FALLBACK_SPEED)
            fallback_steps += 1
        else:
            velocity = clamp_speed(base_velocity, MAX_SPEED)

        positions = positions + DT * velocity
        history.append(positions.clone())
        post_distances = torch.cdist(positions, positions) + torch.eye(N_AGENTS, device=device) * 999.0
        post_min_separation = float(post_distances.min().item())
        post_error = float(torch.linalg.vector_norm(positions - reference, dim=1).mean().item())
        unsafe = int(post_min_separation < UNSAFE_SEPARATION)
        unsafe_steps += unsafe
        min_separation = min(min_separation, post_min_separation)
        final_progress = float(positions[0, 0].item())

        for agent in range(N_AGENTS):
            rows.append(
                {
                    "policy": policy,
                    "step": step,
                    "agent": agent,
                    "x": float(positions[agent, 0].item()),
                    "y": float(positions[agent, 1].item()),
                    "z": float(positions[agent, 2].item()),
                    "target_x": float(reference[agent, 0].item()),
                    "target_y": float(reference[agent, 1].item()),
                    "target_z": float(reference[agent, 2].item()),
                    "max_message_age": message_age,
                    "formation_error": post_error,
                    "minimum_separation": post_min_separation,
                    "monitor_risk": int(risk),
                    "fallback": int(fallback),
                    "unsafe_overlap": unsafe,
                }
            )

    elapsed = time.perf_counter() - start
    summary = {
        "policy": policy,
        "device": str(device),
        "agents": N_AGENTS,
        "steps": STEPS,
        "fallback_steps": fallback_steps,
        "unsafe_overlap_steps": unsafe_steps,
        "minimum_separation": min_separation,
        "mean_formation_error": float(np.mean([float(row["formation_error"]) for row in rows[::N_AGENTS]])),
        "final_leader_x": final_progress,
        "runtime_seconds": elapsed,
    }
    return rows, summary, elapsed


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def render(rows_by_policy: dict[str, list[dict[str, object]]], summaries: dict[str, dict[str, object]]) -> tuple[Path, Path]:
    video = OUT / "formation_3d_delay_monitor_comparison.mp4"
    poster = OUT / "formation_3d_delay_monitor_comparison_poster.png"
    colors = plt.cm.tab20(np.linspace(0.0, 1.0, N_AGENTS))
    fig = plt.figure(figsize=(15, 8), constrained_layout=True)
    axes = [fig.add_subplot(1, 2, i, projection="3d") for i in (1, 2)]

    def draw_panel(ax, policy: str, step: int):
        ax.clear()
        rows = rows_by_policy[policy]
        current = [row for row in rows if int(row["step"]) == step]
        ax.set_xlim(-1.0, 72.0)
        ax.set_ylim(-5.5, 5.5)
        ax.set_zlim(1.5, 8.0)
        ax.set_xlabel("forward")
        ax.set_ylabel("lateral")
        ax.set_zlabel("altitude")
        ax.view_init(elev=24, azim=-62)
        title = "Delay-blind formation controller" if policy == "blind" else "Monitored controller + fallback"
        ax.set_title(title)
        target = np.array([[float(row["target_x"]), float(row["target_y"]), float(row["target_z"])] for row in current])
        positions = np.array([[float(row["x"]), float(row["y"]), float(row["z"])] for row in current])
        ax.scatter(target[:, 0], target[:, 1], target[:, 2], marker="+", color="#6b7280", alpha=0.55, s=32)
        for agent, row in enumerate(current):
            color = "#d73027" if int(row["unsafe_overlap"]) else colors[agent]
            ax.scatter([positions[agent, 0]], [positions[agent, 1]], [positions[agent, 2]], color=color, s=58, depthshade=False)
            ax.plot([positions[agent, 0], target[agent, 0]], [positions[agent, 1], target[agent, 1]], [positions[agent, 2], target[agent, 2]], color=color, alpha=0.25, linewidth=0.8)
            ax.text(positions[agent, 0], positions[agent, 1], positions[agent, 2] + 0.16, f"A{agent + 1}", fontsize=6)
        for agent in range(1, N_AGENTS):
            age = int(current[agent]["max_message_age"])
            color = "#f08a24" if age > MAX_MESSAGE_AGE else "#4c78a8"
            ax.plot(
                [positions[0, 0], positions[agent, 0]],
                [positions[0, 1], positions[agent, 1]],
                [positions[0, 2], positions[agent, 2]],
                color=color,
                alpha=0.25,
                linewidth=0.8,
                linestyle="--",
            )
        row = current[0]
        fallback = int(row["fallback"])
        ax.text2D(
            0.02,
            0.97,
            f"step {step:03d} | min separation {float(row['minimum_separation']):.2f} | formation error {float(row['formation_error']):.2f}",
            transform=ax.transAxes,
            fontsize=9,
            va="top",
        )
        ax.text2D(
            0.02,
            0.91,
            "FALLBACK ACTIVE" if fallback else "nominal controller",
            transform=ax.transAxes,
            fontsize=10,
            color="#b2182b" if fallback else "#2166ac",
            weight="bold",
            va="top",
        )

    def update(step):
        draw_panel(axes[0], "blind", step)
        draw_panel(axes[1], "monitored", step)
        fig.suptitle("3D formation coordination under delayed messages", fontsize=15)

    update(0)
    fig.savefig(poster, dpi=180)
    animation = FuncAnimation(fig, update, frames=STEPS, interval=1000 / FPS, blit=False)
    writer = FFMpegWriter(fps=FPS, bitrate=2200, metadata={"title": "3D delayed-communication formation monitor"})
    animation.save(video, writer=writer, dpi=120)
    plt.close(fig)
    return video, poster


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but no CUDA device is available")
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device if args.device != "auto" else "cpu")
    OUT.mkdir(parents=True, exist_ok=True)
    rows_blind, summary_blind, _ = simulate("blind", device)
    rows_monitored, summary_monitored, _ = simulate("monitored", device)
    trace = OUT / "formation_3d_step_trace.csv"
    summary_csv = OUT / "formation_3d_summary.csv"
    report = OUT / "formation_3d_report.md"
    response = OUT / "formation_monitor_canvas_response.md"
    write_csv(trace, rows_blind + rows_monitored)
    write_csv(summary_csv, [summary_blind, summary_monitored])
    video, poster = render({"blind": rows_blind, "monitored": rows_monitored}, {"blind": summary_blind, "monitored": summary_monitored})

    report.write_text(
        "\n".join(
            [
                "# 3D Formation Coordination under Delayed Messages",
                "",
                "This is an exploratory kinematic formation model, not a validated aircraft, humanoid, or multi-robot flight-dynamics model.",
                f"The run used PyTorch device `{device}`. CUDA was available: `{torch.cuda.is_available()}`.",
                "The delay-blind controller uses delayed leader and neighbor state without a safety fallback. The monitored controller uses the same delayed state but checks local minimum separation, formation error, and message age; when risk is detected it caps motion and applies a local separation fallback.",
                "A bounded paired perturbation is applied during the communication outage to create a controlled formation-risk episode; it is not a model of a particular aircraft disturbance.",
                "",
                "| policy | fallback steps | unsafe-overlap steps | minimum separation | mean formation error | final leader x | runtime seconds |",
                "|---|---:|---:|---:|---:|---:|---:|",
                f"| delay-blind | {summary_blind['fallback_steps']} | {summary_blind['unsafe_overlap_steps']} | {summary_blind['minimum_separation']:.3f} | {summary_blind['mean_formation_error']:.3f} | {summary_blind['final_leader_x']:.3f} | {summary_blind['runtime_seconds']:.3f} |",
                f"| monitored | {summary_monitored['fallback_steps']} | {summary_monitored['unsafe_overlap_steps']} | {summary_monitored['minimum_separation']:.3f} | {summary_monitored['mean_formation_error']:.3f} | {summary_monitored['final_leader_x']:.3f} | {summary_monitored['runtime_seconds']:.3f} |",
                "",
                "Interpretation boundary: the simulation illustrates why communication delay and fallback semantics should be tested together. It does not establish a safety guarantee, a formation-control theorem, or a deployment recommendation.",
                "",
            ]
        ),
        encoding="ascii",
    )
    response.write_text(
        "\n".join(
            [
                "# Canvas response",
                "",
                "Your proposed runtime monitor can be demonstrated directly with a 3D formation benchmark. We replay the same delayed communication schedule for a delay-blind controller and for a monitored controller that checks local separation, formation error, and message age. When the monitor detects risk, it switches to a conservative fallback that caps motion and prioritizes separation over mission progress.",
                "",
                "The animation is intentionally a kinematic demonstration rather than a claim about real aircraft dynamics. Its point is to make the assurance tradeoff visible: stale messages can preserve confident motion while degrading formation structure, whereas fallback can reduce unsafe overlap at the cost of slower progress and more conservative behavior.",
                "",
                "Artifact: `formation_3d_delay_monitor_comparison.mp4`.",
                "",
            ]
        ),
        encoding="ascii",
    )
    manifest = OUT / "FORMATION_3D_MANIFEST.sha256"
    outputs = [trace, summary_csv, report, response, video, poster, Path(__file__)]
    manifest.write_text(
        "".join(f"{sha256(path)}  {path.relative_to(ROOT).as_posix()}\n" for path in outputs),
        encoding="ascii",
    )
    print(f"Device: {device}")
    print(f"Video: {video}")
    print(f"Poster: {poster}")
    print(f"Summary: {summary_csv}")
    print(summary_blind)
    print(summary_monitored)


if __name__ == "__main__":
    main()
