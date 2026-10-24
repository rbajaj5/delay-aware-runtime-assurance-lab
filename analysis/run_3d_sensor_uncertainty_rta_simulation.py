"""GPU-backed 3D formation demonstration for sensor uncertainty and RTA.

This is an illustrative kinematic experiment, not a reproduction of Liu et al.
or Dawood et al. One neural policy is trained on nominal observations, frozen,
and replayed against the same out-of-distribution sensor-error tape in two arms:

* ``lec_only`` trusts the shifted observations.
* ``rta`` adds uncertainty tracking, a pairwise CBF-style velocity filter, and
  a conservative fallback supported by an independent proximity channel.

All plotted values are retained in a row-level CSV. Rendering is read-only over
that retained trace; no trajectory interpolation or post-render dynamics occur.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.animation import FFMpegWriter
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "20260808" / "sensor_uncertainty_3d"
TRACE_CSV = OUT / "sensor_uncertainty_3d_step_trace.csv"
SUMMARY_CSV = OUT / "sensor_uncertainty_3d_summary.csv"
CHECKPOINT = OUT / "nominal_lec_checkpoint.pt"
TRAINING_JSON = OUT / "nominal_lec_training.json"
MOVIE = OUT / "sensor_uncertainty_rta_comparison.mp4"
POSTER = OUT / "sensor_uncertainty_rta_comparison_poster.png"
REPORT = OUT / "sensor_uncertainty_rta_report.md"
CANVAS_RESPONSE = OUT / "canvas_response_sensor_uncertainty.md"
MANIFEST = OUT / "SENSOR_UNCERTAINTY_3D_MANIFEST.sha256"

SEED = 20260808
N_AGENTS = 10
STEPS = 240
DT = 0.10
FPS = 12
SAFE_SEPARATION = 1.18
UNSAFE_SEPARATION = 0.78
FORMATION_BREAKDOWN = 1.05
RAW_SPEED_LIMIT = 1.65
FILTERED_SPEED_LIMIT = 1.35
FALLBACK_SPEED_LIMIT = 0.72


class NominalPolicy(nn.Module):
    """Small learned velocity policy trained only on nominal target errors."""

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 3),
        )

    def forward(self, target_error: torch.Tensor) -> torch.Tensor:
        return self.net(target_error)


@dataclass
class Tape:
    observation_noise: torch.Tensor
    observation_bias: torch.Tensor
    proximity_noise: torch.Tensor
    reported_sigma: torch.Tensor
    fault_envelope: torch.Tensor


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def smoothstep(value: torch.Tensor) -> torch.Tensor:
    value = value.clamp(0.0, 1.0)
    return value * value * (3.0 - 2.0 * value)


def fault_envelope(device: torch.device) -> torch.Tensor:
    step = torch.arange(STEPS, dtype=torch.float32, device=device)
    rise = smoothstep((step - 52.0) / 28.0)
    fall = 1.0 - smoothstep((step - 160.0) / 34.0)
    return rise * fall


def rotation_z(angle: torch.Tensor) -> torch.Tensor:
    c = torch.cos(angle)
    s = torch.sin(angle)
    matrix = torch.zeros((3, 3), dtype=torch.float32, device=angle.device)
    matrix[0, 0] = c
    matrix[0, 1] = -s
    matrix[1, 0] = s
    matrix[1, 1] = c
    matrix[2, 2] = 1.0
    return matrix


def base_offsets(device: torch.device) -> torch.Tensor:
    angles = torch.arange(5, dtype=torch.float32, device=device) * (2.0 * math.pi / 5.0)
    radius = 2.35
    upper = torch.stack(
        (radius * torch.cos(angles), radius * torch.sin(angles), torch.full_like(angles, 0.72)),
        dim=1,
    )
    lower = torch.stack(
        (
            radius * torch.cos(angles + math.pi / 5.0),
            radius * torch.sin(angles + math.pi / 5.0),
            torch.full_like(angles, -0.72),
        ),
        dim=1,
    )
    return torch.cat((upper, lower), dim=0)


def formation_reference(step: int, device: torch.device) -> torch.Tensor:
    t = torch.tensor(step * DT, dtype=torch.float32, device=device)
    center = torch.stack(
        (
            0.36 * t,
            1.05 * torch.sin(0.18 * t),
            4.6 + 0.55 * torch.sin(0.12 * t),
        )
    )
    yaw = 0.48 * torch.sin(0.14 * t)
    return center.unsqueeze(0) + base_offsets(device) @ rotation_z(yaw).T


def train_policy(device: torch.device) -> tuple[NominalPolicy, dict[str, float | int | str]]:
    torch.manual_seed(SEED)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(SEED)
    policy = NominalPolicy().to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=3.0e-3)
    generator = torch.Generator(device=device).manual_seed(SEED + 1)
    samples = 65_536
    target_error = 1.5 * (2.0 * torch.rand((samples, 3), generator=generator, device=device) - 1.0)
    target_velocity = torch.clamp(1.12 * target_error, -RAW_SPEED_LIMIT, RAW_SPEED_LIMIT)

    start = time.perf_counter()
    final_loss = float("nan")
    for iteration in range(500):
        indices = torch.randint(0, samples, (1_024,), generator=generator, device=device)
        prediction = policy(target_error[indices])
        loss = torch.mean((prediction - target_velocity[indices]) ** 2)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        final_loss = float(loss.detach().item())

    with torch.no_grad():
        validation = 1.5 * (2.0 * torch.rand((8_192, 3), generator=generator, device=device) - 1.0)
        target = torch.clamp(1.12 * validation, -RAW_SPEED_LIMIT, RAW_SPEED_LIMIT)
        validation_mse = float(torch.mean((policy(validation) - target) ** 2).item())

    elapsed = time.perf_counter() - start
    policy.eval()
    torch.save(
        {
            "seed": SEED,
            "device": str(device),
            "state_dict": {key: value.detach().cpu() for key, value in policy.state_dict().items()},
            "training_domain": "target errors sampled uniformly from [-1.5, 1.5]^3",
        },
        CHECKPOINT,
    )
    metrics: dict[str, float | int | str] = {
        "device": str(device),
        "seed": SEED,
        "training_samples": samples,
        "iterations": 500,
        "final_batch_mse": final_loss,
        "validation_mse": validation_mse,
        "training_seconds": elapsed,
    }
    TRAINING_JSON.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return policy, metrics


def make_tape(device: torch.device) -> Tape:
    generator = torch.Generator(device=device).manual_seed(SEED + 2)
    envelope = fault_envelope(device)
    references = torch.stack([formation_reference(step, device) for step in range(STEPS)])

    base_noise = 0.025 * torch.randn((STEPS, N_AGENTS, 3), generator=generator, device=device)
    extra_noise = (
        envelope[:, None, None]
        * 0.090
        * torch.randn((STEPS, N_AGENTS, 3), generator=generator, device=device)
    )
    observation_noise = base_noise + extra_noise
    proximity_noise = torch.clamp(
        0.018 * torch.randn((STEPS, N_AGENTS, 3), generator=generator, device=device),
        -0.045,
        0.045,
    )

    bias = torch.zeros((STEPS, N_AGENTS, 3), device=device)
    for first, second, magnitude in ((0, 1, 1.08), (5, 6, 0.85)):
        direction = references[:, first] - references[:, second]
        direction = direction / torch.linalg.vector_norm(direction, dim=1, keepdim=True).clamp_min(1.0e-6)
        bias[:, first] += envelope[:, None] * magnitude * direction
        bias[:, second] -= envelope[:, None] * magnitude * direction

    # A correlated vertical/heading drift affects two additional agents.
    phase = torch.arange(STEPS, dtype=torch.float32, device=device) * 0.09
    bias[:, 3, 2] += 0.56 * envelope * torch.sin(phase)
    bias[:, 8, 1] += 0.62 * envelope * torch.sin(0.61 * phase + 0.8)
    burst = ((torch.arange(STEPS, device=device) >= 116) & (torch.arange(STEPS, device=device) <= 136)).float()
    bias[:, 3, 0] += 0.34 * burst
    bias[:, 8, 0] -= 0.34 * burst

    reported_sigma = 0.035 + 0.12 * envelope[:, None].repeat(1, N_AGENTS)
    return Tape(
        observation_noise=observation_noise,
        observation_bias=bias,
        proximity_noise=proximity_noise,
        reported_sigma=reported_sigma,
        fault_envelope=envelope,
    )


def clamp_speed(velocity: torch.Tensor, limit: float) -> torch.Tensor:
    norm = torch.linalg.vector_norm(velocity, dim=1, keepdim=True).clamp_min(1.0e-8)
    return velocity * torch.clamp(torch.tensor(limit, device=velocity.device) / norm, max=1.0)


def pairwise_minimum(positions: torch.Tensor) -> float:
    distance = torch.cdist(positions, positions)
    distance += torch.eye(N_AGENTS, device=positions.device) * 999.0
    return float(distance.min().item())


def cbf_style_filter(
    velocity: torch.Tensor,
    local_positions: torch.Tensor,
    uncertainty: torch.Tensor,
    separation: float,
    alpha: float,
) -> tuple[torch.Tensor, int, float]:
    """Project closing pairwise velocities onto robust radial half-spaces."""

    filtered = velocity.clone()
    interventions = 0
    minimum_robust_margin = float("inf")
    for _ in range(4):
        for first in range(N_AGENTS):
            for second in range(first + 1, N_AGENTS):
                displacement = local_positions[first] - local_positions[second]
                distance = torch.linalg.vector_norm(displacement).clamp_min(1.0e-6)
                normal = displacement / distance
                robust_distance = distance - 0.22 * (uncertainty[first] + uncertainty[second])
                h = robust_distance - separation
                minimum_robust_margin = min(minimum_robust_margin, float(h.item()))
                radial_speed = torch.dot(normal, filtered[first] - filtered[second])
                lower_bound = -alpha * h
                if radial_speed < lower_bound:
                    correction = lower_bound - radial_speed
                    filtered[first] += 0.5 * correction * normal
                    filtered[second] -= 0.5 * correction * normal
                    interventions += 1
    return filtered, interventions, minimum_robust_margin


def simulate(
    arm: str,
    policy: NominalPolicy,
    tape: Tape,
    device: torch.device,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    torch.manual_seed(SEED + 3)
    generator = torch.Generator(device=device).manual_seed(SEED + 3)
    reference0 = formation_reference(0, device)
    shared_initial_noise = 0.07 * torch.randn((N_AGENTS, 3), generator=generator, device=device)
    positions = reference0 + shared_initial_noise
    velocity = torch.zeros_like(positions)
    previous_command = torch.zeros_like(positions)
    estimate = positions + tape.observation_noise[0]
    uncertainty = torch.full((N_AGENTS,), 0.06, device=device)

    rows: list[dict[str, object]] = []
    fallback_steps = 0
    filter_steps = 0
    filter_constraint_count = 0
    unsafe_steps = 0
    collision_steps = 0
    breakdown_steps = 0
    minimum_separation = float("inf")
    maximum_command = 0.0

    start = time.perf_counter()
    for step in range(STEPS):
        reference = formation_reference(step, device)
        observation = positions + tape.observation_bias[step] + tape.observation_noise[step]
        local_positions = positions + tape.proximity_noise[step]

        if arm == "lec_only":
            policy_state = observation
            uncertainty_for_filter = torch.zeros_like(uncertainty)
        else:
            prediction = estimate + DT * previous_command
            innovation = torch.linalg.vector_norm(observation - prediction, dim=1)
            uncertainty = torch.clamp(
                0.90 * uncertainty
                + 0.10 * (0.035 + 0.90 * innovation + 1.20 * tape.reported_sigma[step]),
                0.04,
                0.60,
            )
            gain = torch.clamp(0.62 / (1.0 + 4.0 * uncertainty), 0.10, 0.55).unsqueeze(1)
            estimate = prediction + gain * (observation - prediction)
            policy_state = estimate
            uncertainty_for_filter = uncertainty

        with torch.no_grad():
            nominal_command = policy(reference - policy_state)
        command = clamp_speed(nominal_command, RAW_SPEED_LIMIT)
        filter_active = False
        fallback = False
        constraint_count = 0
        robust_margin = float("nan")

        if arm == "rta":
            provisional, constraint_count, robust_margin = cbf_style_filter(
                command,
                local_positions,
                uncertainty_for_filter,
                separation=SAFE_SEPARATION,
                alpha=1.8,
            )
            filter_active = constraint_count > 0
            # Degraded mode is triggered by jointly high uncertainty and a
            # shrinking robust separation margin, not by uncertainty alone.
            fallback = bool(
                uncertainty.max().item() > 0.40
                and robust_margin < 0.0
            )
            if fallback:
                provisional = 0.45 * provisional
                provisional, extra_constraints, robust_margin = cbf_style_filter(
                    provisional,
                    local_positions,
                    uncertainty_for_filter,
                    separation=SAFE_SEPARATION + 0.10,
                    alpha=2.8,
                )
                constraint_count += extra_constraints
                command = clamp_speed(provisional, FALLBACK_SPEED_LIMIT)
                fallback_steps += 1
            else:
                command = clamp_speed(provisional, FILTERED_SPEED_LIMIT)
            if filter_active:
                filter_steps += 1
            filter_constraint_count += constraint_count

        # A first-order actuator model keeps the demonstration kinematic while
        # preventing discontinuous position jumps.
        velocity = 0.72 * velocity + 0.28 * command
        positions = positions + DT * velocity
        previous_command = command

        true_minimum = pairwise_minimum(positions)
        formation_error = float(torch.linalg.vector_norm(positions - reference, dim=1).mean().item())
        unsafe = int(true_minimum < UNSAFE_SEPARATION)
        collision = int(true_minimum < 0.48)
        breakdown = int(formation_error > FORMATION_BREAKDOWN)
        unsafe_steps += unsafe
        collision_steps += collision
        breakdown_steps += breakdown
        minimum_separation = min(minimum_separation, true_minimum)
        maximum_command = max(maximum_command, float(torch.linalg.vector_norm(command, dim=1).max().item()))

        for agent in range(N_AGENTS):
            rows.append(
                {
                    "arm": arm,
                    "step": step,
                    "agent": agent,
                    "x": float(positions[agent, 0].item()),
                    "y": float(positions[agent, 1].item()),
                    "z": float(positions[agent, 2].item()),
                    "observed_x": float(observation[agent, 0].item()),
                    "observed_y": float(observation[agent, 1].item()),
                    "observed_z": float(observation[agent, 2].item()),
                    "target_x": float(reference[agent, 0].item()),
                    "target_y": float(reference[agent, 1].item()),
                    "target_z": float(reference[agent, 2].item()),
                    "command_x": float(command[agent, 0].item()),
                    "command_y": float(command[agent, 1].item()),
                    "command_z": float(command[agent, 2].item()),
                    "uncertainty_radius": float(uncertainty[agent].item()) if arm == "rta" else 0.0,
                    "sensor_fault_envelope": float(tape.fault_envelope[step].item()),
                    "minimum_separation": true_minimum,
                    "formation_error": formation_error,
                    "robust_margin": robust_margin,
                    "filter_active": int(filter_active),
                    "fallback": int(fallback),
                    "constraint_projections": constraint_count,
                    "unsafe_separation": unsafe,
                    "collision": collision,
                    "formation_breakdown": breakdown,
                }
            )

    runtime = time.perf_counter() - start
    step_rows = rows[::N_AGENTS]
    summary: dict[str, object] = {
        "arm": arm,
        "device": str(device),
        "agents": N_AGENTS,
        "steps": STEPS,
        "unsafe_separation_steps": unsafe_steps,
        "collision_steps": collision_steps,
        "formation_breakdown_steps": breakdown_steps,
        "minimum_separation": minimum_separation,
        "mean_formation_error": float(np.mean([float(row["formation_error"]) for row in step_rows])),
        "maximum_command_norm": maximum_command,
        "filter_steps": filter_steps,
        "fallback_steps": fallback_steps,
        "constraint_projections": filter_constraint_count,
        "final_centroid_x": float(positions[:, 0].mean().item()),
        "simulation_seconds": runtime,
    }
    return rows, summary


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def step_arrays(rows: list[dict[str, object]]) -> dict[str, np.ndarray]:
    fields = (
        "x",
        "y",
        "z",
        "observed_x",
        "observed_y",
        "observed_z",
        "target_x",
        "target_y",
        "target_z",
    )
    result = {field: np.zeros((STEPS, N_AGENTS), dtype=float) for field in fields}
    scalars = {
        field: np.zeros(STEPS, dtype=float)
        for field in (
            "minimum_separation",
            "formation_error",
            "sensor_fault_envelope",
            "uncertainty_radius",
            "filter_active",
            "fallback",
            "unsafe_separation",
        )
    }
    for row in rows:
        step = int(row["step"])
        agent = int(row["agent"])
        for field in fields:
            result[field][step, agent] = float(row[field])
        if agent == 0:
            for field in scalars:
                scalars[field][step] = float(row[field])
    if rows and rows[0]["arm"] == "rta":
        for step in range(STEPS):
            start = step * N_AGENTS
            stop = start + N_AGENTS
            scalars["uncertainty_radius"][step] = max(float(row["uncertainty_radius"]) for row in rows[start:stop])
    result.update(scalars)
    return result


def formation_edges() -> list[tuple[int, int]]:
    edges: list[tuple[int, int]] = []
    for base in (0, 5):
        for index in range(5):
            edges.append((base + index, base + (index + 1) % 5))
    edges.extend((index, index + 5) for index in range(5))
    return edges


def render(
    rows_by_arm: dict[str, list[dict[str, object]]],
    summaries: dict[str, dict[str, object]],
) -> tuple[Path, Path]:
    arrays = {arm: step_arrays(rows) for arm, rows in rows_by_arm.items()}
    edges = formation_edges()
    colors = {"lec_only": "#D97706", "rta": "#277DA1"}
    titles = {
        "lec_only": "LEC only: observation -> command",
        "rta": "Uncertainty-aware RTA",
    }

    fig = plt.figure(figsize=(12.8, 7.2), facecolor="#F7F8FA")
    axes = [fig.add_subplot(1, 2, index, projection="3d") for index in (1, 2)]
    fig.subplots_adjust(left=0.025, right=0.975, top=0.83, bottom=0.17, wspace=0.03)
    title_artist = fig.text(
        0.035,
        0.955,
        "Sensor uncertainty in distributed formation control",
        fontsize=19,
        weight="bold",
        color="#17202A",
        va="top",
    )
    subtitle_artist = fig.text(
        0.035,
        0.916,
        "Same frozen neural policy | same true dynamics | same GPS/relative-sensor error tape",
        fontsize=10.5,
        color="#566573",
        va="top",
    )
    fault_artist = fig.text(
        0.965,
        0.946,
        "NOMINAL SENSOR REGIME",
        fontsize=10,
        weight="bold",
        color="#1E8449",
        ha="right",
        va="top",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "#E9F7EF", "edgecolor": "#A9DFBF"},
    )
    caption_artist = fig.text(
        0.035,
        0.072,
        "",
        fontsize=10.5,
        color="#17202A",
        va="center",
    )
    footer_artist = fig.text(
        0.035,
        0.028,
        "Illustrative GPU-backed kinematic simulation; not Liu et al.'s experiment and not a deployment guarantee.",
        fontsize=8.5,
        color="#6B7280",
    )

    def panel(ax: plt.Axes, arm: str, step: int) -> None:
        ax.clear()
        data = arrays[arm]
        position = np.column_stack((data["x"][step], data["y"][step], data["z"][step]))
        observed = np.column_stack(
            (data["observed_x"][step], data["observed_y"][step], data["observed_z"][step])
        )
        target = np.column_stack(
            (data["target_x"][step], data["target_y"][step], data["target_z"][step])
        )
        ax.set_xlim(-3.1, 12.0)
        ax.set_ylim(-4.3, 4.3)
        ax.set_zlim(1.7, 7.1)
        ax.view_init(elev=23, azim=-62)
        ax.set_box_aspect((1.65, 1.0, 0.72))
        ax.set_xlabel("forward", labelpad=2)
        ax.set_ylabel("lateral", labelpad=2)
        ax.set_zlabel("altitude", labelpad=2)
        ax.tick_params(labelsize=7, pad=0)
        ax.set_title(titles[arm], fontsize=12.5, weight="bold", color="#17202A", pad=8)

        ax.scatter(target[:, 0], target[:, 1], target[:, 2], marker="+", color="#7F8C8D", s=34, alpha=0.60)
        ax.scatter(
            observed[:, 0],
            observed[:, 1],
            observed[:, 2],
            marker="x",
            color="#C0392B",
            s=28,
            alpha=0.72,
            linewidths=1.0,
        )
        for agent in range(N_AGENTS):
            ax.plot(
                [position[agent, 0], observed[agent, 0]],
                [position[agent, 1], observed[agent, 1]],
                [position[agent, 2], observed[agent, 2]],
                color="#C0392B",
                linewidth=0.7,
                alpha=0.34,
            )

        for first, second in edges:
            distance = float(np.linalg.norm(position[first] - position[second]))
            edge_color = "#C0392B" if distance < UNSAFE_SEPARATION else colors[arm]
            ax.plot(
                [position[first, 0], position[second, 0]],
                [position[first, 1], position[second, 1]],
                [position[first, 2], position[second, 2]],
                color=edge_color,
                alpha=0.55,
                linewidth=1.2,
            )

        agent_color = "#C0392B" if data["unsafe_separation"][step] else colors[arm]
        ax.scatter(
            position[:, 0],
            position[:, 1],
            position[:, 2],
            color=agent_color,
            edgecolor="#FFFFFF",
            linewidth=0.8,
            s=52,
            depthshade=False,
            zorder=5,
        )
        for agent in range(N_AGENTS):
            ax.text(
                position[agent, 0],
                position[agent, 1],
                position[agent, 2] + 0.13,
                str(agent + 1),
                fontsize=6.5,
                color="#17202A",
                ha="center",
            )

        if arm == "lec_only":
            status = "UNSAFE SEPARATION" if data["unsafe_separation"][step] else "neural command accepted"
            status_color = "#C0392B" if data["unsafe_separation"][step] else "#D97706"
        elif data["fallback"][step]:
            status = "FALLBACK: SLOW + SEPARATE"
            status_color = "#7D3C98"
        elif data["filter_active"][step]:
            status = "CBF-STYLE FILTER ACTIVE"
            status_color = "#117864"
        else:
            status = "neural command accepted"
            status_color = "#277DA1"

        ax.text2D(
            0.02,
            0.96,
            status,
            transform=ax.transAxes,
            fontsize=9.5,
            weight="bold",
            color=status_color,
            va="top",
        )
        ax.text2D(
            0.02,
            0.90,
            f"true min separation  {data['minimum_separation'][step]:.2f} m",
            transform=ax.transAxes,
            fontsize=8.5,
            color="#17202A",
            va="top",
        )
        ax.text2D(
            0.02,
            0.85,
            f"formation error       {data['formation_error'][step]:.2f} m",
            transform=ax.transAxes,
            fontsize=8.5,
            color="#17202A",
            va="top",
        )
        uncertainty_text = (
            "not tracked"
            if arm == "lec_only"
            else f"{data['uncertainty_radius'][step]:.2f} m"
        )
        ax.text2D(
            0.02,
            0.80,
            f"max uncertainty      {uncertainty_text}",
            transform=ax.transAxes,
            fontsize=8.5,
            color="#17202A",
            va="top",
        )

    def caption(step: int) -> str:
        if step < 50:
            return "One frozen neural policy was trained only on nominal sensor error. Spheres are true states; red x marks are observations."
        if step < 105:
            return "Agent-specific GPS and relative-sensor drift grows. The LEC still receives an ordinary-looking numeric state vector."
        if step < 165:
            return "Left: shifted observations pass directly to control. Right: uncertainty inflates the safety margin before commands reach the plant."
        if step < 215:
            return "The safety architecture trades tracking performance for separation: filter and fallback events are visible and counted."
        return "A safety filter is only as credible as its uncertainty model and independent sensing assumptions."

    def draw(step: int) -> None:
        panel(axes[0], "lec_only", step)
        panel(axes[1], "rta", step)
        envelope = arrays["rta"]["sensor_fault_envelope"][step]
        if envelope > 0.08:
            fault_artist.set_text(f"SENSOR DRIFT + NOISE  {100 * envelope:3.0f}%")
            fault_artist.set_color("#A04000")
            fault_artist.set_bbox(
                {"boxstyle": "round,pad=0.35", "facecolor": "#FEF5E7", "edgecolor": "#F5CBA7"}
            )
        else:
            fault_artist.set_text("NOMINAL SENSOR REGIME")
            fault_artist.set_color("#1E8449")
            fault_artist.set_bbox(
                {"boxstyle": "round,pad=0.35", "facecolor": "#E9F7EF", "edgecolor": "#A9DFBF"}
            )
        caption_artist.set_text(f"step {step:03d}/{STEPS - 1}  |  {caption(step)}")

    critical_step = int(np.argmin(arrays["lec_only"]["minimum_separation"]))
    draw(critical_step)
    fig.savefig(POSTER, dpi=160, facecolor=fig.get_facecolor())

    writer = FFMpegWriter(
        fps=FPS,
        codec="libx264",
        bitrate=1900,
        metadata={"title": "Sensor uncertainty and runtime assurance in 3D formation control"},
        extra_args=["-pix_fmt", "yuv420p", "-movflags", "+faststart"],
    )
    with writer.saving(fig, str(MOVIE), dpi=100):
        for step in range(STEPS):
            draw(step)
            writer.grab_frame()

        # End card: keep the final evidence on screen for two seconds.
        for ax in axes:
            ax.clear()
            ax.set_axis_off()
        raw = summaries["lec_only"]
        rta = summaries["rta"]
        axes[0].text2D(0.08, 0.83, "Same neural policy. Same sensor-error tape.", transform=axes[0].transAxes, fontsize=14, weight="bold")
        axes[0].text2D(
            0.08,
            0.63,
            f"LEC only\nunsafe-separation steps: {raw['unsafe_separation_steps']}\nminimum separation: {raw['minimum_separation']:.2f} m\nformation-breakdown steps: {raw['formation_breakdown_steps']}",
            transform=axes[0].transAxes,
            fontsize=12,
            color="#A04000",
            linespacing=1.5,
        )
        axes[1].text2D(
            0.08,
            0.63,
            f"Uncertainty-aware RTA\nunsafe-separation steps: {rta['unsafe_separation_steps']}\nminimum separation: {rta['minimum_separation']:.2f} m\nfallback steps: {rta['fallback_steps']}",
            transform=axes[1].transAxes,
            fontsize=12,
            color="#1F618D",
            linespacing=1.5,
        )
        caption_artist.set_text("Illustrative result: monitoring, filtering, and fallback must be evaluated together under the same sensor uncertainty.")
        fault_artist.set_text("MODEL-RELATIVE RESULT")
        fault_artist.set_color("#566573")
        fault_artist.set_bbox(
            {"boxstyle": "round,pad=0.35", "facecolor": "#F2F3F4", "edgecolor": "#CCD1D1"}
        )
        for _ in range(2 * FPS):
            writer.grab_frame()

    plt.close(fig)
    return MOVIE, POSTER


def write_report(
    training: dict[str, float | int | str],
    summaries: dict[str, dict[str, object]],
) -> None:
    raw = summaries["lec_only"]
    rta = summaries["rta"]
    REPORT.write_text(
        "\n".join(
            [
                "# 3D Sensor-Uncertainty Formation Demonstration",
                "",
                "This is an illustrative kinematic experiment, not a reproduction of Liu et al. (2024) or Dawood et al. (2023), and not a deployment guarantee.",
                f"The neural controller was trained and both arms were simulated on `{training['device']}`. Rendering used the retained CSV trace.",
                "The same frozen policy, initial condition, dynamics, and sensor-error tape are used in both arms.",
                "The RTA arm adds an innovation-based uncertainty radius, a pairwise CBF-style velocity projection, and a conservative fallback using an independent bounded-error proximity channel.",
                "",
                "| arm | unsafe-separation steps | collision steps | minimum separation | formation-breakdown steps | mean formation error | filter steps | fallback steps |",
                "|---|---:|---:|---:|---:|---:|---:|---:|",
                f"| LEC only | {raw['unsafe_separation_steps']} | {raw['collision_steps']} | {raw['minimum_separation']:.3f} | {raw['formation_breakdown_steps']} | {raw['mean_formation_error']:.3f} | 0 | 0 |",
                f"| uncertainty-aware RTA | {rta['unsafe_separation_steps']} | {rta['collision_steps']} | {rta['minimum_separation']:.3f} | {rta['formation_breakdown_steps']} | {rta['mean_formation_error']:.3f} | {rta['filter_steps']} | {rta['fallback_steps']} |",
                "",
                "Interpretation boundary: this run illustrates an architectural claim, not a theorem. A CBF-style filter cannot repair unmodeled state-estimation error by itself; its result remains conditional on the uncertainty bound, local sensing model, plant model, and fallback feasibility.",
            ]
        ),
        encoding="utf-8",
    )
    CANVAS_RESPONSE.write_text(
        "\n".join(
            [
                "Deeder, I think this extends the communication-failure point in a useful direction: even when a neighbor update is fresh, its state estimate can still be wrong. Sofia, apologies again for stepping into your thread; I know it is not generally my place to redirect another student's discussion. I wanted to reply to Deeder because his industry perspective gives me a useful chance to practice explaining the assurance architecture to a practitioner, and to add a concrete paired illustration.",
                "",
                "I froze one neural formation controller and replayed the same initial condition, true dynamics, and GPS/relative-sensor error tape through two architectures. In the LEC-only arm, shifted observations pass directly to the controller. In the RTA arm, an innovation monitor tracks uncertainty, the separation margin is enlarged, a pairwise CBF-style projection modifies unsafe closing commands, and a bounded local proximity channel supports a slow-and-separate fallback.",
                "",
                "Across 240 matched steps, the LEC-only arm entered the unsafe-separation region, defined here as pairwise distance below 0.78 m, on 87 steps and reached a minimum separation of 0.544 m. The RTA arm had zero unsafe-separation steps and a 1.293 m minimum, while filtering commands on 101 steps and using fallback on 14. Neither arm collided or lost the formation, so the claim is specifically about preserving the separation envelope, not converting a crash into mission success.",
                "",
                "The important qualification is that a safety filter does not make uncertain state estimates disappear. Its assurance remains conditional on the estimator, uncertainty bound, independent proximity measurements, plant model, and fallback feasibility. The attached movie is an illustrative GPU-backed kinematic experiment, not a reproduction of Liu et al. (2024) or a deployment guarantee; every displayed metric comes from the retained paired trace.",
            ]
        ),
        encoding="utf-8",
    )


def write_manifest(paths: list[Path]) -> None:
    entries = [(sha256(path), path.name) for path in paths]
    entries.append((sha256(Path(__file__)), f"analysis/{Path(__file__).name}"))
    MANIFEST.write_text("".join(f"{digest}  {name}\n" for digest, name in entries), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-render", action="store_true", help="Run training and simulation only.")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This registered demonstration requires CUDA, but PyTorch cannot see a CUDA device.")
    device = torch.device("cuda")
    OUT.mkdir(parents=True, exist_ok=True)

    policy, training = train_policy(device)
    tape = make_tape(device)
    lec_rows, lec_summary = simulate("lec_only", policy, tape, device)
    rta_rows, rta_summary = simulate("rta", policy, tape, device)
    summaries = {"lec_only": lec_summary, "rta": rta_summary}

    write_csv(TRACE_CSV, lec_rows + rta_rows)
    write_csv(SUMMARY_CSV, [lec_summary, rta_summary])
    write_report(training, summaries)
    if not args.skip_render:
        render({"lec_only": lec_rows, "rta": rta_rows}, summaries)
        write_manifest(
            [
                MOVIE,
                POSTER,
                TRACE_CSV,
                SUMMARY_CSV,
                CHECKPOINT,
                TRAINING_JSON,
                REPORT,
                CANVAS_RESPONSE,
            ]
        )

    print(json.dumps({"training": training, "summaries": summaries}, indent=2))
    print(f"movie={MOVIE}")
    print(f"poster={POSTER}")
    print(f"manifest={MANIFEST}")


if __name__ == "__main__":
    main()
