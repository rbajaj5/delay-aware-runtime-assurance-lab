"""CUDA-batched affine-formation conditioning experiment and replay renderer.

The experiment isolates a failure that ordinary tracking-error and pairwise
separation checks can miss. A legal affine command approaches a singular map;
the baseline tracks it, while the RTA arm rejects commands whose leader-span
conditioning falls below a registered threshold and holds the last admissible
affine map.

This is an illustrative first-order multi-agent model, not flight data, a
reproduction of Liu et al., or a formal safety proof.
"""

from __future__ import annotations

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
from matplotlib.collections import LineCollection
from matplotlib.patches import Rectangle


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "20260811" / "affine_conditioning_gpu"
TRACE_CSV = OUT / "affine_conditioning_replay_trace.csv"
EPISODE_CSV = OUT / "affine_conditioning_episode_summary.csv"
BATCH_CSV = OUT / "affine_conditioning_batch_summary.csv"
CONFIG_JSON = OUT / "affine_conditioning_config.json"
MOVIE = OUT / "affine_conditioning_comparison.mp4"
POSTER = OUT / "affine_conditioning_comparison_poster.png"
REPORT = OUT / "affine_conditioning_report.md"
MANIFEST = OUT / "AFFINE_CONDITIONING_GPU_MANIFEST.sha256"

SEED = 20260811
BATCH_SIZE = 4096
N_AGENTS = 12
N_LEADERS = 3
STEPS = 240
DT = 0.08
FPS = 12
CONDITION_THRESHOLD = 0.12
CONDITION_UNCERTAINTY_MARGIN = 0.015
RANK_LOSS_THRESHOLD = 0.04
SEPARATION_THRESHOLD = 0.22
COLLISION_THRESHOLD = 0.12
MAX_SPEED = 1.45


@dataclass
class Parameters:
    min_scale: torch.Tensor
    shear: torch.Tensor
    rotation: torch.Tensor
    gain: torch.Tensor
    noise_amplitude: torch.Tensor
    phase: torch.Tensor
    initial_noise: torch.Tensor

    def select(self, index: int) -> "Parameters":
        return Parameters(
            min_scale=self.min_scale[index : index + 1],
            shear=self.shear[index : index + 1],
            rotation=self.rotation[index : index + 1],
            gain=self.gain[index : index + 1],
            noise_amplitude=self.noise_amplitude[index : index + 1],
            phase=self.phase[index : index + 1],
            initial_noise=self.initial_noise[index : index + 1],
        )


def nominal_configuration(device: torch.device) -> torch.Tensor:
    # Distinct projected coordinates keep pairwise spacing nonzero even when
    # the commanded affine image becomes one-dimensional.
    points = [
        (-3.3, -2.2),
        (0.3, 3.0),
        (3.3, -1.8),
        (-2.7, 1.5),
        (-2.1, -0.4),
        (-1.5, 2.4),
        (-0.9, -1.5),
        (-0.3, 0.9),
        (0.9, -2.5),
        (1.5, 1.8),
        (2.1, -0.2),
        (2.7, 2.6),
    ]
    return torch.tensor(points, dtype=torch.float32, device=device)


def affine_weights_and_stress(q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    leaders = q[:N_LEADERS]
    augmented = torch.cat(
        [leaders.T, torch.ones((1, N_LEADERS), dtype=q.dtype, device=q.device)], dim=0
    )
    weights = []
    for follower in q[N_LEADERS:]:
        rhs = torch.cat([follower, torch.ones((1,), dtype=q.dtype, device=q.device)])
        weights.append(torch.linalg.solve(augmented, rhs))
    weight_matrix = torch.stack(weights)
    constraint = torch.zeros(
        (N_AGENTS - N_LEADERS, N_AGENTS), dtype=q.dtype, device=q.device
    )
    constraint[:, :N_LEADERS] = -weight_matrix
    constraint[:, N_LEADERS:] = torch.eye(
        N_AGENTS - N_LEADERS, dtype=q.dtype, device=q.device
    )
    stress = constraint.T @ constraint
    return weight_matrix, stress


def make_parameters(device: torch.device) -> Parameters:
    generator = torch.Generator(device=device).manual_seed(SEED)
    min_scale = -0.045 + 0.205 * torch.rand(BATCH_SIZE, generator=generator, device=device)
    shear = -0.075 + 0.150 * torch.rand(BATCH_SIZE, generator=generator, device=device)
    rotation = 0.12 + 0.24 * torch.rand(BATCH_SIZE, generator=generator, device=device)
    gain = 1.80 + 0.45 * torch.rand(BATCH_SIZE, generator=generator, device=device)
    noise_amplitude = 0.0025 + 0.0065 * torch.rand(
        BATCH_SIZE, generator=generator, device=device
    )
    phase = 2.0 * math.pi * torch.rand(
        (BATCH_SIZE, N_AGENTS, 2), generator=generator, device=device
    )
    initial_noise = 0.025 * torch.randn(
        (BATCH_SIZE, N_AGENTS, 2), generator=generator, device=device
    )
    return Parameters(
        min_scale=min_scale,
        shear=shear,
        rotation=rotation,
        gain=gain,
        noise_amplitude=noise_amplitude,
        phase=phase,
        initial_noise=initial_noise,
    )


def deformation_progress(step: int) -> float:
    if step < 38:
        return 0.0
    if step < 148:
        tau = (step - 38) / 110.0
        return 0.5 * (1.0 - math.cos(math.pi * tau))
    if step < 176:
        return 1.0
    if step < 234:
        tau = (step - 176) / 58.0
        return 0.5 * (1.0 + math.cos(math.pi * tau))
    return 0.0


def candidate_affine_map(params: Parameters, progress: float) -> torch.Tensor:
    scale = 1.0 + progress * (params.min_scale - 1.0)
    shear = progress * params.shear
    angle = progress * params.rotation
    cosine = torch.cos(angle)
    sine = torch.sin(angle)
    rotation = torch.stack(
        [
            torch.stack([cosine, -sine], dim=-1),
            torch.stack([sine, cosine], dim=-1),
        ],
        dim=-2,
    )
    deformation = torch.zeros_like(rotation)
    deformation[:, 0, 0] = 1.0
    deformation[:, 0, 1] = shear
    deformation[:, 1, 1] = scale
    return rotation @ deformation


def condition_ratio_from_points(points: torch.Tensor) -> torch.Tensor:
    leader_matrix = torch.stack(
        [points[..., 1, :] - points[..., 0, :], points[..., 2, :] - points[..., 0, :]],
        dim=-1,
    )
    singular_values = torch.linalg.svdvals(leader_matrix)
    return singular_values[..., -1] / singular_values[..., 0].clamp_min(1.0e-8)


def clamp_speed(command: torch.Tensor) -> torch.Tensor:
    norm = torch.linalg.vector_norm(command, dim=-1, keepdim=True).clamp_min(1.0e-8)
    return command * torch.clamp(MAX_SPEED / norm, max=1.0)


def pairwise_minimum(positions: torch.Tensor) -> torch.Tensor:
    distance = torch.cdist(positions, positions)
    identity = torch.eye(N_AGENTS, dtype=torch.bool, device=positions.device)
    return distance.masked_fill(identity, float("inf")).amin(dim=(-2, -1))


def disturbance(params: Parameters, step: int, device: torch.device) -> torch.Tensor:
    agent = torch.arange(N_AGENTS, dtype=torch.float32, device=device)[None, :, None]
    axis = torch.arange(2, dtype=torch.float32, device=device)[None, None, :]
    frequency = 0.045 + 0.007 * agent + 0.011 * axis
    return params.noise_amplitude[:, None, None] * torch.sin(
        step * frequency + params.phase
    )


def simulate(
    params: Parameters,
    q: torch.Tensor,
    weights: torch.Tensor,
    record: bool = False,
) -> tuple[dict[str, torch.Tensor], list[dict[str, object]]]:
    device = q.device
    batch = params.min_scale.shape[0]
    positions = q[None, :, :] + params.initial_noise
    positions = torch.stack([positions.clone(), positions.clone()], dim=0)
    velocity = torch.zeros_like(positions)
    last_safe_map = torch.eye(2, dtype=q.dtype, device=device)[None].repeat(batch, 1, 1)

    min_ratio = torch.full((2, batch), float("inf"), device=device)
    min_separation = torch.full((2, batch), float("inf"), device=device)
    tracking_sum = torch.zeros((2, batch), device=device)
    tracking_max = torch.zeros((2, batch), device=device)
    rank_loss_steps = torch.zeros((2, batch), dtype=torch.int32, device=device)
    condition_violation_steps = torch.zeros((2, batch), dtype=torch.int32, device=device)
    separation_violation_steps = torch.zeros((2, batch), dtype=torch.int32, device=device)
    collision_steps = torch.zeros((2, batch), dtype=torch.int32, device=device)
    intervention_steps = torch.zeros((2, batch), dtype=torch.int32, device=device)
    rank_loss_tracking_max = torch.zeros((2, batch), device=device)
    rows: list[dict[str, object]] = []

    for step in range(STEPS):
        progress = deformation_progress(step)
        candidate_map = candidate_affine_map(params, progress)
        translation = torch.tensor(
            [0.55 * math.sin(0.025 * step), 0.30 * math.cos(0.018 * step)],
            dtype=q.dtype,
            device=device,
        )
        candidate_reference = torch.einsum("bij,nj->bni", candidate_map, q) + translation
        candidate_ratio = condition_ratio_from_points(candidate_reference[:, :N_LEADERS])
        robust_candidate_ratio = candidate_ratio - CONDITION_UNCERTAINTY_MARGIN
        admissible = robust_candidate_ratio >= CONDITION_THRESHOLD
        last_safe_map = torch.where(admissible[:, None, None], candidate_map, last_safe_map)
        safe_reference = torch.einsum("bij,nj->bni", last_safe_map, q) + translation
        reference = torch.stack([candidate_reference, safe_reference], dim=0)
        intervention = ~admissible

        command = torch.zeros_like(positions)
        gain = params.gain[None, :, None, None]
        command[:, :, :N_LEADERS] = gain * (
            reference[:, :, :N_LEADERS] - positions[:, :, :N_LEADERS]
        )
        follower_target = torch.einsum(
            "fl,abld->abfd", weights, positions[:, :, :N_LEADERS]
        )
        command[:, :, N_LEADERS:] = 2.15 * (
            follower_target - positions[:, :, N_LEADERS:]
        )
        command = clamp_speed(command)

        common_disturbance = disturbance(params, step, device)
        velocity = 0.76 * velocity + 0.24 * command + common_disturbance[None]
        positions = positions + DT * velocity

        ratio = condition_ratio_from_points(positions[:, :, :N_LEADERS])
        separation = pairwise_minimum(positions)
        tracking = torch.linalg.vector_norm(positions - reference, dim=-1).mean(dim=-1)
        rank_loss = ratio < RANK_LOSS_THRESHOLD
        condition_violation = ratio < CONDITION_THRESHOLD
        separation_violation = separation < SEPARATION_THRESHOLD
        collision = separation < COLLISION_THRESHOLD

        min_ratio = torch.minimum(min_ratio, ratio)
        min_separation = torch.minimum(min_separation, separation)
        tracking_sum += tracking
        tracking_max = torch.maximum(tracking_max, tracking)
        rank_loss_steps += rank_loss.to(torch.int32)
        condition_violation_steps += condition_violation.to(torch.int32)
        separation_violation_steps += separation_violation.to(torch.int32)
        collision_steps += collision.to(torch.int32)
        intervention_steps[1] += intervention.to(torch.int32)
        rank_loss_tracking_max = torch.maximum(
            rank_loss_tracking_max,
            torch.where(rank_loss, tracking, torch.zeros_like(tracking)),
        )

        if record:
            for arm_index, arm in enumerate(("tracking_only", "conditioning_rta")):
                arm_positions = positions[arm_index, 0].detach().cpu()
                arm_reference = reference[arm_index, 0].detach().cpu()
                for agent in range(N_AGENTS):
                    rows.append(
                        {
                            "arm": arm,
                            "step": step,
                            "agent": agent,
                            "x": float(arm_positions[agent, 0]),
                            "y": float(arm_positions[agent, 1]),
                            "reference_x": float(arm_reference[agent, 0]),
                            "reference_y": float(arm_reference[agent, 1]),
                            "leader": int(agent < N_LEADERS),
                            "condition_ratio": float(ratio[arm_index, 0]),
                            "candidate_condition_ratio": float(candidate_ratio[0]),
                            "minimum_separation": float(separation[arm_index, 0]),
                            "tracking_error": float(tracking[arm_index, 0]),
                            "rank_loss": int(rank_loss[arm_index, 0]),
                            "condition_violation": int(condition_violation[arm_index, 0]),
                            "separation_violation": int(separation_violation[arm_index, 0]),
                            "collision": int(collision[arm_index, 0]),
                            "intervention": int(intervention[0]) if arm_index == 1 else 0,
                            "deformation_progress": progress,
                            "commanded_min_scale": float(params.min_scale[0]),
                        }
                    )

    metrics = {
        "min_ratio": min_ratio,
        "min_separation": min_separation,
        "mean_tracking": tracking_sum / STEPS,
        "max_tracking": tracking_max,
        "rank_loss_steps": rank_loss_steps,
        "condition_violation_steps": condition_violation_steps,
        "separation_violation_steps": separation_violation_steps,
        "collision_steps": collision_steps,
        "intervention_steps": intervention_steps,
        "rank_loss_tracking_max": rank_loss_tracking_max,
    }
    return metrics, rows


def choose_representative(metrics: dict[str, torch.Tensor], params: Parameters) -> int:
    baseline_rank = metrics["min_ratio"][0] < 0.02
    baseline_separated = metrics["min_separation"][0] >= SEPARATION_THRESHOLD
    baseline_tracks = metrics["rank_loss_tracking_max"][0] < 0.18
    rta_conditioned = metrics["min_ratio"][1] >= 0.10
    candidates = torch.nonzero(
        baseline_rank & baseline_separated & baseline_tracks & rta_conditioned
    ).flatten()
    if candidates.numel() == 0:
        candidates = torch.nonzero(baseline_rank & rta_conditioned).flatten()
    if candidates.numel() == 0:
        raise RuntimeError("No representative hidden-collapse episode was found.")
    target = torch.abs(params.min_scale[candidates] + 0.005)
    return int(candidates[torch.argmin(target)].item())


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"No rows for {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def episode_rows(metrics: dict[str, torch.Tensor], params: Parameters) -> list[dict[str, object]]:
    cpu = {name: tensor.detach().cpu() for name, tensor in metrics.items()}
    min_scale = params.min_scale.detach().cpu()
    shear = params.shear.detach().cpu()
    rotation = params.rotation.detach().cpu()
    rows: list[dict[str, object]] = []
    for arm_index, arm in enumerate(("tracking_only", "conditioning_rta")):
        for episode in range(BATCH_SIZE):
            hidden = (
                cpu["min_ratio"][arm_index, episode] < RANK_LOSS_THRESHOLD
                and cpu["min_separation"][arm_index, episode] >= SEPARATION_THRESHOLD
                and cpu["rank_loss_tracking_max"][arm_index, episode] < 0.18
            )
            rows.append(
                {
                    "arm": arm,
                    "episode": episode,
                    "min_commanded_scale": float(min_scale[episode]),
                    "shear": float(shear[episode]),
                    "rotation": float(rotation[episode]),
                    "minimum_condition_ratio": float(cpu["min_ratio"][arm_index, episode]),
                    "minimum_separation": float(cpu["min_separation"][arm_index, episode]),
                    "mean_tracking_error": float(cpu["mean_tracking"][arm_index, episode]),
                    "maximum_tracking_error": float(cpu["max_tracking"][arm_index, episode]),
                    "rank_loss_tracking_error": float(cpu["rank_loss_tracking_max"][arm_index, episode]),
                    "rank_loss_steps": int(cpu["rank_loss_steps"][arm_index, episode]),
                    "condition_violation_steps": int(cpu["condition_violation_steps"][arm_index, episode]),
                    "separation_violation_steps": int(cpu["separation_violation_steps"][arm_index, episode]),
                    "collision_steps": int(cpu["collision_steps"][arm_index, episode]),
                    "intervention_steps": int(cpu["intervention_steps"][arm_index, episode]),
                    "hidden_collapse": int(hidden),
                }
            )
    return rows


def aggregate_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for arm in ("tracking_only", "conditioning_rta"):
        arm_rows = [row for row in rows if row["arm"] == arm]
        output.append(
            {
                "arm": arm,
                "episodes": len(arm_rows),
                "rank_loss_episodes": sum(int(row["rank_loss_steps"]) > 0 for row in arm_rows),
                "condition_violation_episodes": sum(
                    int(row["condition_violation_steps"]) > 0 for row in arm_rows
                ),
                "separation_violation_episodes": sum(
                    int(row["separation_violation_steps"]) > 0 for row in arm_rows
                ),
                "collision_episodes": sum(int(row["collision_steps"]) > 0 for row in arm_rows),
                "hidden_collapse_episodes": sum(int(row["hidden_collapse"]) for row in arm_rows),
                "mean_tracking_error": float(
                    np.mean([float(row["mean_tracking_error"]) for row in arm_rows])
                ),
                "median_minimum_condition_ratio": float(
                    np.median([float(row["minimum_condition_ratio"]) for row in arm_rows])
                ),
                "median_minimum_separation": float(
                    np.median([float(row["minimum_separation"]) for row in arm_rows])
                ),
                "mean_intervention_steps": float(
                    np.mean([int(row["intervention_steps"]) for row in arm_rows])
                ),
            }
        )
    return output


def nearest_edges(q: np.ndarray, count: int = 23) -> list[tuple[int, int]]:
    candidates: list[tuple[float, int, int]] = []
    for first in range(N_AGENTS):
        for second in range(first + 1, N_AGENTS):
            candidates.append((float(np.linalg.norm(q[first] - q[second])), first, second))
    edges = [(first, second) for _, first, second in sorted(candidates)[:count]]
    for edge in ((0, 1), (1, 2), (0, 2)):
        if edge not in edges:
            edges.append(edge)
    return edges


def render_replay(
    rows: list[dict[str, object]],
    q: np.ndarray,
    batch_summary: list[dict[str, object]],
) -> None:
    by_arm: dict[str, list[list[dict[str, object]]]] = {}
    for arm in ("tracking_only", "conditioning_rta"):
        arm_rows = [row for row in rows if row["arm"] == arm]
        by_arm[arm] = [
            [row for row in arm_rows if int(row["step"]) == step] for step in range(STEPS)
        ]
    edges = nearest_edges(q)
    plt.rcParams.update({"font.family": "DejaVu Sans"})
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 7.2), dpi=100, facecolor="#f6f8fb")
    fig.subplots_adjust(left=0.055, right=0.965, top=0.80, bottom=0.24, wspace=0.16)

    collections: list[LineCollection] = []
    followers = []
    leaders = []
    references = []
    leader_triangles = []
    statuses = []
    metrics_text = []
    for ax, title, color in zip(
        axes,
        ("Tracking + separation checks", "Conditioning-aware RTA"),
        ("#d97706", "#147d92"),
        strict=True,
    ):
        ax.set_title(title, fontsize=15, color="#182230", weight="bold", pad=10)
        ax.set_xlim(-4.8, 4.8)
        ax.set_ylim(-4.0, 4.0)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, color="#d0d5dd", linewidth=0.7, alpha=0.8)
        ax.set_xlabel("centroid-relative x", fontsize=9)
        ax.set_ylabel("centroid-relative y", fontsize=9)
        ax.tick_params(labelsize=8, colors="#667085")
        collection = LineCollection([], colors=color, linewidths=1.2, alpha=0.50, zorder=2)
        ax.add_collection(collection)
        collections.append(collection)
        followers.append(ax.scatter([], [], s=42, color=color, edgecolor="white", linewidth=0.8, zorder=4))
        leaders.append(ax.scatter([], [], s=90, marker="D", color="#7c3aed", edgecolor="white", linewidth=1.0, zorder=6))
        references.append(ax.scatter([], [], s=30, marker="x", color="#98a2b3", linewidth=1.2, zorder=3))
        triangle, = ax.plot([], [], color="#7c3aed", linewidth=2.4, zorder=5)
        leader_triangles.append(triangle)
        statuses.append(ax.text(0.02, 0.95, "", transform=ax.transAxes, fontsize=11, weight="bold", va="top"))
        metrics_text.append(ax.text(0.02, 0.02, "", transform=ax.transAxes, fontsize=9, color="#344054"))

    fig.text(0.055, 0.94, "Low tracking error can hide affine formation collapse",
             fontsize=23, color="#182230", weight="bold")
    fig.text(0.055, 0.885,
             "Three leaders define the affine map; followers retain fixed design-time affine weights.",
             fontsize=12, color="#526273")
    step_text = fig.text(0.95, 0.94, "", ha="right", fontsize=11, color="#475467", weight="bold")

    gauge_axes = [
        fig.add_axes([0.085, 0.145, 0.36, 0.035]),
        fig.add_axes([0.555, 0.145, 0.36, 0.035]),
    ]
    gauge_markers = []
    for gauge in gauge_axes:
        gauge.set_xlim(0.0, 0.45)
        gauge.set_ylim(0.0, 1.0)
        gauge.axvspan(0.0, CONDITION_THRESHOLD, color="#fee2e2")
        gauge.axvline(CONDITION_THRESHOLD, color="#c43d2f", linestyle="--", linewidth=1.2)
        marker = gauge.axvline(0.4, color="#182230", linewidth=3.0)
        gauge_markers.append(marker)
        gauge.set_yticks([])
        gauge.set_xticks([0.0, CONDITION_THRESHOLD, 0.25, 0.45])
        gauge.tick_params(axis="x", labelsize=8, colors="#667085")
        gauge.set_xlabel("normalized leader conditioning  σmin / σmax", fontsize=8, color="#475467", labelpad=1)
        for spine in gauge.spines.values():
            spine.set_visible(False)

    caption = fig.text(0.055, 0.067, "", fontsize=10.5, color="#344054")
    footer = fig.text(0.055, 0.022,
                      "Illustrative CUDA-batched first-order model; not flight data, a reproduction of Liu et al., or a formal safety proof.",
                      fontsize=8.3, color="#667085")

    summary_lookup = {str(row["arm"]): row for row in batch_summary}
    end_heading = fig.text(0.5, 0.70, "Conditioning is a separate assurance endpoint.", ha="center",
                           fontsize=23, color="#182230", weight="bold", visible=False)
    end_left = fig.text(0.25, 0.43, "", ha="center", fontsize=14, color="#b42318", visible=False)
    end_right = fig.text(0.75, 0.43, "", ha="center", fontsize=14, color="#0f766e", visible=False)
    end_note = fig.text(0.5, 0.20,
                        "A controller can accurately track a reference that has already lost affine rank.",
                        ha="center", fontsize=11, color="#475467", visible=False)

    poster_step = 0
    poster_ratio = float("inf")
    writer = FFMpegWriter(fps=FPS, codec="libx264", bitrate=3000,
                          metadata={"title": "Affine formation conditioning monitor"},
                          extra_args=["-pix_fmt", "yuv420p", "-movflags", "+faststart"])
    with writer.saving(fig, str(MOVIE), dpi=100):
        for step in range(STEPS):
            for arm_index, arm in enumerate(("tracking_only", "conditioning_rta")):
                frame = by_arm[arm][step]
                position = np.array([[float(row["x"]), float(row["y"])] for row in frame])
                reference = np.array([[float(row["reference_x"]), float(row["reference_y"])] for row in frame])
                centered = position - position.mean(axis=0, keepdims=True)
                centered_reference = reference - reference.mean(axis=0, keepdims=True)
                segments = np.array([[centered[first], centered[second]] for first, second in edges])
                collections[arm_index].set_segments(segments)
                followers[arm_index].set_offsets(centered[N_LEADERS:])
                leaders[arm_index].set_offsets(centered[:N_LEADERS])
                references[arm_index].set_offsets(centered_reference)
                triangle_points = np.vstack([centered[:N_LEADERS], centered[0]])
                leader_triangles[arm_index].set_data(triangle_points[:, 0], triangle_points[:, 1])

                ratio = float(frame[0]["condition_ratio"])
                separation = float(frame[0]["minimum_separation"])
                tracking = float(frame[0]["tracking_error"])
                intervention = bool(int(frame[0]["intervention"]))
                gauge_markers[arm_index].set_xdata([ratio, ratio])
                metrics_text[arm_index].set_text(
                    f"tracking error {tracking:.3f}   |   minimum separation {separation:.3f}"
                )
                if arm == "tracking_only":
                    if ratio < RANK_LOSS_THRESHOLD:
                        statuses[arm_index].set_text("AFFINE RANK LOST")
                        statuses[arm_index].set_color("#b42318")
                    elif ratio < CONDITION_THRESHOLD:
                        statuses[arm_index].set_text("NEAR SINGULAR; REFERENCE ACCEPTED")
                        statuses[arm_index].set_color("#d97706")
                    else:
                        statuses[arm_index].set_text("REFERENCE TRACKED")
                        statuses[arm_index].set_color("#9a6700")
                else:
                    statuses[arm_index].set_text(
                        "HOLD LAST WELL-CONDITIONED MAP" if intervention else "REFERENCE TRACKED"
                    )
                    statuses[arm_index].set_color("#7c3aed" if intervention else "#147d92")

            step_text.set_text(f"step {step:03d}/{STEPS - 1}")
            if step < 55:
                caption.set_text("The ordinary monitor sees accurate tracking and safe pairwise spacing.")
            elif step < 135:
                caption.set_text("The commanded affine map compresses; its leader-span conditioning falls toward zero.")
            elif step < 184:
                caption.set_text("Left: low tracking error follows a degenerate command. Right: the conditioning endpoint rejects it.")
            else:
                caption.set_text("The RTA resumes the commanded maneuver only after the proposed map becomes admissible again.")

            baseline_ratio = float(by_arm["tracking_only"][step][0]["condition_ratio"])
            if baseline_ratio < poster_ratio:
                poster_ratio = baseline_ratio
                poster_step = step
            writer.grab_frame()

        for ax in [*axes, *gauge_axes]:
            ax.set_visible(False)
        for artist in [step_text, caption]:
            artist.set_visible(False)
        end_heading.set_visible(True)
        left = summary_lookup["tracking_only"]
        right = summary_lookup["conditioning_rta"]
        end_left.set_text(
            "Tracking + separation checks\n"
            f"hidden-collapse episodes: {int(left['hidden_collapse_episodes'])}/{BATCH_SIZE}\n"
            f"rank-loss episodes: {int(left['rank_loss_episodes'])}/{BATCH_SIZE}"
        )
        end_right.set_text(
            "Conditioning-aware RTA\n"
            f"hidden-collapse episodes: {int(right['hidden_collapse_episodes'])}/{BATCH_SIZE}\n"
            f"rank-loss episodes: {int(right['rank_loss_episodes'])}/{BATCH_SIZE}"
        )
        for artist in [end_left, end_right, end_note]:
            artist.set_visible(True)
        footer.set_text(
            "Descriptive paired CUDA batch. Thresholds and model are illustrative; no deployment-level guarantee is claimed."
        )
        for _ in range(2 * FPS):
            writer.grab_frame()

    # Render the most nearly singular retained frame as a poster.
    for ax in [*axes, *gauge_axes]:
        ax.set_visible(True)
    for artist in [end_heading, end_left, end_right, end_note]:
        artist.set_visible(False)
    step_text.set_visible(True)
    caption.set_visible(True)
    footer.set_text(
        "Illustrative CUDA-batched first-order model; not flight data, a reproduction of Liu et al., or a formal safety proof."
    )
    step = poster_step
    for arm_index, arm in enumerate(("tracking_only", "conditioning_rta")):
        frame = by_arm[arm][step]
        position = np.array([[float(row["x"]), float(row["y"])] for row in frame])
        reference = np.array([[float(row["reference_x"]), float(row["reference_y"])] for row in frame])
        centered = position - position.mean(axis=0, keepdims=True)
        centered_reference = reference - reference.mean(axis=0, keepdims=True)
        collections[arm_index].set_segments(
            np.array([[centered[first], centered[second]] for first, second in edges])
        )
        followers[arm_index].set_offsets(centered[N_LEADERS:])
        leaders[arm_index].set_offsets(centered[:N_LEADERS])
        references[arm_index].set_offsets(centered_reference)
        triangle_points = np.vstack([centered[:N_LEADERS], centered[0]])
        leader_triangles[arm_index].set_data(triangle_points[:, 0], triangle_points[:, 1])
        ratio = float(frame[0]["condition_ratio"])
        gauge_markers[arm_index].set_xdata([ratio, ratio])
        metrics_text[arm_index].set_text(
            f"tracking error {float(frame[0]['tracking_error']):.3f}   |   minimum separation {float(frame[0]['minimum_separation']):.3f}"
        )
    statuses[0].set_text("AFFINE RANK LOST")
    statuses[0].set_color("#b42318")
    statuses[1].set_text("HOLD LAST WELL-CONDITIONED MAP")
    statuses[1].set_color("#7c3aed")
    step_text.set_text(f"step {step:03d}/{STEPS - 1}")
    caption.set_text("Low tracking error and nonzero pairwise spacing do not certify affine rank.")
    fig.savefig(POSTER, dpi=180, facecolor=fig.get_facecolor())
    plt.close(fig)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this experiment, but PyTorch cannot see a GPU.")
    device = torch.device("cuda")
    OUT.mkdir(parents=True, exist_ok=True)
    q = nominal_configuration(device)
    weights, stress = affine_weights_and_stress(q)
    stress_rank = int(torch.linalg.matrix_rank(stress, tol=1.0e-5).item())
    expected_rank = N_AGENTS - 2 - 1
    if stress_rank != expected_rank:
        raise RuntimeError(f"Stress rank {stress_rank} != expected {expected_rank}")
    if torch.linalg.vector_norm(stress @ q).item() > 1.0e-4:
        raise RuntimeError("Stress matrix does not annihilate the nominal configuration.")

    params = make_parameters(device)
    torch.cuda.synchronize()
    start = time.perf_counter()
    metrics, _ = simulate(params, q, weights)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    representative = choose_representative(metrics, params)
    _, replay_rows = simulate(params.select(representative), q, weights, record=True)

    episodes = episode_rows(metrics, params)
    batch_summary = aggregate_rows(episodes)
    write_csv(EPISODE_CSV, episodes)
    write_csv(BATCH_CSV, batch_summary)
    write_csv(TRACE_CSV, replay_rows)

    config = {
        "seed": SEED,
        "device": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "batch_size": BATCH_SIZE,
        "agents": N_AGENTS,
        "leaders": N_LEADERS,
        "steps": STEPS,
        "dt": DT,
        "condition_threshold": CONDITION_THRESHOLD,
        "condition_uncertainty_margin": CONDITION_UNCERTAINTY_MARGIN,
        "rank_loss_threshold": RANK_LOSS_THRESHOLD,
        "separation_threshold": SEPARATION_THRESHOLD,
        "collision_threshold": COLLISION_THRESHOLD,
        "stress_rank": stress_rank,
        "expected_stress_rank": expected_rank,
        "representative_episode": representative,
        "representative_parameters": {
            "min_scale": float(params.min_scale[representative].item()),
            "shear": float(params.shear[representative].item()),
            "rotation": float(params.rotation[representative].item()),
        },
        "gpu_batch_seconds": elapsed,
        "nominal_configuration": q.detach().cpu().tolist(),
        "follower_affine_weights": weights.detach().cpu().tolist(),
        "stress_matrix": stress.detach().cpu().tolist(),
    }
    CONFIG_JSON.write_text(json.dumps(config, indent=2), encoding="utf-8")

    render_replay(replay_rows, q.detach().cpu().numpy(), batch_summary)
    summary_lookup = {str(row["arm"]): row for row in batch_summary}
    baseline = summary_lookup["tracking_only"]
    rta = summary_lookup["conditioning_rta"]
    REPORT.write_text(
        "\n".join(
            [
                "# GPU Affine-Formation Conditioning Experiment",
                "",
                "This is a descriptive, paired CUDA experiment using an illustrative first-order multi-agent model. It is not flight data, a reproduction of Liu et al., or a formal safety proof.",
                "",
                f"- Device: `{config['device']}`; paired episodes per arm: {BATCH_SIZE}; steps per episode: {STEPS}.",
                f"- The design-time PSD stress matrix has rank {stress_rank} = n-d-1 and annihilates the nominal configuration.",
                f"- Tracking/separation arm: {baseline['hidden_collapse_episodes']}/{BATCH_SIZE} hidden-collapse episodes and {baseline['rank_loss_episodes']}/{BATCH_SIZE} rank-loss episodes.",
                f"- Conditioning-RTA arm: {rta['hidden_collapse_episodes']}/{BATCH_SIZE} hidden-collapse episodes and {rta['rank_loss_episodes']}/{BATCH_SIZE} rank-loss episodes.",
                f"- Representative paired episode: {representative}; retained rows: {len(replay_rows)}.",
                "",
                "A hidden collapse is registered descriptively when leader conditioning falls below 0.04, pairwise separation remains at or above 0.22, and tracking error during rank loss remains below 0.18. The RTA rejects a proposed map when its normalized leader conditioning minus the registered uncertainty margin would fall below 0.12, then holds the last admissible map.",
                "",
                "Interpretation boundary: the result demonstrates that tracking error and pairwise separation are not logically equivalent to preservation of affine rank in this model. It does not establish that this monitor or threshold is sufficient for a physical formation-control deployment.",
            ]
        ),
        encoding="utf-8",
    )
    manifest_files = [MOVIE, POSTER, TRACE_CSV, EPISODE_CSV, BATCH_CSV, CONFIG_JSON, REPORT]
    entries = [(sha256(path), path.name) for path in manifest_files]
    entries.append((sha256(Path(__file__)), f"analysis/{Path(__file__).name}"))
    MANIFEST.write_text(
        "".join(f"{digest}  {label}\n" for digest, label in entries), encoding="ascii"
    )
    print(json.dumps({"config": config, "batch_summary": batch_summary}, indent=2))
    print(f"movie={MOVIE}")
    print(f"manifest={MANIFEST}")


if __name__ == "__main__":
    main()
