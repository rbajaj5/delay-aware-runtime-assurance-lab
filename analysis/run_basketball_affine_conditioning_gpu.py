"""Deterministic CUDA basketball affine-formation experiment and replay.

This program creates a reproducible, programmatic half-court simulation.  It
compares a tracking-and-separation monitor with a conditioning-aware runtime
assurance layer while keeping the command sequence, initial state, noise, and
disturbances matched between arms.

The output is illustrative simulation evidence.  It is not real basketball
evidence, a validated player model, or proof of betting profitability.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
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
from matplotlib.patches import Arc, Circle, Rectangle


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "20260812" / "basketball_affine_conditioning_gpu"
MOVIE = OUT / "basketball_affine_conditioning_split_screen.mp4"
POSTER = OUT / "basketball_affine_conditioning_poster.png"
COORDINATES_CSV = OUT / "basketball_frame_coordinates.csv"
EPISODE_CSV = OUT / "basketball_episode_results.csv"
SUMMARY_JSON = OUT / "basketball_simulation_summary.json"
REPORT = OUT / "basketball_simulation_report.md"
MANIFEST = OUT / "BASKETBALL_AFFINE_GPU_MANIFEST.sha256"

SEED = 20260812
BATCH_SIZE = 4096
N_PLAYERS = 5
N_LEADERS = 3
STEPS = 300
FPS = 15
DT = 1.0 / FPS
POSSESSION_PREFIX = "POS-AFFINE"

COURT_WIDTH_FT = 50.0
COURT_LENGTH_FT = 47.0
COURT_CENTER = (25.0, 20.0)

CONDITION_THRESHOLD = 0.12
PROPOSAL_UNCERTAINTY_MARGIN = 0.01
FALLBACK_CHECKPOINT_THRESHOLD = 0.18
RESUME_THRESHOLD = 0.18
RANK_LOSS_THRESHOLD = 0.04
TRACKING_RMSE_THRESHOLD_FT = 1.0
MINIMUM_SPACING_THRESHOLD_FT = 2.0
COLLISION_THRESHOLD_FT = 1.0
MAX_SPEED_FT_S = 18.0

ARMS = ("tracking_separation_only", "conditioning_aware_rta")


@dataclass(frozen=True)
class Player:
    player_id: str
    display_name: str
    short_name: str
    role: str
    role_group: str
    impact_rating: int
    color: str


PLAYERS = (
    Player("P01", "Avery Cole", "Cole", "Primary guard", "guard", 96, "#d1495b"),
    Player("P02", "Marcus Vale", "Vale", "Impact wing", "wing", 93, "#2878b5"),
    Player("P03", "Darius Okafor", "Okafor", "Impact center", "center", 91, "#e98b2a"),
    Player("P04", "Nico Park", "Park", "Spacing wing", "wing", 86, "#1b9e77"),
    Player("P05", "Theo Grant", "Grant", "Stretch forward", "forward", 84, "#a05eb5"),
)


@dataclass
class Parameters:
    min_scale: torch.Tensor
    shear: torch.Tensor
    rotation: torch.Tensor
    leader_gain: torch.Tensor
    follower_gain: torch.Tensor
    disturbance_amplitude: torch.Tensor
    tracking_noise_amplitude: torch.Tensor
    disturbance_phase: torch.Tensor
    noise_phase: torch.Tensor
    initial_noise: torch.Tensor

    def select(self, index: int) -> "Parameters":
        return Parameters(
            min_scale=self.min_scale[index : index + 1],
            shear=self.shear[index : index + 1],
            rotation=self.rotation[index : index + 1],
            leader_gain=self.leader_gain[index : index + 1],
            follower_gain=self.follower_gain[index : index + 1],
            disturbance_amplitude=self.disturbance_amplitude[index : index + 1],
            tracking_noise_amplitude=self.tracking_noise_amplitude[index : index + 1],
            disturbance_phase=self.disturbance_phase[index : index + 1],
            noise_phase=self.noise_phase[index : index + 1],
            initial_noise=self.initial_noise[index : index + 1],
        )


def select_leaders(players: tuple[Player, ...]) -> tuple[int, int, int]:
    guard = max(
        (index for index, player in enumerate(players) if player.role_group == "guard"),
        key=lambda index: players[index].impact_rating,
    )
    wing = max(
        (index for index, player in enumerate(players) if player.role_group == "wing"),
        key=lambda index: players[index].impact_rating,
    )
    center = max(
        (index for index, player in enumerate(players) if player.role_group == "center"),
        key=lambda index: players[index].impact_rating,
    )
    return guard, wing, center


LEADER_INDICES = select_leaders(PLAYERS)
FOLLOWER_INDICES = tuple(index for index in range(N_PLAYERS) if index not in LEADER_INDICES)
assert LEADER_INDICES == (0, 1, 2)
assert FOLLOWER_INDICES == (3, 4)


def nominal_configuration(device: torch.device) -> torch.Tensor:
    """Healthy five-out geometry in centroid-relative court feet."""

    return torch.tensor(
        [
            (0.0, 13.0),   # primary guard above the break
            (-15.0, 3.0),  # impact wing
            (15.0, 3.0),   # impact center spaced to the perimeter
            (-18.0, -10.0),
            (18.0, -10.0),
        ],
        dtype=torch.float32,
        device=device,
    )


def affine_weights(q: torch.Tensor) -> torch.Tensor:
    leaders = q[list(LEADER_INDICES)]
    augmented = torch.cat(
        [leaders.T, torch.ones((1, N_LEADERS), dtype=q.dtype, device=q.device)], dim=0
    )
    rows = []
    for index in FOLLOWER_INDICES:
        rhs = torch.cat([q[index], torch.ones((1,), dtype=q.dtype, device=q.device)])
        rows.append(torch.linalg.solve(augmented, rhs))
    weights = torch.stack(rows)
    reconstructed = weights @ leaders
    if not torch.allclose(reconstructed, q[list(FOLLOWER_INDICES)], atol=1.0e-5):
        raise RuntimeError("Follower affine weights do not reconstruct design geometry.")
    if not torch.allclose(weights.sum(dim=1), torch.ones(2, device=q.device), atol=1.0e-5):
        raise RuntimeError("Follower affine weights do not sum to one.")
    return weights


def make_parameters(device: torch.device) -> Parameters:
    generator = torch.Generator(device=device).manual_seed(SEED)
    return Parameters(
        min_scale=-0.015
        + 0.190 * torch.rand(BATCH_SIZE, generator=generator, device=device),
        shear=-0.060
        + 0.120 * torch.rand(BATCH_SIZE, generator=generator, device=device),
        rotation=-0.085
        + 0.170 * torch.rand(BATCH_SIZE, generator=generator, device=device),
        leader_gain=2.55
        + 0.55 * torch.rand(BATCH_SIZE, generator=generator, device=device),
        follower_gain=2.85
        + 0.55 * torch.rand(BATCH_SIZE, generator=generator, device=device),
        disturbance_amplitude=0.018
        + 0.032 * torch.rand(BATCH_SIZE, generator=generator, device=device),
        tracking_noise_amplitude=0.012
        + 0.022 * torch.rand(BATCH_SIZE, generator=generator, device=device),
        disturbance_phase=2.0
        * math.pi
        * torch.rand(
            (BATCH_SIZE, N_PLAYERS, 2), generator=generator, device=device
        ),
        noise_phase=2.0
        * math.pi
        * torch.rand(
            (BATCH_SIZE, N_PLAYERS, 2), generator=generator, device=device
        ),
        initial_noise=0.14
        * torch.randn(
            (BATCH_SIZE, N_PLAYERS, 2), generator=generator, device=device
        ),
    )


def cosine_ease(value: float) -> float:
    value = min(max(value, 0.0), 1.0)
    return 0.5 * (1.0 - math.cos(math.pi * value))


def phase_for_step(step: int) -> str:
    if step < 40:
        return "healthy five-out"
    if step < 105:
        return "horns / high pick-and-roll"
    if step < 215:
        return "progressive compression"
    if step < 235:
        return "maximum compression"
    if step < 285:
        return "recovery"
    return "healthy reset"


def candidate_affine_map(params: Parameters, step: int) -> torch.Tensor:
    batch = params.min_scale.shape[0]
    identity = torch.eye(2, dtype=torch.float32, device=params.min_scale.device)[None].repeat(
        batch, 1, 1
    )
    horns = torch.tensor(
        [[0.88, 0.18], [-0.05, 0.82]],
        dtype=torch.float32,
        device=params.min_scale.device,
    )[None].repeat(batch, 1, 1)

    cosine = torch.cos(params.rotation)
    sine = torch.sin(params.rotation)
    rotation = torch.stack(
        [
            torch.stack([cosine, -sine], dim=-1),
            torch.stack([sine, cosine], dim=-1),
        ],
        dim=-2,
    )
    compressed = torch.zeros_like(rotation)
    compressed[:, 0, 0] = 1.0
    compressed[:, 0, 1] = params.shear
    compressed[:, 1, 1] = params.min_scale
    compressed = rotation @ compressed

    if step < 40:
        return identity
    if step < 85:
        alpha = cosine_ease((step - 40) / 45.0)
        return (1.0 - alpha) * identity + alpha * horns
    if step < 105:
        return horns
    if step < 215:
        alpha = cosine_ease((step - 105) / 110.0)
        return (1.0 - alpha) * horns + alpha * compressed
    if step < 235:
        return compressed
    if step < 285:
        alpha = cosine_ease((step - 235) / 50.0)
        return (1.0 - alpha) * compressed + alpha * identity
    return identity


def court_translation(step: int, device: torch.device) -> torch.Tensor:
    # The small lateral shift suggests a high-screen possession while preserving
    # the exact same command tape in both arms.
    if step < 40:
        shift = 0.0
    elif step < 105:
        shift = cosine_ease((step - 40) / 65.0)
    elif step < 235:
        shift = 1.0
    elif step < 285:
        shift = 1.0 - cosine_ease((step - 235) / 50.0)
    else:
        shift = 0.0
    return torch.tensor(
        [COURT_CENTER[0] + 0.8 * shift, COURT_CENTER[1] + 0.35 * shift],
        dtype=torch.float32,
        device=device,
    )


def map_metrics(positions: torch.Tensor, q: torch.Tensor) -> tuple[torch.Tensor, ...]:
    design_leaders = q[list(LEADER_INDICES)]
    source = torch.stack(
        [
            design_leaders[1] - design_leaders[0],
            design_leaders[2] - design_leaders[0],
        ],
        dim=-1,
    )
    leaders = positions[..., list(LEADER_INDICES), :]
    target = torch.stack(
        [leaders[..., 1, :] - leaders[..., 0, :], leaders[..., 2, :] - leaders[..., 0, :]],
        dim=-1,
    )
    affine_map = target @ torch.linalg.inv(source)
    singular_values = torch.linalg.svdvals(affine_map)
    sigma_max = singular_values[..., 0]
    sigma_min = singular_values[..., -1]
    ratio = sigma_min / sigma_max.clamp_min(1.0e-8)
    area_scale = torch.abs(torch.linalg.det(affine_map))
    return sigma_min, sigma_max, ratio, area_scale


def candidate_metrics(affine_map: torch.Tensor) -> tuple[torch.Tensor, ...]:
    singular_values = torch.linalg.svdvals(affine_map)
    sigma_max = singular_values[..., 0]
    sigma_min = singular_values[..., -1]
    ratio = sigma_min / sigma_max.clamp_min(1.0e-8)
    area = torch.abs(torch.linalg.det(affine_map))
    return sigma_min, sigma_max, ratio, area


def pairwise_minimum(positions: torch.Tensor) -> torch.Tensor:
    distances = torch.cdist(positions, positions)
    identity = torch.eye(N_PLAYERS, dtype=torch.bool, device=positions.device)
    return distances.masked_fill(identity, float("inf")).amin(dim=(-2, -1))


def rmse(positions: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(torch.mean(torch.sum((positions - reference) ** 2, dim=-1), dim=-1))


def tracking_noise(params: Parameters, step: int, device: torch.device) -> torch.Tensor:
    player = torch.arange(N_PLAYERS, device=device, dtype=torch.float32)[None, :, None]
    axis = torch.arange(2, device=device, dtype=torch.float32)[None, None, :]
    frequency = 0.071 + 0.009 * player + 0.013 * axis
    return params.tracking_noise_amplitude[:, None, None] * torch.sin(
        step * frequency + params.noise_phase
    )


def disturbance(params: Parameters, step: int, device: torch.device) -> torch.Tensor:
    player = torch.arange(N_PLAYERS, device=device, dtype=torch.float32)[None, :, None]
    axis = torch.arange(2, device=device, dtype=torch.float32)[None, None, :]
    frequency = 0.043 + 0.006 * player + 0.010 * axis
    return params.disturbance_amplitude[:, None, None] * torch.sin(
        step * frequency + params.disturbance_phase
    )


def clamp_velocity(command: torch.Tensor) -> torch.Tensor:
    magnitude = torch.linalg.vector_norm(command, dim=-1, keepdim=True).clamp_min(1.0e-8)
    return command * torch.clamp(MAX_SPEED_FT_S / magnitude, max=1.0)


def simulate(
    params: Parameters,
    q: torch.Tensor,
    weights: torch.Tensor,
    record: bool = False,
    possession_id: str | None = None,
) -> tuple[dict[str, torch.Tensor], list[dict[str, object]]]:
    device = q.device
    batch = params.min_scale.shape[0]
    initial = q[None] + torch.tensor(COURT_CENTER, device=device)[None, None, :]
    initial = initial + params.initial_noise
    positions = torch.stack([initial.clone(), initial.clone()], dim=0)
    velocity = torch.zeros_like(positions)

    identity = torch.eye(2, device=device, dtype=q.dtype)[None].repeat(batch, 1, 1)
    accepted_map = torch.stack([identity.clone(), identity.clone()], dim=0)
    rta_fallback_map = identity.clone()
    rta_holding = torch.zeros(batch, dtype=torch.bool, device=device)
    baseline_holding = torch.zeros(batch, dtype=torch.bool, device=device)
    previous_tracking = torch.zeros((2, batch), device=device)
    previous_spacing = torch.full((2, batch), float("inf"), device=device)

    min_ratio = torch.full((2, batch), float("inf"), device=device)
    min_spacing = torch.full((2, batch), float("inf"), device=device)
    tracking_sum = torch.zeros((2, batch), device=device)
    follower_sum = torch.zeros((2, batch), device=device)
    max_tracking = torch.zeros((2, batch), device=device)
    rank_loss_steps = torch.zeros((2, batch), dtype=torch.int32, device=device)
    hidden_collapse_steps = torch.zeros((2, batch), dtype=torch.int32, device=device)
    spacing_violation_steps = torch.zeros((2, batch), dtype=torch.int32, device=device)
    collision_steps = torch.zeros((2, batch), dtype=torch.int32, device=device)
    reject_steps = torch.zeros((2, batch), dtype=torch.int32, device=device)
    hold_steps = torch.zeros((2, batch), dtype=torch.int32, device=device)
    rows: list[dict[str, object]] = []

    for step in range(STEPS):
        candidate_map = candidate_affine_map(params, step)
        translation = court_translation(step, device)
        candidate_reference = torch.einsum("bij,nj->bni", candidate_map, q) + translation
        _, _, candidate_ratio, candidate_area = candidate_metrics(candidate_map)
        robust_candidate_ratio = candidate_ratio - PROPOSAL_UNCERTAINTY_MARGIN

        baseline_pass = (
            (previous_tracking[0] <= TRACKING_RMSE_THRESHOLD_FT)
            & (previous_spacing[0] >= MINIMUM_SPACING_THRESHOLD_FT)
        )
        baseline_new_reject = (~baseline_pass) & (~baseline_holding)
        baseline_resume = baseline_holding & baseline_pass
        baseline_holding = torch.where(baseline_resume, False, baseline_holding | ~baseline_pass)
        accepted_map[0] = torch.where(
            baseline_pass[:, None, None], candidate_map, accepted_map[0]
        )

        checkpoint_admissible = (~rta_holding) & (
            candidate_ratio >= FALLBACK_CHECKPOINT_THRESHOLD
        )
        rta_fallback_map = torch.where(
            checkpoint_admissible[:, None, None], candidate_map, rta_fallback_map
        )
        rta_initial_reject = (~rta_holding) & (
            robust_candidate_ratio < CONDITION_THRESHOLD
        )
        rta_resume = rta_holding & (candidate_ratio >= RESUME_THRESHOLD)
        rta_holding = torch.where(rta_resume, False, rta_holding | rta_initial_reject)
        rta_accept = ~rta_holding
        accepted_map[1] = torch.where(
            rta_accept[:, None, None], candidate_map, rta_fallback_map
        )

        reference = torch.stack(
            [
                torch.einsum("bij,nj->bni", accepted_map[0], q) + translation,
                torch.einsum("bij,nj->bni", accepted_map[1], q) + translation,
            ],
            dim=0,
        )

        measured = positions + tracking_noise(params, step, device)[None]
        command = torch.zeros_like(positions)
        command[:, :, list(LEADER_INDICES), :] = params.leader_gain[None, :, None, None] * (
            reference[:, :, list(LEADER_INDICES), :]
            - measured[:, :, list(LEADER_INDICES), :]
        )
        follower_target = torch.einsum(
            "fl,abld->abfd", weights, positions[:, :, list(LEADER_INDICES), :]
        )
        command[:, :, list(FOLLOWER_INDICES), :] = params.follower_gain[
            None, :, None, None
        ] * (follower_target - measured[:, :, list(FOLLOWER_INDICES), :])
        command = clamp_velocity(command)

        common_disturbance = disturbance(params, step, device)
        velocity = 0.70 * velocity + 0.30 * command + common_disturbance[None]
        positions = positions + DT * velocity

        sigma_min, sigma_max, ratio, area_scale = map_metrics(positions, q)
        spacing = pairwise_minimum(positions)
        tracking = rmse(positions, reference)
        follower_error = rmse(
            positions[:, :, list(FOLLOWER_INDICES), :],
            reference[:, :, list(FOLLOWER_INDICES), :],
        )
        rank_loss = ratio < RANK_LOSS_THRESHOLD
        hidden = (
            (ratio < CONDITION_THRESHOLD)
            & (tracking <= TRACKING_RMSE_THRESHOLD_FT)
            & (spacing >= MINIMUM_SPACING_THRESHOLD_FT)
        )
        spacing_violation = spacing < MINIMUM_SPACING_THRESHOLD_FT
        collision = spacing < COLLISION_THRESHOLD_FT

        left_reject = baseline_new_reject
        left_hold = baseline_holding & ~baseline_new_reject
        right_reject = rta_initial_reject
        right_hold = rta_holding & ~rta_initial_reject

        min_ratio = torch.minimum(min_ratio, ratio)
        min_spacing = torch.minimum(min_spacing, spacing)
        tracking_sum += tracking
        follower_sum += follower_error
        max_tracking = torch.maximum(max_tracking, tracking)
        rank_loss_steps += rank_loss.to(torch.int32)
        hidden_collapse_steps += hidden.to(torch.int32)
        spacing_violation_steps += spacing_violation.to(torch.int32)
        collision_steps += collision.to(torch.int32)
        reject_steps[0] += left_reject.to(torch.int32)
        reject_steps[1] += right_reject.to(torch.int32)
        hold_steps[0] += left_hold.to(torch.int32)
        hold_steps[1] += right_hold.to(torch.int32)

        previous_tracking = tracking
        previous_spacing = spacing

        if record:
            arm_decisions = []
            for arm_index in range(2):
                if bool((left_reject if arm_index == 0 else right_reject)[0].item()):
                    arm_decisions.append("REJECT")
                elif bool((left_hold if arm_index == 0 else right_hold)[0].item()):
                    arm_decisions.append("HOLD LAST SAFE MAP")
                else:
                    arm_decisions.append("ACCEPT")

            actual_cpu = positions[:, 0].detach().cpu()
            reference_cpu = reference[:, 0].detach().cpu()
            candidate_cpu = candidate_reference[0].detach().cpu()
            for arm_index, arm in enumerate(ARMS):
                for player_index, player in enumerate(PLAYERS):
                    rows.append(
                        {
                            "frame": step,
                            "timestamp": round(step * DT, 6),
                            "possession_id": possession_id or f"{POSSESSION_PREFIX}-0000",
                            "arm": arm,
                            "player_id": player.player_id,
                            "player_name": player.display_name,
                            "role": player.role,
                            "impact_rating": player.impact_rating,
                            "x_ft": float(actual_cpu[arm_index, player_index, 0]),
                            "y_ft": float(actual_cpu[arm_index, player_index, 1]),
                            "reference_x_ft": float(reference_cpu[arm_index, player_index, 0]),
                            "reference_y_ft": float(reference_cpu[arm_index, player_index, 1]),
                            "proposed_x_ft": float(candidate_cpu[player_index, 0]),
                            "proposed_y_ft": float(candidate_cpu[player_index, 1]),
                            "designation": (
                                f"L{LEADER_INDICES.index(player_index) + 1}"
                                if player_index in LEADER_INDICES
                                else f"F{FOLLOWER_INDICES.index(player_index) + 1}"
                            ),
                            "sigma_min": float(sigma_min[arm_index, 0]),
                            "sigma_max": float(sigma_max[arm_index, 0]),
                            "normalized_conditioning": float(ratio[arm_index, 0]),
                            "proposed_normalized_conditioning": float(candidate_ratio[0]),
                            "robust_proposed_conditioning": float(robust_candidate_ratio[0]),
                            "area_scale": float(area_scale[arm_index, 0]),
                            "proposed_area_scale": float(candidate_area[0]),
                            "minimum_spacing_ft": float(spacing[arm_index, 0]),
                            "tracking_rmse_ft": float(tracking[arm_index, 0]),
                            "follower_rmse_ft": float(follower_error[arm_index, 0]),
                            "controller_decision": arm_decisions[arm_index],
                            "phase": phase_for_step(step),
                            "affine_rank_loss": int(rank_loss[arm_index, 0]),
                            "hidden_collapse": int(hidden[arm_index, 0]),
                        }
                    )

    metrics = {
        "min_ratio": min_ratio,
        "min_spacing": min_spacing,
        "mean_tracking": tracking_sum / STEPS,
        "mean_follower_rmse": follower_sum / STEPS,
        "max_tracking": max_tracking,
        "rank_loss_steps": rank_loss_steps,
        "hidden_collapse_steps": hidden_collapse_steps,
        "spacing_violation_steps": spacing_violation_steps,
        "collision_steps": collision_steps,
        "reject_steps": reject_steps,
        "hold_steps": hold_steps,
    }
    return metrics, rows


def episode_rows(metrics: dict[str, torch.Tensor], params: Parameters) -> list[dict[str, object]]:
    cpu = {name: tensor.detach().cpu() for name, tensor in metrics.items()}
    min_scale = params.min_scale.detach().cpu()
    shear = params.shear.detach().cpu()
    rotation = params.rotation.detach().cpu()
    rows: list[dict[str, object]] = []
    for arm_index, arm in enumerate(ARMS):
        for episode in range(BATCH_SIZE):
            rows.append(
                {
                    "arm": arm,
                    "episode": episode,
                    "minimum_commanded_scale": float(min_scale[episode]),
                    "commanded_shear": float(shear[episode]),
                    "commanded_rotation_rad": float(rotation[episode]),
                    "minimum_conditioning": float(cpu["min_ratio"][arm_index, episode]),
                    "minimum_spacing_ft": float(cpu["min_spacing"][arm_index, episode]),
                    "mean_tracking_rmse_ft": float(cpu["mean_tracking"][arm_index, episode]),
                    "mean_follower_rmse_ft": float(
                        cpu["mean_follower_rmse"][arm_index, episode]
                    ),
                    "maximum_tracking_rmse_ft": float(cpu["max_tracking"][arm_index, episode]),
                    "rank_loss_steps": int(cpu["rank_loss_steps"][arm_index, episode]),
                    "hidden_collapse_steps": int(
                        cpu["hidden_collapse_steps"][arm_index, episode]
                    ),
                    "spacing_violation_steps": int(
                        cpu["spacing_violation_steps"][arm_index, episode]
                    ),
                    "collision_steps": int(cpu["collision_steps"][arm_index, episode]),
                    "reject_steps": int(cpu["reject_steps"][arm_index, episode]),
                    "hold_steps": int(cpu["hold_steps"][arm_index, episode]),
                }
            )
    return rows


def aggregate_episode_rows(rows: list[dict[str, object]]) -> dict[str, dict[str, object]]:
    output: dict[str, dict[str, object]] = {}
    for arm in ARMS:
        subset = [row for row in rows if row["arm"] == arm]
        output[arm] = {
            "episodes": len(subset),
            "affine_rank_loss_episodes": sum(
                int(row["rank_loss_steps"]) > 0 for row in subset
            ),
            "hidden_collapse_episodes": sum(
                int(row["hidden_collapse_steps"]) > 0 for row in subset
            ),
            "spacing_violation_episodes": sum(
                int(row["spacing_violation_steps"]) > 0 for row in subset
            ),
            "collision_episodes": sum(int(row["collision_steps"]) > 0 for row in subset),
            "mean_tracking_rmse_ft": float(
                np.mean([float(row["mean_tracking_rmse_ft"]) for row in subset])
            ),
            "mean_follower_rmse_ft": float(
                np.mean([float(row["mean_follower_rmse_ft"]) for row in subset])
            ),
            "median_minimum_conditioning": float(
                np.median([float(row["minimum_conditioning"]) for row in subset])
            ),
            "median_minimum_spacing_ft": float(
                np.median([float(row["minimum_spacing_ft"]) for row in subset])
            ),
            "mean_reject_steps": float(
                np.mean([int(row["reject_steps"]) for row in subset])
            ),
            "mean_hold_steps": float(np.mean([int(row["hold_steps"]) for row in subset])),
        }
    return output


def choose_representative(metrics: dict[str, torch.Tensor], params: Parameters) -> int:
    baseline_hidden = metrics["hidden_collapse_steps"][0] > 15
    baseline_rank = metrics["rank_loss_steps"][0] > 8
    baseline_spacing = metrics["min_spacing"][0] >= MINIMUM_SPACING_THRESHOLD_FT
    baseline_tracking = metrics["max_tracking"][0] <= TRACKING_RMSE_THRESHOLD_FT
    rta_conditioned = metrics["min_ratio"][1] >= CONDITION_THRESHOLD
    candidates = torch.nonzero(
        baseline_hidden & baseline_rank & baseline_spacing & baseline_tracking & rta_conditioned
    ).flatten()
    if candidates.numel() == 0:
        candidates = torch.nonzero(baseline_hidden & baseline_rank & rta_conditioned).flatten()
    if candidates.numel() == 0:
        raise RuntimeError("No representative matched basketball episode was found.")
    target = torch.abs(params.min_scale[candidates] - 0.005)
    return int(candidates[torch.argmin(target)].item())


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"No rows supplied for {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def draw_half_court(ax: plt.Axes) -> None:
    floor = "#d7a866"
    line = "#fffaf0"
    paint = "#bd793f"
    ax.set_facecolor(floor)
    for y in np.linspace(0, COURT_LENGTH_FT, 13):
        ax.axhline(y, color="#c99153", linewidth=0.45, alpha=0.55, zorder=0)
    ax.add_patch(Rectangle((0, 0), COURT_WIDTH_FT, COURT_LENGTH_FT, fill=False, ec=line, lw=2.0))
    ax.add_patch(Rectangle((17, 0), 16, 19, fc=paint, ec=line, lw=1.6, alpha=0.38))
    ax.add_patch(Circle((25, 19), 6, fill=False, ec=line, lw=1.4))
    ax.plot([22, 28], [4, 4], color=line, lw=2.0)
    ax.add_patch(Circle((25, 5.25), 0.75, fill=False, ec="#df3b32", lw=2.0))
    ax.add_patch(Arc((25, 5.25), 8, 8, theta1=0, theta2=180, ec=line, lw=1.2))
    corner_y = 5.25 + math.sqrt(23.75**2 - 22.0**2)
    theta = math.degrees(math.acos(22.0 / 23.75))
    ax.plot([3, 3], [0, corner_y], color=line, lw=1.4)
    ax.plot([47, 47], [0, corner_y], color=line, lw=1.4)
    ax.add_patch(
        Arc((25, 5.25), 47.5, 47.5, theta1=theta, theta2=180 - theta, ec=line, lw=1.4)
    )
    ax.axhline(47, color=line, lw=1.8)
    ax.add_patch(Arc((25, 47), 12, 12, theta1=180, theta2=360, ec=line, lw=1.3))
    ax.set_xlim(-0.5, 50.5)
    ax.set_ylim(-0.5, 47.5)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def frame_index(rows: list[dict[str, object]]) -> dict[str, list[list[dict[str, object]]]]:
    indexed: dict[str, list[list[dict[str, object]]]] = {
        arm: [[] for _ in range(STEPS)] for arm in ARMS
    }
    for row in rows:
        indexed[str(row["arm"])][int(row["frame"])].append(row)
    for arm in ARMS:
        for frame in range(STEPS):
            indexed[arm][frame].sort(key=lambda row: str(row["player_id"]))
            if len(indexed[arm][frame]) != N_PLAYERS:
                raise RuntimeError(f"Missing coordinates for {arm} frame {frame}")
    return indexed


def render_replay(
    rows: list[dict[str, object]],
    results: dict[str, dict[str, object]],
    representative: int,
) -> None:
    indexed = frame_index(rows)
    fig = plt.figure(figsize=(19.2, 10.8), dpi=100, facecolor="#f4f6fa")
    fig.subplots_adjust(0, 0, 1, 1)
    left_ax = fig.add_axes([0.035, 0.225, 0.43, 0.62])
    right_ax = fig.add_axes([0.535, 0.225, 0.43, 0.62])
    axes = (left_ax, right_ax)
    arm_titles = ("Tracking + separation only", "Conditioning-aware runtime assurance")
    arm_colors = ("#a35d00", "#146c78")

    fig.text(
        0.035,
        0.965,
        "Affine formation control can track a collapsing basketball shape",
        fontsize=27,
        fontweight="bold",
        color="#172033",
        va="top",
    )
    fig.text(
        0.035,
        0.925,
        "Matched command, disturbance, and noise tapes | gray × marks show proposed positions",
        fontsize=14,
        color="#526176",
        va="top",
    )
    frame_text = fig.text(
        0.965, 0.962, "", ha="right", va="top", fontsize=13, fontweight="bold", color="#39465a"
    )

    timeline_left = 0.14
    timeline_right = 0.86
    timeline_y = 0.885
    fig.add_artist(
        plt.Line2D(
            [timeline_left, timeline_right],
            [timeline_y, timeline_y],
            transform=fig.transFigure,
            color="#adb7c5",
            lw=3,
        )
    )
    hidden_marker_frame = min(
        int(row["frame"])
        for row in rows
        if row["arm"] == ARMS[0]
        and row["player_id"] == PLAYERS[0].player_id
        and int(row["hidden_collapse"]) == 1
    )
    timeline_specs = [
        (0, "HEALTHY", "center"),
        (40, "HORNS / HIGH P&R", "center"),
        (105, "COMPRESSION", "center"),
        (hidden_marker_frame, "HIDDEN COLLAPSE", "right"),
        (235, "RECOVERY", "left"),
    ]
    for step, label, alignment in timeline_specs:
        x = timeline_left + (timeline_right - timeline_left) * step / (STEPS - 1)
        fig.add_artist(
            plt.Line2D([x, x], [timeline_y - 0.008, timeline_y + 0.008], transform=fig.transFigure, color="#65748a", lw=1.5)
        )
        fig.text(x, timeline_y + 0.014, label, ha=alignment, va="bottom", fontsize=9.5, color="#526176")
    timeline_cursor = plt.Line2D(
        [timeline_left, timeline_left],
        [timeline_y - 0.014, timeline_y + 0.014],
        transform=fig.transFigure,
        color="#7c3aed",
        lw=4,
    )
    fig.add_artist(timeline_cursor)

    artists: list[dict[str, object]] = []
    for arm_index, ax in enumerate(axes):
        draw_half_court(ax)
        ax.set_title(arm_titles[arm_index], fontsize=19, fontweight="bold", color="#172033", pad=10)
        proposed = ax.scatter([], [], marker="x", s=55, color="#354052", alpha=0.45, linewidth=1.5, zorder=3)
        proposed_triangle = LineCollection([], colors="#354052", linestyles="dashed", linewidths=1.2, alpha=0.42, zorder=2)
        ax.add_collection(proposed_triangle)
        follower_connections = LineCollection([], colors="#5a6575", linestyles="dotted", linewidths=1.1, alpha=0.55, zorder=4)
        ax.add_collection(follower_connections)
        leader_triangle = LineCollection([], colors="#7138e8", linewidths=3.0, alpha=0.95, zorder=5)
        ax.add_collection(leader_triangle)
        player_scatters = []
        labels = []
        for index, player in enumerate(PLAYERS):
            marker = "D" if index in LEADER_INDICES else "o"
            player_scatters.append(
                ax.scatter([], [], s=210, marker=marker, color=player.color, edgecolor="white", linewidth=2.0, zorder=7)
            )
            labels.append(
                ax.text(
                    0,
                    0,
                    "",
                    ha="center",
                    va="center",
                    fontsize=8.5,
                    color="white",
                    fontweight="bold",
                    zorder=8,
                )
            )
        status = ax.text(
            0.02,
            0.975,
            "",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=12,
            fontweight="bold",
            color=arm_colors[arm_index],
            bbox={"facecolor": "#fffdf8", "edgecolor": "none", "alpha": 0.88, "pad": 4},
            zorder=10,
        )
        phase = ax.text(
            0.98,
            0.975,
            "",
            transform=ax.transAxes,
            va="top",
            ha="right",
            fontsize=10.5,
            color="#223047",
            bbox={"facecolor": "#fffdf8", "edgecolor": "none", "alpha": 0.78, "pad": 3},
            zorder=10,
        )
        artists.append(
            {
                "proposed": proposed,
                "proposed_triangle": proposed_triangle,
                "follower_connections": follower_connections,
                "leader_triangle": leader_triangle,
                "players": player_scatters,
                "labels": labels,
                "status": status,
                "phase": phase,
            }
        )

    telemetry_text: list[dict[str, plt.Text]] = []
    for arm_index, x0 in enumerate((0.04, 0.54)):
        fig.add_artist(
            Rectangle(
                (x0, 0.052),
                0.42,
                0.135,
                transform=fig.transFigure,
                facecolor="#ffffff",
                edgecolor="#d8dee8",
                linewidth=1.0,
                zorder=0,
            )
        )
        entries: dict[str, plt.Text] = {}
        coords = {
            "singular": (x0 + 0.012, 0.163),
            "ratio": (x0 + 0.012, 0.132),
            "area": (x0 + 0.012, 0.101),
            "spacing": (x0 + 0.205, 0.163),
            "tracking": (x0 + 0.205, 0.132),
            "follower": (x0 + 0.205, 0.101),
            "decision": (x0 + 0.012, 0.067),
        }
        for key, (x, y) in coords.items():
            entries[key] = fig.text(
                x,
                y,
                "",
                fontsize=11.2 if key != "decision" else 12.5,
                color="#27364b" if key != "decision" else arm_colors[arm_index],
                fontweight="bold" if key == "decision" else "normal",
                va="center",
            )
        telemetry_text.append(entries)

    batch_text = fig.text(
        0.035,
        0.024,
        "",
        ha="left",
        va="center",
        fontsize=10.5,
        color="#4b596d",
    )
    fig.text(
        0.965,
        0.022,
        "Illustrative simulation only — not real basketball evidence or proof of betting profitability.",
        ha="right",
        va="center",
        fontsize=9.5,
        color="#68758a",
    )

    writer = FFMpegWriter(
        fps=FPS,
        codec="libx264",
        bitrate=9000,
        extra_args=["-pix_fmt", "yuv420p", "-preset", "medium", "-movflags", "+faststart"],
        metadata={
            "title": "Basketball affine formation conditioning simulation",
            "comment": "Deterministic programmatic simulation; illustrative only",
        },
    )

    poster_frame = 220
    poster_written = False
    with writer.saving(fig, str(MOVIE), dpi=100):
        for frame in range(STEPS):
            for arm_index, arm in enumerate(ARMS):
                frame_rows = indexed[arm][frame]
                positions = np.array([[float(row["x_ft"]), float(row["y_ft"])] for row in frame_rows])
                proposed_positions = np.array(
                    [[float(row["proposed_x_ft"]), float(row["proposed_y_ft"])] for row in frame_rows]
                )
                art = artists[arm_index]
                art["proposed"].set_offsets(proposed_positions)  # type: ignore[union-attr]
                proposed_segments = [
                    [proposed_positions[LEADER_INDICES[0]], proposed_positions[LEADER_INDICES[1]]],
                    [proposed_positions[LEADER_INDICES[1]], proposed_positions[LEADER_INDICES[2]]],
                    [proposed_positions[LEADER_INDICES[2]], proposed_positions[LEADER_INDICES[0]]],
                ]
                art["proposed_triangle"].set_segments(proposed_segments)  # type: ignore[union-attr]
                leader_segments = [
                    [positions[LEADER_INDICES[0]], positions[LEADER_INDICES[1]]],
                    [positions[LEADER_INDICES[1]], positions[LEADER_INDICES[2]]],
                    [positions[LEADER_INDICES[2]], positions[LEADER_INDICES[0]]],
                ]
                art["leader_triangle"].set_segments(leader_segments)  # type: ignore[union-attr]
                follower_segments = []
                for follower in FOLLOWER_INDICES:
                    for leader in LEADER_INDICES:
                        follower_segments.append([positions[follower], positions[leader]])
                art["follower_connections"].set_segments(follower_segments)  # type: ignore[union-attr]

                for player_index, player in enumerate(PLAYERS):
                    art["players"][player_index].set_offsets(positions[player_index][None, :])  # type: ignore[index,union-attr]
                    designation = str(frame_rows[player_index]["designation"])
                    art["labels"][player_index].set_position(positions[player_index])  # type: ignore[index,union-attr]
                    art["labels"][player_index].set_text(designation)  # type: ignore[index,union-attr]

                sample = frame_rows[0]
                decision = str(sample["controller_decision"])
                status_color = {
                    "ACCEPT": "#167c52",
                    "REJECT": "#b52e2e",
                    "HOLD LAST SAFE MAP": "#7138e8",
                }[decision]
                art["status"].set_text(decision)  # type: ignore[union-attr]
                art["status"].set_color(status_color)  # type: ignore[union-attr]
                art["phase"].set_text(str(sample["phase"]).upper())  # type: ignore[union-attr]

                telemetry = telemetry_text[arm_index]
                telemetry["singular"].set_text(
                    f"σmin {float(sample['sigma_min']):.3f}    σmax {float(sample['sigma_max']):.3f}"
                )
                telemetry["ratio"].set_text(
                    f"σmin / σmax  {float(sample['normalized_conditioning']):.3f}   threshold {CONDITION_THRESHOLD:.2f}"
                )
                telemetry["area"].set_text(f"area scale |det A|  {float(sample['area_scale']):.3f}")
                telemetry["spacing"].set_text(
                    f"minimum spacing  {float(sample['minimum_spacing_ft']):.2f} ft"
                )
                telemetry["tracking"].set_text(
                    f"formation tracking RMSE  {float(sample['tracking_rmse_ft']):.2f} ft"
                )
                telemetry["follower"].set_text(
                    f"follower RMSE  {float(sample['follower_rmse_ft']):.2f} ft"
                )
                telemetry["decision"].set_text(f"CURRENT STATE: {decision}")
                telemetry["decision"].set_color(status_color)

            x = timeline_left + (timeline_right - timeline_left) * frame / (STEPS - 1)
            timeline_cursor.set_xdata([x, x])
            frame_text.set_text(f"frame {frame:03d}/{STEPS - 1}  |  t={frame * DT:05.2f}s")
            left_result = results[ARMS[0]]
            right_result = results[ARMS[1]]
            batch_text.set_text(
                f"CUDA batch, {BATCH_SIZE:,} matched/arm: rank loss {left_result['affine_rank_loss_episodes']} → {right_result['affine_rank_loss_episodes']}  |  hidden collapse {left_result['hidden_collapse_episodes']} → {right_result['hidden_collapse_episodes']}"
            )
            writer.grab_frame()
            if frame == poster_frame and not poster_written:
                fig.savefig(POSTER, dpi=100, facecolor=fig.get_facecolor())
                poster_written = True

    plt.close(fig)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    OUT.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for this requested experiment.")
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(SEED)
    device = torch.device("cuda")
    q = nominal_configuration(device)
    weights = affine_weights(q)
    params = make_parameters(device)

    start = time.perf_counter()
    metrics, _ = simulate(params, q, weights, record=False)
    torch.cuda.synchronize()
    batch_seconds = time.perf_counter() - start

    episodes = episode_rows(metrics, params)
    results = aggregate_episode_rows(episodes)
    representative = choose_representative(metrics, params)
    representative_id = f"{POSSESSION_PREFIX}-{representative:04d}"
    _, coordinate_rows = simulate(
        params.select(representative),
        q,
        weights,
        record=True,
        possession_id=representative_id,
    )

    expected_rows = STEPS * len(ARMS) * N_PLAYERS
    if len(coordinate_rows) != expected_rows:
        raise RuntimeError(f"Expected {expected_rows} sidecar rows, found {len(coordinate_rows)}")
    write_csv(COORDINATES_CSV, coordinate_rows)
    write_csv(EPISODE_CSV, episodes)

    hidden_frames = [
        int(row["frame"])
        for row in coordinate_rows
        if row["arm"] == ARMS[0]
        and row["player_id"] == PLAYERS[0].player_id
        and int(row["hidden_collapse"]) == 1
    ]
    rank_frames = [
        int(row["frame"])
        for row in coordinate_rows
        if row["arm"] == ARMS[0]
        and row["player_id"] == PLAYERS[0].player_id
        and int(row["affine_rank_loss"]) == 1
    ]

    summary = {
        "schema_version": "1.0",
        "scope": "Illustrative deterministic programmatic simulation; not real basketball evidence or proof of betting profitability.",
        "random_seed": SEED,
        "device": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "simulation_parameters": {
            "matched_episodes_per_arm": BATCH_SIZE,
            "players": N_PLAYERS,
            "leaders": N_LEADERS,
            "simulation_frames": STEPS,
            "fps": FPS,
            "dt_seconds": DT,
            "duration_seconds": STEPS / FPS,
            "court_width_ft": COURT_WIDTH_FT,
            "court_length_ft": COURT_LENGTH_FT,
            "maximum_player_speed_ft_s": MAX_SPEED_FT_S,
            "command_sequence": [
                {"frames": [0, 39], "state": "healthy five-out"},
                {"frames": [40, 104], "state": "horns / high pick-and-roll"},
                {"frames": [105, 214], "state": "progressive compression"},
                {"frames": [215, 234], "state": "maximum compression"},
                {"frames": [235, 284], "state": "recovery"},
                {"frames": [285, 299], "state": "healthy reset"},
            ],
            "gpu_batch_seconds": batch_seconds,
            "matching": "Both arms use identical command maps, initial states, tracking-noise phases, and disturbance phases for each episode.",
        },
        "thresholds": {
            "conditioning_reject_below": CONDITION_THRESHOLD,
            "proposal_uncertainty_margin": PROPOSAL_UNCERTAINTY_MARGIN,
            "fallback_checkpoint_at_or_above": FALLBACK_CHECKPOINT_THRESHOLD,
            "conditioning_resume_at_or_above": RESUME_THRESHOLD,
            "affine_rank_loss_below": RANK_LOSS_THRESHOLD,
            "low_tracking_rmse_at_or_below_ft": TRACKING_RMSE_THRESHOLD_FT,
            "acceptable_pairwise_spacing_at_or_above_ft": MINIMUM_SPACING_THRESHOLD_FT,
            "collision_below_ft": COLLISION_THRESHOLD_FT,
            "hidden_collapse_definition": "normalized conditioning < 0.12 AND tracking RMSE <= 1.0 ft AND minimum pairwise spacing >= 2.0 ft",
        },
        "players": [
            {
                "player_id": player.player_id,
                "name": player.display_name,
                "role": player.role,
                "role_group": player.role_group,
                "impact_rating": player.impact_rating,
                "designation": (
                    f"L{LEADER_INDICES.index(index) + 1}"
                    if index in LEADER_INDICES
                    else f"F{FOLLOWER_INDICES.index(index) + 1}"
                ),
            }
            for index, player in enumerate(PLAYERS)
        ],
        "leader_selection": {
            "L1": PLAYERS[LEADER_INDICES[0]].player_id,
            "L2": PLAYERS[LEADER_INDICES[1]].player_id,
            "L3": PLAYERS[LEADER_INDICES[2]].player_id,
            "rule": "highest-impact primary guard, highest-impact wing, highest-impact center",
        },
        "follower_affine_weights": {
            PLAYERS[FOLLOWER_INDICES[row]].player_id: {
                PLAYERS[LEADER_INDICES[col]].player_id: float(weights[row, col].item())
                for col in range(N_LEADERS)
            }
            for row in range(len(FOLLOWER_INDICES))
        },
        "episode_results": results,
        "representative_episode": {
            "episode_index": representative,
            "possession_id": representative_id,
            "minimum_commanded_scale": float(params.min_scale[representative].item()),
            "commanded_shear": float(params.shear[representative].item()),
            "commanded_rotation_rad": float(params.rotation[representative].item()),
            "first_hidden_collapse_frame": min(hidden_frames) if hidden_frames else None,
            "hidden_collapse_frame_count": len(hidden_frames),
            "first_affine_rank_loss_frame": min(rank_frames) if rank_frames else None,
            "affine_rank_loss_frame_count": len(rank_frames),
            "sidecar_rows": len(coordinate_rows),
        },
    }
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    render_replay(coordinate_rows, results, representative)
    REPORT.write_text(
        "\n".join(
            [
                "# Basketball Affine-Formation Conditioning Simulation",
                "",
                summary["scope"],
                "",
                f"- CUDA device: `{summary['device']}`.",
                f"- Matched episodes per arm: {BATCH_SIZE}; deterministic seed: {SEED}.",
                f"- Video possession: `{representative_id}`; coordinate rows: {len(coordinate_rows)}.",
                f"- Tracking/separation arm: {results[ARMS[0]]['affine_rank_loss_episodes']}/{BATCH_SIZE} rank-loss episodes and {results[ARMS[0]]['hidden_collapse_episodes']}/{BATCH_SIZE} hidden-collapse episodes.",
                f"- Conditioning-RTA arm: {results[ARMS[1]]['affine_rank_loss_episodes']}/{BATCH_SIZE} rank-loss episodes and {results[ARMS[1]]['hidden_collapse_episodes']}/{BATCH_SIZE} hidden-collapse episodes.",
                "",
                "The followers use fixed design-time affine weights relative to the three role-selected leaders. The left controller gates only on prior-frame tracking RMSE and pairwise spacing. The right controller applies the 0.12 threshold to the proposed conditioning minus a disclosed 0.01 uncertainty allowance, returns toward its most recent fallback checkpoint at or above 0.18, and resumes at 0.18 to avoid threshold chatter and inertial undershoot.",
            ]
        ),
        encoding="utf-8",
    )

    manifest_files = [MOVIE, POSTER, COORDINATES_CSV, EPISODE_CSV, SUMMARY_JSON, REPORT]
    entries = [(sha256(path), path.name) for path in manifest_files]
    entries.append((sha256(Path(__file__)), f"analysis/{Path(__file__).name}"))
    MANIFEST.write_text(
        "".join(f"{digest}  {label}\n" for digest, label in entries), encoding="ascii"
    )
    print(json.dumps({"results": results, "representative": summary["representative_episode"]}, indent=2))
    print(f"movie={MOVIE}")
    print(f"coordinates={COORDINATES_CSV}")
    print(f"summary={SUMMARY_JSON}")
    print(f"manifest={MANIFEST}")


if __name__ == "__main__":
    main()
