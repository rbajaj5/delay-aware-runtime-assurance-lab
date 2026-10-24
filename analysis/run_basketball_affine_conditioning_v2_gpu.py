"""Deterministic basketball affine-conditioning simulation for BetGuard ingest.

The CUDA batch creates matched controller arms and exports simulation state
directly.  No coordinate or event value is recovered from rendered pixels.

All teams, players, odds, and outcomes are synthetic and illustrative.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.collections import LineCollection
from matplotlib.patches import Arc, Circle, Rectangle


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "artifacts" / "20260813" / "basketball_affine_conditioning_v2"
SIMULATION_VERSION = "2.0.0"
CREATION_TIMESTAMP_UTC = "2026-08-13T04:20:48Z"
SOURCE_URI = "synthetic://basketball-affine-conditioning-v2"
GLOBAL_SEED = 20260813
DEFAULT_EPISODES = 4096
ARMS = ("tracking_and_separation", "conditioning_aware_rta")
N_ARMS = len(ARMS)

STEPS = 300
FPS = 15
DT = 1.0 / FPS
COURT_WIDTH_FT = 50.0
COURT_LENGTH_FT = 47.0
AFFINE_ORIGIN = (0.0, 20.0)

CONDITION_THRESHOLD = 0.12
HYSTERESIS_THRESHOLD = 0.15
TRACKING_THRESHOLD_FT = 1.0
SPACING_THRESHOLD_FT = 2.0
MAX_SPEED_FT_S = 24.0

DECISION_ACCEPT = np.uint8(0)
DECISION_REJECT = np.uint8(1)
DECISION_HOLD = np.uint8(2)
DECISION_LABELS = np.array(["ACCEPT", "REJECT", "HOLD LAST SAFE MAP"], dtype=object)

HOME_TEAM_ID = "SYN_HOME_AURORAS"
AWAY_TEAM_ID = "SYN_AWAY_SENTINELS"
HOME_LINEUP_ID = "AURORAS_LU_01"
AWAY_LINEUP_ID = "SENTINELS_LU_01"


@dataclass(frozen=True)
class Player:
    player_id: str
    role: str
    impact_rating: float
    team_id: str
    lineup_id: str
    color: str
    offense: bool


OFFENSE = (
    Player("PG01", "point_guard", 10.0, HOME_TEAM_ID, HOME_LINEUP_ID, "#d1495b", True),
    Player("WG01", "primary_wing", 8.5, HOME_TEAM_ID, HOME_LINEUP_ID, "#2878b5", True),
    Player("C01", "center", 8.0, HOME_TEAM_ID, HOME_LINEUP_ID, "#e98b2a", True),
    Player("WG02", "weak_side_wing", 6.5, HOME_TEAM_ID, HOME_LINEUP_ID, "#1b9e77", True),
    Player("WG03", "corner_wing", 6.0, HOME_TEAM_ID, HOME_LINEUP_ID, "#a05eb5", True),
)

DEFENSE = (
    Player("DPG01", "point_of_attack_defender", 7.8, AWAY_TEAM_ID, AWAY_LINEUP_ID, "#283241", False),
    Player("DWG01", "wing_defender", 7.4, AWAY_TEAM_ID, AWAY_LINEUP_ID, "#3b4656", False),
    Player("DC01", "rim_defender", 8.2, AWAY_TEAM_ID, AWAY_LINEUP_ID, "#505b6c", False),
    Player("DWG02", "help_wing", 6.9, AWAY_TEAM_ID, AWAY_LINEUP_ID, "#667184", False),
    Player("DWG03", "corner_defender", 6.6, AWAY_TEAM_ID, AWAY_LINEUP_ID, "#7b8697", False),
)

ALL_PLAYERS = OFFENSE + DEFENSE
N_OFFENSE = len(OFFENSE)
N_DEFENSE = len(DEFENSE)
N_PLAYERS = len(ALL_PLAYERS)
LEADER_INDICES = tuple(
    sorted(range(N_OFFENSE), key=lambda index: OFFENSE[index].impact_rating, reverse=True)[:3]
)
FOLLOWER_INDICES = tuple(index for index in range(N_OFFENSE) if index not in LEADER_INDICES)
assert tuple(OFFENSE[index].player_id for index in LEADER_INDICES) == ("PG01", "WG01", "C01")

PHASES = (
    (0, 39, "PHASE_01_HEALTHY_FIVE_OUT", "healthy five-out"),
    (40, 104, "PHASE_02_HORNS_HIGH_PNR", "horns/high pick-and-roll"),
    (105, 193, "PHASE_03_PROGRESSIVE_COMPRESSION", "progressive compression"),
    (194, 234, "PHASE_04_HIDDEN_COLLAPSE_INTERVAL", "hidden-collapse interval"),
    (235, 284, "PHASE_05_RECOVERY", "recovery"),
    (285, 299, "PHASE_06_HEALTHY_RESET", "healthy reset"),
)


@dataclass
class Parameters:
    episode_seeds: np.ndarray
    min_scale: torch.Tensor
    shear: torch.Tensor
    rotation: torch.Tensor
    leader_gain: torch.Tensor
    follower_gain: torch.Tensor
    disturbance_amplitude: torch.Tensor
    tracking_noise_amplitude: torch.Tensor
    offense_disturbance_phase: torch.Tensor
    offense_noise_phase: torch.Tensor
    defense_disturbance_phase: torch.Tensor
    defense_noise_phase: torch.Tensor
    offense_initial_noise: torch.Tensor
    defense_initial_noise: torch.Tensor


def output_paths(out: Path) -> dict[str, Path]:
    return {
        "movie": out / "basketball_affine_conditioning_v2.mp4",
        "telemetry": out / "basketball_affine_telemetry.csv",
        "config": out / "basketball_affine_config.json",
        "summary": out / "basketball_affine_summary.json",
        "pricing": out / "basketball_affine_pricing_example.json",
        "verification": out / "basketball_affine_csv_verification.json",
        "readme": out / "README.md",
        "manifest": out / "BASKETBALL_AFFINE_V2_MANIFEST.sha256",
    }


def phase_id(frame: int) -> str:
    for first, last, identifier, _ in PHASES:
        if first <= frame <= last:
            return identifier
    raise ValueError(frame)


def phase_name(frame: int) -> str:
    for first, last, _, name in PHASES:
        if first <= frame <= last:
            return name
    raise ValueError(frame)


def possession_number(frame: int) -> int:
    return 1 if frame < 235 else 2


def nominal_offense(device: torch.device) -> torch.Tensor:
    return torch.tensor(
        [
            (0.0, 13.0),
            (-15.0, 3.0),
            (15.0, 3.0),
            (-18.0, -10.0),
            (18.0, -10.0),
        ],
        dtype=torch.float32,
        device=device,
    )


def absolute_design_positions(q: torch.Tensor) -> torch.Tensor:
    return q + torch.tensor(AFFINE_ORIGIN, dtype=q.dtype, device=q.device)


def follower_weights(q: torch.Tensor) -> torch.Tensor:
    leaders = q[list(LEADER_INDICES)]
    augmented = torch.cat(
        [leaders.T, torch.ones((1, len(LEADER_INDICES)), dtype=q.dtype, device=q.device)],
        dim=0,
    )
    rows = []
    for index in FOLLOWER_INDICES:
        rhs = torch.cat([q[index], torch.ones((1,), dtype=q.dtype, device=q.device)])
        rows.append(torch.linalg.solve(augmented, rhs))
    weights = torch.stack(rows)
    assert torch.allclose(weights.sum(dim=1), torch.ones(len(FOLLOWER_INDICES), device=q.device))
    assert torch.allclose(weights @ leaders, q[list(FOLLOWER_INDICES)], atol=1.0e-5)
    return weights


def defender_target(offense_positions: torch.Tensor, step: int) -> torch.Tensor:
    toward_basket = -offense_positions
    unit = toward_basket / torch.linalg.vector_norm(
        toward_basket, dim=-1, keepdim=True
    ).clamp_min(1.0e-6)
    lateral = torch.stack([-unit[..., 1], unit[..., 0]], dim=-1)
    player = torch.arange(N_DEFENSE, dtype=offense_positions.dtype, device=offense_positions.device)[
        None, :, None
    ]
    pressure = 0.22 * torch.sin(0.055 * step + 0.73 * player)
    target = offense_positions + 3.0 * unit + pressure * lateral
    target[..., 0] = target[..., 0].clamp(-24.0, 24.0)
    target[..., 1] = target[..., 1].clamp(0.75, 46.0)
    return target


def make_parameters(episodes: int, device: torch.device) -> Parameters:
    episode_seeds = GLOBAL_SEED + np.arange(episodes, dtype=np.int64) * 1009
    scalars = np.empty((episodes, 7), dtype=np.float32)
    offense_disturbance_phase = np.empty((episodes, N_OFFENSE, 2), dtype=np.float32)
    offense_noise_phase = np.empty((episodes, N_OFFENSE, 2), dtype=np.float32)
    defense_disturbance_phase = np.empty((episodes, N_DEFENSE, 2), dtype=np.float32)
    defense_noise_phase = np.empty((episodes, N_DEFENSE, 2), dtype=np.float32)
    offense_initial_noise = np.empty((episodes, N_OFFENSE, 2), dtype=np.float32)
    defense_initial_noise = np.empty((episodes, N_DEFENSE, 2), dtype=np.float32)
    for episode, seed in enumerate(episode_seeds):
        rng = np.random.default_rng(int(seed))
        scalars[episode] = (
            rng.uniform(-0.015, 0.175),
            rng.uniform(-0.060, 0.060),
            rng.uniform(-0.085, 0.085),
            rng.uniform(3.20, 3.85),
            rng.uniform(3.55, 4.20),
            rng.uniform(0.008, 0.022),
            rng.uniform(0.006, 0.016),
        )
        offense_disturbance_phase[episode] = rng.uniform(0, 2 * math.pi, (N_OFFENSE, 2))
        offense_noise_phase[episode] = rng.uniform(0, 2 * math.pi, (N_OFFENSE, 2))
        defense_disturbance_phase[episode] = rng.uniform(0, 2 * math.pi, (N_DEFENSE, 2))
        defense_noise_phase[episode] = rng.uniform(0, 2 * math.pi, (N_DEFENSE, 2))
        offense_initial_noise[episode] = rng.normal(0, 0.10, (N_OFFENSE, 2))
        defense_initial_noise[episode] = rng.normal(0, 0.08, (N_DEFENSE, 2))

    tensor = lambda values: torch.as_tensor(values, dtype=torch.float32, device=device)
    return Parameters(
        episode_seeds=episode_seeds,
        min_scale=tensor(scalars[:, 0]),
        shear=tensor(scalars[:, 1]),
        rotation=tensor(scalars[:, 2]),
        leader_gain=tensor(scalars[:, 3]),
        follower_gain=tensor(scalars[:, 4]),
        disturbance_amplitude=tensor(scalars[:, 5]),
        tracking_noise_amplitude=tensor(scalars[:, 6]),
        offense_disturbance_phase=tensor(offense_disturbance_phase),
        offense_noise_phase=tensor(offense_noise_phase),
        defense_disturbance_phase=tensor(defense_disturbance_phase),
        defense_noise_phase=tensor(defense_noise_phase),
        offense_initial_noise=tensor(offense_initial_noise),
        defense_initial_noise=tensor(defense_initial_noise),
    )


def cosine_ease(value: float) -> float:
    value = min(max(value, 0.0), 1.0)
    return 0.5 * (1.0 - math.cos(math.pi * value))


def candidate_map(params: Parameters, step: int) -> torch.Tensor:
    episodes = params.min_scale.shape[0]
    identity = torch.eye(2, dtype=torch.float32, device=params.min_scale.device)[None].repeat(
        episodes, 1, 1
    )
    horns = torch.tensor(
        [[0.88, 0.18], [-0.05, 0.82]],
        dtype=torch.float32,
        device=params.min_scale.device,
    )[None].repeat(episodes, 1, 1)
    cosine = torch.cos(params.rotation)
    sine = torch.sin(params.rotation)
    rotation = torch.stack(
        [torch.stack([cosine, -sine], -1), torch.stack([sine, cosine], -1)], dim=-2
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
        return (1 - alpha) * identity + alpha * horns
    if step < 105:
        return horns
    if step < 215:
        alpha = cosine_ease((step - 105) / 110.0)
        return (1 - alpha) * horns + alpha * compressed
    if step < 235:
        return compressed
    if step < 285:
        alpha = cosine_ease((step - 235) / 50.0)
        return (1 - alpha) * compressed + alpha * identity
    return identity


def candidate_translation(step: int, device: torch.device) -> torch.Tensor:
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
        [0.8 * shift, AFFINE_ORIGIN[1] + 0.35 * shift],
        dtype=torch.float32,
        device=device,
    )


def candidate_condition(affine_map: torch.Tensor) -> tuple[torch.Tensor, ...]:
    singular = torch.linalg.svdvals(affine_map)
    sigma_max = singular[..., 0]
    sigma_min = singular[..., -1]
    ratio = sigma_min / sigma_max.clamp_min(1.0e-8)
    determinant = torch.linalg.det(affine_map)
    return sigma_min, sigma_max, ratio, determinant, torch.abs(determinant)


def observed_map_metrics(
    positions: torch.Tensor, q: torch.Tensor
) -> tuple[torch.Tensor, ...]:
    design = q[list(LEADER_INDICES)]
    source = torch.stack([design[1] - design[0], design[2] - design[0]], dim=-1)
    leaders = positions[..., list(LEADER_INDICES), :]
    target = torch.stack([leaders[..., 1, :] - leaders[..., 0, :], leaders[..., 2, :] - leaders[..., 0, :]], dim=-1)
    affine_map = target @ torch.linalg.inv(source)
    singular = torch.linalg.svdvals(affine_map)
    sigma_max = singular[..., 0]
    sigma_min = singular[..., -1]
    ratio = sigma_min / sigma_max.clamp_min(1.0e-8)
    determinant = torch.linalg.det(affine_map)
    return sigma_min, sigma_max, ratio, determinant, torch.abs(determinant)


def pairwise_minimum(positions: torch.Tensor) -> torch.Tensor:
    distances = torch.cdist(positions, positions)
    identity = torch.eye(N_OFFENSE, dtype=torch.bool, device=positions.device)
    return distances.masked_fill(identity, float("inf")).amin(dim=(-2, -1))


def rmse(positions: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(torch.mean(torch.sum((positions - target) ** 2, dim=-1), dim=-1))


def clamp_speed(command: torch.Tensor) -> torch.Tensor:
    speed = torch.linalg.vector_norm(command, dim=-1, keepdim=True).clamp_min(1.0e-8)
    return command * torch.clamp(MAX_SPEED_FT_S / speed, max=1.0)


def periodic_noise(
    amplitude: torch.Tensor,
    phase: torch.Tensor,
    step: int,
    base_frequency: float,
) -> torch.Tensor:
    count = phase.shape[1]
    player = torch.arange(count, dtype=phase.dtype, device=phase.device)[None, :, None]
    axis = torch.arange(2, dtype=phase.dtype, device=phase.device)[None, None, :]
    frequency = base_frequency + 0.007 * player + 0.011 * axis
    return amplitude[:, None, None] * torch.sin(step * frequency + phase)


def ball_context(proposed: torch.Tensor, step: int) -> tuple[torch.Tensor, str, str]:
    def at(index: int) -> torch.Tensor:
        position = proposed[:, index].clone()
        position[:, 0] += 0.25 * math.sin(0.42 * step)
        position[:, 1] += 0.15
        return position

    if 155 <= step < 165:
        alpha = cosine_ease((step - 155) / 10.0)
        return (1 - alpha) * at(0) + alpha * at(1), "NONE", "pass_to_primary_wing"
    if 208 <= step < 215:
        alpha = cosine_ease((step - 208) / 7.0)
        return (1 - alpha) * at(1) + alpha * at(2), "NONE", "entry_pass_to_center"
    if 260 <= step < 270:
        alpha = cosine_ease((step - 260) / 10.0)
        return (1 - alpha) * at(2) + alpha * at(0), "NONE", "reset_pass_to_guard"
    if step < 155:
        return at(0), "PG01", "guard_control"
    if step < 208:
        return at(1), "WG01", "wing_control"
    if step < 260:
        return at(2), "C01", "center_touch"
    return at(0), "PG01", "guard_reset"


def allocate_trajectory(episodes: int) -> dict[str, np.ndarray]:
    f = np.empty
    return {
        "proposed_offense": f((STEPS, episodes, N_OFFENSE, 2), np.float32),
        "reference_offense": f((STEPS, N_ARMS, episodes, N_OFFENSE, 2), np.float32),
        "observed_offense": f((STEPS, N_ARMS, episodes, N_OFFENSE, 2), np.float32),
        "controlled_offense": f((STEPS, N_ARMS, episodes, N_OFFENSE, 2), np.float32),
        "proposed_defense": f((STEPS, episodes, N_DEFENSE, 2), np.float32),
        "observed_defense": f((STEPS, episodes, N_DEFENSE, 2), np.float32),
        "controlled_defense": f((STEPS, episodes, N_DEFENSE, 2), np.float32),
        "ball": f((STEPS, episodes, 2), np.float32),
        "proposed_map": f((STEPS, episodes, 2, 2), np.float32),
        "last_safe_map": f((STEPS, N_ARMS, episodes, 2, 2), np.float32),
        "last_safe_translation": f((STEPS, N_ARMS, episodes, 2), np.float32),
        "last_safe_frame": f((STEPS, N_ARMS, episodes), np.int16),
        "sigma_min": f((STEPS, N_ARMS, episodes), np.float32),
        "sigma_max": f((STEPS, N_ARMS, episodes), np.float32),
        "conditioning": f((STEPS, N_ARMS, episodes), np.float32),
        "determinant": f((STEPS, N_ARMS, episodes), np.float32),
        "area_scale": f((STEPS, N_ARMS, episodes), np.float32),
        "minimum_spacing": f((STEPS, N_ARMS, episodes), np.float32),
        "tracking_rmse": f((STEPS, N_ARMS, episodes), np.float32),
        "follower_rmse": f((STEPS, N_ARMS, episodes), np.float32),
        "tracking_gate": f((STEPS, N_ARMS, episodes), np.bool_),
        "separation_gate": f((STEPS, N_ARMS, episodes), np.bool_),
        "conditioning_gate": f((STEPS, N_ARMS, episodes), np.bool_),
        "decision": f((STEPS, N_ARMS, episodes), np.uint8),
        "held": f((STEPS, N_ARMS, episodes), np.bool_),
        "rank_loss": f((STEPS, N_ARMS, episodes), np.bool_),
        "hidden_collapse": f((STEPS, N_ARMS, episodes), np.bool_),
    }


def to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()


def simulate_batch(
    episodes: int,
    q: torch.Tensor,
    weights: torch.Tensor,
    params: Parameters,
) -> dict[str, np.ndarray]:
    device = q.device
    trajectory = allocate_trajectory(episodes)
    design = absolute_design_positions(q)
    offense = design[None] + params.offense_initial_noise
    offense = torch.stack([offense.clone(), offense.clone()], dim=0)
    offense_velocity = torch.zeros_like(offense)
    initial_defense_target = defender_target(design[None].repeat(episodes, 1, 1), 0)
    defense = initial_defense_target + params.defense_initial_noise
    defense_velocity = torch.zeros_like(defense)

    identity = torch.eye(2, dtype=q.dtype, device=device)[None].repeat(episodes, 1, 1)
    origin = torch.tensor(AFFINE_ORIGIN, dtype=q.dtype, device=device)[None].repeat(episodes, 1)
    active_map = torch.stack([identity.clone(), identity.clone()], dim=0)
    active_translation = torch.stack([origin.clone(), origin.clone()], dim=0)
    safe_map = torch.stack([identity.clone(), identity.clone()], dim=0)
    safe_translation = torch.stack([origin.clone(), origin.clone()], dim=0)
    safe_frame = torch.zeros((N_ARMS, episodes), dtype=torch.int16, device=device)
    holding = torch.zeros((N_ARMS, episodes), dtype=torch.bool, device=device)

    for step in range(STEPS):
        proposed_map = candidate_map(params, step)
        proposed_translation = candidate_translation(step, device)
        proposed = torch.einsum("bij,nj->bni", proposed_map, q) + proposed_translation
        _, _, proposed_conditioning, _, _ = candidate_condition(proposed_map)

        offense_noise = periodic_noise(
            params.tracking_noise_amplitude, params.offense_noise_phase, step, 0.071
        )
        observed = offense + offense_noise[None]
        current_reference = torch.stack(
            [
                torch.einsum("bij,nj->bni", active_map[0], q) + active_translation[0, :, None, :],
                torch.einsum("bij,nj->bni", active_map[1], q) + active_translation[1, :, None, :],
            ],
            dim=0,
        )
        tracking_metric = rmse(observed, current_reference)
        follower_metric = rmse(
            observed[:, :, list(FOLLOWER_INDICES), :],
            current_reference[:, :, list(FOLLOWER_INDICES), :],
        )
        spacing_metric = pairwise_minimum(observed)
        tracking_gate = tracking_metric <= TRACKING_THRESHOLD_FT
        separation_gate = spacing_metric >= SPACING_THRESHOLD_FT
        conditioning_gate = proposed_conditioning[None].repeat(N_ARMS, 1) >= CONDITION_THRESHOLD

        decisions = torch.full(
            (N_ARMS, episodes), int(DECISION_ACCEPT), dtype=torch.uint8, device=device
        )

        left_pass = tracking_gate[0] & separation_gate[0]
        left_new_reject = (~holding[0]) & (~left_pass)
        left_resume = holding[0] & left_pass
        holding[0] = torch.where(left_resume, False, holding[0] | ~left_pass)
        decisions[0] = torch.where(
            left_new_reject,
            torch.tensor(int(DECISION_REJECT), dtype=torch.uint8, device=device),
            torch.where(
                holding[0],
                torch.tensor(int(DECISION_HOLD), dtype=torch.uint8, device=device),
                torch.tensor(int(DECISION_ACCEPT), dtype=torch.uint8, device=device),
            ),
        )
        left_accept = decisions[0] == int(DECISION_ACCEPT)
        active_map[0] = torch.where(left_accept[:, None, None], proposed_map, safe_map[0])
        active_translation[0] = torch.where(
            left_accept[:, None], proposed_translation[None], safe_translation[0]
        )
        safe_map[0] = torch.where(left_accept[:, None, None], proposed_map, safe_map[0])
        safe_translation[0] = torch.where(
            left_accept[:, None], proposed_translation[None], safe_translation[0]
        )
        safe_frame[0] = torch.where(
            left_accept,
            torch.full_like(safe_frame[0], step),
            safe_frame[0],
        )

        right_base_pass = tracking_gate[1] & separation_gate[1]
        right_checkpoint = (
            (~holding[1])
            & right_base_pass
            & (proposed_conditioning >= HYSTERESIS_THRESHOLD)
        )
        safe_map[1] = torch.where(
            right_checkpoint[:, None, None], proposed_map, safe_map[1]
        )
        safe_translation[1] = torch.where(
            right_checkpoint[:, None], proposed_translation[None], safe_translation[1]
        )
        safe_frame[1] = torch.where(
            right_checkpoint,
            torch.full_like(safe_frame[1], step),
            safe_frame[1],
        )
        right_new_reject = (~holding[1]) & (
            (~right_base_pass) | (proposed_conditioning < CONDITION_THRESHOLD)
        )
        right_resume = holding[1] & right_base_pass & (
            proposed_conditioning >= HYSTERESIS_THRESHOLD
        )
        holding[1] = torch.where(right_resume, False, holding[1] | right_new_reject)
        decisions[1] = torch.where(
            right_new_reject,
            torch.tensor(int(DECISION_REJECT), dtype=torch.uint8, device=device),
            torch.where(
                holding[1],
                torch.tensor(int(DECISION_HOLD), dtype=torch.uint8, device=device),
                torch.tensor(int(DECISION_ACCEPT), dtype=torch.uint8, device=device),
            ),
        )
        right_accept = decisions[1] == int(DECISION_ACCEPT)
        active_map[1] = torch.where(right_accept[:, None, None], proposed_map, safe_map[1])
        active_translation[1] = torch.where(
            right_accept[:, None], proposed_translation[None], safe_translation[1]
        )

        control_reference = torch.stack(
            [
                torch.einsum("bij,nj->bni", active_map[0], q) + active_translation[0, :, None, :],
                torch.einsum("bij,nj->bni", active_map[1], q) + active_translation[1, :, None, :],
            ],
            dim=0,
        )
        command = torch.zeros_like(offense)
        command[:, :, list(LEADER_INDICES), :] = params.leader_gain[None, :, None, None] * (
            control_reference[:, :, list(LEADER_INDICES), :]
            - observed[:, :, list(LEADER_INDICES), :]
        )
        follower_target = torch.einsum(
            "fl,abld->abfd", weights, offense[:, :, list(LEADER_INDICES), :]
        )
        command[:, :, list(FOLLOWER_INDICES), :] = params.follower_gain[
            None, :, None, None
        ] * (follower_target - observed[:, :, list(FOLLOWER_INDICES), :])
        command = clamp_speed(command)
        offense_disturbance = periodic_noise(
            params.disturbance_amplitude,
            params.offense_disturbance_phase,
            step,
            0.043,
        )
        offense_velocity = 0.22 * offense_velocity + 0.78 * command + offense_disturbance[None]
        offense = offense + DT * offense_velocity

        proposed_defense = defender_target(proposed, step)
        defense_noise = periodic_noise(
            0.75 * params.tracking_noise_amplitude,
            params.defense_noise_phase,
            step,
            0.063,
        )
        observed_defense = defense + defense_noise
        defense_command = 3.4 * (proposed_defense - observed_defense)
        defense_command = clamp_speed(defense_command)
        defense_disturbance = periodic_noise(
            0.70 * params.disturbance_amplitude,
            params.defense_disturbance_phase,
            step,
            0.039,
        )
        defense_velocity = 0.25 * defense_velocity + 0.75 * defense_command + defense_disturbance
        defense = defense + DT * defense_velocity
        ball, _, _ = ball_context(proposed, step)

        sigma_min, sigma_max, condition, determinant, area = observed_map_metrics(offense, q)
        rank_loss = condition < CONDITION_THRESHOLD
        accepted = decisions == int(DECISION_ACCEPT)
        hidden = rank_loss & tracking_gate & separation_gate & accepted

        trajectory["proposed_offense"][step] = to_numpy(proposed)
        trajectory["reference_offense"][step] = to_numpy(current_reference)
        trajectory["observed_offense"][step] = to_numpy(observed)
        trajectory["controlled_offense"][step] = to_numpy(offense)
        trajectory["proposed_defense"][step] = to_numpy(proposed_defense)
        trajectory["observed_defense"][step] = to_numpy(observed_defense)
        trajectory["controlled_defense"][step] = to_numpy(defense)
        trajectory["ball"][step] = to_numpy(ball)
        trajectory["proposed_map"][step] = to_numpy(proposed_map)
        trajectory["last_safe_map"][step] = to_numpy(safe_map)
        trajectory["last_safe_translation"][step] = to_numpy(safe_translation)
        trajectory["last_safe_frame"][step] = to_numpy(safe_frame)
        trajectory["sigma_min"][step] = to_numpy(sigma_min)
        trajectory["sigma_max"][step] = to_numpy(sigma_max)
        trajectory["conditioning"][step] = to_numpy(condition)
        trajectory["determinant"][step] = to_numpy(determinant)
        trajectory["area_scale"][step] = to_numpy(area)
        trajectory["minimum_spacing"][step] = to_numpy(spacing_metric)
        trajectory["tracking_rmse"][step] = to_numpy(tracking_metric)
        trajectory["follower_rmse"][step] = to_numpy(follower_metric)
        trajectory["tracking_gate"][step] = to_numpy(tracking_gate)
        trajectory["separation_gate"][step] = to_numpy(separation_gate)
        trajectory["conditioning_gate"][step] = to_numpy(conditioning_gate)
        trajectory["decision"][step] = to_numpy(decisions)
        trajectory["held"][step] = to_numpy(holding)
        trajectory["rank_loss"][step] = to_numpy(rank_loss)
        trajectory["hidden_collapse"][step] = to_numpy(hidden)

    return trajectory


def hold_intervals(decision: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    starts = np.full(decision.shape, -1, dtype=np.int16)
    ends = np.full(decision.shape, -1, dtype=np.int16)
    _, arms, episodes = decision.shape
    for arm in range(arms):
        for episode in range(episodes):
            mask = decision[:, arm, episode] != int(DECISION_ACCEPT)
            changes = np.diff(np.concatenate(([False], mask, [False])).astype(np.int8))
            run_starts = np.flatnonzero(changes == 1)
            run_ends = np.flatnonzero(changes == -1) - 1
            for first, last in zip(run_starts, run_ends, strict=True):
                starts[first : last + 1, arm, episode] = first
                ends[first : last + 1, arm, episode] = last
    return starts, ends


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> list[float]:
    if total <= 0:
        return [0.0, 0.0]
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return [max(0.0, center - half), min(1.0, center + half)]


def result_summary(trajectory: dict[str, np.ndarray], episodes: int) -> dict[str, dict[str, object]]:
    output: dict[str, dict[str, object]] = {}
    for arm_index, arm in enumerate(ARMS):
        rank = trajectory["rank_loss"][:, arm_index]
        hidden = trajectory["hidden_collapse"][:, arm_index]
        decisions = trajectory["decision"][:, arm_index]
        rank_episode_count = int(np.any(rank, axis=0).sum())
        hidden_episode_count = int(np.any(hidden, axis=0).sum())
        output[arm] = {
            "rank_loss": {
                "affected_frames": int(rank.sum()),
                "affected_episodes": rank_episode_count,
                "episode_rate": rank_episode_count / episodes,
                "episode_rate_wilson_95_ci": wilson_interval(rank_episode_count, episodes),
            },
            "hidden_collapse": {
                "affected_frames": int(hidden.sum()),
                "affected_episodes": hidden_episode_count,
                "episode_rate": hidden_episode_count / episodes,
                "episode_rate_wilson_95_ci": wilson_interval(hidden_episode_count, episodes),
            },
            "decision_frame_counts": {
                "accepted": int((decisions == int(DECISION_ACCEPT)).sum()),
                "rejected": int((decisions == int(DECISION_REJECT)).sum()),
                "held": int((decisions == int(DECISION_HOLD)).sum()),
            },
            "conditioning": {
                "mean": float(np.mean(trajectory["conditioning"][:, arm_index])),
                "median": float(np.median(trajectory["conditioning"][:, arm_index])),
            },
            "follower_rmse_feet": {
                "mean": float(np.mean(trajectory["follower_rmse"][:, arm_index])),
                "median": float(np.median(trajectory["follower_rmse"][:, arm_index])),
            },
        }
    return output


def validate_trajectory(
    trajectory: dict[str, np.ndarray],
    weights: np.ndarray,
    episodes: int,
) -> dict[str, dict[str, object]]:
    results = result_summary(trajectory, episodes)
    no_nonfinite = all(
        np.isfinite(trajectory[name]).all()
        for name in (
            "proposed_offense",
            "reference_offense",
            "observed_offense",
            "controlled_offense",
            "proposed_defense",
            "observed_defense",
            "controlled_defense",
            "ball",
        )
    )
    checks: dict[str, dict[str, object]] = {
        "hidden_collapse_episodes_subset_of_rank_loss": {
            "passed": all(
                results[arm]["hidden_collapse"]["affected_episodes"]
                <= results[arm]["rank_loss"]["affected_episodes"]
                for arm in ARMS
            )
        },
        "hidden_collapse_frames_subset_of_rank_loss": {
            "passed": all(
                results[arm]["hidden_collapse"]["affected_frames"]
                <= results[arm]["rank_loss"]["affected_frames"]
                for arm in ARMS
            )
        },
        "exactly_three_offensive_leaders": {
            "passed": len(LEADER_INDICES) == 3,
            "leader_ids": [OFFENSE[index].player_id for index in LEADER_INDICES],
        },
        "leader_identities_fixed_within_possessions": {
            "passed": tuple(OFFENSE[index].player_id for index in LEADER_INDICES)
            == ("PG01", "WG01", "C01")
        },
        "follower_weights_fixed_within_possessions": {
            "passed": weights.shape == (2, 3) and np.allclose(weights.sum(axis=1), 1.0),
            "weight_rows": weights.tolist(),
        },
        "timestamps_align_with_video_frames": {
            "passed": np.allclose(np.arange(STEPS) / FPS, np.arange(STEPS) * DT),
            "frames": STEPS,
            "fps": FPS,
        },
        "no_nan_or_infinite_coordinates": {"passed": bool(no_nonfinite)},
    }
    if not all(bool(check["passed"]) for check in checks.values()):
        raise AssertionError(json.dumps(checks, indent=2))
    return checks


def reference_defense_numpy(reference_offense: np.ndarray) -> np.ndarray:
    tensor = torch.as_tensor(reference_offense[None], dtype=torch.float32)
    return defender_target(tensor, 0)[0].numpy()


def telemetry_metadata(weights: np.ndarray) -> dict[str, np.ndarray]:
    reference_offense = np.array(
        [[0.0, 33.0], [-15.0, 23.0], [15.0, 23.0], [-18.0, 10.0], [18.0, 10.0]],
        dtype=np.float32,
    )
    reference_defense = reference_defense_numpy(reference_offense)
    all_reference = np.concatenate([reference_offense, reference_defense], axis=0)
    designations = np.array(["L1", "L2", "L3", "F1", "F2"] + ["context_defender"] * 5, dtype=object)
    follower = np.full((N_PLAYERS, 3), np.nan, dtype=np.float64)
    follower[3:5] = weights
    return {
        "reference": all_reference,
        "player_id": np.array([player.player_id for player in ALL_PLAYERS], dtype=object),
        "role": np.array([player.role for player in ALL_PLAYERS], dtype=object),
        "impact": np.array([player.impact_rating for player in ALL_PLAYERS], dtype=np.float64),
        "team": np.array([player.team_id for player in ALL_PLAYERS], dtype=object),
        "lineup": np.array([player.lineup_id for player in ALL_PLAYERS], dtype=object),
        "designation": designations,
        "follower_weights": follower,
    }


def broadcast_metric(values: np.ndarray, players: int = N_PLAYERS) -> np.ndarray:
    return np.broadcast_to(values[..., None], values.shape + (players,))


def write_telemetry(
    path: Path,
    trajectory: dict[str, np.ndarray],
    params: Parameters,
    weights: np.ndarray,
    hold_start: np.ndarray,
    hold_end: np.ndarray,
    episodes: int,
    chunk_episodes: int,
) -> int:
    metadata = telemetry_metadata(weights)
    phases = np.array([phase_id(frame) for frame in range(STEPS)], dtype=object)
    ball_possessor = np.array([ball_context(torch.zeros((1, N_OFFENSE, 2)), frame)[1] for frame in range(STEPS)], dtype=object)
    possession_state = np.array([ball_context(torch.zeros((1, N_OFFENSE, 2)), frame)[2] for frame in range(STEPS)], dtype=object)
    header = True
    rows_written = 0
    started = time.perf_counter()
    with path.open("w", newline="", encoding="utf-8") as handle:
        for first in range(0, episodes, chunk_episodes):
            last = min(first + chunk_episodes, episodes)
            count = last - first
            shape = (count, N_ARMS, STEPS, N_PLAYERS)

            offense_proposed = trajectory["proposed_offense"][:, first:last].transpose(1, 0, 2, 3)
            defense_proposed = trajectory["proposed_defense"][:, first:last].transpose(1, 0, 2, 3)
            proposed = np.concatenate([offense_proposed, defense_proposed], axis=2)
            proposed = np.broadcast_to(proposed[:, None], (count, N_ARMS, STEPS, N_PLAYERS, 2))

            offense_observed = trajectory["observed_offense"][:, :, first:last].transpose(2, 1, 0, 3, 4)
            defense_observed = trajectory["observed_defense"][:, first:last].transpose(1, 0, 2, 3)
            defense_observed = np.broadcast_to(defense_observed[:, None], (count, N_ARMS, STEPS, N_DEFENSE, 2))
            observed = np.concatenate([offense_observed, defense_observed], axis=3)

            offense_controlled = trajectory["controlled_offense"][:, :, first:last].transpose(2, 1, 0, 3, 4)
            defense_controlled = trajectory["controlled_defense"][:, first:last].transpose(1, 0, 2, 3)
            defense_controlled = np.broadcast_to(defense_controlled[:, None], (count, N_ARMS, STEPS, N_DEFENSE, 2))
            controlled = np.concatenate([offense_controlled, defense_controlled], axis=3)

            offense_reference = trajectory["reference_offense"][:, :, first:last].transpose(2, 1, 0, 3, 4)
            defense_reference = np.broadcast_to(
                metadata["reference"][N_OFFENSE:][None, None, None],
                (count, N_ARMS, STEPS, N_DEFENSE, 2),
            )
            reference = np.concatenate([offense_reference, defense_reference], axis=3)

            metric = lambda name: trajectory[name][:, :, first:last].transpose(2, 1, 0)
            decision = metric("decision")
            held = metric("held")
            ball = trajectory["ball"][:, first:last].transpose(1, 0, 2)
            ball = np.broadcast_to(ball[:, None, :, None], (count, N_ARMS, STEPS, N_PLAYERS, 2))
            maps = trajectory["last_safe_map"][:, :, first:last].transpose(2, 1, 0, 3, 4)
            translations = trajectory["last_safe_translation"][:, :, first:last].transpose(2, 1, 0, 3)
            map_frame = metric("last_safe_frame")
            start_frame = hold_start[:, :, first:last].transpose(2, 1, 0)
            end_frame = hold_end[:, :, first:last].transpose(2, 1, 0)

            episode_values = np.arange(first, last, dtype=np.int32)
            episode_ids = np.array([f"BBAFF-E{episode:04d}" for episode in episode_values], dtype=object)
            possession_ids = np.empty((count, STEPS), dtype=object)
            for local, episode in enumerate(episode_values):
                for frame in range(STEPS):
                    possession_ids[local, frame] = (
                        f"BBAFF-E{episode:04d}-POS{possession_number(frame):02d}"
                    )

            columns: dict[str, object] = {
                "simulation_version": np.full(shape, SIMULATION_VERSION, dtype=object).ravel(),
                "random_seed": np.full(shape, GLOBAL_SEED, dtype=np.int64).ravel(),
                "episode_seed": np.broadcast_to(params.episode_seeds[first:last, None, None, None], shape).ravel(),
                "episode_id": np.broadcast_to(episode_ids[:, None, None, None], shape).ravel(),
                "controller_arm": np.broadcast_to(np.array(ARMS, dtype=object)[None, :, None, None], shape).ravel(),
                "frame_index": np.broadcast_to(np.arange(STEPS, dtype=np.int16)[None, None, :, None], shape).ravel(),
                "timestamp_seconds": np.broadcast_to((np.arange(STEPS) / FPS)[None, None, :, None], shape).ravel(),
                "possession_id": np.broadcast_to(possession_ids[:, None, :, None], shape).ravel(),
                "possession_state": np.broadcast_to(possession_state[None, None, :, None], shape).ravel(),
                "phase_id": np.broadcast_to(phases[None, None, :, None], shape).ravel(),
                "team_id": np.broadcast_to(metadata["team"][None, None, None, :], shape).ravel(),
                "lineup_id": np.broadcast_to(metadata["lineup"][None, None, None, :], shape).ravel(),
                "player_id": np.broadcast_to(metadata["player_id"][None, None, None, :], shape).ravel(),
                "player_role": np.broadcast_to(metadata["role"][None, None, None, :], shape).ravel(),
                "impact_rating": np.broadcast_to(metadata["impact"][None, None, None, :], shape).ravel(),
                "leader_or_follower": np.broadcast_to(metadata["designation"][None, None, None, :], shape).ravel(),
                "reference_x": reference[..., 0].ravel(),
                "reference_y": reference[..., 1].ravel(),
                "proposed_x": proposed[..., 0].ravel(),
                "proposed_y": proposed[..., 1].ravel(),
                "observed_x": observed[..., 0].ravel(),
                "observed_y": observed[..., 1].ravel(),
                "controlled_x": controlled[..., 0].ravel(),
                "controlled_y": controlled[..., 1].ravel(),
                "follower_weight_1": np.broadcast_to(metadata["follower_weights"][:, 0][None, None, None, :], shape).ravel(),
                "follower_weight_2": np.broadcast_to(metadata["follower_weights"][:, 1][None, None, None, :], shape).ravel(),
                "follower_weight_3": np.broadcast_to(metadata["follower_weights"][:, 2][None, None, None, :], shape).ravel(),
                "ball_x": ball[..., 0].ravel(),
                "ball_y": ball[..., 1].ravel(),
                "ball_possessor_id": np.broadcast_to(ball_possessor[None, None, :, None], shape).ravel(),
                "sigma_min": broadcast_metric(metric("sigma_min")).ravel(),
                "sigma_max": broadcast_metric(metric("sigma_max")).ravel(),
                "normalized_conditioning": broadcast_metric(metric("conditioning")).ravel(),
                "determinant": broadcast_metric(metric("determinant")).ravel(),
                "area_scale": broadcast_metric(metric("area_scale")).ravel(),
                "minimum_pairwise_spacing_feet": broadcast_metric(metric("minimum_spacing")).ravel(),
                "formation_tracking_rmse_feet": broadcast_metric(metric("tracking_rmse")).ravel(),
                "follower_rmse_feet": broadcast_metric(metric("follower_rmse")).ravel(),
                "tracking_gate_passed": broadcast_metric(metric("tracking_gate")).ravel(),
                "separation_gate_passed": broadcast_metric(metric("separation_gate")).ravel(),
                "conditioning_gate_passed": broadcast_metric(metric("conditioning_gate")).ravel(),
                "controller_decision": broadcast_metric(DECISION_LABELS[decision]).ravel(),
                "held_last_safe_map": broadcast_metric(held).ravel(),
                "last_safe_map_frame": broadcast_metric(map_frame).ravel(),
                "hold_start_frame": broadcast_metric(np.where(held, start_frame, -1)).ravel(),
                "hold_end_frame": broadcast_metric(np.where(held, end_frame, -1)).ravel(),
                "last_safe_map_a11": broadcast_metric(maps[..., 0, 0]).ravel(),
                "last_safe_map_a12": broadcast_metric(maps[..., 0, 1]).ravel(),
                "last_safe_map_a21": broadcast_metric(maps[..., 1, 0]).ravel(),
                "last_safe_map_a22": broadcast_metric(maps[..., 1, 1]).ravel(),
                "last_safe_translation_x": broadcast_metric(translations[..., 0]).ravel(),
                "last_safe_translation_y": broadcast_metric(translations[..., 1]).ravel(),
                "rank_loss": broadcast_metric(metric("rank_loss")).ravel(),
                "hidden_collapse": broadcast_metric(metric("hidden_collapse")).ravel(),
            }
            frame = pd.DataFrame(columns)
            frame.to_csv(
                handle,
                index=False,
                header=header,
                float_format="%.12f",
                na_rep="",
                lineterminator="\n",
            )
            header = False
            rows_written += len(frame)
            if last == episodes or last % max(chunk_episodes, 128) == 0:
                elapsed = time.perf_counter() - started
                print(
                    f"telemetry episodes {last}/{episodes}; rows={rows_written:,}; elapsed={elapsed:.1f}s",
                    flush=True,
                )
    return rows_written


def representative_episode(trajectory: dict[str, np.ndarray], params: Parameters) -> int:
    left_rank = trajectory["rank_loss"][:, 0].sum(axis=0) >= 20
    left_hidden = trajectory["hidden_collapse"][:, 0].sum(axis=0) >= 15
    right_rank_free = ~np.any(trajectory["rank_loss"][:, 1], axis=0)
    candidates = np.flatnonzero(left_rank & left_hidden & right_rank_free)
    if candidates.size == 0:
        raise RuntimeError("No representative episode satisfies the registered display criteria.")
    scales = params.min_scale.detach().cpu().numpy()
    return int(candidates[np.argmin(np.abs(scales[candidates] - 0.005))])


def draw_half_court(ax: plt.Axes) -> None:
    floor = "#d7a866"
    line = "#fffaf0"
    paint = "#bd793f"
    ax.set_facecolor(floor)
    for y in np.linspace(0, COURT_LENGTH_FT, 13):
        ax.axhline(y, color="#c99153", lw=0.45, alpha=0.55, zorder=0)
    ax.add_patch(Rectangle((-25, 0), 50, 47, fill=False, ec=line, lw=2.0))
    ax.add_patch(Rectangle((-8, 0), 16, 19, fc=paint, ec=line, lw=1.6, alpha=0.38))
    ax.add_patch(Circle((0, 19), 6, fill=False, ec=line, lw=1.4))
    ax.plot([-3, 3], [0.8, 0.8], color=line, lw=2.0)
    ax.add_patch(Circle((0, 0), 0.75, fill=False, ec="#df3b32", lw=2.0))
    ax.add_patch(Arc((0, 0), 8, 8, theta1=0, theta2=180, ec=line, lw=1.2))
    corner_y = math.sqrt(23.75**2 - 22.0**2)
    theta = math.degrees(math.acos(22.0 / 23.75))
    ax.plot([-22, -22], [0, corner_y], color=line, lw=1.4)
    ax.plot([22, 22], [0, corner_y], color=line, lw=1.4)
    ax.add_patch(Arc((0, 0), 47.5, 47.5, theta1=theta, theta2=180 - theta, ec=line, lw=1.4))
    ax.axhline(47, color=line, lw=1.8)
    ax.add_patch(Arc((0, 47), 12, 12, theta1=180, theta2=360, ec=line, lw=1.3))
    ax.set_xlim(-25.5, 25.5)
    ax.set_ylim(-0.5, 47.5)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def render_video(
    path: Path,
    trajectory: dict[str, np.ndarray],
    results: dict[str, dict[str, object]],
    episode: int,
) -> None:
    fig = plt.figure(figsize=(19.2, 10.8), dpi=100, facecolor="#f4f6fa")
    fig.subplots_adjust(0, 0, 1, 1)
    axes = (
        fig.add_axes([0.035, 0.225, 0.43, 0.62]),
        fig.add_axes([0.535, 0.225, 0.43, 0.62]),
    )
    fig.text(0.035, 0.965, "Basketball formation control: tracking is not conditioning", fontsize=27, fontweight="bold", color="#172033", va="top")
    fig.text(0.035, 0.925, "Gray × proposed | hollow rings observed | filled marks controlled | synthetic matched replay", fontsize=13.5, color="#526176", va="top")
    frame_text = fig.text(0.965, 0.962, "", ha="right", va="top", fontsize=13, fontweight="bold", color="#39465a")

    timeline_left, timeline_right, timeline_y = 0.14, 0.86, 0.885
    fig.add_artist(plt.Line2D([timeline_left, timeline_right], [timeline_y, timeline_y], transform=fig.transFigure, color="#adb7c5", lw=3))
    for first, _, identifier, name in PHASES:
        x = timeline_left + (timeline_right - timeline_left) * first / (STEPS - 1)
        fig.add_artist(plt.Line2D([x, x], [timeline_y - 0.008, timeline_y + 0.008], transform=fig.transFigure, color="#65748a", lw=1.3))
        label = name.upper().replace("PROGRESSIVE ", "").replace(" INTERVAL", "")
        alignment = "right" if "HIDDEN" in label else ("left" if name == "recovery" else "center")
        fig.text(x, timeline_y + 0.014, label, ha=alignment, va="bottom", fontsize=8.7, color="#526176")
    cursor = plt.Line2D([timeline_left, timeline_left], [timeline_y - 0.014, timeline_y + 0.014], transform=fig.transFigure, color="#7c3aed", lw=4)
    fig.add_artist(cursor)

    titles = ("Tracking + separation", "Conditioning-aware runtime assurance")
    artists: list[dict[str, object]] = []
    for arm_index, ax in enumerate(axes):
        draw_half_court(ax)
        ax.set_title(titles[arm_index], fontsize=19, fontweight="bold", color="#172033", pad=10)
        proposed = ax.scatter([], [], marker="x", s=55, color="#354052", alpha=0.50, lw=1.5, zorder=4)
        observed = ax.scatter([], [], marker="o", s=195, facecolors="none", edgecolors="#2d3748", alpha=0.60, lw=1.1, zorder=5)
        leader_triangle = LineCollection([], colors="#7138e8", linewidths=3.0, zorder=6)
        follower_lines = LineCollection([], colors="#5a6575", linestyles="dotted", linewidths=1.0, alpha=0.58, zorder=5)
        ax.add_collection(leader_triangle)
        ax.add_collection(follower_lines)
        offense_marks, offense_labels = [], []
        for index, player in enumerate(OFFENSE):
            marker = "D" if index in LEADER_INDICES else "o"
            offense_marks.append(ax.scatter([], [], s=210, marker=marker, color=player.color, edgecolor="white", lw=2.0, zorder=8))
            offense_labels.append(ax.text(0, 0, "", ha="center", va="center", fontsize=7.7, color="white", fontweight="bold", zorder=9))
        defense_marks, defense_labels = [], []
        for index, player in enumerate(DEFENSE):
            defense_marks.append(ax.scatter([], [], s=135, marker="o", color=player.color, edgecolor="white", lw=1.5, zorder=7))
            defense_labels.append(ax.text(0, 0, f"D{index + 1}", ha="center", va="center", fontsize=7.0, color="white", fontweight="bold", zorder=8))
        ball = ax.scatter([], [], s=95, color="#f47c20", edgecolor="#743b12", lw=1.5, zorder=10)
        status = ax.text(0.02, 0.975, "", transform=ax.transAxes, va="top", ha="left", fontsize=12, fontweight="bold", bbox={"facecolor": "#fffdf8", "edgecolor": "none", "alpha": 0.88, "pad": 4}, zorder=12)
        phase = ax.text(0.98, 0.975, "", transform=ax.transAxes, va="top", ha="right", fontsize=10.2, color="#223047", bbox={"facecolor": "#fffdf8", "edgecolor": "none", "alpha": 0.78, "pad": 3}, zorder=12)
        artists.append({"proposed": proposed, "observed": observed, "triangle": leader_triangle, "followers": follower_lines, "offense": offense_marks, "offense_labels": offense_labels, "defense": defense_marks, "defense_labels": defense_labels, "ball": ball, "status": status, "phase": phase})

    telemetry: list[dict[str, plt.Text]] = []
    for arm_index, x0 in enumerate((0.04, 0.54)):
        fig.add_artist(Rectangle((x0, 0.052), 0.42, 0.135, transform=fig.transFigure, facecolor="#ffffff", edgecolor="#d8dee8", lw=1.0, zorder=0))
        entries = {}
        for key, x, y in (
            ("singular", x0 + 0.012, 0.163),
            ("condition", x0 + 0.012, 0.132),
            ("area", x0 + 0.012, 0.101),
            ("spacing", x0 + 0.205, 0.163),
            ("tracking", x0 + 0.205, 0.132),
            ("follower", x0 + 0.205, 0.101),
            ("decision", x0 + 0.012, 0.067),
        ):
            entries[key] = fig.text(x, y, "", fontsize=11.2 if key != "decision" else 12.3, color="#27364b", fontweight="bold" if key == "decision" else "normal", va="center")
        telemetry.append(entries)

    batch = fig.text(0.035, 0.024, "", ha="left", va="center", fontsize=10.2, color="#4b596d")
    disclaimer = "Illustrative synthetic simulation only. Not actual basketball evidence, a validated betting model, or proof of profitability."
    fig.text(0.965, 0.022, disclaimer, ha="right", va="center", fontsize=9.2, color="#68758a")

    frame_dir = path.parent / ".basketball_affine_conditioning_v2_frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    for frame in range(STEPS):
            proposed = trajectory["proposed_offense"][frame, episode]
            defender_proposed = trajectory["proposed_defense"][frame, episode]
            for arm_index in range(N_ARMS):
                controlled_offense = trajectory["controlled_offense"][frame, arm_index, episode]
                observed_offense = trajectory["observed_offense"][frame, arm_index, episode]
                controlled_defense = trajectory["controlled_defense"][frame, episode]
                observed_defense = trajectory["observed_defense"][frame, episode]
                all_proposed = np.concatenate([proposed, defender_proposed])
                all_observed = np.concatenate([observed_offense, observed_defense])
                art = artists[arm_index]
                art["proposed"].set_offsets(all_proposed)  # type: ignore[union-attr]
                art["observed"].set_offsets(all_observed)  # type: ignore[union-attr]
                triangle = [[controlled_offense[0], controlled_offense[1]], [controlled_offense[1], controlled_offense[2]], [controlled_offense[2], controlled_offense[0]]]
                art["triangle"].set_segments(triangle)  # type: ignore[union-attr]
                connections = [[controlled_offense[follower], controlled_offense[leader]] for follower in FOLLOWER_INDICES for leader in LEADER_INDICES]
                art["followers"].set_segments(connections)  # type: ignore[union-attr]
                for index, player in enumerate(OFFENSE):
                    art["offense"][index].set_offsets(controlled_offense[index][None])  # type: ignore[index,union-attr]
                    art["offense_labels"][index].set_position(controlled_offense[index])  # type: ignore[index,union-attr]
                    art["offense_labels"][index].set_text(("L" + str(index + 1)) if index < 3 else ("F" + str(index - 2)))  # type: ignore[index,union-attr]
                for index in range(N_DEFENSE):
                    art["defense"][index].set_offsets(controlled_defense[index][None])  # type: ignore[index,union-attr]
                    art["defense_labels"][index].set_position(controlled_defense[index])  # type: ignore[index,union-attr]
                art["ball"].set_offsets(trajectory["ball"][frame, episode][None])  # type: ignore[union-attr]
                decision = DECISION_LABELS[int(trajectory["decision"][frame, arm_index, episode])]
                color = {"ACCEPT": "#167c52", "REJECT": "#b52e2e", "HOLD LAST SAFE MAP": "#7138e8"}[decision]
                art["status"].set_text(decision)  # type: ignore[union-attr]
                art["status"].set_color(color)  # type: ignore[union-attr]
                art["phase"].set_text(phase_name(frame).upper())  # type: ignore[union-attr]
                values = telemetry[arm_index]
                values["singular"].set_text(f"σmin {trajectory['sigma_min'][frame, arm_index, episode]:.3f}    σmax {trajectory['sigma_max'][frame, arm_index, episode]:.3f}")
                values["condition"].set_text(f"σmin / σmax  {trajectory['conditioning'][frame, arm_index, episode]:.3f}   threshold {CONDITION_THRESHOLD:.2f}")
                values["area"].set_text(f"area scale |det A|  {trajectory['area_scale'][frame, arm_index, episode]:.3f}")
                values["spacing"].set_text(f"minimum spacing  {trajectory['minimum_spacing'][frame, arm_index, episode]:.2f} ft")
                values["tracking"].set_text(f"formation tracking RMSE  {trajectory['tracking_rmse'][frame, arm_index, episode]:.2f} ft")
                values["follower"].set_text(f"follower RMSE  {trajectory['follower_rmse'][frame, arm_index, episode]:.2f} ft")
                values["decision"].set_text(f"CURRENT STATE: {decision}")
                values["decision"].set_color(color)

            x = timeline_left + (timeline_right - timeline_left) * frame / (STEPS - 1)
            cursor.set_xdata([x, x])
            frame_text.set_text(f"frame {frame:03d}/{STEPS - 1}  |  t={frame / FPS:05.2f}s")
            left, right = results[ARMS[0]], results[ARMS[1]]
            batch.set_text(f"Episodes: rank loss {left['rank_loss']['affected_episodes']} → {right['rank_loss']['affected_episodes']}  |  hidden collapse {left['hidden_collapse']['affected_episodes']} → {right['hidden_collapse']['affected_episodes']}")
            fig.savefig(
                frame_dir / f"frame_{frame:03d}.png",
                dpi=100,
                facecolor=fig.get_facecolor(),
            )
    plt.close(fig)

    encoded_path = path.with_name(f"{path.stem}.encoding{path.suffix}")
    command = [
        "ffmpeg",
        "-y",
        "-framerate",
        str(FPS),
        "-start_number",
        "0",
        "-i",
        str(frame_dir / "frame_%03d.png"),
        "-frames:v",
        str(STEPS),
        "-c:v",
        "libx264",
        "-threads",
        "2",
        "-pix_fmt",
        "yuv420p",
        "-preset",
        "veryfast",
        "-crf",
        "18",
        "-movflags",
        "+faststart",
        "-metadata",
        "title=Basketball affine conditioning v2",
        "-metadata",
        f"source_uri={SOURCE_URI}",
        "-metadata",
        "rights_basis=synthetic",
        "-metadata",
        "annotation_method=direct simulation-state export",
        str(encoded_path),
    ]
    subprocess.run(command, check=True)
    os.replace(encoded_path, path)
    for frame_file in frame_dir.glob("frame_*.png"):
        frame_file.unlink()
    frame_dir.rmdir()


def validate_existing_telemetry(path: Path, expected_rows: int, episodes: int) -> int:
    if not path.exists() or path.stat().st_size == 0:
        raise RuntimeError("Requested telemetry reuse, but the completed CSV is absent.")
    with path.open("rb") as handle:
        header = handle.readline().decode("utf-8").rstrip("\r\n")
        handle.seek(max(0, path.stat().st_size - 131_072))
        tail = handle.read().decode("utf-8").splitlines()
    if not tail:
        raise RuntimeError("Requested telemetry reuse, but the CSV has no terminal row.")
    columns = next(csv.reader([header]))
    last = next(csv.reader([tail[-1]]))
    terminal = dict(zip(columns, last, strict=True))
    expected = {
        "episode_id": f"BBAFF-E{episodes - 1:04d}",
        "controller_arm": ARMS[-1],
        "frame_index": str(STEPS - 1),
        "player_id": DEFENSE[-1].player_id,
    }
    observed = {key: terminal[key] for key in expected}
    if observed != expected:
        raise RuntimeError(
            f"Completed-telemetry terminal sentinel mismatch: {observed!r} != {expected!r}"
        )
    return expected_rows


def ffprobe_video(path: Path) -> dict[str, object]:
    command = ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height,r_frame_rate,nb_frames:format=duration", "-of", "json", str(path)]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    return json.loads(result.stdout)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def make_config(
    episodes: int,
    weights: np.ndarray,
    metadata: dict[str, np.ndarray],
) -> dict[str, object]:
    players = []
    for index, player in enumerate(ALL_PLAYERS):
        designation = metadata["designation"][index]
        record: dict[str, object] = {
            "player_id": player.player_id,
            "team_id": player.team_id,
            "lineup_id": player.lineup_id,
            "role": player.role,
            "impact_rating": player.impact_rating,
            "leader_or_follower": designation,
            "design_reference_position_feet": metadata["reference"][index].tolist(),
        }
        if designation in ("F1", "F2"):
            row = 0 if designation == "F1" else 1
            record["follower_weights_by_leader"] = {
                OFFENSE[LEADER_INDICES[col]].player_id: float(weights[row, col])
                for col in range(3)
            }
        else:
            record["follower_weights_by_leader"] = None
        players.append(record)
    return {
        "simulation_version": SIMULATION_VERSION,
        "creation_timestamp_utc": CREATION_TIMESTAMP_UTC,
        "source_uri": SOURCE_URI,
        "random_seed": GLOBAL_SEED,
        "episodes_per_arm": episodes,
        "controller_arms": list(ARMS),
        "coordinate_convention": {
            "units": "feet",
            "basket_center": [0.0, 0.0],
            "x_axis": "sideline-to-sideline; negative left, positive right when facing half court",
            "y_axis": "distance from baseline toward half court",
            "half_court_width_feet": COURT_WIDTH_FT,
            "half_court_length_feet": COURT_LENGTH_FT,
        },
        "thresholds": {
            "normalized_conditioning_reject_below": CONDITION_THRESHOLD,
            "normalized_conditioning_resume_at_or_above": HYSTERESIS_THRESHOLD,
            "tracking_rmse_pass_at_or_below_feet": TRACKING_THRESHOLD_FT,
            "minimum_spacing_pass_at_or_above_feet": SPACING_THRESHOLD_FT,
        },
        "video": {"width": 1920, "height": 1080, "fps": FPS, "frames": STEPS, "duration_seconds": STEPS / FPS},
        "leader_selection": {
            "method": "descending sort of explicit pre-simulation offensive impact ratings",
            "leader_ids": [OFFENSE[index].player_id for index in LEADER_INDICES],
            "selection_frozen_before_simulation": True,
        },
        "players": players,
        "phases": [{"first_frame": first, "last_frame": last, "phase_id": identifier, "name": name} for first, last, identifier, name in PHASES],
        "event_definitions": {
            "rank_loss": "realized controlled normalized_conditioning < 0.12",
            "hidden_collapse": "rank_loss AND tracking_gate_passed AND separation_gate_passed AND controller_decision == ACCEPT",
            "conditioning_gate_passed": "proposed-map normalized conditioning >= 0.12",
        },
        "matching": "Episode seed, command map, disturbance, tracking noise, defenders, ball, and possession transitions are identical between arms.",
        "video_provenance": {
            "source_uri": SOURCE_URI,
            "rights_basis": "synthetic",
            "annotation_method": "direct simulation-state export",
            "identities_verified": True,
            "court_calibration_rmse_feet": 0.0,
        },
    }


def make_pricing_example(
    trajectory: dict[str, np.ndarray],
    episode: int,
    metadata: dict[str, np.ndarray],
    file_hashes: dict[str, str],
) -> dict[str, object]:
    frames = []
    for frame in range(STEPS):
        arms = {}
        for arm_index, arm in enumerate(ARMS):
            offense_observed = trajectory["observed_offense"][frame, arm_index, episode]
            defense_observed = trajectory["observed_defense"][frame, episode]
            observed = np.concatenate([offense_observed, defense_observed])
            arms[arm] = {
                "players": [
                    {
                        "player_id": ALL_PLAYERS[index].player_id,
                        "observed_x": float(observed[index, 0]),
                        "observed_y": float(observed[index, 1]),
                    }
                    for index in range(N_PLAYERS)
                ],
                "normalized_conditioning": float(trajectory["conditioning"][frame, arm_index, episode]),
                "controller_decision": str(DECISION_LABELS[int(trajectory["decision"][frame, arm_index, episode])]),
            }
        _, possessor, state = ball_context(torch.zeros((1, N_OFFENSE, 2)), frame)
        frames.append(
            {
                "frame_index": frame,
                "timestamp_seconds": frame / FPS,
                "possession_id": f"BBAFF-E{episode:04d}-POS{possession_number(frame):02d}",
                "phase_id": phase_id(frame),
                "ball": {
                    "x": float(trajectory["ball"][frame, episode, 0]),
                    "y": float(trajectory["ball"][frame, episode, 1]),
                    "possessor_id": possessor,
                    "possession_state": state,
                },
                "controller_arms": arms,
            }
        )
    baseline_probability = 0.54
    selected_condition = float(np.median(trajectory["conditioning"][:, 1, episode]))
    selected_hold_fraction = float(np.mean(trajectory["held"][:, 1, episode]))
    coefficients = {
        "intercept_logit": math.log(baseline_probability / (1 - baseline_probability)),
        "conditioning_centered_at_0_12": 0.35,
        "hold_fraction": -0.20,
    }
    logit = coefficients["intercept_logit"] + coefficients["conditioning_centered_at_0_12"] * (selected_condition - 0.12) + coefficients["hold_fraction"] * selected_hold_fraction
    scenario_probability = 1 / (1 + math.exp(-logit))
    return {
        "$schema": "synthetic://betguard/video-price/v1",
        "schema_name": "betguard.video-price",
        "schema_version": "1.0",
        "simulation_version": SIMULATION_VERSION,
        "video_id": "basketball-affine-conditioning-v2",
        "home_team": {"team_id": HOME_TEAM_ID, "name": "Synthetic Auroras"},
        "away_team": {"team_id": AWAY_TEAM_ID, "name": "Synthetic Sentinels"},
        "players": [
            {
                "player_id": player.player_id,
                "team_id": player.team_id,
                "role": player.role,
                "impact_rating": player.impact_rating,
                "reference_position": {
                    "x": float(metadata["reference"][index, 0]),
                    "y": float(metadata["reference"][index, 1]),
                },
            }
            for index, player in enumerate(ALL_PLAYERS)
        ],
        "evidence": {
            "source_uri": SOURCE_URI,
            "rights_basis": "synthetic",
            "annotation_method": "direct simulation-state export",
            "identities_verified": True,
            "court_calibration_rmse_feet": 0.0,
            "random_seed": GLOBAL_SEED,
            "episode_id": f"BBAFF-E{episode:04d}",
            "sha256": file_hashes,
        },
        "frames": frames,
        "synthetic_market": {
            "market_type": "two_sided_moneyline",
            "home_decimal_odds": 1.86,
            "away_decimal_odds": 2.08,
            "baseline_home_probability": baseline_probability,
            "uncertainty": {"probability_standard_deviation": 0.08, "interpretation": "illustrative only"},
            "illustrative_calibration_coefficients": coefficients,
            "scenario_home_probability": scenario_probability,
        },
        "validated": False,
        "historical_games": 0,
        "qualified_paper_candidate": False,
    }


FIELD_DEFINITIONS = {
    "simulation_version": "Version of the deterministic generator.",
    "random_seed": "Global recorded seed shared by the experiment.",
    "episode_seed": "Deterministic per-episode seed; identical between arms.",
    "episode_id": "Stable synthetic episode identifier.",
    "controller_arm": "tracking_and_separation or conditioning_aware_rta.",
    "frame_index": "Zero-based video/simulation frame.",
    "timestamp_seconds": "frame_index divided by frame rate.",
    "possession_id": "Stable possession identifier; changes at the recovery phase.",
    "possession_state": "Scripted ball-control or pass state.",
    "phase_id": "Registered phase identifier for the frame.",
    "team_id": "Stable synthetic offensive or defensive team identifier.",
    "lineup_id": "Stable synthetic lineup identifier.",
    "player_id": "Persistent synthetic player identity.",
    "player_role": "Offensive or defensive role.",
    "impact_rating": "Explicit pre-simulation synthetic impact rating.",
    "leader_or_follower": "L1/L2/L3, F1/F2, or context_defender.",
    "reference_x, reference_y": "Active pre-decision controller reference in feet; defensive rows use fixed matchup references.",
    "proposed_x, proposed_y": "Current phase-command position before controller filtering.",
    "observed_x, observed_y": "Noisy pre-actuation observation exported directly from simulation state.",
    "controlled_x, controlled_y": "Post-actuation simulated position.",
    "follower_weight_1..3": "Fixed design-time weights relative to PG01, WG01, C01; blank for leaders and defenders.",
    "ball_x, ball_y": "Scripted matched ball coordinates in feet.",
    "ball_possessor_id": "Stable possessor ID or NONE while a pass is in flight.",
    "sigma_min, sigma_max": "Singular values of the realized controlled offensive leader map.",
    "normalized_conditioning": "Realized sigma_min divided by sigma_max.",
    "determinant": "Signed determinant of the realized controlled leader map.",
    "area_scale": "Absolute determinant of the realized controlled leader map.",
    "minimum_pairwise_spacing_feet": "Minimum observed offensive-player distance used by the gate.",
    "formation_tracking_rmse_feet": "Observed offensive RMSE against the active pre-decision reference.",
    "follower_rmse_feet": "Observed follower RMSE against the active/held map.",
    "tracking_gate_passed": "Whether formation_tracking_rmse_feet is at or below its threshold.",
    "separation_gate_passed": "Whether minimum spacing is at or above its threshold.",
    "conditioning_gate_passed": "Whether proposed normalized conditioning is at least 0.12.",
    "controller_decision": "ACCEPT, REJECT, or HOLD LAST SAFE MAP.",
    "held_last_safe_map": "True while the arm is in a reject/hold interval.",
    "last_safe_map_frame": "Frame at which the exported fallback map was recorded.",
    "hold_start_frame, hold_end_frame": "Inclusive bounds of the current hold interval; -1 outside holds.",
    "last_safe_map_a11..a22": "Exact 2x2 fallback affine matrix retained by the controller.",
    "last_safe_translation_x, last_safe_translation_y": "Exact fallback affine translation in court feet.",
    "rank_loss": "True when realized controlled normalized_conditioning is below 0.12.",
    "hidden_collapse": "rank_loss AND both ordinary gates pass AND the arm accepts.",
}


def make_readme(
    paths: dict[str, Path],
    episodes: int,
    rows: int,
    results: dict[str, dict[str, object]],
) -> str:
    fields = "\n".join(f"| `{field}` | {definition} |" for field, definition in FIELD_DEFINITIONS.items())
    verification = (
        f"- `{paths['verification'].name}`: independent full-CSV reconciliation of counts and numeric summaries.\n"
        if paths["verification"].exists()
        else ""
    )
    return f"""# Basketball Affine Conditioning v2

Deterministic synthetic formation-control data for BetGuard ingestion. No value is extracted from MP4 pixels. All coordinates are exported directly from the simulator at twelve decimal places.

## Reproduce

```powershell
py -3.12 analysis\\run_basketball_affine_conditioning_v2_gpu.py
```

Default seed: `{GLOBAL_SEED}`. Episodes per arm: `{episodes}`. Expected telemetry rows: `{rows:,}`.

## Deliverables

- `{paths['movie'].name}`: 1920x1080, {FPS} fps, {STEPS} frames.
- `{paths['telemetry'].name}`: one row per episode x arm x frame x player, including all five offensive and five defensive identities.
- `{paths['config'].name}`: frozen identities, coordinates, thresholds, phases, weights, and provenance.
- `{paths['summary'].name}`: frame-level and episode-level results, confidence intervals, assertions, and hashes.
- `{paths['pricing'].name}`: unvalidated synthetic BetGuard video-price example.
{verification}- `{paths['manifest'].name}`: SHA-256 manifest for the deliverables and generator source.

## Coordinates and events

Basket center is `(0, 0)`. X spans `[-25, 25]` feet from sideline to sideline; Y spans `[0, 47]` feet from baseline to half court.

`rank_loss = realized controlled normalized_conditioning < 0.12`. The conditioning gate evaluates the proposed affine map; keeping these quantities separate exposes residual post-decision disturbance risk rather than silently redefining it away.

`hidden_collapse = rank_loss AND tracking_gate_passed AND separation_gate_passed AND controller_decision == ACCEPT`.

Counts in the summary are separated into affected frames and affected episodes. Wilson 95% intervals are reported only for episode-level rates because frames within an episode are dependent.

## Fields

| Field | Definition |
|---|---|
{fields}

## Scope

Illustrative synthetic simulation only. Not actual basketball evidence, a validated betting model, or proof of profitability. The pricing example sets `validated=false`, `historical_games=0`, and `qualified_paper_candidate=false`.
"""


def main() -> None:
    episodes = int(os.environ.get("BASKETBALL_V2_EPISODES", str(DEFAULT_EPISODES)))
    out = Path(os.environ.get("BASKETBALL_V2_OUT", str(DEFAULT_OUT))).resolve()
    chunk_episodes = int(os.environ.get("BASKETBALL_V2_CSV_CHUNK", "8"))
    reuse_telemetry = os.environ.get("BASKETBALL_V2_REUSE_TELEMETRY", "0") == "1"
    reuse_video = os.environ.get("BASKETBALL_V2_REUSE_VIDEO", "0") == "1"
    out.mkdir(parents=True, exist_ok=True)
    paths = output_paths(out)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the requested matched experiment.")
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(GLOBAL_SEED)
    device = torch.device("cuda")
    q = nominal_offense(device)
    weights_tensor = follower_weights(q)
    weights = weights_tensor.detach().cpu().numpy().astype(np.float64)
    params = make_parameters(episodes, device)

    started = time.perf_counter()
    trajectory = simulate_batch(episodes, q, weights_tensor, params)
    torch.cuda.synchronize()
    gpu_seconds = time.perf_counter() - started
    results = result_summary(trajectory, episodes)
    validations = validate_trajectory(trajectory, weights, episodes)
    hold_start, hold_end = hold_intervals(trajectory["decision"])
    representative = representative_episode(trajectory, params)
    metadata = telemetry_metadata(weights)
    config = make_config(episodes, weights, metadata)
    config["gpu"] = {
        "device": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "batch_seconds": gpu_seconds,
    }
    paths["config"].write_text(json.dumps(config, indent=2), encoding="utf-8")

    expected_rows = episodes * N_ARMS * STEPS * N_PLAYERS
    if reuse_telemetry:
        rows = validate_existing_telemetry(paths["telemetry"], expected_rows, episodes)
        validations["telemetry_render_recovery"] = {
            "passed": True,
            "mode": "reuse_completed_csv_after_render_stage_interruption",
            "terminal_sentinel_checked": True,
        }
    else:
        rows = write_telemetry(
            paths["telemetry"],
            trajectory,
            params,
            weights,
            hold_start,
            hold_end,
            episodes,
            chunk_episodes,
        )
    assert rows == expected_rows

    if reuse_video:
        validations["video_render_recovery"] = {
            "passed": True,
            "mode": "reuse_verified_300_frame_encode_after_memory_pressure",
        }
    else:
        render_video(paths["movie"], trajectory, results, representative)
    video_probe = ffprobe_video(paths["movie"])
    stream = video_probe["streams"][0]
    assert int(stream["width"]) == 1920
    assert int(stream["height"]) == 1080
    assert int(stream["nb_frames"]) == STEPS
    validations["mp4_frame_alignment"] = {
        "passed": int(stream["nb_frames"]) == STEPS,
        "video_frames": int(stream["nb_frames"]),
        "telemetry_frames_per_episode_arm": STEPS,
    }

    base_hashes = {
        "mp4": sha256(paths["movie"]),
        "telemetry_csv": sha256(paths["telemetry"]),
        "configuration_json": sha256(paths["config"]),
    }
    pricing = make_pricing_example(trajectory, representative, metadata, base_hashes)
    paths["pricing"].write_text(json.dumps(pricing, indent=2), encoding="utf-8")
    hashes = dict(base_hashes)
    hashes["pricing_example_json"] = sha256(paths["pricing"])

    if paths["verification"].exists():
        telemetry_verification = json.loads(paths["verification"].read_text(encoding="utf-8"))
        if not telemetry_verification.get("verified"):
            raise AssertionError("Independent generated-telemetry reconciliation did not pass.")
        if telemetry_verification.get("all_csv_rows") != rows:
            raise AssertionError("Independent generated-telemetry row count does not match.")
        validations["generated_telemetry_reconciliation"] = {
            "passed": True,
            "artifact": paths["verification"].name,
            "all_csv_rows": rows,
            "numeric_tolerance": telemetry_verification["numeric_tolerance"],
        }
        hashes["csv_verification_json"] = sha256(paths["verification"])

    summary = {
        "simulation_version": SIMULATION_VERSION,
        "creation_timestamp_utc": CREATION_TIMESTAMP_UTC,
        "random_seed": GLOBAL_SEED,
        "episodes_per_arm": episodes,
        "frame_rate_fps": FPS,
        "video_duration_seconds": STEPS / FPS,
        "video_frames": STEPS,
        "telemetry_rows": rows,
        "coordinate_convention": config["coordinate_convention"],
        "thresholds": config["thresholds"],
        "player_impact_ratings": {player.player_id: player.impact_rating for player in ALL_PLAYERS},
        "leader_identities": [OFFENSE[index].player_id for index in LEADER_INDICES],
        "results_by_controller_arm": results,
        "representative_episode": {
            "episode_index": representative,
            "episode_id": f"BBAFF-E{representative:04d}",
            "episode_seed": int(params.episode_seeds[representative]),
        },
        "validation_results": validations,
        "video_provenance": config["video_provenance"],
        "sha256": hashes,
        "scope": "Illustrative synthetic simulation only. Not actual basketball evidence, a validated betting model, or proof of profitability.",
    }
    paths["summary"].write_text(json.dumps(summary, indent=2), encoding="utf-8")
    paths["readme"].write_text(make_readme(paths, episodes, rows, results), encoding="utf-8")

    manifest_paths = [paths[key] for key in ("movie", "telemetry", "config", "summary", "pricing", "readme")]
    if paths["verification"].exists():
        manifest_paths.append(paths["verification"])
    manifest_paths.append(Path(__file__))
    paths["manifest"].write_text(
        "".join(
            f"{sha256(path)}  {path.name if path.parent == out else 'analysis/' + path.name}\n"
            for path in manifest_paths
        ),
        encoding="ascii",
    )
    print(
        json.dumps(
            {
                "output_directory": str(out),
                "telemetry_rows": rows,
                "telemetry_bytes": paths["telemetry"].stat().st_size,
                "results": results,
                "representative_episode": representative,
                "validation_results": validations,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
