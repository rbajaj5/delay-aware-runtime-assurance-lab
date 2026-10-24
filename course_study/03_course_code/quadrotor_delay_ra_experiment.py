"""Matched A-E runtime-assurance experiment for delayed PPO quadrotor control.

This is the first full experimental harness beyond the Phase 0 smoke path. It
keeps the scientific scope intentionally narrow:

* safe-control-gym 2D quadrotor tracking
* pretrained PPO controller
* FIFO actuation delay after PPO selection and before env.step(action)
* no CBF/CLBF/MPC/ROS/new controller

Conditions:
    A: nominal PPO, zero delay
    B: delayed PPO, no monitor/intervention
    C: delayed PPO with warning-only delay-aware monitor, no intervention
    D: delayed PPO with delay-blind heuristic intervention
    E: delayed PPO with delay-aware candidate-set intervention using symbolic
       short-horizon prediction over the pending action queue.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
TOOLS_ROOT = REPO_ROOT / "tools"
for path in (SRC_ROOT, TOOLS_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from quadrotor_delay_ra.metrics import (  # noqa: E402
    cumulative_normalized_violation_severity,
    normalized_positive_violation,
    normalized_positive_violation_from_constraint_values,
    per_step_normalized_violation_severity_from_constraint_values,
    violating_step_count,
)
from quadrotor_ppo_trace import _jsonable, _lookup_reference  # noqa: E402
from quadrotor_delay_smoke import _build_controller_and_env_for_smoke  # noqa: E402
from hybrid_delay_aware_monitor import (  # noqa: E402
    HybridDelayAwareMonitor,
    hybrid_config_from_dict,
)


STATE_LOWER = np.array([-2.0, -1.0, 0.0, -1.0, -0.2, -1.5], dtype=float)
STATE_UPPER = np.array([2.0, 1.0, 2.0, 1.0, 0.2, 1.5], dtype=float)
INPUT_LOWER = np.array([0.06, 0.06], dtype=float)
INPUT_UPPER = np.array([0.29, 0.29], dtype=float)
STATE_SCALE = np.maximum.reduce([np.abs(STATE_LOWER), np.abs(STATE_UPPER), STATE_UPPER - STATE_LOWER, np.ones_like(STATE_LOWER)])
INPUT_SCALE = np.maximum.reduce([np.abs(INPUT_LOWER), np.abs(INPUT_UPPER), INPUT_UPPER - INPUT_LOWER, np.ones_like(INPUT_LOWER)])
CONSTRAINT_SCALES = np.concatenate([STATE_SCALE, STATE_SCALE, INPUT_SCALE, INPUT_SCALE])
DELAY_MODE_CHOICES = ["uniform_0_2", "fbm_uniform_0_2", "markov_symmetric"]
DEFAULT_PREDICTION_HORIZON = 3


CONDITIONS = {
    "A_nominal_ppo_zero_delay": {"delay": "zero", "monitor": "none", "intervention": "none"},
    "B_delayed_ppo": {"delay": "uniform_0_2", "monitor": "none", "intervention": "none"},
    "C_warning_only_delay_aware": {"delay": "uniform_0_2", "monitor": "delay_aware", "intervention": "none"},
    "D_delay_blind_intervention": {"delay": "uniform_0_2", "monitor": "delay_blind", "intervention": "delay_blind"},
    "E_delay_aware_intervention": {"delay": "uniform_0_2", "monitor": "delay_aware", "intervention": "delay_aware"},
    "E_plus_velocity_boundary_intervention": {
        "delay": "uniform_0_2",
        "monitor": "delay_aware",
        "intervention": "delay_aware_velocity_boundary",
        "held_action_corrected": True,
        "boundary_penalty_target": "xdot",
        "boundary_penalty_anti": False,
    },
    "E_held_corrected_intervention": {
        "delay": "uniform_0_2",
        "monitor": "delay_aware",
        "intervention": "delay_aware_held_corrected",
        "held_action_corrected": True,
        "boundary_penalty_target": "none",
        "boundary_penalty_anti": False,
    },
    "E_slew_rate_intervention": {
        "delay": "uniform_0_2",
        "monitor": "delay_aware",
        "intervention": "delay_aware_slew_rate",
        "held_action_corrected": True,
        "boundary_penalty_target": "none",
        "boundary_penalty_anti": False,
        "slew_rate_penalty": True,
    },
    "E_xdot_target_intervention": {
        "delay": "uniform_0_2",
        "monitor": "delay_aware",
        "intervention": "delay_aware_xdot_target",
        "held_action_corrected": True,
        "boundary_penalty_target": "xdot",
        "boundary_penalty_anti": False,
    },
    "E_theta_sham_intervention": {
        "delay": "uniform_0_2",
        "monitor": "delay_aware",
        "intervention": "delay_aware_theta_sham",
        "held_action_corrected": True,
        "boundary_penalty_target": "theta",
        "boundary_penalty_anti": False,
    },
    "E_xdot_antitarget_intervention": {
        "delay": "uniform_0_2",
        "monitor": "delay_aware",
        "intervention": "delay_aware_xdot_antitarget",
        "held_action_corrected": True,
        "boundary_penalty_target": "xdot",
        "boundary_penalty_anti": True,
    },
    "Hybrid_vx_penalty": {
        "delay": "uniform_0_2",
        "monitor": "hybrid_delay_aware",
        "intervention": "hybrid_delay_aware",
        "hybrid": {"ablation": "Hybrid_vx_penalty"},
    },
    "Hybrid_boundary": {
        "delay": "uniform_0_2",
        "monitor": "hybrid_delay_aware",
        "intervention": "hybrid_delay_aware",
        "hybrid": {"ablation": "Hybrid_boundary"},
    },
    "Full_Hybrid": {
        "delay": "uniform_0_2",
        "monitor": "hybrid_delay_aware",
        "intervention": "hybrid_delay_aware",
        "hybrid": {"ablation": "Full_Hybrid"},
    },
    "Aggressive_Blind": {
        "delay": "uniform_0_2",
        "monitor": "hybrid_delay_aware",
        "intervention": "hybrid_delay_aware",
        "hybrid": {"ablation": "Aggressive_Blind"},
    },
}


@dataclass(frozen=True)
class ScheduledDelayStep:
    commanded_action: np.ndarray
    applied_action: np.ndarray
    delay_sample: int
    queue_state: list[dict[str, Any]]
    applied_entry: dict[str, Any] | None
    released_entry: dict[str, Any] | None
    held_last_action_used: bool
    applied_action_age: int | None


class ScheduledDelayChannel:
    """FIFO delay channel driven by an externally pre-sampled schedule."""

    def __init__(self, hold_action: np.ndarray | None = None) -> None:
        self._configured_hold_action = None if hold_action is None else np.asarray(hold_action, dtype=float).copy()
        self.reset()

    def reset(self) -> None:
        from collections import deque

        self._queue = deque()
        self._step = 0
        self._last_applied: np.ndarray | None = None
        self._last_applied_entry: dict[str, Any] | None = None

    def update(self, commanded_action: np.ndarray, delay_sample: int) -> ScheduledDelayStep:
        command = np.asarray(commanded_action, dtype=float).copy()
        if self._last_applied is None:
            self._last_applied = self._initial_hold(command)

        entry = {
            "command_step": self._step,
            "release_step": self._step + int(delay_sample),
            "delay_sample": int(delay_sample),
            "action": command,
        }
        self._queue.append(entry)

        released_entry = None
        if self._queue and int(self._queue[0]["release_step"]) <= self._step:
            released = self._queue.popleft()
            self._last_applied = np.asarray(released["action"], dtype=float).copy()
            self._last_applied_entry = {
                "command_step": int(released["command_step"]),
                "release_step": int(released["release_step"]),
                "delay_sample": int(released["delay_sample"]),
                "action": np.asarray(released["action"], dtype=float).tolist(),
            }
            released_entry = dict(self._last_applied_entry)

        held_last_action_used = released_entry is None
        applied_action_age = None
        if self._last_applied_entry is not None:
            applied_action_age = int(self._step - int(self._last_applied_entry["release_step"]))

        out = ScheduledDelayStep(
            commanded_action=command,
            applied_action=np.asarray(self._last_applied, dtype=float).copy(),
            delay_sample=int(delay_sample),
            queue_state=self.queue_state(),
            applied_entry=None if self._last_applied_entry is None else dict(self._last_applied_entry),
            released_entry=released_entry,
            held_last_action_used=held_last_action_used,
            applied_action_age=applied_action_age,
        )
        self._step += 1
        return out

    def queue_state(self) -> list[dict[str, Any]]:
        return [
            {
                "command_step": int(item["command_step"]),
                "release_step": int(item["release_step"]),
                "delay_sample": int(item["delay_sample"]),
                "action": np.asarray(item["action"], dtype=float).tolist(),
            }
            for item in self._queue
        ]

    def last_applied_action(self, fallback_shape: tuple[int, ...] | None = None) -> np.ndarray:
        if self._last_applied is not None:
            return np.asarray(self._last_applied, dtype=float).copy()
        if self._configured_hold_action is not None:
            return np.asarray(self._configured_hold_action, dtype=float).copy()
        if fallback_shape is None:
            return np.array([], dtype=float)
        return np.zeros(fallback_shape, dtype=float)

    def _initial_hold(self, command: np.ndarray) -> np.ndarray:
        if self._configured_hold_action is not None:
            hold = self._configured_hold_action
            if hold.shape != command.shape:
                raise ValueError(f"hold_action shape {hold.shape} does not match command shape {command.shape}")
            return hold.copy()
        return np.zeros_like(command, dtype=float)


def _fgn_latent(seed: int, steps: int, hurst: float, jitter: float = 1e-8) -> np.ndarray:
    """Generate fractional Gaussian noise by Cholesky factorization.

    This is used only for Phase 4 stress schedules. The reported Phase 1-3
    defaults remain the bounded iid uniform schedule.
    """

    if not (0.0 < hurst < 1.0):
        raise ValueError(f"hurst must be in (0, 1), got {hurst}")
    rng = np.random.default_rng(seed)
    idx = np.arange(steps)
    k = np.abs(idx[:, None] - idx[None, :])
    cov = 0.5 * ((k + 1) ** (2.0 * hurst) + np.abs(k - 1) ** (2.0 * hurst) - 2.0 * (k ** (2.0 * hurst)))
    cov += np.eye(steps) * jitter
    try:
        factor = np.linalg.cholesky(cov)
    except np.linalg.LinAlgError:
        factor = np.linalg.cholesky(cov + np.eye(steps) * jitter * 100.0)
    return factor @ rng.standard_normal(steps)


def _quantile_map_to_values(latent: np.ndarray, values: tuple[int, ...] = (0, 1, 2), probs: tuple[float, ...] = (1 / 3, 1 / 3, 1 / 3)) -> list[int]:
    """Map a latent correlated trace to bounded integer delays with fixed marginals."""

    values_arr = np.asarray(values, dtype=int)
    probs_arr = np.asarray(probs, dtype=float)
    if values_arr.ndim != 1 or probs_arr.ndim != 1 or values_arr.size != probs_arr.size:
        raise ValueError("values and probs must be same-length one-dimensional arrays")
    probs_arr = probs_arr / probs_arr.sum()
    order = np.argsort(np.asarray(latent, dtype=float), kind="mergesort")
    ranks = np.empty(order.size, dtype=float)
    ranks[order] = (np.arange(order.size, dtype=float) + 0.5) / float(order.size)
    bins = np.cumsum(probs_arr)
    mapped = values_arr[np.searchsorted(bins, ranks, side="right")]
    return [int(x) for x in mapped]


def delay_schedule_stats(schedule: list[int]) -> dict[str, Any]:
    arr = np.asarray(schedule, dtype=float)
    if arr.size == 0:
        return {}
    high = arr >= 2.0
    longest = 0
    current = 0
    for is_high in high:
        if is_high:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    acf1 = float("nan")
    if arr.size > 1 and np.std(arr[:-1]) > 1e-12 and np.std(arr[1:]) > 1e-12:
        acf1 = float(np.corrcoef(arr[:-1], arr[1:])[0, 1])
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0,
        "min": int(np.min(arr)),
        "max": int(np.max(arr)),
        "fraction_zero": float(np.mean(arr == 0.0)),
        "fraction_high_ge2": float(np.mean(high)),
        "longest_high_run_ge2": int(longest),
        "lag1_autocorrelation": acf1,
    }


def make_markov_symmetric_delay_schedule(seed: int, steps: int, q: float) -> list[int]:
    """Generate a stationary Markov chain directly on delay values {0, 1, 2}."""

    if not 0.0 <= q <= 1.0:
        raise ValueError(f"markov_symmetric persistence q must be in [0, 1], got {q}")
    rng = np.random.default_rng(seed)
    if steps <= 0:
        return []
    schedule = np.empty(steps, dtype=int)
    schedule[0] = int(rng.integers(0, 3))
    for idx in range(1, steps):
        previous = int(schedule[idx - 1])
        if float(rng.random()) < q:
            schedule[idx] = previous
        else:
            others = [value for value in (0, 1, 2) if value != previous]
            schedule[idx] = int(others[int(rng.integers(0, 2))])
    return [int(x) for x in schedule]


def make_delay_schedule(seed: int, steps: int, mode: str, fbm_hurst: float = 0.75, markov_q: float = 0.70) -> list[int]:
    if mode == "zero":
        return [0] * steps
    if mode == "fixed_one":
        return [1] * steps
    if mode == "uniform_0_2":
        rng = np.random.default_rng(seed)
        return [int(x) for x in rng.integers(0, 3, size=steps)]
    if mode == "fbm_uniform_0_2":
        # fBM-derived stress schedule: keep the same bounded {0,1,2}
        # marginal support while injecting persistence through the latent path.
        fgn = _fgn_latent(seed=seed, steps=steps, hurst=fbm_hurst)
        fbm_path = np.cumsum(fgn)
        return _quantile_map_to_values(fbm_path)
    if mode == "markov_symmetric":
        return make_markov_symmetric_delay_schedule(seed=seed, steps=steps, q=markov_q)
    raise ValueError(f"Unknown delay mode {mode}")


def schedule_hash(schedule: list[int]) -> str:
    raw = ",".join(str(x) for x in schedule).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


def flatten_constraint_values(values: Any) -> np.ndarray:
    if values is None:
        return np.array([], dtype=float)
    parts = [np.asarray(part, dtype=float).reshape(-1) for part in np.atleast_1d(values)]
    return np.concatenate(parts) if parts else np.array([], dtype=float)


def constraint_severity(values: np.ndarray) -> float:
    if values.size == CONSTRAINT_SCALES.size:
        return per_step_normalized_violation_severity_from_constraint_values(values, CONSTRAINT_SCALES)
    return float(np.sum(np.maximum(values, 0.0)))


def max_state_constraint_value(values: np.ndarray) -> float:
    if values.size >= 12:
        return float(np.max(values[:12]))
    if values.size:
        return float(np.max(values))
    return float("-inf")


def state_violation_severity(state: np.ndarray) -> float:
    return float(np.sum(normalized_positive_violation(state, STATE_LOWER, STATE_UPPER)))


def input_violation_severity(physical_action: np.ndarray) -> float:
    return float(np.sum(normalized_positive_violation(physical_action, INPUT_LOWER, INPUT_UPPER)))


def denormalized_clipped_action(env: Any, normalized_action: np.ndarray) -> np.ndarray:
    physical = np.asarray(env.denormalize_action(np.asarray(normalized_action, dtype=float)), dtype=float).reshape(-1)
    low, high = env.physical_action_bounds
    return np.clip(physical, np.asarray(low, dtype=float), np.asarray(high, dtype=float))


def normalized_from_physical_action(env: Any, physical_action: np.ndarray) -> np.ndarray:
    """Invert safe-control-gym's linear action normalization."""

    low, high = env.physical_action_bounds
    low_arr = np.asarray(low, dtype=float).reshape(-1)
    high_arr = np.asarray(high, dtype=float).reshape(-1)
    physical = np.clip(np.asarray(physical_action, dtype=float).reshape(-1), low_arr, high_arr)
    span = np.maximum(high_arr - low_arr, 1e-12)
    normalized = 2.0 * (physical - low_arr) / span - 1.0
    return np.clip(normalized, -1.0, 1.0)


def _uses_normalized_action_space(env: Any) -> bool:
    """Return the environment's action-coordinate convention."""

    if hasattr(env, "NORMALIZED_RL_ACTION_SPACE"):
        return bool(env.NORMALIZED_RL_ACTION_SPACE)
    return bool(getattr(env, "normalized_rl_action_space", True))


def apply_action_conversion(
    controller_output: np.ndarray,
    effectiveness: float,
    env: Any,
    controller_outputs_normalized: bool = True,
    use_corrected_action_conversion: bool = True,
    u_min: np.ndarray | None = None,
    u_max: np.ndarray | None = None,
    audit_log: list[dict[str, Any]] | None = None,
) -> np.ndarray:
    """Apply actuator effectiveness in physical thrust coordinates.

    The 2D PPO and monitor arms produce hover-centered normalized actions, so
    their commands are first converted with ``env.denormalize_action``. A
    physical controller such as LQR can set ``controller_outputs_normalized``
    to ``False`` and provide thrust directly in Newtons. In both cases the
    intended plant model is

        u_eff = clip(effectiveness * u_physical, u_min, u_max).

    Corrected runs map ``u_eff`` back with ``env.normalize_action``, the inverse
    of safe-control-gym's hover-centered de-normalization. The legacy branch is
    retained solely to reproduce archived runs: it clips the pre-decay command
    and maps actuator bounds linearly to [-1, 1]. That bounds-based map is not
    the inverse of hover-centered de-normalization.
    """

    if not math.isfinite(float(effectiveness)) or float(effectiveness) < 0.0:
        raise ValueError(f"effectiveness must be finite and nonnegative, got {effectiveness}")
    if u_min is None or u_max is None:
        if not hasattr(env, "physical_action_bounds"):
            raise AttributeError("Environment does not expose physical_action_bounds")
        env_low, env_high = env.physical_action_bounds
        if u_min is None:
            u_min = np.asarray(env_low, dtype=float)
        if u_max is None:
            u_max = np.asarray(env_high, dtype=float)
    low = np.asarray(u_min, dtype=float).reshape(-1)
    high = np.asarray(u_max, dtype=float).reshape(-1)
    output = np.asarray(controller_output, dtype=float).reshape(-1)
    if output.size != low.size or low.shape != high.shape:
        raise ValueError(
            f"Action/bound shape mismatch: output={output.shape}, low={low.shape}, high={high.shape}"
        )

    raw_physical = (
        np.asarray(env.denormalize_action(output), dtype=float).reshape(-1)
        if controller_outputs_normalized
        else output.copy()
    )
    if use_corrected_action_conversion:
        physical_for_multiplier = raw_physical
    else:
        # Exact archived behavior: the old helper clipped before multiplying.
        physical_for_multiplier = np.clip(raw_physical, low, high)
    decayed = np.clip(float(effectiveness) * physical_for_multiplier, low, high)

    normalized_space = _uses_normalized_action_space(env)
    if not normalized_space:
        action_to_env = decayed.copy()
    elif use_corrected_action_conversion:
        action_to_env = np.asarray(env.normalize_action(decayed), dtype=float).reshape(-1)
    else:
        span = np.maximum(high - low, 1e-12)
        action_to_env = np.clip(2.0 * (decayed - low) / span - 1.0, -1.0, 1.0)

    recovered_physical = (
        np.asarray(env.denormalize_action(action_to_env), dtype=float).reshape(-1)
        if normalized_space
        else action_to_env.copy()
    )
    if audit_log is not None:
        audit_log.append(
            {
                "effectiveness": float(effectiveness),
                "controller_outputs_normalized": bool(controller_outputs_normalized),
                "use_corrected_action_conversion": bool(use_corrected_action_conversion),
                "u_physical_raw": raw_physical.copy(),
                "u_physical_for_multiplier": physical_for_multiplier.copy(),
                "u_decayed": decayed.copy(),
                "action_to_env": action_to_env.copy(),
                "recovered_physical": recovered_physical.copy(),
            }
        )
    return action_to_env


def verify_action_conversion_audit(
    audit_log: list[dict[str, Any]],
    u_min: np.ndarray,
    u_max: np.ndarray,
    tol: float = 1e-12,
) -> dict[str, float | int]:
    """Verify physical multiplication and final action-coordinate round trip."""

    low = np.asarray(u_min, dtype=float).reshape(-1)
    high = np.asarray(u_max, dtype=float).reshape(-1)
    multiplier_mismatches = 0
    roundtrip_mismatches = 0
    checked = 0
    max_multiplier_error = 0.0
    max_roundtrip_error = 0.0
    for record in audit_log:
        eta = float(record["effectiveness"])
        if math.isclose(eta, 1.0, rel_tol=0.0, abs_tol=tol):
            continue
        checked += 1
        source = np.asarray(record["u_physical_for_multiplier"], dtype=float)
        expected = np.clip(eta * source, low, high)
        decayed = np.asarray(record["u_decayed"], dtype=float)
        recovered = np.asarray(record["recovered_physical"], dtype=float)
        multiplier_error = float(np.max(np.abs(decayed - expected)))
        roundtrip_error = float(np.max(np.abs(recovered - expected)))
        max_multiplier_error = max(max_multiplier_error, multiplier_error)
        max_roundtrip_error = max(max_roundtrip_error, roundtrip_error)
        multiplier_mismatches += int(multiplier_error > tol)
        roundtrip_mismatches += int(roundtrip_error > tol)
    return {
        "total_checked": checked,
        "multiplier_mismatches": multiplier_mismatches,
        "roundtrip_mismatches": roundtrip_mismatches,
        "max_multiplier_error": max_multiplier_error,
        "max_roundtrip_error": max_roundtrip_error,
    }


def verify_audit(
    audit_log: list[dict[str, Any]],
    u_min: np.ndarray,
    u_max: np.ndarray,
    tol: float = 1e-12,
) -> tuple[int, int, float]:
    """Compatibility wrapper returning final physical round-trip mismatches."""

    result = verify_action_conversion_audit(audit_log, u_min, u_max, tol=tol)
    return (
        int(result["roundtrip_mismatches"]),
        int(result["total_checked"]),
        float(result["max_roundtrip_error"]),
    )


def apply_thrust_effectiveness_decay(
    env: Any,
    normalized_action: np.ndarray,
    effectiveness: float,
    use_corrected_action_conversion: bool = True,
    audit_log: list[dict[str, Any]] | None = None,
) -> np.ndarray:
    """Apply thrust effectiveness before the FIFO; corrected by default."""

    return apply_action_conversion(
        controller_output=normalized_action,
        effectiveness=effectiveness,
        env=env,
        controller_outputs_normalized=True,
        use_corrected_action_conversion=use_corrected_action_conversion,
        audit_log=audit_log,
    )


def thrust_effectiveness_at_step(step: int, steps: int, final_effectiveness: float) -> float:
    if steps <= 1:
        return float(final_effectiveness)
    alpha = float(step) / float(max(steps - 1, 1))
    return float(1.0 + alpha * (float(final_effectiveness) - 1.0))


def predict_next_state(env: Any, state: np.ndarray, normalized_action: np.ndarray) -> np.ndarray:
    physical = denormalized_clipped_action(env, normalized_action)
    result = env.symbolic.fd_func(x0=np.asarray(state, dtype=float), p=physical)
    return np.asarray(result["xf"], dtype=float).reshape(-1)


def candidate_actions(ppo_action: np.ndarray, state: np.ndarray) -> list[tuple[str, np.ndarray]]:
    ppo_clipped = np.clip(np.asarray(ppo_action, dtype=float), -1.0, 1.0)
    theta = float(state[4]) if state.size > 4 else 0.0
    theta_dot = float(state[5]) if state.size > 5 else 0.0
    correction = float(np.clip(2.0 * theta + 0.25 * theta_dot, -0.8, 0.8))
    candidates = [
        ("ppo_clipped", ppo_clipped),
        ("hover", np.array([0.0, 0.0], dtype=float)),
        ("gentle_up", np.array([0.35, 0.35], dtype=float)),
        ("gentle_down", np.array([-0.35, -0.35], dtype=float)),
        ("diff_left", np.array([0.55, -0.55], dtype=float)),
        ("diff_right", np.array([-0.55, 0.55], dtype=float)),
        ("theta_correct_a", np.array([-correction, correction], dtype=float)),
        ("theta_correct_b", np.array([correction, -correction], dtype=float)),
    ]
    # Preserve order while removing duplicate numeric actions.
    unique: list[tuple[str, np.ndarray]] = []
    seen: set[tuple[float, float]] = set()
    for name, action in candidates:
        key = tuple(np.round(np.asarray(action, dtype=float), 6))
        if key not in seen:
            unique.append((name, np.asarray(action, dtype=float)))
            seen.add(key)
    return unique


def queue_actions_for_prediction(
    queue_state: list[dict[str, Any]],
    current_step: int,
    candidate: np.ndarray,
    delay_sample: int,
    last_applied_action: np.ndarray | None = None,
    prediction_horizon: int = DEFAULT_PREDICTION_HORIZON,
) -> list[np.ndarray]:
    if int(prediction_horizon) < 1:
        raise ValueError("prediction_horizon must be at least 1")
    entries = [
        (int(item["command_step"]), int(item["release_step"]), np.asarray(item["action"], dtype=float))
        for item in queue_state
    ]
    entries.append((current_step, current_step + int(delay_sample), np.asarray(candidate, dtype=float)))
    entries.sort(key=lambda item: (item[0], item[1]))
    actions: list[np.ndarray] = []
    step = current_step
    queue = entries[:]
    if last_applied_action is None:
        last = np.zeros_like(candidate, dtype=float)
    else:
        last = np.asarray(last_applied_action, dtype=float).reshape(np.asarray(candidate, dtype=float).shape).copy()
    for _ in range(int(prediction_horizon)):
        if queue and int(queue[0][1]) <= step:
            _, _, last = queue.pop(0)
        actions.append(np.asarray(last, dtype=float).copy())
        step += 1
    return actions


BOUNDARY_PENALTY_INDEX = {
    "xdot": 1,
    "theta": 4,
    "thetadot": 5,
}


def boundary_candidate_penalty(
    state: np.ndarray,
    previous_state: np.ndarray,
    predicted_state: np.ndarray,
    weight: float,
    target: str = "xdot",
    anti_target: bool = False,
) -> float:
    """Penalize target-axis motion toward or away from the relevant bound."""

    target = str(target or "none").lower()
    if target in {"", "none"} or weight <= 0.0:
        return 0.0
    index = BOUNDARY_PENALTY_INDEX.get(target)
    if index is None:
        raise ValueError(f"Unknown boundary penalty target {target!r}. Valid targets: {sorted(BOUNDARY_PENALTY_INDEX)}")
    if state.size <= index or predicted_state.size <= index or previous_state.size <= index:
        return 0.0
    max_abs = max(abs(float(STATE_LOWER[index])), abs(float(STATE_UPPER[index])), 1e-12)
    safe_threshold = 0.70 * max_abs
    current = float(state[index])
    previous = float(previous_state[index])
    predicted = float(predicted_state[index])
    boundary_proximity = max(0.0, (abs(current) - safe_threshold) / max(max_abs - safe_threshold, 1e-12))
    boundary_proximity = min(1.0, boundary_proximity)
    if abs(current) <= 1e-12:
        return 0.0
    predicted_change = predicted - previous
    outward_change = math.copysign(1.0, current) * predicted_change
    penalized_change = max(0.0, -outward_change if anti_target else outward_change)
    return float(weight) * boundary_proximity * penalized_change


def xdot_boundary_proximity(state: np.ndarray, threshold_fraction: float = 0.70) -> float:
    state_arr = np.asarray(state, dtype=float).reshape(-1)
    index = BOUNDARY_PENALTY_INDEX["xdot"]
    if state_arr.size <= index:
        return 0.0
    max_abs = max(abs(float(STATE_LOWER[index])), abs(float(STATE_UPPER[index])), 1e-12)
    safe_threshold = float(threshold_fraction) * max_abs
    current = abs(float(state_arr[index]))
    proximity = max(0.0, (current - safe_threshold) / max(max_abs - safe_threshold, 1e-12))
    return float(min(1.0, proximity))


def slew_rate_candidate_penalty(
    state: np.ndarray,
    queue_state: list[dict[str, Any]],
    candidate: np.ndarray,
    last_applied_action: np.ndarray | None,
    weight: float,
) -> tuple[float, float, float, float]:
    """Penalize large candidate jumps when queued actions or xdot risk make timing brittle."""

    if weight <= 0.0:
        return 0.0, 0.0, 0.0, 0.0
    candidate_arr = np.asarray(candidate, dtype=float).reshape(-1)
    if last_applied_action is None:
        last_arr = np.zeros_like(candidate_arr, dtype=float)
    else:
        last_arr = np.asarray(last_applied_action, dtype=float).reshape(candidate_arr.shape)
    action_delta_norm = float(np.linalg.norm(candidate_arr - last_arr))
    queue_factor = 1.0 if len(queue_state) > 0 else 0.0
    proximity = xdot_boundary_proximity(state)
    conditioning = max(queue_factor, proximity)
    penalty = float(weight) * conditioning * action_delta_norm
    return penalty, action_delta_norm, queue_factor, proximity


def velocity_gated_pitch_penalty(
    predicted_terminal_state: np.ndarray,
    enabled: bool,
    weight: float,
    minimum_forward_velocity: float,
) -> tuple[float, float, bool, float, float]:
    """Return a terminal kinetic-energy-derivative proxy and its cost.

    For the 2D plant's sign convention, horizontal kinetic energy
    ``V_v = 0.5 * v_x**2`` has ``V_dot_v = v_x * a_x`` and the thrust-vector
    contribution to ``a_x`` is proportional to ``sin(theta)``.  Thus,
    positive forward velocity with non-negative pitch is the symmetric-action
    trap seen in E's IID-delay failures: hover/common-thrust candidates do not
    rotate the thrust vector into braking pitch.

    The activation gate keeps nominal eta=1 behavior unchanged at low speed.
    It is intentionally based on the candidate's predicted terminal velocity,
    so a candidate that reduces v_x below the threshold is not penalized.
    """

    terminal = np.asarray(predicted_terminal_state, dtype=float).reshape(-1)
    terminal_vx = float(terminal[1]) if terminal.size > 1 else float("nan")
    terminal_theta = float(terminal[4]) if terminal.size > 4 else float("nan")
    valid = math.isfinite(terminal_vx) and math.isfinite(terminal_theta)
    proxy = (
        max(0.0, terminal_vx) * max(0.0, math.sin(terminal_theta))
        if valid
        else 0.0
    )
    active = bool(
        enabled
        and valid
        and float(weight) > 0.0
        and terminal_vx >= float(minimum_forward_velocity)
        and proxy > 0.0
    )
    penalty = float(weight) * proxy if active else 0.0
    return float(penalty), float(proxy), active, terminal_vx, terminal_theta


def state_boundary_pressure(state: np.ndarray) -> float:
    """Smooth proxy for proximity to the externally specified safety box."""

    state_arr = np.asarray(state, dtype=float).reshape(-1)
    n = min(state_arr.size, STATE_LOWER.size)
    if n == 0:
        return 0.0
    lower = STATE_LOWER[:n]
    upper = STATE_UPPER[:n]
    center = 0.5 * (lower + upper)
    half_width = np.maximum(0.5 * (upper - lower), 1e-12)
    normalized_distance = np.abs((state_arr[:n] - center) / half_width)
    pressure = float(np.max(normalized_distance))
    if n > 1:
        x = float(state_arr[0])
        x_dot = float(state_arr[1])
        outward_speed = max(0.0, x_dot if x >= 0.0 else -x_dot)
        pressure += 0.25 * outward_speed / max(abs(float(STATE_UPPER[1])), 1e-12)
    return pressure


def choose_adversarial_saturation_command(
    env: Any,
    state: np.ndarray,
    command: np.ndarray,
    queue_state: list[dict[str, Any]],
    current_step: int,
    delay_sample: int,
    effort: float,
) -> tuple[np.ndarray, np.ndarray, str, float]:
    """Choose a bounded normalized-action disturbance that worsens predicted safety pressure."""

    base = np.clip(np.asarray(command, dtype=float).reshape(-1), -1.0, 1.0)
    effort = float(max(0.0, effort))
    if effort <= 1e-12:
        return base, np.zeros_like(base), "none", 0.0

    raw_directions = []
    for i in range(base.size):
        unit = np.zeros_like(base)
        unit[i] = 1.0
        raw_directions.append((f"+u{i}", unit.copy()))
        raw_directions.append((f"-u{i}", -unit.copy()))
    if base.size == 2:
        raw_directions.extend(
            [
                ("+diff", np.array([1.0, -1.0], dtype=float)),
                ("-diff", np.array([-1.0, 1.0], dtype=float)),
                ("+common", np.array([1.0, 1.0], dtype=float)),
                ("-common", np.array([-1.0, -1.0], dtype=float)),
            ]
        )

    best_score = float("-inf")
    best_name = "none"
    best_action = base.copy()
    for name, direction in raw_directions:
        direction = np.asarray(direction, dtype=float).reshape(-1)
        direction = np.clip(direction, -1.0, 1.0)
        disturbed = np.clip(base + effort * direction, -1.0, 1.0)
        pred_state = np.asarray(state, dtype=float).copy()
        worst_severity = 0.0
        worst_pressure = state_boundary_pressure(pred_state)
        for action in queue_actions_for_prediction(queue_state, current_step, disturbed, delay_sample):
            pred_state = predict_next_state(env, pred_state, action)
            worst_severity = max(worst_severity, state_violation_severity(pred_state))
            worst_pressure = max(worst_pressure, state_boundary_pressure(pred_state))
        saturation_pressure = float(np.max(np.abs(disturbed)))
        score = 100.0 * worst_severity + worst_pressure + 0.05 * saturation_pressure
        if score > best_score:
            best_score = score
            best_name = name
            best_action = disturbed

    delta = best_action - base
    return best_action, delta, best_name, float(best_score)


def score_candidate_delay_aware(
    env: Any,
    state: np.ndarray,
    queue_state: list[dict[str, Any]],
    current_step: int,
    candidate: np.ndarray,
    delay_sample: int,
    reference: np.ndarray,
    last_applied_action: np.ndarray | None = None,
    velocity_boundary_weight: float = 0.0,
    boundary_penalty_target: str = "xdot",
    boundary_penalty_anti: bool = False,
    slew_rate_weight: float = 0.0,
    use_velocity_gated_penalty: bool = False,
    lambda_penalty: float = 100.0,
    velocity_gated_min_vx: float = 0.80,
    prediction_horizon: int = DEFAULT_PREDICTION_HORIZON,
) -> tuple[float, dict[str, float]]:
    pred_state = np.asarray(state, dtype=float).copy()
    worst_state_severity = 0.0
    final_tracking = 0.0
    velocity_boundary_penalty = 0.0
    actions = queue_actions_for_prediction(
        queue_state,
        current_step,
        candidate,
        delay_sample,
        last_applied_action=last_applied_action,
        prediction_horizon=prediction_horizon,
    )
    for action in actions:
        prev_state = pred_state.copy()
        pred_state = predict_next_state(env, pred_state, action)
        worst_state_severity = max(worst_state_severity, state_violation_severity(pred_state))
        velocity_boundary_penalty += boundary_candidate_penalty(
            state=state,
            previous_state=prev_state,
            predicted_state=pred_state,
            weight=velocity_boundary_weight,
            target=boundary_penalty_target,
            anti_target=boundary_penalty_anti,
        )
    slew_penalty, slew_action_delta_norm, slew_queue_factor, slew_xdot_proximity = slew_rate_candidate_penalty(
        state=state,
        queue_state=queue_state,
        candidate=candidate,
        last_applied_action=last_applied_action,
        weight=slew_rate_weight,
    )
    if reference.size:
        n = min(reference.size, pred_state.size)
        final_tracking = float(np.sqrt(np.mean(np.square(pred_state[:n] - reference[:n]))))
    phys = denormalized_clipped_action(env, candidate)
    input_sev = input_violation_severity(phys)
    action_norm = float(np.linalg.norm(candidate))
    score_without_boundary = (
        100.0 * worst_state_severity
        + 5.0 * input_sev
        + 0.05 * final_tracking
        + 0.005 * action_norm
    )
    score_without_slew = score_without_boundary + velocity_boundary_penalty
    score_without_velocity_gated_penalty = score_without_slew + slew_penalty
    (
        velocity_gated_penalty,
        velocity_gated_proxy,
        velocity_gated_active,
        predicted_terminal_vx,
        predicted_terminal_theta,
    ) = velocity_gated_pitch_penalty(
        predicted_terminal_state=pred_state,
        enabled=use_velocity_gated_penalty,
        weight=lambda_penalty,
        minimum_forward_velocity=velocity_gated_min_vx,
    )
    score = score_without_velocity_gated_penalty + velocity_gated_penalty
    return score, {
        "candidate_score_without_boundary": score_without_boundary,
        "candidate_score_without_slew": score_without_slew,
        "predicted_state_severity": worst_state_severity,
        "predicted_tracking_rmse": final_tracking,
        "candidate_input_severity": input_sev,
        "candidate_action_norm": action_norm,
        "candidate_velocity_boundary_penalty": velocity_boundary_penalty,
        "candidate_boundary_penalty": velocity_boundary_penalty,
        "candidate_slew_rate_penalty": slew_penalty,
        "candidate_slew_action_delta_norm": slew_action_delta_norm,
        "candidate_slew_queue_factor": slew_queue_factor,
        "candidate_slew_xdot_proximity": slew_xdot_proximity,
        "candidate_score_without_velocity_gated_penalty": score_without_velocity_gated_penalty,
        "candidate_velocity_gated_penalty": velocity_gated_penalty,
        "candidate_velocity_gated_energy_derivative_proxy": velocity_gated_proxy,
        "candidate_velocity_gated_penalty_active": float(velocity_gated_active),
        "candidate_velocity_gated_terminal_vx": predicted_terminal_vx,
        "candidate_velocity_gated_terminal_theta": predicted_terminal_theta,
    }


def choose_delay_aware_action(
    env: Any,
    state: np.ndarray,
    ppo_action: np.ndarray,
    queue_state: list[dict[str, Any]],
    current_step: int,
    delay_sample: int,
    reference: np.ndarray,
    last_applied_action: np.ndarray | None = None,
    velocity_boundary_weight: float = 0.0,
    boundary_penalty_target: str = "xdot",
    boundary_penalty_anti: bool = False,
    slew_rate_weight: float = 0.0,
    use_velocity_gated_penalty: bool = False,
    lambda_penalty: float = 100.0,
    velocity_gated_min_vx: float = 0.80,
    prediction_horizon: int = DEFAULT_PREDICTION_HORIZON,
) -> tuple[np.ndarray, str, float, list[dict[str, Any]]]:
    scored = []
    for name, action in candidate_actions(ppo_action, state):
        score, details = score_candidate_delay_aware(
            env,
            state,
            queue_state,
            current_step,
            action,
            delay_sample,
            reference,
            last_applied_action=last_applied_action,
            velocity_boundary_weight=velocity_boundary_weight,
            boundary_penalty_target=boundary_penalty_target,
            boundary_penalty_anti=boundary_penalty_anti,
            slew_rate_weight=slew_rate_weight,
            use_velocity_gated_penalty=use_velocity_gated_penalty,
            lambda_penalty=lambda_penalty,
            velocity_gated_min_vx=velocity_gated_min_vx,
            prediction_horizon=prediction_horizon,
        )
        scored.append({"name": name, "action": action, "score": score, **details})
    scored.sort(key=lambda item: item["score"])
    best = scored[0]
    json_scores = [
        {
            "name": item["name"],
            "score": float(item["score"]),
            "score_without_boundary": float(item.get("candidate_score_without_boundary", item["score"])),
            "score_without_slew": float(item.get("candidate_score_without_slew", item["score"])),
            "action": item["action"].tolist(),
            "predicted_state_severity": float(item["predicted_state_severity"]),
            "predicted_tracking_rmse": float(item["predicted_tracking_rmse"]),
            # Unweighted score components retained for preregistered score audits.
            "constraint_violation": float(item["predicted_state_severity"]),
            "action_violation": float(item["candidate_input_severity"]),
            "rmse": float(item["predicted_tracking_rmse"]),
            "l2_penalty": float(item["candidate_action_norm"]),
            "final_score": float(item["score"]),
            "candidate_velocity_boundary_penalty": float(item.get("candidate_velocity_boundary_penalty", 0.0)),
            "candidate_boundary_penalty": float(item.get("candidate_boundary_penalty", 0.0)),
            "candidate_slew_rate_penalty": float(item.get("candidate_slew_rate_penalty", 0.0)),
            "candidate_slew_action_delta_norm": float(item.get("candidate_slew_action_delta_norm", 0.0)),
            "candidate_slew_queue_factor": float(item.get("candidate_slew_queue_factor", 0.0)),
            "candidate_slew_xdot_proximity": float(item.get("candidate_slew_xdot_proximity", 0.0)),
            "score_without_velocity_gated_penalty": float(
                item.get("candidate_score_without_velocity_gated_penalty", item["score"])
            ),
            "candidate_velocity_gated_penalty": float(
                item.get("candidate_velocity_gated_penalty", 0.0)
            ),
            "candidate_velocity_gated_energy_derivative_proxy": float(
                item.get("candidate_velocity_gated_energy_derivative_proxy", 0.0)
            ),
            "candidate_velocity_gated_penalty_active": bool(
                item.get("candidate_velocity_gated_penalty_active", False)
            ),
            "candidate_velocity_gated_terminal_vx": float(
                item.get("candidate_velocity_gated_terminal_vx", float("nan"))
            ),
            "candidate_velocity_gated_terminal_theta": float(
                item.get("candidate_velocity_gated_terminal_theta", float("nan"))
            ),
        }
        for item in scored
    ]
    return np.asarray(best["action"], dtype=float), str(best["name"]), float(best["predicted_state_severity"]), json_scores


def hybrid_candidate_dicts(ppo_action: np.ndarray, state: np.ndarray) -> list[dict[str, Any]]:
    """Return the existing candidate library in the schema expected by the hybrid wrapper."""

    return [{"name": name, "action": action} for name, action in candidate_actions(ppo_action, state)]


def score_candidates_for_hybrid(
    env: Any,
    state: np.ndarray,
    candidates: list[dict[str, Any]],
    delay_queue: list[dict[str, Any]],
    context: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Use the existing E scorer as the base score for hybrid anti-drift selection."""

    current_step = int(context["current_step"])
    delay_sample = int(context["delay_sample"])
    reference = np.asarray(context.get("reference", np.array([])), dtype=float).reshape(-1)
    last_applied_action = context.get("last_applied_action")
    out: list[dict[str, Any]] = []
    for cand in candidates:
        name = str(cand.get("name", "candidate"))
        action = np.asarray(cand["action"], dtype=float)
        base_score, details = score_candidate_delay_aware(
            env=env,
            state=state,
            queue_state=delay_queue,
            current_step=current_step,
            candidate=action,
            delay_sample=delay_sample,
            reference=reference,
            last_applied_action=last_applied_action,
            velocity_boundary_weight=0.0,
            boundary_penalty_target="none",
            boundary_penalty_anti=False,
            slew_rate_weight=0.0,
        )
        pred_state = np.asarray(state, dtype=float).copy()
        for predicted_action in queue_actions_for_prediction(
            delay_queue,
            current_step,
            action,
            delay_sample,
            last_applied_action=last_applied_action,
        ):
            pred_state = predict_next_state(env, pred_state, predicted_action)
        out.append(
            {
                "candidate": cand,
                "candidate_name": name,
                "action": action,
                "predicted_state": pred_state,
                "base_score": base_score,
                "score": base_score,
                **details,
            }
        )
    return out


def hybrid_scores_for_log(decision: Any) -> list[dict[str, Any]]:
    """Serialize hybrid-ranked candidates into the existing candidate_scores column."""

    rows = []
    for item in decision.scores:
        rows.append(
            {
                "name": item["candidate_name"],
                "score": float(item.get("hybrid_score", item.get("base_score", 0.0))),
                "base_score": float(item.get("base_score", 0.0)),
                "action": np.asarray(item["action"], dtype=float).tolist(),
                "predicted_state_severity": float(item.get("predicted_state_severity", 0.0)),
                "candidate_hybrid_vx_penalty": float(item.get("vx_penalty", 0.0)),
                "candidate_hybrid_boundary_penalty": float(item.get("boundary_penalty", 0.0)),
                "candidate_hybrid_ttb_penalty": float(item.get("ttb_penalty", 0.0)),
                "candidate_hybrid_blind_brake_penalty": float(item.get("blind_brake_penalty", 0.0)),
                "candidate_hybrid_vx_impact": float(item.get("vx_impact", 0.0)),
                "candidate_hybrid_outward_vx_delta": float(item.get("outward_vx_delta", 0.0)),
                "candidate_hybrid_inward_vx_delta": float(item.get("inward_vx_delta", 0.0)),
                "candidate_hybrid_vx_brake_credit": float(item.get("vx_brake_credit", 0.0)),
                "candidate_vx": float(item.get("candidate_vx", float("nan"))),
            }
        )
    return rows


def score_candidate_delay_blind(
    env: Any,
    state: np.ndarray,
    candidate: np.ndarray,
    reference: np.ndarray,
    prediction_horizon: int = DEFAULT_PREDICTION_HORIZON,
) -> tuple[float, dict[str, float]]:
    """Score a candidate under the delay-blind assumption.

    This baseline intentionally uses the same candidate set, score weights, and
    symbolic model as the delay-aware condition, but it ignores pending queued
    actions and assumes the selected command is what matters immediately. D-vs-E
    is therefore a fairer stack-level comparison than a crude fixed hover
    fallback, not a pure single-variable queue-awareness ablation.
    """

    pred_state = np.asarray(state, dtype=float).copy()
    worst_state_severity = 0.0
    final_tracking = 0.0
    if int(prediction_horizon) < 1:
        raise ValueError("prediction_horizon must be at least 1")
    for _ in range(int(prediction_horizon)):
        pred_state = predict_next_state(env, pred_state, candidate)
        worst_state_severity = max(worst_state_severity, state_violation_severity(pred_state))
    if reference.size:
        n = min(reference.size, pred_state.size)
        final_tracking = float(np.sqrt(np.mean(np.square(pred_state[:n] - reference[:n]))))
    phys = denormalized_clipped_action(env, candidate)
    input_sev = input_violation_severity(phys)
    action_norm = float(np.linalg.norm(candidate))
    score = 100.0 * worst_state_severity + 5.0 * input_sev + 0.05 * final_tracking + 0.005 * action_norm
    return score, {
        "predicted_state_severity": worst_state_severity,
        "predicted_tracking_rmse": final_tracking,
        "candidate_input_severity": input_sev,
        "candidate_action_norm": action_norm,
    }


def choose_delay_blind_action(
    env: Any,
    state: np.ndarray,
    ppo_action: np.ndarray,
    reference: np.ndarray,
    prediction_horizon: int = DEFAULT_PREDICTION_HORIZON,
) -> tuple[np.ndarray, str, float, list[dict[str, Any]]]:
    scored = []
    for name, action in candidate_actions(ppo_action, state):
        score, details = score_candidate_delay_blind(
            env,
            state,
            action,
            reference,
            prediction_horizon=prediction_horizon,
        )
        scored.append({"name": name, "action": action, "score": score, **details})
    scored.sort(key=lambda item: item["score"])
    best = scored[0]
    json_scores = [
        {
            "name": item["name"],
            "score": float(item["score"]),
            "action": item["action"].tolist(),
            "predicted_state_severity": float(item["predicted_state_severity"]),
            "predicted_tracking_rmse": float(item["predicted_tracking_rmse"]),
            # Keep the same audit schema as the queue-aware scorer.
            "constraint_violation": float(item["predicted_state_severity"]),
            "action_violation": float(item["candidate_input_severity"]),
            "rmse": float(item["predicted_tracking_rmse"]),
            "l2_penalty": float(item["candidate_action_norm"]),
            "final_score": float(item["score"]),
        }
        for item in scored
    ]
    return np.asarray(best["action"], dtype=float), str(best["name"]), float(best["predicted_state_severity"]), json_scores


def tracking_error(state: np.ndarray, reference: np.ndarray) -> float:
    if state.size == 0 or reference.size == 0:
        return float("nan")
    n = min(state.size, reference.size)
    return float(np.sqrt(np.mean(np.square(state[:n] - reference[:n]))))


def candidate_score_by_name(candidate_scores: list[dict[str, Any]], name: str) -> float:
    for item in candidate_scores:
        if item.get("name") == name:
            return float(item.get("score", float("nan")))
    return float("nan")


def candidate_field_by_name(candidate_scores: list[dict[str, Any]], name: str, field: str) -> float:
    for item in candidate_scores:
        if item.get("name") == name:
            return float(item.get(field, float("nan")))
    return float("nan")


def candidate_rank_diagnostics(candidate_scores: list[dict[str, Any]]) -> dict[str, Any]:
    finite_scores: list[dict[str, Any]] = []
    for item in candidate_scores:
        try:
            score = float(item.get("score", float("nan")))
            score_without = float(item.get("score_without_boundary", item.get("candidate_score_without_boundary", score)))
            score_without_slew = float(item.get("score_without_slew", item.get("candidate_score_without_slew", score)))
            score_without_velocity_gated = float(
                item.get("score_without_velocity_gated_penalty", score)
            )
        except (TypeError, ValueError):
            continue
        if (
            not math.isfinite(score)
            or not math.isfinite(score_without)
            or not math.isfinite(score_without_slew)
            or not math.isfinite(score_without_velocity_gated)
        ):
            continue
        enriched = dict(item)
        enriched["score"] = score
        enriched["score_without_boundary"] = score_without
        enriched["score_without_slew"] = score_without_slew
        enriched["score_without_velocity_gated_penalty"] = score_without_velocity_gated
        finite_scores.append(enriched)

    if not finite_scores:
        return {
            "boundary_top_candidate_with_boundary": "",
            "boundary_top_candidate_without_boundary": "",
            "score_gap_top2_with_boundary": float("nan"),
            "score_gap_top2_without_boundary": float("nan"),
            "boundary_would_have_flipped": False,
            "boundary_penalty_any_active": False,
            "boundary_penalty_nonzero_count": 0,
            "boundary_penalty_max": 0.0,
            "boundary_penalty_mean": 0.0,
            "slew_top_candidate_with_slew": "",
            "slew_top_candidate_without_slew": "",
            "score_gap_top2_with_slew": float("nan"),
            "score_gap_top2_without_slew": float("nan"),
            "slew_would_have_flipped": False,
            "slew_penalty_any_active": False,
            "slew_penalty_nonzero_count": 0,
            "slew_penalty_max": 0.0,
            "slew_penalty_mean": 0.0,
            "velocity_gated_top_candidate_with_penalty": "",
            "velocity_gated_top_candidate_without_penalty": "",
            "score_gap_top2_with_velocity_gated_penalty": float("nan"),
            "score_gap_top2_without_velocity_gated_penalty": float("nan"),
            "velocity_gated_penalty_would_have_flipped": False,
            "velocity_gated_penalty_any_active": False,
            "velocity_gated_penalty_nonzero_count": 0,
            "velocity_gated_penalty_max": 0.0,
            "velocity_gated_penalty_mean": 0.0,
        }

    with_boundary = sorted(finite_scores, key=lambda item: (float(item["score"]), str(item.get("name", ""))))
    without_boundary = sorted(
        finite_scores,
        key=lambda item: (float(item["score_without_boundary"]), str(item.get("name", ""))),
    )
    without_slew = sorted(
        finite_scores,
        key=lambda item: (float(item["score_without_slew"]), str(item.get("name", ""))),
    )
    without_velocity_gated = sorted(
        finite_scores,
        key=lambda item: (
            float(item["score_without_velocity_gated_penalty"]),
            str(item.get("name", "")),
        ),
    )

    def top2_gap(rows: list[dict[str, Any]], key: str) -> float:
        if len(rows) < 2:
            return float("nan")
        return float(rows[1][key] - rows[0][key])

    penalties = np.array(
        [float(item.get("candidate_boundary_penalty", item.get("candidate_velocity_boundary_penalty", 0.0))) for item in finite_scores],
        dtype=float,
    )
    nonzero_count = int(np.sum(np.abs(penalties) > 1e-12))
    top_with = str(with_boundary[0].get("name", ""))
    top_without = str(without_boundary[0].get("name", ""))
    slew_penalties = np.array(
        [float(item.get("candidate_slew_rate_penalty", 0.0)) for item in finite_scores],
        dtype=float,
    )
    slew_nonzero_count = int(np.sum(np.abs(slew_penalties) > 1e-12))
    slew_top_without = str(without_slew[0].get("name", ""))
    velocity_gated_penalties = np.array(
        [float(item.get("candidate_velocity_gated_penalty", 0.0)) for item in finite_scores],
        dtype=float,
    )
    velocity_gated_nonzero_count = int(
        np.sum(np.abs(velocity_gated_penalties) > 1e-12)
    )
    velocity_gated_top_without = str(without_velocity_gated[0].get("name", ""))
    return {
        "boundary_top_candidate_with_boundary": top_with,
        "boundary_top_candidate_without_boundary": top_without,
        "score_gap_top2_with_boundary": top2_gap(with_boundary, "score"),
        "score_gap_top2_without_boundary": top2_gap(without_boundary, "score_without_boundary"),
        "boundary_would_have_flipped": bool(top_with != top_without),
        "boundary_penalty_any_active": bool(nonzero_count > 0),
        "boundary_penalty_nonzero_count": nonzero_count,
        "boundary_penalty_max": float(np.max(penalties)) if penalties.size else 0.0,
        "boundary_penalty_mean": float(np.mean(penalties)) if penalties.size else 0.0,
        "slew_top_candidate_with_slew": top_with,
        "slew_top_candidate_without_slew": slew_top_without,
        "score_gap_top2_with_slew": top2_gap(with_boundary, "score"),
        "score_gap_top2_without_slew": top2_gap(without_slew, "score_without_slew"),
        "slew_would_have_flipped": bool(top_with != slew_top_without),
        "slew_penalty_any_active": bool(slew_nonzero_count > 0),
        "slew_penalty_nonzero_count": slew_nonzero_count,
        "slew_penalty_max": float(np.max(slew_penalties)) if slew_penalties.size else 0.0,
        "slew_penalty_mean": float(np.mean(slew_penalties)) if slew_penalties.size else 0.0,
        "velocity_gated_top_candidate_with_penalty": top_with,
        "velocity_gated_top_candidate_without_penalty": velocity_gated_top_without,
        "score_gap_top2_with_velocity_gated_penalty": top2_gap(with_boundary, "score"),
        "score_gap_top2_without_velocity_gated_penalty": top2_gap(
            without_velocity_gated,
            "score_without_velocity_gated_penalty",
        ),
        "velocity_gated_penalty_would_have_flipped": bool(
            top_with != velocity_gated_top_without
        ),
        "velocity_gated_penalty_any_active": bool(velocity_gated_nonzero_count > 0),
        "velocity_gated_penalty_nonzero_count": velocity_gated_nonzero_count,
        "velocity_gated_penalty_max": float(np.max(velocity_gated_penalties))
        if velocity_gated_penalties.size
        else 0.0,
        "velocity_gated_penalty_mean": float(np.mean(velocity_gated_penalties))
        if velocity_gated_penalties.size
        else 0.0,
    }


def state_violation_contribution(state: np.ndarray, index: int) -> float:
    state_arr = np.asarray(state, dtype=float).reshape(-1)
    if index >= state_arr.size or index >= STATE_LOWER.size:
        return float("nan")
    lower = max(float(STATE_LOWER[index] - state_arr[index]), 0.0) / float(STATE_SCALE[index])
    upper = max(float(state_arr[index] - STATE_UPPER[index]), 0.0) / float(STATE_SCALE[index])
    return float(lower + upper)


def queue_head_field(queue_state: list[dict[str, Any]], field: str) -> Any:
    if not queue_state:
        return None
    return queue_state[0].get(field)


def first_predicted_action_for_audit(
    queue_state: list[dict[str, Any]],
    current_step: int,
    candidate: np.ndarray,
    delay_sample: int,
    last_applied_action: np.ndarray | None = None,
) -> np.ndarray:
    actions = queue_actions_for_prediction(
        queue_state,
        current_step,
        candidate,
        delay_sample,
        last_applied_action=last_applied_action,
    )
    if not actions:
        return np.zeros_like(candidate, dtype=float)
    return np.asarray(actions[0], dtype=float).copy()


def run_episode(
    condition: str,
    seed: int,
    steps: int,
    warning_buffer: float,
    output_rows: list[dict[str, Any]],
    delayed_delay_mode: str | None = None,
    fbm_hurst: float = 0.75,
    markov_q: float = 0.70,
    disturbance_mode: str = "none",
    thrust_decay_final: float = 0.70,
    adversarial_effort: float = 0.0,
    velocity_boundary_weight: float = 1.0,
    slew_rate_weight: float = 0.0,
    use_corrected_action_conversion: bool = True,
    use_velocity_gated_penalty: bool = False,
    lambda_penalty: float = 100.0,
    velocity_gated_min_vx: float = 0.80,
    prediction_horizon: int = DEFAULT_PREDICTION_HORIZON,
) -> dict[str, Any]:
    if int(prediction_horizon) < 1:
        raise ValueError("prediction_horizon must be at least 1")
    spec = dict(CONDITIONS[condition])
    if delayed_delay_mode and spec["delay"] != "zero":
        spec["delay"] = delayed_delay_mode
    config, env, ctrl, checkpoint = _build_controller_and_env_for_smoke()
    obs, info = env.reset(seed=seed)
    ctrl.reset_before_run(obs, info, env=env)
    delay_schedule = make_delay_schedule(seed=seed, steps=steps, mode=spec["delay"], fbm_hurst=fbm_hurst, markov_q=markov_q)
    delay_stats = delay_schedule_stats(delay_schedule)
    channel = ScheduledDelayChannel(hold_action=np.zeros(2, dtype=float))
    hybrid_monitor = None
    if spec["monitor"] == "hybrid_delay_aware":
        hybrid_monitor = HybridDelayAwareMonitor(
            delay_aware_scorer=lambda state, candidates, delay_queue, context: score_candidates_for_hybrid(
                env=env,
                state=np.asarray(state, dtype=float),
                candidates=list(candidates),
                delay_queue=list(delay_queue),
                context=context,
            ),
            config=hybrid_config_from_dict(spec.get("hybrid", {})),
        )

    severities: list[float] = []
    rewards: list[float] = []
    rmses: list[float] = []
    warning_count = 0
    intervention_count = 0
    first_violation_step = None
    first_warning_step = None
    first_intervention_step = None
    done_reason = ""
    info_keys: list[str] = []
    action_conversion_audit: list[dict[str, Any]] = []
    velocity_gated_penalty_for_e = bool(
        use_velocity_gated_penalty and condition == "E_delay_aware_intervention"
    )
    start = time.perf_counter()

    for step in range(steps):
        step_start = time.perf_counter()
        state_before = np.asarray(getattr(env, "state", np.array([])), dtype=float).reshape(-1)
        reference = _lookup_reference(env, step)
        ppo_action = np.asarray(ctrl.select_action(obs, info), dtype=float).reshape(-1)
        delay_sample = delay_schedule[step]
        queue_state_before = channel.queue_state()
        c_values_before = flatten_constraint_values(info.get("constraint_values"))
        current_max_constraint = max_state_constraint_value(c_values_before)
        current_severity_hint = constraint_severity(c_values_before)
        current_warning = bool(current_max_constraint > -warning_buffer or current_severity_hint > 0.0)

        warning = False
        intervention = False
        command = ppo_action.copy()
        action_source = "ppo"
        selected_candidate = ""
        predicted_severity = float("nan")
        candidate_scores: list[dict[str, Any]] = []
        hybrid_log_fields: dict[str, Any] = {}

        if spec["monitor"] == "delay_blind":
            warning = current_warning
            if warning and spec["intervention"] == "delay_blind":
                command, selected_candidate, predicted_severity, candidate_scores = choose_delay_blind_action(
                    env=env,
                    state=state_before,
                    ppo_action=ppo_action,
                    reference=reference,
                    prediction_horizon=prediction_horizon,
                )
                intervention = True
                action_source = f"delay_blind_{selected_candidate}"
        elif spec["monitor"] == "delay_aware":
            use_corrected_rollout = bool(spec.get("held_action_corrected", False))
            boundary_penalty_target = str(spec.get("boundary_penalty_target", "none"))
            boundary_penalty_anti = bool(spec.get("boundary_penalty_anti", False))
            scorer_last_applied = (
                channel.last_applied_action(fallback_shape=ppo_action.shape) if use_corrected_rollout else None
            )
            scorer_velocity_weight = velocity_boundary_weight if boundary_penalty_target != "none" else 0.0
            scorer_slew_weight = slew_rate_weight if bool(spec.get("slew_rate_penalty", False)) else 0.0
            # Score the PPO-clipped action first as a warning proxy. Candidate
            # selection is only used for condition E.
            _, _, predicted_severity, preview_scores = choose_delay_aware_action(
                env=env,
                state=state_before,
                ppo_action=ppo_action,
                queue_state=queue_state_before,
                current_step=step,
                delay_sample=delay_sample,
                reference=reference,
                last_applied_action=scorer_last_applied,
                velocity_boundary_weight=scorer_velocity_weight,
                boundary_penalty_target=boundary_penalty_target,
                boundary_penalty_anti=boundary_penalty_anti,
                slew_rate_weight=scorer_slew_weight,
                use_velocity_gated_penalty=velocity_gated_penalty_for_e,
                lambda_penalty=lambda_penalty,
                velocity_gated_min_vx=velocity_gated_min_vx,
                **(
                    {}
                    if int(prediction_horizon) == DEFAULT_PREDICTION_HORIZON
                    else {"prediction_horizon": int(prediction_horizon)}
                ),
            )
            candidate_scores = preview_scores
            warning = bool(current_warning or predicted_severity > 0.0)
            if warning and str(spec["intervention"]).startswith("delay_aware"):
                command, selected_candidate, predicted_severity, candidate_scores = choose_delay_aware_action(
                    env=env,
                    state=state_before,
                    ppo_action=ppo_action,
                    queue_state=queue_state_before,
                    current_step=step,
                    delay_sample=delay_sample,
                    reference=reference,
                    last_applied_action=scorer_last_applied,
                    velocity_boundary_weight=scorer_velocity_weight,
                    boundary_penalty_target=boundary_penalty_target,
                    boundary_penalty_anti=boundary_penalty_anti,
                    slew_rate_weight=scorer_slew_weight,
                    use_velocity_gated_penalty=velocity_gated_penalty_for_e,
                    lambda_penalty=lambda_penalty,
                    velocity_gated_min_vx=velocity_gated_min_vx,
                    **(
                        {}
                        if int(prediction_horizon) == DEFAULT_PREDICTION_HORIZON
                        else {"prediction_horizon": int(prediction_horizon)}
                    ),
                )
                intervention = True
                action_source = f"{spec['intervention']}_{selected_candidate}"
        elif spec["monitor"] == "hybrid_delay_aware":
            scorer_last_applied = channel.last_applied_action(fallback_shape=ppo_action.shape)
            _, _, predicted_severity, preview_scores = choose_delay_aware_action(
                env=env,
                state=state_before,
                ppo_action=ppo_action,
                queue_state=queue_state_before,
                current_step=step,
                delay_sample=delay_sample,
                reference=reference,
                last_applied_action=scorer_last_applied,
                velocity_boundary_weight=0.0,
                boundary_penalty_target="none",
                boundary_penalty_anti=False,
                slew_rate_weight=0.0,
            )
            candidate_scores = preview_scores
            warning = bool(current_warning or predicted_severity > 0.0)
            if warning and spec["intervention"] == "hybrid_delay_aware":
                if hybrid_monitor is None:
                    raise RuntimeError("Hybrid condition selected but hybrid monitor was not initialized.")
                decision = hybrid_monitor.select_action(
                    state=state_before,
                    candidates=hybrid_candidate_dicts(ppo_action, state_before),
                    delay_queue=queue_state_before,
                    context={
                        "seed": seed,
                        "condition": condition,
                        "current_step": step,
                        "delay_sample": delay_sample,
                        "reference": reference,
                        "last_applied_action": scorer_last_applied,
                    },
                )
                command = np.asarray(decision.action, dtype=float).copy()
                selected_candidate = str(decision.candidate_name)
                predicted_severity = float(decision.scores[0].get("predicted_state_severity", predicted_severity))
                candidate_scores = hybrid_scores_for_log(decision)
                hybrid_log_fields = dict(decision.log_fields)
                intervention = True
                action_source = f"{decision.mode}_{selected_candidate}"

        monitor_command = command.copy()
        audit_selected_candidate = selected_candidate or "ppo_clipped"
        selected_candidate_score = candidate_score_by_name(candidate_scores, audit_selected_candidate)
        selected_candidate_score_without_boundary = candidate_field_by_name(
            candidate_scores,
            audit_selected_candidate,
            "score_without_boundary",
        )
        selected_candidate_boundary_penalty = candidate_field_by_name(
            candidate_scores,
            audit_selected_candidate,
            "candidate_boundary_penalty",
        )
        selected_candidate_slew_penalty = candidate_field_by_name(
            candidate_scores,
            audit_selected_candidate,
            "candidate_slew_rate_penalty",
        )
        selected_candidate_slew_delta_norm = candidate_field_by_name(
            candidate_scores,
            audit_selected_candidate,
            "candidate_slew_action_delta_norm",
        )
        selected_candidate_velocity_gated_penalty = candidate_field_by_name(
            candidate_scores,
            audit_selected_candidate,
            "candidate_velocity_gated_penalty",
        )
        selected_candidate_velocity_gated_proxy = candidate_field_by_name(
            candidate_scores,
            audit_selected_candidate,
            "candidate_velocity_gated_energy_derivative_proxy",
        )
        selected_candidate_velocity_gated_active = candidate_field_by_name(
            candidate_scores,
            audit_selected_candidate,
            "candidate_velocity_gated_penalty_active",
        )
        selected_candidate_velocity_gated_terminal_vx = candidate_field_by_name(
            candidate_scores,
            audit_selected_candidate,
            "candidate_velocity_gated_terminal_vx",
        )
        selected_candidate_velocity_gated_terminal_theta = candidate_field_by_name(
            candidate_scores,
            audit_selected_candidate,
            "candidate_velocity_gated_terminal_theta",
        )
        ppo_candidate_score = candidate_score_by_name(candidate_scores, "ppo_clipped")
        ppo_candidate_score_without_boundary = candidate_field_by_name(
            candidate_scores,
            "ppo_clipped",
            "score_without_boundary",
        )
        ppo_candidate_velocity_gated_penalty = candidate_field_by_name(
            candidate_scores,
            "ppo_clipped",
            "candidate_velocity_gated_penalty",
        )
        candidate_rank_info = candidate_rank_diagnostics(candidate_scores)
        if math.isfinite(selected_candidate_score) and math.isfinite(ppo_candidate_score):
            score_gap_selected_vs_ppo = selected_candidate_score - ppo_candidate_score
        else:
            score_gap_selected_vs_ppo = float("nan")
        predicted_first_applied_action = first_predicted_action_for_audit(
            queue_state_before,
            step,
            monitor_command,
            delay_sample,
            last_applied_action=scorer_last_applied if bool(spec.get("held_action_corrected", False)) else None,
        )
        predicted_state_after_first = predict_next_state(env, state_before, predicted_first_applied_action)

        delay_channel_command = monitor_command.copy()
        thrust_effectiveness = 1.0
        adversarial_delta = np.zeros_like(monitor_command)
        adversarial_direction = ""
        adversarial_score = float("nan")
        action_conversion_record: dict[str, Any] = {}
        if disturbance_mode == "thrust_decay":
            thrust_effectiveness = thrust_effectiveness_at_step(step, steps, thrust_decay_final)
            delay_channel_command = apply_thrust_effectiveness_decay(
                env,
                monitor_command,
                thrust_effectiveness,
                use_corrected_action_conversion=use_corrected_action_conversion,
                audit_log=action_conversion_audit,
            )
            action_conversion_record = action_conversion_audit[-1]
        elif disturbance_mode == "adversarial_saturation":
            delay_channel_command, adversarial_delta, adversarial_direction, adversarial_score = choose_adversarial_saturation_command(
                env=env,
                state=state_before,
                command=monitor_command,
                queue_state=channel.queue_state(),
                current_step=step,
                delay_sample=delay_sample,
                effort=adversarial_effort,
            )
        elif disturbance_mode != "none":
            raise ValueError(f"Unknown disturbance_mode {disturbance_mode}")

        delay_result = channel.update(delay_channel_command, delay_sample)
        next_obs, reward, done, info = env.step(delay_result.applied_action)
        runtime_ms = (time.perf_counter() - step_start) * 1000.0

        state_after = np.asarray(getattr(env, "state", np.array([])), dtype=float).reshape(-1)
        reference_after = _lookup_reference(env, step)
        c_values = flatten_constraint_values(info.get("constraint_values"))
        severity = constraint_severity(c_values)
        rmse = tracking_error(state_after, reference_after)
        n_pred = min(state_after.size, predicted_state_after_first.size)
        if n_pred:
            state_pred_delta = state_after[:n_pred] - predicted_state_after_first[:n_pred]
            state_pred_error_l2 = float(np.linalg.norm(state_pred_delta))
            state_pred_error_max = float(np.max(np.abs(state_pred_delta)))
        else:
            state_pred_error_l2 = float("nan")
            state_pred_error_max = float("nan")
        applied_action_prediction_error_l2 = float(
            np.linalg.norm(np.asarray(delay_result.applied_action, dtype=float) - predicted_first_applied_action)
        )
        applied_entry = delay_result.applied_entry or {}

        if severity > 0.0 and first_violation_step is None:
            first_violation_step = step
        if warning and first_warning_step is None:
            first_warning_step = step
        if intervention and first_intervention_step is None:
            first_intervention_step = step
        if warning:
            warning_count += 1
        if intervention:
            intervention_count += 1
        if not info_keys:
            info_keys = sorted(str(k) for k in info.keys())
        if done and not done_reason:
            done_reason = "out_of_bounds" if bool(info.get("out_of_bounds", False)) else "done"

        severities.append(severity)
        rewards.append(float(reward))
        if not math.isnan(rmse):
            rmses.append(rmse)

        output_rows.append(
            {
                "seed": seed,
                "condition": condition,
                "step": step,
                "delay_mode": spec["delay"],
                "delay_markov_q": markov_q if spec["delay"] == "markov_symmetric" else "",
                "disturbance_mode": disturbance_mode,
                "thrust_decay_final": thrust_decay_final,
                "thrust_effectiveness": thrust_effectiveness,
                "use_corrected_action_conversion": use_corrected_action_conversion,
                "action_conversion_u_physical_raw": json.dumps(
                    _jsonable(action_conversion_record.get("u_physical_raw", []))
                ),
                "action_conversion_u_physical_for_multiplier": json.dumps(
                    _jsonable(action_conversion_record.get("u_physical_for_multiplier", []))
                ),
                "action_conversion_u_decayed": json.dumps(
                    _jsonable(action_conversion_record.get("u_decayed", []))
                ),
                "action_conversion_action_to_env": json.dumps(
                    _jsonable(action_conversion_record.get("action_to_env", []))
                ),
                "action_conversion_recovered_physical": json.dumps(
                    _jsonable(action_conversion_record.get("recovered_physical", []))
                ),
                "adversarial_effort": adversarial_effort,
                "held_action_corrected_rollout": bool(spec.get("held_action_corrected", False)),
                "boundary_penalty_target": str(spec.get("boundary_penalty_target", "none")),
                "boundary_penalty_anti": bool(spec.get("boundary_penalty_anti", False)),
                "velocity_boundary_weight": velocity_boundary_weight
                if str(spec.get("boundary_penalty_target", "none")) != "none"
                else 0.0,
                "slew_rate_penalty_enabled": bool(spec.get("slew_rate_penalty", False)),
                "slew_rate_weight": slew_rate_weight if bool(spec.get("slew_rate_penalty", False)) else 0.0,
                "velocity_gated_penalty_enabled": velocity_gated_penalty_for_e,
                "velocity_gated_lambda_penalty": lambda_penalty
                if velocity_gated_penalty_for_e
                else 0.0,
                "velocity_gated_min_vx": velocity_gated_min_vx,
                "queue_len_before": len(queue_state_before),
                "queue_head_command_step_before": queue_head_field(queue_state_before, "command_step"),
                "queue_head_release_step_before": queue_head_field(queue_state_before, "release_step"),
                "queue_head_delay_sample_before": queue_head_field(queue_state_before, "delay_sample"),
                "candidate_delay_sample": delay_sample,
                "candidate_release_step": step + int(delay_sample),
                "applied_command_step": applied_entry.get("command_step"),
                "applied_release_step": applied_entry.get("release_step"),
                "applied_original_delay": applied_entry.get("delay_sample"),
                "applied_action_age": delay_result.applied_action_age,
                "held_last_action_used": delay_result.held_last_action_used,
                "delay_sample": delay_sample,
                "delay_schedule_hash": schedule_hash(delay_schedule),
                "monitor": spec["monitor"],
                "intervention_mode": spec["intervention"],
                "warning": warning,
                "intervention": intervention,
                "action_source": action_source,
                "selected_candidate": selected_candidate,
                "hybrid_mode": hybrid_log_fields.get("hybrid_mode", ""),
                "hybrid_reason": hybrid_log_fields.get("hybrid_reason", ""),
                "hybrid_mode_switch": hybrid_log_fields.get("hybrid_mode_switch", False),
                "hybrid_x": hybrid_log_fields.get("hybrid_x", float("nan")),
                "hybrid_vx": hybrid_log_fields.get("hybrid_vx", float("nan")),
                "hybrid_abs_vx": hybrid_log_fields.get("hybrid_abs_vx", float("nan")),
                "hybrid_dist_to_x_bound": hybrid_log_fields.get("hybrid_dist_to_x_bound", float("nan")),
                "hybrid_ttb_steps": hybrid_log_fields.get("hybrid_ttb_steps", float("nan")),
                "hybrid_selected_candidate": hybrid_log_fields.get("hybrid_selected_candidate", ""),
                "hybrid_selected_score": hybrid_log_fields.get("hybrid_selected_score", float("nan")),
                "hybrid_selected_base_score": hybrid_log_fields.get("hybrid_selected_base_score", float("nan")),
                "hybrid_selected_vx_penalty": hybrid_log_fields.get("hybrid_selected_vx_penalty", float("nan")),
                "hybrid_selected_boundary_penalty": hybrid_log_fields.get("hybrid_selected_boundary_penalty", float("nan")),
                "hybrid_selected_ttb_penalty": hybrid_log_fields.get("hybrid_selected_ttb_penalty", float("nan")),
                "hybrid_selected_blind_brake_penalty": hybrid_log_fields.get("hybrid_selected_blind_brake_penalty", float("nan")),
                "hybrid_selected_vx_impact": hybrid_log_fields.get("hybrid_selected_vx_impact", float("nan")),
                "hybrid_selected_outward_vx_delta": hybrid_log_fields.get("hybrid_selected_outward_vx_delta", float("nan")),
                "hybrid_selected_inward_vx_delta": hybrid_log_fields.get("hybrid_selected_inward_vx_delta", float("nan")),
                "hybrid_selected_vx_brake_credit": hybrid_log_fields.get("hybrid_selected_vx_brake_credit", float("nan")),
                "ppo_commanded": json.dumps(_jsonable(ppo_action)),
                "monitor_commanded": json.dumps(_jsonable(monitor_command)),
                "delay_channel_commanded": json.dumps(_jsonable(delay_channel_command)),
                "adversarial_delta": json.dumps(_jsonable(adversarial_delta)),
                "adversarial_direction": adversarial_direction,
                "adversarial_score": adversarial_score,
                "predicted_first_applied_action": json.dumps(_jsonable(predicted_first_applied_action)),
                "actual_applied_action": json.dumps(_jsonable(delay_result.applied_action)),
                "applied_action_prediction_error_l2": applied_action_prediction_error_l2,
                "state_pred_error_l2": state_pred_error_l2,
                "state_pred_error_max": state_pred_error_max,
                "selected_candidate_score": selected_candidate_score,
                "selected_candidate_score_without_boundary": selected_candidate_score_without_boundary,
                "selected_candidate_boundary_penalty": selected_candidate_boundary_penalty,
                "selected_candidate_slew_penalty": selected_candidate_slew_penalty,
                "selected_candidate_slew_delta_norm": selected_candidate_slew_delta_norm,
                "selected_candidate_velocity_gated_penalty": selected_candidate_velocity_gated_penalty,
                "selected_candidate_velocity_gated_energy_derivative_proxy": selected_candidate_velocity_gated_proxy,
                "selected_candidate_velocity_gated_penalty_active": bool(
                    selected_candidate_velocity_gated_active
                )
                if math.isfinite(selected_candidate_velocity_gated_active)
                else False,
                "selected_candidate_velocity_gated_terminal_vx": selected_candidate_velocity_gated_terminal_vx,
                "selected_candidate_velocity_gated_terminal_theta": selected_candidate_velocity_gated_terminal_theta,
                "ppo_candidate_score": ppo_candidate_score,
                "ppo_candidate_score_without_boundary": ppo_candidate_score_without_boundary,
                "ppo_candidate_velocity_gated_penalty": ppo_candidate_velocity_gated_penalty,
                "score_gap_selected_vs_ppo": score_gap_selected_vs_ppo,
                "boundary_top_candidate_with_boundary": candidate_rank_info["boundary_top_candidate_with_boundary"],
                "boundary_top_candidate_without_boundary": candidate_rank_info["boundary_top_candidate_without_boundary"],
                "score_gap_top2_with_boundary": candidate_rank_info["score_gap_top2_with_boundary"],
                "score_gap_top2_without_boundary": candidate_rank_info["score_gap_top2_without_boundary"],
                "boundary_would_have_flipped": candidate_rank_info["boundary_would_have_flipped"],
                "boundary_penalty_any_active": candidate_rank_info["boundary_penalty_any_active"],
                "boundary_penalty_nonzero_count": candidate_rank_info["boundary_penalty_nonzero_count"],
                "boundary_penalty_max": candidate_rank_info["boundary_penalty_max"],
                "boundary_penalty_mean": candidate_rank_info["boundary_penalty_mean"],
                "slew_top_candidate_with_slew": candidate_rank_info["slew_top_candidate_with_slew"],
                "slew_top_candidate_without_slew": candidate_rank_info["slew_top_candidate_without_slew"],
                "score_gap_top2_with_slew": candidate_rank_info["score_gap_top2_with_slew"],
                "score_gap_top2_without_slew": candidate_rank_info["score_gap_top2_without_slew"],
                "slew_would_have_flipped": candidate_rank_info["slew_would_have_flipped"],
                "slew_penalty_any_active": candidate_rank_info["slew_penalty_any_active"],
                "slew_penalty_nonzero_count": candidate_rank_info["slew_penalty_nonzero_count"],
                "slew_penalty_max": candidate_rank_info["slew_penalty_max"],
                "slew_penalty_mean": candidate_rank_info["slew_penalty_mean"],
                "velocity_gated_top_candidate_with_penalty": candidate_rank_info[
                    "velocity_gated_top_candidate_with_penalty"
                ],
                "velocity_gated_top_candidate_without_penalty": candidate_rank_info[
                    "velocity_gated_top_candidate_without_penalty"
                ],
                "score_gap_top2_with_velocity_gated_penalty": candidate_rank_info[
                    "score_gap_top2_with_velocity_gated_penalty"
                ],
                "score_gap_top2_without_velocity_gated_penalty": candidate_rank_info[
                    "score_gap_top2_without_velocity_gated_penalty"
                ],
                "velocity_gated_penalty_would_have_flipped": candidate_rank_info[
                    "velocity_gated_penalty_would_have_flipped"
                ],
                "velocity_gated_penalty_any_active": candidate_rank_info[
                    "velocity_gated_penalty_any_active"
                ],
                "velocity_gated_penalty_nonzero_count": candidate_rank_info[
                    "velocity_gated_penalty_nonzero_count"
                ],
                "velocity_gated_penalty_max": candidate_rank_info[
                    "velocity_gated_penalty_max"
                ],
                "velocity_gated_penalty_mean": candidate_rank_info[
                    "velocity_gated_penalty_mean"
                ],
                "x_violation_contribution": state_violation_contribution(state_after, 0),
                "x_dot_violation_contribution": state_violation_contribution(state_after, 1),
                "theta_violation_contribution": state_violation_contribution(state_after, 4),
                "theta_dot_violation_contribution": state_violation_contribution(state_after, 5),
                "delayed_applied": json.dumps(_jsonable(delay_result.applied_action)),
                "queue_state": json.dumps(_jsonable(delay_result.queue_state)),
                "state": json.dumps(_jsonable(state_after)),
                "reference_state": json.dumps(_jsonable(reference_after)),
                "observation": json.dumps(_jsonable(obs)),
                "constraint_values": json.dumps(_jsonable(c_values)),
                "current_max_constraint_before_action": current_max_constraint,
                "normalized_violation_severity": severity,
                "predicted_state_severity": predicted_severity,
                "candidate_scores": json.dumps(_jsonable(candidate_scores)),
                "tracking_rmse": rmse,
                "reward": float(reward),
                "done": bool(done),
                "done_reason": done_reason,
                "out_of_bounds": bool(info.get("out_of_bounds", False)),
                "runtime_ms": runtime_ms,
            }
        )

        obs = next_obs
        if done:
            break

    action_low, action_high = env.physical_action_bounds
    action_conversion_verification = verify_action_conversion_audit(
        action_conversion_audit,
        np.asarray(action_low, dtype=float),
        np.asarray(action_high, dtype=float),
    )
    env.close()
    try:
        ctrl.close()
    except Exception:
        pass
    elapsed = time.perf_counter() - start
    return {
        "seed": seed,
        "condition": condition,
        "delay_mode": spec["delay"],
        "delay_markov_q": markov_q if spec["delay"] == "markov_symmetric" else "",
        "disturbance_mode": disturbance_mode,
        "thrust_decay_final": thrust_decay_final,
        "use_corrected_action_conversion": use_corrected_action_conversion,
        "action_conversion_total_checked": action_conversion_verification["total_checked"],
        "action_conversion_multiplier_mismatches": action_conversion_verification["multiplier_mismatches"],
        "action_conversion_roundtrip_mismatches": action_conversion_verification["roundtrip_mismatches"],
        "action_conversion_max_multiplier_error": action_conversion_verification["max_multiplier_error"],
        "action_conversion_max_roundtrip_error": action_conversion_verification["max_roundtrip_error"],
        "adversarial_effort": adversarial_effort,
        "slew_rate_weight": slew_rate_weight if bool(spec.get("slew_rate_penalty", False)) else 0.0,
        "velocity_gated_penalty_enabled": velocity_gated_penalty_for_e,
        "velocity_gated_lambda_penalty": lambda_penalty
        if velocity_gated_penalty_for_e
        else 0.0,
        "velocity_gated_min_vx": velocity_gated_min_vx,
        "monitor": spec["monitor"],
        "intervention_mode": spec["intervention"],
        "delay_schedule_hash": schedule_hash(delay_schedule),
        "delay_schedule_mean": delay_stats.get("mean", float("nan")),
        "delay_schedule_std": delay_stats.get("std", float("nan")),
        "delay_schedule_max": delay_stats.get("max", float("nan")),
        "delay_schedule_fraction_zero": delay_stats.get("fraction_zero", float("nan")),
        "delay_schedule_fraction_high_ge2": delay_stats.get("fraction_high_ge2", float("nan")),
        "delay_schedule_longest_high_run_ge2": delay_stats.get("longest_high_run_ge2", float("nan")),
        "delay_schedule_lag1_autocorrelation": delay_stats.get("lag1_autocorrelation", float("nan")),
        "steps_requested": steps,
        "steps_recorded": len(severities),
        "checkpoint": str(checkpoint),
        "cumulative_reward": float(np.sum(rewards)),
        "tracking_rmse": float(np.mean(rmses)) if rmses else float("nan"),
        "cumulative_normalized_violation_severity": cumulative_normalized_violation_severity(severities),
        "violating_step_count": violating_step_count(severities),
        "max_normalized_violation_severity": float(np.max(severities)) if severities else 0.0,
        "zero_violation_episode": bool(not any(s > 1e-12 for s in severities)),
        "warning_count": warning_count,
        "intervention_count": intervention_count,
        "first_warning_step": first_warning_step,
        "first_intervention_step": first_intervention_step,
        "first_violation_step": first_violation_step,
        "done_reason": done_reason,
        "info_keys": info_keys,
        "runtime_seconds": elapsed,
        "mean_runtime_ms": float(np.mean([row["runtime_ms"] for row in output_rows if row["seed"] == seed and row["condition"] == condition])),
        "config_constraints": _jsonable(config.task_config.constraints),
    }


def aggregate(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    by_condition: dict[str, list[dict[str, Any]]] = {}
    for ep in episodes:
        by_condition.setdefault(ep["condition"], []).append(ep)
    aggregates: dict[str, Any] = {}
    fields = [
        "cumulative_normalized_violation_severity",
        "violating_step_count",
        "tracking_rmse",
        "cumulative_reward",
        "warning_count",
        "intervention_count",
        "runtime_seconds",
    ]
    for condition, rows in by_condition.items():
        out: dict[str, Any] = {"n": len(rows)}
        for field in fields:
            vals = np.array([float(row[field]) for row in rows], dtype=float)
            out[field] = {
                "mean": float(np.nanmean(vals)),
                "std": float(np.nanstd(vals, ddof=1)) if len(vals) > 1 else 0.0,
                "min": float(np.nanmin(vals)),
                "max": float(np.nanmax(vals)),
            }
        out["zero_violation_fraction"] = float(np.mean([bool(row["zero_violation_episode"]) for row in rows]))
        out["out_of_bounds_fraction"] = float(np.mean([row["done_reason"] == "out_of_bounds" for row in rows]))
        aggregates[condition] = out

    by_seed = {(row["seed"], row["condition"]): row for row in episodes}
    seeds = sorted({row["seed"] for row in episodes})
    d_condition = "D_delay_blind_intervention"
    for compare_condition in sorted(name for name in by_condition if name != d_condition):
        paired: list[dict[str, Any]] = []
        for seed in seeds:
            d = by_seed.get((seed, d_condition))
            other = by_seed.get((seed, compare_condition))
            if d and other:
                dsev = float(d["cumulative_normalized_violation_severity"])
                other_sev = float(other["cumulative_normalized_violation_severity"])
                reduction = (dsev - other_sev) / dsev if dsev > 1e-12 else float("nan")
                paired.append({"seed": seed, "D": dsev, compare_condition: other_sev, "relative_reduction": reduction})
        if paired:
            vals = np.array([p["relative_reduction"] for p in paired if not math.isnan(p["relative_reduction"])], dtype=float)
            key = f"paired_{condition_short_label(compare_condition).replace('+', 'plus')}_vs_D"
            aggregates[key] = {
                "condition": compare_condition,
                "n": len(vals),
                "mean_relative_reduction": float(np.mean(vals)) if vals.size else float("nan"),
                "median_relative_reduction": float(np.median(vals)) if vals.size else float("nan"),
                "supports_delay_aware_stack_band": bool(vals.size and np.mean(vals) >= 0.30),
                "practical_null_band": bool(vals.size and np.mean(vals) < 0.10),
                "paired_rows": paired,
            }
    return aggregates


def condition_short_label(name: str) -> str:
    if name.startswith("E_plus"):
        return "E+"
    if name.startswith("E_held"):
        return "H"
    if name.startswith("E_slew"):
        return "HS"
    if name.startswith("E_xdot_target"):
        return "X"
    if name.startswith("E_theta_sham"):
        return "S"
    if name.startswith("E_xdot_antitarget"):
        return "A"
    return name.split("_", 1)[0]


def write_plots(output_dir: Path, aggregates: dict[str, Any]) -> list[str]:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return []

    names = [name for name in CONDITIONS if name in aggregates]
    labels = [condition_short_label(name) for name in names]
    palette = {
        "A": "#449161",
        "B": "#bc5048",
        "C": "#de8d3d",
        "D": "#7460aa",
        "E": "#3771bb",
        "E+": "#1b9e77",
        "H": "#1b9e77",
        "X": "#66a61e",
        "S": "#e6ab02",
        "A": "#d95f02",
    }
    colors = [palette.get(label, "#555555") for label in labels]
    plot_specs = [
        ("cumulative_normalized_violation_severity", "Cumulative normalized severity", "phase3_severity_by_condition.png"),
        ("violating_step_count", "Constraint-violating steps", "phase3_violating_steps_by_condition.png"),
        ("tracking_rmse", "Tracking RMSE", "phase3_tracking_rmse_by_condition.png"),
        ("intervention_count", "Interventions / episode", "phase3_interventions_by_condition.png"),
    ]
    paths: list[str] = []
    for field, ylabel, filename in plot_specs:
        means = [aggregates[name][field]["mean"] for name in names]
        stds = [aggregates[name][field]["std"] for name in names]
        fig, ax = plt.subplots(figsize=(8.5, 5.0), dpi=160)
        lower_err = [min(float(mean), float(std)) for mean, std in zip(means, stds)]
        upper_err = [float(std) for std in stds]
        ax.bar(labels, means, yerr=np.array([lower_err, upper_err]), color=colors, capsize=4)
        ax.set_title(ylabel + " by condition")
        ax.set_xlabel("Condition (A zero, B delay, C warning-only, D blind, E aware)")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.25)
        for i, mean in enumerate(means):
            ax.text(i, mean, f"{mean:.3g}", ha="center", va="bottom", fontsize=9)
        fig.tight_layout()
        out = output_dir / filename
        fig.savefig(out)
        plt.close(fig)
        paths.append(str(out))
    return paths


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--steps", type=int, default=250)
    parser.add_argument("--seed-start", type=int, default=1337)
    parser.add_argument("--warning-buffer", type=float, default=0.05)
    parser.add_argument(
        "--delayed-delay-mode",
        "--delay-mode",
        dest="delayed_delay_mode",
        choices=DELAY_MODE_CHOICES,
        default=None,
        help="Optional override for nonzero-delay conditions B-E. Defaults to each condition's declared mode.",
    )
    parser.add_argument("--fbm-hurst", type=float, default=0.75, help="Hurst exponent for fbm_uniform_0_2 stress schedules.")
    parser.add_argument(
        "--markov-q",
        type=float,
        default=0.70,
        help="Persistence probability for markov_symmetric delay schedules.",
    )
    parser.add_argument(
        "--conditions",
        default=",".join(CONDITIONS.keys()),
        help="Comma-separated condition names to run. Defaults to all A-E conditions.",
    )
    parser.add_argument(
        "--disturbance-mode",
        choices=["none", "thrust_decay", "adversarial_saturation"],
        default="none",
        help="Optional Phase 5/6 pre-delay command disturbance. Default preserves Phase 0-4 paths.",
    )
    parser.add_argument(
        "--thrust-decay-final",
        type=float,
        default=0.70,
        help="Final Phase 5 thrust effectiveness at the last step; linearly interpolated from 1.0.",
    )
    parser.add_argument(
        "--use-corrected-action-conversion",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Apply thrust effectiveness in physical space and return through env.normalize_action "
            "(default). Use --no-use-corrected-action-conversion only to reproduce archived runs."
        ),
    )
    parser.add_argument(
        "--adversarial-effort",
        type=float,
        default=0.10,
        help="Maximum normalized-action additive effort for Phase 6 adversarial saturation.",
    )
    parser.add_argument(
        "--velocity-boundary-weight",
        type=float,
        default=1.0,
        help="Phase 7 E+ penalty weight for predicted x-dot increases near the x-dot boundary.",
    )
    parser.add_argument(
        "--slew-rate-weight",
        type=float,
        default=0.01,
        help="Phase 10 H+Slew penalty weight for command changes relative to the last applied action.",
    )
    parser.add_argument(
        "--use-velocity-gated-penalty",
        "--use_velocity_gated_penalty",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Opt in to the velocity-gated terminal pitch penalty for arm E only. "
            "Default false preserves the original scorer."
        ),
    )
    parser.add_argument(
        "--lambda-penalty",
        "--lambda_penalty",
        type=float,
        default=100.0,
        help="Weight for E's velocity-gated terminal pitch penalty (default: 100.0).",
    )
    parser.add_argument(
        "--velocity-gated-min-vx",
        "--velocity_gated_min_vx",
        type=float,
        default=0.80,
        help=(
            "Minimum predicted terminal forward velocity that activates the penalty "
            "(default: 0.80 m/s)."
        ),
    )
    parser.add_argument(
        "--prediction-horizon",
        type=int,
        default=DEFAULT_PREDICTION_HORIZON,
        help="Candidate-scoring rollout horizon N (default: 3).",
    )
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "reports" / "quadrotor_delay_ra_experiment"))
    args = parser.parse_args()

    if args.lambda_penalty < 0.0:
        raise ValueError("--lambda-penalty must be non-negative")
    if args.velocity_gated_min_vx < 0.0:
        raise ValueError("--velocity-gated-min-vx must be non-negative")
    if args.prediction_horizon < 1:
        raise ValueError("--prediction-horizon must be at least 1")

    conditions_to_run = [part.strip() for part in str(args.conditions).split(",") if part.strip()]
    unknown_conditions = [name for name in conditions_to_run if name not in CONDITIONS]
    if unknown_conditions:
        raise ValueError(f"Unknown condition(s): {unknown_conditions}. Valid names: {list(CONDITIONS)}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(
        "delay_config "
        + json.dumps(
            {
                "delayed_delay_mode_override": args.delayed_delay_mode,
                "fbm_hurst": args.fbm_hurst,
                "markov_q": args.markov_q,
                "markov_symmetric_values": [0, 1, 2],
                "markov_symmetric_transition": {
                    "stay_probability": args.markov_q,
                    "other_value_probability_each": (1.0 - args.markov_q) / 2.0,
                },
                "use_corrected_action_conversion": args.use_corrected_action_conversion,
                "use_velocity_gated_penalty": args.use_velocity_gated_penalty,
                "lambda_penalty": args.lambda_penalty,
                "velocity_gated_min_vx": args.velocity_gated_min_vx,
                "prediction_horizon": args.prediction_horizon,
            },
            sort_keys=True,
        ),
        flush=True,
    )

    step_rows: list[dict[str, Any]] = []
    episodes: list[dict[str, Any]] = []
    seeds = [args.seed_start + i for i in range(args.seeds)]
    run_start = time.perf_counter()
    for seed in seeds:
        for condition in conditions_to_run:
            print(f"running seed={seed} condition={condition}", flush=True)
            episodes.append(
                run_episode(
                    condition,
                    seed,
                    args.steps,
                    args.warning_buffer,
                    step_rows,
                    delayed_delay_mode=args.delayed_delay_mode,
                    fbm_hurst=args.fbm_hurst,
                    markov_q=args.markov_q,
                    disturbance_mode=args.disturbance_mode,
                    thrust_decay_final=args.thrust_decay_final,
                    adversarial_effort=args.adversarial_effort,
                    velocity_boundary_weight=args.velocity_boundary_weight,
                    slew_rate_weight=args.slew_rate_weight,
                    use_corrected_action_conversion=args.use_corrected_action_conversion,
                    use_velocity_gated_penalty=args.use_velocity_gated_penalty,
                    lambda_penalty=args.lambda_penalty,
                    velocity_gated_min_vx=args.velocity_gated_min_vx,
                    prediction_horizon=args.prediction_horizon,
                )
            )

    step_csv = output_dir / "quadrotor_delay_ra_steps.csv"
    episode_csv = output_dir / "quadrotor_delay_ra_episodes.csv"
    summary_json = output_dir / "quadrotor_delay_ra_summary.json"
    manifest_json = output_dir / "quadrotor_delay_ra_manifest.json"

    with step_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(step_rows[0].keys()) if step_rows else ["seed"])
        writer.writeheader()
        writer.writerows(step_rows)

    with episode_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(episodes[0].keys()) if episodes else ["seed"])
        writer.writeheader()
        writer.writerows(episodes)

    aggregates = aggregate(episodes)
    plot_paths = write_plots(output_dir, aggregates)
    summary = {
        "aggregates": aggregates,
        "episodes": episodes,
        "plots": plot_paths,
        "delay_mode_override": args.delayed_delay_mode,
        "fbm_hurst": args.fbm_hurst,
        "markov_q": args.markov_q,
        "disturbance_mode": args.disturbance_mode,
        "thrust_decay_final": args.thrust_decay_final,
        "use_corrected_action_conversion": args.use_corrected_action_conversion,
        "adversarial_effort": args.adversarial_effort,
        "velocity_boundary_weight": args.velocity_boundary_weight,
        "slew_rate_weight": args.slew_rate_weight,
        "use_velocity_gated_penalty": args.use_velocity_gated_penalty,
        "lambda_penalty": args.lambda_penalty,
        "velocity_gated_min_vx": args.velocity_gated_min_vx,
        "prediction_horizon": args.prediction_horizon,
        "scope": "Matched A-E experiment with heuristic delay-blind versus delay-aware intervention stacks. D and E share candidates, score weights, and symbolic dynamics, but differ in rollout model, trigger, and sampled-delay use. This is Phase 1+ evidence, not a formal CBF/MPC shield.",
    }
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    manifest = {
        "script": str(Path(__file__).resolve()),
        "repo_root": str(REPO_ROOT),
        "seeds": seeds,
        "steps": args.steps,
        "warning_buffer": args.warning_buffer,
        "delayed_delay_mode_override": args.delayed_delay_mode,
        "fbm_hurst": args.fbm_hurst,
        "markov_q": args.markov_q,
        "conditions": {name: CONDITIONS[name] for name in conditions_to_run},
        "disturbance_mode": args.disturbance_mode,
        "thrust_decay_final": args.thrust_decay_final,
        "use_corrected_action_conversion": args.use_corrected_action_conversion,
        "adversarial_effort": args.adversarial_effort,
        "velocity_boundary_weight": args.velocity_boundary_weight,
        "slew_rate_weight": args.slew_rate_weight,
        "use_velocity_gated_penalty": args.use_velocity_gated_penalty,
        "lambda_penalty": args.lambda_penalty,
        "velocity_gated_min_vx": args.velocity_gated_min_vx,
        "prediction_horizon": args.prediction_horizon,
        "runtime_seconds": time.perf_counter() - run_start,
        "outputs": {
            "step_csv": str(step_csv),
            "episode_csv": str(episode_csv),
            "summary_json": str(summary_json),
            "manifest_json": str(manifest_json),
            "plots": plot_paths,
        },
    }
    manifest_json.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
