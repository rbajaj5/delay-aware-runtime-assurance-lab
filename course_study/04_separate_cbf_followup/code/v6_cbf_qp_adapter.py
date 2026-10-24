#!/usr/bin/env python3
"""Runtime adapter for the preregistered V6 CBF-QP formal shield.

The frozen experiment harness is never edited. This module temporarily
installs a chooser with the same callback contract, then restores the
original chooser and condition table entry when an episode ends.
"""

from __future__ import annotations

import contextlib
import json
import math
import time
from pathlib import Path
from typing import Any, Iterator

import numpy as np


STATE_NAMES = ("x", "vx", "z", "vz", "theta", "theta_dot")
STATE_LOWER = np.array([-2.0, -1.0, 0.0, -1.0, -0.2, -1.5], dtype=float)
STATE_UPPER = np.array([2.0, 1.0, 2.0, 1.0, 0.2, 1.5], dtype=float)
BARRIER_NAMES = tuple(
    [f"{name}_lower" for name in STATE_NAMES]
    + [f"{name}_upper" for name in STATE_NAMES]
)
GAMMA = 0.20
CONSTRAINT_TOL = 1.0e-7
ACTIVE_TOL = 1.0e-6
TEMP_CONDITION = "_V6_CBF_QP_ADAPTER"


def compact_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def parse_json(value: Any) -> Any:
    if isinstance(value, str):
        return json.loads(value)
    return value


def parse_vector(value: Any) -> np.ndarray:
    return np.asarray(parse_json(value), dtype=float).reshape(-1)


def as_bool(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes"}


def barrier_margins(state: np.ndarray) -> np.ndarray:
    state = np.asarray(state, dtype=float).reshape(-1)
    return np.concatenate([state - STATE_LOWER, STATE_UPPER - state])


def margin_dict(values: np.ndarray) -> dict[str, float]:
    return {
        name: float(value)
        for name, value in zip(BARRIER_NAMES, np.asarray(values, dtype=float))
    }


class FormalShieldChooser:
    """Queue-blind or queue-compensated one-step CBF-QP chooser."""

    def __init__(
        self,
        exp: Any,
        phase0: Any,
        osqp_module: Any,
        variant: str,
    ) -> None:
        if variant not in {"F-blind", "F-comp"}:
            raise ValueError(f"Unknown formal variant: {variant}")
        if abs(float(phase0.GAMMA) - GAMMA) > 0.0:
            raise AssertionError(
                f"Phase-0 gamma differs: expected {GAMMA}, got {phase0.GAMMA}"
            )
        self.exp = exp
        self.phase0 = phase0
        self.osqp_module = osqp_module
        self.variant = variant
        self._cached_step: int | None = None
        self._cached_result: tuple[np.ndarray, str, float, list[dict[str, Any]]] | None = None
        self.solve_count = 0

    def _queue_anchor(
        self,
        env: Any,
        state: np.ndarray,
        ppo_action: np.ndarray,
        queue_state: list[dict[str, Any]],
        current_step: int,
        delay_sample: int,
        last_applied_action: np.ndarray,
    ) -> tuple[np.ndarray, int, list[list[float]], float]:
        if self.variant == "F-blind":
            return state.copy(), 0, [], 0.0

        offset = self.phase0.first_controllable_offset(
            self.exp,
            queue_state,
            current_step,
            delay_sample,
            last_applied_action,
        )
        actions = self.exp.queue_actions_for_prediction(
            queue_state=queue_state,
            current_step=current_step,
            candidate=ppo_action,
            delay_sample=delay_sample,
            last_applied_action=last_applied_action,
            prediction_horizon=max(offset + 1, 1),
        )
        anchor = state.copy()
        started = time.perf_counter_ns()
        for action in actions[:offset]:
            anchor = self.exp.predict_next_state(env, anchor, action)
        elapsed_ms = (time.perf_counter_ns() - started) / 1.0e6
        prefix = [np.asarray(action, dtype=float).tolist() for action in actions[:offset]]
        return anchor, int(offset), prefix, float(elapsed_ms)

    def __call__(self, *args: Any, **kwargs: Any) -> tuple[np.ndarray, str, float, list[dict[str, Any]]]:
        if args:
            raise AssertionError("Formal chooser requires keyword invocation")
        env = kwargs["env"]
        state = np.asarray(kwargs["state"], dtype=float).reshape(-1)
        ppo_action = np.asarray(kwargs["ppo_action"], dtype=float).reshape(-1)
        queue_state = list(kwargs["queue_state"])
        current_step = int(kwargs["current_step"])
        delay_sample = int(kwargs["delay_sample"])
        last_applied = kwargs.get("last_applied_action")
        if last_applied is None:
            raise AssertionError(
                "Formal adapter requires held-action-corrected queue state"
            )
        last_applied = np.asarray(last_applied, dtype=float).reshape(-1)

        if self._cached_step == current_step and self._cached_result is not None:
            return self._cached_result

        selection_started = time.perf_counter_ns()
        anchor, offset, fixed_prefix, propagation_ms = self._queue_anchor(
            env,
            state,
            ppo_action,
            queue_state,
            current_step,
            delay_sample,
            last_applied,
        )
        action_low, action_high = env.physical_action_bounds
        u_lower = np.asarray(action_low, dtype=float).reshape(-1)
        u_upper = np.asarray(action_high, dtype=float).reshape(-1)
        u_ppo = self.exp.denormalized_clipped_action(env, ppo_action)

        x_next_ref, action_jacobian, linearization_ms = (
            self.phase0.linearize_discrete_action(
                env,
                anchor,
                u_ppo,
                u_lower,
                u_upper,
            )
        )
        g, b, h_current, h_required = self.phase0.barrier_system(
            anchor,
            x_next_ref,
            action_jacobian,
            u_ppo,
        )
        hard = self.phase0.solve_hard_qp(
            self.osqp_module,
            g,
            b,
            u_ppo,
            u_lower,
            u_upper,
        )

        fallback_used = not bool(hard["feasible"])
        fallback_wall_ms = 0.0
        fallback_iterations = 0
        fallback_status = "not_used"
        if hard["feasible"]:
            selected_physical = np.asarray(hard["u"], dtype=float)
            slack = np.zeros(len(BARRIER_NAMES), dtype=float)
            selected_name = "cbf_qp"
        else:
            relaxed = self.phase0.solve_slack_penalized_fallback(
                self.osqp_module,
                g,
                b,
                u_ppo,
                u_lower,
                u_upper,
            )
            selected_physical = np.asarray(relaxed["u"], dtype=float)
            slack = np.asarray(relaxed["slack"], dtype=float)
            selected_name = "cbf_slack_fallback"
            fallback_wall_ms = float(relaxed["solve_wall_ms"])
            fallback_iterations = int(relaxed["solver_iterations"])
            fallback_status = str(relaxed["status"])

        selected_normalized = np.asarray(
            env.normalize_action(selected_physical),
            dtype=float,
        ).reshape(-1)
        predicted_linear = x_next_ref + action_jacobian @ (
            selected_physical - u_ppo
        )
        predicted_nonlinear = self.phase0.symbolic_step_physical(
            env,
            anchor,
            selected_physical,
        )
        h_next_nonlinear = barrier_margins(predicted_nonlinear)
        nonlinear_residual = h_next_nonlinear - h_required
        linearized_residual = g @ selected_physical - b
        active_constraints = [
            BARRIER_NAMES[index]
            for index in range(len(BARRIER_NAMES))
            if linearized_residual[index] <= ACTIVE_TOL
            or slack[index] > ACTIVE_TOL
        ]
        total_wall_ms = (time.perf_counter_ns() - selection_started) / 1.0e6
        action_delta = float(np.linalg.norm(selected_normalized - np.clip(ppo_action, -1.0, 1.0)))
        trigger_value = max(
            action_delta,
            float(np.sum(slack)) if fallback_used else 0.0,
        )
        if trigger_value <= 1.0e-12:
            trigger_value = 0.0

        payload: dict[str, Any] = {
            "name": selected_name,
            "action": selected_normalized.tolist(),
            "score": trigger_value,
            "score_without_boundary": trigger_value,
            "score_without_slew": trigger_value,
            "score_without_velocity_gated_penalty": trigger_value,
            "predicted_state_severity": trigger_value,
            "predicted_tracking_rmse": 0.0,
            "candidate_boundary_penalty": 0.0,
            "candidate_slew_rate_penalty": 0.0,
            "candidate_slew_action_delta_norm": 0.0,
            "candidate_velocity_gated_penalty": 0.0,
            "candidate_velocity_gated_energy_derivative_proxy": 0.0,
            "candidate_velocity_gated_penalty_active": False,
            "candidate_velocity_gated_terminal_vx": float(predicted_nonlinear[1]),
            "candidate_velocity_gated_terminal_theta": float(predicted_nonlinear[4]),
            "formal_variant": self.variant,
            "formal_gamma": GAMMA,
            "formal_step": current_step,
            "formal_queue_before": queue_state,
            "formal_queue_len_before": len(queue_state),
            "formal_delay_sample": delay_sample,
            "formal_last_applied_normalized": last_applied.tolist(),
            "formal_first_controllable_offset_steps": offset,
            "formal_target_realized_step": current_step + offset,
            "formal_fixed_queue_prefix_actions_normalized": fixed_prefix,
            "formal_state_at_decision": state.tolist(),
            "formal_barrier_anchor_state": anchor.tolist(),
            "formal_ppo_action_normalized": ppo_action.tolist(),
            "formal_ppo_action_physical": u_ppo.tolist(),
            "formal_qp_action_normalized": selected_normalized.tolist(),
            "formal_qp_action_physical": selected_physical.tolist(),
            "formal_hard_qp_feasible": bool(hard["feasible"]),
            "formal_hard_qp_status": str(hard["status"]),
            "formal_hard_qp_status_val": int(hard["status_val"]),
            "formal_fallback_used": fallback_used,
            "formal_fallback_disposition": (
                "slack_penalized_highs_lp" if fallback_used else "none"
            ),
            "formal_fallback_status": fallback_status,
            "formal_slack_used": bool(np.max(slack) > CONSTRAINT_TOL),
            "formal_slack_sum": float(np.sum(slack)),
            "formal_slack_max": float(np.max(slack)),
            "formal_slack_by_constraint": margin_dict(slack),
            "formal_active_constraints": active_constraints,
            "formal_h_current": margin_dict(h_current),
            "formal_h_required": margin_dict(h_required),
            "formal_predicted_next_state_linearized": predicted_linear.tolist(),
            "formal_predicted_next_state_nonlinear": predicted_nonlinear.tolist(),
            "formal_predicted_h_next_nonlinear": margin_dict(h_next_nonlinear),
            "formal_nonlinear_dcbf_residual": margin_dict(nonlinear_residual),
            "formal_linearized_barrier_residual": margin_dict(linearized_residual),
            "formal_minimum_nonlinear_dcbf_residual": float(np.min(nonlinear_residual)),
            "formal_minimum_linearized_barrier_residual": float(np.min(linearized_residual)),
            "formal_nonlinear_dcbf_satisfied_without_slack": bool(
                np.min(nonlinear_residual) >= -CONSTRAINT_TOL
            ),
            "formal_claimed_safe_model": bool(hard["feasible"]),
            "formal_linearization_error_linf": float(
                np.max(np.abs(predicted_nonlinear - predicted_linear))
            ),
            "formal_action_deviation_from_ppo_normalized_l2": action_delta,
            "formal_queue_propagation_ms": propagation_ms,
            "formal_linearization_ms": float(linearization_ms),
            "formal_hard_qp_solve_wall_ms": float(hard["solve_wall_ms"]),
            "formal_hard_qp_solver_run_time_ms": float(hard["solver_run_time_ms"]),
            "formal_hard_qp_iterations": int(hard["solver_iterations"]),
            "formal_fallback_solve_wall_ms": fallback_wall_ms,
            "formal_fallback_iterations": fallback_iterations,
            "formal_total_selection_wall_ms": float(total_wall_ms),
        }
        result = (
            selected_normalized,
            selected_name,
            trigger_value,
            [payload],
        )
        self._cached_step = current_step
        self._cached_result = result
        self.solve_count += 1
        return result


@contextlib.contextmanager
def formal_adapter_context(
    exp: Any,
    phase0: Any,
    osqp_module: Any,
    variant: str,
) -> Iterator[tuple[str, FormalShieldChooser]]:
    """Install the formal chooser without modifying the frozen source file."""

    original_chooser = exp.choose_delay_aware_action
    previous_condition = exp.CONDITIONS.get(TEMP_CONDITION)
    chooser = FormalShieldChooser(exp, phase0, osqp_module, variant)
    exp.CONDITIONS[TEMP_CONDITION] = {
        "delay": "uniform_0_2",
        "monitor": "delay_aware",
        "intervention": f"delay_aware_cbf_qp_{variant.lower()}",
        "held_action_corrected": True,
        "boundary_penalty_target": "none",
        "boundary_penalty_anti": False,
    }
    exp.choose_delay_aware_action = chooser
    try:
        yield TEMP_CONDITION, chooser
    finally:
        exp.choose_delay_aware_action = original_chooser
        if previous_condition is None:
            del exp.CONDITIONS[TEMP_CONDITION]
        else:
            exp.CONDITIONS[TEMP_CONDITION] = previous_condition


def enrich_formal_step_rows(rows: list[dict[str, Any]]) -> None:
    """Flatten formal payloads and align model predictions to realized states."""

    lookup = {int(row["step"]): row for row in rows}
    for row in rows:
        raw_scores = row.pop("candidate_scores", "[]")
        scores = parse_json(raw_scores)
        if len(scores) != 1:
            raise AssertionError(
                f"Formal step has {len(scores)} score rows at seed={row['seed']} "
                f"step={row['step']}"
            )
        payload = dict(scores[0])
        for key, value in payload.items():
            if not key.startswith("formal_"):
                continue
            row[key] = (
                compact_json(value)
                if isinstance(value, (dict, list, tuple))
                else value
            )

        target_step = int(payload["formal_target_realized_step"])
        target_row = lookup.get(target_step)
        if target_row is None:
            row["formal_realized_disposition"] = "unavailable_episode_terminated"
            row["formal_realized_state"] = ""
            row["formal_realized_h"] = ""
            row["formal_prediction_error_l2"] = ""
            row["formal_prediction_error_linf"] = ""
            row["formal_predicted_safe_realized_unsafe"] = ""
            continue

        realized = parse_vector(target_row["state"])
        predicted = np.asarray(
            payload["formal_predicted_next_state_nonlinear"],
            dtype=float,
        )
        delta = realized - predicted
        realized_h = barrier_margins(realized)
        row["formal_realized_disposition"] = "matched_target_step"
        row["formal_realized_state"] = compact_json(realized.tolist())
        row["formal_realized_h"] = compact_json(margin_dict(realized_h))
        row["formal_prediction_error_l2"] = float(np.linalg.norm(delta))
        row["formal_prediction_error_linf"] = float(np.max(np.abs(delta)))
        row["formal_predicted_safe_realized_unsafe"] = bool(
            payload["formal_claimed_safe_model"]
            and np.min(realized_h) < -CONSTRAINT_TOL
        )


def formal_episode_summary(
    variant: str,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    hard_infeasible = sum(
        not as_bool(row["formal_hard_qp_feasible"]) for row in rows
    )
    fallback = sum(as_bool(row["formal_fallback_used"]) for row in rows)
    claimed_safe = sum(as_bool(row["formal_claimed_safe_model"]) for row in rows)
    unavailable = sum(
        row["formal_realized_disposition"] != "matched_target_step"
        for row in rows
    )
    mismatch = sum(
        as_bool(row["formal_predicted_safe_realized_unsafe"])
        for row in rows
        if row["formal_predicted_safe_realized_unsafe"] != ""
    )
    return {
        "formal_variant": variant,
        "formal_gamma": GAMMA,
        "formal_solver_invocations": len(rows),
        "formal_hard_infeasible_count": hard_infeasible,
        "formal_fallback_count": fallback,
        "formal_claimed_safe_count": claimed_safe,
        "formal_realized_comparison_unavailable_count": unavailable,
        "formal_predicted_safe_realized_unsafe_count": mismatch,
        "formal_mean_total_selection_wall_ms": float(
            np.mean([float(row["formal_total_selection_wall_ms"]) for row in rows])
        ),
        "formal_max_total_selection_wall_ms": float(
            np.max([float(row["formal_total_selection_wall_ms"]) for row in rows])
        ),
    }

