"""Per-step logging schema for the Phase 0 quadrotor delay harness."""

from __future__ import annotations

STEP_LOG_FIELDS = [
    "seed",
    "step",
    "condition",
    "observation",
    "state",
    "PPO_commanded",
    "delayed_applied",
    "delay_sample",
    "queue_state",
    "constraint_values",
    "violation",
    "normalized_violation_severity",
    "reward",
    "runtime_ms",
]

MECHANISM_OPTIONAL_FIELDS = [
    "queue_len_before",
    "queue_head_command_step_before",
    "queue_head_release_step_before",
    "queue_head_delay_sample_before",
    "candidate_delay_sample",
    "candidate_release_step",
    "applied_command_step",
    "applied_release_step",
    "applied_original_delay",
    "applied_action_age",
    "held_last_action_used",
    "warning",
    "intervention",
    "action_source",
    "selected_candidate",
    "monitor_commanded",
    "predicted_first_applied_action",
    "actual_applied_action",
    "applied_action_prediction_error_l2",
    "state_pred_error_l2",
    "state_pred_error_max",
    "selected_candidate_score",
    "ppo_candidate_score",
    "score_gap_selected_vs_ppo",
    "x_violation_contribution",
    "x_dot_violation_contribution",
    "theta_violation_contribution",
    "theta_dot_violation_contribution",
    "predicted_state_severity",
    "candidate_scores",
    "out_of_bounds",
]

REQUIRED_STEP_LOG_FIELDS = tuple(STEP_LOG_FIELDS)
OPTIONAL_STEP_LOG_FIELDS = tuple(MECHANISM_OPTIONAL_FIELDS)


def validate_step_log_row(row: dict) -> None:
    """Raise ``ValueError`` if a per-step log row is missing required fields."""

    missing = [field for field in REQUIRED_STEP_LOG_FIELDS if field not in row]
    if missing:
        raise ValueError(f"Missing required step log field(s): {missing}")
