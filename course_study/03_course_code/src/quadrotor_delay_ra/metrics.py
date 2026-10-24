"""Pure safety-envelope metrics for the Phase 0 quadrotor harness."""

from __future__ import annotations

import numpy as np


def signed_normalized_margin(
    values: np.ndarray | list[float] | tuple[float, ...],
    lower_bounds: np.ndarray | list[float] | tuple[float, ...],
    upper_bounds: np.ndarray | list[float] | tuple[float, ...],
) -> np.ndarray:
    """Return signed normalized bound margins for each bounded component.

    Negative or zero values are inside the envelope. Positive values are
    normalized violations. For each component ``x_i``:

    ``max((lower_i - x_i) / scale_i, (x_i - upper_i) / scale_i)``

    where ``scale_i = max(abs(lower_i), abs(upper_i), upper_i - lower_i, 1)``.
    """

    vals = np.asarray(values, dtype=float)
    lows = np.asarray(lower_bounds, dtype=float)
    highs = np.asarray(upper_bounds, dtype=float)
    if vals.shape != lows.shape or vals.shape != highs.shape:
        raise ValueError("values, lower_bounds, and upper_bounds must have identical shapes")
    scale = np.maximum.reduce([np.abs(lows), np.abs(highs), highs - lows, np.ones_like(vals)])
    lower_violation = (lows - vals) / scale
    upper_violation = (vals - highs) / scale
    return np.maximum(lower_violation, upper_violation)


def normalized_positive_violation(
    values: np.ndarray | list[float] | tuple[float, ...],
    lower_bounds: np.ndarray | list[float] | tuple[float, ...],
    upper_bounds: np.ndarray | list[float] | tuple[float, ...],
) -> np.ndarray:
    """Return per-component normalized positive violations."""

    return np.maximum(signed_normalized_margin(values, lower_bounds, upper_bounds), 0.0)


def normalized_positive_violation_from_constraint_values(
    constraint_values: np.ndarray | list[float] | tuple[float, ...],
    normalization_scales: np.ndarray | list[float] | tuple[float, ...],
) -> np.ndarray:
    """Normalize positive safe-control-gym ``g(x) <= 0`` constraint values."""

    values = np.asarray(constraint_values, dtype=float)
    scales = np.asarray(normalization_scales, dtype=float)
    if values.shape != scales.shape:
        raise ValueError("constraint_values and normalization_scales must have identical shapes")
    safe_scales = np.maximum(scales, 1e-12)
    return np.maximum(values, 0.0) / safe_scales


def per_step_normalized_violation_severity(
    values: np.ndarray | list[float] | tuple[float, ...],
    lower_bounds: np.ndarray | list[float] | tuple[float, ...],
    upper_bounds: np.ndarray | list[float] | tuple[float, ...],
) -> float:
    """Return the per-step normalized violation severity.

    Phase 0 uses the V5 proposal endpoint in its simplest envelope form: sum of
    positive normalized state/input bound violations at a step.
    """

    return float(np.sum(normalized_positive_violation(values, lower_bounds, upper_bounds)))


def per_step_normalized_violation_severity_from_constraint_values(
    constraint_values: np.ndarray | list[float] | tuple[float, ...],
    normalization_scales: np.ndarray | list[float] | tuple[float, ...],
) -> float:
    """Return normalized severity from safe-control-gym constraint values."""

    return float(np.sum(normalized_positive_violation_from_constraint_values(constraint_values, normalization_scales)))


def cumulative_normalized_violation_severity(per_step_severities: np.ndarray | list[float]) -> float:
    """Return cumulative normalized violation severity across a rollout."""

    return float(np.sum(np.asarray(per_step_severities, dtype=float)))


def violating_step_count(per_step_severities: np.ndarray | list[float], atol: float = 1e-12) -> int:
    """Count steps with any positive normalized violation severity."""

    return int(np.sum(np.asarray(per_step_severities, dtype=float) > atol))
