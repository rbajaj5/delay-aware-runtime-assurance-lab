"""Mechanism diagnostics for delay-aware quadrotor RA failures.

The project-level failure mechanism is a dwell process, not a generic
branching-ratio congestion story: E-condition OOB episodes enter a sustained
forward-velocity (`x_dot`) upper-bound dwell while the delay queue is saturated
and the short-horizon scorer remains optimistic about later recovery.

This module keeps that vocabulary reusable across analysis scripts, paper
audits, and lightweight simulations.
"""

from __future__ import annotations

import ast
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


CONDITION_D = "D_delay_blind_intervention"
CONDITION_E = "E_delay_aware_intervention"
CONDITION_H = "E_held_corrected_intervention"

STATE_NAMES = ("x", "x_dot", "z", "z_dot", "theta", "theta_dot")
BOUNDS: dict[str, tuple[float, float]] = {
    "x": (-2.0, 2.0),
    "x_dot": (-1.0, 1.0),
    "z": (0.0, 2.0),
    "z_dot": (-1.0, 1.0),
    "theta": (-0.2, 0.2),
    "theta_dot": (-1.5, 1.5),
}

MECHANISM_COLUMNS = (
    "x_dot",
    "x_dot_upper_margin",
    "x_dot_dwell_active",
    "x_dot_dwell_run_id",
    "x_dot_dwell_age",
    "queue_len_mechanism",
    "queue_saturated",
    "dominant_violation_component",
)


@dataclass(frozen=True)
class RunLayout:
    """Resolved on-disk layout for one experiment report directory."""

    run_dir: Path
    root_steps_csv: Path | None
    chunk_steps_csvs: tuple[Path, ...]
    episodes_csv: Path | None
    oob_trace_csv: Path | None

    @property
    def has_full_steps(self) -> bool:
        return self.root_steps_csv is not None or bool(self.chunk_steps_csvs)


def discover_run_layout(run_dir: str | Path) -> RunLayout:
    """Discover standard and P0/chunked quadrotor report artifacts."""

    root = Path(run_dir)
    root_steps = root / "quadrotor_delay_ra_steps.csv"
    root_eps = root / "quadrotor_delay_ra_episodes.csv"
    p0_eps = root / "p0_combined_episodes.csv"
    p0_trace = root / "p0_oob_decision_traces_E_H.csv"
    chunk_steps = tuple(sorted(root.glob("chunk_*_seed*_n*/quadrotor_delay_ra_steps.csv")))
    return RunLayout(
        run_dir=root,
        root_steps_csv=root_steps if root_steps.exists() else None,
        chunk_steps_csvs=chunk_steps,
        episodes_csv=root_eps if root_eps.exists() else (p0_eps if p0_eps.exists() else None),
        oob_trace_csv=p0_trace if p0_trace.exists() else None,
    )


def default_run_dirs(root: str | Path) -> list[Path]:
    """Known report directories, ordered from newest endpoint run to fallback."""

    base = Path(root)
    return [
        base / "reports" / "quadrotor_delay_ra_p0_uniform_1000",
        base / "reports" / "quadrotor_delay_ra_experiment_phase123_20260629_final50",
    ]


def find_default_run_dir(root: str | Path) -> Path:
    for run_dir in default_run_dirs(root):
        layout = discover_run_layout(run_dir)
        if layout.episodes_csv or layout.oob_trace_csv or layout.has_full_steps:
            return run_dir
    raise FileNotFoundError("No quadrotor delay-RA report directory found")


def parse_literal(value: Any, default: Any = None) -> Any:
    """Parse JSON/Python literal cells emitted by the CSV logger."""

    if value is None:
        return default
    if isinstance(value, float) and math.isnan(value):
        return default
    if isinstance(value, (list, tuple, dict)):
        return value
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped or stripped.lower() == "nan":
            return default
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            try:
                return ast.literal_eval(stripped)
            except (ValueError, SyntaxError):
                return default
    return value


def parse_vector(value: Any, *, min_len: int = 0) -> np.ndarray:
    parsed = parse_literal(value, default=[])
    arr = np.asarray(parsed, dtype=float).reshape(-1)
    if arr.size < min_len:
        out = np.full((min_len,), np.nan, dtype=float)
        out[: arr.size] = arr
        return out
    return arr


def queue_len_from_cell(value: Any) -> int:
    parsed = parse_literal(value, default=[])
    if isinstance(parsed, list):
        return len(parsed)
    return 0


def read_episode_table(layout_or_dir: RunLayout | str | Path) -> pd.DataFrame:
    layout = layout_or_dir if isinstance(layout_or_dir, RunLayout) else discover_run_layout(layout_or_dir)
    if layout.episodes_csv is None:
        raise FileNotFoundError(f"No episode table found under {layout.run_dir}")
    return pd.read_csv(layout.episodes_csv)


def read_oob_trace(layout_or_dir: RunLayout | str | Path) -> pd.DataFrame:
    layout = layout_or_dir if isinstance(layout_or_dir, RunLayout) else discover_run_layout(layout_or_dir)
    if layout.oob_trace_csv is None:
        raise FileNotFoundError(f"No P0 OOB trace table found under {layout.run_dir}")
    return pd.read_csv(layout.oob_trace_csv)


def find_chunk_for_seed(layout_or_dir: RunLayout | str | Path, seed: int) -> Path | None:
    """Find the P0 chunk CSV whose directory name covers a seed."""

    layout = layout_or_dir if isinstance(layout_or_dir, RunLayout) else discover_run_layout(layout_or_dir)
    pat = re.compile(r"chunk_\d+_seed(?P<start>\d+)_n(?P<n>\d+)")
    for csv_path in layout.chunk_steps_csvs:
        match = pat.search(str(csv_path.parent.name))
        if not match:
            continue
        start = int(match.group("start"))
        n = int(match.group("n"))
        if start <= int(seed) < start + n:
            return csv_path
    return None


def read_steps(
    layout_or_dir: RunLayout | str | Path,
    *,
    conditions: Sequence[str] | None = None,
    seeds: Sequence[int] | None = None,
    usecols: Sequence[str] | None = None,
    max_chunks: int | None = None,
) -> pd.DataFrame:
    """Read step rows from a standard run or chunked P0 run.

    Chunked P0 runs are large. Pass `seeds` when possible so only the relevant
    chunk is read.
    """

    layout = layout_or_dir if isinstance(layout_or_dir, RunLayout) else discover_run_layout(layout_or_dir)
    csvs: list[Path] = []
    if layout.root_steps_csv is not None:
        csvs = [layout.root_steps_csv]
    elif seeds:
        seen: set[Path] = set()
        for seed in seeds:
            chunk = find_chunk_for_seed(layout, int(seed))
            if chunk is not None and chunk not in seen:
                csvs.append(chunk)
                seen.add(chunk)
    else:
        csvs = list(layout.chunk_steps_csvs[:max_chunks])
    if not csvs:
        raise FileNotFoundError(f"No step CSVs found under {layout.run_dir}")

    frames: list[pd.DataFrame] = []
    for csv_path in csvs:
        frame = pd.read_csv(csv_path, usecols=usecols)
        if conditions is not None and "condition" in frame.columns:
            frame = frame[frame["condition"].isin(list(conditions))]
        if seeds is not None and "seed" in frame.columns:
            frame = frame[frame["seed"].isin([int(seed) for seed in seeds])]
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).sort_values(["condition", "seed", "step"]).reset_index(drop=True)


def add_mechanism_columns(df: pd.DataFrame, *, saturation_threshold: int = 2) -> pd.DataFrame:
    """Add reusable dwell, queue, and component-dominance columns."""

    out = df.copy()
    if "state" in out.columns:
        states = np.vstack(out["state"].map(lambda value: parse_vector(value, min_len=6)).to_numpy())
        for idx, name in enumerate(STATE_NAMES):
            out[name] = states[:, idx]
    if "x_dot" not in out.columns:
        raise KeyError("Expected `state` or `x_dot` to compute the dwell mechanism")

    out["x_dot_upper_margin"] = out["x_dot"].astype(float) - BOUNDS["x_dot"][1]
    out["x_dot_dwell_active"] = out["x_dot_upper_margin"] > 0.0

    if "queue_len_before" in out.columns:
        out["queue_len_mechanism"] = pd.to_numeric(out["queue_len_before"], errors="coerce").fillna(0).astype(int)
    elif "queue_len" in out.columns:
        out["queue_len_mechanism"] = pd.to_numeric(out["queue_len"], errors="coerce").fillna(0).astype(int)
    elif "queue_state" in out.columns:
        out["queue_len_mechanism"] = out["queue_state"].map(queue_len_from_cell)
    else:
        out["queue_len_mechanism"] = 0
    out["queue_saturated"] = out["queue_len_mechanism"] >= int(saturation_threshold)

    component_cols = [
        "x_violation_contribution",
        "x_dot_violation_contribution",
        "theta_violation_contribution",
        "theta_dot_violation_contribution",
    ]
    if all(col in out.columns for col in component_cols):
        contrib = out[component_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
        out["dominant_violation_component"] = contrib.idxmax(axis=1).str.replace("_violation_contribution", "", regex=False)
        out.loc[contrib.max(axis=1) <= 0.0, "dominant_violation_component"] = "none"
    else:
        out["dominant_violation_component"] = np.where(out["x_dot_dwell_active"], "x_dot", "none")

    out = annotate_dwell_runs(out)
    return out


def annotate_dwell_runs(df: pd.DataFrame) -> pd.DataFrame:
    """Annotate active x_dot dwell runs per condition/seed."""

    out = df.sort_values([c for c in ("condition", "seed", "step") if c in df.columns]).copy()
    out["x_dot_dwell_run_id"] = -1
    out["x_dot_dwell_age"] = 0
    group_cols = [c for c in ("condition", "seed") if c in out.columns]
    groups = out.groupby(group_cols, sort=False) if group_cols else [(None, out)]
    for _, group in groups:
        run_id = -1
        age = 0
        prev_active = False
        for idx, active in group["x_dot_dwell_active"].astype(bool).items():
            if active and not prev_active:
                run_id += 1
                age = 1
            elif active:
                age += 1
            else:
                age = 0
            if active:
                out.at[idx, "x_dot_dwell_run_id"] = run_id
                out.at[idx, "x_dot_dwell_age"] = age
            prev_active = bool(active)
    return out


def discover_oob_seeds(steps: pd.DataFrame | None = None, episodes: pd.DataFrame | None = None) -> dict[str, list[int]]:
    """Return condition -> seeds with OOB termination/rows."""

    out: dict[str, list[int]] = {}
    source = episodes if episodes is not None and not episodes.empty else steps
    if source is None or source.empty:
        return out
    for condition, group in source.groupby("condition"):
        if "done_reason" in group.columns:
            mask = group["done_reason"].astype(str).str.contains("out_of_bounds|OOB", case=False, na=False)
        elif "out_of_bounds" in group.columns:
            mask = group["out_of_bounds"].astype(bool)
        else:
            mask = pd.Series(False, index=group.index)
        out[str(condition)] = sorted(group.loc[mask, "seed"].astype(int).unique().tolist())
    return out


def summarize_oob_trace(trace: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summarize the P0 decision-trace rows used in the paper mechanism."""

    if trace.empty:
        return pd.DataFrame(), pd.DataFrame()
    df = add_mechanism_columns(trace)
    rows: list[dict[str, Any]] = []
    group_cols = ["condition"]
    if "trace_role" in df.columns:
        group_cols.append("trace_role")
    for keys, group in df.groupby(group_cols):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys))
        row.update(
            n_rows=int(len(group)),
            n_seeds=int(group["seed"].nunique()) if "seed" in group.columns else 0,
            intervention_rate=float(group.get("intervention", pd.Series(False, index=group.index)).astype(bool).mean()),
            queue_saturated_rate=float(group["queue_saturated"].mean()),
            mean_queue_len=float(group["queue_len_mechanism"].mean()),
            mean_xdot_upper_margin=float(group["x_dot_upper_margin"].mean()),
            xdot_active_rate=float(group["x_dot_dwell_active"].mean()),
            xdot_dominant_rate=float((group["dominant_violation_component"] == "x_dot").mean()),
            mean_predicted_severity=float(pd.to_numeric(group.get("predicted_state_severity"), errors="coerce").mean())
            if "predicted_state_severity" in group
            else math.nan,
            mean_realized_severity=float(pd.to_numeric(group.get("normalized_violation_severity"), errors="coerce").mean())
            if "normalized_violation_severity" in group
            else math.nan,
            mean_applied_action_error=float(pd.to_numeric(group.get("applied_action_prediction_error_l2"), errors="coerce").mean())
            if "applied_action_prediction_error_l2" in group
            else math.nan,
            mean_state_prediction_error=float(pd.to_numeric(group.get("state_pred_error_l2"), errors="coerce").mean())
            if "state_pred_error_l2" in group
            else math.nan,
            mean_selected_score_gap_vs_ppo=float(pd.to_numeric(group.get("score_gap_selected_vs_ppo"), errors="coerce").mean())
            if "score_gap_selected_vs_ppo" in group
            else math.nan,
        )
        rows.append(row)
    summary = pd.DataFrame(rows)
    hist = candidate_histogram(df, group_cols=group_cols)
    return summary, hist


def candidate_histogram(df: pd.DataFrame, *, group_cols: Sequence[str] = ("condition",)) -> pd.DataFrame:
    candidate_col = "selected_candidate" if "selected_candidate" in df.columns else "chosen_candidate"
    if candidate_col not in df.columns:
        return pd.DataFrame()
    sub = df.copy()
    sub[candidate_col] = sub[candidate_col].fillna("ppo")
    hist = (
        sub.groupby([*group_cols, candidate_col])
        .size()
        .rename("count")
        .reset_index()
        .rename(columns={candidate_col: "candidate"})
    )
    totals = hist.groupby(list(group_cols))["count"].transform("sum")
    hist["fraction"] = hist["count"] / totals
    return hist.sort_values([*group_cols, "count"], ascending=[True] * len(group_cols) + [False])


def rollout_calibration(steps: pd.DataFrame, *, horizon: int = 5) -> pd.DataFrame:
    """Compare logged predicted severity to realized future severity.

    Positive `signed_error` is pessimism. Negative `signed_error` is optimism.
    """

    if steps.empty:
        return pd.DataFrame()
    required = {"seed", "condition", "step", "intervention", "normalized_violation_severity", "predicted_state_severity"}
    missing = sorted(required - set(steps.columns))
    if missing:
        raise KeyError(f"Missing calibration columns: {missing}")
    rows: list[dict[str, Any]] = []
    for (condition, seed), group in steps.groupby(["condition", "seed"]):
        g = group.sort_values("step").reset_index(drop=True)
        severity = pd.to_numeric(g["normalized_violation_severity"], errors="coerce").fillna(0.0).to_numpy()
        predicted = pd.to_numeric(g["predicted_state_severity"], errors="coerce").to_numpy()
        intervention = g["intervention"].astype(bool).to_numpy()
        for pos in np.flatnonzero(intervention):
            if np.isnan(predicted[pos]):
                continue
            end = min(len(g), pos + 1 + int(horizon))
            realized = float(np.sum(severity[pos + 1 : end]))
            signed_error = float(predicted[pos] - realized)
            rows.append(
                {
                    "condition": condition,
                    "seed": int(seed),
                    "step": int(g.loc[pos, "step"]),
                    "predicted": float(predicted[pos]),
                    "realized_future": realized,
                    "signed_error": signed_error,
                    "optimism": max(0.0, -signed_error),
                    "horizon_truncated": bool(end - (pos + 1) < int(horizon)),
                }
            )
    return pd.DataFrame(rows)


def summarize_rollout_calibration(calibration: pd.DataFrame, oob_seeds: Mapping[str, Sequence[int]] | None = None) -> pd.DataFrame:
    if calibration.empty:
        return pd.DataFrame()
    cal = calibration.copy()
    if oob_seeds:
        cal["oob_seed"] = [
            int(row.seed) in set(int(seed) for seed in oob_seeds.get(str(row.condition), []))
            for row in cal.itertuples()
        ]
        group_cols = ["condition", "oob_seed"]
    else:
        group_cols = ["condition"]
    return (
        cal.groupby(group_cols)
        .agg(
            n_interventions=("signed_error", "size"),
            mean_signed_error=("signed_error", "mean"),
            median_signed_error=("signed_error", "median"),
            mean_optimism=("optimism", "mean"),
            frac_optimistic=("optimism", lambda s: float((s > 0).mean())),
            mean_predicted=("predicted", "mean"),
            mean_realized_future=("realized_future", "mean"),
        )
        .reset_index()
    )


def dwell_summary(steps_or_trace: pd.DataFrame) -> pd.DataFrame:
    if steps_or_trace.empty:
        return pd.DataFrame()
    df = add_mechanism_columns(steps_or_trace)
    rows: list[dict[str, Any]] = []
    for (condition, seed), group in df.groupby(["condition", "seed"]):
        active = group[group["x_dot_dwell_active"]]
        rows.append(
            {
                "condition": condition,
                "seed": int(seed),
                "any_xdot_dwell": bool(not active.empty),
                "first_xdot_dwell_step": int(active["step"].min()) if not active.empty and "step" in active.columns else math.nan,
                "max_xdot_dwell_age": int(group["x_dot_dwell_age"].max()),
                "xdot_active_steps": int(group["x_dot_dwell_active"].sum()),
                "max_xdot_upper_margin": float(group["x_dot_upper_margin"].max()),
                "queue_saturated_during_xdot_rate": float(active["queue_saturated"].mean()) if not active.empty else 0.0,
            }
        )
    return pd.DataFrame(rows)


def seed_mechanism_trace(
    steps: pd.DataFrame,
    *,
    seed: int = 1373,
    condition: str = CONDITION_E,
    step_min: int = 10,
    step_max: int = 90,
) -> pd.DataFrame:
    df = add_mechanism_columns(steps)
    sub = df[
        (df["seed"].astype(int) == int(seed))
        & (df["condition"] == condition)
        & (df["step"].astype(int).between(int(step_min), int(step_max)))
    ].copy()
    if sub.empty:
        return pd.DataFrame()
    cols = [
        "seed",
        "condition",
        "step",
        "action_source",
        "selected_candidate",
        "intervention",
        "queue_len_mechanism",
        "queue_saturated",
        "held_last_action_used",
        "x_dot",
        "x_dot_upper_margin",
        "x_dot_dwell_age",
        "theta_dot",
        "normalized_violation_severity",
        "predicted_state_severity",
        "applied_action_prediction_error_l2",
        "dominant_violation_component",
    ]
    return sub[[col for col in cols if col in sub.columns]]


def simulation_rows(steps: pd.DataFrame, *, seed: int = 1373, conditions: Sequence[str] = (CONDITION_D, CONDITION_E, CONDITION_H)) -> list[dict[str, Any]]:
    """Build a compact JSON-friendly trace for the browser simulation."""

    df = add_mechanism_columns(steps)
    rows: list[dict[str, Any]] = []
    for row in df[df["seed"].astype(int) == int(seed)].sort_values(["condition", "step"]).itertuples(index=False):
        rec = row._asdict()
        condition = str(rec.get("condition"))
        if condition not in conditions:
            continue
        state = parse_vector(rec.get("state"), min_len=6)
        rows.append(
            {
                "seed": int(rec.get("seed")),
                "condition": condition,
                "step": int(rec.get("step")),
                "x": float(state[0]),
                "x_dot": float(state[1]),
                "z": float(state[2]),
                "z_dot": float(state[3]),
                "theta": float(state[4]),
                "theta_dot": float(state[5]),
                "severity": float(rec.get("normalized_violation_severity", 0.0) or 0.0),
                "predicted_severity": _safe_float(rec.get("predicted_state_severity")),
                "queue_len": int(rec.get("queue_len_mechanism", 0) or 0),
                "queue_saturated": bool(rec.get("queue_saturated", False)),
                "selected_candidate": _safe_str(rec.get("selected_candidate"), "ppo"),
                "action_source": _safe_str(rec.get("action_source"), "ppo"),
                "intervention": bool(rec.get("intervention", False)),
                "out_of_bounds": bool(rec.get("out_of_bounds", False)),
                "done_reason": _safe_str(rec.get("done_reason"), ""),
                "x_dot_dwell_age": int(rec.get("x_dot_dwell_age", 0) or 0),
                "dominant_violation_component": _safe_str(rec.get("dominant_violation_component"), "none"),
                "applied_action_prediction_error_l2": _safe_float(rec.get("applied_action_prediction_error_l2")),
            }
        )
    return rows


def _safe_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(out) else out


def _safe_str(value: Any, default: str = "") -> str:
    if value is None:
        return default
    if isinstance(value, float) and math.isnan(value):
        return default
    text = str(value)
    return default if text.lower() == "nan" else text
