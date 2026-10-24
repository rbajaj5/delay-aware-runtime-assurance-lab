#!/usr/bin/env python3
"""Run the exploratory N={5,6,8,9} dense horizon-spectrum extension.

The four horizons run on the same 1,000 schedules as the registered N=3/N=4
block. Each horizon writes to a separate directory, allowing one process per
horizon without concurrent writes to any artifact.
"""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import math
import os
import shutil
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy.stats import beta


REPO_ROOT = Path(r"C:\Users\anaxe\RAGforAssurredAutonomy")
HARNESS_PATH = REPO_ROOT / "tools" / "quadrotor_delay_ra_experiment.py"
PREREG_PATH = (
    REPO_ROOT / "analysis" / "prereg" / "horizon_dense_extension_20260725.md"
)
FREEZE_MANIFEST_PATH = PREREG_PATH.with_suffix(".manifest.json")
OUTPUT_DIR = (
    REPO_ROOT
    / "analysis"
    / "out"
    / "horizon_spectrum_20260723"
    / "dense_extension_20260725"
)
REFERENCE_EPISODES = (
    REPO_ROOT
    / "analysis"
    / "out"
    / "horizon_spectrum_20260723"
    / "phase2b"
    / "confirmation"
    / "episodes.csv"
)
REFERENCE_N3_STEPS = (
    REPO_ROOT
    / "analysis"
    / "out"
    / "horizon_spectrum_20260723"
    / "phase2b"
    / "confirmation"
    / "full_steps"
    / "E-N3_steps.csv"
)
PHASE2B_SUMMARY = (
    REPO_ROOT
    / "analysis"
    / "out"
    / "horizon_spectrum_20260723"
    / "phase2b"
    / "horizon_ablation_summary.csv"
)
PILOT_SUMMARY = (
    REPO_ROOT
    / "analysis"
    / "out"
    / "horizon_spectrum_20260723"
    / "pilot"
    / "horizon_spectrum_pilot.csv"
)
N7_SUMMARY = (
    REPO_ROOT
    / "analysis"
    / "out"
    / "horizon_spectrum_20260723"
    / "exploratory_2x2_20260723"
    / "part_b"
    / "en7_exploratory_summary.csv"
)

HORIZONS = (5, 6, 8, 9)
SEEDS = tuple(range(61000, 62000))
SMOKE_SEEDS = (61000, 61004, 61137, 61665, 61999)
STEPS = 100
WARNING_BUFFER = 0.05
CONDITION = "E_delay_aware_intervention"
PIPELINE = "physical_multiplier"
EXPECTED_HARNESS_SHA256 = (
    "8fb2df2b0ea09e9740540cf0ff0cb889fae3f9a2b38aab689f2572e828f5a1a5"
)
PAYLOAD_STEP_MIN = 20
PAYLOAD_STEP_MAX = 35


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compact_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes"}


def parse_vector(value: Any) -> np.ndarray:
    parsed = json.loads(value) if isinstance(value, str) else value
    return np.asarray(parsed, dtype=float).reshape(-1)


def rounded_hash(array: Any, decimals: int = 12) -> str:
    canonical = np.round(np.asarray(array, dtype=float), decimals).astype("<f8")
    return hashlib.sha256(canonical.tobytes()).hexdigest()[:12]


def episode_hashes(rows: list[dict[str, Any]]) -> dict[str, str]:
    states = np.vstack([parse_vector(row["state"]) for row in rows])
    actions = np.vstack([parse_vector(row["actual_applied_action"]) for row in rows])
    terminal = rounded_hash(states[-1])
    applied = rounded_hash(actions)
    return {
        "terminal_state_hash_12": terminal,
        "applied_action_sequence_hash_12": applied,
        "state_sequence_hash_12": rounded_hash(states),
        "trajectory_family_key": f"{terminal}_{applied}",
    }


def first_oob_step(rows: list[dict[str, Any]]) -> int | str:
    for row in rows:
        if as_bool(row.get("out_of_bounds")):
            return int(row["step"])
    return ""


def ensure_runtime_paths() -> None:
    env_root = REPO_ROOT / ".mamba-safe-control-gym-3d"
    os.environ["PATH"] = ";".join(
        [
            str(env_root / "Library" / "bin"),
            str(env_root / "Scripts"),
            str(env_root),
            os.environ.get("PATH", ""),
        ]
    )
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    for path in (REPO_ROOT / "tools", REPO_ROOT / "src"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))


def load_harness(tag: str) -> Any:
    ensure_runtime_paths()
    module_name = f"horizon_dense_harness_{tag}_{os.getpid()}"
    spec = importlib.util.spec_from_file_location(module_name, HARNESS_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import harness: {HARNESS_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=list(rows[0].keys()),
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def append_rows(
    writer: csv.DictWriter,
    rows: Iterable[dict[str, Any]],
) -> None:
    for row in rows:
        writer.writerow(row)


def clopper_pearson(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    lower = 0.0 if k == 0 else float(beta.ppf(alpha / 2.0, k, n - k + 1))
    upper = 1.0 if k == n else float(beta.ppf(1.0 - alpha / 2.0, k + 1, n - k))
    return lower, upper


def candidate_rows(raw_value: Any) -> list[dict[str, Any]]:
    if isinstance(raw_value, str):
        if not raw_value.strip():
            return []
        return list(json.loads(raw_value))
    return list(raw_value or [])


def discrimination_row(
    arm: str,
    horizon: int,
    seed: int,
    row: dict[str, Any],
) -> dict[str, Any]:
    candidates = candidate_rows(row.get("candidate_scores", "[]"))
    scored = sorted(
        candidates,
        key=lambda item: (float(item.get("final_score", math.inf)), str(item.get("name", ""))),
    )
    scores = np.asarray(
        [float(item.get("final_score", math.nan)) for item in scored], dtype=float
    )
    violations = np.asarray(
        [float(item.get("constraint_violation", math.nan)) for item in scored],
        dtype=float,
    )
    finite_scores = scores[np.isfinite(scores)]
    if finite_scores.size >= 2:
        score_range = float(np.max(finite_scores) - np.min(finite_scores))
        top2_gap = float(finite_scores[1] - finite_scores[0])
        norm_margin = top2_gap / score_range if score_range > 0.0 else 0.0
    else:
        score_range = float("nan")
        top2_gap = float("nan")
        norm_margin = float("nan")
    selected = str(row.get("selected_candidate", "") or "")
    winner = selected or (str(scored[0].get("name", "")) if scored else "")
    return {
        "arm": arm,
        "prediction_horizon": horizon,
        "seed": seed,
        "step": int(row["step"]),
        "intervention": as_bool(row.get("intervention")),
        "winner": winner,
        "selected_candidate": selected,
        "candidate_count": len(candidates),
        "feasible_support": int(np.sum(violations <= 1e-9)) if violations.size else 0,
        "saturated_step": bool(
            violations.size and np.all(np.isfinite(violations)) and np.all(violations > 1e-9)
        ),
        "norm_top2_margin": norm_margin,
        "top2_score_gap": top2_gap,
        "score_range": score_range,
        "minimum_constraint_violation": float(np.min(violations))
        if violations.size
        else float("nan"),
        "maximum_constraint_violation": float(np.max(violations))
        if violations.size
        else float("nan"),
        "gentle_down_winner": winner == "gentle_down",
        "delay_sample": int(row.get("delay_sample", 0)),
        "queue_len_before": int(row.get("queue_len_before", 0)),
        "state": row.get("state", ""),
    }


def score_payload_rows(
    arm: str,
    horizon: int,
    seed: int,
    row: dict[str, Any],
) -> list[dict[str, Any]]:
    step = int(row["step"])
    if step < PAYLOAD_STEP_MIN or step > PAYLOAD_STEP_MAX:
        return []
    selected = str(row.get("selected_candidate", "") or "")
    payload: list[dict[str, Any]] = []
    for rank, item in enumerate(
        sorted(
            candidate_rows(row.get("candidate_scores", "[]")),
            key=lambda value: (
                float(value.get("final_score", math.inf)),
                str(value.get("name", "")),
            ),
        ),
        start=1,
    ):
        name = str(item.get("name", ""))
        payload.append(
            {
                "arm": arm,
                "prediction_horizon": horizon,
                "seed": seed,
                "step": step,
                "candidate_rank": rank,
                "candidate_name": name,
                "selected_candidate": selected,
                "is_selected": name == selected,
                "constraint_violation": float(
                    item.get("constraint_violation", math.nan)
                ),
                "action_violation": float(item.get("action_violation", math.nan)),
                "rmse": float(item.get("rmse", math.nan)),
                "l2_penalty": float(item.get("l2_penalty", math.nan)),
                "final_score": float(item.get("final_score", math.nan)),
                "candidate_action": compact_json(item.get("action", [])),
            }
        )
    return payload


def strip_candidate_payload(
    arm: str,
    horizon: int,
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    cleaned: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        item.pop("candidate_scores", None)
        item["block"] = "exploratory_dense_horizon"
        item["pipeline"] = PIPELINE
        item["arm"] = arm
        item["prediction_horizon"] = horizon
        item["requested_seed"] = int(item["seed"])
        cleaned.append(item)
    return cleaned


def verify_freeze() -> dict[str, Any]:
    if not FREEZE_MANIFEST_PATH.exists():
        raise FileNotFoundError(FREEZE_MANIFEST_PATH)
    manifest = json.loads(FREEZE_MANIFEST_PATH.read_text(encoding="utf-8"))
    if manifest.get("status") != "FROZEN":
        raise AssertionError("Dense-extension manifest is not FROZEN")
    checks = {
        "protocol": PREREG_PATH,
        "runner": Path(__file__).resolve(),
        "harness": HARNESS_PATH,
        "reference_episodes": REFERENCE_EPISODES,
        "reference_n3_steps": REFERENCE_N3_STEPS,
        "phase2b_summary": PHASE2B_SUMMARY,
        "pilot_summary": PILOT_SUMMARY,
        "n7_summary": N7_SUMMARY,
    }
    verified: dict[str, Any] = {}
    for label, path in checks.items():
        expected = manifest["inputs"][label]["sha256"]
        actual = sha256(path)
        if actual != expected:
            raise AssertionError(
                f"Freeze hash mismatch for {label}: expected={expected}, actual={actual}"
            )
        verified[label] = {
            "path": str(path),
            "sha256": actual,
            "size_bytes": path.stat().st_size,
        }
    if verified["harness"]["sha256"] != EXPECTED_HARNESS_SHA256:
        raise AssertionError("Harness does not match the registered horizon harness")
    return {"freeze_manifest": manifest, "verified_inputs": verified}


def reference_n3_rows() -> dict[int, dict[str, str]]:
    rows = [
        row
        for row in read_csv(REFERENCE_EPISODES)
        if row.get("arm") == "E-N3"
    ]
    by_seed = {int(row["seed"]): row for row in rows}
    if set(by_seed) != set(SEEDS):
        raise AssertionError("Reference E-N3 seed set is not 61000-61999")
    return by_seed


def run_smoke_gate(
    harness: Any,
    reference: dict[int, dict[str, str]],
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    for seed in SMOKE_SEEDS:
        rows: list[dict[str, Any]] = []
        episode = harness.run_episode(
            condition=CONDITION,
            seed=seed,
            steps=STEPS,
            warning_buffer=WARNING_BUFFER,
            output_rows=rows,
            delayed_delay_mode="uniform_0_2",
            disturbance_mode="none",
            use_corrected_action_conversion=True,
            prediction_horizon=3,
        )
        hashes = episode_hashes(rows)
        observed_oob = str(episode.get("done_reason", "")) == "out_of_bounds"
        observed_first = first_oob_step(rows)
        expected = reference[seed]
        comparisons = {
            "state_sequence_hash_12": (
                hashes["state_sequence_hash_12"],
                expected["state_sequence_hash_12"],
            ),
            "applied_action_sequence_hash_12": (
                hashes["applied_action_sequence_hash_12"],
                expected["applied_action_sequence_hash_12"],
            ),
            "delay_schedule_hash": (
                str(episode["delay_schedule_hash"]),
                expected["delay_schedule_hash"],
            ),
            "oob": (observed_oob, as_bool(expected["oob"])),
            "first_oob_step": (
                str(observed_first),
                str(expected["first_oob_step"]),
            ),
        }
        mismatches = {
            key: {"observed": observed, "expected": wanted}
            for key, (observed, wanted) in comparisons.items()
            if observed != wanted
        }
        if mismatches:
            raise AssertionError(f"N=3 smoke mismatch for seed {seed}: {mismatches}")
        checks.append(
            {
                "seed": seed,
                "status": "PASS",
                **hashes,
                "delay_schedule_hash": episode["delay_schedule_hash"],
                "oob": observed_oob,
                "first_oob_step": observed_first,
            }
        )
    return {
        "status": "PASS",
        "seeds": list(SMOKE_SEEDS),
        "checks": checks,
    }


def run_horizon(horizon: int) -> dict[str, Any]:
    ensure_runtime_paths()
    try:
        import torch

        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except Exception:
        pass

    harness = load_harness(f"n{horizon}")
    arm = f"E-N{horizon}-dense"
    arm_dir = OUTPUT_DIR / f"n{horizon}"
    arm_dir.mkdir(parents=True, exist_ok=False)
    full_steps_path = arm_dir / f"{arm}_steps.csv"
    diagnostics_path = arm_dir / f"{arm}_discrimination.csv"
    payload_path = arm_dir / f"{arm}_steps20_35_scores.csv"
    episodes_path = arm_dir / "episodes.csv"

    episode_rows: list[dict[str, Any]] = []
    start = time.perf_counter()
    step_stream = None
    diagnostic_stream = None
    payload_stream = None
    try:
        for index, seed in enumerate(SEEDS, start=1):
            rows: list[dict[str, Any]] = []
            episode = harness.run_episode(
                condition=CONDITION,
                seed=seed,
                steps=STEPS,
                warning_buffer=WARNING_BUFFER,
                output_rows=rows,
                delayed_delay_mode="uniform_0_2",
                disturbance_mode="none",
                use_corrected_action_conversion=True,
                prediction_horizon=horizon,
            )
            hashes = episode_hashes(rows)
            oob = str(episode.get("done_reason", "")) == "out_of_bounds"
            episode.update(
                {
                    "block": "exploratory_dense_horizon",
                    "pipeline": PIPELINE,
                    "arm": arm,
                    "base_condition": CONDITION,
                    "prediction_horizon": horizon,
                    "requested_seed": seed,
                    "oob": oob,
                    "first_oob_step": first_oob_step(rows),
                    **hashes,
                }
            )
            episode_rows.append(episode)

            cleaned = strip_candidate_payload(arm, horizon, rows)
            diagnostics = [
                discrimination_row(arm, horizon, seed, row) for row in rows
            ]
            payload = [
                item
                for row in rows
                for item in score_payload_rows(arm, horizon, seed, row)
            ]

            if step_stream is None:
                step_stream = full_steps_path.open(
                    "w", encoding="utf-8", newline=""
                )
                step_writer = csv.DictWriter(
                    step_stream,
                    fieldnames=list(cleaned[0].keys()),
                    extrasaction="ignore",
                    lineterminator="\n",
                )
                step_writer.writeheader()
            if diagnostic_stream is None:
                diagnostic_stream = diagnostics_path.open(
                    "w", encoding="utf-8", newline=""
                )
                diagnostic_writer = csv.DictWriter(
                    diagnostic_stream,
                    fieldnames=list(diagnostics[0].keys()),
                    extrasaction="ignore",
                    lineterminator="\n",
                )
                diagnostic_writer.writeheader()
            if payload_stream is None:
                payload_stream = payload_path.open(
                    "w", encoding="utf-8", newline=""
                )
                payload_writer = csv.DictWriter(
                    payload_stream,
                    fieldnames=list(payload[0].keys()),
                    extrasaction="ignore",
                    lineterminator="\n",
                )
                payload_writer.writeheader()

            append_rows(step_writer, cleaned)
            append_rows(diagnostic_writer, diagnostics)
            append_rows(payload_writer, payload)

            if index == 1 or index % 25 == 0:
                print(
                    compact_json(
                        {
                            "event": "dense_horizon_progress",
                            "horizon": horizon,
                            "seeds_complete": index,
                            "seeds_total": len(SEEDS),
                            "elapsed_seconds": time.perf_counter() - start,
                        }
                    ),
                    flush=True,
                )
                step_stream.flush()
                diagnostic_stream.flush()
                payload_stream.flush()
    finally:
        for stream in (step_stream, diagnostic_stream, payload_stream):
            if stream is not None:
                stream.close()

    write_csv(episodes_path, episode_rows)
    runtime = time.perf_counter() - start
    runtime_payload = {
        "arm": arm,
        "prediction_horizon": horizon,
        "episodes": len(episode_rows),
        "runtime_seconds": runtime,
        "mean_wall_seconds_per_episode": runtime / len(episode_rows),
    }
    (arm_dir / "runtime.json").write_text(
        json.dumps(runtime_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return runtime_payload


def quantiles(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if not array.size:
        return {
            "mean": float("nan"),
            "p25": float("nan"),
            "p50": float("nan"),
            "p75": float("nan"),
        }
    return {
        "mean": float(np.mean(array)),
        "p25": float(np.percentile(array, 25)),
        "p50": float(np.percentile(array, 50)),
        "p75": float(np.percentile(array, 75)),
    }


def summarize_horizon(horizon: int) -> tuple[dict[str, Any], dict[str, Any]]:
    arm = f"E-N{horizon}-dense"
    arm_dir = OUTPUT_DIR / f"n{horizon}"
    episodes = read_csv(arm_dir / "episodes.csv")
    diagnostics = read_csv(arm_dir / f"{arm}_discrimination.csv")
    n = len(episodes)
    k = sum(as_bool(row["oob"]) for row in episodes)
    lower, upper = clopper_pearson(k, n)
    survivors = [row for row in episodes if not as_bool(row["oob"])]
    total_severity = sum(
        float(row["cumulative_normalized_violation_severity"]) for row in episodes
    )
    total_steps = sum(int(row["steps_recorded"]) for row in episodes)
    summary = {
        "arm": arm,
        "prediction_horizon": horizon,
        "evidence_class": "exploratory",
        "n": n,
        "oob_count": k,
        "oob_rate": k / n,
        "cp95_lower": lower,
        "cp95_upper": upper,
        "survivor_mean_cumulative_severity": float(
            np.mean(
                [
                    float(row["cumulative_normalized_violation_severity"])
                    for row in survivors
                ]
            )
        )
        if survivors
        else float("nan"),
        "exposure_weighted_per_step_severity": total_severity / total_steps,
        "mean_intervention_count": float(
            np.mean([float(row["intervention_count"]) for row in episodes])
        ),
        "mean_runtime_seconds": float(
            np.mean([float(row["runtime_seconds"]) for row in episodes])
        ),
        "trajectory_family_count": len(
            {row["trajectory_family_key"] for row in episodes}
        ),
        "failure_family_count": len(
            {
                row["trajectory_family_key"]
                for row in episodes
                if as_bool(row["oob"])
            }
        ),
        "source": str(arm_dir / "episodes.csv"),
    }

    intervention_rows = [
        row for row in diagnostics if as_bool(row["intervention"])
    ]
    winners = Counter(row["winner"] for row in intervention_rows)
    by_seed: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in intervention_rows:
        by_seed[int(row["seed"])].append(row)
    churn_values: list[float] = []
    for rows in by_seed.values():
        ordered = sorted(rows, key=lambda item: int(item["step"]))
        if len(ordered) < 2:
            churn_values.append(0.0)
            continue
        changes = sum(
            left["winner"] != right["winner"]
            for left, right in zip(ordered, ordered[1:])
        )
        churn_values.append(changes / (len(ordered) - 1))
    margins = [float(row["norm_top2_margin"]) for row in intervention_rows]
    supports = [float(row["feasible_support"]) for row in intervention_rows]
    mechanism = {
        "arm": arm,
        "prediction_horizon": horizon,
        "intervention_step_rows": len(intervention_rows),
        "mean_feasible_support": float(np.mean(supports)) if supports else float("nan"),
        "median_feasible_support": float(np.median(supports))
        if supports
        else float("nan"),
        "saturated_step_fraction": (
            sum(as_bool(row["saturated_step"]) for row in intervention_rows)
            / len(intervention_rows)
            if intervention_rows
            else float("nan")
        ),
        "norm_top2_margin_mean": quantiles(margins)["mean"],
        "norm_top2_margin_p25": quantiles(margins)["p25"],
        "norm_top2_margin_p50": quantiles(margins)["p50"],
        "norm_top2_margin_p75": quantiles(margins)["p75"],
        "argmin_churn_mean": float(np.mean(churn_values))
        if churn_values
        else float("nan"),
        "gentle_down_winner_count": winners.get("gentle_down", 0),
        "gentle_down_winner_share": (
            winners.get("gentle_down", 0) / len(intervention_rows)
            if intervention_rows
            else float("nan")
        ),
        "winner_counts_json": compact_json(dict(sorted(winners.items()))),
        "source": str(arm_dir / f"{arm}_discrimination.csv"),
    }
    return summary, mechanism


def independent_step_audit(horizon: int) -> dict[str, Any]:
    arm = f"E-N{horizon}-dense"
    arm_dir = OUTPUT_DIR / f"n{horizon}"
    episodes = {int(row["seed"]): row for row in read_csv(arm_dir / "episodes.csv")}
    observed: dict[int, dict[str, Any]] = {}
    with (arm_dir / f"{arm}_steps.csv").open(
        "r", encoding="utf-8-sig", newline=""
    ) as stream:
        for row in csv.DictReader(stream):
            seed = int(row["seed"])
            item = observed.setdefault(
                seed, {"rows": 0, "oob": False, "first_oob_step": ""}
            )
            item["rows"] += 1
            if as_bool(row.get("out_of_bounds")) and not item["oob"]:
                item["oob"] = True
                item["first_oob_step"] = int(row["step"])
    mismatches = []
    for seed in SEEDS:
        expected = episodes[seed]
        actual = observed.get(seed)
        if actual is None:
            mismatches.append({"seed": seed, "reason": "missing_step_rows"})
            continue
        if actual["oob"] != as_bool(expected["oob"]):
            mismatches.append(
                {
                    "seed": seed,
                    "reason": "oob_mismatch",
                    "step_log": actual["oob"],
                    "episode": expected["oob"],
                }
            )
        if str(actual["first_oob_step"]) != str(expected["first_oob_step"]):
            mismatches.append(
                {
                    "seed": seed,
                    "reason": "first_oob_step_mismatch",
                    "step_log": actual["first_oob_step"],
                    "episode": expected["first_oob_step"],
                }
            )
        if actual["rows"] != int(expected["steps_recorded"]):
            mismatches.append(
                {
                    "seed": seed,
                    "reason": "row_count_mismatch",
                    "step_log": actual["rows"],
                    "episode": expected["steps_recorded"],
                }
            )
    if mismatches:
        raise AssertionError(f"Step-log endpoint audit failed for N={horizon}: {mismatches[:10]}")
    return {
        "arm": arm,
        "status": "PASS",
        "episode_count": len(episodes),
        "oob_count_recomputed": sum(item["oob"] for item in observed.values()),
        "mismatches": [],
    }


def structural_gates(
    reference: dict[int, dict[str, str]],
    smoke: dict[str, Any],
) -> dict[str, Any]:
    rows_by_horizon: dict[int, list[dict[str, str]]] = {}
    for horizon in HORIZONS:
        rows_by_horizon[horizon] = read_csv(OUTPUT_DIR / f"n{horizon}" / "episodes.csv")

    gate2_errors: dict[str, Any] = {}
    for horizon, rows in rows_by_horizon.items():
        seeds = [int(row["seed"]) for row in rows]
        requested = [int(row["requested_seed"]) for row in rows]
        duplicates = sorted(seed for seed, count in Counter(seeds).items() if count != 1)
        missing = sorted(set(SEEDS) - set(seeds))
        extra = sorted(set(seeds) - set(SEEDS))
        echo = [
            {"seed": int(row["seed"]), "requested_seed": int(row["requested_seed"])}
            for row in rows
            if int(row["seed"]) != int(row["requested_seed"])
        ]
        if len(rows) != len(SEEDS) or duplicates or missing or extra or echo:
            gate2_errors[str(horizon)] = {
                "row_count": len(rows),
                "duplicates": duplicates,
                "missing": missing,
                "extra": extra,
                "echo_mismatches": echo,
            }
    if gate2_errors:
        raise AssertionError(f"Seed/row gates failed: {gate2_errors}")

    delay_mismatches = []
    for seed in SEEDS:
        expected = reference[seed]["delay_schedule_hash"]
        observed = {
            horizon: next(
                row["delay_schedule_hash"]
                for row in rows_by_horizon[horizon]
                if int(row["seed"]) == seed
            )
            for horizon in HORIZONS
        }
        if any(value != expected for value in observed.values()):
            delay_mismatches.append(
                {"seed": seed, "reference": expected, "observed": observed}
            )
    if delay_mismatches:
        raise AssertionError(f"Delay-hash gate failed: {delay_mismatches[:10]}")

    config_errors = []
    for horizon, rows in rows_by_horizon.items():
        for row in rows:
            if (
                row["delay_mode"] != "uniform_0_2"
                or row["disturbance_mode"] != "none"
                or row["monitor"] != "delay_aware"
                or row["intervention_mode"] != "delay_aware"
                or int(row["prediction_horizon"]) != horizon
                or row["base_condition"] != CONDITION
                or row["pipeline"] != PIPELINE
            ):
                config_errors.append(
                    {
                        "horizon": horizon,
                        "seed": row["seed"],
                        "delay_mode": row["delay_mode"],
                        "disturbance_mode": row["disturbance_mode"],
                        "monitor": row["monitor"],
                        "intervention_mode": row["intervention_mode"],
                        "prediction_horizon": row["prediction_horizon"],
                        "base_condition": row["base_condition"],
                        "pipeline": row["pipeline"],
                    }
                )
    if config_errors:
        raise AssertionError(f"Configuration gate failed: {config_errors[:10]}")

    endpoint_audits = [independent_step_audit(horizon) for horizon in HORIZONS]
    return {
        "G1_N3_reproduction": smoke,
        "G2_seed_rows_and_echo": {
            "status": "PASS",
            "required_seeds": len(SEEDS),
            "horizons": list(HORIZONS),
            "errors": {},
        },
        "G3_delay_hash_matching": {
            "status": "PASS",
            "matched_seeds": len(SEEDS),
            "reference_arm": "E-N3 registered",
            "errors": [],
        },
        "G4_configuration": {
            "status": "PASS",
            "pipeline": PIPELINE,
            "delay_mode": "uniform_0_2",
            "disturbance_mode": "none",
            "errors": [],
        },
        "G5_step_log_endpoint_recomputation": {
            "status": "PASS",
            "audits": endpoint_audits,
        },
    }


def dense_spectrum_tables(
    new_summaries: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    phase2b = read_csv(PHASE2B_SUMMARY)
    pilot = read_csv(PILOT_SUMMARY)
    n7 = read_csv(N7_SUMMARY)[0]

    primary: list[dict[str, Any]] = []
    for horizon in (3, 4):
        row = next(
            item
            for item in phase2b
            if item["arm"] == f"E-N{horizon}"
        )
        primary.append(
            {
                "prediction_horizon": horizon,
                "arm": row["arm"],
                "evidence_class": "registered",
                "n": int(row["n"]),
                "oob_count": int(row["oob_count"]),
                "oob_rate": float(row["oob_rate"]),
                "cp95_lower": float(row["cp95_lower"]),
                "cp95_upper": float(row["cp95_upper"]),
                "source": str(PHASE2B_SUMMARY),
            }
        )
    for row in new_summaries:
        primary.append(
            {
                "prediction_horizon": int(row["prediction_horizon"]),
                "arm": row["arm"],
                "evidence_class": "exploratory",
                "n": int(row["n"]),
                "oob_count": int(row["oob_count"]),
                "oob_rate": float(row["oob_rate"]),
                "cp95_lower": float(row["cp95_lower"]),
                "cp95_upper": float(row["cp95_upper"]),
                "source": row["source"],
            }
        )
    primary.append(
        {
            "prediction_horizon": 7,
            "arm": n7["arm"],
            "evidence_class": "exploratory",
            "n": int(n7["n"]),
            "oob_count": int(n7["oob_count"]),
            "oob_rate": float(n7["oob_rate"]),
            "cp95_lower": float(n7["cp95_lower"]),
            "cp95_upper": float(n7["cp95_upper"]),
            "source": str(N7_SUMMARY),
        }
    )
    n10 = next(row for row in pilot if int(row["prediction_horizon"]) == 10)
    primary.append(
        {
            "prediction_horizon": 10,
            "arm": n10["arm"],
            "evidence_class": "pilot",
            "n": int(n10["n"]),
            "oob_count": int(n10["oob_count"]),
            "oob_rate": float(n10["oob_rate"]),
            "cp95_lower": float(n10["cp95_lower"]),
            "cp95_upper": float(n10["cp95_upper"]),
            "source": str(PILOT_SUMMARY),
        }
    )
    primary.sort(key=lambda row: int(row["prediction_horizon"]))

    all_points = list(primary)
    for row in pilot:
        all_points.append(
            {
                "prediction_horizon": int(row["prediction_horizon"]),
                "arm": row["arm"],
                "evidence_class": "pilot",
                "n": int(row["n"]),
                "oob_count": int(row["oob_count"]),
                "oob_rate": float(row["oob_rate"]),
                "cp95_lower": float(row["cp95_lower"]),
                "cp95_upper": float(row["cp95_upper"]),
                "source": str(PILOT_SUMMARY),
            }
        )
    all_points.sort(
        key=lambda row: (
            int(row["prediction_horizon"]),
            {"pilot": 0, "registered": 1, "exploratory": 2}[row["evidence_class"]],
        )
    )
    return primary, all_points


def write_manifest(root: Path) -> None:
    rows = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        if path.name == "FILE_MANIFEST.csv":
            continue
        rows.append(
            {
                "relative_path": path.relative_to(root).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    write_csv(root / "FILE_MANIFEST.csv", rows)


def main() -> int:
    provenance = verify_freeze()
    if OUTPUT_DIR.exists():
        raise FileExistsError(
            f"Output directory already exists; refusing to append: {OUTPUT_DIR}"
        )
    OUTPUT_DIR.mkdir(parents=True)
    frozen_dir = OUTPUT_DIR / "frozen_inputs"
    frozen_dir.mkdir()
    shutil.copy2(PREREG_PATH, frozen_dir / PREREG_PATH.name)
    shutil.copy2(Path(__file__).resolve(), frozen_dir / Path(__file__).name)
    shutil.copy2(FREEZE_MANIFEST_PATH, frozen_dir / FREEZE_MANIFEST_PATH.name)

    launch = {
        "status": "STARTED",
        "timestamp_local": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "evidence_class": "exploratory",
        "horizons": list(HORIZONS),
        "seed_start": SEEDS[0],
        "seed_stop": SEEDS[-1],
        "steps": STEPS,
        "parallel_workers": len(HORIZONS),
        "provenance": provenance,
    }
    (OUTPUT_DIR / "run_started.json").write_text(
        json.dumps(launch, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print("DENSE_EXTENSION_FREEZE_VERIFIED", flush=True)

    reference = reference_n3_rows()
    harness = load_harness("smoke")
    smoke = run_smoke_gate(harness, reference)
    (OUTPUT_DIR / "n3_reproduction_gate.json").write_text(
        json.dumps(smoke, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print("N3_REPRODUCTION_GATE_PASS", flush=True)

    worker_results = []
    with ProcessPoolExecutor(max_workers=len(HORIZONS)) as pool:
        future_map = {pool.submit(run_horizon, horizon): horizon for horizon in HORIZONS}
        for future in as_completed(future_map):
            horizon = future_map[future]
            result = future.result()
            worker_results.append(result)
            print(
                compact_json(
                    {
                        "event": "dense_horizon_complete",
                        "horizon": horizon,
                        "result": result,
                    }
                ),
                flush=True,
            )

    summaries: list[dict[str, Any]] = []
    mechanisms: list[dict[str, Any]] = []
    for horizon in HORIZONS:
        summary, mechanism = summarize_horizon(horizon)
        summaries.append(summary)
        mechanisms.append(mechanism)
    write_csv(OUTPUT_DIR / "dense_extension_summary.csv", summaries)
    write_csv(OUTPUT_DIR / "dense_extension_mechanism_summary.csv", mechanisms)

    gates = structural_gates(reference, smoke)
    (OUTPUT_DIR / "structural_gates.json").write_text(
        json.dumps(gates, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    primary, all_points = dense_spectrum_tables(summaries)
    write_csv(OUTPUT_DIR / "dense_spectrum_primary_points.csv", primary)
    write_csv(OUTPUT_DIR / "dense_spectrum_all_points.csv", all_points)

    completion = {
        "status": "COMPLETE",
        "evidence_class": "exploratory",
        "horizons": list(HORIZONS),
        "seed_start": SEEDS[0],
        "seed_stop": SEEDS[-1],
        "summaries": summaries,
        "mechanism_summaries": mechanisms,
        "structural_gates": gates,
        "worker_results": sorted(
            worker_results, key=lambda row: int(row["prediction_horizon"])
        ),
        "hypothesis_tests_executed": 0,
    }
    (OUTPUT_DIR / "completion.json").write_text(
        json.dumps(completion, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_manifest(OUTPUT_DIR)
    print("DENSE_EXTENSION_COMPLETE", flush=True)
    print(json.dumps(summaries, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

