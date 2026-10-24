#!/usr/bin/env python3
"""Run the frozen horizon-spectrum Phase 2b confirmation and guardrail.

Authorized arms:
    E-N3, E-N4, AP-N3, D-N3 (confirmation)
    E-N3, E-N4, AP-N3       (gradual physical_multiplier guardrail)

The historical AP chooser is loaded from its hash-pinned source and is not
reimplemented here. A compatibility wrapper only removes newer no-op keyword
arguments and enriches returned score dictionaries for logging.
"""

from __future__ import annotations

import contextlib
import csv
import hashlib
import importlib.util
import json
import math
import os
import random
import shutil
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np
from scipy.stats import beta


REPO_ROOT = Path(r"C:\Users\anaxe\RAGforAssurredAutonomy")
HARNESS_PATH = REPO_ROOT / "tools" / "quadrotor_delay_ra_experiment.py"
PREREG_PATH = (
    REPO_ROOT / "analysis" / "prereg" / "horizon_spectrum_ap_confirmation_20260723.md"
)
FREEZE_MANIFEST_PATH = PREREG_PATH.with_suffix(".manifest.json")
PILOT_DIR = REPO_ROOT / "analysis" / "out" / "horizon_spectrum_20260723" / "pilot"
PILOT_ADVANCEMENT_PATH = PILOT_DIR / "pilot_advancement.json"
PILOT_FILE_MANIFEST = PILOT_DIR / "FILE_MANIFEST.csv"
AP_SOURCE_PATH = Path(
    r"C:\Users\anaxe\Documents\Codex\2026-05-05"
    r"\master-codex-prompt-agentic-rag-with\outputs"
    r"\uniform_thrust_provenance_ADDENDUM_bundle_20260720"
    r"\run_followup_experiments_20260708.py"
)
AP_PROVENANCE_PATH = (
    REPO_ROOT
    / "analysis"
    / "out"
    / "horizon_spectrum_phase0_20260723"
    / "ap_provenance.txt"
)
ARCHIVED_A_EPISODES = (
    REPO_ROOT
    / "analysis"
    / "out"
    / "markov_robustness_arm_2026-07-06"
    / "markov_robustness_episodes.csv"
)
OUTPUT_ROOT = (
    REPO_ROOT / "analysis" / "out" / "horizon_spectrum_20260723" / "phase2b"
)

CONFIRMATION_TOKEN = "CONFIRM-E-N3,E-N4,AP-N3,D-N3"
CONFIRMATION_SEEDS = tuple(range(61000, 62000))
GUARDRAIL_SEEDS = tuple(range(62000, 62100))
STEPS = 100
WARNING_BUFFER = 0.05
PIPELINE = "physical_multiplier"
ARCHIVED_A_SEQUENCE_HASH_12 = "b74f787d735d04c6"
EXPECTED_HARNESS_SHA256 = (
    "8fb2df2b0ea09e9740540cf0ff0cb889fae3f9a2b38aab689f2572e828f5a1a5"
)
EXPECTED_AP_SHA256 = (
    "9cf5740426f35b2a13d5e3dfcd1a270452c9be345b94f00fb335d9ed87d151f5"
)
PAYLOAD_STEP_MIN = 20
PAYLOAD_STEP_MAX = 35
PAYLOAD_RANDOM_SEED = 20260723

CONFIRMATION_ARMS = (
    ("E-N3", "E_delay_aware_intervention", 3, False),
    ("E-N4", "E_delay_aware_intervention", 4, False),
    ("AP-N3", "AP_candidate_projection_intervention", 3, True),
    ("D-N3", "D_delay_blind_intervention", 3, False),
)
GUARDRAIL_ARMS = CONFIRMATION_ARMS[:3]

KNOWN_FAMILIES = {
    "F1": "39e7c99b7933_2c6be4ccedec",
    "F2": "fb7d25581ddc_2c6fea9771d0",
    "F3": "1a99aa1f0676_d10f8cdbb9cc",
    "F4": "6b91763e8338_d83e5d15466c",
    "F5": "0f7451715530_8b78cf3a2d47",
    "F6": "3ced655a6ce0_dd47aace5543",
}
G3B_REQUIRED = {"F1", "F2", "F3"}


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
    for path in (REPO_ROOT / "tools", REPO_ROOT / "src"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))


def import_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def verify_input_provenance() -> dict[str, Any]:
    freeze = json.loads(FREEZE_MANIFEST_PATH.read_text(encoding="utf-8"))
    if freeze.get("status") != "FROZEN":
        raise AssertionError("Frozen preregistration manifest status is not FROZEN")
    expected = {
        PREREG_PATH: freeze["preregistration"]["sha256"],
        HARNESS_PATH: EXPECTED_HARNESS_SHA256,
        AP_SOURCE_PATH: EXPECTED_AP_SHA256,
        ARCHIVED_A_EPISODES: freeze["archived_A_episodes"]["sha256"],
    }
    checks = []
    for path, expected_hash in expected.items():
        actual = sha256(path)
        if actual != expected_hash:
            raise AssertionError(
                f"Input hash mismatch: {path}; expected={expected_hash}; actual={actual}"
            )
        checks.append({"path": str(path), "sha256": actual})

    pilot_manifest_rows = list(
        csv.DictReader(PILOT_FILE_MANIFEST.open(newline="", encoding="utf-8"))
    )
    advancement_relative = PILOT_ADVANCEMENT_PATH.relative_to(PILOT_DIR).as_posix()
    advancement_entry = next(
        (
            row
            for row in pilot_manifest_rows
            if row["relative_path"] == advancement_relative
        ),
        None,
    )
    if advancement_entry is None:
        raise AssertionError("pilot_advancement.json is absent from pilot manifest")
    if sha256(PILOT_ADVANCEMENT_PATH) != advancement_entry["sha256"]:
        raise AssertionError("Pilot advancement artifact hash mismatch")
    advancement = json.loads(PILOT_ADVANCEMENT_PATH.read_text(encoding="utf-8"))
    expected_arms = ["E-N3", "E-N4", "AP-N3", "D-N3"]
    if advancement.get("confirmation_arms") != expected_arms:
        raise AssertionError(
            f"Pilot arm list differs: {advancement.get('confirmation_arms')}"
        )
    if CONFIRMATION_TOKEN != "CONFIRM-" + ",".join(expected_arms):
        raise AssertionError("Compiled confirmation token differs from pilot arm list")
    return {
        "confirmation_token": CONFIRMATION_TOKEN,
        "freeze_manifest": str(FREEZE_MANIFEST_PATH),
        "freeze_manifest_sha256": sha256(FREEZE_MANIFEST_PATH),
        "pilot_advancement": str(PILOT_ADVANCEMENT_PATH),
        "pilot_advancement_sha256": sha256(PILOT_ADVANCEMENT_PATH),
        "verified_inputs": checks,
    }


def append_csv(
    path: Path,
    rows: Iterable[dict[str, Any]],
    fieldnames: list[str] | None,
) -> list[str] | None:
    materialized = list(rows)
    if not materialized:
        return fieldnames
    if fieldnames is None:
        fieldnames = list(materialized[0].keys())
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=fieldnames, extrasaction="ignore", lineterminator="\n"
        )
        if write_header:
            writer.writeheader()
        writer.writerows(materialized)
    return fieldnames


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fields.append(key)
                seen.add(key)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=fields, extrasaction="ignore", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def clopper_pearson(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    lower = 0.0 if k == 0 else float(beta.ppf(alpha / 2.0, k, n - k + 1))
    upper = 1.0 if k == n else float(beta.ppf(1.0 - alpha / 2.0, k + 1, n - k))
    return lower, upper


def exact_mcnemar(a_only: int, b_only: int) -> float:
    discordant = int(a_only + b_only)
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, i) for i in range(min(a_only, b_only) + 1))
    return float(min(1, 2 * tail / (2**discordant)))


@contextlib.contextmanager
def historical_ap_context(exp: Any, historical: Any) -> Iterator[None]:
    """Install the exact historical chooser with a logging-only adapter."""

    arm = historical.Arm(
        name="AP-N3",
        base_condition="AP_candidate_projection_intervention",
        patch_mode="candidate_projection",
    )
    with historical.patched_arm(exp, arm):
        frozen_projection_choose = exp.choose_delay_aware_action

        def compatibility_logging_adapter(*args: Any, **kwargs: Any) -> Any:
            prediction_horizon = int(kwargs.pop("prediction_horizon", 3))
            velocity_penalty = bool(kwargs.pop("use_velocity_gated_penalty", False))
            kwargs.pop("lambda_penalty", None)
            kwargs.pop("velocity_gated_min_vx", None)
            if prediction_horizon != 3:
                raise AssertionError(f"AP-N3 received N={prediction_horizon}")
            if velocity_penalty:
                raise AssertionError("AP-N3 received velocity-gated penalty")
            result = frozen_projection_choose(*args, **kwargs)
            action, name, predicted, scores = result
            env = kwargs.get("env", args[0] if args else None)
            if env is None:
                raise AssertionError("AP logging adapter could not locate env")
            for item in scores:
                candidate = np.asarray(item["action"], dtype=float)
                physical = exp.denormalized_clipped_action(env, candidate)
                item["constraint_violation"] = float(
                    item["predicted_state_severity"]
                )
                item["action_violation"] = float(
                    exp.input_violation_severity(physical)
                )
                item["rmse"] = float(item["predicted_tracking_rmse"])
                item["l2_penalty"] = float(np.linalg.norm(candidate))
                item["final_score"] = float(item["score"])
            return action, name, predicted, scores

        exp.choose_delay_aware_action = compatibility_logging_adapter
        try:
            yield
        finally:
            exp.choose_delay_aware_action = frozen_projection_choose


def parse_candidate_scores(row: dict[str, Any]) -> list[dict[str, Any]]:
    value = row.get("candidate_scores", "[]")
    return list(json.loads(value) if isinstance(value, str) else value)


def payload_rows(
    block: str,
    pipeline: str,
    arm: str,
    horizon: int,
    seed: int,
    step_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    payload: list[dict[str, Any]] = []
    for row in step_rows:
        step = int(row["step"])
        if step < PAYLOAD_STEP_MIN or step > PAYLOAD_STEP_MAX:
            continue
        for rank, item in enumerate(parse_candidate_scores(row), start=1):
            payload.append(
                {
                    "block": block,
                    "pipeline": pipeline,
                    "arm": arm,
                    "prediction_horizon": horizon,
                    "seed": seed,
                    "step": step,
                    "warning": as_bool(row["warning"]),
                    "intervention": as_bool(row["intervention"]),
                    "selected_candidate": row.get("selected_candidate", ""),
                    "candidate_rank": rank,
                    "candidate_name": item["name"],
                    "candidate_action": compact_json(item.get("action", [])),
                    "is_selected": str(item["name"])
                    == str(row.get("selected_candidate", "")),
                    "constraint_violation": float(
                        item.get(
                            "constraint_violation",
                            item["predicted_state_severity"],
                        )
                    ),
                    "action_violation": float(item.get("action_violation", 0.0)),
                    "rmse": float(
                        item.get("rmse", item["predicted_tracking_rmse"])
                    ),
                    "l2_penalty": float(item.get("l2_penalty", 0.0)),
                    "final_score": float(item.get("final_score", item["score"])),
                    "raw_delay_aware_score": item.get(
                        "raw_delay_aware_score", ""
                    ),
                    "projection_distance_to_ppo": item.get(
                        "projection_distance_to_ppo", ""
                    ),
                    "projection_feasible": item.get("projection_feasible", ""),
                }
            )
    return payload


def ap_intervention_rows(
    block: str,
    pipeline: str,
    seed: int,
    step_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for row in step_rows:
        if not as_bool(row.get("intervention")):
            continue
        scores = parse_candidate_scores(row)
        if not scores or any("projection_feasible" not in item for item in scores):
            raise AssertionError(
                f"AP branch fields absent: block={block} seed={seed} step={row['step']}"
            )
        selected = str(row.get("selected_candidate", ""))
        selected_rows = [item for item in scores if str(item["name"]) == selected]
        if len(selected_rows) != 1:
            raise AssertionError(
                f"AP selected candidate lookup failed: seed={seed} step={row['step']}"
            )
        item = selected_rows[0]
        feasible_count = sum(bool(score["projection_feasible"]) for score in scores)
        result.append(
            {
                "block": block,
                "pipeline": pipeline,
                "seed": seed,
                "step": int(row["step"]),
                "selected_candidate": selected,
                "projection_feasible": bool(item["projection_feasible"]),
                "feasible_candidate_count": feasible_count,
                "fallback_used": feasible_count == 0,
                "raw_delay_aware_score": float(item["raw_delay_aware_score"]),
                "projection_distance_to_ppo": float(
                    item["projection_distance_to_ppo"]
                ),
                "projection_score": float(item["score"]),
                "candidate_count": len(scores),
            }
        )
    return result


def clean_step_rows(
    block: str,
    pipeline: str,
    arm: str,
    base_condition: str,
    horizon: int,
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    cleaned: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        item.pop("candidate_scores", None)
        item["block"] = block
        item["pipeline"] = pipeline
        item["arm"] = arm
        item["prediction_horizon"] = horizon
        item["base_condition"] = base_condition
        item["condition"] = arm
        cleaned.append(item)
    return cleaned


def run_arm_episode(
    exp: Any,
    historical: Any,
    block: str,
    pipeline: str,
    arm_spec: tuple[str, str, int, bool],
    seed: int,
    disturbance_mode: str,
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    arm, condition, horizon, is_ap = arm_spec
    rows: list[dict[str, Any]] = []
    context = (
        historical_ap_context(exp, historical)
        if is_ap
        else contextlib.nullcontext()
    )
    with context:
        episode = exp.run_episode(
            condition=condition,
            seed=seed,
            steps=STEPS,
            warning_buffer=WARNING_BUFFER,
            output_rows=rows,
            delayed_delay_mode="uniform_0_2",
            disturbance_mode=disturbance_mode,
            thrust_decay_final=0.70,
            use_corrected_action_conversion=True,
            use_velocity_gated_penalty=False,
            prediction_horizon=horizon,
        )
    hashes = episode_hashes(rows)
    episode.update(
        {
            "block": block,
            "pipeline": pipeline,
            "arm": arm,
            "base_condition": condition,
            "condition": arm,
            "prediction_horizon": horizon,
            "requested_seed": seed,
            "oob": str(episode.get("done_reason", "")) == "out_of_bounds",
            "first_oob_step": first_oob_step(rows),
            **hashes,
        }
    )
    return (
        episode,
        clean_step_rows(block, pipeline, arm, condition, horizon, rows),
        payload_rows(block, pipeline, arm, horizon, seed, rows),
        ap_intervention_rows(block, pipeline, seed, rows) if is_ap else [],
    )


def run_gate_a(exp: Any, output_root: Path) -> tuple[dict[str, Any], str]:
    rows: list[dict[str, Any]] = []
    episode = exp.run_episode(
        condition="A_nominal_ppo_zero_delay",
        seed=CONFIRMATION_SEEDS[0],
        steps=STEPS,
        warning_buffer=WARNING_BUFFER,
        output_rows=rows,
        delayed_delay_mode="uniform_0_2",
        disturbance_mode="none",
        use_corrected_action_conversion=True,
        prediction_horizon=3,
    )
    state_hash = rounded_hash(
        np.vstack([parse_vector(row["state"]) for row in rows])
    )
    # The archived Markov hash uses compact JSON, so retain that exact check too.
    compact_states = np.round(
        np.vstack([parse_vector(row["state"]) for row in rows]), 12
    ).tolist()
    archived_style = hashlib.sha256(
        compact_json(compact_states).encode("utf-8")
    ).hexdigest()[:16]
    episode["standard_state_sequence_hash_12"] = state_hash
    episode["archived_style_sequence_hash_12"] = archived_style
    write_csv(
        output_root / "gate_A_reference_steps.csv",
        clean_step_rows(
            "gate_A",
            PIPELINE,
            "A-reference",
            "A_nominal_ppo_zero_delay",
            3,
            rows,
        ),
    )
    (output_root / "gate_A_reference_episode.json").write_text(
        json.dumps(episode, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return episode, archived_style


def run_block(
    exp: Any,
    historical: Any,
    block: str,
    seeds: tuple[int, ...],
    arms: tuple[tuple[str, str, int, bool], ...],
    disturbance_mode: str,
    output_dir: Path,
) -> tuple[list[dict[str, Any]], Path]:
    output_dir.mkdir(parents=True)
    (output_dir / "full_steps").mkdir()
    (output_dir / "universal_payload_capture").mkdir()
    episodes: list[dict[str, Any]] = []
    step_fields = {arm[0]: None for arm in arms}
    payload_fields = {arm[0]: None for arm in arms}
    ap_fields: list[str] | None = None
    ap_path = output_dir / "ap_intervention_branches.csv"
    block_start = time.perf_counter()

    for index, seed in enumerate(seeds, start=1):
        for arm_spec in arms:
            arm = arm_spec[0]
            episode, steps, payload, ap_rows = run_arm_episode(
                exp,
                historical,
                block,
                PIPELINE,
                arm_spec,
                seed,
                disturbance_mode,
            )
            episodes.append(episode)
            step_fields[arm] = append_csv(
                output_dir / "full_steps" / f"{arm}_steps.csv",
                steps,
                step_fields[arm],
            )
            payload_fields[arm] = append_csv(
                output_dir
                / "universal_payload_capture"
                / f"{arm}_steps20_35_all_scores.csv",
                payload,
                payload_fields[arm],
            )
            if ap_rows:
                ap_fields = append_csv(ap_path, ap_rows, ap_fields)
        if index == 1 or index % 10 == 0:
            print(
                compact_json(
                    {
                        "event": "block_progress",
                        "block": block,
                        "seeds_complete": index,
                        "seeds_total": len(seeds),
                        "elapsed_seconds": time.perf_counter() - block_start,
                    }
                ),
                flush=True,
            )
    write_csv(output_dir / "episodes.csv", episodes)
    runtime = {
        "block": block,
        "seeds": len(seeds),
        "arms": [arm[0] for arm in arms],
        "wall_clock_seconds": time.perf_counter() - block_start,
    }
    (output_dir / "runtime.json").write_text(
        json.dumps(runtime, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return episodes, ap_path


def structural_gates(
    block: str,
    episodes: list[dict[str, Any]],
    seeds: tuple[int, ...],
    arms: tuple[tuple[str, str, int, bool], ...],
    gate_a_hash: str,
    confirmation: bool,
) -> dict[str, Any]:
    arm_names = [arm[0] for arm in arms]
    expected_seeds = set(seeds)
    by_arm: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for episode in episodes:
        by_arm[str(episode["arm"])].append(episode)

    matched_hashes = 0
    for seed in seeds:
        values = {
            str(row["delay_schedule_hash"])
            for row in episodes
            if int(row["seed"]) == seed
        }
        if len(values) == 1:
            matched_hashes += 1
    row_counts = {arm: len(by_arm[arm]) for arm in arm_names}
    seed_errors = {
        arm: {
            "missing": sorted(
                expected_seeds - {int(row["seed"]) for row in by_arm[arm]}
            ),
            "extra": sorted(
                {int(row["seed"]) for row in by_arm[arm]} - expected_seeds
            ),
        }
        for arm in arm_names
        if {int(row["seed"]) for row in by_arm[arm]} != expected_seeds
    }
    pair_counts = Counter(
        (str(row["arm"]), int(row["seed"])) for row in episodes
    )
    duplicates = [
        {"arm": arm, "seed": seed, "count": count}
        for (arm, seed), count in pair_counts.items()
        if count != 1
    ]
    echoes = [
        {
            "arm": row["arm"],
            "seed": row["seed"],
            "requested_seed": row["requested_seed"],
        }
        for row in episodes
        if int(row["seed"]) != int(row["requested_seed"])
    ]
    gates: dict[str, Any] = {
        "G1": {
            "status": "PASS" if matched_hashes == len(seeds) else "FAIL",
            "matched_delay_hash_seeds": matched_hashes,
            "required": len(seeds),
        },
        "G2": {
            "status": "PASS"
            if gate_a_hash == ARCHIVED_A_SEQUENCE_HASH_12
            else "FAIL",
            "observed_A_archived_style_hash": gate_a_hash,
            "required_A_archived_style_hash": ARCHIVED_A_SEQUENCE_HASH_12,
        },
        "G4": {
            "status": "PASS"
            if all(row_counts.get(arm) == len(seeds) for arm in arm_names)
            else "FAIL",
            "row_counts": row_counts,
            "required_per_arm": len(seeds),
        },
        "G5": {
            "status": "PASS" if not seed_errors else "FAIL",
            "registered_seed_start": seeds[0],
            "registered_seed_stop": seeds[-1],
            "errors": seed_errors,
        },
        "G6": {
            "status": "PASS" if not duplicates and not echoes else "FAIL",
            "duplicate_or_gap_pairs": duplicates,
            "echoed_seed_mismatches": echoes,
        },
    }
    if confirmation:
        e3 = by_arm["E-N3"]
        e3_exits = [row for row in e3 if as_bool(row["oob"])]
        e3_rate = len(e3_exits) / len(e3)
        present_keys = {str(row["trajectory_family_key"]) for row in e3_exits}
        present_known = sorted(
            family
            for family, key in KNOWN_FAMILIES.items()
            if key in present_keys
        )
        missing_required = sorted(G3B_REQUIRED - set(present_known))
        gates["G3a"] = {
            "status": "PASS" if 0.12 <= e3_rate <= 0.22 else "FAIL",
            "E_N3_oob_count": len(e3_exits),
            "n": len(e3),
            "rate": e3_rate,
            "required_inclusive_rate": [0.12, 0.22],
        }
        gates["G3b"] = {
            "status": "PASS" if not missing_required else "FAIL",
            "required_largest_families": sorted(G3B_REQUIRED),
            "present_known_families": present_known,
            "missing_required_families": missing_required,
            "all_six_signatures": KNOWN_FAMILIES,
        }
    return {"block": block, "gates": gates}


def assert_gates(gate_result: dict[str, Any], output_dir: Path) -> None:
    path = output_dir / "gates.json"
    path.write_text(
        json.dumps(gate_result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    failed = [
        name
        for name, result in gate_result["gates"].items()
        if result["status"] != "PASS"
    ]
    if failed:
        raise AssertionError(
            f"{gate_result['block']} gate failure: {failed}; details={path}"
        )


def summarize_arms(
    episodes: list[dict[str, Any]], output_path: Path
) -> list[dict[str, Any]]:
    by_arm: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in episodes:
        by_arm[str(row["arm"])].append(row)
    rows: list[dict[str, Any]] = []
    for arm, values in sorted(by_arm.items()):
        k = sum(as_bool(row["oob"]) for row in values)
        lower, upper = clopper_pearson(k, len(values))
        survivors = [row for row in values if not as_bool(row["oob"])]
        total_severity = sum(
            float(row["cumulative_normalized_violation_severity"])
            for row in values
        )
        total_steps = sum(int(row["steps_recorded"]) for row in values)
        rows.append(
            {
                "block": values[0]["block"],
                "pipeline": values[0]["pipeline"],
                "arm": arm,
                "prediction_horizon": values[0]["prediction_horizon"],
                "n": len(values),
                "oob_count": k,
                "oob_rate": k / len(values),
                "cp95_lower": lower,
                "cp95_upper": upper,
                "survivor_mean_cumulative_severity": float(
                    np.mean(
                        [
                            float(
                                row[
                                    "cumulative_normalized_violation_severity"
                                ]
                            )
                            for row in survivors
                        ]
                    )
                )
                if survivors
                else float("nan"),
                "exposure_weighted_per_step_severity": total_severity
                / total_steps,
                "mean_intervention_count": float(
                    np.mean([float(row["intervention_count"]) for row in values])
                ),
                "mean_runtime_seconds": float(
                    np.mean([float(row["runtime_seconds"]) for row in values])
                ),
                "episodes_csv": str(output_path.parent / "episodes.csv"),
            }
        )
    write_csv(output_path, rows)
    return rows


def paired_outputs(
    episodes: list[dict[str, Any]], output_dir: Path
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    indexed = {
        (int(row["seed"]), str(row["arm"])): row for row in episodes
    }
    seeds = sorted({int(row["seed"]) for row in episodes})
    arms = ["E-N3", "E-N4", "AP-N3", "D-N3"]
    long_rows = []
    for seed in seeds:
        for arm in arms:
            row = indexed[(seed, arm)]
            long_rows.append(
                {
                    "seed": seed,
                    "arm": arm,
                    "oob": as_bool(row["oob"]),
                    "first_oob_step": row["first_oob_step"],
                    "delay_schedule_hash": row["delay_schedule_hash"],
                    "trajectory_family_key": row["trajectory_family_key"],
                    "requested_seed": row["requested_seed"],
                    "episodes_csv": str(output_dir / "episodes.csv"),
                }
            )
    write_csv(OUTPUT_ROOT / "horizon_ablation_paired_outcomes.csv", long_rows)

    tests: dict[str, Any] = {
        "alpha_per_test": 0.025,
        "test": "exact two-sided McNemar",
        "hypotheses": {},
    }
    for hypothesis, comparator in (("HZ1", "E-N4"), ("HA1", "AP-N3")):
        both = comparator_only = e3_only = neither = 0
        for seed in seeds:
            comp = as_bool(indexed[(seed, comparator)]["oob"])
            e3 = as_bool(indexed[(seed, "E-N3")]["oob"])
            if comp and e3:
                both += 1
            elif comp:
                comparator_only += 1
            elif e3:
                e3_only += 1
            else:
                neither += 1
        tests["hypotheses"][hypothesis] = {
            "comparator": comparator,
            "reference": "E-N3",
            "both_fail": both,
            "comparator_only": comparator_only,
            "E_N3_only": e3_only,
            "neither": neither,
            "discordant_total": comparator_only + e3_only,
            "exact_two_sided_mcnemar_p": exact_mcnemar(
                comparator_only, e3_only
            ),
        }
    (OUTPUT_ROOT / "horizon_ablation_paired_tests.json").write_text(
        json.dumps(tests, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return long_rows, tests


def write_family_membership(
    block: str,
    episodes: list[dict[str, Any]],
    output_dir: Path,
) -> dict[str, list[int]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    by_arm: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in episodes:
        by_arm[str(row["arm"])].append(row)
    representatives: dict[str, list[int]] = {}
    for arm, values in sorted(by_arm.items()):
        exits = [row for row in values if as_bool(row["oob"])]
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in exits:
            groups[str(row["trajectory_family_key"])].append(row)
        records = []
        reps = []
        for index, (key, members) in enumerate(
            sorted(groups.items(), key=lambda item: (-len(item[1]), item[0])),
            start=1,
        ):
            representative = min(int(row["seed"]) for row in members)
            reps.append(representative)
            terminal, applied = key.split("_", maxsplit=1)
            records.append(
                {
                    "block": block,
                    "arm": arm,
                    "family_id": f"{arm}-F{index:03d}",
                    "basin_size": len(members),
                    "representative_seed": representative,
                    "terminal_state_hash_12": terminal,
                    "applied_action_sequence_hash_12": applied,
                    "trajectory_family_key": key,
                    "known_E_N3_family_id": next(
                        (
                            family
                            for family, known_key in KNOWN_FAMILIES.items()
                            if known_key == key
                        ),
                        "",
                    ),
                    "member_seeds": compact_json(
                        sorted(int(row["seed"]) for row in members)
                    ),
                }
            )
        representatives[arm] = reps
        write_csv(output_dir / f"{block}_{arm}_families.csv", records)
    return representatives


def filter_payload(
    source: Path,
    destination: Path,
    selected_roles: dict[int, set[str]],
    episode_by_seed: dict[int, dict[str, Any]],
) -> int:
    destination.parent.mkdir(parents=True, exist_ok=True)
    writer = None
    stream_out = None
    count = 0
    try:
        with source.open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            fields = list(reader.fieldnames or []) + [
                "payload_role",
                "episode_oob",
                "trajectory_family_key",
                "state_sequence_hash_12",
            ]
            stream_out = destination.open("w", newline="", encoding="utf-8")
            writer = csv.DictWriter(
                stream_out, fieldnames=fields, extrasaction="ignore", lineterminator="\n"
            )
            writer.writeheader()
            for row in reader:
                seed = int(row["seed"])
                if seed not in selected_roles:
                    continue
                episode = episode_by_seed[seed]
                row.update(
                    {
                        "payload_role": "+".join(sorted(selected_roles[seed])),
                        "episode_oob": as_bool(episode["oob"]),
                        "trajectory_family_key": episode["trajectory_family_key"],
                        "state_sequence_hash_12": episode["state_sequence_hash_12"],
                    }
                )
                writer.writerow(row)
                count += 1
    finally:
        if stream_out is not None:
            stream_out.close()
    return count


def select_payloads(
    block: str,
    episodes: list[dict[str, Any]],
    block_dir: Path,
    family_reps: dict[str, list[int]],
    output_dir: Path,
    matched_e3_reps_for_e4: bool,
) -> list[dict[str, Any]]:
    by_arm: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in episodes:
        by_arm[str(row["arm"])].append(row)
    selection_rows = []
    e3_reps = set(family_reps.get("E-N3", []))
    for arm, values in sorted(by_arm.items()):
        episode_by_seed = {int(row["seed"]): row for row in values}
        roles: dict[int, set[str]] = defaultdict(set)
        for seed in family_reps.get(arm, []):
            roles[seed].add("observed_exit_family_representative")
        survivors = [
            int(row["seed"]) for row in values if not as_bool(row["oob"])
        ]
        arm_offset = sum(ord(char) for char in f"{block}:{arm}")
        rng = random.Random(PAYLOAD_RANDOM_SEED + arm_offset)
        for seed in rng.sample(survivors, min(4, len(survivors))):
            roles[seed].add("random_survivor")
        if matched_e3_reps_for_e4 and arm == "E-N4":
            for seed in sorted(e3_reps):
                if seed in episode_by_seed and not as_bool(
                    episode_by_seed[seed]["oob"]
                ):
                    roles[seed].add("matched_E_N3_family_representative_survivor")
        source = (
            block_dir
            / "universal_payload_capture"
            / f"{arm}_steps20_35_all_scores.csv"
        )
        destination = (
            output_dir / f"{block}_{arm}_steps20_35_scores.csv"
        )
        row_count = filter_payload(source, destination, roles, episode_by_seed)
        for seed, seed_roles in sorted(roles.items()):
            selection_rows.append(
                {
                    "block": block,
                    "arm": arm,
                    "seed": seed,
                    "roles": "+".join(sorted(seed_roles)),
                    "episode_oob": as_bool(episode_by_seed[seed]["oob"]),
                    "trajectory_family_key": episode_by_seed[seed][
                        "trajectory_family_key"
                    ],
                    "payload_path": str(destination),
                    "payload_file_total_rows": row_count,
                }
            )
        if matched_e3_reps_for_e4 and arm == "E-N4":
            matched = [
                seed
                for seed, seed_roles in roles.items()
                if "matched_E_N3_family_representative_survivor" in seed_roles
            ]
            if not matched:
                raise AssertionError("No matched E-N4 survivor payloads selected")
            required_steps = set(range(24, 33))
            seen: dict[int, set[int]] = defaultdict(set)
            with destination.open(newline="", encoding="utf-8") as stream:
                for row in csv.DictReader(stream):
                    seed = int(row["seed"])
                    if seed in matched:
                        seen[seed].add(int(row["step"]))
            missing = {
                seed: sorted(required_steps - seen[seed])
                for seed in matched
                if not required_steps.issubset(seen[seed])
            }
            if missing:
                raise AssertionError(
                    f"E-N4 drift-onset payload steps 24-32 missing: {missing}"
                )
    return selection_rows


def ap_fallback_summary(
    seeds: tuple[int, ...],
    branch_path: Path,
    output_path: Path,
) -> list[dict[str, Any]]:
    by_seed: dict[int, list[dict[str, Any]]] = defaultdict(list)
    if branch_path.exists():
        with branch_path.open(newline="", encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                by_seed[int(row["seed"])].append(row)
    rows = []
    for seed in seeds:
        values = by_seed.get(seed, [])
        fallback = sum(as_bool(row["fallback_used"]) for row in values)
        feasible = len(values) - fallback
        rows.append(
            {
                "seed": seed,
                "intervention_count": len(values),
                "feasible_branch_count": feasible,
                "fallback_branch_count": fallback,
                "fallback_rate": fallback / len(values) if values else 0.0,
                "branch_log": str(branch_path),
            }
        )
    write_csv(output_path, rows)
    return rows


def write_output_manifest(root: Path, provenance: dict[str, Any]) -> None:
    paths = sorted(
        [path for path in root.rglob("*") if path.is_file() and path.name != "MANIFEST.sha256"],
        key=lambda path: str(path).lower(),
    )
    lines = [f"{sha256(path)} *{path}" for path in paths]
    input_paths = [
        PREREG_PATH,
        FREEZE_MANIFEST_PATH,
        HARNESS_PATH,
        AP_SOURCE_PATH,
        PILOT_ADVANCEMENT_PATH,
        PILOT_FILE_MANIFEST,
        ARCHIVED_A_EPISODES,
    ]
    lines.extend(f"{sha256(path)} *{path}" for path in input_paths)
    (root / "MANIFEST.sha256").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> int:
    ensure_runtime_paths()
    provenance = verify_input_provenance()
    if OUTPUT_ROOT.exists():
        raise FileExistsError(
            f"Phase 2b output already exists; refusing append/rerun: {OUTPUT_ROOT}"
        )
    OUTPUT_ROOT.mkdir(parents=True)
    provenance["runner_path"] = str(Path(__file__).resolve())
    provenance["runner_sha256"] = sha256(Path(__file__).resolve())
    provenance["launch_timestamp_local"] = time.strftime(
        "%Y-%m-%dT%H:%M:%S%z"
    )
    (OUTPUT_ROOT / "run_started.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    shutil.copyfile(AP_PROVENANCE_PATH, OUTPUT_ROOT / "ap_provenance.txt")
    print("PHASE2B_INPUTS_VERIFIED", flush=True)
    print(json.dumps(provenance, indent=2, sort_keys=True), flush=True)

    exp = import_module(HARNESS_PATH, "horizon_phase2b_harness")
    historical = import_module(AP_SOURCE_PATH, "horizon_phase2b_historical_ap")
    historical.add_condition_specs(exp)

    _, gate_a_hash = run_gate_a(exp, OUTPUT_ROOT)

    confirmation_dir = OUTPUT_ROOT / "confirmation"
    confirmation_episodes, confirmation_ap_log = run_block(
        exp,
        historical,
        "confirmation",
        CONFIRMATION_SEEDS,
        CONFIRMATION_ARMS,
        "none",
        confirmation_dir,
    )
    confirmation_gates = structural_gates(
        "confirmation",
        confirmation_episodes,
        CONFIRMATION_SEEDS,
        CONFIRMATION_ARMS,
        gate_a_hash,
        confirmation=True,
    )
    assert_gates(confirmation_gates, confirmation_dir)

    confirmation_summary = summarize_arms(
        confirmation_episodes, OUTPUT_ROOT / "horizon_ablation_summary.csv"
    )
    _, paired_tests = paired_outputs(confirmation_episodes, confirmation_dir)
    confirmation_reps = write_family_membership(
        "confirmation",
        confirmation_episodes,
        OUTPUT_ROOT / "family_membership",
    )
    payload_selection = select_payloads(
        "confirmation",
        confirmation_episodes,
        confirmation_dir,
        confirmation_reps,
        OUTPUT_ROOT / "onset_payloads",
        matched_e3_reps_for_e4=True,
    )
    confirmation_ap_fallback = ap_fallback_summary(
        CONFIRMATION_SEEDS,
        confirmation_ap_log,
        OUTPUT_ROOT / "ap_fallback_rate.csv",
    )

    guardrail_dir = OUTPUT_ROOT / "guardrail_physical_multiplier"
    guardrail_episodes, guardrail_ap_log = run_block(
        exp,
        historical,
        "gradual_guardrail",
        GUARDRAIL_SEEDS,
        GUARDRAIL_ARMS,
        "thrust_decay",
        guardrail_dir,
    )
    guardrail_gates = structural_gates(
        "gradual_guardrail",
        guardrail_episodes,
        GUARDRAIL_SEEDS,
        GUARDRAIL_ARMS,
        gate_a_hash,
        confirmation=False,
    )
    assert_gates(guardrail_gates, guardrail_dir)
    guardrail_summary = summarize_arms(
        guardrail_episodes, OUTPUT_ROOT / "guardrail_summary.csv"
    )
    guardrail_reps = write_family_membership(
        "gradual_guardrail",
        guardrail_episodes,
        OUTPUT_ROOT / "family_membership",
    )
    payload_selection.extend(
        select_payloads(
            "gradual_guardrail",
            guardrail_episodes,
            guardrail_dir,
            guardrail_reps,
            OUTPUT_ROOT / "onset_payloads",
            matched_e3_reps_for_e4=False,
        )
    )
    write_csv(OUTPUT_ROOT / "payload_selection.csv", payload_selection)
    ap_fallback_summary(
        GUARDRAIL_SEEDS,
        guardrail_ap_log,
        OUTPUT_ROOT / "ap_guardrail_fallback_rate.csv",
    )

    mechanical_lines = [
        "===================================================",
        "HORIZON SPECTRUM PHASE 2B MECHANICAL REPORT",
        "===================================================",
        "",
        f"Confirmation token: {CONFIRMATION_TOKEN}",
        "Confirmation seeds: 61000-61999",
        "Guardrail seeds: 62000-62099",
        f"Guardrail pipeline: {PIPELINE}",
        "",
        "CONFIRMATION GATES:",
    ]
    mechanical_lines.extend(
        f"- {name}: {result['status']}"
        for name, result in confirmation_gates["gates"].items()
    )
    mechanical_lines.extend(["", "CONFIRMATION OOB COUNTS:"])
    mechanical_lines.extend(
        f"- {row['arm']}: {row['oob_count']}/{row['n']}"
        for row in confirmation_summary
    )
    mechanical_lines.extend(["", "REGISTERED PAIRED TESTS:"])
    for hypothesis, result in paired_tests["hypotheses"].items():
        mechanical_lines.append(
            f"- {hypothesis}: comparator_only={result['comparator_only']}, "
            f"E_N3_only={result['E_N3_only']}, both={result['both_fail']}, "
            f"neither={result['neither']}, p={result['exact_two_sided_mcnemar_p']}"
        )
    mechanical_lines.extend(["", "GUARDRAIL GATES:"])
    mechanical_lines.extend(
        f"- {name}: {result['status']}"
        for name, result in guardrail_gates["gates"].items()
    )
    mechanical_lines.extend(["", "GUARDRAIL OOB COUNTS:"])
    mechanical_lines.extend(
        f"- {row['arm']}: {row['oob_count']}/{row['n']}"
        for row in guardrail_summary
    )
    total_ap_interventions = sum(
        row["intervention_count"] for row in confirmation_ap_fallback
    )
    total_ap_fallback = sum(
        row["fallback_branch_count"] for row in confirmation_ap_fallback
    )
    mechanical_lines.extend(
        [
            "",
            "AP BRANCH LOG:",
            f"- confirmation intervention rows: {total_ap_interventions}",
            f"- confirmation fallback rows: {total_ap_fallback}",
            f"- path: {confirmation_ap_log}",
            "",
            "No interpretation is included.",
        ]
    )
    (OUTPUT_ROOT / "mechanical_report.txt").write_text(
        "\n".join(mechanical_lines) + "\n", encoding="utf-8"
    )
    completion = {
        "status": "COMPLETE",
        "confirmation_token": CONFIRMATION_TOKEN,
        "confirmation_gates": confirmation_gates,
        "confirmation_summary": confirmation_summary,
        "paired_tests": paired_tests,
        "guardrail_gates": guardrail_gates,
        "guardrail_summary": guardrail_summary,
        "ap_confirmation_intervention_rows": total_ap_interventions,
        "ap_confirmation_fallback_rows": total_ap_fallback,
    }
    (OUTPUT_ROOT / "completion.json").write_text(
        json.dumps(completion, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_output_manifest(OUTPUT_ROOT, provenance)
    print("\n".join(mechanical_lines), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
