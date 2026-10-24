#!/usr/bin/env python3
"""Run the authorized V6 CBF-QP confirmation and guardrail blocks."""

from __future__ import annotations

import contextlib
import csv
import hashlib
import inspect
import json
import math
import platform
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np


ROOT = Path(r"C:\Users\anaxe\RAGforAssurredAutonomy")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis import run_v6_cbf_qp_pilot as pilot  # noqa: E402
from analysis.v6_exact_tests import (  # noqa: E402
    exact_mcnemar_one_sided,
    exact_mcnemar_two_sided,
)


ANALYSIS = ROOT / "analysis"
HARNESS = ROOT / "tools" / "quadrotor_delay_ra_experiment.py"
ADAPTER = ANALYSIS / "v6_cbf_qp_adapter.py"
PHASE0 = ANALYSIS / "run_v6_phase0_feasibility.py"
PHASE2B = ANALYSIS / "run_horizon_spectrum_phase2b.py"
PILOT_RUNNER = ANALYSIS / "run_v6_cbf_qp_pilot.py"
EXACT_HELPER = ANALYSIS / "v6_exact_tests.py"
PREREG_ORIGINAL = ANALYSIS / "prereg" / "v6_cbf_qp_formal_shield_20260724.md"
PREREG_AMENDED = (
    ANALYSIS / "prereg" / "v6_cbf_qp_formal_shield_20260724_amendment1.md"
)
AMENDMENT_MANIFEST = (
    ANALYSIS
    / "out"
    / "v6_cbf_qp_prereg_20260724"
    / "V6_AMENDMENT1_MANIFEST.sha256"
)
AP_SOURCE = Path(
    r"C:\Users\anaxe\Documents\Codex\2026-05-05"
    r"\master-codex-prompt-agentic-rag-with\outputs"
    r"\uniform_thrust_provenance_ADDENDUM_bundle_20260720"
    r"\run_followup_experiments_20260708.py"
)
FREEZE_DIR = ANALYSIS / "out" / "v6_cbf_qp_confirmation_freeze_20260724"
FREEZE_MANIFEST = FREEZE_DIR / "V6_CONFIRMATION_FREEZE_MANIFEST.sha256"
OUTPUT = ANALYSIS / "out" / "v6_cbf_qp_confirmation_20260724"

AUTHORIZATION = "AUTHORIZE-V6-CONFIRMATION"
SEEDS = tuple(range(73000, 74000))
RG_SEEDS = tuple(range(61000, 61005))
STEPS = 100
WARNING_BUFFER = 0.05
GAMMA = 0.20
TOL = 1.0e-7
PIPELINE = "physical_multiplier"

EXPECTED_ORIGINAL_PREREG_SHA = (
    "a220eef71bc201eedd5810b1af47e1e5592463c6944a689ca0b415530b0e810f"
)
EXPECTED_AMENDED_PREREG_SHA = (
    "b1a007ef4dfecdbb309e58db4fbf1f941fb72f2f9719f5da92699922d01df494"
)
EXPECTED_HARNESS_SHA = (
    "8fb2df2b0ea09e9740540cf0ff0cb889fae3f9a2b38aab689f2572e828f5a1a5"
)

RG_ARMS = pilot.RG_ARMS
COMPARATOR_ARMS = pilot.COMPARATOR_ARMS
ALL_ARMS = pilot.ALL_ARMS
FORMAL_ARMS = ("F-blind", "F-comp")

BOUNDS = (
    ("x", 0, -2.0, 2.0),
    ("vx", 1, -1.0, 1.0),
    ("z", 2, 0.0, 2.0),
    ("vz", 3, -1.0, 1.0),
    ("theta", 4, -0.2, 0.2),
    ("theta_dot", 5, -1.5, 1.5),
)
CONSTRAINTS = (
    "x_lower",
    "x_upper",
    "vx_lower",
    "vx_upper",
    "z_lower",
    "z_upper",
    "vz_lower",
    "vz_upper",
    "theta_lower",
    "theta_upper",
    "theta_dot_lower",
    "theta_dot_upper",
)

SECTION10_NATIVE_FILES = (
    "episodes.csv",
    "summary.csv",
    "full_state_census.csv",
    "vx_bound_episode_census.csv",
    "vx_bound_summary.csv",
    "infeasibility_fallback_ledger.csv",
    "solver_timing.csv",
    "model_mismatch_ledger.csv",
    "vx_upper_early_warning_census.csv",
    "failure_family_inventory.csv",
    "intervention_family_inventory.csv",
    "fallback_family_inventory.csv",
    "binding_channel_episode_census.csv",
    "binding_channel_summary.csv",
    "constraint_slack_summary.csv",
    "active_constraint_census.csv",
    "paired_outcomes_and_tests.json",
    "runtime_config.json",
    "gates.json",
    "block_report.txt",
)


@dataclass(frozen=True)
class BlockSpec:
    name: str
    fault_profile: str
    disturbance_mode: str
    pipeline: str


BLOCKS = (
    BlockSpec("uniform_primary", "nominal", "none", "nominal_uniform_iid"),
    BlockSpec(
        "gradual_guardrail",
        "gradual_1p0_to_0p70_steps_0_99",
        "thrust_decay",
        PIPELINE,
    ),
    BlockSpec(
        "persistent_guardrail",
        "persistent_0p70_from_step_35",
        "thrust_decay",
        PIPELINE,
    ),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compact_json(value: Any) -> str:
    return json.dumps(value, separators=(",", ":"), sort_keys=True)


def parse_json(value: Any) -> Any:
    return pilot.parse_json(value)


def parse_vector(value: Any) -> np.ndarray:
    return pilot.parse_vector(value)


def as_bool(value: Any) -> bool:
    return pilot.as_bool(value)


def write_csv(
    path: Path,
    rows: list[dict[str, Any]],
    fieldnames: list[str] | tuple[str, ...],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def ensure_csv_header(path: Path, fieldnames: list[str]) -> None:
    if path.exists():
        return
    write_csv(path, [], fieldnames)


def append_csv(
    path: Path,
    rows: list[dict[str, Any]],
    fieldnames: list[str] | None,
) -> list[str]:
    if fieldnames is None:
        if not rows:
            raise AssertionError(f"Cannot establish schema for {path}")
        fieldnames = list(rows[0])
    ensure_csv_header(path, fieldnames)
    if rows:
        with path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
            writer.writerows(rows)
    return fieldnames


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_manifest(paths: list[Path], manifest: Path) -> None:
    unique = sorted(
        {path.resolve() for path in paths},
        key=lambda path: str(path).lower(),
    )
    manifest.write_text(
        "".join(f"{sha256(path)} *{path}\n" for path in unique),
        encoding="utf-8",
    )


def verify_manifest(manifest: Path) -> int:
    count = 0
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        expected, raw_path = line.split(" ", 1)
        path = Path(raw_path.lstrip("*"))
        actual = sha256(path)
        if actual.lower() != expected.lower():
            raise AssertionError(
                f"Manifest mismatch: {path}: {actual} != {expected}"
            )
        count += 1
    return count


def verify_frozen_inputs() -> list[dict[str, str]]:
    expected = {
        HARNESS: EXPECTED_HARNESS_SHA,
        PREREG_ORIGINAL: EXPECTED_ORIGINAL_PREREG_SHA,
        PREREG_AMENDED: EXPECTED_AMENDED_PREREG_SHA,
    }
    verified = []
    for path, expected_hash in expected.items():
        actual = sha256(path)
        if actual != expected_hash:
            raise AssertionError(
                f"Frozen input mismatch: {path}: {actual} != {expected_hash}"
            )
        verified.append({"path": str(path), "sha256": actual})
    if not FREEZE_MANIFEST.exists():
        raise AssertionError(f"Confirmation freeze manifest missing: {FREEZE_MANIFEST}")
    freeze_entries = {}
    for line in FREEZE_MANIFEST.read_text(encoding="utf-8").splitlines():
        digest, raw_path = line.split(" ", 1)
        freeze_entries[str(Path(raw_path.lstrip("*")).resolve()).lower()] = digest
    for path in (Path(__file__), EXACT_HELPER):
        expected_hash = freeze_entries.get(str(path.resolve()).lower())
        if expected_hash is None:
            raise AssertionError(f"Source absent from freeze manifest: {path}")
        actual = sha256(path)
        if actual != expected_hash:
            raise AssertionError(
                f"Post-freeze source change: {path}: {actual} != {expected_hash}"
            )
        verified.append({"path": str(path), "sha256": actual})
    return verified


@contextlib.contextmanager
def thrust_profile_context(exp: Any, spec: BlockSpec) -> Iterator[None]:
    original = exp.thrust_effectiveness_at_step
    if spec.name != "persistent_guardrail":
        yield
        return

    def persistent_effectiveness(
        step: int,
        steps: int,
        final_effectiveness: float,
    ) -> float:
        del steps
        return 1.0 if int(step) < 35 else float(final_effectiveness)

    exp.thrust_effectiveness_at_step = persistent_effectiveness
    try:
        yield
    finally:
        exp.thrust_effectiveness_at_step = original


def verify_persistent_profile() -> None:
    class Dummy:
        @staticmethod
        def thrust_effectiveness_at_step(
            step: int,
            steps: int,
            final_effectiveness: float,
        ) -> float:
            return float(step + steps + final_effectiveness)

    dummy = Dummy()
    spec = next(block for block in BLOCKS if block.name == "persistent_guardrail")
    with thrust_profile_context(dummy, spec):
        values = [
            dummy.thrust_effectiveness_at_step(step, STEPS, 0.70)
            for step in (0, 34, 35, 99)
        ]
    if values != [1.0, 1.0, 0.70, 0.70]:
        raise AssertionError(f"Persistent profile anchor mismatch: {values}")


def run_formal_episode(
    exp: Any,
    phase0: Any,
    adapter: Any,
    osqp_module: Any,
    spec: BlockSpec,
    variant: str,
    seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    with thrust_profile_context(exp, spec):
        with adapter.formal_adapter_context(
            exp,
            phase0,
            osqp_module,
            variant,
        ) as (condition, chooser):
            episode = exp.run_episode(
                condition=condition,
                seed=seed,
                steps=STEPS,
                warning_buffer=WARNING_BUFFER,
                output_rows=rows,
                delayed_delay_mode="uniform_0_2",
                disturbance_mode=spec.disturbance_mode,
                thrust_decay_final=0.70,
                use_corrected_action_conversion=True,
                use_velocity_gated_penalty=False,
                prediction_horizon=3,
            )
    adapter.enrich_formal_step_rows(rows)
    if chooser.solve_count != len(rows):
        raise AssertionError(
            f"{spec.name}/{variant}/{seed}: solver rows differ "
            f"({chooser.solve_count} != {len(rows)})"
        )
    episode.update(
        {
            "block": spec.name,
            "fault_profile": spec.fault_profile,
            "pipeline": spec.pipeline,
            "arm": variant,
            "base_condition": condition,
            "condition": variant,
            "prediction_horizon": "queue_anchor_one_step",
            "requested_seed": seed,
            "oob": str(episode.get("done_reason", "")) == "out_of_bounds",
            "first_oob_step": pilot.first_oob_step(rows),
            **pilot.episode_hashes(rows),
            **adapter.formal_episode_summary(variant, rows),
        }
    )
    for row in rows:
        row.update(
            {
                "block": spec.name,
                "fault_profile": spec.fault_profile,
                "pipeline": spec.pipeline,
                "arm": variant,
                "base_condition": condition,
                "condition": variant,
                "prediction_horizon": "queue_anchor_one_step",
            }
        )
    return episode, rows


def run_comparator_episode(
    exp: Any,
    phase2b: Any,
    historical: Any,
    spec: BlockSpec,
    arm_spec: tuple[str, str, int, bool],
    seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    with thrust_profile_context(exp, spec):
        episode, rows, _, ap_rows = phase2b.run_arm_episode(
            exp,
            historical,
            spec.name,
            spec.pipeline,
            arm_spec,
            seed,
            spec.disturbance_mode,
        )
    episode["fault_profile"] = spec.fault_profile
    for row in rows:
        row["fault_profile"] = spec.fault_profile
    for row in ap_rows:
        row["fault_profile"] = spec.fault_profile
    return episode, rows, ap_rows


def verify_episode(
    episode: dict[str, Any],
    rows: list[dict[str, Any]],
    expected_hash: str,
) -> None:
    pilot.validate_episode_gate(episode, rows, expected_hash)
    if len(rows) != int(episode["steps_recorded"]):
        raise AssertionError("Step-row count does not match episode exposure")


def confirmation_hash_inventory(exp: Any) -> tuple[dict[int, str], dict[str, Any]]:
    inventory = {
        seed: exp.schedule_hash(
            exp.make_delay_schedule(seed, STEPS, "uniform_0_2")
        )
        for seed in SEEDS
    }
    if len(set(inventory.values())) != len(SEEDS):
        raise AssertionError("Confirmation schedules are not unique by seed")
    prior, scanned = pilot.collect_prior_delay_hashes(exp)
    overlap = sorted(set(inventory.values()) & prior)
    if overlap:
        raise AssertionError(
            f"Confirmation delay hashes overlap prior retained hashes: {overlap}"
        )
    return inventory, {
        "seed_start": SEEDS[0],
        "seed_end": SEEDS[-1],
        "confirmation_distinct_hashes": len(set(inventory.values())),
        "prior_distinct_hashes": len(prior),
        "overlap_count": 0,
        "compute_only_timing_seeds_included": list(range(70000, 70010)),
        "scanned_sources": scanned,
    }


def state_episode_rows(
    spec: BlockSpec,
    arm: str,
    seed: int,
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    states = np.vstack([parse_vector(row["state"]) for row in rows])
    result = []
    vx_row: dict[str, Any] = {}
    for name, index, lower, upper in BOUNDS:
        values = states[:, index]
        lower_steps = [
            int(row["step"])
            for row, value in zip(rows, values)
            if float(value) < lower
        ]
        upper_steps = [
            int(row["step"])
            for row, value in zip(rows, values)
            if float(value) > upper
        ]
        record = {
            "block": spec.name,
            "fault_profile": spec.fault_profile,
            "pipeline": spec.pipeline,
            "arm": arm,
            "seed": seed,
            "state": name,
            "lower_bound": lower,
            "upper_bound": upper,
            "episode_min": float(np.min(values)),
            "episode_max": float(np.max(values)),
            "lower_crossed": bool(lower_steps),
            "upper_crossed": bool(upper_steps),
            "first_lower_step": min(lower_steps) if lower_steps else "",
            "first_upper_step": min(upper_steps) if upper_steps else "",
        }
        result.append(record)
        if name == "vx":
            vx_row = {
                **record,
                "lower_tail_excess": max(0.0, lower - float(np.min(values))),
                "upper_tail_excess": max(0.0, float(np.max(values)) - upper),
            }
    return result, vx_row


def summarize_full_state(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["arm"], row["state"])].append(row)
    for arm in ALL_ARMS:
        for name, _, lower, upper in BOUNDS:
            group = groups[(arm, name)]
            mins = np.array([row["episode_min"] for row in group], dtype=float)
            maxs = np.array([row["episode_max"] for row in group], dtype=float)
            lower_steps = [
                int(row["first_lower_step"])
                for row in group
                if row["first_lower_step"] != ""
            ]
            upper_steps = [
                int(row["first_upper_step"])
                for row in group
                if row["first_upper_step"] != ""
            ]
            output.append(
                {
                    "block": group[0]["block"],
                    "fault_profile": group[0]["fault_profile"],
                    "pipeline": group[0]["pipeline"],
                    "arm": arm,
                    "state": name,
                    "lower_bound": lower,
                    "upper_bound": upper,
                    "lower_crossing_episodes": sum(
                        as_bool(row["lower_crossed"]) for row in group
                    ),
                    "upper_crossing_episodes": sum(
                        as_bool(row["upper_crossed"]) for row in group
                    ),
                    "absolute_min": float(np.min(mins)),
                    "mean_episode_min": float(np.mean(mins)),
                    "p05_episode_min": float(np.percentile(mins, 5)),
                    "absolute_max": float(np.max(maxs)),
                    "mean_episode_max": float(np.mean(maxs)),
                    "p95_episode_max": float(np.percentile(maxs, 95)),
                    "first_lower_step_min": min(lower_steps) if lower_steps else "",
                    "first_lower_step_median": (
                        float(np.median(lower_steps)) if lower_steps else ""
                    ),
                    "first_upper_step_min": min(upper_steps) if upper_steps else "",
                    "first_upper_step_median": (
                        float(np.median(upper_steps)) if upper_steps else ""
                    ),
                }
            )
    return output


def summarize_vx_bounds(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for arm in ALL_ARMS:
        group = [row for row in rows if row["arm"] == arm]
        mins = np.array([row["episode_min"] for row in group], dtype=float)
        maxs = np.array([row["episode_max"] for row in group], dtype=float)
        output.append(
            {
                "block": group[0]["block"],
                "fault_profile": group[0]["fault_profile"],
                "pipeline": group[0]["pipeline"],
                "arm": arm,
                "episodes": len(group),
                "vx_lower_bound": -1.0,
                "vx_upper_bound": 1.0,
                "lower_tail_crossing_episodes": sum(
                    as_bool(row["lower_crossed"]) for row in group
                ),
                "upper_tail_crossing_episodes": sum(
                    as_bool(row["upper_crossed"]) for row in group
                ),
                "both_tail_crossing_episodes": sum(
                    as_bool(row["lower_crossed"]) and as_bool(row["upper_crossed"])
                    for row in group
                ),
                "absolute_min_vx": float(np.min(mins)),
                "mean_episode_min_vx": float(np.mean(mins)),
                "p05_episode_min_vx": float(np.percentile(mins, 5)),
                "absolute_max_vx": float(np.max(maxs)),
                "mean_episode_max_vx": float(np.mean(maxs)),
                "p95_episode_max_vx": float(np.percentile(maxs, 95)),
                "mean_lower_tail_excess": float(
                    np.mean([row["lower_tail_excess"] for row in group])
                ),
                "max_lower_tail_excess": float(
                    np.max([row["lower_tail_excess"] for row in group])
                ),
                "mean_upper_tail_excess": float(
                    np.mean([row["upper_tail_excess"] for row in group])
                ),
                "max_upper_tail_excess": float(
                    np.max([row["upper_tail_excess"] for row in group])
                ),
            }
        )
    return output


def binding_episode_row(
    spec: BlockSpec,
    arm: str,
    seed: int,
    episode: dict[str, Any],
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    uncertifiable = [
        row
        for row in rows
        if as_bool(row["intervention"])
        and not as_bool(row["formal_hard_qp_feasible"])
    ]
    vx_count = 0
    theta_count = 0
    for row in uncertifiable:
        slack = parse_json(row["formal_slack_by_constraint"])
        vx_count += float(slack["vx_upper"]) > TOL
        theta_count += (
            float(slack["theta_lower"]) > TOL
            or float(slack["theta_upper"]) > TOL
        )
    n = len(uncertifiable)
    f_vx = float(vx_count / n) if n else float("nan")
    f_theta = float(theta_count / n) if n else float("nan")
    if n == 0:
        label = "no_uncertifiable_states"
    elif f_vx > f_theta:
        label = "vx_upper_dominant"
    elif f_theta > f_vx:
        label = "theta_dominant"
    else:
        label = "mixed_or_tied"
    states = np.vstack([parse_vector(row["state"]) for row in rows])
    oob = as_bool(episode["oob"])
    max_x = float(np.max(states[:, 0]))
    return {
        "block": spec.name,
        "fault_profile": spec.fault_profile,
        "pipeline": spec.pipeline,
        "arm": arm,
        "seed": seed,
        "n_uncertifiable": n,
        "vx_upper_binding_count": vx_count,
        "theta_binding_count": theta_count,
        "f_vx_upper": f_vx,
        "f_theta": f_theta,
        "channel_class": label,
        "oob": oob,
        "steps_recorded": len(rows),
        "max_x": max_x,
        "subsequently_exits_or_approaches_x_bound": bool(oob or max_x >= 1.95),
        "survivor_away_from_x_bound": bool(
            not oob and len(rows) == STEPS and max_x < 1.95
        ),
    }


def summarize_binding(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summary = []
    for arm in FORMAL_ARMS:
        arm_rows = [row for row in rows if row["arm"] == arm]
        vx_rows = [
            row for row in arm_rows if row["channel_class"] == "vx_upper_dominant"
        ]
        theta_rows = [
            row for row in arm_rows if row["channel_class"] == "theta_dominant"
        ]
        vx_success = sum(
            row["subsequently_exits_or_approaches_x_bound"] for row in vx_rows
        )
        theta_success = sum(
            row["survivor_away_from_x_bound"] for row in theta_rows
        )
        vx_ci = pilot.clopper_pearson(vx_success, len(vx_rows)) if vx_rows else ("", "")
        theta_ci = (
            pilot.clopper_pearson(theta_success, len(theta_rows))
            if theta_rows
            else ("", "")
        )
        supported = bool(
            len(vx_rows) >= 10
            and len(theta_rows) >= 10
            and vx_success / len(vx_rows) >= 0.90
            and theta_success / len(theta_rows) >= 0.90
        )
        summary.append(
            {
                "block": arm_rows[0]["block"],
                "fault_profile": arm_rows[0]["fault_profile"],
                "pipeline": arm_rows[0]["pipeline"],
                "arm": arm,
                "vx_upper_dominant_n": len(vx_rows),
                "vx_upper_dominant_exit_or_approach_n": vx_success,
                "vx_upper_dominant_exit_or_approach_fraction": (
                    vx_success / len(vx_rows) if vx_rows else ""
                ),
                "vx_upper_dominant_cp95_lower": vx_ci[0],
                "vx_upper_dominant_cp95_upper": vx_ci[1],
                "theta_dominant_n": len(theta_rows),
                "theta_dominant_survive_away_n": theta_success,
                "theta_dominant_survive_away_fraction": (
                    theta_success / len(theta_rows) if theta_rows else ""
                ),
                "theta_dominant_cp95_lower": theta_ci[0],
                "theta_dominant_cp95_upper": theta_ci[1],
                "mixed_or_tied_n": sum(
                    row["channel_class"] == "mixed_or_tied" for row in arm_rows
                ),
                "no_uncertifiable_states_n": sum(
                    row["channel_class"] == "no_uncertifiable_states"
                    for row in arm_rows
                ),
                "registered_discriminator_supported": supported,
            }
        )
    return summary


def early_warning_episode(
    spec: BlockSpec,
    arm: str,
    seed: int,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    tau_slack = ""
    tau_cross = ""
    for row in rows:
        step = int(row["step"])
        slack = parse_json(row["formal_slack_by_constraint"])
        if (
            tau_slack == ""
            and as_bool(row["intervention"])
            and not as_bool(row["formal_hard_qp_feasible"])
            and float(slack["vx_upper"]) > TOL
        ):
            tau_slack = step
        if tau_cross == "" and float(parse_vector(row["state"])[1]) > 1.0:
            tau_cross = step
    warned = tau_slack != "" and tau_cross != "" and int(tau_slack) < int(tau_cross)
    return {
        "block": spec.name,
        "fault_profile": spec.fault_profile,
        "pipeline": spec.pipeline,
        "arm": arm,
        "seed": seed,
        "tau_slack": tau_slack,
        "tau_cross": tau_cross,
        "warned_crossing": warned,
        "lead_time": int(tau_cross) - int(tau_slack) if warned else "",
        "false_warning_without_crossing": tau_slack != "" and tau_cross == "",
    }


def summarize_early_warning(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for arm in FORMAL_ARMS:
        group = [row for row in rows if row["arm"] == arm]
        crossed = [row for row in group if row["tau_cross"] != ""]
        warned = [row for row in crossed if row["warned_crossing"]]
        lead = [int(row["lead_time"]) for row in warned]
        fraction = len(warned) / len(crossed) if crossed else float("nan")
        median = float(np.median(lead)) if lead else float("nan")
        result.append(
            {
                "block": group[0]["block"],
                "fault_profile": group[0]["fault_profile"],
                "pipeline": group[0]["pipeline"],
                "arm": arm,
                "episodes": len(group),
                "vx_upper_crossings": len(crossed),
                "warned_crossings": len(warned),
                "missed_crossings": len(crossed) - len(warned),
                "warned_crossing_fraction": fraction,
                "false_warnings_without_crossing": sum(
                    row["false_warning_without_crossing"] for row in group
                ),
                "lead_time_min": min(lead) if lead else "",
                "lead_time_median": median,
                "lead_time_max": max(lead) if lead else "",
                "registered_early_warning_supported": bool(
                    crossed and fraction >= 0.90 and median >= 1.0
                ),
            }
        )
    return result


def summarize_episodes(
    spec: BlockSpec,
    episodes: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    output = []
    for arm in ALL_ARMS:
        group = [row for row in episodes if row["arm"] == arm]
        oob = [row for row in group if as_bool(row["oob"])]
        lower, upper = pilot.clopper_pearson(len(oob), len(group))
        steps = np.array([int(row["steps_recorded"]) for row in group], dtype=float)
        severity = np.array(
            [float(row["cumulative_normalized_violation_severity"]) for row in group],
            dtype=float,
        )
        exits = sorted(int(row["first_oob_step"]) for row in oob)
        output.append(
            {
                "block": spec.name,
                "fault_profile": spec.fault_profile,
                "pipeline": spec.pipeline,
                "arm": arm,
                "n": len(group),
                "oob_k": len(oob),
                "oob_rate": len(oob) / len(group),
                "oob_cp95_lower": lower,
                "oob_cp95_upper": upper,
                "first_exit_min": min(exits) if exits else "",
                "first_exit_median": float(np.median(exits)) if exits else "",
                "first_exit_max": max(exits) if exits else "",
                "first_exit_sorted": compact_json(exits),
                "mean_steps_recorded": float(np.mean(steps)),
                "mean_cumulative_severity": float(np.mean(severity)),
                "per_step_severity": float(np.sum(severity) / np.sum(steps)),
                "mean_interventions_per_episode": float(
                    np.mean([float(row["intervention_count"]) for row in group])
                ),
                "mean_reward": float(
                    np.mean([float(row["cumulative_reward"]) for row in group])
                ),
                "mean_runtime_seconds": float(
                    np.mean([float(row["runtime_seconds"]) for row in group])
                ),
                "severity_right_censored_by_oob": True,
            }
        )
    return output


def family_inventory(
    episodes: list[dict[str, Any]],
    predicate: Any,
    inventory_type: str,
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for episode in episodes:
        if predicate(episode):
            groups[(episode["arm"], episode["trajectory_family_key"])].append(episode)
    rows = []
    for (arm, family), members in sorted(groups.items()):
        rows.append(
            {
                "block": members[0]["block"],
                "fault_profile": members[0]["fault_profile"],
                "pipeline": members[0]["pipeline"],
                "inventory_type": inventory_type,
                "arm": arm,
                "trajectory_family_key": family,
                "terminal_state_hash_12": members[0]["terminal_state_hash_12"],
                "applied_action_sequence_hash_12": members[0][
                    "applied_action_sequence_hash_12"
                ],
                "basin_size": len(members),
                "seeds": compact_json(sorted(int(row["seed"]) for row in members)),
                "oob_count": sum(as_bool(row["oob"]) for row in members),
                "mean_intervention_count": float(
                    np.mean([float(row["intervention_count"]) for row in members])
                ),
                "mean_formal_fallback_count": float(
                    np.mean(
                        [float(row.get("formal_fallback_count") or 0) for row in members]
                    )
                ),
            }
        )
    return rows


FAMILY_FIELDS = (
    "block",
    "fault_profile",
    "pipeline",
    "inventory_type",
    "arm",
    "trajectory_family_key",
    "terminal_state_hash_12",
    "applied_action_sequence_hash_12",
    "basin_size",
    "seeds",
    "oob_count",
    "mean_intervention_count",
    "mean_formal_fallback_count",
)


def paired_counts(
    lookup: dict[tuple[str, int], bool],
    left: str,
    right: str,
) -> dict[str, int]:
    both = sum(lookup[(left, seed)] and lookup[(right, seed)] for seed in SEEDS)
    left_only = sum(
        lookup[(left, seed)] and not lookup[(right, seed)] for seed in SEEDS
    )
    right_only = sum(
        not lookup[(left, seed)] and lookup[(right, seed)] for seed in SEEDS
    )
    return {
        "both": both,
        f"{left}_only": left_only,
        f"{right}_only": right_only,
        "neither": len(SEEDS) - both - left_only - right_only,
    }


def paired_results(
    spec: BlockSpec,
    episodes: list[dict[str, Any]],
    vx_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    oob_lookup = {
        (row["arm"], int(row["seed"])): as_bool(row["oob"]) for row in episodes
    }
    reverse_lookup = {
        (row["arm"], int(row["seed"])): as_bool(row["lower_crossed"])
        for row in vx_rows
    }
    oob_fe = paired_counts(oob_lookup, "F-comp", "E-N4")
    oob_fb = paired_counts(oob_lookup, "F-comp", "F-blind")
    reverse = paired_counts(reverse_lookup, "F-comp", "E-N4")
    payload: dict[str, Any] = {
        "block": spec.name,
        "fault_profile": spec.fault_profile,
        "pipeline": spec.pipeline,
        "guardrail_tests_registered": spec.name == "uniform_primary",
        "F-comp_vs_E-N4_oob": oob_fe,
        "F-comp_vs_E-N4_reverse_vx": reverse,
        "F-comp_vs_F-blind_oob": oob_fb,
    }
    if spec.name != "uniform_primary":
        payload["test_note"] = (
            "Guardrail block: paired counts are descriptive; no exact test is "
            "registered or computed."
        )
        return payload

    f_oob = oob_fe["F-comp_only"]
    e_oob = oob_fe["E-N4_only"]
    oob_fe["exact_two_sided_mcnemar_p"] = exact_mcnemar_two_sided(f_oob, e_oob)
    oob_fe["HF1_endpoint_matching"] = bool(
        sum(
            as_bool(row["oob"])
            for row in episodes
            if row["arm"] == "F-comp"
        )
        == 0
        and f_oob == 0
    )

    f_reverse = reverse["F-comp_only"]
    e_reverse = reverse["E-N4_only"]
    reverse["alternative"] = "F-comp has fewer min_t(vx_t) < -1.0 crossings"
    reverse["exact_one_sided_mcnemar_p_lower_tail"] = exact_mcnemar_one_sided(
        f_reverse,
        e_reverse,
        alternative="left_fewer",
    )
    differences = np.array(
        [
            int(reverse_lookup[("F-comp", seed)])
            - int(reverse_lookup[("E-N4", seed)])
            for seed in SEEDS
        ],
        dtype=float,
    )
    rng = np.random.default_rng(20260724)
    sampled = rng.choice(differences, size=(10000, len(differences)), replace=True)
    boot = np.mean(sampled, axis=1)
    reverse["paired_risk_difference_Fcomp_minus_E-N4"] = float(
        np.mean(differences)
    )
    reverse["paired_bootstrap_95_lower"] = float(np.percentile(boot, 2.5))
    reverse["paired_bootstrap_95_upper"] = float(np.percentile(boot, 97.5))
    reverse["paired_bootstrap_resamples"] = 10000
    reverse["paired_bootstrap_rng_seed"] = 20260724
    reverse["HF2_supported"] = bool(
        reverse["exact_one_sided_mcnemar_p_lower_tail"] < 0.025
        and e_reverse > f_reverse
    )

    fb_f = oob_fb["F-comp_only"]
    fb_b = oob_fb["F-blind_only"]
    oob_fb["exact_two_sided_mcnemar_p"] = exact_mcnemar_two_sided(fb_f, fb_b)
    return payload


def solver_timing_summary(
    spec: BlockSpec,
    timing: dict[str, dict[str, list[float]]],
) -> list[dict[str, Any]]:
    rows = []
    for arm in FORMAL_ARMS:
        values = timing[arm]
        total = np.asarray(values["total"], dtype=float)
        hard_wall = np.asarray(values["hard_wall"], dtype=float)
        hard_reported = np.asarray(values["hard_reported"], dtype=float)
        hard_iterations = np.asarray(values["hard_iterations"], dtype=float)
        fallback_wall = np.asarray(values["fallback_wall"], dtype=float)
        fallback_iterations = np.asarray(values["fallback_iterations"], dtype=float)
        rows.append(
            {
                "block": spec.name,
                "fault_profile": spec.fault_profile,
                "pipeline": spec.pipeline,
                "arm": arm,
                "n_solver_invocations": len(total),
                "total_mean_ms": float(np.mean(total)),
                "total_median_ms": float(np.median(total)),
                "total_p95_ms": float(np.percentile(total, 95)),
                "total_max_ms": float(np.max(total)),
                "fraction_gt_5ms": float(np.mean(total > 5.0)),
                "fraction_gt_10ms": float(np.mean(total > 10.0)),
                "fraction_gt_20ms": float(np.mean(total > 20.0)),
                "hard_qp_wall_mean_ms": float(np.mean(hard_wall)),
                "hard_qp_solver_reported_mean_ms": float(np.mean(hard_reported)),
                "hard_qp_iterations_mean": float(np.mean(hard_iterations)),
                "hard_qp_iterations_p95": float(np.percentile(hard_iterations, 95)),
                "fallback_invocations": len(fallback_wall),
                "fallback_wall_mean_ms": (
                    float(np.mean(fallback_wall)) if len(fallback_wall) else ""
                ),
                "fallback_iterations_mean": (
                    float(np.mean(fallback_iterations))
                    if len(fallback_iterations)
                    else ""
                ),
            }
        )
    return rows


def section10_check(
    block_dir: Path,
    *,
    final: bool,
) -> dict[str, Any]:
    checks = {}
    for relative in SECTION10_NATIVE_FILES:
        if not final and relative in {"gates.json", "block_report.txt"}:
            continue
        path = block_dir / relative
        present = path.exists() and path.stat().st_size > 0
        checks[relative] = {
            "present": present,
            "size_bytes": path.stat().st_size if path.exists() else 0,
        }
        if not present:
            raise AssertionError(f"Missing/empty Section-10 output: {path}")
    for arm in ALL_ARMS:
        path = block_dir / "full_steps" / f"{arm}_steps.csv"
        present = path.exists() and path.stat().st_size > 0
        checks[f"full_steps/{arm}_steps.csv"] = {
            "present": present,
            "size_bytes": path.stat().st_size if path.exists() else 0,
        }
        if not present:
            raise AssertionError(f"Missing native step log: {path}")
    return checks


def run_block(
    exp: Any,
    phase0: Any,
    phase2b: Any,
    adapter: Any,
    historical: Any,
    osqp_module: Any,
    spec: BlockSpec,
    schedule_hashes: dict[int, str],
    verified_inputs: list[dict[str, str]],
    rg_paths: list[Path],
) -> tuple[Path, dict[str, Any]]:
    block_dir = OUTPUT / "blocks" / spec.name
    block_dir.mkdir(parents=True)
    (block_dir / "full_steps").mkdir()
    episodes: list[dict[str, Any]] = []
    state_rows: list[dict[str, Any]] = []
    vx_rows: list[dict[str, Any]] = []
    binding_rows: list[dict[str, Any]] = []
    early_rows: list[dict[str, Any]] = []
    timing: dict[str, dict[str, list[float]]] = {
        arm: defaultdict(list) for arm in FORMAL_ARMS
    }
    slack_sum = {arm: Counter() for arm in FORMAL_ARMS}
    slack_max = {arm: Counter() for arm in FORMAL_ARMS}
    active_counts = {arm: Counter() for arm in FORMAL_ARMS}
    step_fields: dict[str, list[str] | None] = {arm: None for arm in ALL_ARMS}
    fallback_fields: list[str] | None = None
    mismatch_fields: list[str] | None = None
    ap_fields: list[str] | None = None
    comparator_lookup = {item[0]: item for item in COMPARATOR_ARMS}
    start = time.perf_counter()

    fallback_path = block_dir / "infeasibility_fallback_ledger.csv"
    mismatch_path = block_dir / "model_mismatch_ledger.csv"
    ap_path = block_dir / "ap_intervention_branches.csv"

    for index, seed in enumerate(SEEDS, start=1):
        for arm in ALL_ARMS:
            if arm in FORMAL_ARMS:
                episode, rows = run_formal_episode(
                    exp,
                    phase0,
                    adapter,
                    osqp_module,
                    spec,
                    arm,
                    seed,
                )
                ap_rows: list[dict[str, Any]] = []
            else:
                episode, rows, ap_rows = run_comparator_episode(
                    exp,
                    phase2b,
                    historical,
                    spec,
                    comparator_lookup[arm],
                    seed,
                )
            verify_episode(episode, rows, schedule_hashes[seed])
            episodes.append(episode)
            path = block_dir / "full_steps" / f"{arm}_steps.csv"
            step_fields[arm] = append_csv(path, rows, step_fields[arm])
            states, vx = state_episode_rows(spec, arm, seed, rows)
            state_rows.extend(states)
            vx_rows.append(vx)

            if arm == "AP-N3":
                if ap_rows:
                    ap_fields = append_csv(ap_path, ap_rows, ap_fields)

            if arm in FORMAL_ARMS:
                binding_rows.append(
                    binding_episode_row(spec, arm, seed, episode, rows)
                )
                early_rows.append(early_warning_episode(spec, arm, seed, rows))
                fallback = [
                    row for row in rows if as_bool(row["formal_fallback_used"])
                ]
                mismatch = [
                    row
                    for row in rows
                    if as_bool(row["formal_predicted_safe_realized_unsafe"])
                ]
                fallback_fields = append_csv(
                    fallback_path,
                    fallback,
                    fallback_fields or list(rows[0]),
                )
                mismatch_fields = append_csv(
                    mismatch_path,
                    mismatch,
                    mismatch_fields or list(rows[0]),
                )
                for row in rows:
                    timing[arm]["total"].append(
                        float(row["formal_total_selection_wall_ms"])
                    )
                    timing[arm]["hard_wall"].append(
                        float(row["formal_hard_qp_solve_wall_ms"])
                    )
                    timing[arm]["hard_reported"].append(
                        float(row["formal_hard_qp_solver_run_time_ms"])
                    )
                    timing[arm]["hard_iterations"].append(
                        float(row["formal_hard_qp_iterations"])
                    )
                    if row["formal_fallback_solve_wall_ms"] not in ("", None):
                        timing[arm]["fallback_wall"].append(
                            float(row["formal_fallback_solve_wall_ms"])
                        )
                        timing[arm]["fallback_iterations"].append(
                            float(row["formal_fallback_iterations"])
                        )
                    slack = parse_json(row["formal_slack_by_constraint"])
                    for constraint in CONSTRAINTS:
                        value = float(slack[constraint])
                        slack_sum[arm][constraint] += value
                        slack_max[arm][constraint] = max(
                            slack_max[arm][constraint],
                            value,
                        )
                    for constraint in parse_json(row["formal_active_constraints"]):
                        active_counts[arm][constraint] += 1

        if index == 1 or index % 10 == 0 or index == len(SEEDS):
            print(
                f"{spec.name} progress: {index}/{len(SEEDS)} seeds; "
                f"elapsed={time.perf_counter() - start:.1f}s",
                flush=True,
            )

    expected_cells = {(arm, seed) for arm in ALL_ARMS for seed in SEEDS}
    actual_cells = {(row["arm"], int(row["seed"])) for row in episodes}
    if actual_cells != expected_cells or len(episodes) != len(expected_cells):
        raise AssertionError(f"{spec.name}: incomplete/duplicated arm-by-seed grid")
    for seed in SEEDS:
        hashes = {
            row["delay_schedule_hash"]
            for row in episodes
            if int(row["seed"]) == seed
        }
        if hashes != {schedule_hashes[seed]}:
            raise AssertionError(f"{spec.name}: schedule mismatch seed={seed}")

    episode_fields = list(episodes[0])
    write_csv(block_dir / "episodes.csv", episodes, episode_fields)
    summary = summarize_episodes(spec, episodes)
    write_csv(block_dir / "summary.csv", summary, list(summary[0]))
    full_state = summarize_full_state(state_rows)
    write_csv(
        block_dir / "full_state_census.csv",
        full_state,
        list(full_state[0]),
    )
    write_csv(
        block_dir / "vx_bound_episode_census.csv",
        vx_rows,
        list(vx_rows[0]),
    )
    vx_summary = summarize_vx_bounds(vx_rows)
    write_csv(
        block_dir / "vx_bound_summary.csv",
        vx_summary,
        list(vx_summary[0]),
    )
    write_csv(
        block_dir / "binding_channel_episode_census.csv",
        binding_rows,
        list(binding_rows[0]),
    )
    binding_summary = summarize_binding(binding_rows)
    write_csv(
        block_dir / "binding_channel_summary.csv",
        binding_summary,
        list(binding_summary[0]),
    )
    early_summary = summarize_early_warning(early_rows)
    write_csv(
        block_dir / "vx_upper_early_warning_census.csv",
        early_summary,
        list(early_summary[0]),
    )
    timing_rows = solver_timing_summary(spec, timing)
    write_csv(
        block_dir / "solver_timing.csv",
        timing_rows,
        list(timing_rows[0]),
    )

    slack_rows = []
    active_rows = []
    for arm in FORMAL_ARMS:
        for constraint in CONSTRAINTS:
            slack_rows.append(
                {
                    "block": spec.name,
                    "fault_profile": spec.fault_profile,
                    "pipeline": spec.pipeline,
                    "arm": arm,
                    "constraint": constraint,
                    "slack_sum": slack_sum[arm][constraint],
                    "slack_max": slack_max[arm][constraint],
                }
            )
            active_rows.append(
                {
                    "block": spec.name,
                    "fault_profile": spec.fault_profile,
                    "pipeline": spec.pipeline,
                    "arm": arm,
                    "constraint": constraint,
                    "active_count": active_counts[arm][constraint],
                }
            )
    write_csv(
        block_dir / "constraint_slack_summary.csv",
        slack_rows,
        list(slack_rows[0]),
    )
    write_csv(
        block_dir / "active_constraint_census.csv",
        active_rows,
        list(active_rows[0]),
    )

    failure_families = family_inventory(
        episodes,
        lambda episode: as_bool(episode["oob"]),
        "failure",
    )
    intervention_families = family_inventory(
        episodes,
        lambda episode: float(episode["intervention_count"]) > 0,
        "intervention",
    )
    fallback_families = family_inventory(
        episodes,
        lambda episode: float(episode.get("formal_fallback_count") or 0) > 0,
        "fallback",
    )
    write_csv(
        block_dir / "failure_family_inventory.csv",
        failure_families,
        FAMILY_FIELDS,
    )
    write_csv(
        block_dir / "intervention_family_inventory.csv",
        intervention_families,
        FAMILY_FIELDS,
    )
    write_csv(
        block_dir / "fallback_family_inventory.csv",
        fallback_families,
        FAMILY_FIELDS,
    )

    paired = paired_results(spec, episodes, vx_rows)
    write_json(block_dir / "paired_outcomes_and_tests.json", paired)
    runtime_seconds = time.perf_counter() - start
    runtime = {
        "authorization": AUTHORIZATION,
        "block": spec.name,
        "fault_profile": spec.fault_profile,
        "pipeline": spec.pipeline,
        "seed_start": SEEDS[0],
        "seed_end": SEEDS[-1],
        "steps": STEPS,
        "delay_mode": "uniform_0_2",
        "gamma": GAMMA,
        "arms": list(ALL_ARMS),
        "python": sys.version,
        "platform": platform.platform(),
        "numpy_version": np.__version__,
        "osqp_version": str(osqp_module.__version__),
        "runtime_seconds": runtime_seconds,
        "runner_sha256": sha256(Path(__file__)),
        "exact_helper_sha256": sha256(EXACT_HELPER),
        "original_prereg_sha256": EXPECTED_ORIGINAL_PREREG_SHA,
        "amended_prereg_sha256": EXPECTED_AMENDED_PREREG_SHA,
    }
    write_json(block_dir / "runtime_config.json", runtime)

    if ap_fields is None:
        write_csv(
            ap_path,
            [],
            [
                "block",
                "pipeline",
                "arm",
                "seed",
                "step",
                "projection_feasible",
                "selected_candidate",
                "fallback_used",
            ],
        )

    checks = section10_check(block_dir, final=False)
    gates = {
        "block": spec.name,
        "seed_hash": {
            "pass": True,
            "episode_cells": len(episodes),
            "matched_seed_hashes": len(SEEDS),
            "prior_overlap_count": 0,
        },
        "rows": {
            "pass": True,
            "episode_rows": len(episodes),
            "step_rows_by_arm": {
                arm: sum(int(row["steps_recorded"]) for row in episodes if row["arm"] == arm)
                for arm in ALL_ARMS
            },
        },
        "no_silent_fallback": {"pass": True},
        "conversion": {"pass": True},
        "certificate_accounting": {"pass": True},
        "comparator_anchors": {
            "pass": True,
            "arms": ["E-N4", "AP-N3", "D-N3"],
        },
        "amendment1": {
            "pass": True,
            "criterion_unmodified": True,
            "class_counts": {
                row["arm"]: {
                    "vx_upper_dominant_n": row["vx_upper_dominant_n"],
                    "theta_dominant_n": row["theta_dominant_n"],
                    "mixed_or_tied_n": row["mixed_or_tied_n"],
                    "no_uncertifiable_states_n": row["no_uncertifiable_states_n"],
                    "registered_discriminator_supported": row[
                        "registered_discriminator_supported"
                    ],
                }
                for row in binding_summary
            },
        },
        "vx_registered_descriptive": {
            "pass": True,
            "all_arms": list(ALL_ARMS),
            "both_bounds": [-1.0, 1.0],
            "both_tails": True,
            "HF2_remains_reverse_channel_only": True,
        },
        "section10_native_outputs": {
            "pass": True,
            "files": checks,
            "manifest_written_and_verified_before_completion": True,
        },
    }
    write_json(block_dir / "gates.json", gates)
    report_lines = [
        "===================================================",
        f"V6 CONFIRMATION BLOCK: {spec.name}",
        "===================================================",
        "",
        f"Authorization: {AUTHORIZATION}",
        f"Seeds: {SEEDS[0]}-{SEEDS[-1]}",
        f"Fault profile: {spec.fault_profile}",
        f"Pipeline: {spec.pipeline}",
        f"Runtime seconds: {runtime_seconds:.17g}",
        "",
        "ENDPOINTS",
        *(compact_json(row) for row in summary),
        "",
        "AMENDMENT-1",
        *(compact_json(row) for row in binding_summary),
        "",
        "VX BOUND CENSUS",
        *(compact_json(row) for row in vx_summary),
        "",
        "PAIRED OUTPUT",
        json.dumps(paired, indent=2, sort_keys=True),
        "",
        "SECTION-10 NATIVE OUTPUTS: PASS",
        "No interpretation or advancement decision is made.",
        "AWAITING_SIGNOFF",
    ]
    (block_dir / "block_report.txt").write_text(
        "\n".join(report_lines) + "\n",
        encoding="utf-8",
    )

    # Recheck after gates/report exist, then freeze the complete block payload.
    section10_check(block_dir, final=True)
    result_paths = [
        path
        for path in block_dir.rglob("*")
        if path.is_file() and path.name != "BLOCK_MANIFEST.sha256"
    ]
    source_paths = [
        HARNESS,
        ADAPTER,
        PHASE0,
        PHASE2B,
        PILOT_RUNNER,
        EXACT_HELPER,
        Path(__file__),
        PREREG_ORIGINAL,
        PREREG_AMENDED,
        AMENDMENT_MANIFEST,
        AP_SOURCE,
        FREEZE_MANIFEST,
        *rg_paths,
    ]
    manifest = block_dir / "BLOCK_MANIFEST.sha256"
    write_manifest(source_paths + result_paths, manifest)
    verified_count = verify_manifest(manifest)
    print(
        f"{spec.name} COMPLETE: manifest {verified_count}/{verified_count}; "
        "AWAITING_SIGNOFF",
        flush=True,
    )
    return manifest, {
        "block": spec.name,
        "runtime_seconds": runtime_seconds,
        "manifest": str(manifest),
        "manifest_sha256": sha256(manifest),
        "manifest_entries": verified_count,
        "summary": summary,
        "binding_summary": binding_summary,
        "vx_summary": vx_summary,
        "paired": paired,
    }


def main() -> int:
    if AUTHORIZATION != "AUTHORIZE-V6-CONFIRMATION":
        raise AssertionError("Confirmation authorization token mismatch")
    if OUTPUT.exists():
        raise AssertionError(f"Refusing to overwrite confirmation output: {OUTPUT}")
    verified_inputs = verify_frozen_inputs()
    verify_persistent_profile()
    pilot.ensure_runtime_paths()
    phase0 = pilot.import_module(PHASE0, "v6_confirmation_phase0")
    phase2b = pilot.import_module(PHASE2B, "v6_confirmation_phase2b")
    adapter = pilot.import_module(ADAPTER, "v6_confirmation_adapter")
    historical = pilot.import_module(AP_SOURCE, "v6_confirmation_historical")
    exp = historical.load_experiment_module(correct_thrust_decay=True)
    historical.add_condition_specs(exp)
    osqp_module = phase0.load_osqp()

    OUTPUT.mkdir(parents=True)
    (OUTPUT / "regression_gate").mkdir()
    (OUTPUT / "blocks").mkdir()
    run_start = time.perf_counter()

    # This is the confirmation harness instance; RG precedes every F episode.
    (
        rg_rows,
        normalization_events,
        normalization_census,
        ap_branch_rows,
    ) = pilot.run_regression_gate(exp, phase2b, historical)
    rg_paths = [
        OUTPUT / "regression_gate" / "rg_bit_identical_results.csv",
        OUTPUT / "regression_gate" / "rg_null_normalization_events.csv",
        OUTPUT / "regression_gate" / "rg_null_normalization_census.csv",
        OUTPUT / "regression_gate" / "rg_ap_intervention_branch_comparison.csv",
    ]
    write_csv(rg_paths[0], rg_rows, list(rg_rows[0]))
    write_csv(
        rg_paths[1],
        normalization_events,
        list(normalization_events[0]),
    )
    write_csv(
        rg_paths[2],
        normalization_census,
        list(normalization_census[0]),
    )
    write_csv(rg_paths[3], ap_branch_rows, list(ap_branch_rows[0]))
    print(f"CONFIRMATION RG PASS: {len(rg_rows)}/{len(RG_SEEDS) * len(RG_ARMS)}")

    schedule_hashes, hash_gate = confirmation_hash_inventory(exp)
    hash_inventory_path = OUTPUT / "confirmation_delay_hash_inventory.csv"
    write_csv(
        hash_inventory_path,
        [
            {"seed": seed, "delay_schedule_hash": schedule_hashes[seed]}
            for seed in SEEDS
        ],
        ["seed", "delay_schedule_hash"],
    )
    hash_gate_path = OUTPUT / "confirmation_delay_hash_disjointness.json"
    write_json(hash_gate_path, hash_gate)
    print(
        f"DELAY-HASH GATE PASS: {len(schedule_hashes)}/{len(SEEDS)} matched; "
        "prior overlap=0",
        flush=True,
    )

    block_manifests = []
    block_results = []
    for spec in BLOCKS:
        manifest, result = run_block(
            exp,
            phase0,
            phase2b,
            adapter,
            historical,
            osqp_module,
            spec,
            schedule_hashes,
            verified_inputs,
            rg_paths,
        )
        block_manifests.append(manifest)
        block_results.append(result)

    total_runtime = time.perf_counter() - run_start
    completion = {
        "authorization": AUTHORIZATION,
        "status": "AWAITING_SIGNOFF",
        "runner_sha256": sha256(Path(__file__)),
        "exact_helper_sha256": sha256(EXACT_HELPER),
        "freeze_manifest_sha256": sha256(FREEZE_MANIFEST),
        "seed_start": SEEDS[0],
        "seed_end": SEEDS[-1],
        "gamma": GAMMA,
        "blocks": block_results,
        "RG": {
            "pass": True,
            "cells": len(rg_rows),
            "expected_cells": len(RG_SEEDS) * len(RG_ARMS),
        },
        "delay_hash_gate": hash_gate,
        "total_runtime_seconds": total_runtime,
        "section10_native_outputs_all_blocks": True,
        "confirmation_runner_reports_completion_only_after_manifest_verification": True,
    }
    completion_path = OUTPUT / "confirmation_completion.json"
    write_json(completion_path, completion)
    report_path = OUTPUT / "confirmation_report.txt"
    report_path.write_text(
        "\n".join(
            [
                "===================================================",
                "V6 CBF-QP CONFIRMATION — MECHANICAL REPORT",
                "===================================================",
                "",
                f"Authorization: {AUTHORIZATION}",
                f"Seeds: {SEEDS[0]}-{SEEDS[-1]}",
                f"Gamma: {GAMMA:.17g}",
                f"RG: {len(rg_rows)}/{len(RG_SEEDS) * len(RG_ARMS)} PASS",
                "Delay hashes: 1000/1000 matched, zero prior overlap",
                f"Total runtime seconds: {total_runtime:.17g}",
                "",
                *(
                    f"{item['block']}: manifest "
                    f"{item['manifest_entries']}/{item['manifest_entries']} verified"
                    for item in block_results
                ),
                "",
                "Section-10 native-output completion gate: PASS for all blocks",
                "No interpretation or advancement decision is made.",
                "AWAITING_SIGNOFF",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    top_manifest = OUTPUT / "CONFIRMATION_MANIFEST.sha256"
    top_paths = [
        HARNESS,
        ADAPTER,
        PHASE0,
        PHASE2B,
        PILOT_RUNNER,
        EXACT_HELPER,
        Path(__file__),
        PREREG_ORIGINAL,
        PREREG_AMENDED,
        AMENDMENT_MANIFEST,
        AP_SOURCE,
        FREEZE_MANIFEST,
        hash_inventory_path,
        hash_gate_path,
        completion_path,
        report_path,
        *rg_paths,
        *block_manifests,
    ]
    write_manifest(top_paths, top_manifest)
    count = verify_manifest(top_manifest)
    print(report_path.read_text(encoding="utf-8"), end="")
    print(f"Top-level manifest: {count}/{count} verified")
    print(f"Manifest: {top_manifest}")
    print("AWAITING_SIGNOFF")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
