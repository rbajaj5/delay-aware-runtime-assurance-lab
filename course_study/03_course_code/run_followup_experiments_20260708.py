#!/usr/bin/env python3
"""Run RAG-suggested follow-up experiments without modifying the main harness.

This script imports the existing July 6 quadrotor delay harness and injects
temporary condition variants by monkey-patching only within one episode call.
The original P0/Markov/thrust code paths remain unchanged.

The experiments are intentionally labeled as screening/diagnostic arms:
  * AP_candidate_projection: candidate-library action projection proxy.
  * E_queue_empty / E_queue_shift_plus1: queue-observability/model mismatch.
  * E_xdot_bias +/- 0.2 and E_stale_state1: scorer-state perturbations.
  * Thrust_Etrigger_Dchoice: E trigger retained; D candidate choice on second
    delay-aware call of each warning step under thrust decay.

These are not formal CBF/reachability filters or full sensor-fault monitors.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import importlib.util
import json
import math
import os
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator

import numpy as np


ROOT = Path(r"C:/Users/anaxe/Documents/Codex/2026-07-06/thrust-degradation-expansion-pre-registered-protocol")
RAG_ROOT = Path(r"C:/Users/anaxe/RAGforAssurredAutonomy")
PY_ENV_ROOT = RAG_ROOT / ".mamba-safe-control-gym-3d"
EXPERIMENT_PATH = RAG_ROOT / "tools" / "quadrotor_delay_ra_experiment.py"
OUT_ROOT = ROOT / "analysis" / "followup_experiments_20260708"


def ensure_windows_dll_path() -> None:
    parts = [
        str(PY_ENV_ROOT / "Library" / "bin"),
        str(PY_ENV_ROOT / "Scripts"),
        str(PY_ENV_ROOT),
    ]
    os.environ["PATH"] = ";".join(parts + [os.environ.get("PATH", "")])


def load_experiment_module(correct_thrust_decay: bool = True) -> Any:
    ensure_windows_dll_path()
    for path in (RAG_ROOT / "tools", RAG_ROOT / "src"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    spec = importlib.util.spec_from_file_location("followup_quadrotor_experiment", EXPERIMENT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load experiment module: {EXPERIMENT_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["followup_quadrotor_experiment"] = module
    spec.loader.exec_module(module)

    if correct_thrust_decay:
        def corrected_apply_thrust_effectiveness_decay(env: Any, normalized_action: Any, effectiveness: float) -> np.ndarray:
            action = np.asarray(normalized_action, dtype=float).reshape(-1)
            physical = np.asarray(env.denormalize_action(action), dtype=float).reshape(-1)
            decayed = float(effectiveness) * physical
            return np.asarray(env.normalize_action(decayed), dtype=float).reshape(action.shape)

        module.apply_thrust_effectiveness_decay = corrected_apply_thrust_effectiveness_decay
    return module


def csv_write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
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
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def parse_json_cell(value: Any, default: Any) -> Any:
    if value is None:
        return default
    if isinstance(value, (list, dict)):
        return value
    text = str(value).strip()
    if not text:
        return default
    try:
        return json.loads(text)
    except Exception:
        return default


def rounded_sequence_hash(rows: list[dict[str, Any]], places: int, terminal_only: bool = False) -> str:
    seq: list[Any] = []
    source_rows = rows[-1:] if terminal_only and rows else rows
    for row in source_rows:
        state = parse_json_cell(row.get("state"), [])
        rounded = [round(float(x), places) for x in state]
        seq.append(rounded)
    raw = json.dumps(seq, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


def mcnemar_exact_two_sided(a_only: int, b_only: int) -> float:
    n = int(a_only + b_only)
    if n == 0:
        return 1.0
    k = min(int(a_only), int(b_only))
    cdf = sum(math.comb(n, i) for i in range(k + 1)) / (2**n)
    return min(1.0, 2.0 * cdf)


@dataclass(frozen=True)
class Arm:
    name: str
    base_condition: str
    patch_mode: str = "none"
    disturbance_mode: str = "none"
    thrust_decay_final: float = 0.70


@contextlib.contextmanager
def patched_arm(exp: Any, arm: Arm) -> Iterator[None]:
    original_choose_aware = exp.choose_delay_aware_action
    original_score_aware = exp.score_candidate_delay_aware

    def projection_choose(
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
    ) -> tuple[np.ndarray, str, float, list[dict[str, Any]]]:
        ppo = np.clip(np.asarray(ppo_action, dtype=float).reshape(-1), -1.0, 1.0)
        scored: list[dict[str, Any]] = []
        for name, action in exp.candidate_actions(ppo_action, state):
            score, details = original_score_aware(
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
            )
            distance = float(np.linalg.norm(np.asarray(action, dtype=float).reshape(-1) - ppo))
            feasible = float(details["predicted_state_severity"]) <= 1e-9
            # Candidate-library proxy for action projection: prefer predicted-safe
            # candidates closest to PPO; fall back to the normal severity score.
            projection_score = (0.0 if feasible else 1_000_000.0 + score) + distance
            scored.append(
                {
                    "name": name,
                    "action": np.asarray(action, dtype=float),
                    "score": projection_score,
                    "raw_delay_aware_score": score,
                    "projection_distance_to_ppo": distance,
                    "projection_feasible": feasible,
                    **details,
                }
            )
        scored.sort(key=lambda item: (float(item["score"]), str(item["name"])))
        best = scored[0]
        json_scores = [
            {
                "name": item["name"],
                "score": float(item["score"]),
                "raw_delay_aware_score": float(item["raw_delay_aware_score"]),
                "projection_distance_to_ppo": float(item["projection_distance_to_ppo"]),
                "projection_feasible": bool(item["projection_feasible"]),
                "action": item["action"].tolist(),
                "predicted_state_severity": float(item["predicted_state_severity"]),
                "predicted_tracking_rmse": float(item["predicted_tracking_rmse"]),
                "candidate_boundary_penalty": float(item.get("candidate_boundary_penalty", 0.0)),
                "candidate_slew_rate_penalty": float(item.get("candidate_slew_rate_penalty", 0.0)),
            }
            for item in scored
        ]
        return np.asarray(best["action"], dtype=float), str(best["name"]), float(best["predicted_state_severity"]), json_scores

    def queue_empty_choose(*args: Any, **kwargs: Any) -> tuple[np.ndarray, str, float, list[dict[str, Any]]]:
        args = list(args)
        # Positional layout: env, state, ppo_action, queue_state, current_step, ...
        if len(args) >= 4:
            args[3] = []
        if "queue_state" in kwargs:
            kwargs["queue_state"] = []
        return original_choose_aware(*args, **kwargs)

    def queue_shift_choose(*args: Any, **kwargs: Any) -> tuple[np.ndarray, str, float, list[dict[str, Any]]]:
        args = list(args)
        if len(args) >= 4:
            queue_state = args[3]
            shifted = []
            for item in queue_state:
                copy = dict(item)
                if "release_step" in copy:
                    copy["release_step"] = int(copy["release_step"]) + 1
                shifted.append(copy)
            args[3] = shifted
        if len(args) >= 6:
            args[5] = min(2, int(args[5]) + 1)
        if "queue_state" in kwargs:
            shifted = []
            for item in kwargs["queue_state"]:
                copy = dict(item)
                if "release_step" in copy:
                    copy["release_step"] = int(copy["release_step"]) + 1
                shifted.append(copy)
            kwargs["queue_state"] = shifted
        if "delay_sample" in kwargs:
            kwargs["delay_sample"] = min(2, int(kwargs["delay_sample"]) + 1)
        return original_choose_aware(*args, **kwargs)

    def xdot_bias_choose(delta: float) -> Callable[..., tuple[np.ndarray, str, float, list[dict[str, Any]]]]:
        def wrapped(*args: Any, **kwargs: Any) -> tuple[np.ndarray, str, float, list[dict[str, Any]]]:
            args = list(args)
            if len(args) >= 2:
                state = np.asarray(args[1], dtype=float).copy()
                if state.size > 1:
                    state[1] += delta
                args[1] = state
            elif "state" in kwargs:
                state = np.asarray(kwargs["state"], dtype=float).copy()
                if state.size > 1:
                    state[1] += delta
                kwargs["state"] = state
            return original_choose_aware(*args, **kwargs)

        return wrapped

    def stale_state_choose() -> Callable[..., tuple[np.ndarray, str, float, list[dict[str, Any]]]]:
        last_true_state: np.ndarray | None = None
        cached_step: int | None = None
        cached_observed_state: np.ndarray | None = None

        def wrapped(*args: Any, **kwargs: Any) -> tuple[np.ndarray, str, float, list[dict[str, Any]]]:
            nonlocal last_true_state, cached_step, cached_observed_state
            args = list(args)
            state = np.asarray(args[1] if len(args) >= 2 else kwargs["state"], dtype=float).copy()
            current_step = int(args[4] if len(args) >= 5 else kwargs["current_step"])
            if cached_step != current_step:
                cached_observed_state = last_true_state.copy() if last_true_state is not None else state.copy()
                cached_step = current_step
                last_true_state = state.copy()
            observed = cached_observed_state.copy() if cached_observed_state is not None else state
            if len(args) >= 2:
                args[1] = observed
            else:
                kwargs["state"] = observed
            return original_choose_aware(*args, **kwargs)

        return wrapped

    def etrigger_dchoice_choose() -> Callable[..., tuple[np.ndarray, str, float, list[dict[str, Any]]]]:
        call_counts: dict[int, int] = defaultdict(int)

        def wrapped(
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
        ) -> tuple[np.ndarray, str, float, list[dict[str, Any]]]:
            call_counts[int(current_step)] += 1
            if call_counts[int(current_step)] == 1:
                return original_choose_aware(
                    env,
                    state,
                    ppo_action,
                    queue_state,
                    current_step,
                    delay_sample,
                    reference,
                    last_applied_action=last_applied_action,
                    velocity_boundary_weight=velocity_boundary_weight,
                    boundary_penalty_target=boundary_penalty_target,
                    boundary_penalty_anti=boundary_penalty_anti,
                    slew_rate_weight=slew_rate_weight,
                )
            action, name, pred, scores = exp.choose_delay_blind_action(env, state, ppo_action, reference)
            for item in scores:
                item["etrigger_dchoice_second_call"] = True
            return action, f"Dchoice_{name}", pred, scores

        return wrapped

    try:
        if arm.patch_mode == "candidate_projection":
            exp.choose_delay_aware_action = projection_choose
        elif arm.patch_mode == "queue_empty":
            exp.choose_delay_aware_action = queue_empty_choose
        elif arm.patch_mode == "queue_shift_plus1":
            exp.choose_delay_aware_action = queue_shift_choose
        elif arm.patch_mode == "xdot_bias_plus0p2":
            exp.choose_delay_aware_action = xdot_bias_choose(+0.2)
        elif arm.patch_mode == "xdot_bias_minus0p2":
            exp.choose_delay_aware_action = xdot_bias_choose(-0.2)
        elif arm.patch_mode == "stale_state1":
            exp.choose_delay_aware_action = stale_state_choose()
        elif arm.patch_mode == "etrigger_dchoice":
            exp.choose_delay_aware_action = etrigger_dchoice_choose()
        yield
    finally:
        exp.choose_delay_aware_action = original_choose_aware


def add_condition_specs(exp: Any) -> None:
    exp.CONDITIONS.setdefault(
        "AP_candidate_projection_intervention",
        {
            "delay": "uniform_0_2",
            "monitor": "delay_aware",
            "intervention": "delay_aware_candidate_projection",
            "held_action_corrected": True,
            "boundary_penalty_target": "none",
            "boundary_penalty_anti": False,
        },
    )
    for name in [
        "E_queue_empty_intervention",
        "E_queue_shift_plus1_intervention",
        "E_xdot_bias_plus0p2_intervention",
        "E_xdot_bias_minus0p2_intervention",
        "E_stale_state1_intervention",
        "Thrust_Etrigger_Dchoice_intervention",
    ]:
        exp.CONDITIONS.setdefault(
            name,
            {
                "delay": "uniform_0_2",
                "monitor": "delay_aware",
                "intervention": "delay_aware_" + name,
                "held_action_corrected": True,
                "boundary_penalty_target": "none",
                "boundary_penalty_anti": False,
            },
        )


def run_arm(exp: Any, arm: Arm, seed: int, steps: int, warning_buffer: float) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    with patched_arm(exp, arm):
        ep = exp.run_episode(
            condition=arm.base_condition,
            seed=seed,
            steps=steps,
            warning_buffer=warning_buffer,
            output_rows=rows,
            delayed_delay_mode="uniform_0_2",
            disturbance_mode=arm.disturbance_mode,
            thrust_decay_final=arm.thrust_decay_final,
        )
    original_condition = ep["condition"]
    ep["base_condition"] = original_condition
    ep["condition"] = arm.name
    ep["patch_mode"] = arm.patch_mode
    ep["sequence_hash_12"] = rounded_sequence_hash(rows, 12, terminal_only=False)
    ep["terminal_hash_12"] = rounded_sequence_hash(rows, 12, terminal_only=True)
    ep["terminal_hash_10"] = rounded_sequence_hash(rows, 10, terminal_only=True)
    for row in rows:
        row["base_condition"] = row["condition"]
        row["condition"] = arm.name
        row["patch_mode"] = arm.patch_mode
    return ep, rows


def summarize(episodes: list[dict[str, Any]], baseline: str = "D_delay_blind_intervention") -> dict[str, Any]:
    by_condition: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for ep in episodes:
        by_condition[str(ep["condition"])].append(ep)
    condition_rows: list[dict[str, Any]] = []
    for condition, rows in sorted(by_condition.items()):
        sev = np.asarray([float(r["cumulative_normalized_violation_severity"]) for r in rows], dtype=float)
        rewards = np.asarray([float(r["cumulative_reward"]) for r in rows], dtype=float)
        interventions = np.asarray([float(r["intervention_count"]) for r in rows], dtype=float)
        steps = np.asarray([float(r["steps_recorded"]) for r in rows], dtype=float)
        oobs = [r for r in rows if str(r.get("done_reason")) == "out_of_bounds"]
        condition_rows.append(
            {
                "condition": condition,
                "n": len(rows),
                "oob_count": len(oobs),
                "oob_rate": len(oobs) / len(rows) if rows else float("nan"),
                "mean_severity": float(np.nanmean(sev)) if sev.size else float("nan"),
                "max_severity": float(np.nanmax(sev)) if sev.size else float("nan"),
                "mean_reward": float(np.nanmean(rewards)) if rewards.size else float("nan"),
                "mean_interventions": float(np.nanmean(interventions)) if interventions.size else float("nan"),
                "mean_steps_recorded": float(np.nanmean(steps)) if steps.size else float("nan"),
                "distinct_sequence_hash_12": len({str(r.get("sequence_hash_12", "")) for r in rows}),
                "distinct_terminal_hash_12": len({str(r.get("terminal_hash_12", "")) for r in rows}),
            }
        )

    paired_rows: list[dict[str, Any]] = []
    by_seed_condition = {(int(ep["seed"]), str(ep["condition"])): ep for ep in episodes}
    seeds = sorted({int(ep["seed"]) for ep in episodes})
    for condition in sorted(by_condition):
        if condition == baseline:
            continue
        both_fail = baseline_only = condition_only = neither = 0
        n_pairs = 0
        for seed in seeds:
            b = by_seed_condition.get((seed, baseline))
            c = by_seed_condition.get((seed, condition))
            if b is None or c is None:
                continue
            n_pairs += 1
            b_fail = str(b.get("done_reason")) == "out_of_bounds"
            c_fail = str(c.get("done_reason")) == "out_of_bounds"
            if b_fail and c_fail:
                both_fail += 1
            elif b_fail and not c_fail:
                baseline_only += 1
            elif c_fail and not b_fail:
                condition_only += 1
            else:
                neither += 1
        if n_pairs:
            paired_rows.append(
                {
                    "baseline": baseline,
                    "condition": condition,
                    "n_pairs": n_pairs,
                    "both_fail": both_fail,
                    "baseline_only": baseline_only,
                    "condition_only": condition_only,
                    "neither": neither,
                    "exact_mcnemar_p": mcnemar_exact_two_sided(baseline_only, condition_only),
                }
            )
    return {"condition_summary": condition_rows, "paired_summary": paired_rows}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-start", type=int, default=30000)
    parser.add_argument("--seeds", type=int, default=200)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--warning-buffer", type=float, default=0.05)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--skip-uniform", action="store_true")
    parser.add_argument("--skip-thrust", action="store_true")
    parser.add_argument("--raw-thrust-decay", action="store_true")
    parser.add_argument("--output-root", type=Path, default=OUT_ROOT)
    args = parser.parse_args()

    exp = load_experiment_module(correct_thrust_decay=not args.raw_thrust_decay)
    add_condition_specs(exp)

    seeds = [args.seed_start + i for i in range(args.seeds)]
    uniform_arms = [
        Arm("D_delay_blind_intervention", "D_delay_blind_intervention"),
        Arm("E_delay_aware_intervention", "E_delay_aware_intervention"),
        Arm("AP_candidate_projection_intervention", "AP_candidate_projection_intervention", "candidate_projection"),
        Arm("E_queue_empty_intervention", "E_queue_empty_intervention", "queue_empty"),
        Arm("E_queue_shift_plus1_intervention", "E_queue_shift_plus1_intervention", "queue_shift_plus1"),
        Arm("E_xdot_bias_plus0p2_intervention", "E_xdot_bias_plus0p2_intervention", "xdot_bias_plus0p2"),
        Arm("E_xdot_bias_minus0p2_intervention", "E_xdot_bias_minus0p2_intervention", "xdot_bias_minus0p2"),
        Arm("E_stale_state1_intervention", "E_stale_state1_intervention", "stale_state1"),
    ]
    thrust_arms = [
        Arm("Thrust_B_delayed_ppo", "B_delayed_ppo", disturbance_mode="thrust_decay"),
        Arm("Thrust_D_delay_blind_intervention", "D_delay_blind_intervention", disturbance_mode="thrust_decay"),
        Arm("Thrust_E_delay_aware_intervention", "E_delay_aware_intervention", disturbance_mode="thrust_decay"),
        Arm(
            "Thrust_Etrigger_Dchoice_intervention",
            "Thrust_Etrigger_Dchoice_intervention",
            "etrigger_dchoice",
            disturbance_mode="thrust_decay",
        ),
    ]
    if args.smoke:
        seeds = seeds[:2]

    out_dir = args.output_root / ("smoke" if args.smoke else f"run_seed{seeds[0]}_{seeds[-1]}_{time.strftime('%Y%m%d_%H%M%S')}")
    out_dir.mkdir(parents=True, exist_ok=True)
    episodes: list[dict[str, Any]] = []
    kept_steps: list[dict[str, Any]] = []
    raw_lines: list[str] = []
    raw_lines.append(
        json.dumps(
            {
                "experiment_path": str(EXPERIMENT_PATH),
                "output_dir": str(out_dir),
                "seed_start": seeds[0],
                "seed_end": seeds[-1],
                "n_seeds": len(seeds),
                "steps": args.steps,
                "warning_buffer": args.warning_buffer,
                "corrected_thrust_decay_patch": not args.raw_thrust_decay,
                "uniform_arms": [arm.__dict__ for arm in uniform_arms],
                "thrust_arms": [arm.__dict__ for arm in thrust_arms],
                "notes": [
                    "Action projection is a candidate-library proxy, not a continuous reachability shield.",
                    "State perturbation arms alter scorer inputs; current-warning trigger still sees true simulator constraints.",
                    "Thrust_Etrigger_Dchoice retains E preview on first delay-aware call and forces D candidate choice on the second warning-step call.",
                    "No policy-generalization run is included because only one quadrotor PPO checkpoint was found.",
                ],
            },
            sort_keys=True,
        )
    )

    start = time.perf_counter()
    for seed in seeds:
        if not args.skip_uniform:
            for arm in uniform_arms:
                print(f"running uniform seed={seed} arm={arm.name}", flush=True)
                ep, rows = run_arm(exp, arm, seed, args.steps, args.warning_buffer)
                episodes.append(ep)
                if ep.get("done_reason") == "out_of_bounds" or seed in seeds[:5]:
                    kept_steps.extend(rows)
        if not args.skip_thrust:
            thrust_seed = 1337 + (seed - seeds[0])
            for arm in thrust_arms:
                print(f"running thrust seed={thrust_seed} arm={arm.name}", flush=True)
                ep, rows = run_arm(exp, arm, thrust_seed, args.steps, args.warning_buffer)
                episodes.append(ep)
                if ep.get("done_reason") == "out_of_bounds" or thrust_seed in list(range(1337, 1342)):
                    kept_steps.extend(rows)

    runtime = time.perf_counter() - start
    summary = summarize(episodes)
    raw_lines.append(json.dumps({"runtime_seconds": runtime}, sort_keys=True))
    raw_lines.append("CONDITION_SUMMARY")
    for row in summary["condition_summary"]:
        raw_lines.append(json.dumps(row, sort_keys=True))
    raw_lines.append("PAIRED_SUMMARY_VS_D")
    for row in summary["paired_summary"]:
        raw_lines.append(json.dumps(row, sort_keys=True))

    csv_write_rows(out_dir / "followup_episodes.csv", episodes)
    csv_write_rows(out_dir / "followup_steps_kept_oob_plus_audit.csv", kept_steps)
    csv_write_rows(out_dir / "followup_condition_summary.csv", summary["condition_summary"])
    csv_write_rows(out_dir / "followup_paired_summary.csv", summary["paired_summary"])
    (out_dir / "followup_raw_output.txt").write_text("\n".join(raw_lines) + "\n", encoding="utf-8")
    (out_dir / "followup_manifest.json").write_text(
        json.dumps(
            {
                "script": str(Path(__file__).resolve()),
                "experiment_path": str(EXPERIMENT_PATH),
                "output_dir": str(out_dir),
                "runtime_seconds": runtime,
                "seed_start": seeds[0],
                "seed_end": seeds[-1],
                "n_seeds": len(seeds),
                "steps": args.steps,
                "smoke": args.smoke,
                "skip_uniform": args.skip_uniform,
                "skip_thrust": args.skip_thrust,
                "corrected_thrust_decay_patch": not args.raw_thrust_decay,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print("\n".join(raw_lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
