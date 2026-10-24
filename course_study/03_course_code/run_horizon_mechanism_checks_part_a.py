from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(r"C:\Users\anaxe\RAGforAssurredAutonomy")
STUDY_ROOT = ROOT / "analysis" / "out" / "horizon_spectrum_20260723"
PHASE2B_ROOT = STUDY_ROOT / "phase2b"
DX_ROOT = STUDY_ROOT / "pilot_diagnostics"
OUTPUT_ROOT = STUDY_ROOT / "mechanism_checks_readonly_20260723"
REPORT_PATH = OUTPUT_ROOT / "chkA_report.md"
CHK1_PATH = OUTPUT_ROOT / "chk1_rates.csv"
CHK2_PATH = OUTPUT_ROOT / "chk2_payload_rankings.csv"
MANIFEST_PATH = OUTPUT_ROOT / "CHK_MANIFEST.sha256"
SCRIPT_PATH = ROOT / "analysis" / "run_horizon_mechanism_checks_part_a.py"

DX_MANIFEST = DX_ROOT / "DX_MANIFEST.sha256"
PHASE2B_MANIFEST = PHASE2B_ROOT / "MANIFEST.sha256"

DX2_RATES = DX_ROOT / "dx2_intervention_rates.csv"
CONFIRMATION_EPISODES = PHASE2B_ROOT / "confirmation" / "episodes.csv"
CONFIRMATION_E3_STEPS = (
    PHASE2B_ROOT / "confirmation" / "full_steps" / "E-N3_steps.csv"
)
CONFIRMATION_E4_STEPS = (
    PHASE2B_ROOT / "confirmation" / "full_steps" / "E-N4_steps.csv"
)
E3_PAYLOAD = (
    PHASE2B_ROOT
    / "onset_payloads"
    / "confirmation_E-N3_steps20_35_scores.csv"
)
E4_PAYLOAD = (
    PHASE2B_ROOT
    / "onset_payloads"
    / "confirmation_E-N4_steps20_35_scores.csv"
)
PAYLOAD_SELECTION = PHASE2B_ROOT / "payload_selection.csv"

INPUTS = (
    DX2_RATES,
    CONFIRMATION_EPISODES,
    CONFIRMATION_E3_STEPS,
    CONFIRMATION_E4_STEPS,
    E3_PAYLOAD,
    E4_PAYLOAD,
    PAYLOAD_SELECTION,
)

PRECOMMITTED_READINGS = (
    "- Rates close (N4 within ~±15% of N3's mean) AND rankings flipped → "
    "the one-step fix is **scorer-side repricing**; the trigger channel did "
    'not materially move at the shallow end. The "accidental sweet spot / '
    'trigger neither too quiet nor too loud" hypothesis is rejected on '
    "existing data.",
    "- Rates materially higher at N4 AND rankings unchanged where "
    "interventions occur → **trigger-mediated** rescue; repricing story "
    "rejected.",
    "- Both moved → **entangled**; the 2×2 off-diagonals become the deciding "
    "evidence rather than confirmatory decoration.",
)

COMPONENT_WEIGHTS = {
    "constraint_violation": 100.0,
    "action_violation": 5.0,
    "rmse": 0.05,
    "l2_penalty": 0.005,
}
TARGET_CANDIDATES = ("gentle_down", "diff_left", "theta_correct_b")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def manifest_entries(path: Path) -> dict[str, str]:
    entries: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        if not raw_line.strip():
            continue
        recorded_hash, recorded_path = raw_line.split(" *", maxsplit=1)
        entries[str(Path(recorded_path)).lower()] = recorded_hash
    return entries


def verify_inputs() -> list[dict[str, str]]:
    recorded = {
        **manifest_entries(DX_MANIFEST),
        **manifest_entries(PHASE2B_MANIFEST),
    }
    verified: list[dict[str, str]] = []
    for path in INPUTS:
        if not path.exists():
            raise FileNotFoundError(path)
        expected = recorded.get(str(path).lower())
        if expected is None:
            raise AssertionError(f"No manifest hash records {path}")
        actual = sha256(path)
        if actual != expected:
            raise AssertionError(
                f"Hash mismatch for {path}: expected {expected}, got {actual}"
            )
        verified.append(
            {"path": str(path), "sha256": actual, "status": "PASS"}
        )
    return verified


def as_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series
    return series.astype(str).str.lower().isin({"true", "1", "yes"})


def summarize_rates(
    scope: str,
    per_seed: pd.DataFrame,
    raw_file_by_arm: dict[str, str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for (arm, horizon), group in per_seed.groupby(
        ["arm", "prediction_horizon"], sort=True
    ):
        rate = group["interventions_per_100_executed_steps"]
        rows.append(
            {
                "scope": scope,
                "arm": arm,
                "prediction_horizon": int(horizon),
                "n_seeds": int(group["seed"].nunique()),
                "mean_interventions_per_100_steps": float(rate.mean()),
                "min_interventions_per_100_steps": float(rate.min()),
                "max_interventions_per_100_steps": float(rate.max()),
                "p50_interventions_per_100_steps": float(rate.median()),
                "raw_file": raw_file_by_arm[arm],
            }
        )
    return rows


def build_chk1() -> tuple[pd.DataFrame, dict[str, float]]:
    pilot = pd.read_csv(DX2_RATES)
    required_pilot_arms = {"E-N3", "E-N4", "E-N5", "E-N7", "E-N10"}
    if set(pilot["arm"]) != required_pilot_arms:
        raise AssertionError(
            f"Unexpected pilot arms: {sorted(pilot['arm'].unique())}"
        )
    if pilot.groupby("arm")["seed"].nunique().ne(200).any():
        raise AssertionError("Pilot arm seed cardinality is not 200")

    confirmation_rows: list[pd.DataFrame] = []
    confirmation_paths = {
        "E-N3": CONFIRMATION_E3_STEPS,
        "E-N4": CONFIRMATION_E4_STEPS,
    }
    confirmation_horizons = {"E-N3": 3, "E-N4": 4}
    for arm, path in confirmation_paths.items():
        steps = pd.read_csv(
            path, usecols=["seed", "step", "intervention"]
        )
        steps["intervention"] = as_bool(steps["intervention"])
        per_seed = (
            steps.groupby("seed", as_index=False)
            .agg(
                executed_steps=("step", "size"),
                intervention_count=("intervention", "sum"),
            )
            .assign(
                arm=arm,
                prediction_horizon=confirmation_horizons[arm],
            )
        )
        per_seed["interventions_per_100_executed_steps"] = (
            100.0
            * per_seed["intervention_count"]
            / per_seed["executed_steps"]
        )
        confirmation_rows.append(per_seed)
    confirmation = pd.concat(confirmation_rows, ignore_index=True)
    if confirmation.groupby("arm")["seed"].nunique().ne(1000).any():
        raise AssertionError("Confirmation arm seed cardinality is not 1,000")

    summary = pd.DataFrame(
        summarize_rates(
            "pilot",
            pilot,
            {arm: str(DX2_RATES) for arm in required_pilot_arms},
        )
        + summarize_rates(
            "confirmation",
            confirmation,
            {arm: str(path) for arm, path in confirmation_paths.items()},
        )
    )
    summary.to_csv(CHK1_PATH, index=False)

    means = {
        f"{row.scope}_{row.arm}": float(
            row.mean_interventions_per_100_steps
        )
        for row in summary.itertuples(index=False)
    }
    return summary, means


def weighted_component_columns(frame: pd.DataFrame, suffix: str) -> None:
    for name, weight in COMPONENT_WEIGHTS.items():
        frame[f"{name}_weighted_{suffix}"] = frame[f"{name}_{suffix}"] * weight


def gap_change_term(
    rows: pd.DataFrame,
    competitor: str,
) -> tuple[float, float, bool, str, float]:
    indexed = rows.set_index("candidate_name")
    gentle = indexed.loc["gentle_down"]
    other = indexed.loc[competitor]
    gap_n3 = float(gentle["final_score_n3"] - other["final_score_n3"])
    gap_n4 = float(gentle["final_score_n4"] - other["final_score_n4"])
    order_flip = (gap_n3 < 0.0) != (gap_n4 < 0.0)
    changes: dict[str, float] = {}
    for component in COMPONENT_WEIGHTS:
        weighted = f"{component}_weighted"
        changes[component] = float(
            (
                gentle[f"{weighted}_n4"]
                - other[f"{weighted}_n4"]
            )
            - (
                gentle[f"{weighted}_n3"]
                - other[f"{weighted}_n3"]
            )
        )
    largest = max(changes, key=lambda name: abs(changes[name]))
    return gap_n3, gap_n4, order_flip, largest, changes[largest]


def ranking_string(rows: pd.DataFrame, suffix: str) -> str:
    ranked = rows.sort_values(f"candidate_rank_{suffix}")
    return " > ".join(
        f"{item.candidate_name}:{int(getattr(item, f'candidate_rank_{suffix}'))}"
        for item in ranked.itertuples(index=False)
    )


def build_chk2() -> tuple[pd.DataFrame, dict[str, Any]]:
    selection = pd.read_csv(PAYLOAD_SELECTION)
    matched = selection[
        (selection["block"] == "confirmation")
        & (selection["arm"] == "E-N4")
        & (
            selection["roles"]
            == "matched_E_N3_family_representative_survivor"
        )
    ].copy()
    seeds = sorted(int(seed) for seed in matched["seed"])
    if len(seeds) != 7:
        raise AssertionError(f"Expected seven matched seeds, got {seeds}")

    n3 = pd.read_csv(E3_PAYLOAD)
    n4 = pd.read_csv(E4_PAYLOAD)
    n3 = n3[n3["seed"].isin(seeds) & n3["step"].between(24, 32)].copy()
    n4 = n4[n4["seed"].isin(seeds) & n4["step"].between(24, 32)].copy()

    for label, frame in (("N3", n3), ("N4", n4)):
        counts = frame.groupby(["seed", "step"]).size()
        if len(counts) != 7 * 9 or not counts.eq(8).all():
            raise AssertionError(
                f"{label} payload is incomplete: groups={len(counts)}, "
                f"sizes={sorted(counts.unique())}"
            )
        names = frame.groupby(["seed", "step"])["candidate_name"].agg(set)
        if not all(len(value) == 8 for value in names):
            raise AssertionError(f"{label} candidate names are not unique")
        missing = set(TARGET_CANDIDATES) - set(frame["candidate_name"])
        if missing:
            raise AssertionError(f"{label} missing candidates: {missing}")

    merge_columns = [
        "seed",
        "step",
        "candidate_name",
        "candidate_action",
        "candidate_rank",
        "is_selected",
        "constraint_violation",
        "action_violation",
        "rmse",
        "l2_penalty",
        "final_score",
        "payload_role",
        "episode_oob",
    ]
    merged = n3[merge_columns].merge(
        n4[merge_columns],
        on=["seed", "step", "candidate_name"],
        how="inner",
        suffixes=("_n3", "_n4"),
        validate="one_to_one",
    )
    if len(merged) != 7 * 9 * 8:
        raise AssertionError(f"Unexpected merged payload rows: {len(merged)}")

    weighted_component_columns(merged, "n3")
    weighted_component_columns(merged, "n4")
    for suffix in ("n3", "n4"):
        reconstructed = sum(
            merged[f"{component}_weighted_{suffix}"]
            for component in COMPONENT_WEIGHTS
        )
        maximum_error = float(
            np.max(np.abs(reconstructed - merged[f"final_score_{suffix}"]))
        )
        if maximum_error > 1e-10:
            raise AssertionError(
                f"{suffix} score decomposition error {maximum_error}"
            )

    context_rows: list[dict[str, Any]] = []
    for (seed, step), rows in merged.groupby(["seed", "step"], sort=True):
        indexed = rows.set_index("candidate_name")
        winner_n3 = str(
            rows.loc[rows["candidate_rank_n3"].idxmin(), "candidate_name"]
        )
        winner_n4 = str(
            rows.loc[rows["candidate_rank_n4"].idxmin(), "candidate_name"]
        )
        inversion_n3 = bool(
            indexed.loc["gentle_down", "candidate_rank_n3"]
            < indexed.loc["diff_left", "candidate_rank_n3"]
            and indexed.loc["gentle_down", "candidate_rank_n3"]
            < indexed.loc["theta_correct_b", "candidate_rank_n3"]
        )
        inversion_n4 = bool(
            indexed.loc["gentle_down", "candidate_rank_n4"]
            < indexed.loc["diff_left", "candidate_rank_n4"]
            and indexed.loc["gentle_down", "candidate_rank_n4"]
            < indexed.loc["theta_correct_b", "candidate_rank_n4"]
        )
        diff = gap_change_term(rows, "diff_left")
        theta = gap_change_term(rows, "theta_correct_b")
        context_rows.append(
            {
                "seed": int(seed),
                "step": int(step),
                "winner_n3": winner_n3,
                "winner_n4": winner_n4,
                "winner_changed": winner_n3 != winner_n4,
                "full_ranking_n3": ranking_string(rows, "n3"),
                "full_ranking_n4": ranking_string(rows, "n4"),
                "gentle_down_rank_n3": int(
                    indexed.loc["gentle_down", "candidate_rank_n3"]
                ),
                "gentle_down_rank_n4": int(
                    indexed.loc["gentle_down", "candidate_rank_n4"]
                ),
                "diff_left_rank_n3": int(
                    indexed.loc["diff_left", "candidate_rank_n3"]
                ),
                "diff_left_rank_n4": int(
                    indexed.loc["diff_left", "candidate_rank_n4"]
                ),
                "theta_correct_b_rank_n3": int(
                    indexed.loc["theta_correct_b", "candidate_rank_n3"]
                ),
                "theta_correct_b_rank_n4": int(
                    indexed.loc["theta_correct_b", "candidate_rank_n4"]
                ),
                "n3_style_inversion_present_n3": inversion_n3,
                "n3_style_inversion_present_n4": inversion_n4,
                "gentle_down_vs_diff_left_gap_n3": diff[0],
                "gentle_down_vs_diff_left_gap_n4": diff[1],
                "gentle_down_vs_diff_left_order_flip": diff[2],
                "gentle_down_vs_diff_left_largest_gap_change_term": diff[3],
                "gentle_down_vs_diff_left_largest_gap_change": diff[4],
                "gentle_down_vs_theta_correct_b_gap_n3": theta[0],
                "gentle_down_vs_theta_correct_b_gap_n4": theta[1],
                "gentle_down_vs_theta_correct_b_order_flip": theta[2],
                "gentle_down_vs_theta_correct_b_largest_gap_change_term": (
                    theta[3]
                ),
                "gentle_down_vs_theta_correct_b_largest_gap_change": theta[4],
                "specified_pair_order_flip_any": bool(diff[2] or theta[2]),
            }
        )
    context = pd.DataFrame(context_rows)
    output = merged.merge(
        context, on=["seed", "step"], how="left", validate="many_to_one"
    )
    output = output.sort_values(
        ["seed", "step", "candidate_rank_n4", "candidate_name"]
    )
    output.to_csv(CHK2_PATH, index=False)

    flipped_terms: list[str] = []
    for row in context.itertuples(index=False):
        if row.gentle_down_vs_diff_left_order_flip:
            flipped_terms.append(
                row.gentle_down_vs_diff_left_largest_gap_change_term
            )
        if row.gentle_down_vs_theta_correct_b_order_flip:
            flipped_terms.append(
                row.gentle_down_vs_theta_correct_b_largest_gap_change_term
            )
    metrics = {
        "matched_seeds": seeds,
        "seed_step_pairs": int(len(context)),
        "candidate_rows": int(len(output)),
        "winner_changes": int(context["winner_changed"].sum()),
        "n3_inversion_count": int(
            context["n3_style_inversion_present_n3"].sum()
        ),
        "n4_inversion_count": int(
            context["n3_style_inversion_present_n4"].sum()
        ),
        "diff_left_order_flips": int(
            context["gentle_down_vs_diff_left_order_flip"].sum()
        ),
        "theta_correct_b_order_flips": int(
            context["gentle_down_vs_theta_correct_b_order_flip"].sum()
        ),
        "any_specified_pair_flip_pairs": int(
            context["specified_pair_order_flip_any"].sum()
        ),
        "flipped_pair_term_counts": {
            str(name): int(count)
            for name, count in pd.Series(flipped_terms).value_counts().items()
        },
        "winner_counts_n3": {
            str(name): int(count)
            for name, count in (
                context["winner_n3"].value_counts().sort_index().items()
            )
        },
        "winner_counts_n4": {
            str(name): int(count)
            for name, count in (
                context["winner_n4"].value_counts().sort_index().items()
            )
        },
    }
    return output, metrics


def markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    display = frame[columns].copy()
    for column in display.select_dtypes(include=[np.number]).columns:
        if "horizon" not in column and not column.startswith("n_"):
            display[column] = display[column].map(
                lambda value: f"{float(value):.12g}"
            )
    header = "| " + " | ".join(display.columns) + " |"
    divider = "| " + " | ".join("---" for _ in display.columns) + " |"
    rows = [
        "| " + " | ".join(str(value) for value in row) + " |"
        for row in display.itertuples(index=False, name=None)
    ]
    return "\n".join([header, divider, *rows])


def append_report(
    verified: list[dict[str, str]],
    chk1: pd.DataFrame,
    means: dict[str, float],
    chk2_metrics: dict[str, Any],
) -> str:
    existing = REPORT_PATH.read_text(encoding="utf-8")
    for reading in PRECOMMITTED_READINGS:
        if reading not in existing:
            raise AssertionError(
                "Pre-committed reading is absent before computation: "
                f"{reading}"
            )

    pilot_relative = (
        100.0
        * (means["pilot_E-N4"] - means["pilot_E-N3"])
        / means["pilot_E-N3"]
    )
    confirmation_relative = (
        100.0
        * (means["confirmation_E-N4"] - means["confirmation_E-N3"])
        / means["confirmation_E-N3"]
    )
    rates_close = (
        abs(pilot_relative) <= 15.0 and abs(confirmation_relative) <= 15.0
    )
    rankings_flipped = (
        chk2_metrics["any_specified_pair_flip_pairs"] > 0
        and chk2_metrics["winner_changes"] > 0
    )
    if rates_close and rankings_flipped:
        verdict = "REPRICING"
    elif (
        means["confirmation_E-N4"] > 1.15 * means["confirmation_E-N3"]
        and not rankings_flipped
    ):
        verdict = "TRIGGER-MEDIATED"
    else:
        verdict = "ENTANGLED"

    source_table = pd.DataFrame(verified)
    chk1_columns = [
        "scope",
        "arm",
        "prediction_horizon",
        "n_seeds",
        "mean_interventions_per_100_steps",
        "min_interventions_per_100_steps",
        "max_interventions_per_100_steps",
        "p50_interventions_per_100_steps",
        "raw_file",
    ]
    term_counts = json.dumps(
        chk2_metrics["flipped_pair_term_counts"], sort_keys=True
    )
    winner_n3 = json.dumps(chk2_metrics["winner_counts_n3"], sort_keys=True)
    winner_n4 = json.dumps(chk2_metrics["winner_counts_n4"], sort_keys=True)
    computed = f"""

## Input hash verification

{markdown_table(source_table, ["status", "path", "sha256"])}

All Part A inputs matched their pre-existing manifest hashes before the data
were read.

## CHK1 - Intervention rates

{markdown_table(chk1, chk1_columns)}

Pilot E-N4 versus E-N3 mean-rate difference: {pilot_relative:.12g}% relative
to E-N3. Confirmation E-N4 versus E-N3 mean-rate difference:
{confirmation_relative:.12g}% relative to E-N3. Both absolute differences are
within the pre-committed ±15% band: **{str(rates_close).upper()}**.

Raw sources: `{DX2_RATES}`, `{CONFIRMATION_E3_STEPS}`, and
`{CONFIRMATION_E4_STEPS}`. Derived table: `{CHK1_PATH}`.

## CHK2 - E-N4 survivor payload rankings at drift onset

- Matched E-N4 survivor seeds: {chk2_metrics["matched_seeds"]}.
- Evaluated seed-step pairs: {chk2_metrics["seed_step_pairs"]} (seven seeds,
  steps 24-32).
- Full candidate rows: {chk2_metrics["candidate_rows"]} (eight candidates per
  seed-step).
- N3-style inversion present under N3: {chk2_metrics["n3_inversion_count"]}/63.
- N3-style inversion present under N4: {chk2_metrics["n4_inversion_count"]}/63.
- Winner changed from N3 to N4: {chk2_metrics["winner_changes"]}/63.
- `gentle_down` versus `diff_left` order flips: {chk2_metrics["diff_left_order_flips"]}/63.
- `gentle_down` versus `theta_correct_b` order flips:
  {chk2_metrics["theta_correct_b_order_flips"]}/63.
- Seed-step pairs with at least one specified pair-order flip:
  {chk2_metrics["any_specified_pair_flip_pairs"]}/63.
- Largest weighted gap-change terms across the flipped pairs: `{term_counts}`.
- N3 winner counts: `{winner_n3}`.
- N4 winner counts: `{winner_n4}`.

The candidate-level CSV records both complete rankings, the four unweighted
score terms, their weighted contributions, both final scores, both winners,
the two specified pairwise gaps, order-flip flags, and the largest component
of each gap change for every seed-step.

Raw sources: `{E3_PAYLOAD}`, `{E4_PAYLOAD}`, and `{PAYLOAD_SELECTION}`.
Derived table: `{CHK2_PATH}`.

## Mechanical adjudication

- Rates within the pre-committed ±15% band: **{str(rates_close).upper()}**.
- Rankings flipped: **{str(rankings_flipped).upper()}**.
- No hypothesis test was computed.

CHK-VERDICT: {verdict}
"""
    REPORT_PATH.write_text(
        existing.replace(
            "<!-- Computed results are appended only after source-hash "
            "verification. -->",
            computed.strip(),
        )
        + "\n",
        encoding="utf-8",
    )
    return verdict


def write_manifest() -> None:
    manifest_targets = (CHK1_PATH, CHK2_PATH, REPORT_PATH, SCRIPT_PATH)
    lines = [f"{sha256(path)} *{path}" for path in manifest_targets]
    MANIFEST_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    if not REPORT_PATH.exists():
        raise FileNotFoundError(
            f"Pre-committed report scaffold is missing: {REPORT_PATH}"
        )
    if CHK1_PATH.exists() or CHK2_PATH.exists() or MANIFEST_PATH.exists():
        raise FileExistsError(
            "Part A derived outputs already exist; refusing to overwrite"
        )
    verified = verify_inputs()
    chk1, means = build_chk1()
    _, chk2_metrics = build_chk2()
    verdict = append_report(verified, chk1, means, chk2_metrics)
    write_manifest()
    print("PART_A_COMPLETE")
    print(f"CHK1={CHK1_PATH}")
    print(f"CHK2={CHK2_PATH}")
    print(f"REPORT={REPORT_PATH}")
    print(f"MANIFEST={MANIFEST_PATH}")
    print(f"CHK-VERDICT: {verdict}")
    print("AWAITING_SIGNOFF")


if __name__ == "__main__":
    main()
