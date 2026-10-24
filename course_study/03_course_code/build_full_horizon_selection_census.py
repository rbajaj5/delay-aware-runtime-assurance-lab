#!/usr/bin/env python3
"""Build a full-sequence selected-candidate census for horizons N=3..10.

This is read-only postprocessing over retained step logs. It provides one
mechanism metric available for every measured integer horizon: the fraction
of intervention steps selecting each candidate, especially gentle_down.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(r"C:\Users\anaxe\RAGforAssurredAutonomy")
HORIZON_ROOT = ROOT / "analysis" / "out" / "horizon_spectrum_20260723"
OUTPUT_DIR = HORIZON_ROOT / "dense_extension_20260725"
OUTPUT_CSV = OUTPUT_DIR / "full_horizon_selection_census.csv"
OUTPUT_JSON = OUTPUT_DIR / "full_horizon_selection_census_provenance.json"

SOURCES = {
    3: HORIZON_ROOT
    / "phase2b"
    / "confirmation"
    / "full_steps"
    / "E-N3_steps.csv",
    4: HORIZON_ROOT
    / "phase2b"
    / "confirmation"
    / "full_steps"
    / "E-N4_steps.csv",
    5: OUTPUT_DIR / "n5" / "E-N5-dense_steps.csv",
    6: OUTPUT_DIR / "n6" / "E-N6-dense_steps.csv",
    7: HORIZON_ROOT
    / "exploratory_2x2_20260723"
    / "part_b"
    / "en7_coupled"
    / "full_steps"
    / "E-N7-exploratory_steps.csv",
    8: OUTPUT_DIR / "n8" / "E-N8-dense_steps.csv",
    9: OUTPUT_DIR / "n9" / "E-N9-dense_steps.csv",
    10: HORIZON_ROOT / "pilot" / "full_steps" / "E-N10_steps.csv",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def as_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def census(horizon: int, path: Path) -> dict[str, Any]:
    winners: Counter[str] = Counter()
    steps_by_seed: dict[int, list[tuple[int, str]]] = defaultdict(list)
    total_rows = 0
    intervention_rows = 0
    seeds: set[int] = set()

    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            total_rows += 1
            seed = int(row["seed"])
            seeds.add(seed)
            if not as_bool(row.get("intervention")):
                continue
            intervention_rows += 1
            winner = str(row.get("selected_candidate", "") or "")
            if not winner:
                raise AssertionError(
                    f"Missing selected_candidate at intervention: N={horizon}, "
                    f"seed={seed}, step={row.get('step')}"
                )
            winners[winner] += 1
            steps_by_seed[seed].append((int(row["step"]), winner))

    churn = []
    for values in steps_by_seed.values():
        ordered = sorted(values)
        if len(ordered) < 2:
            churn.append(0.0)
        else:
            changes = sum(
                left[1] != right[1]
                for left, right in zip(ordered, ordered[1:])
            )
            churn.append(changes / (len(ordered) - 1))

    expected_n = 200 if horizon == 10 else 1000
    if len(seeds) != expected_n:
        raise AssertionError(
            f"N={horizon} seed count {len(seeds)} != expected {expected_n}"
        )
    if intervention_rows != sum(winners.values()):
        raise AssertionError(f"N={horizon} winner census does not sum")

    def share(name: str) -> float:
        return winners.get(name, 0) / intervention_rows if intervention_rows else float("nan")

    symmetric_count = winners.get("gentle_down", 0) + winners.get("hover", 0)
    differential_count = sum(
        winners.get(name, 0)
        for name in ("diff_left", "diff_right", "theta_correct_a", "theta_correct_b")
    )
    return {
        "prediction_horizon": horizon,
        "n": len(seeds),
        "total_step_rows": total_rows,
        "intervention_step_rows": intervention_rows,
        "gentle_down_count": winners.get("gentle_down", 0),
        "gentle_down_share": share("gentle_down"),
        "hover_count": winners.get("hover", 0),
        "hover_share": share("hover"),
        "symmetric_count": symmetric_count,
        "symmetric_share": symmetric_count / intervention_rows,
        "differential_count": differential_count,
        "differential_share": differential_count / intervention_rows,
        "ppo_clipped_count": winners.get("ppo_clipped", 0),
        "ppo_clipped_share": share("ppo_clipped"),
        "argmin_churn_mean": float(np.mean(churn)),
        "winner_counts_json": json.dumps(dict(sorted(winners.items())), separators=(",", ":")),
        "source_path": str(path),
        "source_sha256": sha256(path),
    }


def main() -> int:
    missing = [str(path) for path in SOURCES.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing source step logs: {missing}")
    rows = [census(horizon, SOURCES[horizon]) for horizon in sorted(SOURCES)]
    write_csv(OUTPUT_CSV, rows)
    provenance = {
        "status": "COMPLETE",
        "simulation_steps_executed": 0,
        "output": {
            "path": str(OUTPUT_CSV),
            "sha256": sha256(OUTPUT_CSV),
        },
        "sources": [
            {
                "prediction_horizon": horizon,
                "path": str(path),
                "size_bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for horizon, path in sorted(SOURCES.items())
        ],
    }
    OUTPUT_JSON.write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(rows, indent=2, sort_keys=True))
    print(f"OUTPUT={OUTPUT_CSV}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

