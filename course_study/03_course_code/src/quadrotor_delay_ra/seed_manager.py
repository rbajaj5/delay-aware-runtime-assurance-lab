"""Deterministic seed and delay-schedule management for batch experiments.

The current quadrotor experiments rely on matched seed-indexed delay schedules:
conditions B-H must see the same exogenous delay realization for a given base
seed. This module centralizes that contract so future parallel/vectorized
runners can share read-only schedules instead of sampling inside workers.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np


DEFAULT_CONDITION_OFFSETS: dict[str, int] = {
    "A_nominal_ppo_zero_delay": 0,
    "B_delayed_ppo": 10_000,
    "C_warning_only_delay_aware": 20_000,
    "D_delay_blind_intervention": 30_000,
    "E_delay_aware_intervention": 40_000,
    "E_held_corrected_intervention": 50_000,
    "E_slew_rate_intervention": 60_000,
}


def schedule_hash(schedule: np.ndarray | Sequence[int]) -> str:
    """Return the short hash convention used by the experiment logs."""

    arr = np.asarray(schedule, dtype=np.int16).reshape(-1)
    raw = ",".join(str(int(x)) for x in arr).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


@dataclass(frozen=True)
class SeedRecord:
    """Resolved deterministic seed metadata for one base seed."""

    seed_index: int
    base_seed: int
    env_seeds: dict[str, int]
    delay_schedule_hash: str


class DeterministicSeedManager:
    """Generate and persist matched delay schedules for batch experiments.

    Delay schedules are keyed only by the base seed, not by condition. Workers
    may use condition-specific environment seeds if desired, but must index the
    same schedule row for all matched conditions.
    """

    def __init__(
        self,
        *,
        seed_start: int,
        num_seeds: int,
        steps: int,
        max_delay: int = 2,
        condition_offsets: Mapping[str, int] | None = None,
    ) -> None:
        if num_seeds <= 0:
            raise ValueError("num_seeds must be positive")
        if steps <= 0:
            raise ValueError("steps must be positive")
        if max_delay < 0:
            raise ValueError("max_delay must be nonnegative")
        self.seed_start = int(seed_start)
        self.num_seeds = int(num_seeds)
        self.steps = int(steps)
        self.max_delay = int(max_delay)
        self.condition_offsets = dict(condition_offsets or DEFAULT_CONDITION_OFFSETS)
        self.base_seeds = np.arange(self.seed_start, self.seed_start + self.num_seeds, dtype=np.int64)

    def env_seed(self, base_seed: int, condition: str) -> int:
        """Return a reproducible condition-specific environment seed."""

        return int(base_seed) + int(self.condition_offsets.get(condition, 0))

    def delay_schedule(self, base_seed: int, mode: str = "uniform_0_2") -> np.ndarray:
        """Return the matched integer delay schedule for one base seed."""

        if mode == "zero":
            return np.zeros(self.steps, dtype=np.int16)
        if mode == "fixed_one":
            return np.ones(self.steps, dtype=np.int16)
        if mode != "uniform_0_2":
            raise ValueError(f"Unsupported deterministic schedule mode: {mode}")
        rng = np.random.default_rng(int(base_seed))
        return rng.integers(0, self.max_delay + 1, size=self.steps, dtype=np.int16)

    def records(self, conditions: Sequence[str], mode: str = "uniform_0_2") -> list[SeedRecord]:
        """Return audit records for all base seeds."""

        out: list[SeedRecord] = []
        for idx, base_seed in enumerate(self.base_seeds):
            schedule = self.delay_schedule(int(base_seed), mode=mode)
            out.append(
                SeedRecord(
                    seed_index=idx,
                    base_seed=int(base_seed),
                    env_seeds={condition: self.env_seed(int(base_seed), condition) for condition in conditions},
                    delay_schedule_hash=schedule_hash(schedule),
                )
            )
        return out

    def write_delay_memmap(self, path: str | Path, mode: str = "uniform_0_2") -> Path:
        """Write schedules to a read-only-shareable ``.npy`` memmap file."""

        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        mm = np.lib.format.open_memmap(
            out,
            mode="w+",
            dtype=np.int16,
            shape=(self.num_seeds, self.steps),
        )
        for idx, base_seed in enumerate(self.base_seeds):
            mm[idx, :] = self.delay_schedule(int(base_seed), mode=mode)
        mm.flush()
        del mm
        return out

    def write_manifest(self, path: str | Path, *, conditions: Sequence[str], schedule_path: str | Path) -> Path:
        """Write JSON metadata that lets workers reproduce schedule indexing."""

        manifest_path = Path(path)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "seed_start": self.seed_start,
            "num_seeds": self.num_seeds,
            "steps": self.steps,
            "max_delay": self.max_delay,
            "condition_offsets": self.condition_offsets,
            "conditions": list(conditions),
            "schedule_path": str(Path(schedule_path)),
            "records": [record.__dict__ for record in self.records(conditions)],
        }
        manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return manifest_path


def load_delay_memmap(path: str | Path, *, readonly: bool = True) -> np.ndarray:
    """Load a schedule memmap without copying it into process memory."""

    mode = "r" if readonly else "r+"
    return np.load(Path(path), mmap_mode=mode)
