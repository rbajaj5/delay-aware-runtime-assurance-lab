"""Fixed-shape streaming loggers for large quadrotor batches.

The original research harness writes CSVs because that is easy to audit. For
larger runs, per-step Python dict/list accumulation becomes the bottleneck. This
module provides a dependency-light memmap logger that stores dense arrays in
``.npy`` files and a separate sparse CSV writer for decision traces.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np


@dataclass(frozen=True)
class MemmapLogSpec:
    """Shape contract for dense per-step experiment logging."""

    num_conditions: int
    num_seeds: int
    max_steps: int
    state_dim: int = 6
    action_dim: int = 2
    constraint_dim: int = 16
    max_queue_len: int = 3


class MemmapStepLogger:
    """Preallocated dense logger backed by NumPy ``.npy`` memmaps."""

    def __init__(
        self,
        root: str | Path,
        spec: MemmapLogSpec,
        *,
        condition_names: Iterable[str],
        seed_values: Iterable[int],
    ) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.spec = spec
        self.condition_names = list(condition_names)
        self.seed_values = [int(s) for s in seed_values]
        if len(self.condition_names) != spec.num_conditions:
            raise ValueError("condition_names length must match spec.num_conditions")
        if len(self.seed_values) != spec.num_seeds:
            raise ValueError("seed_values length must match spec.num_seeds")
        self.condition_to_index = {name: i for i, name in enumerate(self.condition_names)}
        self.seed_to_index = {seed: i for i, seed in enumerate(self.seed_values)}
        c, n, t = spec.num_conditions, spec.num_seeds, spec.max_steps
        self.arrays: dict[str, np.memmap] = {
            "state": self._open("state.npy", np.float32, (c, n, t, spec.state_dim), np.nan),
            "reference_state": self._open("reference_state.npy", np.float32, (c, n, t, spec.state_dim), np.nan),
            "ppo_command": self._open("ppo_command.npy", np.float32, (c, n, t, spec.action_dim), np.nan),
            "monitor_command": self._open("monitor_command.npy", np.float32, (c, n, t, spec.action_dim), np.nan),
            "applied_action": self._open("applied_action.npy", np.float32, (c, n, t, spec.action_dim), np.nan),
            "constraint_values": self._open("constraint_values.npy", np.float32, (c, n, t, spec.constraint_dim), np.nan),
            "queue_actions": self._open("queue_actions.npy", np.float32, (c, n, t, spec.max_queue_len, spec.action_dim), np.nan),
            "queue_release_steps": self._open("queue_release_steps.npy", np.int32, (c, n, t, spec.max_queue_len), -1),
            "queue_mask": self._open("queue_mask.npy", np.bool_, (c, n, t, spec.max_queue_len), False),
            "delay_sample": self._open("delay_sample.npy", np.int16, (c, n, t), -1),
            "queue_len": self._open("queue_len.npy", np.int16, (c, n, t), -1),
            "severity": self._open("severity.npy", np.float32, (c, n, t), np.nan),
            "reward": self._open("reward.npy", np.float32, (c, n, t), np.nan),
            "done": self._open("done.npy", np.bool_, (c, n, t), False),
            "out_of_bounds": self._open("out_of_bounds.npy", np.bool_, (c, n, t), False),
            "written": self._open("written.npy", np.bool_, (c, n, t), False),
        }
        self.write_manifest()

    def _open(self, name: str, dtype: Any, shape: tuple[int, ...], fill_value: Any) -> np.memmap:
        path = self.root / name
        arr = np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=shape)
        arr[...] = fill_value
        arr.flush()
        return arr

    def write_step(
        self,
        *,
        condition: str | int,
        seed: int,
        step: int,
        state: np.ndarray,
        reference_state: np.ndarray | None = None,
        ppo_command: np.ndarray | None = None,
        monitor_command: np.ndarray | None = None,
        applied_action: np.ndarray | None = None,
        constraint_values: np.ndarray | None = None,
        delay_sample: int = -1,
        queue_state: list[Mapping[str, Any]] | None = None,
        severity: float = np.nan,
        reward: float = np.nan,
        done: bool = False,
        out_of_bounds: bool = False,
    ) -> None:
        """Write one dense step row by numeric location."""

        ci = int(condition) if isinstance(condition, int) else self.condition_to_index[condition]
        si = self.seed_to_index[int(seed)]
        st = int(step)
        if st < 0 or st >= self.spec.max_steps:
            raise IndexError(f"step {st} outside max_steps={self.spec.max_steps}")
        self._assign_vector("state", (ci, si, st), state, self.spec.state_dim)
        if reference_state is not None:
            self._assign_vector("reference_state", (ci, si, st), reference_state, self.spec.state_dim)
        if ppo_command is not None:
            self._assign_vector("ppo_command", (ci, si, st), ppo_command, self.spec.action_dim)
        if monitor_command is not None:
            self._assign_vector("monitor_command", (ci, si, st), monitor_command, self.spec.action_dim)
        if applied_action is not None:
            self._assign_vector("applied_action", (ci, si, st), applied_action, self.spec.action_dim)
        if constraint_values is not None:
            self._assign_vector("constraint_values", (ci, si, st), constraint_values, self.spec.constraint_dim)
        self.arrays["delay_sample"][ci, si, st] = int(delay_sample)
        self.arrays["severity"][ci, si, st] = float(severity)
        self.arrays["reward"][ci, si, st] = float(reward)
        self.arrays["done"][ci, si, st] = bool(done)
        self.arrays["out_of_bounds"][ci, si, st] = bool(out_of_bounds)
        self.arrays["written"][ci, si, st] = True
        self._assign_queue(ci, si, st, queue_state or [])

    def _assign_vector(self, name: str, idx: tuple[int, int, int], values: np.ndarray, dim: int) -> None:
        arr = np.asarray(values, dtype=np.float32).reshape(-1)
        clipped = np.full((dim,), np.nan, dtype=np.float32)
        n = min(dim, arr.size)
        clipped[:n] = arr[:n]
        self.arrays[name][idx] = clipped

    def _assign_queue(self, ci: int, si: int, st: int, queue_state: list[Mapping[str, Any]]) -> None:
        max_q = self.spec.max_queue_len
        count = min(len(queue_state), max_q)
        self.arrays["queue_len"][ci, si, st] = len(queue_state)
        self.arrays["queue_actions"][ci, si, st, :, :] = np.nan
        self.arrays["queue_release_steps"][ci, si, st, :] = -1
        self.arrays["queue_mask"][ci, si, st, :] = False
        for slot, item in enumerate(queue_state[:count]):
            action = np.asarray(item.get("action", []), dtype=np.float32).reshape(-1)
            n = min(action.size, self.spec.action_dim)
            self.arrays["queue_actions"][ci, si, st, slot, :n] = action[:n]
            self.arrays["queue_release_steps"][ci, si, st, slot] = int(item.get("release_step", -1))
            self.arrays["queue_mask"][ci, si, st, slot] = True

    def flush(self) -> None:
        for arr in self.arrays.values():
            arr.flush()

    def close(self) -> None:
        self.flush()
        self.arrays.clear()

    def write_manifest(self) -> Path:
        path = self.root / "manifest.json"
        payload = {
            "spec": self.spec.__dict__,
            "condition_names": self.condition_names,
            "seed_values": self.seed_values,
            "arrays": {name: str(self.root / f"{name}.npy") for name in self.arrays},
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return path


class SparseDecisionTraceWriter:
    """Append-only CSV for intervention/OOB-window diagnostics."""

    def __init__(self, path: str | Path, fieldnames: list[str]) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fieldnames = list(fieldnames)
        self._f = self.path.open("w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._f, fieldnames=self.fieldnames)
        self._writer.writeheader()

    def write(self, row: Mapping[str, Any]) -> None:
        self._writer.writerow({field: row.get(field, "") for field in self.fieldnames})

    def close(self) -> None:
        self._f.close()
