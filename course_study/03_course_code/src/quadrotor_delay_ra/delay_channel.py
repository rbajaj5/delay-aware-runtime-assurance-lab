"""Action-delay channel for normalized quadrotor controller outputs.

The channel sits after the PPO controller selects a normalized action and before
``env.step(action)``. It models actuator/software delay without dropping or
reordering commands.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np


DelayMode = Literal["zero", "fixed_one", "uniform_0_2"]


@dataclass(frozen=True)
class DelayStep:
    """Result returned by one delay-channel update."""

    commanded_action: np.ndarray
    applied_action: np.ndarray
    delay_sample: int
    queue_state: list[dict[str, Any]]


class DelayChannel:
    """FIFO delay channel for normalized actions.

    Commands are appended with a target release step. At each step, at most one
    matured command is released in FIFO order. If a later command has an earlier
    sampled release time, it still waits behind older commands; this preserves
    issue order and avoids dropping commands.

    For nonzero delay before any command has matured, the channel applies a
    configurable hold action. The default hold is a zero vector matching the first
    command shape.
    """

    def __init__(
        self,
        mode: DelayMode = "zero",
        seed: int | None = None,
        hold_action: np.ndarray | list[float] | tuple[float, ...] | None = None,
    ) -> None:
        if mode not in {"zero", "fixed_one", "uniform_0_2"}:
            raise ValueError(f"Unsupported delay mode: {mode}")
        self.mode = mode
        self._seed = seed
        self._rng = np.random.default_rng(seed)
        self._configured_hold_action = None if hold_action is None else np.asarray(hold_action, dtype=float).copy()
        self.reset(seed=seed)

    def reset(self, seed: int | None = None) -> None:
        """Clear queued commands and optionally reset the deterministic RNG."""

        if seed is not None:
            self._seed = seed
            self._rng = np.random.default_rng(seed)
        elif self._seed is not None:
            self._rng = np.random.default_rng(self._seed)
        self._queue: deque[dict[str, Any]] = deque()
        self._step = 0
        self._last_applied: np.ndarray | None = None

    def sample_delay(self) -> int:
        """Sample the configured integer control-step delay."""

        if self.mode == "zero":
            return 0
        if self.mode == "fixed_one":
            return 1
        return int(self._rng.integers(0, 3))

    def update(self, commanded_action: np.ndarray | list[float] | tuple[float, ...]) -> DelayStep:
        """Append one command and return the action to apply this control step."""

        command = np.asarray(commanded_action, dtype=float).copy()
        if self._last_applied is None:
            self._last_applied = self._initial_hold(command)

        delay_sample = self.sample_delay()
        entry = {
            "command_step": self._step,
            "release_step": self._step + delay_sample,
            "delay_sample": delay_sample,
            "action": command,
        }
        self._queue.append(entry)

        if self._queue and int(self._queue[0]["release_step"]) <= self._step:
            released = self._queue.popleft()
            self._last_applied = np.asarray(released["action"], dtype=float).copy()

        result = DelayStep(
            commanded_action=command,
            applied_action=np.asarray(self._last_applied, dtype=float).copy(),
            delay_sample=delay_sample,
            queue_state=self.queue_state(),
        )
        self._step += 1
        return result

    def queue_state(self) -> list[dict[str, Any]]:
        """Return a JSON-serializable snapshot of pending delayed commands."""

        return [
            {
                "command_step": int(item["command_step"]),
                "release_step": int(item["release_step"]),
                "delay_sample": int(item["delay_sample"]),
                "action": np.asarray(item["action"], dtype=float).tolist(),
            }
            for item in self._queue
        ]

    def _initial_hold(self, command: np.ndarray) -> np.ndarray:
        if self._configured_hold_action is not None:
            hold = self._configured_hold_action
            if hold.shape != command.shape:
                raise ValueError(f"hold_action shape {hold.shape} does not match command shape {command.shape}")
            return hold.copy()
        return np.zeros_like(command, dtype=float)
