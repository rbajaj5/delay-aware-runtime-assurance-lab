"""Tensorized FIFO actuation-delay channel.

This module is a GPU-friendly counterpart to the scalar delay channel used in
the safe-control-gym harness. It preserves the same semantics:

* commands are appended with absolute release steps;
* issue order is FIFO, so later commands cannot overtake the head;
* if the head has not matured, the plant receives the held last action.

The implementation keeps a fixed-capacity tensor buffer and updates all
environments in one batched tensor operation. It intentionally does not replace
safe-control-gym physics; it is the delay primitive a future vectorized runner
can share with batched PPO inference and batched monitor scoring.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TensorDelayStep:
    """Batched result from one tensorized delay update."""

    commanded_action: "torch.Tensor"
    applied_action: "torch.Tensor"
    delay_sample: "torch.Tensor"
    queue_len: "torch.Tensor"
    held_last_action_used: "torch.Tensor"
    released_action: "torch.Tensor"
    released_mask: "torch.Tensor"


class TensorFIFOActuationDelay:
    """Batched FIFO delay channel using PyTorch tensors.

    Parameters
    ----------
    num_envs:
        Number of independent environments in the batch.
    action_dim:
        Normalized action dimension.
    max_delay:
        Maximum integer delay sample. Capacity defaults to ``max_delay + 1``,
        which is sufficient for one command issued per control tick.
    device:
        CPU or CUDA device string.
    """

    def __init__(
        self,
        *,
        num_envs: int,
        action_dim: int,
        max_delay: int = 2,
        device: str | None = None,
        dtype: "torch.dtype | None" = None,
        hold_action: "torch.Tensor | None" = None,
        capacity: int | None = None,
    ) -> None:
        import torch

        if num_envs <= 0:
            raise ValueError("num_envs must be positive")
        if action_dim <= 0:
            raise ValueError("action_dim must be positive")
        if max_delay < 0:
            raise ValueError("max_delay must be nonnegative")
        self.num_envs = int(num_envs)
        self.action_dim = int(action_dim)
        self.max_delay = int(max_delay)
        self.capacity = int(capacity if capacity is not None else max_delay + 1)
        if self.capacity < self.max_delay + 1:
            raise ValueError("capacity must be at least max_delay + 1")
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.dtype = dtype or torch.float32
        if hold_action is None:
            hold = torch.zeros((self.num_envs, self.action_dim), dtype=self.dtype, device=self.device)
        else:
            hold = hold_action.to(device=self.device, dtype=self.dtype)
            if hold.ndim == 1:
                hold = hold.view(1, -1).expand(self.num_envs, -1).contiguous()
            if tuple(hold.shape) != (self.num_envs, self.action_dim):
                raise ValueError(f"hold_action shape {tuple(hold.shape)} does not match {(self.num_envs, self.action_dim)}")
        self._configured_hold = hold.clone()
        self.reset()

    def reset(self) -> None:
        """Clear every environment queue and restore the configured hold action."""

        import torch

        shape = (self.num_envs, self.capacity)
        self.actions = torch.zeros((*shape, self.action_dim), dtype=self.dtype, device=self.device)
        self.release_steps = torch.zeros(shape, dtype=torch.int64, device=self.device)
        self.delay_samples = torch.zeros(shape, dtype=torch.int16, device=self.device)
        self.valid = torch.zeros(shape, dtype=torch.bool, device=self.device)
        self.queue_len = torch.zeros((self.num_envs,), dtype=torch.int64, device=self.device)
        self.last_applied = self._configured_hold.clone()
        self.current_step = 0

    def update(self, commanded_action: "torch.Tensor", delay_sample: "torch.Tensor | int") -> TensorDelayStep:
        """Append commands and return the batched action applied this step."""

        import torch

        command = commanded_action.to(device=self.device, dtype=self.dtype)
        if tuple(command.shape) != (self.num_envs, self.action_dim):
            raise ValueError(f"commanded_action shape {tuple(command.shape)} does not match {(self.num_envs, self.action_dim)}")
        if isinstance(delay_sample, int):
            delays = torch.full((self.num_envs,), int(delay_sample), dtype=torch.int64, device=self.device)
        else:
            delays = delay_sample.to(device=self.device, dtype=torch.int64).view(-1)
            if delays.numel() != self.num_envs:
                raise ValueError(f"delay_sample has {delays.numel()} entries, expected {self.num_envs}")
        delays = torch.clamp(delays, 0, self.max_delay)
        if torch.any(self.queue_len >= self.capacity):
            raise RuntimeError("FIFO delay buffer overflow before append; increase capacity or inspect delay semantics")

        env_idx = torch.arange(self.num_envs, device=self.device)
        tail_idx = self.queue_len
        self.actions[env_idx, tail_idx, :] = command
        self.release_steps[env_idx, tail_idx] = int(self.current_step) + delays
        self.delay_samples[env_idx, tail_idx] = delays.to(torch.int16)
        self.valid[env_idx, tail_idx] = True
        self.queue_len = self.queue_len + 1

        head_valid = self.valid[:, 0]
        released_mask = head_valid & (self.release_steps[:, 0] <= int(self.current_step))
        released_action = self.actions[:, 0, :].clone()
        applied = torch.where(released_mask.view(-1, 1), released_action, self.last_applied)
        self.last_applied = applied.clone()

        self._pop_released_heads(released_mask)
        held = ~released_mask
        out = TensorDelayStep(
            commanded_action=command.clone(),
            applied_action=applied.clone(),
            delay_sample=delays.clone(),
            queue_len=self.queue_len.clone(),
            held_last_action_used=held.clone(),
            released_action=released_action,
            released_mask=released_mask.clone(),
        )
        self.current_step += 1
        return out

    def _pop_released_heads(self, released_mask: "torch.Tensor") -> None:
        """Remove matured heads in FIFO order without per-environment loops."""

        import torch

        if not bool(torch.any(released_mask)):
            return
        slot_idx = torch.arange(self.capacity, device=self.device).view(1, -1).expand(self.num_envs, -1)
        shift = released_mask.to(torch.int64).view(-1, 1)
        src_idx = torch.clamp(slot_idx + shift, max=self.capacity - 1)
        gather_actions = src_idx.unsqueeze(-1).expand(-1, -1, self.action_dim)
        self.actions = torch.gather(self.actions, 1, gather_actions)
        self.release_steps = torch.gather(self.release_steps, 1, src_idx)
        self.delay_samples = torch.gather(self.delay_samples, 1, src_idx)
        self.valid = torch.gather(self.valid, 1, src_idx)
        self.queue_len = self.queue_len - released_mask.to(torch.int64)
        valid_by_len = slot_idx < self.queue_len.view(-1, 1)
        self.valid = self.valid & valid_by_len
        self.actions = torch.where(valid_by_len.unsqueeze(-1), self.actions, torch.zeros_like(self.actions))
        self.release_steps = torch.where(valid_by_len, self.release_steps, torch.zeros_like(self.release_steps))
        self.delay_samples = torch.where(valid_by_len, self.delay_samples, torch.zeros_like(self.delay_samples))

    def queue_snapshot_cpu(self) -> list[list[dict[str, object]]]:
        """Return a JSON-friendly debug snapshot. This intentionally uses CPU."""

        actions = self.actions.detach().cpu()
        release = self.release_steps.detach().cpu()
        delays = self.delay_samples.detach().cpu()
        valid = self.valid.detach().cpu()
        out: list[list[dict[str, object]]] = []
        for env in range(self.num_envs):
            rows: list[dict[str, object]] = []
            for slot in range(self.capacity):
                if bool(valid[env, slot]):
                    rows.append(
                        {
                            "release_step": int(release[env, slot]),
                            "delay_sample": int(delays[env, slot]),
                            "action": actions[env, slot].numpy().astype(float).tolist(),
                        }
                    )
            out.append(rows)
        return out
