"""Delayed signaling and cross-play protocol formation diagnostic.

This toy benchmark is inspired by the OvercookedV2 emphasis on asymmetric
information, stochasticity, cross-play, and test-time protocol formation. One
agent observes a changing hidden bit and sends a convention-dependent signal.
The partner must act under delayed, jittered signals and delayed feedback.

The code is a mechanism diagnostic, not an implementation of OvercookedV2 and
not a claim about the paper's reported results.
"""

from __future__ import annotations

import csv
import hashlib
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "20260730" / "crossplay_protocol"
ROUNDS = 140
DELAYS = (0, 1, 2, 3, 5)
POLICIES = ("fixed_protocol", "delay_blind_adaptive", "queue_aware_adaptive", "known_protocol")
CONVENTIONS = (0, 1)
GAMES_PER_CELL = 20


@dataclass(frozen=True)
class Signal:
    seq: int
    value: int
    due: int


@dataclass(frozen=True)
class Feedback:
    seq: int
    signal: int
    action: int
    reward: int
    due: int


def run_game(delay: int, policy: str, sender_convention: int, seed: int) -> dict[str, float | int | str]:
    rng = random.Random(seed)
    hidden = rng.choice((0, 1))
    signal_queue: list[Signal] = []
    feedback_queue: list[Feedback] = []
    received_signals: list[Signal] = []
    received_feedback: list[Feedback] = []
    signal_by_seq: dict[int, Signal] = {}
    action_by_seq: dict[int, tuple[int, int]] = {}
    posterior_c0 = 0.5
    latest_signal: Signal | None = None
    correct = 0
    total = 0
    convention_updates = 0
    misattributed_updates = 0
    signal_age_total = 0
    signal_age_count = 0
    learned_at: int | None = None

    for now in range(ROUNDS):
        if rng.random() < 0.08:
            hidden = 1 - hidden

        due_signals = [message for message in signal_queue if message.due <= now]
        signal_queue = [message for message in signal_queue if message.due > now]
        due_feedback = [message for message in feedback_queue if message.due <= now]
        feedback_queue = [message for message in feedback_queue if message.due > now]
        received_signals.extend(due_signals)
        received_feedback.extend(due_feedback)

        if policy == "queue_aware_adaptive":
            for message in sorted(received_feedback, key=lambda item: item.seq):
                original = signal_by_seq.get(message.seq)
                if original is None or message.seq not in action_by_seq:
                    continue
                inferred_state = message.action if message.reward else 1 - message.action
                inferred_c = original.value ^ inferred_state
                posterior_c0 = 0.95 if inferred_c == 0 else 0.05
                convention_updates += 1
                if inferred_c != sender_convention:
                    misattributed_updates += 1
            received_feedback.clear()
            if due_signals:
                latest_signal = max(received_signals, key=lambda item: item.seq)
        else:
            for message in received_feedback:
                if policy == "delay_blind_adaptive":
                    original = latest_signal
                else:
                    original = signal_by_seq.get(message.seq)
                if original is None or message.seq not in action_by_seq:
                    continue
                inferred_state = message.action if message.reward else 1 - message.action
                inferred_c = original.value ^ inferred_state
                posterior_c0 = 0.95 if inferred_c == 0 else 0.05
                convention_updates += 1
                if inferred_c != sender_convention:
                    misattributed_updates += 1
            received_feedback.clear()
            if received_signals:
                latest_signal = max(
                    received_signals,
                    key=lambda item: item.due if policy == "delay_blind_adaptive" else item.seq,
                )

        if latest_signal is None:
            action = 0
            signal_age = delay + 1
        elif policy == "known_protocol":
            action = latest_signal.value ^ sender_convention
            signal_age = now - latest_signal.seq
        elif policy == "fixed_protocol":
            action = latest_signal.value
            signal_age = now - latest_signal.seq
        else:
            guessed_c = 0 if posterior_c0 >= 0.5 else 1
            action = latest_signal.value ^ guessed_c
            signal_age = now - latest_signal.seq
        signal_age_total += signal_age
        signal_age_count += 1

        reward = int(action == hidden)
        correct += reward
        total += 1
        if learned_at is None and posterior_c0 in (0.05, 0.95):
            learned_at = now

        signal_value = hidden ^ sender_convention
        signal = Signal(now, signal_value, now + delay + rng.choice((0, 1)))
        signal_by_seq[now] = signal
        signal_queue.append(signal)
        action_by_seq[now] = (action, reward)
        feedback_queue.append(
            Feedback(now, signal_value, action, reward, now + delay + rng.choice((0, 1)))
        )

    return {
        "delay": delay,
        "policy": policy,
        "sender_convention": sender_convention,
        "seed": seed,
        "reward_rate": correct / total,
        "mean_signal_age": signal_age_total / signal_age_count,
        "convention_updates": convention_updates,
        "misattributed_updates": misattributed_updates,
        "learned_by_end": int(learned_at is not None),
        "learned_at": learned_at if learned_at is not None else ROUNDS,
    }


def aggregate(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[int, str, int], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[(int(row["delay"]), str(row["policy"]), int(row["sender_convention"]))].append(row)
    summary: list[dict[str, object]] = []
    for (delay, policy, convention), group in sorted(groups.items()):
        summary.append(
            {
                "delay": delay,
                "policy": policy,
                "sender_convention": convention,
                "n_games": len(group),
                "mean_reward_rate": sum(float(row["reward_rate"]) for row in group) / len(group),
                "mean_signal_age": sum(float(row["mean_signal_age"]) for row in group) / len(group),
                "mean_convention_updates": sum(int(row["convention_updates"]) for row in group) / len(group),
                "mean_misattributed_updates": sum(int(row["misattributed_updates"]) for row in group) / len(group),
                "fraction_learned_by_end": sum(int(row["learned_by_end"]) for row in group) / len(group),
                "mean_learned_at": sum(int(row["learned_at"]) for row in group) / len(group),
            }
        )
    return summary


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def render(summary: list[dict[str, object]], path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), constrained_layout=False)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.80, bottom=0.18, wspace=0.28)
    colors = {
        "fixed_protocol": "#d55e00",
        "delay_blind_adaptive": "#cc79a7",
        "queue_aware_adaptive": "#009e73",
        "known_protocol": "#0072b2",
    }
    labels = {
        "fixed_protocol": "fixed protocol",
        "delay_blind_adaptive": "delay-blind adaptive",
        "queue_aware_adaptive": "queue-aware adaptive",
        "known_protocol": "known protocol",
    }
    for policy in POLICIES:
        for convention, linestyle in ((0, "-"), (1, "--")):
            rows = [row for row in summary if row["policy"] == policy and int(row["sender_convention"]) == convention]
            rows.sort(key=lambda row: int(row["delay"]))
            axes[0].plot(
                [int(row["delay"]) for row in rows],
                [float(row["mean_reward_rate"]) for row in rows],
                marker="o",
                linewidth=2,
                linestyle=linestyle,
                color=colors[policy],
                label=f"{labels[policy]} / c={convention}",
            )
            axes[1].plot(
                [int(row["delay"]) for row in rows],
                [float(row["mean_misattributed_updates"]) for row in rows],
                marker="o",
                linewidth=2,
                linestyle=linestyle,
                color=colors[policy],
                label=f"{labels[policy]} / c={convention}",
            )
    axes[0].set_title("Cross-play reward rate")
    axes[0].set_ylabel("fraction correct")
    axes[1].set_title("Misattributed feedback updates")
    axes[1].set_ylabel("mean updates / episode")
    for axis in axes:
        axis.set_xlabel("base communication delay")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    fig.suptitle("Delayed protocol formation under cross-play", fontsize=16, fontweight="bold")
    fig.text(0.01, 0.025, "Hidden bit switches stochastically; 20 games/cell; dashed lines use sender convention 1; exploratory only.", fontsize=9)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    seed = 20260730
    for delay in DELAYS:
        for policy in POLICIES:
            for convention in CONVENTIONS:
                for _ in range(GAMES_PER_CELL):
                    rows.append(run_game(delay, policy, convention, seed))
                    seed += 1
    summary = aggregate(rows)
    raw_path = OUT / "crossplay_delay_protocol_games.csv"
    summary_path = OUT / "crossplay_delay_protocol_summary.csv"
    figure_path = OUT / "crossplay_delay_protocol.png"
    report_path = OUT / "crossplay_delay_protocol_report.md"
    manifest_path = OUT / "CROSSPLAY_DELAY_PROTOCOL_MANIFEST.sha256"
    write_csv(raw_path, rows)
    write_csv(summary_path, summary)
    render(summary, figure_path)
    report_path.write_text(
        """# Delayed Protocol Formation under Cross-Play

This toy diagnostic takes the design lessons of Gessler et al., `OvercookedV2: Rethinking Overcooked for Zero-Shot Coordination` (arXiv:2503.17821), as a template: one agent has asymmetric information, the partner acts from partial observations, communication is stochastic and delayed, and the pair may need to form a protocol at test time.

The sender observes a hidden bit that can switch over time and transmits the bit through a convention-dependent binary signal. Feedback reports whether the receiver's action matched the current hidden bit. Signal and feedback delays include one-step jitter, creating possible reordering.

- `fixed_protocol`: assumes convention 0 and never adapts.
- `delay_blind_adaptive`: updates its convention belief using the latest arrived signal, ignoring sequence identity.
- `queue_aware_adaptive`: pairs feedback with the sequence-indexed signal/action record.
- `known_protocol`: reference policy given the sender's convention.

The plotted curves separate sender convention 0 and 1 to expose cross-play mismatch. This is not an implementation of OvercookedV2, and the results are not evidence about that benchmark. They are an exploratory test of whether sequence-aware feedback helps a minimal protocol learner under delay.
""",
        encoding="utf-8",
    )
    outputs = [raw_path, summary_path, figure_path, report_path]
    manifest_path.write_text("".join(f"{sha256(path)}  *{path}\n" for path in outputs), encoding="utf-8")
    print(f"Games: {len(rows)}")
    print(f"Summary: {summary_path}")
    print(f"Figure: {figure_path}")
    print(f"Report: {report_path}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
