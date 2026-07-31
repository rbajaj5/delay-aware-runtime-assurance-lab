"""Global-average versus local-concentration diagnostic for delayed coordination.

The mathematical analogy is limited and explicit: a global event mass can be
the same while its local concentration differs sharply. Here we compare a
spread delay schedule with a bursty schedule having the same mean delay, then
measure stale-decision mass and its temporal/spatial concentration.
"""

from __future__ import annotations

import csv
import hashlib
import math
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "20260730" / "delay_concentration"
AGENTS = 12
ROUNDS = 240
CAPACITY = 2
BASE_DELAY = 3
BURST_DELAY = 12
WINDOW = 20
GAMES_PER_CELL = 40
POLICIES = ("fifo_blind", "queue_aware_latest")
SCHEDULES = ("spread", "clustered_same_mean")


@dataclass(frozen=True)
class Message:
    sender: int
    created: int
    due: int
    value: int
    sequence: int


def neighbors(agent: int) -> tuple[int, int]:
    return ((agent - 1) % AGENTS, (agent + 1) % AGENTS)


def delay_for(schedule: str, now: int) -> int:
    if schedule == "spread":
        return BASE_DELAY
    # One 12-step burst every four rounds and zero otherwise: mean 3.
    return BURST_DELAY if now % 4 == 0 else 0


def choose_due(queue: list[Message], policy: str, now: int) -> tuple[list[Message], list[Message]]:
    due = [message for message in queue if message.due <= now]
    future = [message for message in queue if message.due > now]
    if policy == "fifo_blind":
        chosen = sorted(due, key=lambda message: message.sequence)[:CAPACITY]
    else:
        latest: dict[int, Message] = {}
        for message in due:
            if message.sender not in latest or message.sequence > latest[message.sender].sequence:
                latest[message.sender] = message
        chosen = sorted(latest.values(), key=lambda message: message.sequence)[:CAPACITY]
    selected = {message.sequence for message in chosen}
    remaining = future + [message for message in due if message.sequence not in selected]
    return chosen, remaining


def gini(values: list[float]) -> float:
    ordered = sorted(values)
    total = sum(ordered)
    if total == 0:
        return 0.0
    weighted = sum((index + 1) * value for index, value in enumerate(ordered))
    return (2 * weighted) / (len(ordered) * total) - (len(ordered) + 1) / len(ordered)


def run_game(schedule: str, policy: str, seed: int) -> dict[str, float | int | str]:
    rng = random.Random(seed)
    state = [rng.choice((-1, 1)) for _ in range(AGENTS)]
    latest_seen: list[dict[int, int]] = [dict() for _ in range(AGENTS)]
    queues: list[list[Message]] = [[] for _ in range(AGENTS)]
    sequence = 0
    events_by_round: list[int] = []
    events_by_agent = [0] * AGENTS
    total_decisions = 0

    for now in range(ROUNDS):
        for agent in range(AGENTS):
            if rng.random() < 0.08:
                state[agent] *= -1

        for receiver in range(AGENTS):
            chosen, queues[receiver] = choose_due(queues[receiver], policy, now)
            for message in chosen:
                latest_seen[receiver][message.sender] = message.value

        round_events = 0
        for receiver in range(AGENTS):
            for sender in neighbors(receiver):
                if latest_seen[receiver].get(sender) != state[sender]:
                    round_events += 1
                    events_by_agent[receiver] += 1
                total_decisions += 1
        events_by_round.append(round_events)

        for sender in range(AGENTS):
            delay = delay_for(schedule, now)
            for receiver in neighbors(sender):
                sequence += 1
                queues[receiver].append(Message(sender, now, now + delay, state[sender], sequence))

    total_events = sum(events_by_round)
    global_rate = total_events / total_decisions
    window_rates = [
        sum(events_by_round[start : start + WINDOW]) / (WINDOW * AGENTS * 2)
        for start in range(ROUNDS - WINDOW + 1)
    ]
    max_window_rate = max(window_rates)
    return {
        "schedule": schedule,
        "policy": policy,
        "seed": seed,
        "mean_delay": sum(delay_for(schedule, round_index) for round_index in range(ROUNDS)) / ROUNDS,
        "stale_event_rate": global_rate,
        "max_agent_stale_rate": max(events_by_agent) / (ROUNDS * 2),
        "gini_agent_stale_events": gini([float(value) for value in events_by_agent]),
        "max_window_stale_rate": max_window_rate,
        "temporal_concentration_ratio": max_window_rate / global_rate if global_rate else 0.0,
        "total_stale_events": total_events,
    }


def aggregate(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["schedule"]), str(row["policy"]))].append(row)
    summary: list[dict[str, object]] = []
    for (schedule, policy), group in sorted(groups.items()):
        summary.append(
            {
                "schedule": schedule,
                "policy": policy,
                "n_games": len(group),
                "mean_delay": sum(float(row["mean_delay"]) for row in group) / len(group),
                "mean_stale_event_rate": sum(float(row["stale_event_rate"]) for row in group) / len(group),
                "mean_max_agent_stale_rate": sum(float(row["max_agent_stale_rate"]) for row in group) / len(group),
                "mean_gini_agent_stale_events": sum(float(row["gini_agent_stale_events"]) for row in group) / len(group),
                "mean_max_window_stale_rate": sum(float(row["max_window_stale_rate"]) for row in group) / len(group),
                "mean_temporal_concentration_ratio": sum(float(row["temporal_concentration_ratio"]) for row in group) / len(group),
                "mean_total_stale_events": sum(int(row["total_stale_events"]) for row in group) / len(group),
            }
        )
    return summary


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def render(summary: list[dict[str, object]], path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=False)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.82, bottom=0.18, wspace=0.28)
    schedules = list(SCHEDULES)
    labels = {"spread": "spread delay", "clustered_same_mean": "clustered delay"}
    colors = {"fifo_blind": "#d55e00", "queue_aware_latest": "#0072b2"}
    policy_labels = {"fifo_blind": "blind FIFO", "queue_aware_latest": "queue-aware latest"}
    x = [0, 1]
    for policy in POLICIES:
        rows = [next(row for row in summary if row["schedule"] == schedule and row["policy"] == policy) for schedule in schedules]
        axes[0].plot(x, [float(row["mean_stale_event_rate"]) for row in rows], marker="o", linewidth=2.5, color=colors[policy], label=policy_labels[policy])
        axes[0].plot(x, [float(row["mean_max_window_stale_rate"]) for row in rows], marker="s", linestyle="--", linewidth=2, color=colors[policy], alpha=0.7, label=f"{policy_labels[policy]} / max window")
        axes[1].plot(x, [float(row["mean_gini_agent_stale_events"]) for row in rows], marker="o", linewidth=2.5, color=colors[policy], label=policy_labels[policy])
        axes[1].plot(x, [float(row["mean_temporal_concentration_ratio"]) for row in rows], marker="s", linestyle="--", linewidth=2, color=colors[policy], alpha=0.7, label=f"{policy_labels[policy]} / temporal ratio")
    axes[0].set_title("Global stale mass vs local burst")
    axes[0].set_ylabel("rate")
    axes[1].set_title("Spatial and temporal concentration")
    axes[1].set_ylabel("Gini / max-window-to-mean ratio")
    for axis in axes:
        axis.set_xticks(x, [labels[schedule] for schedule in schedules])
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    fig.suptitle("Same mean delay, different concentration of stale decisions", fontsize=16, fontweight="bold")
    fig.text(0.01, 0.025, f"12-agent ring; mean delay {BASE_DELAY}; window {WINDOW}; {GAMES_PER_CELL} games/cell; exploratory analogy only.", fontsize=9)
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
    for schedule in SCHEDULES:
        for policy in POLICIES:
            for _ in range(GAMES_PER_CELL):
                rows.append(run_game(schedule, policy, seed))
                seed += 1
    summary = aggregate(rows)
    spread_fifo = next(row for row in summary if row["schedule"] == "spread" and row["policy"] == "fifo_blind")
    clustered_fifo = next(row for row in summary if row["schedule"] == "clustered_same_mean" and row["policy"] == "fifo_blind")
    raw_path = OUT / "delay_concentration_games.csv"
    summary_path = OUT / "delay_concentration_summary.csv"
    figure_path = OUT / "delay_concentration.png"
    report_path = OUT / "delay_concentration_report.md"
    manifest_path = OUT / "DELAY_CONCENTRATION_MANIFEST.sha256"
    write_csv(raw_path, rows)
    write_csv(summary_path, summary)
    render(summary, figure_path)
    report_path.write_text(
        f"""# Delay Concentration Diagnostic

This experiment applies a limited methodological analogy from counterexample-driven analysis: equal global mass need not imply equal local concentration. It compares two delay schedules with the same mean delay of {BASE_DELAY}: a spread schedule with delay {BASE_DELAY} every round, and a clustered schedule with delay {BURST_DELAY} every fourth round and zero otherwise.

The multi-agent system is a {AGENTS}-agent ring. Each agent receives useful state updates from two neighbors and makes a stale decision when its latest applied neighbor state differs from the current state. `fifo_blind` processes messages by arrival order; `queue_aware_latest` retains the newest due update per sender.

Reported quantities separate global stale-event rate from concentration: maximum sliding-window rate, maximum per-agent rate, a spatial Gini index, and a temporal concentration ratio. In this run, spread FIFO had global stale-event rate {float(spread_fifo['mean_stale_event_rate']):.3f} and maximum-window rate {float(spread_fifo['mean_max_window_stale_rate']):.3f}; clustered FIFO had {float(clustered_fifo['mean_stale_event_rate']):.3f} and {float(clustered_fifo['mean_max_window_stale_rate']):.3f}. Thus equal mean delay did not preserve equal stale-event mass, and clustering did not automatically increase concentration in this model. The analogy is diagnostic only. It does not identify a plurisubharmonic function, a Monge-Ampere measure, or a theorem about queue dynamics.

All results are exploratory.
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
