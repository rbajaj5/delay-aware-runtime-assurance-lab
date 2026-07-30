"""Multi-agent queue discipline sweep with bounded background aging.

The experiment compares FIFO, strict useful-message priority, and a priority
policy that gives one service slot to sufficiently old background traffic.
It measures both coordination freshness and scheduler starvation. This is an
exploratory queueing diagnostic, not a safety certificate.
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
OUT = ROOT / "artifacts" / "20260730" / "fair_queue"
AGENTS = 12
ROUNDS = 180
CAPACITY = 2
AGING_THRESHOLD = 6
DELAYS = (0, 1, 2, 3, 5)
BACKGROUND_RATES = (0.0, 1.0, 2.0, 4.0, 8.0)
GAMES_PER_CELL = 10
POLICIES = ("fifo_blind", "priority_deduplicating", "priority_with_aging")


@dataclass(frozen=True)
class Message:
    sender: int | None
    created: int
    due: int
    value: int | None
    useful: bool
    sequence: int


def neighbors(agent: int) -> tuple[int, int]:
    return ((agent - 1) % AGENTS, (agent + 1) % AGENTS)


def poisson(rng: random.Random, rate: float) -> int:
    if rate <= 0:
        return 0
    threshold = math.exp(-rate)
    product = 1.0
    draws = 0
    while product > threshold:
        draws += 1
        product *= rng.random()
    return draws - 1


def select_due(queue: list[Message], policy: str, now: int) -> tuple[list[Message], list[Message]]:
    due = [message for message in queue if message.due <= now]
    future = [message for message in queue if message.due > now]
    if policy == "fifo_blind":
        chosen = sorted(due, key=lambda message: message.sequence)[:CAPACITY]
    else:
        latest_useful: dict[int, Message] = {}
        for message in due:
            if message.useful and (
                message.sender not in latest_useful
                or message.sequence > latest_useful[message.sender].sequence
            ):
                latest_useful[message.sender] = message
        useful = sorted(latest_useful.values(), key=lambda message: message.sequence)
        background = sorted(
            (message for message in due if not message.useful),
            key=lambda message: message.sequence,
        )
        chosen: list[Message] = []
        if policy == "priority_with_aging":
            aged = [message for message in background if now - message.created >= AGING_THRESHOLD]
            if aged:
                chosen.append(aged[0])
        chosen.extend(useful[: CAPACITY - len(chosen)])
        if len(chosen) < CAPACITY:
            selected = {message.sequence for message in chosen}
            chosen.extend(
                message
                for message in background
                if message.sequence not in selected
            )
            chosen = chosen[:CAPACITY]
    selected = {message.sequence for message in chosen}
    remaining = future + [message for message in due if message.sequence not in selected]
    return chosen, remaining


def run_game(delay: int, background_rate: float, policy: str, seed: int) -> dict[str, float | int | str]:
    rng = random.Random(seed)
    state = [rng.choice((-1, 1)) for _ in range(AGENTS)]
    latest_seen: list[dict[int, int]] = [dict() for _ in range(AGENTS)]
    queues: list[list[Message]] = [[] for _ in range(AGENTS)]
    sequence = 0
    useful_processed = 0
    background_processed = 0
    useful_wait_total = 0.0
    useful_wait_count = 0
    background_wait_total = 0.0
    background_wait_count = 0
    stale_decisions = 0
    decisions = 0
    total_backlog = 0
    max_backlog = 0

    for now in range(ROUNDS):
        for agent in range(AGENTS):
            if rng.random() < 0.08:
                state[agent] *= -1

        for receiver in range(AGENTS):
            chosen, queues[receiver] = select_due(queues[receiver], policy, now)
            for message in chosen:
                wait = now - message.created
                if message.useful:
                    useful_processed += 1
                    useful_wait_total += wait
                    useful_wait_count += 1
                    latest_seen[receiver][message.sender] = message.value
                else:
                    background_processed += 1
                    background_wait_total += wait
                    background_wait_count += 1

        for receiver in range(AGENTS):
            for neighbor in neighbors(receiver):
                if latest_seen[receiver].get(neighbor) != state[neighbor]:
                    stale_decisions += 1
                decisions += 1

        for sender in range(AGENTS):
            for receiver in neighbors(sender):
                sequence += 1
                queues[receiver].append(Message(sender, now, now + delay, state[sender], True, sequence))
            for _ in range(poisson(rng, background_rate)):
                sequence += 1
                queues[sender].append(Message(None, now, now, None, False, sequence))

        backlog = sum(len(queue) for queue in queues)
        total_backlog += backlog
        max_backlog = max(max_backlog, backlog)

    return {
        "delay": delay,
        "background_rate": background_rate,
        "policy": policy,
        "seed": seed,
        "stale_decision_rate": stale_decisions / decisions,
        "useful_processed": useful_processed,
        "background_processed": background_processed,
        "mean_useful_wait": useful_wait_total / useful_wait_count if useful_wait_count else 0.0,
        "mean_background_wait": background_wait_total / background_wait_count if background_wait_count else 0.0,
        "mean_backlog": total_backlog / ROUNDS,
        "max_backlog": max_backlog,
    }


def aggregate(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[int, float, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[(int(row["delay"]), float(row["background_rate"]), str(row["policy"]))].append(row)
    summary: list[dict[str, object]] = []
    for (delay, rate, policy), group in sorted(groups.items()):
        summary.append(
            {
                "delay": delay,
                "background_rate": rate,
                "policy": policy,
                "n_games": len(group),
                "mean_stale_decision_rate": sum(float(row["stale_decision_rate"]) for row in group) / len(group),
                "mean_useful_processed": sum(int(row["useful_processed"]) for row in group) / len(group),
                "mean_background_processed": sum(int(row["background_processed"]) for row in group) / len(group),
                "mean_useful_wait": sum(float(row["mean_useful_wait"]) for row in group) / len(group),
                "mean_background_wait": sum(float(row["mean_background_wait"]) for row in group) / len(group),
                "mean_backlog": sum(float(row["mean_backlog"]) for row in group) / len(group),
                "mean_max_backlog": sum(int(row["max_backlog"]) for row in group) / len(group),
            }
        )
    return summary


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def render(summary: list[dict[str, object]], path: Path) -> None:
    delay = 3
    subset = [row for row in summary if int(row["delay"]) == delay]
    rates = list(BACKGROUND_RATES)
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=False)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.88, bottom=0.14, hspace=0.38, wspace=0.23)
    panels = (
        ("mean_stale_decision_rate", "Stale decision rate", "fraction"),
        ("mean_useful_wait", "Useful-message wait", "rounds"),
        ("mean_background_processed", "Background processed", "messages / game"),
        ("mean_background_wait", "Background-message wait", "rounds"),
    )
    colors = {
        "fifo_blind": "#d55e00",
        "priority_deduplicating": "#0072b2",
        "priority_with_aging": "#009e73",
    }
    labels = {
        "fifo_blind": "blind FIFO",
        "priority_deduplicating": "priority + dedup",
        "priority_with_aging": "priority + aging",
    }
    for axis, (field, title, ylabel) in zip(axes.flat, panels):
        for policy in POLICIES:
            values = [
                float(next(row[field] for row in subset if float(row["background_rate"]) == rate and row["policy"] == policy))
                for rate in rates
            ]
            axis.plot(rates, values, marker="o", linewidth=2.5, color=colors[policy], label=labels[policy])
        axis.set_title(title)
        axis.set_xlabel("Background messages / agent / round")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.25)
        axis.legend()
    fig.suptitle(f"Fair queueing for delayed multi-agent coordination (delay = {delay})", fontsize=16, fontweight="bold")
    fig.text(0.01, 0.015, f"12-agent ring; capacity {CAPACITY}/agent/round; aging threshold {AGING_THRESHOLD}; {GAMES_PER_CELL} games/cell; exploratory only.", fontsize=9)
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
        for rate in BACKGROUND_RATES:
            for policy in POLICIES:
                for _ in range(GAMES_PER_CELL):
                    rows.append(run_game(delay, rate, policy, seed))
                    seed += 1
    summary = aggregate(rows)
    raw_path = OUT / "fair_queue_multiagent_games.csv"
    summary_path = OUT / "fair_queue_multiagent_summary.csv"
    figure_path = OUT / "fair_queue_multiagent.png"
    report_path = OUT / "fair_queue_multiagent_report.md"
    manifest_path = OUT / "FAIR_QUEUE_MULTIAGENT_MANIFEST.sha256"
    write_csv(raw_path, rows)
    write_csv(summary_path, summary)
    render(summary, figure_path)
    report_path.write_text(
        f"""# Fair Queueing for Delayed Multi-Agent Coordination

This sweep compares three finite-capacity queue disciplines on a {AGENTS}-agent ring. Each agent receives useful state updates from its two neighbors and low-value background messages. The service capacity is {CAPACITY} messages per agent per round, and useful messages are delayed by the tested horizon.

- `fifo_blind`: process due messages by arrival order.
- `priority_deduplicating`: retain the newest useful message per sender and serve useful traffic first.
- `priority_with_aging`: use the same useful-message priority, but reserve service for the oldest background message once it has waited at least {AGING_THRESHOLD} rounds.

The downstream metric is stale-decision rate: an agent's decision is stale when its latest applied neighbor state differs from that neighbor's current state. The figure shows the delay-3 slice; the CSV includes all tested delays and loads.

The aging policy is a trade-off probe, not a deployment recommendation. Its parameters, starvation bounds, message taxonomy, authentication, auditability, and failure handling would need to be specified before use in a safety-relevant system.

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
