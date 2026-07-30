"""Exploratory message-passing simulation with low-value queue pollution.

Useful messages represent coordination updates. Background messages represent
low-value work that still consumes finite service capacity. We compare blind
FIFO processing with a queue-aware policy that prioritizes useful messages and
deduplicates older useful messages from the same sender.
"""

from __future__ import annotations

import csv
import hashlib
import math
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis" / "out" / "hex_delay_exploratory_20260730"
AGENTS = 12
ROUNDS = 180
DELAYS = (0, 1, 2, 3, 5)
BACKGROUND_RATES = (0.0, 1.0, 2.0, 4.0, 8.0)
POLICIES = ("fifo_blind", "priority_deduplicating")
GAMES = 20
CAPACITY = 2
STALE_AFTER = 1
SEED_BASE = 2026073900


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def neighbors(agent: int) -> tuple[int, int]:
    return ((agent - 1) % AGENTS, (agent + 1) % AGENTS)


def poisson(rng: random.Random, rate: float) -> int:
    if rate <= 0.0:
        return 0
    threshold = math.exp(-rate)
    product = 1.0
    count = 0
    while product > threshold:
        product *= rng.random()
        count += 1
    return count - 1


def simulate(delay: int, background_rate: float, policy: str, seed: int) -> dict[str, float | int | str]:
    rng = random.Random(seed)
    queues: dict[int, list[dict[str, int | str]]] = {agent: [] for agent in range(AGENTS)}
    pending: list[dict[str, int | str]] = []
    sequence = {agent: 0 for agent in range(AGENTS)}
    arrival_order = 0
    useful_processed = 0
    useful_fresh = 0
    useful_stale = 0
    useful_dropped = 0
    background_processed = 0
    backlog_sum = 0
    max_backlog = 0
    useful_wait_sum = 0
    useful_wait_count = 0

    def enqueue(message: dict[str, int | str]) -> None:
        nonlocal arrival_order
        message["arrival_order"] = arrival_order
        arrival_order += 1
        queues[int(message["receiver"])].append(message)

    for round_index in range(ROUNDS):
        due = [message for message in pending if int(message["due"]) <= round_index]
        pending = [message for message in pending if int(message["due"]) > round_index]
        for message in sorted(due, key=lambda item: int(item["arrival_order"])):
            enqueue(message)

        for sender in range(AGENTS):
            for receiver in neighbors(sender):
                sequence[sender] += 1
                pending.append({
                    "due": round_index + delay,
                    "receiver": receiver,
                    "sender": sender,
                    "kind": "useful",
                    "sent": round_index,
                    "sequence": sequence[sender],
                    "arrival_order": arrival_order,
                })
                arrival_order += 1

        for receiver in range(AGENTS):
            for _ in range(poisson(rng, background_rate)):
                pending.append({
                    "due": round_index,
                    "receiver": receiver,
                    "sender": -1,
                    "kind": "background",
                    "sent": round_index,
                    "sequence": 0,
                    "arrival_order": arrival_order,
                })
                arrival_order += 1

        for receiver in range(AGENTS):
            queue = queues[receiver]
            if policy == "priority_deduplicating":
                latest: dict[int, dict[str, int | str]] = {}
                background: list[dict[str, int | str]] = []
                for message in queue:
                    if message["kind"] == "useful":
                        sender = int(message["sender"])
                        if sender not in latest or int(message["sequence"]) > int(latest[sender]["sequence"]):
                            latest[sender] = message
                    else:
                        background.append(message)
                useful_queue = sorted(latest.values(), key=lambda item: int(item["arrival_order"]))
                queue = useful_queue + sorted(background, key=lambda item: int(item["arrival_order"]))
                queues[receiver] = queue

            selected = queue[:CAPACITY]
            queues[receiver] = queue[CAPACITY:]
            useful_selected = 0
            for message in selected:
                if message["kind"] == "useful":
                    useful_selected += 1
                    useful_processed += 1
                    wait = round_index - int(message["sent"])
                    useful_wait_sum += wait
                    useful_wait_count += 1
                    if wait <= STALE_AFTER:
                        useful_fresh += 1
                    else:
                        useful_stale += 1
                else:
                    background_processed += 1
            useful_dropped += max(0, min(CAPACITY, len([item for item in selected if item["kind"] == "useful"])) - useful_selected)

        backlog = sum(len(queue) for queue in queues.values()) + len(pending)
        backlog_sum += backlog
        max_backlog = max(max_backlog, backlog)

    total_useful_expected = AGENTS * 2 * ROUNDS
    return {
        "delay": delay,
        "background_rate": background_rate,
        "policy": policy,
        "seed": seed,
        "agents": AGENTS,
        "rounds": ROUNDS,
        "capacity_per_agent": CAPACITY,
        "useful_expected": total_useful_expected,
        "useful_processed": useful_processed,
        "useful_fresh": useful_fresh,
        "useful_stale": useful_stale,
        "useful_delivery_rate": useful_processed / total_useful_expected,
        "useful_fresh_rate": useful_fresh / useful_processed if useful_processed else 0.0,
        "stale_useful_rate": useful_stale / useful_processed if useful_processed else 0.0,
        "background_processed": background_processed,
        "mean_useful_wait": useful_wait_sum / useful_wait_count if useful_wait_count else 0.0,
        "mean_backlog": backlog_sum / ROUNDS,
        "max_backlog": max_backlog,
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for delay in DELAYS:
        for background_rate in BACKGROUND_RATES:
            for index in range(GAMES):
                seed = SEED_BASE + delay * 100000 + int(background_rate * 1000) * 100 + index
                for policy in POLICIES:
                    rows.append(simulate(delay, background_rate, policy, seed))

    summary = []
    for delay in DELAYS:
        for background_rate in BACKGROUND_RATES:
            for policy in POLICIES:
                subset = [row for row in rows if row["delay"] == delay and row["background_rate"] == background_rate and row["policy"] == policy]
                summary.append({
                    "delay": delay,
                    "background_rate": background_rate,
                    "policy": policy,
                    "n_games": len(subset),
                    "mean_useful_delivery_rate": sum(row["useful_delivery_rate"] for row in subset) / len(subset),
                    "mean_useful_fresh_rate": sum(row["useful_fresh_rate"] for row in subset) / len(subset),
                    "mean_stale_useful_rate": sum(row["stale_useful_rate"] for row in subset) / len(subset),
                    "mean_useful_wait": sum(row["mean_useful_wait"] for row in subset) / len(subset),
                    "mean_backlog": sum(row["mean_backlog"] for row in subset) / len(subset),
                    "mean_max_backlog": sum(row["max_backlog"] for row in subset) / len(subset),
                    "mean_background_processed": sum(row["background_processed"] for row in subset) / len(subset),
                })

    detail = OUT / "semantic_queue_pollution_games.csv"
    summary_path = OUT / "semantic_queue_pollution_summary.csv"
    write_csv(detail, rows)
    write_csv(summary_path, summary)

    colors = {"fifo_blind": "#D55E00", "priority_deduplicating": "#0072B2"}
    fig, axes = plt.subplots(2, 2, figsize=(13.0, 9.0))
    selected_delay = 3
    for policy in POLICIES:
        selected = [row for row in summary if row["delay"] == selected_delay and row["policy"] == policy]
        label = "blind FIFO" if policy == "fifo_blind" else "priority + dedup"
        axes[0, 0].plot([row["background_rate"] for row in selected], [row["mean_useful_fresh_rate"] for row in selected], marker="o", linewidth=3, color=colors[policy], label=label)
        axes[0, 1].plot([row["background_rate"] for row in selected], [row["mean_stale_useful_rate"] for row in selected], marker="o", linewidth=3, color=colors[policy], label=label)
        axes[1, 0].plot([row["background_rate"] for row in selected], [row["mean_useful_wait"] for row in selected], marker="o", linewidth=3, color=colors[policy], label=label)
        axes[1, 1].plot([row["background_rate"] for row in selected], [row["mean_backlog"] for row in selected], marker="o", linewidth=3, color=colors[policy], label=label)
    axes[0, 0].set_title("Fresh useful delivery")
    axes[0, 0].set_ylabel("Rate")
    axes[0, 1].set_title("Stale useful delivery")
    axes[0, 1].set_ylabel("Rate")
    axes[1, 0].set_title("Useful-message wait")
    axes[1, 0].set_ylabel("Mean rounds")
    axes[1, 1].set_title("Total queue backlog")
    axes[1, 1].set_ylabel("Mean messages")
    for axis in axes.flat:
        axis.set_xlabel("Background messages / agent / round")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    fig.suptitle("Queue pollution under delayed useful messages (delay = 3)", fontsize=16, fontweight="bold")
    fig.text(0.01, 0.01, f"12-agent ring; capacity {CAPACITY}/agent/round; {GAMES} games/cell; exploratory only.", fontsize=8)
    fig.tight_layout(rect=[0, 0.04, 1, 0.95])
    figure = OUT / "semantic_queue_pollution.png"
    fig.savefig(figure, dpi=220, facecolor="white")
    plt.close(fig)

    report = OUT / "semantic_queue_pollution_report.md"
    report.write_text(
        "# Semantic Queue-Pollution Simulation\n\n"
        "Useful messages are coordination updates. Background messages are low-value work that still consumes finite processing capacity. Each agent can process two messages per round. `fifo_blind` processes arrival order. `priority_deduplicating` prioritizes useful messages and retains only the newest useful message from each sender.\n\n"
        "The experiment is a queueing abstraction, not a model of human worth or semantic meaning. It isolates a systems question: can low-value traffic make a delay-blind system miss or act on stale high-value updates?\n\n"
        "The figure shows delay 3. The CSV includes all delays and background-load levels.\n\n"
        "| delay | background rate | policy | games | fresh useful rate | stale useful rate | useful wait | mean backlog |\n|---:|---:|---|---:|---:|---:|---:|---:|\n"
        + "\n".join(
            f"| {row['delay']} | {row['background_rate']} | {row['policy']} | {row['n_games']} | {row['mean_useful_fresh_rate']:.17g} | {row['mean_stale_useful_rate']:.17g} | {row['mean_useful_wait']:.17g} | {row['mean_backlog']:.17g} |"
            for row in summary
        )
        + "\n\nAll results are exploratory. The priority policy is a queue discipline, not a safety certificate; a deployment would need an explicit policy for message classes, starvation, auditability, and failure handling.\n",
        encoding="utf-8",
    )
    manifest = OUT / "SEMANTIC_QUEUE_POLLUTION_MANIFEST.sha256"
    manifest.write_text("\n".join(f"{sha256(path)} *{path}" for path in [detail, summary_path, figure, report, Path(__file__)]) + "\n", encoding="utf-8")
    print(f"Games: {len(rows)}")
    print(f"Summary: {summary_path}")
    print(f"Figure: {figure}")
    print(f"Report: {report}")
    print(f"Manifest: {manifest}")


if __name__ == "__main__":
    main()
