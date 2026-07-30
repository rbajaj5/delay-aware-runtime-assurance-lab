"""Exploratory multi-agent Hex delay/cascade simulation.

Four agents share one board: agents 1-2 play blue and 3-4 play yellow.
Each agent has a local board and receives broadcast move messages with delay.
The red-line metric is a deliberately simple local threat threshold: an
agent flags the opponent as near a crossing when its local shortest-path cost
is <= 2. We compare blind local views with queue-aware pending overlays.
"""

from __future__ import annotations

import csv
import hashlib
import heapq
import math
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis" / "out" / "hex_delay_exploratory_20260730"
N = 7
DELAYS = (0, 1, 2, 3, 5)
POLICIES = ("blind_fleet", "queue_aware_fleet")
AGENTS = (1, 2, 3, 4)
COLORS = {1: 1, 2: 1, 3: 2, 4: 2}
GAMES = 20
SEED_BASE = 2026073400
RED_LINE_COST = 2.0


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def neighbors(r: int, c: int):
    return [(r, c - 1), (r, c + 1), (r - 1, c), (r + 1, c), (r - 1, c + 1), (r + 1, c - 1)]


def copy_board(board):
    return [row[:] for row in board]


def path_cost(board, color: int) -> float:
    heap: list[tuple[float, tuple[int, int]]] = []
    distances: dict[tuple[int, int], float] = {}
    starts = [(r, 0) for r in range(N)] if color == 1 else [(0, c) for c in range(N)]
    target = (lambda r, c: c == N - 1) if color == 1 else (lambda r, c: r == N - 1)
    for cell in starts:
        r, c = cell
        if board[r][c] == 3 - color:
            continue
        cost = 0.0 if board[r][c] == color else 1.0
        if cost < distances.get(cell, math.inf):
            distances[cell] = cost
            heapq.heappush(heap, (cost, cell))
    while heap:
        cost, cell = heapq.heappop(heap)
        if cost != distances.get(cell):
            continue
        r, c = cell
        if target(r, c):
            return cost
        for nr, nc in neighbors(r, c):
            if not (0 <= nr < N and 0 <= nc < N) or board[nr][nc] == 3 - color:
                continue
            next_cost = cost + (0.0 if board[nr][nc] == color else 1.0)
            if next_cost < distances.get((nr, nc), math.inf):
                distances[(nr, nc)] = next_cost
                heapq.heappush(heap, (next_cost, (nr, nc)))
    return float(N * N)


def choose_move(board, color: int, rng: random.Random):
    legal = [(r, c) for r in range(N) for c in range(N) if board[r][c] == 0]
    if not legal:
        return None
    scored = []
    opponent = 3 - color
    for r, c in legal:
        candidate = copy_board(board)
        candidate[r][c] = color
        own = path_cost(candidate, color)
        opp = path_cost(candidate, opponent)
        scored.append(((-own, opp), (r, c)))
    best_key = max(key for key, _ in scored)
    return rng.choice([cell for key, cell in scored if key == best_key])


def overlay(local, pending, agent: int):
    board = copy_board(local)
    for _, recipient, r, c, color in pending:
        if recipient == agent and board[r][c] == 0:
            board[r][c] = color
    return board


def simulate(delay: int, policy: str, seed: int):
    rng = random.Random(seed)
    true_board = [[0 for _ in range(N)] for _ in range(N)]
    local = {agent: copy_board(true_board) for agent in AGENTS}
    pending = []
    conflicts = 0
    false_red_lines = 0
    missed_red_lines = 0
    red_line_events = 0
    cascade_turns = 0
    prior_red = {agent: False for agent in AGENTS}
    moves = 0
    turn = 0
    while moves < N * N and turn < N * N * 4:
        agent = AGENTS[turn % len(AGENTS)]
        color = COLORS[agent]
        due = [item for item in pending if item[0] <= turn and item[1] == agent]
        pending = [item for item in pending if not (item[0] <= turn and item[1] == agent)]
        for _, _, r, c, payload_color in due:
            local[agent][r][c] = payload_color
        decision = overlay(local[agent], pending, agent) if policy == "queue_aware_fleet" else copy_board(local[agent])
        local_threat = path_cost(decision, 3 - color) <= RED_LINE_COST
        true_threat = path_cost(true_board, 3 - color) <= RED_LINE_COST
        red_line_events += int(local_threat)
        false_red_lines += int(local_threat and not true_threat)
        missed_red_lines += int((not local_threat) and true_threat)
        if local_threat and prior_red[agent]:
            cascade_turns += 1
        prior_red[agent] = local_threat
        move = choose_move(decision, color, rng)
        if move is not None:
            r, c = move
            if true_board[r][c] != 0:
                conflicts += 1
            else:
                true_board[r][c] = color
                local[agent][r][c] = color
                moves += 1
                if delay == 0:
                    for recipient in AGENTS:
                        if recipient != agent:
                            local[recipient][r][c] = color
                else:
                    for recipient in AGENTS:
                        if recipient != agent:
                            pending.append((turn + delay, recipient, r, c, color))
        turn += 1
    return {
        "delay": delay,
        "policy": policy,
        "seed": seed,
        "agents": len(AGENTS),
        "turns": turn,
        "moves": moves,
        "stale_conflicts": conflicts,
        "red_line_events": red_line_events,
        "false_red_lines": false_red_lines,
        "missed_red_lines": missed_red_lines,
        "cascade_turns": cascade_turns,
        "full_board": moves == N * N,
    }


def write_csv(path: Path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for delay in DELAYS:
        for index in range(GAMES):
            seed = SEED_BASE + delay * 10000 + index
            for policy in POLICIES:
                rows.append(simulate(delay, policy, seed))

    summary = []
    for delay in DELAYS:
        for policy in POLICIES:
            subset = [row for row in rows if row["delay"] == delay and row["policy"] == policy]
            summary.append({
                "delay": delay,
                "policy": policy,
                "n_games": len(subset),
                "mean_stale_conflicts": sum(row["stale_conflicts"] for row in subset) / len(subset),
                "mean_false_red_lines": sum(row["false_red_lines"] for row in subset) / len(subset),
                "mean_missed_red_lines": sum(row["missed_red_lines"] for row in subset) / len(subset),
                "mean_cascade_turns": sum(row["cascade_turns"] for row in subset) / len(subset),
                "full_board_rate": sum(row["full_board"] for row in subset) / len(subset),
            })

    detail = OUT / "hex_multiagent_delay_cascade_games.csv"
    summary_path = OUT / "hex_multiagent_delay_cascade_summary.csv"
    write_csv(detail, rows)
    write_csv(summary_path, summary)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.0))
    colors = {"blind_fleet": "#D55E00", "queue_aware_fleet": "#0072B2"}
    for policy in POLICIES:
        selected = [row for row in summary if row["policy"] == policy]
        axes[0].plot([row["delay"] for row in selected], [row["mean_stale_conflicts"] for row in selected], marker="o", linewidth=3, color=colors[policy], label=policy)
        axes[1].plot([row["delay"] for row in selected], [row["mean_false_red_lines"] for row in selected], marker="o", linewidth=3, color=colors[policy], label=f"{policy}: false red-line")
        axes[1].plot([row["delay"] for row in selected], [row["mean_missed_red_lines"] for row in selected], marker="x", linewidth=2, linestyle="--", color=colors[policy], label=f"{policy}: missed red-line")
    axes[0].set_title("Four-agent stale conflicts")
    axes[0].set_ylabel("Mean conflicts per game")
    axes[1].set_title("Local red-line disagreement")
    axes[1].set_ylabel("Mean events per game")
    for axis in axes:
        axis.set_xlabel("Broadcast delay (moves)")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    fig.suptitle("Multi-agent delayed state: blind versus queue-aware fleet", fontsize=15, fontweight="bold")
    fig.text(0.01, 0.01, f"7x7 Hex; 4 agents; {GAMES} games per cell; red-line metric is a toy path-cost threshold; exploratory only.", fontsize=8)
    fig.tight_layout(rect=[0, 0.04, 1, 0.95])
    figure = OUT / "hex_multiagent_delay_cascade.png"
    fig.savefig(figure, dpi=220, facecolor="white")
    plt.close(fig)

    report = OUT / "hex_multiagent_delay_cascade_report.md"
    report.write_text(
        "# Multi-Agent Delayed-State Cascade\n\n"
        f"Four agents share a {N}x{N} Hex board, two per color. Every applied move is broadcast to the other three agents with the listed delay. `blind_fleet` acts from delivered local state; `queue_aware_fleet` overlays pending messages before acting.\n\n"
        "A red-line event is a toy threshold (`opponent shortest-path cost <= 2`) evaluated on the acting agent's local board. A false red-line occurs when the local view triggers but the authoritative board does not; a missed red-line is the reverse. Consecutive local triggers by the same agent are counted as cascade turns.\n\n"
        "This is a mechanism-level analogue of delayed multi-agent synchronization and rigid automated response. It is not a forecast, military model, or evidence for the source paper's scenarios.\n\n"
        "| delay | policy | games | mean conflicts | false red-lines | missed red-lines | cascade turns | full board rate |\n|---:|---|---:|---:|---:|---:|---:|---:|\n"
        + "\n".join(f"| {row['delay']} | {row['policy']} | {row['n_games']} | {row['mean_stale_conflicts']:.17g} | {row['mean_false_red_lines']:.17g} | {row['mean_missed_red_lines']:.17g} | {row['mean_cascade_turns']:.17g} | {row['full_board_rate']:.17g} |" for row in summary)
        + "\n",
        encoding="utf-8",
    )
    manifest = OUT / "HEX_MULTIAGENT_DELAY_CASCADE_MANIFEST.sha256"
    manifest.write_text("\n".join(f"{sha256(path)} *{path}" for path in [detail, summary_path, figure, report, Path(__file__)]) + "\n", encoding="utf-8")
    print(f"Games: {len(rows)}")
    print(f"Summary: {summary_path}")
    print(f"Figure: {figure}")
    print(f"Manifest: {manifest}")


if __name__ == "__main__":
    main()
