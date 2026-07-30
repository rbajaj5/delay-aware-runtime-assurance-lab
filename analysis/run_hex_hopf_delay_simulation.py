"""Exploratory Hex simulation with a delayed Hopf-fiber state.

The standard Hopf map S^3 -> S^2 is used as a coordinate construction. Each
cell has a fixed projected base point and a time-varying hidden S^1 fiber
phase. The phase is coupled locally, then transmitted with delay. Blind play
uses delivered phase messages; queue-aware play overlays pending phase
messages. This is a toy physics-inspired communication model, not evidence
for the external paper's unified-field claims and not v6 evidence.
"""

from __future__ import annotations

import csv
import hashlib
import heapq
import json
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
POLICIES = ("blind_delay", "queue_aware")
GAMES = 30
SEED_BASE = 2026073100
PAPER_URL = "https://philpapers.org/archive/NIETTU.pdf"
TRANSPORT_MODE = "adversarial_selective_hold"


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


def copy_phase(phase):
    return [row[:] for row in phase]


def circular_distance(a: float, b: float) -> float:
    return abs(math.atan2(math.sin(a - b), math.cos(a - b)))


def hopf_coordinates(eta: float, delta: float, fiber_phase: float):
    """Return (z1,z2,h) with the Hopf fiber phase explicit."""
    xi1 = fiber_phase + delta / 2.0
    xi2 = fiber_phase - delta / 2.0
    z1 = math.cos(eta) * complex(math.cos(xi1), math.sin(xi1))
    z2 = math.sin(eta) * complex(math.cos(xi2), math.sin(xi2))
    base = (2 * (z1 * z2.conjugate()).real, 2 * (z1 * z2.conjugate()).imag, abs(z1) ** 2 - abs(z2) ** 2)
    return z1, z2, base


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


def crossing(board: list[list[int]], color: int) -> bool:
    starts = [(r, 0) for r in range(N)] if color == 1 else [(0, c) for c in range(N)]
    target = (lambda r, c: c == N - 1) if color == 1 else (lambda r, c: r == N - 1)
    stack = [cell for cell in starts if board[cell[0]][cell[1]] == color]
    seen = set(stack)
    while stack:
        r, c = stack.pop()
        if target(r, c):
            return True
        for nr, nc in neighbors(r, c):
            if 0 <= nr < N and 0 <= nc < N and (nr, nc) not in seen and board[nr][nc] == color:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return False


def phase_disagreement(phase, r: int, c: int) -> float:
    values = []
    for nr, nc in neighbors(r, c):
        if 0 <= nr < N and 0 <= nc < N and phase[nr][nc] is not None:
            values.append(circular_distance(phase[r][c], phase[nr][nc]))
    return sum(values) / len(values) if values else math.pi


def queue_overlay(local_board, local_phase, pending, player: int):
    board = copy_board(local_board)
    phase = copy_phase(local_phase)
    for _, recipient, r, c, payload_player, payload_phase in pending:
        if recipient != player:
            continue
        if board[r][c] == 0:
            board[r][c] = payload_player
        phase[r][c] = payload_phase
    return board, phase


def choose_move(board, phase, player: int, rng: random.Random):
    legal = [(r, c) for r in range(N) for c in range(N) if board[r][c] == 0]
    if not legal:
        return None
    scored = []
    opponent = 3 - player
    for r, c in legal:
        candidate = copy_board(board)
        candidate[r][c] = player
        own_cost = path_cost(candidate, player)
        opp_cost = path_cost(candidate, opponent)
        disagreement = phase_disagreement(phase, r, c)
        scored.append(((-own_cost, opp_cost, -disagreement), (r, c)))
    best_key = max(key for key, _ in scored)
    candidates = [cell for key, cell in scored if key == best_key]
    return rng.choice(candidates)


def initial_phase(seed: int):
    rng = random.Random(seed)
    phase = []
    for r in range(N):
        row = []
        for c in range(N):
            row.append(rng.uniform(-math.pi, math.pi))
        phase.append(row)
    return phase


def evolve_phase(phase, omega, rng: random.Random):
    next_phase = copy_phase(phase)
    for r in range(N):
        for c in range(N):
            local = [phase[nr][nc] for nr, nc in neighbors(r, c) if 0 <= nr < N and 0 <= nc < N]
            mean_sin = sum(math.sin(value) for value in local) / len(local)
            mean_cos = sum(math.cos(value) for value in local) / len(local)
            mean_phase = math.atan2(mean_sin, mean_cos)
            drift = omega[r][c] + 0.18 * math.sin(mean_phase - phase[r][c]) + rng.gauss(0.0, 0.025)
            next_phase[r][c] = math.atan2(math.sin(phase[r][c] + drift), math.cos(phase[r][c] + drift))
    return next_phase


def transport_delay(base_delay: int, payload_phase: float, r: int, c: int, hold_strength: int = 1) -> int:
    """Apply a deliberately selective hold to phase-bearing messages."""
    if base_delay == 0:
        return 0
    trigger = math.sin(payload_phase + 0.31 * r - 0.17 * c) > 0.0
    return base_delay + (hold_strength if trigger else 0)


def simulate_game(delay: int, policy: str, seed: int, keep_trace: bool = False, hold_strength: int = 1):
    rng = random.Random(seed)
    board = [[0 for _ in range(N)] for _ in range(N)]
    phase = initial_phase(seed + 17)
    omega = [[rng.uniform(-0.08, 0.08) for _ in range(N)] for _ in range(N)]
    local_board = {1: copy_board(board), 2: copy_board(board)}
    local_phase = {1: copy_phase(phase), 2: copy_phase(phase)}
    pending = []
    trace = []
    stale_conflicts = 0
    phase_errors = []
    extra_holds = []
    moves = 0
    turn = 0
    while moves < N * N and turn < N * N * 5:
        phase = evolve_phase(phase, omega, rng)
        player = 1 if turn % 2 == 0 else 2
        due = [item for item in pending if item[0] <= turn and item[1] == player]
        pending = [item for item in pending if not (item[0] <= turn and item[1] == player)]
        for _, _, r, c, payload_player, payload_phase in due:
            local_board[player][r][c] = payload_player
            local_phase[player][r][c] = payload_phase
        if policy == "queue_aware":
            decision_board, decision_phase = queue_overlay(local_board[player], local_phase[player], pending, player)
        else:
            decision_board, decision_phase = copy_board(local_board[player]), copy_phase(local_phase[player])
        move = choose_move(decision_board, decision_phase, player, rng)
        stale = False
        applied = False
        selected_phase_error = None
        if move is not None:
            r, c = move
            selected_phase_error = circular_distance(decision_phase[r][c], phase[r][c])
            phase_errors.append(selected_phase_error)
            if board[r][c] != 0:
                stale_conflicts += 1
                stale = True
            else:
                board[r][c] = player
                local_board[player][r][c] = player
                local_phase[player][r][c] = phase[r][c]
                if delay == 0:
                    local_board[3 - player][r][c] = player
                    local_phase[3 - player][r][c] = phase[r][c]
                else:
                    actual_delay = transport_delay(delay, phase[r][c], r, c, hold_strength)
                    extra_holds.append(actual_delay - delay)
                    due_turn = turn + actual_delay
                    pending.append((due_turn, 3 - player, r, c, player, phase[r][c]))
                moves += 1
                applied = True
        if keep_trace:
            trace.append({"turn": turn, "player": player, "r": move[0] if move else None, "c": move[1] if move else None, "applied": applied, "stale_conflict": stale, "selected_phase_error": selected_phase_error, "true_board": copy_board(board), "pending_count": len(pending)})
        turn += 1
    return {"delay": delay, "policy": policy, "seed": seed, "turns": turn, "moves": moves, "stale_conflicts": stale_conflicts, "mean_phase_error": sum(phase_errors) / len(phase_errors), "p95_phase_error": sorted(phase_errors)[int(0.95 * (len(phase_errors) - 1))], "mean_extra_hold": sum(extra_holds) / len(extra_holds) if extra_holds else 0.0, "full_board": moves == N * N, "blue_win": crossing(board, 1), "yellow_win": crossing(board, 2), "trace": trace}


def write_csv(path: Path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    traces = {}
    for delay in DELAYS:
        for index in range(GAMES):
            seed = SEED_BASE + delay * 10000 + index
            for policy in POLICIES:
                result = simulate_game(delay, policy, seed, keep_trace=(delay == 3 and index == 0))
                if result["trace"]:
                    traces[policy] = result["trace"]
                rows.append({key: value for key, value in result.items() if key != "trace"})

    summary = []
    for delay in DELAYS:
        for policy in POLICIES:
            subset = [row for row in rows if row["delay"] == delay and row["policy"] == policy]
            conflicts = [int(row["stale_conflicts"]) for row in subset]
            phase_errors = [float(row["mean_phase_error"]) for row in subset]
            summary.append({"delay": delay, "policy": policy, "transport_mode": TRANSPORT_MODE, "n_games": len(subset), "mean_stale_conflicts": sum(conflicts) / len(conflicts), "any_conflict_rate": sum(value > 0 for value in conflicts) / len(conflicts), "mean_phase_error": sum(phase_errors) / len(phase_errors), "p95_phase_error": sorted(float(row["p95_phase_error"]) for row in subset)[int(0.95 * (len(subset) - 1))], "mean_extra_hold": sum(float(row["mean_extra_hold"]) for row in subset) / len(subset), "full_board_rate": sum(bool(row["full_board"]) for row in subset) / len(subset), "blue_win_rate": sum(bool(row["blue_win"]) for row in subset) / len(subset), "yellow_win_rate": sum(bool(row["yellow_win"]) for row in subset) / len(subset)})

    detail = OUT / "hex_hopf_delay_simulation_games.csv"
    summary_path = OUT / "hex_hopf_delay_simulation_summary.csv"
    trace_path = OUT / "hex_hopf_delay_trace_delay3_seed0.json"
    write_csv(detail, rows)
    write_csv(summary_path, summary)
    trace_path.write_text(json.dumps(traces, indent=2), encoding="utf-8")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
    for policy, color in [("blind_delay", "#D55E00"), ("queue_aware", "#0072B2")]:
        selected = [row for row in summary if row["policy"] == policy]
        axes[0].plot([row["delay"] for row in selected], [row["mean_phase_error"] for row in selected], marker="o", linewidth=3, label=policy, color=color)
        axes[1].plot([row["delay"] for row in selected], [row["mean_stale_conflicts"] for row in selected], marker="o", linewidth=3, label=policy, color=color)
    axes[0].set_title("Hopf-fiber phase error")
    axes[0].set_ylabel("Mean circular phase error (radians)")
    axes[1].set_title("Authoritative stale-cell conflicts")
    axes[1].set_ylabel("Mean conflicts per game")
    for axis in axes:
        axis.set_xlabel("Message delay (moves)")
        axis.grid(alpha=0.25)
        axis.legend()
    fig.suptitle("Exploratory Hex delay sweep with a hidden Hopf-fiber phase", fontsize=15, fontweight="bold")
    fig.text(0.01, 0.01, f"7x7 Hex; {GAMES} games per cell; Hopf coordinates are a toy internal state; not v6 evidence.", fontsize=8)
    fig.tight_layout(rect=[0, 0.04, 1, 0.95])
    figure = OUT / "hex_hopf_delay_simulation.png"
    fig.savefig(figure, dpi=220, facecolor="white")
    plt.close(fig)

    report = OUT / "hex_hopf_delay_simulation_report.md"
    report.write_text("# Exploratory Hex Hopf-Fiber Delay Simulation\n\n" + f"The simulation uses the standard Hopf coordinate construction `z1=cos(eta)e^(i xi1)`, `z2=sin(eta)e^(i xi2)`, with a time-varying fiber phase and a fixed projected base coordinate per cell. Neighbor-coupled phase drift is transmitted through the same delayed move queue as board occupancy.\n\nTransport mode: `{TRANSPORT_MODE}`. For nonzero base delay, phase-bearing messages satisfying a deterministic phase/cell predicate receive one additional hidden hold step. This is deliberately adversarial transport, not an honest FIFO channel.\n\nThe `blind_delay` policy uses delivered local phase state; `queue_aware` overlays pending phase payloads before selecting a move. The phase model is a toy communication/oscillator model. It is not a physical derivation from the supplied paper and does not support the paper's unified-field claims.\n\nExternal context: `" + PAPER_URL + "`.\n\n" + "| delay | policy | games | mean phase error | p95 phase error | mean extra hold | mean stale conflicts | any conflict rate |\n|---:|---|---:|---:|---:|---:|---:|---:|\n" + "\n".join(f"| {row['delay']} | {row['policy']} | {row['n_games']} | {row['mean_phase_error']:.17g} | {row['p95_phase_error']:.17g} | {row['mean_extra_hold']:.17g} | {row['mean_stale_conflicts']:.17g} | {row['any_conflict_rate']:.17g} |" for row in summary) + "\n", encoding="utf-8")
    manifest = OUT / "HEX_HOPF_DELAY_SIMULATION_MANIFEST.sha256"
    manifest.write_text("\n".join(f"{sha256(path)} *{path}" for path in [detail, summary_path, trace_path, figure, report, Path(__file__)]) + "\n", encoding="utf-8")
    print(f"Games: {len(rows)}")
    print(f"Summary: {summary_path}")
    print(f"Figure: {figure}")
    print(f"Manifest: {manifest}")


if __name__ == "__main__":
    main()
