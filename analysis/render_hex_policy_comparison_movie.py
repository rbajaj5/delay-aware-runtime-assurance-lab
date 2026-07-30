"""Render a trace-driven Hex policy comparison movie.

The movie reads only the retained trace emitted by
run_delay_aware_hex_simulation.py. It is an exploratory teaching artifact,
not v6 evidence and not a model of a physical plant.
"""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
from collections import deque
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis" / "out" / "hex_delay_exploratory_20260730"
TRACE = OUT / "hex_delay_trace_delay3_seed0.json"
N = 7
DELAY = 3
W, H, FPS = 1920, 1080, 24
BLUE = (35, 105, 190)
YELLOW = (234, 167, 35)
BLACK = (35, 35, 35)
GRAY = (105, 110, 116)
WHITE = (250, 251, 252)
GREEN = (0, 150, 90)
RED = (190, 55, 55)


def font(size: int, bold: bool = False):
    path = r"C:\Windows\Fonts\segoeuib.ttf" if bold else r"C:\Windows\Fonts\segoeui.ttf"
    return ImageFont.truetype(path, size)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def neighbors(r: int, c: int):
    return [(r, c - 1), (r, c + 1), (r - 1, c), (r + 1, c), (r - 1, c + 1), (r + 1, c - 1)]


def crossing(board, color: int):
    starts = [(r, 0) for r in range(N)] if color == 1 else [(0, c) for c in range(N)]
    target = (lambda r, c: c == N - 1) if color == 1 else (lambda r, c: r == N - 1)
    queue = deque(cell for cell in starts if board[cell[0]][cell[1]] == color)
    parent = {cell: None for cell in queue}
    goal = None
    while queue:
        r, c = queue.popleft()
        if target(r, c):
            goal = (r, c)
            break
        for nr, nc in neighbors(r, c):
            if 0 <= nr < N and 0 <= nc < N and board[nr][nc] == color and (nr, nc) not in parent:
                parent[(nr, nc)] = (r, c)
                queue.append((nr, nc))
    if goal is None:
        return []
    path = []
    while goal is not None:
        path.append(goal)
        goal = parent[goal]
    return list(reversed(path))


def cell_center(r: int, c: int, origin_x: int, origin_y: int, size: int = 33):
    return origin_x + c * size * 1.48 + r * size * 0.74, origin_y + r * size * 1.28


def draw_board(draw, board, path_cells, origin_x, origin_y, title, subtitle, conflict_cell=None, size=33):
    draw.text((origin_x, origin_y - 70), title, fill=BLACK, font=font(25, True))
    draw.text((origin_x, origin_y - 40), subtitle, fill=GRAY, font=font(15))
    radius = 21
    for r in range(N):
        for c in range(N):
            cx, cy = cell_center(r, c, origin_x, origin_y, size)
            points = []
            for angle in range(0, 360, 60):
                rad = math.radians(angle)
                points.append((cx + radius * math.cos(rad), cy + radius * math.sin(rad)))
            value = board[r][c]
            fill = BLUE if value == 1 else YELLOW if value == 2 else WHITE
            outline = (20, 65, 130) if value == 1 else (160, 110, 15) if value == 2 else (155, 162, 170)
            width = 4 if conflict_cell == (r, c) else 1
            draw.polygon(points, fill=fill, outline=RED if width > 1 else outline, width=width)
            if (r, c) in path_cells:
                draw.ellipse((cx - 7, cy - 7, cx + 7, cy + 7), fill=GREEN, outline="white", width=2)


def card(title, lines):
    image = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(image)
    draw.text((W // 2, 230), title, anchor="ma", fill=BLACK, font=font(48, True))
    draw.multiline_text((W // 2, 400), "\n".join(lines), anchor="ma", align="center", fill=GRAY, font=font(25), spacing=18)
    return image


def frame(traces, step: int) -> Image.Image:
    image = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(image)
    draw.text((W // 2, 22), "HEX: DELAY-BLIND VS QUEUE-AWARE PLAY", anchor="ma", fill=BLACK, font=font(34, True))
    draw.text((W // 2, 66), f"7x7 board | fixed message delay = {DELAY} moves | trace seed = 2026073000", anchor="ma", fill=GRAY, font=font(18))
    for index, (policy, x) in enumerate((("blind_delay", 70), ("queue_aware", 970))):
        trace = traces[policy]
        item = trace[min(step, len(trace) - 1)]
        true_board = item["true_board"]
        local_board = item["local_board"]
        true_path = crossing(true_board, 1) or crossing(true_board, 2)
        conflict = (item["r"], item["c"]) if item["stale_conflict"] and item["r"] is not None else None
        color = RED if item["stale_conflict"] else GREEN
        short_policy = "blind" if policy == "blind_delay" else "queue"
        draw_board(draw, true_board, true_path, x, 205, f"{short_policy}: true board", "The authoritative board that exists", conflict_cell=conflict)
        draw_board(draw, local_board, [], x + 380, 205, "Local view", "Delivered board; pending count shown below", conflict_cell=None)
        draw.rounded_rectangle((x, 760, x + 780, 890), radius=12, fill=(247, 248, 249), outline=(220, 222, 225), width=2)
        status = "STALE CELL CONFLICT" if item["stale_conflict"] else "move applied"
        draw.text((x + 22, 780), status, fill=color, font=font(22, True))
        draw.text((x + 22, 820), f"turn {item['turn']} | player {'BLUE' if item['player'] == 1 else 'YELLOW'} | pending messages {item['pending_count']}", fill=BLACK, font=font(17))
        draw.text((x + 22, 850), f"selected cell: {item['r']},{item['c']} | applied: {item['applied']}", fill=GRAY, font=font(17))
    draw.text((30, H - 31), "Trace-derived exploratory Hex simulation; stale conflicts are message-visibility diagnostics, not v6 quadrotor evidence.", fill=GRAY, font=font(14))
    return image


def main() -> None:
    if not TRACE.exists():
        raise FileNotFoundError(TRACE)
    traces = json.loads(TRACE.read_text(encoding="utf-8"))
    assert set(traces) == {"blind_delay", "queue_aware"}
    max_steps = max(len(trace) for trace in traces.values())
    output = OUT / "hex_policy_comparison_delay3_seed0.mp4"
    poster = OUT / "hex_policy_comparison_delay3_seed0_poster.png"
    report = OUT / "hex_policy_comparison_delay3_seed0_report.md"
    manifest = OUT / "HEX_POLICY_COMPARISON_MANIFEST.sha256"
    ffmpeg = ["ffmpeg", "-y", "-f", "rawvideo", "-vcodec", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-an", "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", str(output)]
    process = subprocess.Popen(ffmpeg, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert process.stdin is not None
    try:
        title = card("HEX WITH A DELAYED BOARD", ["Same board size, same delay, two information policies", "blind_delay acts from delivered state; queue_aware overlays pending move messages", "Exploratory teaching artifact"])
        for _ in range(FPS * 2):
            process.stdin.write(title.tobytes())
        for step in range(max_steps):
            image = frame(traces, step)
            for _ in range(4):
                process.stdin.write(image.tobytes())
        blind_conflicts = sum(1 for item in traces["blind_delay"] if item["stale_conflict"])
        aware_conflicts = sum(1 for item in traces["queue_aware"] if item["stale_conflict"])
        end = card("THE INFORMATION POLICY CHANGES THE PLAY", [f"Delay-3 trace: blind conflicts = {blind_conflicts}; queue-aware conflicts = {aware_conflicts}", "The authoritative board is shown separately from the acting player's delivered view.", "Exploratory only; no physical or v6 safety claim."])
        for _ in range(FPS * 2):
            process.stdin.write(end.tobytes())
    finally:
        process.stdin.close()
    stderr = process.stderr.read().decode("utf-8", errors="replace") if process.stderr else ""
    if process.wait() != 0:
        raise RuntimeError(stderr[-4000:])
    poster_frame = frame(traces, max_steps - 1)
    poster_frame.save(poster)
    report.write_text(
        "# Trace-Driven Hex Policy Comparison\n\n"
        f"Source: `{TRACE}`; source SHA-256: `{sha256(TRACE)}`. The movie renders {max_steps} trace steps from the delay-3 seed-0 records. It displays the authoritative board, the acting player's delivered local board, pending-message count, and the logged stale-conflict flag. No state is simulated during rendering.\n\n"
        "The movie is an exploratory combinatorial teaching artifact. It does not model a physical plant, formal safety constraints, or the v6 controller.\n",
        encoding="utf-8",
    )
    manifest.write_text("\n".join(f"{sha256(path)} *{path}" for path in [TRACE, output, poster, report, Path(__file__)]) + "\n", encoding="utf-8")
    print(f"MP4: {output}")
    print(f"Poster: {poster}")
    print(f"Manifest: {manifest}")


if __name__ == "__main__":
    main()
