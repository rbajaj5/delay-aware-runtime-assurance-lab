# Exploratory Hex Delay Simulation

Board: 7x7; games per cell: 50; delays: [0, 1, 2, 3, 5] moves.
`blind_delay` chooses from delivered local board state. `queue_aware` overlays pending move payloads before choosing. A stale-cell conflict occurs when the selected cell is already occupied on the authoritative board.

| delay | policy | games | mean conflicts | p95 conflicts | any-conflict rate | mean turns | full-board rate | blue win rate | yellow win rate |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | blind_delay | 50 | 0 | 0 | 0 | 49 | 1 | 0.41999999999999998 | 0.57999999999999996 |
| 0 | queue_aware | 50 | 0 | 0 | 0 | 49 | 1 | 0.41999999999999998 | 0.57999999999999996 |
| 1 | blind_delay | 50 | 0 | 0 | 0 | 49 | 1 | 0.57999999999999996 | 0.41999999999999998 |
| 1 | queue_aware | 50 | 0 | 0 | 0 | 49 | 1 | 0.57999999999999996 | 0.41999999999999998 |
| 2 | blind_delay | 50 | 6.3600000000000003 | 9 | 1 | 55.359999999999999 | 1 | 0.56000000000000005 | 0.44 |
| 2 | queue_aware | 50 | 0 | 0 | 0 | 49 | 1 | 0.41999999999999998 | 0.57999999999999996 |
| 3 | blind_delay | 50 | 6.2000000000000002 | 10 | 1 | 55.200000000000003 | 1 | 0.62 | 0.38 |
| 3 | queue_aware | 50 | 0 | 0 | 0 | 49 | 1 | 0.59999999999999998 | 0.40000000000000002 |
| 5 | blind_delay | 50 | 9.8599999999999994 | 13 | 1 | 58.859999999999999 | 1 | 0.41999999999999998 | 0.57999999999999996 |
| 5 | queue_aware | 50 | 0 | 0 | 0 | 49 | 1 | 0.56000000000000005 | 0.44 |

This is a small strategic/message-visibility model. It does not model a physical plant, formal safety constraints, or the v6 controller. The Hex crossing check is a topology diagnostic; the delay sweep is exploratory.
