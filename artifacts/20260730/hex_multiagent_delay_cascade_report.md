# Multi-Agent Delayed-State Cascade

Four agents share a 7x7 Hex board, two per color. Every applied move is broadcast to the other three agents with the listed delay. `blind_fleet` acts from delivered local state; `queue_aware_fleet` overlays pending messages before acting.

A red-line event is a toy threshold (`opponent shortest-path cost <= 2`) evaluated on the acting agent's local board. A false red-line occurs when the local view triggers but the authoritative board does not; a missed red-line is the reverse. Consecutive local triggers by the same agent are counted as cascade turns.

This is a mechanism-level analogue of delayed multi-agent synchronization and rigid automated response. It is not a forecast, military model, or evidence for the source paper's scenarios.

| delay | policy | games | mean conflicts | false red-lines | missed red-lines | cascade turns | full board rate |
|---:|---|---:|---:|---:|---:|---:|---:|
| 0 | blind_fleet | 20 | 0 | 0 | 0 | 16.699999999999999 | 1 |
| 0 | queue_aware_fleet | 20 | 0 | 0 | 0 | 16.699999999999999 | 1 |
| 1 | blind_fleet | 20 | 0 | 0 | 0 | 16.699999999999999 | 1 |
| 1 | queue_aware_fleet | 20 | 0 | 0 | 0 | 16.699999999999999 | 1 |
| 2 | blind_fleet | 20 | 9.5999999999999996 | 1.05 | 1.6000000000000001 | 18.600000000000001 | 1 |
| 2 | queue_aware_fleet | 20 | 0 | 0 | 0 | 16.449999999999999 | 1 |
| 3 | blind_fleet | 20 | 15.449999999999999 | 0.5 | 3.1000000000000001 | 19.600000000000001 | 1 |
| 3 | queue_aware_fleet | 20 | 0 | 0 | 0 | 17.100000000000001 | 1 |
| 5 | blind_fleet | 20 | 16.5 | 0.10000000000000001 | 2.3500000000000001 | 20.949999999999999 | 1 |
| 5 | queue_aware_fleet | 20 | 0 | 0 | 0 | 16.800000000000001 | 1 |
