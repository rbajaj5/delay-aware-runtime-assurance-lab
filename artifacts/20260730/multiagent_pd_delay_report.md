# Multi-Agent Prisoner's Dilemma with Delayed Receipts

This experiment is inspired by the Veritasium presentation and its references to Axelrod's iterated Prisoner's Dilemma and cooperation under noise. Source: `https://www.veritasium.com/videos/2024/1/15/what-the-prisoners-dilemma-reveals-about-life-the-universe-and-everything`.

A ring of 12 agents plays 200 repeated edge games. `blind_tft` copies the latest received action regardless of receipt age. `queue_aware_forgiving` cooperates when the latest receipt is older than one round.

| delay | policy | games | cooperation rate | payoff/action | stale retaliations | forgiving actions | defection bursts |
|---:|---|---:|---:|---:|---:|---:|---:|
| 0 | blind_tft | 40 | 1 | 3 | 0 | 0 | 0 |
| 0 | queue_aware_forgiving | 40 | 1 | 3 | 0 | 24 | 0 |
| 1 | blind_tft | 40 | 1 | 3 | 0 | 0 | 0 |
| 1 | queue_aware_forgiving | 40 | 1 | 3 | 0 | 24 | 0 |
| 2 | blind_tft | 40 | 1 | 3 | 0 | 0 | 0 |
| 2 | queue_aware_forgiving | 40 | 1 | 3 | 0 | 48 | 0 |
| 3 | blind_tft | 40 | 1 | 3 | 0 | 0 | 0 |
| 3 | queue_aware_forgiving | 40 | 1 | 3 | 0 | 72 | 0 |
| 5 | blind_tft | 40 | 1 | 3 | 0 | 0 | 0 |
| 5 | queue_aware_forgiving | 40 | 1 | 3 | 0 | 120 | 0 |

This is a small repeated-game communication model, not a reproduction of the video and not a claim about biological cooperation or real-world fleets.
