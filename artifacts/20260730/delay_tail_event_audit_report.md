# Finite-Run Tail Event Audit

Input: `C:\Users\anaxe\Documents\Codex\2026-05-05\master-codex-prompt-agentic-rag-with\analysis\out\hex_delay_exploratory_20260730\delay_random_matrix_loop_games.csv`

The supplied excerpt separates a regular event from its exceptional complement and shows that the complement can be controlled before taking expectations. This audit follows that bookkeeping pattern only: it defines empirical tail events from the retained finite-run data. It does not infer an exponential probability bound or claim the source theorem's hypotheses.

## Data-defined thresholds

- `mean_receipt_age`: 3.9444444444444446
- `mean_loop_residual`: 0.23529411764705885
- `max_disagreement`: 0.81358736086236316

Tail membership is `value >= threshold`.

| delay | policy | games | age tail rate | loop tail rate | disagreement tail rate |
|---:|---|---:|---:|---:|---:|
| 0 | blind_delay | 30 | 0 | 1 | 0.033333333333333333 |
| 0 | queue_aware | 30 | 0 | 1 | 0.033333333333333333 |
| 1 | blind_delay | 30 | 0 | 1 | 0.13333333333333333 |
| 1 | queue_aware | 30 | 0 | 1 | 0.13333333333333333 |
| 2 | blind_delay | 30 | 0 | 0 | 0 |
| 2 | queue_aware | 30 | 0 | 1 | 0.13333333333333333 |
| 3 | blind_delay | 30 | 0 | 0 | 0 |
| 3 | queue_aware | 30 | 0 | 1 | 0.066666666666666666 |
| 5 | blind_delay | 30 | 1 | 0 | 0 |
| 5 | queue_aware | 30 | 0 | 1 | 0 |

The tail rates are descriptive diagnostics for separating ordinary and exceptional episodes. They are not hypothesis tests and should not be presented as asymptotic rare-event probabilities.
