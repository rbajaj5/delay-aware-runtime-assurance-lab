# Factual handoff: JHU Data Visualization

Implemented a separate experiment at `C:\Users\anaxe\Documents\Codex\2026-05-05\master-codex-prompt-agentic-rag-with\delay-aware-runtime-assurance-lab\experiments\20261014_bridge_fleet`. It uses the supplied historical
seven-bridge schematic, not real Kaliningrad streets, and compares arrival-order
FIFO dispatch, priority/coalesced updates, and priority plus freshness monitoring.
It reuses queue/delay and rendering patterns from the existing assured-autonomy
repository. No existing experiment was replaced; nothing was published or submitted.

Ran 100 matched seeds per scenario, three scenarios and three policies (900
episodes). All inputs are synthetic. Controllers receive delayed messages, never
the hidden closure schedule. Actual closures block every policy, including
in-progress motion under an explicit schematic stop rule. Representative replay
seed: 2026101400, selected by the saved median-distance rule.

## Actual findings

| Scenario | Policy | Completion | Mean pickup wait (s) | Uncompleted | Blocked entries | RTA fleet-min |
|---|---|---:|---:|---:|---:|---:|
| all_open | Latest received | 45.9% | 185.1 | 25.94 | 0.00 | 0.00 |
| all_open | Priority updates | 96.3% | 89.2 | 2.06 | 0.00 | 0.00 |
| all_open | Freshness monitor | 94.2% | 114.5 | 3.19 | 0.00 | 5.24 |
| detour | Latest received | 43.5% | 191.5 | 27.05 | 0.79 | 0.00 |
| detour | Priority updates | 90.3% | 132.4 | 5.13 | 1.29 | 0.00 |
| detour | Freshness monitor | 86.9% | 152.3 | 6.84 | 0.47 | 5.27 |
| district_isolation | Latest received | 36.0% | 172.2 | 30.53 | 1.21 | 0.00 |
| district_isolation | Priority updates | 74.9% | 196.9 | 12.38 | 1.99 | 0.00 |
| district_isolation | Freshness monitor | 75.2% | 199.2 | 12.25 | 0.74 | 19.81 |

Main comparison in the detour scenario: priority versus monitor completion is
90.3% versus 86.9%;
mean blocked attempts are 1.29 versus 0.47;
mean picked-up passenger waits are 132.4s versus 152.3s.
This is a tradeoff, not universal superiority. Baseline FIFO is deliberately
overloaded by background messages. Do not generalize its penalty to uncongested queues.

Independent checks reconciled 42,354 request records and
61,649 completed link traversals across all 900
episodes. No vehicle progresses on a closed edge. Means and 10th-90th percentiles
come from saved summaries; the renderer does not invent outcomes. Counts, waiting,
empty/occupied travel and selected replay results were independently verified.

## Deliverables

Final video: `C:\Users\anaxe\Documents\Codex\2026-05-05\master-codex-prompt-agentic-rag-with\delay-aware-runtime-assurance-lab\experiments\20261014_bridge_fleet\output\fleet_changing_networks.mp4`
Code and README: `C:\Users\anaxe\Documents\Codex\2026-05-05\master-codex-prompt-agentic-rag-with\delay-aware-runtime-assurance-lab\experiments\20261014_bridge_fleet`
CSV tables, logs, transcript and validation: `C:\Users\anaxe\Documents\Codex\2026-05-05\master-codex-prompt-agentic-rag-with\delay-aware-runtime-assurance-lab\experiments\20261014_bridge_fleet\output`
Paper stills: `C:\Users\anaxe\Documents\Codex\2026-05-05\master-codex-prompt-agentic-rag-with\delay-aware-runtime-assurance-lab\experiments\20261014_bridge_fleet\output\01_euler.png` through `C:\Users\anaxe\Documents\Codex\2026-05-05\master-codex-prompt-agentic-rag-with\delay-aware-runtime-assurance-lab\experiments\20261014_bridge_fleet\output\04_repeated_results.png`

## Remaining incomplete by design

No current motor-vehicle map, empirically calibrated taxi demand, real vehicle
physics, large-network or broad hyperparameter study, or operational flight/driving
validation. No Tableau workbook is built for this new experiment; the tables are
ready for that later step. GPU is used for video encoding, not graph simulation.
No claim of new theory or deployed Waymo performance. Final media QA is recorded
separately in `media_validation.json`.
