# Delay-Aware Runtime Assurance Lab

Small, reproducible simulations for studying how delayed messages, finite queues, and queue-aware processing affect multi-agent coordination.

This repository is an exploratory companion to the delay-aware runtime-assurance work. It is intentionally separate from the manuscript bundles: the code here is a transparent sandbox for mechanism discovery, not registered evidence and not a safety case.

## What is here

- `analysis/`: standalone Python simulations and read-only diagnostics.
- `artifacts/20260730/`: CSV outputs, plots, replay artifacts, reports, and SHA-256 manifests from the runs.
- `REPO_MANIFEST.sha256`: hash manifest for the repository's tracked files, generated after assembly.

The most recent experiments are:

- `run_semantic_queue_pollution_simulation.py`: useful coordination traffic competing with low-value background traffic under finite service capacity.
- `run_queue_semantic_decision_simulation.py`: downstream decision staleness caused by acting on delayed neighbor state.
- `run_fair_queue_multiagent_simulation.py`: FIFO, strict priority, and bounded-aging queue disciplines, measuring the trade-off between stale decisions and background-message starvation.
- `run_crossplay_delay_protocol_simulation.py`: an OvercookedV2-inspired asymmetric-information signaling game with cross-play convention mismatch and delayed, sequence-aware feedback.

The bounded-aging run is the current multi-agent extension. At delay 3 and background load 4 messages per agent per round, the exploratory mean stale-decision rates were approximately 0.496 for FIFO, 0.217 for strict priority, and 0.362 for priority with aging. The aging policy served roughly 2,125 background messages per game, while strict priority served roughly 46. These values are mechanism diagnostics, not registered claims.
- `run_delay_random_matrix_loop_diagnostic.py`: a finite random-matrix diagnostic for delayed receipt-age structure and loop residuals.
- `run_delay_ensemble_transfer.py`: transfer of the same queue policies across ring, random-regular, and two-block topologies.
- `run_delay_branch_sector_diagnostic.py`: branch/sector separation diagnostics for a complex phase surrogate.
- `run_hex_multiagent_delay_cascade.py`: delayed coordination in a small Hex-like multi-agent board.

## Reproduce the newest runs

From the repository root:

```powershell
python analysis/run_semantic_queue_pollution_simulation.py
python analysis/run_queue_semantic_decision_simulation.py
```

The scripts write outputs to the sibling workspace layout used during development. To run them entirely inside a clone, set the output location in the scripts to a repository-local artifact directory before execution.

The simulations use Python, NumPy, pandas, and Matplotlib where needed. No simulator harness, manuscript, or registered result is modified by these scripts. In a clone, outputs are written under `analysis/out/`; the checked-in run products are retained under `artifacts/20260730/`.

## Interpretation boundary

The queue policies are deliberately simple abstractions. `priority_deduplicating` is not automatically safe: it can starve background work, requires an explicit message taxonomy, and needs boundedness, authentication, auditability, and failure-handling rules before deployment. The random-matrix and branch diagnostics are finite empirical probes; they do not establish Wigner, Sine-beta, Airy-beta, or universality claims.

Every result should be read together with its report and source manifest. Numbers in this repository are exploratory unless a separate artifact explicitly says otherwise.

## License

Code is released under the MIT License. Generated artifacts are provided for reproducibility and inspection.
