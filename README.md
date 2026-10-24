# Delay-Aware Runtime Assurance Lab

## Public Research Collection

The October 24, 2026 source release adds the [final course study](course_study/README.md), later formation/risk-allocation demonstrations, and the [multiscale infrastructure-planning laboratory](experiments/20261014_bikeshare_multiscale/README_PLANNING.md). [Release scope and exclusions](docs/PUBLIC_RELEASE_20261024.md) distinguish registered evidence from exploratory extensions and identify large files retained locally. The current working tree includes source and compact evidence, not every raw trace or map cache.

The class literature-retrieval system is a separate repository: [Assured Autonomy Agentic RAG](https://github.com/rbajaj5/assured-autonomy-rag). The RAG is not a substitute for simulation or formal assurance evidence.

## Original Exploratory Sandbox

Small, reproducible simulations for studying how delayed messages, finite queues, and queue-aware processing affect multi-agent coordination.

The original repository was an exploratory companion to the delay-aware runtime-assurance work. Its sandbox experiments remain exploratory, not registered evidence or a safety case. The separately labeled `course_study/` directory now carries the final manuscript and selected registered evidence; those labels do not transfer to other folders.

## What is here

- `analysis/`: standalone Python simulations and read-only diagnostics.
- `paper/queue-aware-coordination-draft.md`: the current self-contained manuscript draft for the queue benchmark.
- `paper/queue-aware-coordination-draft.docx`: formatted DOCX version of the manuscript draft.
- `paper/build_queue_paper_docx.py`: reproducible DOCX builder.
- `artifacts/20260730/`: CSV outputs, plots, replay artifacts, reports, and SHA-256 manifests from the runs.
- `REPO_MANIFEST.sha256`: hash manifest for the repository's tracked files, generated after assembly.

The most recent experiments are:

- `run_semantic_queue_pollution_simulation.py`: useful coordination traffic competing with low-value background traffic under finite service capacity.
- `run_queue_semantic_decision_simulation.py`: downstream decision staleness caused by acting on delayed neighbor state.
- `run_fair_queue_multiagent_simulation.py`: FIFO, strict priority, and bounded-aging queue disciplines, measuring the trade-off between stale decisions and background-message starvation.
- `run_crossplay_delay_protocol_simulation.py`: an OvercookedV2-inspired asymmetric-information signaling game with cross-play convention mismatch and delayed, sequence-aware feedback.
- `run_delay_concentration_diagnostic.py`: compares global stale-event mass with spatial and temporal concentration under equal-mean spread versus clustered delay schedules.
- `run_jl_projection_safety_diagnostic.py`: tests whether random-projection geometry preservation also preserves a coordinatewise safety predicate; in this toy setting it does not.
- `run_helicopter_3d_landing_simulation.py`: renders an exploratory 3D helicopter landing-coordination replay under delayed clearances, comparing delay-blind and queue-aware freshness handling.
- `run_3d_formation_delay_monitor_simulation.py`: runs a CUDA-capable 3D formation-monitor demonstration with delayed messages, a bounded perturbation, and a separation fallback.
- `analyze_helicopter_network.py`: derives temporal conflict-graph, component, motif, and spectral diagnostics from the retained helicopter trace without rerunning the simulation.
- `derive_prime_graph_signature.py`: derives an optional prime-labeled arithmetic fingerprint from the retained conflict graph without rerunning the simulation.

The bounded-aging run is the current multi-agent extension. At delay 3 and background load 4 messages per agent per round, the exploratory mean stale-decision rates were approximately 0.496 for FIFO, 0.217 for strict priority, and 0.362 for priority with aging. The aging policy served roughly 2,125 background messages per game, while strict priority served roughly 46. These values are mechanism diagnostics, not registered claims.

The literature notes in `docs/` include scoped connections to zero-shot coordination, oracle-consistency evaluation, and physics-informed kernel surrogates. They are conceptual guides, not additional evidence for the simulation results.
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

Every result should be read together with its report and source manifest. Numbers in this repository are exploratory unless a separate artifact explicitly says otherwise. The original `REPO_MANIFEST.sha256` is a historical assembly manifest, not a hash list for the later additions; see `docs/PUBLIC_RELEASE_MANIFEST.json` for this source release.

## License

Code is released under the MIT License. Generated artifacts are provided for reproducibility and inspection.
