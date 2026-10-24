# Final Course Study

**Prediction-Horizon Sensitivity in Delay-Aware Runtime Assurance** by Ravi Bajaj, August 2026. The final IEEE revision and recording-ready presentation are in `01_course/`, with searchable text and notes. These are retained artifacts, not newly rewritten results.

The system is a PPO-controlled 2D quadrotor in safe-control-gym with a FIFO actuation-delay channel and finite candidate library. The study compares queue-aware and delay-blind assurance and investigates non-monotone horizon sensitivity. The registered fresh comparison records 175/1000 OOB exits at N=3 and 0/1000 at N=4. N=5 is 0/1000 in an exploratory extension; N=6,7,8,9 give 454,479,821,981/1000; N=10 is 200/200 in a pilot. Zero observed exits do not prove safety. N=4 is a configuration-scoped repair, not a universal safe-horizon rule.

## Evidence Map

- `01_course/`: final paper, presentation, text extracts.
- `02_horizon_evidence/`: frozen protocols, summaries, gate results, claims ledger, mechanism checks. Registered, pilot, and exploratory classes remain distinct.
- `03_course_code/`: retained experiment sources and support modules. The main harness and AP source match their registered runner's pinned hashes.
- `04_separate_cbf_followup/`: separately scoped queue-anchored CBF-QP follow-up, protocol, statistical erratum, and descriptive all-seed mechanism census. Do not conflate this with AP-N3's finite candidate-projection artifact.

The v2 action-conversion inconsistency is quarantined in the final paper. Its affected comparisons are not pooled into the final result. This is unrelated to the later bikeshare all-open routing sentinel bug.

## Reproduction Boundary

This directory is an evidence/source snapshot, not a self-contained executable archive. Historical sources retain absolute paths and refer to trained checkpoints, simulator checkouts, environments, and large step logs not included here. Do not execute a historical runner blindly: inspect its inputs, output paths, hash gates, and prerequisites first. Preserving those sources byte-for-byte avoids silently breaking their provenance contracts. Existing source manifests may refer to omitted files.

The work used AI-assisted development and drafting. It is simulation research, not hardware flight validation, a deployment safety certificate, or an accepted-conference claim. Third-party frameworks and cited publications retain their own licenses; no third-party paper corpus is redistributed here.
