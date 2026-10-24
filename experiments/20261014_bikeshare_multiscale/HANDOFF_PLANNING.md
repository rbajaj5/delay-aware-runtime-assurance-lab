# Handoff

The operative checkpoint is `CHECKPOINT.md`. The application is a local infrastructure-planning laboratory, not a cloud dashboard. Start `serve.py --port 8766` with the configured Python 3.11 and open http://127.0.0.1:8766/.

## Deliverables

- Dashboard: `app/`, with separate saved-experiment and new-scenario modes.
- Narrated dashboard demonstration: `output/movie/river_links_planning_narrated.mp4`.
- Reproducible walkthrough: `output/movie/walkthrough.json` and `walkthrough_resolved.json`.
- Full retained replay coverage/audit: `output/journeys/manifest.json` and `output/planning/journey_validation.json`.
- Corrected baseline: `output/corrected/`; original `output/` logs/results preserved.
- Fixed hypothetical definitions and 744 investment episodes: `output/infrastructure/`.
- Exported tables/figures/quality checks: `output/planning/`.
- Custom scenario configurations and paired results: `output/custom_runs/`.
- Teaching guide, field interpretation and caveats: `README_PLANNING.md`.

## Validation gates

`output/planning/browser_validation.json`, `runner_browser_validation.json`, `journey_validation.json`, and `output/movie/movie_validation.json` must say `passed: true`. The movie audit confirms a full decode and pixel comparisons to actual dashboard screenshots. No claim of completion should be inferred only from a silent intermediate movie.

Original assumptions and limitations remain visible. The important finding is resolution-dependent planning sensitivity, not proof that a hypothetical bridge should be built. In particular, preserve zero and negative candidate effects and seed variability. Do not reuse the old all-open baseline for planning claims.

## Restart behavior

Saved-run exploration uses only local files. The runner has a single calculation worker and a bounded queue. Completed custom configurations are content-addressed and reused. An interrupted in-memory job can be resubmitted with the same settings after server restart; original experiment outputs are never edited. The result JSON records source hashes, exact closure timing, initial inventory, source demand hash and both arms.

No Canvas access, cloud publication, submission, or repository push was performed.
