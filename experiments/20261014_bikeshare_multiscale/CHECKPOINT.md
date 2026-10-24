# Checkpoint: River Links Planning Laboratory

Updated 2026-10-14. **Resume here; do not restart preparation or replace the experiment.**

## Preserved inputs and correction

- Original 992 episodes, April 1-7 2025 Capital Bikeshare demand, 31 saved tapes, both OSM extracts, detailed/200/600/1400 m models, routing audits and smoke outputs remain unchanged.
- A deterministic replay reproduced all non-wall-clock summary fields and all 1,102,619 retained events. Original request detail now covers 445,824 outcomes, not only two example seeds.
- The original all-open arm had a sentinel bug: ordinary edges with `crossing=None` matched the absent closure and waited until reopening. `corrected_model.py` fixes this separately. `output/corrected/` contains 248 recomputed all-open episodes plus 744 unchanged closure episodes. **Old baseline-dependent service/ranking conclusions are superseded**, not evidence that closures improve service.
- Previous checkpoint archived at `output/CHECKPOINT_pre_planning_extension.md`; old application preserved at `app/index_legacy.html`.

## Completed extension

- Six fixed hypothetical candidates: two riverbank links plus an approach connection per city. Endpoints, lengths, access, directions and speeds: `output/infrastructure/interventions.json`. No candidate replaced because of a zero/negative benefit.
- Smoke: 48 paired-design episodes checked before the full 744 investment episodes. Every candidate/resolution graph rebuilt; no stale route cache. All 552 station OD pairs reachable per candidate/resolution. No combined investments simulated or permitted.
- 992 corrected study views + 744 investment views + 992 retained audit views = 2,728 catalog entries referencing 1,984 complete physical replay files.
- Journey audit passes: 2,728 CSV summary rows reconciled to event logs; 1,015,744 snapshot checks; 905,301 timed movements; 2,251,898 event checks; original source hashes unchanged. At 597 rounded-time boundaries a snapshot is checked against an event-order prefix. UI inventories are right-continuous ledger values.
- Local app: `app/index.html`, `app/app.js`, `app/domain.js`, `app/style.css`; `serve.py` on loopback port 8766. Linked maps, station inventories, routes, replay/scrubbing, seed/day/policy/scale controls, investment and recovery rankings, reduced/detailed comparison, visible coverage and caveats.
- Browser checks: 35 saved-app checks and six scenario-runner checks; desktop 1920/1600 and mobile 390 px. Unit tests: 14 pass. Validation JSON and screenshots are in `output/planning/`.
- New-scenario runner: `scenario_runner.py`; supports one existing crossing closure, one fixed hypothetical candidate, or all-open. Closure start/duration must fit horizon; all four scales; saved workloads/seeds; both policies; starting inventory 4-12 per station. Both arms use matched inputs. Capacity stays 16; crew target stays 8, with unchanged dispatch thresholds. Full configs/tapes/results are in `output/custom_runs/`.
- Pending settings hide previous trajectories. Exact saved matches bypass computation. Custom results explicitly describe one seed; they do not inherit aggregate rankings. Jobs are serialized, with a three-job pending limit.
- CSV tables, data-quality inventory, research figures and independent audits: `output/planning/`. Reading/teaching guide: `README_PLANNING.md`.

## Findings to preserve

Washington controlled/no redistribution: I1 mean benefit +0.583 trips detailed, -0.583 at 200 m; highest-mean candidate changes I1 -> I3. Mean route error at 200 m remains only 1.358%. At 1400 m Washington mean route error is 69.217%, with much larger worst-OD distortion. Kaliningrad/cargo: I1 benefit +1.917 detailed versus +0.167 at 200 m; I3 has the higher reduced-model mean. These small and variable effects are descriptive sensitivity results, not reliable construction recommendations. Kaliningrad I2 has zero selected OD usage and zero measured benefit; retain it.

Investment benefits compare against corrected existing-open infrastructure. Recovery benefits compare reopening against a timed disruption. Never merge their baselines. Travel-time deltas use identical request IDs completed in both arms. Whiskers are empirical p10-p90, not confidence intervals.

## Movie and final checks

The movie is recorded from the **actual verified dashboard**, not the earlier separate Pillow renderer. `output/movie/walkthrough.json` freezes all scene settings and timing, including request 17 using Washington I1 and a custom B04 closure (70-minute start, 40-minute duration, 7 bikes per station). `walkthrough_resolved.json` records exact episodes/results; `dashboard_frame_index.jsonl` records each captured state. `app/demo.css` is only a compact demonstration layout.

Completed deliverable: `output/movie/river_links_planning_narrated.mp4`, 180 s, 1920x1080, 24 fps; 12 deterministic UI captures/s, each held for two frames. Local Zira narration, burned captions and subtitle track. NVENC uses the GPU for encoding; event simulation/routing are CPU-based. **`output/movie/movie_validation.json` reports `passed: true`: all 4,320 encoded frames decode successfully, and sampled encoded images match the dashboard captures.** The 2,160 captured states and demonstrated scenario configurations are exported alongside the movie. No simulation or recording work remains pending for this refinement.

## Remaining scope limits

No multi-intervention combinations, demand-response/intensity control, user-edited candidate positions or connection speeds. No construction costs/engineering feasibility, traffic congestion, grade, field-verified station access, or historical road reconstruction. Finite buffers and unresolved via-way restrictions remain; the half-width Kaliningrad buffer loses 147 station OD routes. April trips use October 2026 maps. Observed days are completed rentals, not denied demand or monthly observations. Kaliningrad and cross-city workloads are hypothetical. Topology, routing resolution and Hausdorff dimension are not interchangeable.

## Exact commands (PowerShell)

```powershell
Set-Location 'C:\Users\anaxe\Documents\Codex\2026-05-05\master-codex-prompt-agentic-rag-with\delay-aware-runtime-assurance-lab\experiments\20261014_bikeshare_multiscale'
$py = 'C:\Users\anaxe\AppData\Local\Programs\Python\Python311\python.exe'
& $py serve.py --port 8766
```

Open `http://127.0.0.1:8766/`. Keep that terminal running. In a second terminal in the same directory:

```powershell
$py = 'C:\Users\anaxe\AppData\Local\Programs\Python\Python311\python.exe'
& $py -m unittest -v test_model.py test_replay.py test_runner.py
node qa_app.cjs
node qa_runner.cjs
& $py audit_replays.py
# Only if the movie needs regeneration:
& $py prepare_walkthrough.py
powershell -NoProfile -ExecutionPolicy Bypass -File '..\20261014_bridge_fleet\narrate.ps1' -OutputDirectory '.\output\movie'
& $py -c 'from make_movie import align_audio; align_audio()'
node record_dashboard.cjs
& $py finalize_dashboard.py
```

If derived planning tables are missing: `& $py planning_analysis.py`, then `& $py audit_replays.py`. If complete journey exports are missing: `& $py replay_export.py` (hash-guarded resume), then `planning_analysis.py`. If investment runs are missing: `& $py infrastructure.py` (hash-guarded reuse). Do not invoke `prepare.py` or the overwriting `simulate.py` run entrypoint. Do not use the unrelated September ZIP. Do not access Canvas, publish, submit or push this work.
