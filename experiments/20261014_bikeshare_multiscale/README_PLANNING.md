# River Links: Infrastructure-Planning Laboratory

Central question: **Which hypothetical changes improve service of fixed bikeshare requests, and how far can the network be reduced without changing that planning comparison?**

This extends the existing experiment. It does not replace its April 2025 demand, OSM extracts, network audits, candidate locations, or original 992 episodes. Nothing is published or submitted.

## Open locally

From this directory, using Python 3.11 with NetworkX, NumPy, SciPy, Shapely and pyproj:

```powershell
& 'C:\Users\anaxe\AppData\Local\Programs\Python\Python311\python.exe' serve.py --port 8766
```

Open http://127.0.0.1:8766/. The server is loopback-only. No account, password, cloud visualization service or API credit is needed. Stop with Ctrl+C.

## Two modes

- **Explore a saved experiment:** selects exactly calculated city/workload/seed/policy/scale/intervention combinations. Playback time is not an experiment parameter. Scrubbing reconstructs inventories from the event ledger and interpolates riders along their timed, expanded street routes.
- **Configure and run a new scenario:** calculates a matched open baseline and one selected intervention, or identifies an exact saved match. It supports a single existing-crossing closure, one fixed hypothetical link, or all-open; closure start/duration within the episode; both operating policies; existing demand tapes; all four resolutions; and initial inventory of 4–12 bikes per station. Capacity stays 16. Crew dispatch thresholds remain 3/11 and its target remains 8 bikes. Changing settings hides old trajectories. Jobs are serialized and their complete configuration is saved under `output/custom_runs/`.

No simultaneous interventions, demand-response model, demand-intensity slider, altered travel speeds, or hypothetical-link duration slider is exposed. Individual benefits are never added together. New single-seed runs do not inherit the aggregate rankings or uncertainty of the original experiment.

## Completed evidence

- 992 retained episodes reproduced from 31 saved input tapes, with every non-timing summary field and 1,102,619 events matching. Wall-clock runtime is excluded from equality.
- 445,824 request outcomes recovered for the retained experiment. Coverage is complete, not the previous two-seed movie subset.
- 248 corrected open-network episodes plus 744 unchanged closure episodes form the corrected study. Original data remain untouched.
- 744 investment episodes: three fixed candidates per city, four routing resolutions, both operating policies, 12 controlled seeds per city and seven Washington observed days.
- Every intervention has two explicit directed edges and documented endpoints, geometric length rule, cycling access and speeds in `output/infrastructure/interventions.json`. All 24 intervention/resolution models were rebuilt, with new route caches. All 552 selected station OD pairs remain reachable. Candidate I2 in Kaliningrad is unused by the selected shortest station routes and is retained as a zero-benefit case.
- 2,728 catalog views reference 1,984 unique complete replays. The journey audit checks every compressed movement, snapshot/event consistency and unchanged original source hashes. See `output/planning/journey_validation.json`.

## Important correction

The original router compared `edge.crossing == crossing` even when the requested closure was `None`. Ordinary edges also have `crossing=None`, so the all-open arm could wait until a nonexistent reopening. `corrected_model.py` isolates the fix without editing `routing.py` or retained results. Closure arms are unchanged. The old all-open service comparisons and rankings are **superseded**; original replay is available only as an explicitly labeled audit view.

## How to read the measures

- Additional completed trips = intervention completions minus baseline completions for the same input tape. These are not newly generated riders.
- Served-fraction change is in percentage points, with every modeled request in the denominator.
- Unserved = failed pickup plus unreachable route. Unfinished = bicycle still riding or waiting for a dock at the horizon. Neither is silently removed.
- Travel-time change uses identical request IDs completed in both arms. It excludes failed/unfinished trips and is not the mean time over all requests. Aggregate tables average the episode-level differences, not a pooled rider sample.
- Empty-station time is the sum of empty durations across stations, in station-hours. Full-return events can include repeated attempts by the same rider.
- Cargo effort includes physically routed crew travel and handling time. Bikes cannot be moved instantaneously by the dashboard.
- Ranking whiskers are empirical p10–p90 across 12 seeds or seven observed days, not confidence intervals. Ranking means use ties, with 1e-10 numerical tolerance. A descriptive highest mean is not a statistically established best construction option.
- Routing resolution is a model-reduction parameter in metres, not a Hausdorff dimension. Station-group visualization does not pool bicycle inventories.

## Results and limits

Washington controlled demand without redistribution: I1 averages +0.583 additional completions with detailed routing and -0.583 at 200 m. The descriptive top candidate changes from I1 to I3. These effects are small and variable. Even 200 m mean route-length error of 1.36% does not certify a delicate service ranking. At 1400 m Washington mean route error is 69.22%, with far worse individual OD paths.

Kaliningrad with cargo redistribution: I1 averages +1.917 completions with detailed routing versus +0.167 at 200 m; I3 then has the highest mean. This does not establish superiority of I3 outside the finite experiment.

Recovery comparisons reopen an existing crossing relative to its closure. Investment comparisons add one link relative to the corrected existing open network. Those baselines must not be conflated.

Finite map boundaries, unresolved via-way restrictions, map-screened connectors, fixed speeds, synthetic station capacities/inventory, one crude cargo policy, absent traffic/grade/cost/engineering constraints, and missing denied demand limit external validity. April 2025 trips are routed on October 2026 maps. The controlled cross-city workload is hypothetical. The observed workload is one week, not monthly data. These are sensitivity experiments, not construction recommendations or a demand forecast.

## Teaching walkthrough

1. Start with Washington, controlled demand, no redistribution, detailed routing, all-open. Ask what it takes to complete a trip besides finding a path: bicycles at origins and free docks at destinations.
2. Enable I1. Select “Changed route / outcome” in journeys, then request 17. Compare its blue baseline and purple intervention route and the linked origin inventory.
3. Switch to “Detailed vs reduced” at 200 m. Ask why the mean I1 effect and highest-mean candidate change despite a small average routing error. Then use 1400 m to reveal larger service distortion.
4. Try Kaliningrad / cargo crew / I1 / detailed then 200 m. Separate changes in travel from changes in inventory timing.
5. Try Washington / observed April week / April 4 / I1. Discuss why seven recorded-rental days with tied effects do not establish a broadly robust ranking.

The three answerable questions are: which fixed candidates improve modeled completions (sorted matched-effect bars), how resolution changes their effects/rankings (scale plot and tied-rank table), and where timing/inventory differences arise (linked maps, event counters and station timelines). Association in observed rentals is not evidence that building a bridge would create real demand.

## Data and movie

CSV exports in `output/planning/` are suitable for Tableau/Power BI: `episode_results.csv`, `matched_benefits.csv`, `ranked_benefits.csv`, `benefit_scale_errors.csv`, and `ranking_agreement.csv`. `data_quality.json` inventories fields, numeric/string types, row counts and missingness. Full request tables are gzip CSVs under `output/journeys/` and `output/planning/`.

There are 1,736 corrected/investment episode rows, 1,488 matched benefit rows, 144 aggregated benefit rows and 1,488 scale-error rows. The 1,344 blank `day` values in the episode table belong to controlled workloads and are expected, not missing observed dates. Treat seed, scale and scenario as grouping dimensions; do not sum duplicate views or average already-aggregated means across incompatible workloads.

`output/movie/river_links_planning_narrated.mp4` is a three-minute recording of the actual dashboard in its compact demonstration layout. It has local automated narration, burned captions and a selectable subtitle track. The 24 fps file uses 12 deterministic dashboard captures per second, each shown for two frames. GPU use is NVENC video encoding, not a claim of GPU-based event simulation.

`output/movie/walkthrough.json` freezes control settings and simulation-time intervals. `walkthrough_resolved.json` binds them to exact episode IDs and results. `dashboard_frame_index.jsonl` records every captured dashboard state. `movie_validation.json` records decode, stream and pixel checks.
