# Autonomous Fleets on Changing Networks
## From Konigsberg to Kaliningrad: a schematic prototype

A reproducible, synthetic multi-agent dispatch experiment and a narrated,
captioned 180-second, 1920 x 1080, 24 fps teaching movie. Nothing was published,
uploaded to Canvas, or submitted. Existing repository experiments were preserved.

## Scope and sources

The supplied historical graph has seven undirected edges and degrees 5,3,3,3.
NetworkX verifies connectivity, absence of an Euler trail, a detour after closing
a+b, and isolation of B after closing a+b+f. Euler's coverage problem is NOT a
requirement for taxi journeys. Modern Kaliningrad is not modeled geographically.
The permitted historical-schematic fallback avoids presenting unverified roads
as a current, access-checked map. All demand, times, delays and closures are synthetic.

Historical context: [MAA Convergence](https://old.maa.org/press/periodicals/convergence/leonard-eulers-solution-to-the-konigsberg-bridge-problem).
Source review date: 2026-10-14. `sources.json` identifies sources and repository
components adapted. OSM access guidance informed the decision to defer modern
road mapping; no OSM road dataset is used. No DC bikeshare data are used.
The fleet is inspired by the idea of Waymo taxis, not its software or measured performance.

## Protocol

- Six capacity-one taxis; 900 seconds in one-second steps; synthetic Poisson
  demand at 0.065 requests/second through second 720. Origins favor B; destinations
  are drawn from the other three nodes. No cancellation or abandonment is hidden.
- Seven fixed synthetic traversal times (28-48 seconds), stored in `config.json`.
  These are link-service times, not geometric lengths or observed road speeds.
- All-open baseline; a+b close at 240 and reopen at 540; a+b+f close at 240 and
  reopen at 600, temporarily isolating B. Closures apply equally to every arm.
- Base transmission lag is 1-8 seconds; common delay bursts add 18-30 seconds
  during [180,330) and [600,660). Periodic road and status messages are timestamped.
- Every recipient services at most one arrived packet per second. The dispatcher
  has 1.483 offered messages/second before boundary bursts, exceeding service
  capacity. This deliberately congested information channel explains much of the
  FIFO penalty, even with all links open. It is not evidence that FIFO is always bad.
- Policy 1 uses arrival-order FIFO and the newest processed timestamp per source.
  Policy 2 prioritizes commands, then roads, then status, then background traffic;
  it also coalesces obsolete arrived road/status snapshots. It is a policy package,
  not an isolated test of priority alone; background work can be starved.
- Policy 3 uses policy 2 plus a 25-second idle-status age bound and 45-second
  road-observation age bound. Dispatch checks pickup and passenger routes; each
  vehicle checks its current leg at every node departure. It waits at nodes when
  no fresh known route exists, or uses a fresh known detour. It does not reassign
  passengers or hold an obsolete route forever. No future closure schedule enters
  a controller's knowledge. An initial all-open inventory is shared by all arms.
- Actual closures block entry. If a link closes during traversal, progress freezes
  until reopening; interruption time is recorded. This intentionally simple rule
  is not a realistic safe-stopping/braking model. Zero movement through a closed
  link is a common environmental constraint, NOT an RTA achievement.
- A local blocked-entry observation updates only that taxi's knowledge of that
  link. Other knowledge arrives through delayed packets. Both status reservations
  and monotone timestamps prevent double assignments and old-state overwrites.

## Matching, variation, and replay

100 seeds per scenario (2026101400-2026101499), three scenarios, three policies:
900 policy episodes. The same 100 seed identities recur across scenarios.
NumPy SeedSequence creates separate demand, fleet, delay, and background streams.
All exogenous arrays are generated before a policy runs and never mutated.
Policy-dependent command times index the same pre-generated delay field; packet
payloads naturally differ with vehicle state. Scenario configuration and tape
hashes establish matching. `input_tapes/` retains all 100 tapes.

The replay uses seed 2026101400 in the detour scenario, selected
by minimum robust-scaled squared distance to the nine-dimensional median across
the three policies' completion, blocked entries, and capped pickup wait.
Ties use the smallest seed. `replay_selection.json` records the full rule and
scaling. It is representative by that stated rule, not representative of every
possible metric. The replay is rerun and checked against its original result.

## Computed results

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

Entries are means of episode-level measurements. Pickup wait includes all picked-up
riders, including those still onboard; it excludes those still waiting. Therefore
read it with uncompleted counts. `capped_pickup_wait_s` averages elapsed waiting
over ALL requests, using the horizon for those not picked up; it is a censored,
finite-horizon measure, not eventual waiting time. Completion is averaged by episode,
not pooled across riders. `summary.csv` also reports SD, median and empirical p10/p90.
Whiskers show simulation variability, NOT confidence intervals for a real city.

The freshness monitor reduces mean blocked entry attempts relative to priority
updates in both closure scenarios. In the detour case it also reduces completion
and increases pickup waiting. In the isolation case average completion is similar;
the table, paired differences and variability should be read together. Fewer raw
entry attempts can reflect less service exposure; `movement_exposure.csv` includes
attempts per 100 edge-entry intents. No collision or injury outcome is modeled.

## Files and units

- `fleet_changing_networks.mp4`: final narrated movie with burned-in captions
  and a selectable English subtitle track. `captions.srt` and `transcript.txt`
  contain the same narration. Voice: local Microsoft Zira Desktop, not a human.
- `01_euler.png` through `04_repeated_results.png`: four 1080p stills.
- `episode_results.csv`: one row per scenario, seed, policy (900 rows).
- `summary.csv`, `paired_differences.csv`: aggregated and within-seed comparisons.
- `requests.csv`: one request per policy episode, with arrival, assignment,
  pickup and completion times. Blank pickup/completion is censoring, not zero.
- `events.csv.gz`: full timestamped event CSV, gzip-compressed. Decompress for BI.
  `replay_events.csv` is an immediately importable, uncompressed representative log.
- `replay_timeline.csv`, `replay_vehicles.csv`: one-second replay metrics and agent
  coordinates. Layout coordinates are arbitrary schematic units, not GPS/meters.
- `movie_frame_index.csv`: maps every rendered frame to its replay second or
  results scenario; `replay.json` supplies all plotted state.
- Validation JSONs and SHA-256 manifest cover inputs, output, playback and checks.

A fleet-minute is summed vehicle time divided by 60. Empty/occupied travel count
actual moving seconds, not forced stops; occupied journey elapsed time can include
stops. Freshness fallback counts proactive route/status holds only. Forced closure
stops are reported separately for ALL policies. A waiting passenger is not yet
picked up; uncompleted equals waiting plus onboard at the horizon.

## Reproduction

Python 3.11+, pinned packages in `requirements.txt`, and FFmpeg/FFprobe on PATH:

```powershell
python -m pip install -r requirements.txt
python test_simulation.py
python simulate.py
python verify.py
python render.py --encoder h264_nvenc
powershell.exe -NoProfile -ExecutionPolicy Bypass -File narrate.ps1
python assemble_audio.py
python finalize.py
```

Use `--encoder libx264` without NVIDIA hardware. Graph simulation and rendering
are CPU-based; NVIDIA NVENC accelerates encoding only. `render.py --stills-only`
prepares figures and narration text without the movie. Non-Windows users can
retain the captioned silent video or generate `narration.wav` using their own TTS.
The numeric experiment does not depend on TTS. Rebuilding the movie does not rerun
the simulation. No credentials, paid APIs, or uploads are used.

## Limitations and next steps

This is an illustrative engineering experiment, not preregistered confirmatory
evidence. The configuration was smoke-tested before the final matched sweep.
Road capacity, signals, braking, pedestrians, vehicle dynamics, communication
loss, real taxi demand and real access restrictions are absent. Modern mapping,
queue-load and threshold sweeps, scalable networks and empirical data remain
future work. Freshness alone is not truth: a new closure can invalidate a recent
observation. These results do not establish general safety or Waymo performance.
