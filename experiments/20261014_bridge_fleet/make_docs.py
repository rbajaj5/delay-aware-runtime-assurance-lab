"""Generate a factual handoff and small BI exports from final retained outputs."""
import csv
import gzip
import hashlib
import json
from pathlib import Path
import shutil

HERE=Path(__file__).resolve().parent


def main():
    out=HERE/'output';config=json.loads((out/'config_used.json').read_text());replay=json.loads((out/'replay.json').read_text())
    summary={}
    with (out/'summary.csv').open(newline='') as f:
        for r in csv.DictReader(f):summary[r['scenario'],r['policy'],r['metric']]=float(r['mean'])
    def m(s,p,k):return summary[s,p,k]
    labels={'latest_received':'Latest received','priority_updates':'Priority updates','freshness_monitor':'Freshness monitor'}
    table=['| Scenario | Policy | Completion | Mean pickup wait (s) | Uncompleted | Blocked entries | RTA fleet-min |',
           '|---|---|---:|---:|---:|---:|---:|']
    for s in config['scenarios']:
        for p in config['policies']:
            table.append(f"| {s['id']} | {labels[p]} | {100*m(s['id'],p,'completion_fraction'):.1f}% | {m(s['id'],p,'mean_pickup_wait_s'):.1f} | {m(s['id'],p,'uncompleted'):.2f} | {m(s['id'],p,'closed_entry_attempts'):.2f} | {m(s['id'],p,'fallback_vehicle_min'):.2f} |")
    results='\n'.join(table)
    offered=len(config['edges'])/config['road_update_period_s']+config['fleet_size']/config['status_update_period_s']+config['dispatcher_background_rate_per_s']
    readme=f'''# Autonomous Fleets on Changing Networks
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
  has {offered:.3f} offered messages/second before boundary bursts, exceeding service
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

The replay uses seed {replay['selection']['seed']} in the detour scenario, selected
by minimum robust-scaled squared distance to the nine-dimensional median across
the three policies' completion, blocked entries, and capped pickup wait.
Ties use the smallest seed. `replay_selection.json` records the full rule and
scaling. It is representative by that stated rule, not representative of every
possible metric. The replay is rerun and checked against its original result.

## Computed results

{results}

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
'''
    (HERE/'README.md').write_text(readme,encoding='utf-8')
    # Small plain CSVs are directly consumable by either BI tool.
    with gzip.open(out/'requests.csv.gz','rb') as src,(out/'requests.csv').open('wb') as dst:shutil.copyfileobj(src,dst)
    events=[];vehicle_rows=[]
    from render import Movie
    movie=Movie(out)
    for policy,data in replay['policies'].items():
        events.extend(data['events'])
        for f in data['frames']:
            for v in f['vehicles']:
                xy=movie.edge_position(v['edge'],v['progress_s']/movie.edges[v['edge']]['travel_s'],(0,0,1,1),v['node']) if v['edge'] else config['nodes'][v['node']]
                vehicle_rows.append({'seed':replay['selection']['seed'],'scenario':replay['selection']['scenario'],'policy':policy,
                    'time_s':f['time_s'],'vehicle_id':v['id']+1,'schematic_x':float(xy[0]),'schematic_y':float(xy[1]),
                    'node_or_departure':v['node'],'edge':v['edge'] or '', 'edge_progress_s':v['progress_s'],
                    'passenger_request':v['task'] if v['task'] is not None else '', 'occupied':v['onboard'],'hold_reason':v['hold'] or ''})
    from simulate import write_csv
    write_csv(out/'replay_events.csv',events);write_csv(out/'replay_vehicles.csv',vehicle_rows)
    seeds=[]
    for file in sorted((out/'input_tapes').glob('*.json.gz')):
        with gzip.open(file,'rt',encoding='utf-8') as f:tape=json.load(f)
        seeds.append({'seed':tape['seed'],'requests':len(tape['requests']),'input_sha256':tape['sha256']})
    write_csv(out/'seeds.csv',seeds)
    audit=json.loads((out/'independent_validation.json').read_text())
    handoff=f'''# Factual handoff: JHU Data Visualization

Implemented a separate experiment at `{HERE}`. It uses the supplied historical
seven-bridge schematic, not real Kaliningrad streets, and compares arrival-order
FIFO dispatch, priority/coalesced updates, and priority plus freshness monitoring.
It reuses queue/delay and rendering patterns from the existing assured-autonomy
repository. No existing experiment was replaced; nothing was published or submitted.

Ran 100 matched seeds per scenario, three scenarios and three policies (900
episodes). All inputs are synthetic. Controllers receive delayed messages, never
the hidden closure schedule. Actual closures block every policy, including
in-progress motion under an explicit schematic stop rule. Representative replay
seed: {replay['selection']['seed']}, selected by the saved median-distance rule.

## Actual findings

{results}

Main comparison in the detour scenario: priority versus monitor completion is
{100*m('detour','priority_updates','completion_fraction'):.1f}% versus {100*m('detour','freshness_monitor','completion_fraction'):.1f}%;
mean blocked attempts are {m('detour','priority_updates','closed_entry_attempts'):.2f} versus {m('detour','freshness_monitor','closed_entry_attempts'):.2f};
mean picked-up passenger waits are {m('detour','priority_updates','mean_pickup_wait_s'):.1f}s versus {m('detour','freshness_monitor','mean_pickup_wait_s'):.1f}s.
This is a tradeoff, not universal superiority. Baseline FIFO is deliberately
overloaded by background messages. Do not generalize its penalty to uncongested queues.

Independent checks reconciled {audit['request_rows']:,} request records and
{audit['edge_exits_reconstructed']:,} completed link traversals across all 900
episodes. No vehicle progresses on a closed edge. Means and 10th-90th percentiles
come from saved summaries; the renderer does not invent outcomes. Counts, waiting,
empty/occupied travel and selected replay results were independently verified.

## Deliverables

Final video: `{out/'fleet_changing_networks.mp4'}`
Code and README: `{HERE}`
CSV tables, logs, transcript and validation: `{out}`
Paper stills: `{out/'01_euler.png'}` through `{out/'04_repeated_results.png'}`

## Remaining incomplete by design

No current motor-vehicle map, empirically calibrated taxi demand, real vehicle
physics, large-network or broad hyperparameter study, or operational flight/driving
validation. No Tableau workbook is built for this new experiment; the tables are
ready for that later step. GPU is used for video encoding, not graph simulation.
No claim of new theory or deployed Waymo performance. Final media QA is recorded
separately in `media_validation.json`.
'''
    (HERE/'HANDOFF.md').write_text(handoff,encoding='utf-8')
    provenance_file=out/'starting_code_provenance.json'
    provenance=json.loads(provenance_file.read_text()) if provenance_file.exists() else {}
    repo=HERE.parents[1]
    for name in ['analysis/run_fair_queue_multiagent_simulation.py','analysis/run_3d_formation_delay_monitor_simulation.py','analysis/portfolio_rta_demo.py','analysis/render_sensor_uncertainty_voiceover.ps1']:
        if (repo/name).exists():provenance[name]=hashlib.sha256((repo/name).read_bytes()).hexdigest()
    provenance_file.write_text(json.dumps(provenance,indent=2))
    print('README, factual handoff, source provenance and BI-ready replay tables generated.')


if __name__=='__main__':main()
