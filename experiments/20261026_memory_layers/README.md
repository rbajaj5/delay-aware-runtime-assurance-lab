# Long-term Memory vs Working Memory / 长期记忆与工作记忆

A 2:08 narrated, captioned teaching video using a reproducible four-drone,
two-pad coordination model. This extends the earlier
[working-memory explainer](../20261026_working_memory/README.md), without changing
the course-study results or the separate humanoid repositories.

[![Watch the memory-layers comparison](delivery/memory_layers_poster.jpg)](https://github.com/rbajaj5/delay-aware-runtime-assurance-lab/raw/refs/heads/main/experiments/20261026_memory_layers/delivery/memory_layers_drone_fleet.mp4)

**[Watch/download the 1080p MP4](https://github.com/rbajaj5/delay-aware-runtime-assurance-lab/raw/refs/heads/main/experiments/20261026_memory_layers/delivery/memory_layers_drone_fleet.mp4)**
| [Captions](delivery/memory_layers.srt)
| [Episode summaries](evidence/episode_summaries.json)
| [Aggregates](evidence/aggregates.json)
| [Validation](evidence/validation.json)

GitHub's binary-file page may not preview the film; the links above address the
MP4 directly. English narration is generated locally with Microsoft Zira.
Chinese labels identify the concepts; this is not a native-Mandarin voice lesson.

## The distinction

- **长期记忆 / long-term memory:** reusable map and procedural knowledge, identical
  in both controllers. In this model it is static configuration, not a learned
  memory system or a persistence benchmark.
- **工作记忆 / working memory:** active task records containing pad ownership,
  command identity, acknowledgment state, and observation age.
- **消息队列 / message queue:** messages in transit, separate from the active
  task record. All due messages are processed at each tick; there is no
  compute-capacity bottleneck in this particular experiment.

These are functional engineering analogies. The model does not establish how
human memory works. Both arms have working state: the comparison is between
two reservation-release policies, not between a system with memory and one
without it, nor a claim that long-term memory is inferior to working memory.

## Executed model

`simulate.py` runs CPU discrete-time kinematics, not GPU flight physics. Four
drones (A-D) have fixed identities and two fixed pad assignments. Commands
authorize approach, landing service, and departure. Overlapping occupants are
counted as resource conflicts; no collision response is modeled. An unsafe
episode can therefore still finish all four tasks. Completion alone is not
safety.

Both policies share:

- The same pad map and one-drone-per-pad procedure.
- A 0.1 s tick and 14 s episode horizon.
- Matched travel, dwell, command-delay, ACK-delay, and observation-delay draws.
- Timestamp-monotone observation acceptance and a 0.6 s freshness gate.
- Unique command identities and exactly one modeled execution per identity.
- A fresh observed clear-pad check before issuing a clearance.

Only release of an unresolved reservation differs:

1. **Timeout-only:** if no acceptance ACK arrives within 1.2 s, remove the
   reservation. Otherwise retain it until verified completion.
2. **Queue-aware:** do not treat a missing ACK as cancellation. Retain the
   reservation until a fresh observation includes that command's completion
   and shows its pad clear.

An acceptance ACK is not a completion receipt. A physically empty pad is not
necessarily available: a drone may already be authorized and approaching it.
This is the failure mechanism. At 4.5 s an old empty-pad observation is delivered;
both controllers reject it. Thus the comparison does not confound reservation
retention with different timestamp filters.

The authority is centralized, clocks agree, telemetry is exact when available,
command delivery is bounded and reliable, and drones obey their clearance.
The protocol has no restart persistence, authentication, clock-error model,
distributed consensus, collision-avoidance controller, battery constraint, or
communication recovery procedure. Unresolved ownership is not a time-expiring
lease with independently enforced fencing.

## Inputs and results

Seeds 20261027-20261058: 32 matched seeds x 3 profiles x 2 policies = 192 runs.
Travel times are uniform on [2.6, 2.9] s, service dwell on [2.0, 2.3] s,
command/ACK/telemetry delays on [0.1, 0.2] s, and departure takes 0.8 s.
These narrowly chosen timing ranges demonstrate a constructed mechanism. They
are not representative samples of deployed drone fleets.

| Profile | Timeout conflict episodes | Queue-aware conflict episodes | Finished tasks, timeout / aware |
|---|---:|---:|---:|
| Timely messages | 0/32 | 0/32 | 4 / 4 |
| All acceptance ACKs lost | 32/32 | 0/32 | 4 / 4 |
| ACKs lost; telemetry stops at 1 s | 32/32 | 0/32 | 4 / 2 |

Mean time with at least one conflicting pad is 1.04375 s for the timeout arm
under both fault profiles. Queue-aware retention trades progress for avoidance
of this resource conflict during the observation outage. These counts come
from `evidence/`, not constants in the renderer. The outcome pattern is strongly
determined by the protocol and chosen timings; it is not a learned advantage
or a statistical guarantee.

`mean_issue_time_s` averages only issued commands; it must not be compared
across arms with different numbers issued as a standalone performance score.
The plots instead expose unfinished tasks explicitly. Conflict duration is
the number of sampled conflicting steps multiplied by 0.1 s, not an exact
continuous-time collision duration.

Inputs are paired by seed and hashed. Observed message payloads can differ
between controllers because their actions change the simulated world. This
is matched random input, not identical observation tapes.

## Replay coverage and media

The video uses only seed 20261027, not a compilation of best runs. All six
policy/profile traces for that seed are retained as JSON with configuration,
inputs, events, and per-step drone coordinates and controller state.

- 00:00-00:24: definitions and labeled schematic example, not a replay.
- 00:24-01:24: matched lost-ACK episode, slowed from simulated 0.0-13.9 s.
- 01:24-01:44: matched telemetry-outage episode, simulated 0.0-13.9 s.
- 01:44-01:56: all 192 episode summaries, grouped by profile and policy.
- 01:56-02:08: interpretation and sources.

For each replay, 140 of 141 saved states appear; the 14.0 s endpoint is omitted
by the half-open video interval. Completed-task counts are already terminal by
the final rendered state. `evidence/video_frame_index.json` maps every rendered
frame to its exact profile and saved state index, or null for explanatory
sections. Rendered geometry and telemetry come from that same saved state.
The map shows the simulated physical state of active drones; departed drones
move to the finished-ID list rather than remaining overlaid at the exit.
Active records show the coordinator's state. Pad labels include occupant counts.
Rotor animation is decorative only.

## Reproduce

Python 3.11+, the packages in `requirements.txt`, FFmpeg and FFprobe on PATH.
Current typography uses Windows Segoe UI and Microsoft YaHei. Narration uses
Windows System.Speech / Microsoft Zira Desktop. No paid API is required.

From this directory:

```powershell
python simulate.py
python -m unittest test_simulate -v
python video.py --prepare --preview
.\narrate.ps1
python video.py --render --mux
python verify.py
```

The program fails rather than silently omitting narration or skipping an audit.
`build/` contains ignored intermediate video, audio, and preview images.
`delivery/` contains the final MP4, poster, and captions. `evidence/` contains
replay data, compact episode summaries, and validation. The new code and
authored graphics follow the repository's MIT license.

## Sources and attribution

- N. Cowan, [Working Memory Underpins Cognitive Development, Learning, and
  Education](https://pmc.ncbi.nlm.nih.gov/articles/PMC4207727/), 2014: cognitive
  background for the term, not validation of this engineering analogy.
- R. D. Yates et al., [Age of Information: An Introduction and
  Survey](https://arxiv.org/abs/2007.08564), preprint 2020 / published 2021:
  timestamped update freshness. Freshness alone does not resolve ownership.
- Ravi Bajaj's [queue-aware research collection](../../README.md) and
  [earlier working-memory demonstration](../20261026_working_memory/README.md):
  project context. This is newly authored teaching code, not a rerun of the
  original registered quadrotor study.

Prepared with OpenAI Codex assistance. No external footage, proprietary model
weights, financial forecasts, or claims from the Neural Hanzi news briefing
are used in this experiment.
