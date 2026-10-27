# Queue-Aware Drone Fleets: Active Task State in GPU Flight Physics

An illustration of **Ravi Bajaj's queue-aware assured-autonomy work**, extended
to four physical agents sharing two landing resources. The research question is:
**When does retaining unresolved work prevent conflicting actions, and what
progress is lost when the evidence needed to release that work disappears?**

[![Watch the narrated GPU fleet comparison](delivery/preview_3.jpg)](https://github.com/rbajaj5/delay-aware-runtime-assurance-lab/releases/download/gpu-fleet-active-state-2026-10-27/queue_aware_gpu_fleet.mp4)

**[Narrated MP4 with Mandarin terms and captions](https://github.com/rbajaj5/delay-aware-runtime-assurance-lab/releases/download/gpu-fleet-active-state-2026-10-27/queue_aware_gpu_fleet.mp4)**
| [Script](lesson.json) | [Captions](delivery/captions.srt)
| [Measured results](evidence/aggregates.json) | [State audit](evidence/validation.json)

**Measured final render + encode: 176.89 s (2 min 56.89 s); audio/subtitle mux:
2.30 s.** This is one observed run for a 3:42 movie, not total development time.
It excludes physics simulation, speech synthesis, renderer setup, audio assembly
and validation. Intel Core Ultra 9 275HX, 32 GB RAM; Intel Graphics rendered the
replay and CPU libx264 encoded it. NVIDIA RTX 5070 Ti Laptop GPU / 12 GB ran the
physics. Repeated paused images are cached; every frame still links to retained
simulation data. [Exact timing](delivery/timing.json) and
[final media audit](delivery/validation.json).

This is a **new exploratory extension**, not a rerun or enlargement of the
[registered course study](../../course_study/README.md). The earlier
[192-run kinematic fleet example](../20261026_memory_layers/README.md) is preserved
and its sample counts must not be pooled with these physics episodes.

## Connection to the Assured Autonomy Study

The course study tests prediction-horizon sensitivity for a PPO-controlled 2D
quadrotor with a FIFO actuation-delay channel and a finite candidate library.
This extension asks a different, multi-agent question: does the supervisor keep
track of valid commands that can still execute against a shared resource?
The common methodological idea is to account for pending actions explicitly,
compare matched inputs, explain the failure mechanism, and measure progress
alongside safety. A favorable fleet result does not reverse or generalize the
course study's non-monotone horizon result. The Chinese terminology helps explain
these distinctions; it is not the research claim.

## Measured Comparison

Eight fixed seeds (45200-45207), three message profiles and two policies:
**48 CUDA-integrated episodes**, each 40 s, with four drones in each world.
Parallel worlds are independent episodes, not one 192-drone swarm.

| Communication profile | Pad-conflict episodes: timeout / queue-aware | Mean completed tasks: timeout / queue-aware |
|---|---:|---:|
| Timely messages | 0/8 / 0/8 | 4.000 / 4.000 |
| Acceptance ACKs lost | 8/8 / 0/8 | 0.375 / 4.000 |
| ACKs lost; telemetry stops at 1 s | 8/8 / 0/8 | 0.375 / 2.000 |

All conflict episodes also cross the illustrative **0.8 m center-separation
threshold**. With fresh telemetry, queue-aware runs have minimum separation
at least 1.9906 m and maximum tilt 6.604 degrees. When telemetry stops, they
leave two tasks unfinished, with no tested separation violation. These are
deliberately constructed fault cases with narrow timing/noise ranges, **not
reliability probabilities or a general guarantee that queue awareness wins**.

Pad conflict means at least two measured vehicle centers lie within 0.85 m
horizontally of one pad and below 0.70 m altitude. This is a resource-zone
violation, not a count of physical impacts. Contacts affect the rigid bodies,
but post-contact motion has no damage or emergency-shutdown model. Do not
interpret its exact tumble trajectory or later completion count as a validated
crash prediction. Separation and conflict are sampled every 20 ms; between-step
violations could be missed. Counts are recomputed from retained coordinates.

## The Mechanism

Both arms have the same map, initial states, rotor controller, fresh-observation
gate, unique command identities and exogenous wind/delay draws. Both reject
out-of-order observations. Only release of an unresolved reservation differs:

1. **Timeout-only:** without an acceptance ACK for 1.2 s, remove the reservation.
2. **Queue-aware:** retain it until fresh telemetry confirms that its particular
   drone finished and the pad is clear.

An acceptance ACK is not completion. An empty pad can already be committed to
an approaching drone. Each drone obeys its received clearance, while a central
coordinator maintains the shared ledger. A new clearance requires fresh telemetry
(age at most 0.6 s), a reported clear pad, and no retained owner.
Each drone has one command in this bounded experiment; deduplication by drone
identity is not a general multi-mission idempotency implementation.

The supervisor does **not** get privileged current coordinates for its release
decision. Ground-truth coordinates are used for physics, local flight feedback,
and retrospective evaluation. Telemetry carries delayed observations generated
by that arm's world; matched random inputs do not mean identical endogenous
observation payloads. Current task state is more precise here than a generic
claim about working-memory capacity. No LLM is controlling this fleet.

When telemetry fails, already accepted missions continue locally, but the
coordinator cannot verify their completion. The remaining pair waits. An
indefinite hover is **not** an operationally safe fallback with limited battery;
energy-aware recovery is outside this model. This is an experimental runtime
supervisor, not a verified barrier-function or reachability safety certificate.

## Actual GPU Physics

- MuJoCo Warp on an NVIDIA RTX 5070 Ti Laptop GPU; CUDA required, no CPU fallback.
- Four free rigid bodies per world, 24 velocity degrees of freedom, 16 rotor actuators.
- Each vehicle: 1.2 kg, diagonal inertia (0.025, 0.025, 0.045) kg m^2.
- Four rotors at (+/-0.23, +/-0.23) m; 0-6 N each, with alternating yaw reaction.
- Gravity 9.81 m/s^2; 2 ms physics integration; 20 ms local control and state logs.
- Measured-position/velocity feedback, geometric attitude feedback and rotor mixing.
- Five-second smooth approach/departure **references**, not prescribed positions.
  Actual arrival and completion depend on measured motion; a reference is not
  direct assignment to the body's position or velocity.
- Hull and four landing feet contact the ground and pads. Rotor disks are visual
  geometry only. No CFD, downwash, propeller damage, motor lag, battery, GPS error,
  perception, learned policy or hardware validation.

Task scheduling and the feedback calculation run on CPU; **integration of the
positions, orientations, velocities and contacts runs on CUDA**. The video uses
OpenGL to replay those recorded states and CPU libx264 to encode them. GPU physics
must not be confused with GPU rendering or video encoding.

All 48 worlds ran in one batch. The measured simulation/control/recording loop
took **31.81 s**, excluding **12.57 s** of initialization and cached-kernel setup,
and excluding archive compression, rendering, TTS and validation. It is one
observed run, not a pure GPU-kernel benchmark.

## Validation and Development Record

The independent audit checks **96,048 world states**, finite values, unit
quaternions, rotor bounds, input hashes, replay links, endpoint recomputation,
and the queue-aware ownership invariant. Timely paired trajectories agree within
2e-5 absolute tolerance. Two controller unit tests check mass/inertia, actuator
dimensions, hover thrust, saturation and motor-off commands.
Six additional protocol tests check unresolved ownership, timely and late ACKs,
stale/out-of-order observations and owner-specific completion plus pad clearance.
All eight tests pass.

Separate CUDA sanity checks hold altitude with zero measured drift for 0.2 s
and match analytic free fall within 0.001963 m, consistent with the finite-step
integrator. These are narrow implementation checks, not full model validation.

The unsuccessful pilots remain in `pilot_v1/` and `pilot_v2/`:

- v1 used a spherical contact proxy that rolled with motors off, producing
  nominal tilt near 177 degrees. It is rejected as the main comparison model.
- v2 uses the hull and landing feet, eliminating that nominal instability, but
  its 30 s horizon ends before the second pair finishes. The final horizon is
  40 s for both arms and all profiles, with no change to the fault timing.
- The first launch failed before an episode because a Warp spatial-vector array
  needed its six-component axis during force assignment. That integration bug
  was fixed before the retained v1 pilot.

No pilot is silently relabeled as a successful final run. Final and v2 source
snapshots are retained alongside their outputs. v1 retains its scene XML, config,
hashes and trajectories, but not a complete source snapshot; exact v1 code
reconstruction is therefore not claimed. Contact outcomes can be numerically
sensitive; fresh GPU runs need not reproduce post-contact trajectories bitwise.
This directory disables Git line-ending conversion to preserve the exact bytes
covered by source and media hashes on Windows and other platforms.

## Chinese Engineering Vocabulary

| Term | Translation | Role here |
|---|---|---|
| 当前任务状态 | Current task state / active task state | Current evidence, unresolved commands and ownership. Our preferred label for the implemented records. |
| 工作记忆 | Working memory | Earlier explanatory analogy; not a cognitive or context-capacity benchmark. |
| 长期记忆 | Long-term memory | Persistent knowledge; the map is a static illustration, not a learned memory store. |
| 消息队列 | Message queue | Messages in transit, separate from the task ledger. |
| 资源预留 | Resource reservation | Authority retained for a pending or underway mission. |
| 团队记忆 | Team memory | Shared knowledge is not itself an execution reservation. |
| 观测时效 | Observation freshness | Our timestamp-based evidence-age gate. |
| 运行时保障 | Runtime assurance | The experimental execution gate, not a formal safety proof. |
| 匹配对照 | Matched comparisons | Same exogenous timing/noise inputs across policies. |

[Alibaba AgentRun](https://help.aliyun.com/zh/agentrun/memory-storage-overview)
distinguishes long-term memory, session history and session state.
[Tencent Agent Memory](https://cloud.tencent.cn/document/product/1813/132100)
distinguishes task context, persistent memory and shared team assets.
[Alibaba Bailian](https://help.aliyun.com/zh/model-studio/long-term-memory)
describes stored memory fragments. These engineering sources motivate vocabulary,
not the flight model or our supervisor's safety claims. No psychology sources or
vendor implementation tests are used. Sources checked October 27, 2026.

## Movie Coverage

The 221.76 s movie contains 5,544 frames at 1920x1080 / 25 fps. It illustrates
seed 45200 in three separate communication profiles, with both policies shown
at the same simulated time. It is not an uninterrupted 221.76 s flight.

| Replay profile | Distinct retained time samples shown per arm | Available samples per arm |
|---|---:|---:|
| Timely messages | 607 | 2,001 |
| Acceptance ACKs lost | 1,477 | 2,001 |
| ACKs and telemetry lost | 573 | 2,001 |

Each profile includes its start and end state; accelerated playback skips some
20 ms samples, while definitions and conclusions pause on a measured state.
The [frame manifest](delivery/frame_manifest.json) links every video frame to
its exact world and retained state. The [media audit](delivery/validation.json)
checks decoding, 17 audible narration cues, frame links and nine encoded-image
comparisons. Waveform checks are not a pronunciation or comprehension benchmark.

## Reproduce

Use the pinned package versions in `evidence/summary.json`, CUDA and the existing
MuJoCo Warp environment. `edge-tts==7.3.1` belongs in a separate speech environment;
only the public lesson text is sent to Microsoft's online speech service.
The tested host uses Python 3.11 on Windows, FFmpeg/ffprobe on PATH and the
Windows Segoe UI/Microsoft YaHei fonts referenced in `video.py`. Rendering on
another OS requires replacing those font paths. Install rendering dependencies
from `requirements-render.txt`; physics and voice have separate requirement files.

```powershell
python simulate.py --seeds 8 --out reproduced
python validate.py --out reproduced
python -m unittest validate.ControllerTests test_protocol -v
# Narration/rendering use the retained evidence/ directory:
python speech.py
python video.py --preview
python video.py
python verify_media.py
```

The simulator refuses to overwrite an existing summary. The film shows fixed
seed 45200, not selected best trajectories. Its frame manifest records exact
sample coverage; paused explanations and accelerated playback are explicit.
All aggregate counts come from eight seeds, never from one illustrated trace.
Neural narration uses Andrew in English and Xiaoxiao in Mandarin, at -5% rate,
without speeding the resulting audio up. Online TTS may change between runs.

Authored with OpenAI Codex assistance. No real-world aircraft operation,
commercial memory-product validation or deployment-ready safety claim is made.
