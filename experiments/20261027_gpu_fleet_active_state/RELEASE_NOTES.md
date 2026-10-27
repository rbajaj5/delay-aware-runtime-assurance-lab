# Queue-Aware Drone Fleets with GPU Flight Physics

A narrated illustration of Ravi Bajaj's queue-aware assured-autonomy work:
four flying agents share two landing pads while commands, acknowledgments and
observations travel through distinct message queues.

**[Methods, results, measured rendering time and source](https://github.com/rbajaj5/delay-aware-runtime-assurance-lab/tree/main/experiments/20261027_gpu_fleet_active_state)**

- `queue_aware_gpu_fleet.mp4`: 3:42, 1920x1080, 25 fps, English neural narration,
  Mandarin term pronunciations, burned-in captions and a subtitle track.
- `captions.srt`: the synchronized captions as a separate file.
- MuJoCo Warp CUDA integrates mass-bearing rigid bodies, rotor forces and
  contacts. The coordinator and local feedback run on CPU; rendering/encoding
  are not presented as GPU physics.
- 48 episodes: eight matched seeds, three communication profiles, two policies.
  Lost acceptance acknowledgments produce pad conflicts in 8/8 timeout episodes
  versus 0/8 queue-aware episodes. With telemetry lost as well, the queue-aware
  fleet completes two tasks and leaves two waiting.
- Full retained poses, per-episode summaries, source snapshots, narrow GPU
  calibration checks and a frame-to-state manifest are in the experiment folder.

These are constructed exploratory mechanism checks, not fleet reliability
estimates, validated crash predictions, a formal safety certificate, or an
expansion of the registered course study. No learned controller or LLM operates
the fleet. Older physics and kinematic demonstrations remain available.
