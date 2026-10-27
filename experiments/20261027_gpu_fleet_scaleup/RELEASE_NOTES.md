# Sixteen-Drone GPU Flight Trajectories

A narrated, captioned 3:21 illustration of Ravi Bajaj's queue-aware autonomy
research. Sixteen physical drones share four pads. Persistent measured xyz
trails, altitude histories, pad ownership and completion counts are synchronized.

MuJoCo Warp integrates rigid-body flight/contact physics on NVIDIA CUDA.
Task coordination and local feedback use CPU; OpenGL replay and CPU video
encoding are separate operations. This is not a learned controller, CFD,
hardware validation, or a new registered course-study result.

Eight matched seeds, three communication profiles, two policies: 48 worlds,
16 drones per world, 100 s each. With lost acceptance receipts, timeout-only
produces sampled pad conflicts in 8/8 episodes and completes a mean of 10
missions (range 6-14); queue-awareness produces 0/8 such episodes and completes
16. With telemetry also unavailable, queue-awareness completes only four and
leaves twelve waiting. No battery-safe indefinite-hover claim is made.

Both arms share a latched 70-degree tilt disarm. The failed preliminary batch
is excluded and documented. The nominal pair audit uses explicitly documented
position/attitude tolerances rather than claiming bitwise float32 agreement.

## Assets

- `sixteen_drone_gpu_trajectories.mp4`: 1080p, 25 fps, English neural narration,
  Mandarin terminology, burned captions and selectable subtitle track.
- `captions.srt`: exact synchronized captions.
- `trajectories.npz`: all 48 worlds' retained physics; place in `evidence_v2/`
  beside the checked-in summary for replay and independent audit.
- `featured_flight_paths.csv.gz`: 480,096 readable position/orientation/velocity
  rows for the six featured first-seed worlds. This CSV does not cover all seeds.

See `experiments/20261027_gpu_fleet_scaleup/README.md` for measured timings,
methods, coverage, limitations, scripts and verification records. The earlier
four-drone experiment and its release are preserved separately.
