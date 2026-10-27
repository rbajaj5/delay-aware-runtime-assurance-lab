# Sixteen-Drone GPU Fleet Checkpoint

User requested a larger multi-agent fleet, explicit physical trajectories, and
an upgraded narrated video on GitHub. Preserve the earlier four-drone study.

## Current Implementation

- New experiment copied from `20261027_gpu_fleet_active_state`, not a rewrite
  of its evidence. Sixteen drones, four fixed pad assignments, four waves.
- 96 velocity DOFs, 64 rotor actuators per world; mass/inertia/rotor limits
  unchanged. Gravity and hull/foot contacts integrated on CUDA (MuJoCo Warp).
- 100 s episodes, 2 ms physics, 20 ms control/logging; approach/departure
  references 8 s, dwell 2 s. Waiting rows y=4,7,10,13; departure parking is
  filled farthest first to avoid stationary agents blocking later flights.
- GPU overflow flags checked every control step; a flagged run fails.
- Pilot: six worlds at seed 45300; nominal both finish 16, ACK loss timeout
  finishes 10 with conflicts / queue-aware finishes 16 without conflicts;
  outage timeout finishes 0 / queue-aware 4 with 12 waiting. Not pooled.
- Nine controller/protocol unit tests pass, including latched excessive-tilt
  disarm applied identically to both policies.
- First 48-world run aborted with nonfinite state near 99.54 s. No valid results
  retained; failure record and source snapshot in `evidence/`. Failing world ID
  was not captured by its original failure handler, so exact cause is unknown.
- Added motor disarm above 70 degrees in both arms; physics continues under
  gravity/contact. This is a new shared modeling assumption, not damage validation.
  New failure handler saves invalid state and world IDs. Never pool first run.

## Completed Work

- Revised main simulation completed: 48 worlds = 8 seeds x 3 profiles x 2
  policies in `evidence_v2/`. GPU stepping/CPU control/logging 165.72 s;
  setup/kernel loading 12.81 s. No overflow or nonfinite states.
- Audit passed: 240,048 world states, source/input hashes, summary endpoints,
  quaternion norms, rotor bounds, queue ownership, GPU hover/free fall.
- Nominal pair events are exactly equal. Inherited mixed-coordinate 2e-5
  tolerance failed on small float32 contact differences. Inspected all pairs;
  now separate 0.1 mm position / 0.01 degree attitude tolerances. Measured
  maxima: 0.0383 mm / 0.00217 degrees. Documented post-run audit correction.
- Nominal both complete 16; ACK loss timeout mean 10 (6-14) with conflicts in
  8/8, queue-aware 16 with 0/8 conflicts. Outage timeout 0 with 8/8 conflicts,
  queue-aware 4 completed / 12 waiting with 0/8 conflicts. Small synthetic suite.
- Featured CSV export verified all 480,096 rows (six worlds at seed 45300).
  Full 214 MB trajectory archive and 42 MB CSV are release assets, not Git.
- Neural speech complete: 201.314 s, seven chapters, thirteen cues. One empty
  speech-service response retried; revised narration discloses the disarm rule.
- Updated renderer has measured xyz trails and altitude histories, four pad
  ownership indicators and 16 persistent agent identities. Previews inspected;
  fixed camera widened/rotated to include paths after falls; labels deconflicted.
- Rendering completed: 5,033 frames / 201.32 s. Render/encode 418.1273 s;
  mux 3.15194 s. Intel Graphics OpenGL replay / CPU libx264, separate from CUDA.
- Media verifier checks exact frame links, audio, encoded frames and framing
  of every featured position, including unshown intermediate samples.
- Media audit passed all 5,033 frame links, 13 audio segments, full decode,
  seven encoded comparisons and 480,096 projected position samples. Inspected
  encoded frames visually. MP4 SHA-256:
  `6586784a56722f2893f0b03687b4b67ac0bfe1bf88eb983baef4b18aea0fb514`.
- Measured timings and replay coverage are now documented beside the video.

## Publication Complete

- Research source/evidence/front page pushed in `22b8ddf7947a1effc3f3fc368b1690dcfc0fa556`.
- Public release `gpu-fleet-trajectories-2026-10-27` targets that commit.
  Movie, captions, full main NPZ and featured CSV uploaded; all four GitHub
  asset digests and byte lengths match local files. See `delivery/publication.json`.
- Gardening related-example link pushed in `607e40f`. No laundry changes.
- Earlier four-drone source, results and movie preserved. No active jobs remain.
- No required implementation remains. Optional future work: battery-limited
  recovery, more seeds/layouts, decentralized ownership and hardware validation.
  These are not claims of this demonstration.

## Resume Commands

From this directory, physics/render Python is
`../../../.venv-laundry-gpu/Scripts/python.exe`, voice Python is
`../../../.venv-gardening-voice/Scripts/python.exe`.

```powershell
../../../.venv-laundry-gpu/Scripts/python.exe validate.py --out evidence_v2
../../../.venv-laundry-gpu/Scripts/python.exe -m unittest validate.ControllerTests test_protocol -v
../../../.venv-laundry-gpu/Scripts/python.exe export_paths.py
../../../.venv-laundry-gpu/Scripts/python.exe video.py --preview
../../../.venv-laundry-gpu/Scripts/python.exe video.py
../../../.venv-laundry-gpu/Scripts/python.exe verify_media.py
```

Do not overwrite completed evidence. Choose a fresh `--out` for another run.
Only this new experiment, its research README feature and the gardening link
are in scope for commits; unrelated historical dirty files must remain intact.

Release tag: `gpu-fleet-trajectories-2026-10-27` in
`rbajaj5/delay-aware-runtime-assurance-lab`. Movie name:
`sixteen_drone_gpu_trajectories.mp4`. Do not replace the previous release.

## Later Neural Hanzi Camera and Lesson Edition

Completed in `neural_hanzi/`, using the same recordings, no new physics.
Six original memory terms now drive a 189.68 s narrated lesson with large
matched landing close-ups, retained full-fleet overviews and linked altitude
charts. All 4,742 frame links, audio/caption checks, overview/landing-center
framing checks and nine existing unit tests passed. Render/encode 511.61518 s;
mux 2.66569 s. Coverage and boundaries are in the lesson README.

Source/front page commit `72a77fa526a3b371f6585deab97c198b96d8b6f7`; new release
`neural-hanzi-landings-2026-10-27`. Both uploaded asset digests verified against
local copies. Gardening related link commit `acc1350`. Prior releases preserved.
No active jobs or pending implementation. See `neural_hanzi/CHECKPOINT.md`.
