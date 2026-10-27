# GPU Fleet Checkpoint

October 27, 2026. User asked to upgrade videos using drone fleets, specifically
GPU flight physics, to illustrate Ravi's queue-aware multi-agent research rather
than making the Chinese vocabulary the main contribution.

Repository: delay-aware-runtime-assurance-lab. New isolated experiment only.
Keep the existing dirty/untracked historical artifacts and deleted old formation
MP4 untouched. Preserve the old 192-run kinematic video, gardening and laundry.

## Completed

- CUDA MuJoCo Warp plant: 4 free quadrotor bodies, 16 bounded motor forces, contacts.
- Pilot v1 rejected: spherical support rolled when motors stopped. Retained.
- Pilot v2 stable with hull/feet but 30 s horizon cut off the second pair. Retained.
- Main: 48 worlds (8 seeds x 3 profiles x 2 arms), 40 s, 2 ms integration.
- Main loop 31.81 s; setup/cached compilation 12.57 s, separately recorded.
- 96,048-world-state audit, source hashes, paired inputs, independent outcome
  recomputation and ownership invariants passed. Two controller tests passed.
- CUDA analytic hover/freefall sanity checks passed. Six additional protocol
  tests pass, for eight unit tests including the two controller tests.
- Outcomes: timely both 4 tasks; ACK loss 8/8 conflict episodes timeout vs 0/8
  queue-aware, mean completion .375 vs 4. Outage queue-aware finishes 2, holds 2.
- Documentation distinguishes constructed examples from original registered work,
  centralized task authority from distributed consensus, and GPU dynamics from
  rendering. Post-contact exact motion is not a validated crash/damage model.
- English/Mandarin neural speech complete: 17 cues, 221.746 s; 5544 video frames
  at 25 fps will give 221.76 s. Nine chapter previews inspected.
- Root research README features the new experiment while retaining older movies.
- First partial render was stopped to fix repeated NPZ decompression per frame.
  Renderer now loads qpos/qvel once. Physics evidence is unchanged.
- First complete render: 304.50 s + 2.50 s mux. Replaced after plot-label overlap
  was found in encoded frames. Final renderer moves labels above plots and caches
  identical paused images; six cached/uncached image comparisons are exact.
- FINAL MOVIE complete and audited: 221.76 s, 5544 frames, 1080p25, English neural
  narration, eight Mandarin term pronunciations, burned and embedded captions.
- Final measured render/encode 176.89275290002115 s; mux 2.297382800024934 s.
- All 5544 frame links, 17 audible cues, full decode and nine encoded image
  comparisons passed. Replay coverage per arm: timely 607/2001; lost ACK
  1477/2001; outage 573/2001. Each includes t=0 and t=40, seed 45200 only.
- Final MP4 SHA-256: 8aa928e454da4c74405dfc791e9a65a0029e235972f7ed3ec1a3a8ab85031fbf.
- Eight unit tests pass. Exact-byte Git attributes protect source hashes from
  checkout line-ending conversions. Existing historical artifacts untouched.

## Publication Complete

- No simulation/render processes remain active. Do not rerun completed evidence.
- Research source and front page pushed at
  `a045fd75ecd1e6a57c25f25bd398c0d3f7422cc0`.
- Public release `gpu-fleet-active-state-2026-10-27` targets that exact commit.
  MP4 (11,631,944 bytes) and captions uploaded. Remote MP4 digest matches the
  audited local SHA-256 above. No old release was replaced.
- Gardening related-example link pushed at `513002d`. Laundry untouched.
- No implementation work remains for this bounded upgrade. Further mission
  profiles, decentralized coordination or battery-aware recovery would be new
  experiments, not claims established by this film.

Description page:
https://github.com/rbajaj5/delay-aware-runtime-assurance-lab/tree/main/experiments/20261027_gpu_fleet_active_state

Release:
https://github.com/rbajaj5/delay-aware-runtime-assurance-lab/releases/tag/gpu-fleet-active-state-2026-10-27

## Commands

From this experiment directory (local Python paths, not portable requirements):

```powershell
../../../.venv-laundry-gpu/Scripts/python.exe -m unittest validate.ControllerTests test_protocol -v
../../../.venv-laundry-gpu/Scripts/python.exe validate.py
../../../.venv-gardening-voice/Scripts/python.exe speech.py
../../../.venv-laundry-gpu/Scripts/python.exe video.py --preview
../../../.venv-laundry-gpu/Scripts/python.exe video.py
../../../.venv-laundry-gpu/Scripts/python.exe verify_media.py
```

Do not rerun final evidence. For a new run use `simulate.py --out NEW_DIRECTORY`.
The narration source is `lesson.json`; only public text goes to online TTS.
Rendering timing must be measured, not guessed; keep it beside the movie as
requested. No opening timing card is needed.
