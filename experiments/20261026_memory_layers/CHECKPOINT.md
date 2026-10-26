# Memory-layers video checkpoint

## Scope

Create and publish one narrated video contrasting long-term memory and working
memory using a four-drone landing protocol. Keep message transport distinct.
Repository: `rbajaj5/delay-aware-runtime-assurance-lab`, branch `main`.
No changes to registered evidence, the existing explainer, or humanoid repos.

## Completed

- Deterministic CPU kinematic simulator and full configuration.
- 192 runs: 32 matched seeds, 3 profiles, 2 controllers.
- Six full featured-seed replays, all episode summaries, and aggregate results.
- 11 tests passed, including exact reproduction of all episode summaries and
  featured traces, common-input pairing, geometry-derived conflicts, and the
  timely-message control.
- Video authoring, Chinese labels, English voice generation, and preview QA.
- Front-page preview and direct MP4 links prepared.
- Final MP4 rendered: 128 s, 1920x1080, 24 fps, H.264/AAC and embedded captions.
- All 3,072 frames decoded; 11 encoded keyframes compared with their source
  rendering; audio duration/peak/RMS checked; final visual review completed.
- MP4 SHA-256: `02c53bd68563c59c002f4071d061b190a3d4783ed20944dc9ae890a1462bc0af`.

## Pending at this checkpoint

- Commit only this folder and the root README; push and verify the public files.

## Exact resume commands

From this directory, using any Python 3.11+ environment with NumPy/Pillow:

```powershell
python -m unittest test_simulate -v
python video.py --prepare --preview
.\narrate.ps1
python video.py --render --mux
python verify.py
```

The local environment used here is the workspace's
`.venv-laundry-gpu\Scripts\python.exe`; using this environment does not imply GPU
execution. This example performs CPU kinematics, rendering, and encoding.

## Preserve

The root working tree contains many pre-existing untracked historical artifacts
and a deleted `artifacts/20260803/formation_3d/formation_3d_delay_monitor_comparison.mp4`.
Do not stage, delete, restore, or publish those changes for this task.

## Limitations

Fixed map, two pads, four agents, reliable bounded command delivery, exact
telemetry when available, synchronized clocks, central coordinator, no crashes
or restart semantics. 32 seeds vary narrow timing ranges, not scene geometry.
No aerodynamics, learned policy, GPU, hardware, or cognitive validation.
Movie replay covers 140/141 states in each shown episode; the final 14.0 s
sample is not displayed, and terminal completion counts already agree at 13.9 s.
