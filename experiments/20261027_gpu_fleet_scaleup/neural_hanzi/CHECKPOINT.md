# Neural Hanzi Landing Lesson Checkpoint

## Request and Scope

Keep the 16-drone overview, add synchronized large landing close-ups, and center
the lesson on exactly six terms: 工作记忆, 会话状态, 团队记忆, 记忆片段, 观测时效, 长期记忆.
This is an edit to the presentation, not a new physics experiment. Preserve
all prior source, recordings, results, movies and release tags.

## Completed

- Read the existing experiment checkpoint and original gardening glossary.
- Rechecked Alibaba AgentRun, Tencent Agent Memory and Alibaba Bailian Chinese
  engineering documentation. Sources and bounded interpretations in README.
- New six-chapter public script and neural speech complete: 12 cues, all six
  Mandarin pronunciations, English explanations. Duration 189.672 s before
  rounding to 25 fps. Reused cached cues after transient empty TTS responses.
- New renderer: common fixed P1 close-up camera in both arms, full-fleet
  overviews, measured paths, live owner/occupancy/freshness/task fields and
  synchronized P1 altitude histories. No position interpolation or CPU physics.
- Close-up shows drones 01/05/09/13; all physical bodies remain in scene.
  Preview audit: all 24,311 P1 landing-center samples fit the close-up.
- Six preview images inspected. The final chapter returns the large panels
  to the full fleet at simulation time 80 s to avoid an empty-pad ending.
- Parent renderer and physics code are unchanged. Parent experiment inputs
  and all prior validations remain intact.
- Final render complete: 4,742 frames / 189.68 s, 1080p25. Render/encode
  511.61518 s; audio/subtitle mux 2.66569 s. Timing exclusions in README.
- Media audit passed every frame link, full decode, 12 audio cues, subtitle
  track, six encoded-image comparisons, 480,096 overview centers and 24,311
  near-P1 landing centers. Encoded opening/conclusion inspected visually.
- All nine existing controller/protocol unit tests passed. Original physics
  source and evidence have no changes against HEAD.
- README records selected-window coverage and production timing.
- Research front page, experiment banner and gardening related link edited.

## Publication Complete

- No running simulation/render jobs. Prior rendering attempt was stopped early
  to improve the conclusion camera; none of its output is delivered evidence.
- Source and front page pushed in `72a77fa526a3b371f6585deab97c198b96d8b6f7`.
- Release `neural-hanzi-landings-2026-10-27` targets that exact commit in
  `rbajaj5/delay-aware-runtime-assurance-lab`. MP4 and SRT uploaded; both
  GitHub asset digests and sizes match local files. See `delivery/publication.json`.
- Full trajectory input remains in `gpu-fleet-trajectories-2026-10-27`.
- Gardening related link pushed in `acc1350`. No laundry changes.
- No required implementation remains. Historical unrelated dirty artifacts
  were neither staged nor changed. Any future experiment needs a separate
  evidence directory; do not overwrite the current recordings.

## Resume

From this directory, physics/render Python:
`../../../../.venv-laundry-gpu/Scripts/python.exe`.
Voice Python: `../../../../.venv-gardening-voice/Scripts/python.exe`.

```powershell
../../../../.venv-laundry-gpu/Scripts/python.exe render.py --preview
../../../../.venv-laundry-gpu/Scripts/python.exe render.py
../../../../.venv-laundry-gpu/Scripts/python.exe verify.py
```

Use `speech.py` only if narration changes. It sends the public script to online
TTS and caches completed segments. Do not rerun physical simulations for this edit.
