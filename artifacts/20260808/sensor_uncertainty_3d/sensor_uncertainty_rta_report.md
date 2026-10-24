# 3D Sensor-Uncertainty Formation Demonstration

This is an illustrative kinematic experiment, not a reproduction of Liu et al. (2024) or Dawood et al. (2023), and not a deployment guarantee.
The neural controller was trained and both arms were simulated on `cuda`. Rendering used the retained CSV trace.
The same frozen policy, initial condition, dynamics, and sensor-error tape are used in both arms.
The RTA arm adds an innovation-based uncertainty radius, a pairwise CBF-style velocity projection, and a conservative fallback using an independent bounded-error proximity channel.
The displayed unsafe-separation threshold is 0.78 m; collision is defined as separation below 0.48 m, and formation breakdown as mean target error above 1.05 m.

| arm | unsafe-separation steps | collision steps | minimum separation | formation-breakdown steps | mean formation error | filter steps | fallback steps |
|---|---:|---:|---:|---:|---:|---:|---:|
| LEC only | 87 | 0 | 0.544 | 0 | 0.440 | 0 | 0 |
| uncertainty-aware RTA | 0 | 0 | 1.293 | 0 | 0.407 | 101 | 14 |

Interpretation boundary: this run illustrates an architectural claim, not a theorem. A CBF-style filter cannot repair unmodeled state-estimation error by itself; its result remains conditional on the uncertainty bound, local sensing model, plant model, and fallback feasibility.

The narrated MP4 carries the same on-screen captions as the silent version. Its exact voice-over text and generated WAV are retained beside the movies; the narration adds no analytical claims beyond the logged trace and this report.
