# Diagram Audit: Delayed Multi-Agent Hex

This audit applies the Jaffe–Liu picture-language principle: each picture is
an abstraction of a defined object, and each simulation picture must be
reversible to logged state or an explicitly stated derived metric.

## Diagram Inventory

| Diagram | What it shows | Evidence class | Audit judgment |
|---|---|---|---|
| `hex_delay_stale_conflicts.png` | Mean stale-cell conflicts versus delay for blind and queue-aware policies | Exploratory simulation summary | Strongest quantitative diagram. Delay 1 is correctly zero after recipient-order correction; blind conflicts begin at delay 2. |
| `hex_policy_comparison_delay3_seed0_poster.png` | Authoritative and local boards for the same delay-3 trace | Logged trace | Good mechanism picture. Red outlines identify logged stale selections; pending count is logged. |
| `hex_hopf_delay_simulation.png` | Hidden fiber-phase error and stale conflicts under selective holds | Exploratory toy dynamics | Useful physics-inspired comparison, but not a physical Hopf-fibration claim. |
| `hex_hopf_adversarial_sweep.png` | Sensitivity to selective-hold strength 0, 1, and 2 | Exploratory parameter sweep | Separates the queue effect from one adversary setting. Keep the `adversarial` label visible. |
| `hex_multiagent_delay_cascade.png` | Four-agent fleet conflicts and local false/missed red-line triggers | Exploratory multi-agent simulation | Directly supports the delayed-state coordination story. The red-line threshold is toy and must stay in the caption. |
| `hex_hopf_topology_bridge.png` | Standard Hopf-map picture beside a completed Hex crossing | Mathematical visualization/analogy | Clear if read as a topology bridge. It must not be presented as validation of the external unified-field interpretation. |
| `hopf_model_selection_metrics.png` | Spline edge surrogate versus MLP on smooth phase and discrete conflict tasks | Exploratory model-selection check | Correctly labels the spline as a surrogate, not a full KAN. The negative result is more informative than a decorative architecture swap. |

## Picture-Language Semantics

- Blue and yellow cells always denote the two Hex players.
- Orange always denotes blind/local-state behavior in summary plots.
- Blue always denotes queue-aware behavior in summary plots.
- A green dot denotes a global crossing-path cell, not a safe-state marker.
- A red outline denotes a stale-cell conflict against the authoritative board.
- `T`, `L_a`, `Q_a`, `O_a`, `R_a`, `F_a`, `M_a`, and `C` are defined in
  `PICTURE_LANGUAGE_SPEC.md`.

## Viewer-Risk Checks

1. The Hopf bridge is explicitly labeled a visual analogy and not a v6 result.
2. The multi-agent red-line metric states its path-cost threshold in the footer
   and report.
3. The KAN-inspired model is not called a KAN in the figure title.
4. The corrected plain Hex summary must be used; the earlier pre-correction
   output with delay-1 conflicts is superseded by the regenerated manifest.
5. None of these diagrams should be merged into registered v6 evidence panels.

## Recommended Presentation Order

1. `hex_policy_comparison_delay3_seed0_poster.png`: local board versus
   authoritative board.
2. `hex_delay_stale_conflicts.png`: delay sweep and queue effect.
3. `hex_multiagent_delay_cascade.png`: same mechanism with four agents.
4. `hex_hopf_delay_simulation.png`: hidden fiber state as an optional research
   extension.
5. `hex_hopf_topology_bridge.png`: optional mathematical afterword, not an
   empirical result.

All outputs remain exploratory and separate from the registered quadrotor
manuscript claims.
