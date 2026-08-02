# Network-Science Diagnostic for the Altara Helicopter Trace

Source: `artifacts/20260802/helicopter_3d/helicopter_3d_step_trace.csv`
Source SHA-256: `3c9545e333b3faf783098bf68940305dbf7a0df512b482a2d4d2493a83e5cccc`

This is read-only postprocessing of the retained kinematic trace. Each frame is a temporal conflict graph: helicopters are nodes, and an edge joins two active helicopters assigned to the same pad within the replay's 2.0-unit separation threshold.

The metrics are descriptive network diagnostics, not a claim that a centrality or spectral quantity is a safety certificate. The network lens makes the cascade visible: a small number of stale clearance decisions can create connected conflict components that persist over several steps.

| policy | conflict steps | total conflict edges | peak component | triangle steps | most exposed |
|---|---:|---:|---:|---:|---|
| delay_blind | 132 | 846 | 3 | 115 | H1 (132 steps) |
| queue_aware | 115 | 115 | 2 | 0 | H1 (115 steps) |
