# Basketball Affine-Formation Conditioning Simulation

Illustrative deterministic programmatic simulation; not real basketball evidence or proof of betting profitability.

- CUDA device: `NVIDIA GeForce RTX 5070 Ti Laptop GPU`.
- Matched episodes per arm: 4096; deterministic seed: 20260812.
- Video possession: `POS-AFFINE-2082`; coordinate rows: 3000.
- Tracking/separation arm: 1220/4096 rank-loss episodes and 2992/4096 hidden-collapse episodes.
- Conditioning-RTA arm: 0/4096 rank-loss episodes and 0/4096 hidden-collapse episodes.

The followers use fixed design-time affine weights relative to the three role-selected leaders. The left controller gates only on prior-frame tracking RMSE and pairwise spacing. The right controller applies the 0.12 threshold to the proposed conditioning minus a disclosed 0.01 uncertainty allowance, returns toward its most recent fallback checkpoint at or above 0.18, and resumes at 0.18 to avoid threshold chatter and inertial undershoot.