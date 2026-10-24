# GPU Affine-Formation Conditioning Experiment

This is a descriptive, paired CUDA experiment using an illustrative first-order multi-agent model. It is not flight data, a reproduction of Liu et al., or a formal safety proof.

- Device: `NVIDIA GeForce RTX 5070 Ti Laptop GPU`; paired episodes per arm: 4096; steps per episode: 240.
- The design-time PSD stress matrix has rank 9 = n-d-1 and annihilates the nominal configuration.
- Tracking/separation arm: 1101/4096 hidden-collapse episodes and 2304/4096 rank-loss episodes.
- Conditioning-RTA arm: 0/4096 hidden-collapse episodes and 0/4096 rank-loss episodes.
- Representative paired episode: 2335; retained rows: 5760.

A hidden collapse is registered descriptively when leader conditioning falls below 0.04, pairwise separation remains at or above 0.22, and tracking error during rank loss remains below 0.18. The RTA rejects a proposed map when its normalized leader conditioning minus the registered uncertainty margin would fall below 0.12, then holds the last admissible map.

Interpretation boundary: the result demonstrates that tracking error and pairwise separation are not logically equivalent to preservation of affine rank in this model. It does not establish that this monitor or threshold is sufficient for a physical formation-control deployment.