# Delay / Random-Matrix Loop Diagnostic

Conceptual source: [Bourgade and Huang, *Loop Equations Characterize Random Matrix Statistics*](https://arxiv.org/abs/2607.07617). The source proves uniqueness results for Sine_beta and Airy_beta point processes from loop-equation hierarchies and motivates checking approximate loop equations as a route to universality.

## Operational translation

At each round, the experiment forms a 24-by-24 receipt-age matrix on a ring. It is symmetrized, centered, and scaled by 1/sqrt(N). The spectral-edge proxy is the largest eigenvalue. For z=2+0.5i, the reported first residual is |m(z)^2 + z m(z) + 1|, using m(z)=N^-1 Tr((H-zI)^-1), the semicircle loop-equation form as a diagnostic reference.

The blind policy uses the latest delivered receipt. The queue-aware policy overlays the newest pending payload before updating. This is a toy coordination model: the matrix is not assumed to be Wigner, no Sine_beta/Airy_beta limit is claimed, and the residual is not a safety certificate. The diagnostic asks whether receipt-age structure changes the spectral summaries that a later, larger fleet study could test more seriously.

## Connection to the attached excerpt

The supplied excerpt gives two useful controls for this interpretation: a Gronwall-type exponential envelope for a growth quantity, and a single-entry resolvent-stability estimate of the form C N^-1/2 Lambda^3 under explicit bounds on resolvent entries and matrix entries. We use those ideas to justify tracking both a growth/disagreement trace and the resolvent observable. We do not claim that the theorem hypotheses hold for this receipt-age matrix. The next rigorous extension would perturb one age edge at a time, measure the resulting change in m(z), and check the stated assumptions before comparing against a stability bound.

This run performs that perturbation check descriptively. For each nondegenerate age matrix, the largest off-diagonal age edge is increased symmetrically by 0.05/sqrt(N). `single_entry_delta_m` is the observed change in m(z); `stability_proxy` is N^-1/2 Lambda^3 with the constant omitted. The ratio is not a pass/fail theorem test because the source bound has an unspecified constant and our matrices are not sampled from the source ensembles.

The derivative identity is also checked numerically. For the selected off-diagonal edge, the central finite difference of m(z) is compared with `-(2/N)(G^2)ij`; the reported relative error is a calculus check, not a physical metric.

| delay | policy | games | spectral edge | loop residual | receipt age | delta m | stability proxy | delta/proxy | derivative rel. error | disagreement | edge exceedance | state overshoot |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | blind_delay | 30 | 0 | 0.23529411764705885 | 0 | 0 | 0 | 0 | 0 | 0.35939583411626536 | 0 | 0 |
| 0 | queue_aware | 30 | 0 | 0.23529411764705885 | 0 | 0 | 0 | 0 | 0 | 0.35939583411626536 | 0 | 0 |
| 1 | blind_delay | 30 | 0 | 0.23529411764705885 | 0 | 0 | 0 | 0 | 0 | 0.35908129784286374 | 0 | 0 |
| 1 | queue_aware | 30 | 0 | 0.23529411764705885 | 0 | 0 | 0 | 0 | 0 | 0.35908129784286374 | 0 | 0 |
| 2 | blind_delay | 30 | 1.4543641155891442 | 0.04963164936503011 | 0.9944444444444438 | 0.00026890288239395381 | 0.046138066454225482 | 0.0057958427399004055 | 8.5143283384102013e-05 | 0.25336478724078482 | 0 | 0 |
| 2 | queue_aware | 30 | 0 | 0.23529411764705885 | 0 | 0 | 0 | 0 | 0 | 0.35936662403234881 | 0 | 0 |
| 3 | blind_delay | 30 | 1.4543641155891442 | 0.04963164936503011 | 1.9833333333333341 | 0.00026890288239395381 | 0.046138066454225482 | 0.0057958427399004055 | 8.5143283384102013e-05 | 0.23003592365380307 | 0 | 0 |
| 3 | queue_aware | 30 | 0 | 0.23529411764705885 | 0 | 0 | 0 | 0 | 0 | 0.35850053961228456 | 0 | 0 |
| 5 | blind_delay | 30 | 1.4543641155891442 | 0.04963164936503011 | 3.9444444444444433 | 0.00026890288239395381 | 0.046138066454225482 | 0.0057958427399004055 | 8.5143283384102013e-05 | 0.23455784945646255 | 0 | 0 |
| 5 | queue_aware | 30 | 0 | 0.23529411764705885 | 0 | 0 | 0 | 0 | 0 | 0.35694758622689043 | 0 | 0 |

Interpretation is deliberately limited: a difference in the spectral proxy would motivate a larger ensemble study; it does not establish random-matrix universality or a causal safety mechanism.
