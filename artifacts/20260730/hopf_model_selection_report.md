# Hopf/Hex Model Selection Check

The smooth task predicts the next hidden Hopf-fiber phase. The discrete task predicts stale-cell conflicts from retained exploratory Hex trace rows. The spline model is an additive triangular edge-function surrogate inspired by the KAN parameterization; it is not presented as a full KAN implementation.

| task | model | metric | value | fit seconds |
|---|---|---|---:|---:|
| smooth_hopf_phase | spline_edge_surrogate | circular_mae_radians | 0.17155529002940209 | 0.0046582000213675201 |
| smooth_hopf_phase | mlp | circular_mae_radians | 0.0056308630232179456 | 3.0603804999846034 |
| discrete_stale_conflict | spline_edge_surrogate | accuracy | 0.88465298142717497 | 0 |
| discrete_stale_conflict | spline_edge_surrogate | positive_recall | 0.70078740157480313 | 0 |
| discrete_stale_conflict | mlp | accuracy | 0.88465298142717497 | 1.9469265999796335 |
| discrete_stale_conflict | mlp | positive_recall | 0.72440944881889768 | 1.9469265999796335 |

The result is a model-selection diagnostic, not evidence that KANs improve runtime assurance. It tests the paper's central practical caution: a smooth function-space representation can be useful on a smooth phase task while offering no automatic solution to discrete queue conflicts.
