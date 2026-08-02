# Prime-Labeled Conflict-Graph Fingerprint

Source: `artifacts/20260802/helicopter_3d/helicopter_3d_step_trace.csv`
Source SHA-256: `3c9545e333b3faf783098bf68940305dbf7a0df512b482a2d4d2493a83e5cccc`

This is deterministic read-only postprocessing of the retained trace. It is an optional serialization aid, not a safety certificate, causal model, or complexity result.

Assign agent H1-H8 the first eight primes `2, 3, 5, 7, 11, 13, 17, 19`. For a conflict edge (i,j), define the multiplicative edge code `q_ij = p_i * p_j`. At step t:

`A_t = sum_(i,j in E_t) q_ij` is the additive conflict code; `N_t = A_t / max(1, C(|V_t|, 2))` is the normalized division form; and `P_t = product_(i,j in E_t) q_ij^(r_ij(t))` is the persistence form, where r_ij(t) is the current consecutive edge run length. The CSV stores log10(P_t) to avoid overflow.

The policy contrast is the subtraction `Delta_t = N_t(delay_blind) - N_t(queue_aware)`. In this trace its largest value occurs at step 66 and equals 13.297619. This arithmetic is a reproducible fingerprint of graph structure; the substantive findings remain the conflict counts and throughput reported in the main paper.

The prime labels are intentionally kept out of the paper's primary claim. Replacing them with another injective node-label scheme would preserve the underlying graph comparison.
