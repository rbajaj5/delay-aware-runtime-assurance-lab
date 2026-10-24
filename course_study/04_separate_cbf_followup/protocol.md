# Pre-Registered Protocol: V6 Formal CBF-QP Shield

**Frozen:** 2026-07-24, before adapter implementation, pilot execution, or
any V6 closed-loop episode.

**Authorization:** `AUTHORIZE-V6-FREEZE`.

**Primary barrier contraction parameter:** gamma = 0.20 for both F-blind and
F-comp.

## 1. Purpose and scope

This study tests whether replacing the finite eight-candidate heuristic
selector with a model-relative control-barrier-function quadratic program
(CBF-QP) improves safety under the same delayed 2D quadrotor harness. It does
not assume that delay compensation is protective. The formal variants are
tested as implementations whose model, queue semantics, action conversion,
solver feasibility, fallback behavior, and realized full-state outcomes are
all auditable.

The frozen v5 harness and all v5 artifacts remain untouched. Shield work is
confined to an adapter copy whose source is hash-pinned before execution.
This protocol authorizes no pilot by itself; the pilot remains gated on a
separate token.

## 2. Model-relative safety construction

The state is `[x, vx, z, vz, theta, theta_dot]`. The registered state box is:

| State | Lower | Upper |
|---|---:|---:|
| x | -2.0 | 2.0 |
| vx | -1.0 | 1.0 |
| z | 0.0 | 2.0 |
| vz | -1.0 | 1.0 |
| theta | -0.2 | 0.2 |
| theta_dot | -1.5 | 1.5 |

For every lower and upper box face, the discrete barrier condition is

`h_j(x_(k+1)) >= (1 - gamma) h_j(x_k)`, for j = 1,...,12.

The registered primary value is `gamma = 0.20`, so the next modeled margin
must retain at least 80% of the current margin. The action is selected by

`minimize ||u - u_PPO||_2^2`

subject to the twelve locally linearized DCBF inequalities and the physical
two-motor thrust bounds. The one-step map is the safe-control-gym CasADi
CVODES map documented in `v6_design.md`; the action Jacobian uses a
deterministic finite difference of `1e-6 N`.

The certificate is model-relative. It is not called a proof about the
PyBullet trajectory. Predicted-safe/realized-unsafe events are registered
disclosures.

## 3. Formal arms and comparators

| Arm | Frozen definition |
|---|---|
| F-blind | Evaluate the DCBF at the current logged state and ignore pending FIFO commands. |
| F-comp | Reuse the existing FIFO release helper, propagate through pending commands to the first controllable step, then enforce the DCBF at that queue-anchored state. |
| E-N4 | Registered repaired finite-library comparator with four-step shared trigger/scorer horizon. |
| AP-N3 | Exact dual-branch action-projection artifact as previously confirmed; endpoint/horizon qualification retained. |
| D-N3 | Delay-blind finite-library comparator and regression anchor. |
| B | Delayed PPO without monitor intervention. |

Both formal arms use gamma = 0.20. The optional `F-blind@gamma=0.35` pilot is
not included in this frozen protocol. Gamma 0.35 remains a labeled
logged-state sweep result only.

## 4. Gamma choice fixed before closed-loop data

The logged-state sweep evaluated gamma in {0.10, 0.20, 0.35, 0.60} on
F-blind and F-comp across 175 registered E-N3 fatal onsets, 200 registered
E-N4 survivor states, and 200 exploratory `(trigger=3, scorer=10)` saturated
states.

Gamma 0.20 is selected because it lies in the observed dissociation window:

- At fatal onsets, F-comp is hard-QP and nonlinear-DCBF infeasible on 175/175,
  while F-blind is feasible on 144/175.
- Gamma 0.10 makes both variants infeasible on 175/175 and therefore removes
  the desired delay-compensation contrast.
- At gamma 0.35, F-comp remains infeasible on 175/175 and F-blind becomes
  feasible on 175/175, but survivor hard feasibility is effectively flat
  relative to gamma 0.20: F-comp is 160/200 at 0.20 and 161/200 at 0.35;
  F-blind is 158/200 at both.
- Gamma 0.20 imposes the stronger contraction of those two values and avoids
  choosing the weaker barrier merely to increase feasibility.

The model-relative emptiness finding is frozen as follows:

> No admissible bounded action satisfies the queue-anchored nonlinear DCBF at
> any of the 175 fatal-onset states for gamma <= 0.35 under the frozen
> numerical feasibility procedure. At gamma = 0.60, 156/175 remain
> infeasible and 19/175 are feasible.

F-blind fatal-state feasibility is reported separately as delay-blind
certificate optimism; it is not averaged with F-comp.

At the primary gamma, linearized and nonlinear feasibility agree on
1150/1150 evaluations. Across the complete sweep, they agree on 4598/4600:
one optimistic and one conservative disagreement, each 1/4600. The complete
four-cell cross-tab remains the source of record. The operational adapter
uses the linearized hard-QP result and its preregistered fallback; the
nonlinear census is a model-level audit.

## 5. Infeasibility fallback

Every intervention first attempts the hard OSQP CBF-QP. If it is infeasible,
the sole fallback is the preregistered SciPy HiGHS relaxation:

`minimize sum(s) + 1e-4 ||u - u_PPO||_1`

subject to `G u + s >= b`, `s >= 0`, and physical actuator bounds. There is
no silent hover, clipped-PPO, or alternate branch. Every infeasible solve
must produce exactly one fallback-disposition row with per-constraint slack.

The zero-compute census shows that hard infeasibility is normal rather than
exceptional: among F-comp survivor states it ranges from 27/200 (13.5%) to
55/200 (27.5%) across the tested gammas. Infeasibility and fallback
engagement are therefore ordinary operating outcomes and are not themselves
classified as safety failures.

## 6. Blocks, seeds, and matching

All episodes use 100 command steps, fixed initial-state semantics, the frozen
PPO checkpoint, and matched IID uniform delays drawn from `{0,1,2}`.

| Block | Seeds | n | Arms | Fault semantics |
|---|---:|---:|---|---|
| Gated pilot | 72000-72199 | 200 | F-blind, F-comp, E-N4, AP-N3, D-N3, B | nominal, uniform IID delay |
| Confirmation primary | 73000-73999 | 1000 | same six arms | nominal, uniform IID delay |
| Gradual guardrail | 73000-73999 | 1000 | same six arms | `physical_multiplier`; eta decreases linearly from 1.0 to 0.70 over steps 0-99 |
| Persistent guardrail | 73000-73999 | 1000 | same six arms | `physical_multiplier`; eta = 0.70 from command step 35 onward |

The pilot and confirmation ranges were not found in the retained
preregistration or manifest text at freeze. Before the first episode in each
range, the generator must create the complete delay-hash inventory and assert
zero overlap with every retained prior delay hash, including compute-only
timing seeds 70000-70009. Within each block, every seed must have one identical
delay schedule and hash across all six arms.

## 7. Endpoints and registered tests

### HF1: Uniform endpoint matching criterion (primary)

F-comp must record `0/1000` OOB exits and zero F-comp-only OOB discordances
against E-N4. This strict operational criterion is called endpoint matching,
not a claim that a nonsignificant superiority test proves equivalence.
Clopper-Pearson 95% intervals and the paired discordance table are reported.

### HF2: Full-state reverse-velocity superiority (co-primary)

For the event `min_t(vx_t) < -1.0`, compare F-comp with E-N4 by exact
one-sided McNemar at alpha = 0.025, with the alternative that F-comp has fewer
crossings. Report the paired risk difference, its paired bootstrap 95%
interval using 10000 seed resamples and RNG seed 20260724, and the complete
distribution of episode minimum vx. HF2 is supported only when the exact
one-sided p-value is below 0.025 and the observed discordance direction favors
F-comp.

### HF3: Delay compensation at the formal layer (secondary)

Compare F-comp with F-blind on uniform OOB occurrence using exact two-sided
McNemar at alpha = 0.05. Report reverse-bound crossings, minimum-vx
distributions, and family inventories descriptively. No directional
assumption is registered.

### Guardrail readings

Gradual and persistent blocks report OOB, all twelve full-state crossings,
fallback rate, model mismatch, and solver timing for every arm. They are
pipeline-labeled transfer censuses, not additional primary hypothesis tests.
The verdict cannot be "the pivot was right" unless F-comp meets HF1, has a
cleaner full-state census under HF2, has bounded and fully disclosed fallback
behavior, and transfers without a new material failure surface in both
guardrails. Any missing component yields a predeclared partial verdict.

## 8. Registered disclosures

The following are reported for every arm and block regardless of outcome:

1. OOB count/rate with Clopper-Pearson 95% interval; first-exit distribution;
   cumulative and per-step severity with explicit right-censoring warning.
2. All twelve state-bound crossing counts, episode minima/maxima, and first
   crossing steps.
3. Hard-QP infeasibility and fallback engagements per 100 executed steps;
   per-constraint slack sum/max and active/binding faces.
4. Solver wall time, solver-reported time, iterations, and fractions exceeding
   5, 10, and 20 ms.
5. PPO command, QP command, pre/post-conversion physical commands, FIFO input,
   released simulator action, and all conversion deltas.
6. Model-predicted next state/barriers, realized next state/barriers,
   prediction error, and every predicted-safe/realized-unsafe event.
7. Intervention and fallback family inventories using terminal-state plus
   full applied-action-sequence hashes at 12-decimal rounding.

### Rider 3: vx_upper slack as a falsifiable early-warning disclosure

For each arm and block define:

- `tau_slack`: first step with hard-QP infeasibility and relaxed
  `vx_upper` slack greater than `1e-7`;
- `tau_cross`: first realized step with `vx > 1.0`;
- lead time: `tau_cross - tau_slack`.

The early-warning hypothesis is supported only if at least 90% of episodes
with a later `vx_upper` crossing have `tau_slack < tau_cross` and the median
lead time among warned crossings is at least one step. Otherwise it is not
supported. Report warned crossings, missed crossings, false warnings among
episodes without a crossing, lead-time distribution, and arm/block
denominators. This is a registered disclosure, not a co-primary test.

## 9. Hard gates

1. **RG:** before any F episode, the adapter reproduces E-N3, E-N4, AP-N3,
   and D-N3 on seeds 61000-61004 against retained artifacts, including row
   count, delay schedule/hash, flags, selected candidate, commands, released
   actions, 12-decimal state sequences/hashes, terminal result, and OOB.
2. **Seed/hash:** executed seed sets equal the registered ranges; output rows
   echo seeds without duplicates/gaps; delay hashes are matched across arms
   and disjoint from every prior retained hash.
3. **Rows:** episode and required step-row counts are exact per arm.
4. **No silent fallback:** every hard-infeasible solve has exactly one logged
   fallback row and every hard-feasible solve has none.
5. **Conversion:** every logged post-conversion command recomputes under the
   registered pipeline; mismatch halts.
6. **Certificate accounting:** every claimed-safe prediction has a realized
   next-state comparison or an explicit queue-offset disposition.
7. **Comparator anchors:** E-N4, AP-N3, and D-N3 run through their frozen
   adapter paths without monitor-logic changes.

Any failed gate halts execution with a descriptive assertion. No fix-and-
continue is permitted without a new authorization.

## 10. Logging and deliverables

Full per-step state and applied-action logs are retained for every arm.
Formal-arm intervention rows additionally retain variant, gamma, hard-QP
status, fallback disposition, active constraints, all twelve slack values,
solve timing/iterations, queue before/after, sampled delay/release indices,
all command representations, predicted and realized states/barriers, and
prediction errors.

Required outputs are episode and step CSVs; paired-outcome and exact-test
files; full-state census; infeasibility/fallback ledger; solver-timing table;
model-mismatch ledger; vx_upper early-warning census; family inventories for
every arm with exits; runtime configuration; gate report; and a SHA-256
manifest covering source, protocol, and results.

## 11. Phase boundary

This frozen document does not authorize adapter implementation or a pilot.
The 200-seed pilot remains gated on its own exact authorization token. The
confirmation and both 1000-seed guardrails require a later explicit token
after the pilot package is audited.

## 12. Frozen source identities

| Artifact | SHA-256 |
|---|---|
| V6-0 design | `7cae8fff093d1d2cc5e4493bb83b094ee70776af7162817277e3b160894741c3` |
| V6-0 feasibility CSV | `71fac7dffb81ae5424f26e35085e4298716da5fb439a3b9bb35f6e8946db7885` |
| V6-0 manifest | `26966b596b7013605903b61497993458ba117ed1a57dd4bad997aee278477d00` |
| RIDER-1 report | `1b736b52e3401af1a3fe1b71d9fb965209696d71efce3cec496cd949bc19c1c1` |
| Gamma sweep | `d6bd88f41ee89dc5a307fde293da8c12e1676636e35fdf7a6db6ac82248176a6` |
| Feasibility cross-tab | `e554fd866dcbab661bde66d004c71c9b0def4de50238c1351b26566fd85c4c87` |
| RIDER-1 manifest | `0c267e1147aea5d3d4abb433da5b5a1c8226d8d2974e04c841b6b2e038fd714a` |
| All-set binding-slack census | `{CENSUS_SHA256_INSERTED_IN_FREEZE_MANIFEST}` |

The all-set census is frozen beside this document and its exact hash is
recorded in the freeze manifest and JSON metadata. The placeholder above is
intentional to avoid self-referential document rewriting after freeze.

The V6-0 manifest records the design document's CRLF byte representation as
`6516a8c0493a7fe0147fcf5d578655856b31ed5c11d706cba2434526520a3539`.
The current LF representation has the hash shown in this table; conversion
back to CRLF reproduces the manifested hash exactly. This is a line-ending
normalization only, not a textual-content change.
