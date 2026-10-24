# Pre-registered protocol: horizon dose-response spectrum and AP-N3 confirmation

Frozen: 2026-07-23, before any Phase 2a pilot episode.

## Scope and chronology

This study tests whether extending the queue-aware scorer's prediction horizon
suppresses Arm E's IID uniform-delay out-of-bounds (OOB) tail and confirms the
historical candidate-projection artifact AP-N3 exactly as implemented.

Before this freeze, only the following horizon-study executions occurred:

- Phase 0 code-equivalence runs at N=3 on seeds 1340, 1373, and 1665. These
  reproduce retained trajectories and were used only to verify that
  parameterizing the horizon leaves the N=3 score, ranking, selected action,
  and terminal-RMSE calculation unchanged.
- Phase 0 compute-only timing runs at N=3 and N=10 on seeds 70000-70009. These
  generated endpoint outcomes as an unavoidable by-product of timing. The
  seeds and schedule hashes are excluded from all study blocks, and their
  outcomes were not used to alter the hypotheses, advancement rule, gates,
  interpretations, arms, or seed ranges.
- Historical AP-N3 executions on seeds 30000-30199, which motivated but do not
  enter the registered test.

No seed in the Phase 2a pilot range has been executed before this freeze.

## Plant, channel, endpoint, and fixed scorer

- Plant/task: the existing safe-control-gym 2D quadrotor trajectory-tracking
  harness.
- Episode horizon: 100 simulation steps.
- IID delay: each seed generates one schedule with values sampled uniformly
  from {0, 1, 2}; that schedule is matched across arms.
- Action conversion: the corrected physical-multiplier path already used by
  the current harness. No de-normalization, clipping, conversion, trigger, or
  queue-release rule is changed.
- Primary endpoint: OOB occurrence by the frozen 100-step endpoint.
- Secondary endpoints: survivor cumulative normalized violation severity,
  exposure-weighted per-step severity, intervention count, and runtime.
- Candidate library: unchanged.
- Velocity-gated pitch penalty: disabled in every arm.

The generalized queue-aware score is

J_N(c) = 100 max_{k in {1,...,N}} v(x_hat_{t+k})
       + 5 v_u(c)
       + 0.05 RMSE(x_hat_{t+N}, r_t)
       + 0.005 ||c||_2.

All scalar weights remain fixed at every N. The pending FIFO is rolled forward,
the candidate is inserted at release step t+d_t, and that candidate is held
through the remaining predicted steps. Extending N changes no release,
de-normalization, clipping, or conversion semantics.

## Arms

| Arm | Frozen definition |
| --- | --- |
| E-N3 | Existing delay-aware stack; N=3 anchor |
| E-N4 | Existing delay-aware stack; N=4 |
| E-N5 | Existing delay-aware stack; N=5 |
| E-N7 | Existing delay-aware stack; N=7 |
| E-N10 | Existing delay-aware stack; N=10 |
| D-N3 | Existing delay-blind stack; N=3 reproduction anchor, not hypothesis-tested |
| AP-N3 | Exact historical dual-branch candidate-projection artifact; N=3 |

## AP-N3 identity and sign-off amendment

AP-N3 is frozen to:

`C:\Users\anaxe\Documents\Codex\2026-05-05\master-codex-prompt-agentic-rag-with\outputs\uniform_thrust_provenance_ADDENDUM_bundle_20260720\run_followup_experiments_20260708.py`

SHA-256:
`9cf5740426f35b2a13d5e3dfcd1a270452c9be345b94f00fb335d9ed87d151f5`

Implementation: lines 143-207. The selection rule is:

```python
distance = float(np.linalg.norm(np.asarray(action, dtype=float).reshape(-1) - ppo))
feasible = float(details["predicted_state_severity"]) <= 1e-9
# Candidate-library proxy for action projection: prefer predicted-safe
# candidates closest to PPO; fall back to the normal severity score.
projection_score = (0.0 if feasible else 1_000_000.0 + score) + distance
scored.sort(key=lambda item: (float(item["score"]), str(item["name"])))
```

The feasible branch selects the predicted-safe library candidate nearest PPO.
When no candidate is predicted safe, the artifact falls back to the ordinary
delay-aware score plus distance after adding a common 1,000,000 offset. The
user explicitly approved confirmation of this dual-branch artifact with:

`SIGNOFF-AP-FALLBACK-AS-IMPLEMENTED`

Every AP-N3 intervention row in the 1,000-seed confirmation will log
`projection_feasible`, selected candidate, `raw_delay_aware_score`,
`projection_distance_to_ppo`, and `projection_score`. The descriptive
deliverable `ap_fallback_rate.csv` will report feasible-branch and fallback-
branch counts per seed. It has no separate hypothesis test.

## Seed blocks and schedule-hash disjointness

| Block | Seeds | N | Purpose |
| --- | ---: | ---: | --- |
| Pilot | 60000-60199 | 200 | E horizon spectrum and advancement |
| Confirmation | 61000-61999 | 1,000 | Registered HZ1 and HA1 tests |
| Gradual guardrail | 62000-62099 | 100 | Descriptive transfer check |

Before freeze, 7,328 distinct delay hashes were read from 90 retained episode
artifacts. Each proposed block contains one distinct hash per seed. Hash
overlap is zero between each proposed block and the retained hashes, and zero
between every pair of proposed blocks. Range arithmetic alone is not treated
as evidence of disjointness.

The guardrail uses the gradual-loss profile under corrected
`physical_multiplier` conversion. Its comparability target is the corrected
96/1000 uniform-pipeline gradual cell, not the canonical-wrapper 302/1000
fault-profile cell. It is descriptive only; it does not test a historical
cell.

## Phase 2a pilot

The pilot runs E-N3, E-N4, E-N5, E-N7, and E-N10 on all 200 matched pilot
seeds. It runs no hypothesis test.

### Pilot anchor gate G0

E-N3 must record 22-45 OOB episodes, inclusive, out of 200. This is the
Clopper-Pearson-consistent band for the established approximately 16% rate.
Any result outside the band is pipeline drift and halts the study before the
advancement rule.

The known 1,000-seed failure-family basin sizes are {72, 40, 31, 9, 6, 1}.
A 200-seed pilot therefore carries approximately 30 expected exits and may
miss small families. A pilot zero is suggestive rather than conclusive; the
confirmation block is required.

### Graded advancement rule

Apply these rules in order after G0 and all pilot structural gates pass:

1. **Zero boundary.** If any E-N* records 0/200, confirm the smallest such N
   and its immediate predecessor in the tested spectrum.
2. **Inflection boundary.** If no horizon reaches zero, define a clear
   inflection as E-N7 < 5/200 and E-N10 < 5/200 while E-N3, E-N4, and E-N5
   each exceed 15/200. Confirm E-N5, E-N7, and E-N10.
3. **Monotonic decrease, no threshold.** If neither prior rule fires and the
   counts are non-increasing over N={3,4,5,7,10}, with at least one strict
   decrease, confirm E-N7 and E-N10.
4. **Flat-spectrum residual.** If none of the prior mechanical rules fires,
   confirm E-N10 only. Defining this as the residual branch makes the rule
   total and prevents post-pilot selection for a nonmonotonic spectrum.

Every branch additionally confirms mandatory E-N3, AP-N3, and D-N3. The pilot
runner records the fired branch and resulting arm list mechanically, prints
`AWAITING_SIGNOFF`, and stops. No Phase 2b implementation or execution is
permitted until the user's message contains the exact string
`CONFIRM-[arm list]`.

## Phase 2b confirmation and guardrail

After exact arm-list sign-off:

1. Run the 1,000-seed confirmation block on the advancement arms plus E-N3,
   AP-N3, and D-N3.
2. Run the 100-seed gradual guardrail on every confirmed E horizon plus E-N3
   and AP-N3.
3. Label every gradual artifact with `pipeline=physical_multiplier`.
4. Apply all structural gates before reading registered endpoints.

## Registered hypotheses

| ID | Registered comparison | Test |
| --- | --- | --- |
| HZ1 | Largest confirmed E horizon has fewer IID OOB exits than E-N3 | Exact two-sided McNemar, alpha=0.025 |
| HA1 | AP-N3 has fewer IID OOB exits than E-N3 | Exact two-sided McNemar, alpha=0.025 |

HZ1's referent is defined in every advancement branch. The smallest sufficient
horizon, if observed, is descriptive rather than a separate test. HZ1 and HA1
are Bonferroni-adjusted co-primary hypotheses at alpha=0.05/2. D-N3 is a
reproduction check with full logging and no registered hypothesis test.

### AP-N3 interpretation amendment

Confirmation of AP-N3 confirms the dual-branch artifact as implemented, and
attributes nothing to the predicted-safe restriction in isolation.

## Pre-committed outcome interpretations

- **Boundary reached (some confirmed N at 0/1000):** horizon extension is **sufficient** to suppress the uniform-delay tail at the frozen endpoint, and the failing smaller N's are individually insufficient — i.e., horizon depth is the binding constraint for this scorer configuration. The manuscript's mechanism language upgrades to "short-horizon myopia in the tested scorer configuration," with the caveat that the score's terminal-RMSE step co-varies with N.
- **Partial reduction (improvement, no zero):** horizon depth interacts with base-score candidate pricing; neither alone is binding. Report as partial with mandatory family analysis; no causal upgrade.
- **Plateau (no material improvement by N=10):** horizon is not the binding constraint; the mechanism is base-score ranker mispricing. Because AP proves the existing 8-candidate library contains survival paths, the registered contingency is **score reformulation, not library expansion** (library changes would confound the diagnosis by moving library and score landscape simultaneously). The manuscript's "short-horizon myopia" language must be downgraded to "ranking blind spot at the tested horizon."

## Structural gates

All gates are hard assertions. A failure produces a descriptive stack trace and
halts; there is no fix-and-continue without explicit sign-off.

| Gate | Frozen check |
| --- | --- |
| G0 | Pilot E-N3 OOB count is in [22,45]/200 before advancement |
| G1 | Per-seed delay-schedule hash is identical across all arms in a block |
| G2 | Gate-only Arm A state-sequence hash equals archived `b74f787d735d04c6` |
| G3a | Confirmation E-N3 OOB rate is in [12%,22%] |
| G3b | The three largest known E-N3 basin signatures appear; report all six |
| G4 | Exactly one episode row per registered seed and arm |
| G5 | Executed seed set equals the registered range exactly |
| G6 | Every row echoes its requested seed; no duplicate or missing arm-seed pair |

G2 provenance:

- Archived table:
  `analysis/out/markov_robustness_arm_2026-07-06/markov_robustness_episodes.csv`
- Archived table SHA-256:
  `7631fcb674af1bae4cdc53795b2c55cb3347d50784933eba3b0d4418aa2ef06f`
- All 1,000 archived Arm A rows share state-sequence hash:
  `b74f787d735d04c6`

The six known E-N3 failure signatures are frozen as the pair
`terminal_state_hash_12 + applied_action_sequence_hash_12`:

| Family | Basin | Terminal hash | Applied-action hash | Required by G3b |
| --- | ---: | --- | --- | --- |
| F1 | 72 | `39e7c99b7933` | `2c6be4ccedec` | yes |
| F2 | 40 | `fb7d25581ddc` | `2c6fea9771d0` | yes |
| F3 | 31 | `1a99aa1f0676` | `d10f8cdbb9cc` | yes |
| F4 | 9 | `6b91763e8338` | `d83e5d15466c` | report |
| F5 | 6 | `0f7451715530` | `8b78cf3a2d47` | report |
| F6 | 1 | `3ced655a6ce0` | `dd47aace5543` | report |

## Logging and audit artifacts

- Retain per-step state vectors and applied actions for every arm, including
  D-N3.
- Capture candidate score payloads universally during execution for steps
  20-35, then retain one representative seed per observed trajectory family
  and four pseudorandom survivors per arm.
- Log one row per `(seed, step, candidate)` with the unweighted components
  `constraint_violation`, `action_violation`, `rmse`, `l2_penalty`, and
  `final_score`.
- A representative rerun is permitted only if universal payload capture fails;
  it must reproduce the original episode's 12-decimal state-sequence hash.
- Family keys use the 12-decimal terminal-state hash paired with the
  12-decimal applied-action-sequence hash.
- Produce Clopper-Pearson 95% intervals for every reported OOB count.
- Hash the preregistration and every output artifact in the study manifest.

The score fields added before freeze are logging-only. A repeated N=3
equivalence audit on seeds 1340, 1373, and 1665 compared 1,922 candidate rows:
the maximum score and terminal-RMSE deviations were zero, candidate rankings
and selected actions matched exactly, and no near-tie swap occurred.

## Frozen stopping rule

Phase 2a ends after the pilot gates and advancement branch are reported.
The final output must contain the literal line:

`AWAITING_SIGNOFF`

No pilot hypothesis tests are permitted. No confirmation code or run is
permitted before exact user arm-list sign-off.
