# Exploratory Dense Horizon-Spectrum Extension

**Status:** FROZEN BEFORE EXECUTION  
**Freeze date:** 2026-07-25  
**Authorization:** User request to run `N={5,6,8,9}` before revising the v6 mechanism figure.

## Purpose

The existing horizon program measured every integer prediction depth from
`N=3` through `N=10` except `N=6`, `N=8`, and `N=9`; `N=5` had only a
200-seed pilot result. This extension fills those four cells on the same
1,000 matched schedules used for the registered `N=3`/`N=4` comparison and
the exploratory `N=7` scale-up.

This is a post hoc, exploratory characterization. It runs no hypothesis test
and does not retroactively change the evidence class of any registered result.

## Frozen Design

- Arms: coupled queue-aware monitor with shared trigger and scorer horizon
  `N in {5,6,8,9}`.
- Seeds: `61000-61999`, inclusive.
- Episodes per arm: 1,000.
- Episode length: 100 simulator steps.
- Delay mode: IID uniform delay in `{0,1,2}`.
- Disturbance mode: `none`.
- Action-conversion setting: corrected path enabled, although the no-fault
  block does not invoke thrust-effectiveness conversion.
- Candidate library, score weights, warning buffer, PPO checkpoint, plant,
  controller, and queue semantics: unchanged from the frozen horizon study.
- Harness: `tools/quadrotor_delay_ra_experiment.py`, hash-pinned in the
  companion freeze manifest.

The prediction depth `N` remains a shared algorithm parameter. It controls
both the trigger rollout and all eight candidate-scoring rollouts. Physical
delay remains sampled from `{0,1,2}`.

## Endpoints

- Primary descriptive endpoint: OOB occurrence by the 100-step endpoint.
- Clopper-Pearson 95% interval for each OOB rate.
- Secondary descriptive endpoints: survivor cumulative normalized violation
  severity, exposure-weighted per-step severity, intervention count, and
  runtime.
- Mechanism diagnostics at intervention steps:
  feasible candidate support, saturated-step indicator, normalized top-two
  score margin, selected-candidate distribution, `gentle_down` share, and
  per-seed argmin churn.

## Retained Artifacts

For every arm:

1. One episode row per seed.
2. Full per-step state/action/queue log with the bulky candidate payload
   removed.
3. Per-step discrimination diagnostics.
4. Full eight-candidate decomposed scores for steps 20-35.
5. Endpoint and mechanism summaries.

## Hard Gates

1. A five-seed `N=3` reproduction must match the retained registered `N=3`
   state-sequence hash, applied-action-sequence hash, delay hash, OOB status,
   and first-OOB step exactly.
2. Every new arm must contain exactly the registered seeds `61000-61999`,
   with no duplicate or missing arm-seed pair and exact requested-seed echo.
3. The delay-schedule hash for each seed must match across all four new arms
   and the retained registered `N=3` arm.
4. Every row must report uniform delay, no disturbance, and the coupled
   queue-aware monitor at the requested prediction horizon.
5. OOB status and first-OOB step independently recomputed from retained step
   logs must match the episode table.
6. Any gate failure halts reporting; no fix-and-continue is permitted.

## Reporting Boundaries

- New `N=5`, `N=6`, `N=8`, and `N=9` results are labeled exploratory
  `(n=1000)`.
- Existing `N=3` and `N=4` results remain registered `(n=1000)`.
- Existing `N=7` remains exploratory `(n=1000)`.
- Existing `N=10` remains pilot `(n=200)` unless separately scaled later.
- No monotonicity theorem or continuous interpolation is claimed. The dense
  integer sweep characterizes this implementation on the tested schedule
  distribution and endpoint.

