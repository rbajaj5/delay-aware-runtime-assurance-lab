# Part A - Zero-compute mechanism checks

## Scope

This report uses only existing Phase 2b and pilot-diagnostic artifacts. It
contains no new simulation runs and no hypothesis tests. Part B remains gated.

## Pre-committed readings

- Rates close (N4 within ~±15% of N3's mean) AND rankings flipped → the one-step fix is **scorer-side repricing**; the trigger channel did not materially move at the shallow end. The "accidental sweet spot / trigger neither too quiet nor too loud" hypothesis is rejected on existing data.
- Rates materially higher at N4 AND rankings unchanged where interventions occur → **trigger-mediated** rescue; repricing story rejected.
- Both moved → **entangled**; the 2×2 off-diagonals become the deciding evidence rather than confirmatory decoration.

## Input hash verification

| status | path | sha256 |
| --- | --- | --- |
| PASS | C:\Users\anaxe\RAGforAssurredAutonomy\analysis\out\horizon_spectrum_20260723\pilot_diagnostics\dx2_intervention_rates.csv | 4d6c3adc3b0e4b8568203aba529f4596aae1c74d86fb11ae6210a5a60801742d |
| PASS | C:\Users\anaxe\RAGforAssurredAutonomy\analysis\out\horizon_spectrum_20260723\phase2b\confirmation\episodes.csv | e0c6112eb3d7ac7d2d1247183aa2595ec3d1d015756af2393ee83b07537b0c5c |
| PASS | C:\Users\anaxe\RAGforAssurredAutonomy\analysis\out\horizon_spectrum_20260723\phase2b\confirmation\full_steps\E-N3_steps.csv | a30b8c773cd16bb0dcc2276dfd2fad01c3ed9629794e31d246f488ef8660e1f6 |
| PASS | C:\Users\anaxe\RAGforAssurredAutonomy\analysis\out\horizon_spectrum_20260723\phase2b\confirmation\full_steps\E-N4_steps.csv | 3eb5ddaef25938fce953c7a659bd3b6f63e506d6a4518f8092808f5031e36c59 |
| PASS | C:\Users\anaxe\RAGforAssurredAutonomy\analysis\out\horizon_spectrum_20260723\phase2b\onset_payloads\confirmation_E-N3_steps20_35_scores.csv | 0e1a17092a13d153e7274d5c4fd950b80726e78025fa928c8f822c63f0bed3af |
| PASS | C:\Users\anaxe\RAGforAssurredAutonomy\analysis\out\horizon_spectrum_20260723\phase2b\onset_payloads\confirmation_E-N4_steps20_35_scores.csv | 5220404657e4977fb997249bde1bd8a570805ec526edd3282c2e23e760c91d24 |
| PASS | C:\Users\anaxe\RAGforAssurredAutonomy\analysis\out\horizon_spectrum_20260723\phase2b\payload_selection.csv | 16b20da0b0fbf9ba6f8228aafab87db5fffecde7bd9facfa6f355695f52aeddd |

All Part A inputs matched their pre-existing manifest hashes before the data
were read.

## CHK1 - Intervention rates

| scope | arm | prediction_horizon | n_seeds | mean_interventions_per_100_steps | min_interventions_per_100_steps | max_interventions_per_100_steps | p50_interventions_per_100_steps | raw_file |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pilot | E-N10 | 10 | 200 | 95.8527948126 | 93.0555555556 | 97.4025974026 | 96 | C:\Users\anaxe\RAGforAssurredAutonomy\analysis\out\horizon_spectrum_20260723\pilot_diagnostics\dx2_intervention_rates.csv |
| pilot | E-N3 | 3 | 200 | 83.7745950469 | 51 | 98 | 89 | C:\Users\anaxe\RAGforAssurredAutonomy\analysis\out\horizon_spectrum_20260723\pilot_diagnostics\dx2_intervention_rates.csv |
| pilot | E-N4 | 4 | 200 | 77.115 | 53 | 95 | 78 | C:\Users\anaxe\RAGforAssurredAutonomy\analysis\out\horizon_spectrum_20260723\pilot_diagnostics\dx2_intervention_rates.csv |
| pilot | E-N5 | 5 | 200 | 74.355 | 50 | 96 | 72 | C:\Users\anaxe\RAGforAssurredAutonomy\analysis\out\horizon_spectrum_20260723\pilot_diagnostics\dx2_intervention_rates.csv |
| pilot | E-N7 | 7 | 200 | 94.894439761 | 86.1538461538 | 98 | 95 | C:\Users\anaxe\RAGforAssurredAutonomy\analysis\out\horizon_spectrum_20260723\pilot_diagnostics\dx2_intervention_rates.csv |
| confirmation | E-N3 | 3 | 1000 | 82.7306220886 | 51 | 98 | 89 | C:\Users\anaxe\RAGforAssurredAutonomy\analysis\out\horizon_spectrum_20260723\phase2b\confirmation\full_steps\E-N3_steps.csv |
| confirmation | E-N4 | 4 | 1000 | 77.274 | 51 | 96 | 73 | C:\Users\anaxe\RAGforAssurredAutonomy\analysis\out\horizon_spectrum_20260723\phase2b\confirmation\full_steps\E-N4_steps.csv |

Pilot E-N4 versus E-N3 mean-rate difference: -7.94942075593% relative
to E-N3. Confirmation E-N4 versus E-N3 mean-rate difference:
-6.59564977373% relative to E-N3. Both absolute differences are
within the pre-committed ±15% band: **TRUE**.

Raw sources: `C:\Users\anaxe\RAGforAssurredAutonomy\analysis\out\horizon_spectrum_20260723\pilot_diagnostics\dx2_intervention_rates.csv`, `C:\Users\anaxe\RAGforAssurredAutonomy\analysis\out\horizon_spectrum_20260723\phase2b\confirmation\full_steps\E-N3_steps.csv`, and
`C:\Users\anaxe\RAGforAssurredAutonomy\analysis\out\horizon_spectrum_20260723\phase2b\confirmation\full_steps\E-N4_steps.csv`. Derived table: `C:\Users\anaxe\RAGforAssurredAutonomy\analysis\out\horizon_spectrum_20260723\mechanism_checks_readonly_20260723\chk1_rates.csv`.

## CHK2 - E-N4 survivor payload rankings at drift onset

- Matched E-N4 survivor seeds: [61001, 61004, 61016, 61019, 61111, 61433, 61546].
- Evaluated seed-step pairs: 63 (seven seeds,
  steps 24-32).
- Full candidate rows: 504 (eight candidates per
  seed-step).
- N3-style inversion present under N3: 63/63.
- N3-style inversion present under N4: 49/63.
- Winner changed from N3 to N4: 34/63.
- `gentle_down` versus `diff_left` order flips: 4/63.
- `gentle_down` versus `theta_correct_b` order flips:
  10/63.
- Seed-step pairs with at least one specified pair-order flip:
  14/63.
- Largest weighted gap-change terms across the flipped pairs: `{"constraint_violation": 14}`.
- N3 winner counts: `{"gentle_down": 32, "hover": 31}`.
- N4 winner counts: `{"diff_left": 2, "diff_right": 6, "gentle_down": 11, "hover": 23, "ppo_clipped": 5, "theta_correct_a": 12, "theta_correct_b": 4}`.

The candidate-level CSV records both complete rankings, the four unweighted
score terms, their weighted contributions, both final scores, both winners,
the two specified pairwise gaps, order-flip flags, and the largest component
of each gap change for every seed-step.

Raw sources: `C:\Users\anaxe\RAGforAssurredAutonomy\analysis\out\horizon_spectrum_20260723\phase2b\onset_payloads\confirmation_E-N3_steps20_35_scores.csv`, `C:\Users\anaxe\RAGforAssurredAutonomy\analysis\out\horizon_spectrum_20260723\phase2b\onset_payloads\confirmation_E-N4_steps20_35_scores.csv`, and `C:\Users\anaxe\RAGforAssurredAutonomy\analysis\out\horizon_spectrum_20260723\phase2b\payload_selection.csv`.
Derived table: `C:\Users\anaxe\RAGforAssurredAutonomy\analysis\out\horizon_spectrum_20260723\mechanism_checks_readonly_20260723\chk2_payload_rankings.csv`.

## Mechanical adjudication

- Rates within the pre-committed ±15% band: **TRUE**.
- Rankings flipped: **TRUE**.
- No hypothesis test was computed.

CHK-VERDICT: REPRICING

