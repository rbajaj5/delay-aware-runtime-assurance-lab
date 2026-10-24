# V6 Pilot One-Sided McNemar Tail Erratum

Authorization: `AUTHORIZE-V6-PVALUE-TAIL-FIX`

Date: 2026-07-24

## Scope

This erratum is deposited alongside, not inside, the frozen pilot bundle:

`analysis/out/v6_cbf_qp_pilot_20260724_rerun2/`

The frozen pilot runner and payload were not edited. Its manifest remains:

- `PILOT_MANIFEST.sha256`
- SHA-256: `35376ae6029ac6bcba23a6feef34618c42f3d7b85562aa62d28d5097d665a5a9`
- Entries reverified after this fix: 43/43

## Superseded value

The following field in the frozen pilot output is superseded:

- File: `analysis/out/v6_cbf_qp_pilot_20260724_rerun2/paired_outcomes_and_tests.json`
- File SHA-256: `c0f6fd51ad97e9cb7a1270986dbbc8d308b471e1596a3afe9e2b5f4e8b3969bb`
- Field: `F-comp_vs_E-N4_reverse_vx.exact_one_sided_mcnemar_p_alternative_Fcomp_fewer`
- Emitted value: `1.0`
- Discordant table: F-comp only = 0; E-N4 only = 6
- Correct directed alternative: F-comp has fewer reverse-velocity crossings
- Correct exact lower-tail p-value: `0.015625`
- Exact fraction: `1/64`

The source paired table is unchanged and was correct. The defect affected only
the one-sided tail extraction.

## Correction

The frozen runner's helper at
`analysis/run_v6_cbf_qp_pilot.py:320-336` used the upper tail for every
one-sided request. It remains unchanged to preserve the frozen bundle.

The corrected shared implementation is:

- `analysis/v6_exact_tests.py:27-58`: explicit lower/upper directional tails
- `analysis/v6_exact_tests.py:78-100`: exact two-sided test
- SHA-256: `b9c16860b134735b61904ae9ae1c0d24ba306b2af280db125e2f4ee84853bb70`

The exact source-level change is recorded in `pvalue_tail_fix.diff`.
Any V6 confirmation runner must import the corrected shared helper before its
freeze. This erratum does not authorize a confirmation freeze or launch.

## Regression anchors

The unit tests at `tests/test_v6_exact_tests.py:26-85` verify:

1. The 0-vs-6 lower tail is exactly `1/64 = 0.015625`.
2. The archived HZ1 175-vs-0 two-sided test reproduces
   `4.176194859519056e-53` to full floating-point precision.
3. For a symmetric 5-vs-5 table, each inclusive one-sided tail is
   `319/512`, while the doubled-smaller-tail two-sided value is `1`.

Test file SHA-256:
`8c1d6bde39c6c8f7be67f429cf0f0576423625637b93432d115507f8352338dc`

Archived HZ1 source:

- `analysis/out/horizon_spectrum_20260723/phase2b/horizon_ablation_paired_tests.json`
- SHA-256: `b311598a53b5c792a48a4435b4c85e1810ca784ddcb57d91e56dbe7f23ecde25`

Raw unit-test output is retained in `unit_test_output.txt`.

## Status

The one-sided helper defect is corrected outside the frozen pilot bundle.
All upper-tail p-values emitted by the affected pilot helper are superseded
where their registered alternative requires the lower tail. In the retained
pilot outputs, the only affected emitted field located by audit is the
F-comp-versus-E-N4 reverse-velocity comparison identified above.

`AWAITING_CONFIRMATION_FREEZE_AUTHORIZATION`
