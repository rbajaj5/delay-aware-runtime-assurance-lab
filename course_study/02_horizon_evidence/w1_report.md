# V5 Phase W1 Report

Date: 2026-07-23  
Scope: read-only CHK-V, claims ledger, and fixed-structure outline only. No simulations, hypothesis tests, figure regeneration, or manuscript drafting were performed.

## Source gate

All numerical sources used in `v5_claims_ledger.csv`, `v5_outline.md`, and this report were SHA-256 checked before their contents were read. The source and output hashes are recorded in `W1_MANIFEST.sha256`.

CHK-V read six full step logs already covered by the Phase 2b manifest: E-N3, E-N4, and AP-N3 in the 1,000-seed confirmation and 100-seed corrected-`physical_multiplier` gradual guardrail blocks. Its episode census contains 3,300 rows.

## Pre-committed CHK-V readings

Copied verbatim before adjudication:

> E-N4 (and AP) show no material shift toward the reverse bound relative to E-N3 survivors → the repairs are presented without the λ50-style endpoint qualification, and the contrast with λ50 becomes a paper point in the repairs section.

> Either repair shows a λ50-style shift → the repair claim carries the same horizon-and-endpoint-bounded qualification λ50 received, in the abstract and the repairs section, not only in limitations.

## CHK-V results

### All episodes

| Block | Arm | n | OOB | mean min(vx) | p5 min(vx) | p50 min(vx) | absolute min(vx) | vx < -1 count | mean max abs(vx) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Confirmation | E-N3 | 1000 | 175 | -0.6288780002009202 | -1.0546628586828652 | -0.7288531882803282 | -1.1727065786185054 | 352 | 1.3223009233104142 |
| Confirmation | E-N4 | 1000 | 0 | -0.7951282743519873 | -0.9549593608004636 | -0.8714779834845104 | -1.0244352180148022 | 22 | 1.0202998742307983 |
| Confirmation | AP-N3 | 1000 | 0 | -0.9462127155853817 | -1.0369520823016791 | -0.9293465446327777 | -1.1538484946259020 | 135 | 0.9910870023574093 |
| Gradual guardrail, physical_multiplier | E-N3 | 100 | 11 | -0.2394720542461401 | -0.4286717355488942 | -0.2313352506693489 | -0.4584645252747813 | 0 | 1.1304170155230220 |
| Gradual guardrail, physical_multiplier | E-N4 | 100 | 0 | -0.1798983213262504 | -0.4373831582438941 | -0.2739450127196481 | -0.5207483861329446 | 0 | 0.9328898486267324 |
| Gradual guardrail, physical_multiplier | AP-N3 | 100 | 0 | -0.8101875077336174 | -1.3390188243268162 | -0.6913777764957703 | -1.3769255491288890 | 20 | 0.9794455150331661 |

### Paired E-N3 survivor comparisons

| Block | Comparison | paired n | E-N3 mean min(vx) | repair mean min(vx) | mean paired delta | E-N3 vx < -1 | repair vx < -1 | E-N3 mean max abs(vx) | repair mean max abs(vx) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Confirmation | E-N4 | 825 | -0.7623359383375032 | -0.8065171582278999 | -0.0441812198903967 | 352 | 22 | 1.1264803102910468 | 1.0243339936868308 |
| Confirmation | AP-N3 | 825 | -0.7623359383375032 | -0.9533487266445497 | -0.1910127883070465 | 352 | 134 | 1.1264803102910468 | 0.9990956894490712 |
| Gradual guardrail, physical_multiplier | E-N4 | 89 | -0.2691364775703908 | -0.1954841869526212 | 0.0736522906177696 | 0 | 0 | 0.9915103778975255 | 0.9298404923914941 |
| Gradual guardrail, physical_multiplier | AP-N3 | 89 | -0.2691364775703908 | -0.8161404198999049 | -0.5470039423295140 | 0 | 20 | 0.9915103778975255 | 0.9877809733971661 |

## CHK-V reading

**Split adjudication:**

- **E-N4:** The first pre-committed reading applies to E-N4. Across the confirmation dual survivors, E-N4 has fewer reverse-bound crossings than E-N3 (22 versus 352), and the corrected-`physical_multiplier` gradual guardrail has zero crossings in both arms with E-N4's mean minimum velocity moving away from the reverse bound. E-N4 does not reproduce the lambda=50 pattern of more reverse-bound crossings and a systematic paired shift.
- **AP-N3:** The second pre-committed reading applies to AP. Although confirmation crossings fall from 352 to 134, the corrected-`physical_multiplier` gradual guardrail moves paired mean minimum velocity from -0.2691364775703908 to -0.8161404198999049 and introduces 20 reverse-bound crossings where E-N3 has none. AP's repair claim therefore carries an explicit horizon-and-endpoint qualification in the abstract-equivalent opening and Section 7, not only in limitations.

This split must remain visible. A sentence treating E-N4 and AP as having the same full-state qualification would erase the result of CHK-V.

## Claims-ledger checks

- C1 includes both range-scoped E-N3 rates, both D-N3 zero counts, the six original named families, and fresh unnamed two-seed family `E-N3-F006`.
- C2 includes HZ1, the registered p-value and alpha, guardrail outcome, and E-N4's CHK-V qualification.
- C3 describes AP as the confirmed dual-branch artifact, includes the 57.6% fallback rate, and carries the AP-specific CHK-V qualification.
- C4 contains only measured N values; it contains no interpolation or trend test.
- C5 reports exploratory 2x2 outcomes and exact-signature overlap without p-values.
- C6 separates measured saturation, fixation, repricing, intervention-rate, and trajectory-distribution diagnostics.
- C7 is split into pipeline-scoped fault profiles, fully qualified lambda=50 mitigation, and severity/semantics methodology.

## Mandatory corrections checklist

- [x] **AP description:** Replace every AP gloss with: "The candidate-projection arm selects the predicted-safe library member nearest the clipped PPO action when at least one exists; when no candidate is predicted safe, it falls back to the infeasible candidate minimizing the raw delay-aware score plus its distance to the PPO action. It is a discrete proxy without reachability guarantees; step logs show the fallback branch is exercised on a majority of interventions." Delete the v4 sentence claiming that predicted-safe restriction itself avoids the exit mode.
- [x] **Mechanism name:** Retire "short-horizon myopia." When historical wording is needed, use "the ranking blind spot at the tested horizon, later identified as the boundary-shell expression of the saturation defect."
- [x] **Causal referent:** Replace "horizon depth is the binding constraint" with "the shared trigger-plus-scorer lookahead." At the same claim location, report CHK1's 6.6%-7.9% reduction in trigger/intervention rate, which is opposite the trigger-mediated prediction.
- [x] **Trigger terminology:** Replace "quiet trigger" with "baseline trigger" throughout; E-N3 intervenes 83.8 times per 100 steps in the pilot.
- [x] **Range scope:** Never write an unqualified E-N3 rate. Use 159/1000 for the original block and 175/1000 for the fresh confirmation.
- [x] **Pipeline labels:** Every gradual-cell value must name its conversion pipeline. In particular, the 302/1000 result belongs to the canonical pre-reconciliation fault-profile conversion pipeline; the horizon guardrail belongs to the corrected `physical_multiplier` pipeline.

The checklist is satisfied at the outline and ledger level. It becomes a W2 drafting gate on every section, caption, table, and abstract sentence.

## Outline check

`v5_outline.md` follows the required 11-section map without restructuring. Its main-text estimate is 15.50 pages before appendices. It places the seven existing provenance-stamped figures in evidence order and forbids paper figure regeneration from anything except committed `figure_data/` CSVs.

The current W1 source set does not contain a verified course-template page limit. This remains flagged for W2 formatting review rather than being decided here.

## Deliverables

- `chkv_fullstate_census.csv`
- `chkv_summary.json` (supporting machine-readable summary)
- `v5_claims_ledger.csv`
- `v5_outline.md`
- `w1_report.md`
- `W1_MANIFEST.sha256`

Phase W2 has not started.

AWAITING_SIGNOFF
