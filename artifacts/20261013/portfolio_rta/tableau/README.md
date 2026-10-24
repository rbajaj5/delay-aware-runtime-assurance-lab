# Multi-agent teaching dashboard

## Status and files

The simulation and its data were completed and audited before this dashboard
was built. No dashboard result feeds back into the simulation. Nothing has
been published or uploaded to Tableau Public. No sign-in was needed.

- `dashboard_review.html`: self-contained interactive browser review, with two
  global filters, measured summaries, charts, and a synchronized time cursor.
- `Multi_agent_risk_and_assurance.twbx`: candidate native Tableau packaged
  workbook, including the Hyper extract. Open in Tableau Desktop/Public Edition.
- `dataset_profile.json`: source field names, inferred types, counts, missingness.
- `dashboard_validation.json`: actual Tableau Hyper SQL checks against JSON.
- `preview_validation.json`: automated browser checks of every filter combination.

IMPORTANT: native Tableau Desktop is not installed on the generation machine.
Its layout, import compatibility, and parameter controls have NOT been tested
in Desktop. XML parsing is not a substitute for that test. The HTML preview is
not Tableau and does not establish that the native layout renders identically.
Before publication, open the TWBX, confirm the four charts, and exercise both
parameter controls against the expected counts below. Publish only after review.

## Dataset inspection

| Table | Rows | Fields | Grain |
|---|---:|---:|---|
| frame_metrics.csv | 2,160 | 26 | One frame |
| action_weights.csv | 12,960 | 8 | One action per frame |
| dashboard_data.csv / Hyper extract | 25,920 | 21 | One frame, action, and proposed/applied allocation |

The original tables have zero missing cells. Numeric fields include time,
eigenvalues, unitless ratios, and distances/errors in feet; categorical fields
include phase, coverage, and controller decision. Boolean fields distinguish
identity continuity and hidden collapse. Full field-level types and unique
counts are in `dataset_profile.json`. There is just ONE independent episode.

The flattened Tableau table repeats frame metrics 12 times. Use COUNTD(Frame),
not row count, for exposure and event counts. AVG is appropriate for replicated
frame metrics here because the replication count is equal for every frame.
Never sum a conditioning ratio or a leading eigenvalue share. Weight means are
computed within Action and Allocation; each proposed/applied mix sums to one.
All fields come from the frozen JSON, not recreated illustrative values.

## Three questions and chart choices

1. **When does motion concentrate in a common direction?** A line of leading
   eigenvalue share against time shows timing and persistence. Larger share
   means concentration of variance, not automatically danger. The defense change
   and other scripted processes co-evolve; this chart alone is not a causal test.
2. **How does the proposed action mix differ from the applied mix?** Paired
   horizontal bars compare average weights for the same six actions without
   requiring the student to interpret stacked areas. Gold is proposed, green is
   applied. During HOLD, the proposed allocation can continue changing.
3. **Can small tracking error coexist with poor formation geometry?** A scatter
   of shadow tracking RMSE against shadow conditioning directly compares the
   two quantities. In the review preview, reference lines mark 0.5 ft and 0.12.
   Red marks additionally satisfy spacing >= 2 ft. A companion time-series plot
   contrasts proposed and applied geometry to show when the shield intervenes.

## Teaching walkthrough

Designed for an upper-undergraduate student familiar with vectors and variance.
Begin by asking which plot answers a performance question and which answers a
geometry question. Then ask whether low tracking error is sufficient to conclude
the shape is usable. Finally, ask why the optimizer's proposal and the applied
weights can disagree without the optimization having failed.

Try these two combinations:

- **Stable five-out + ACCEPT:** 360 frames, mean leading share 0.3614363872,
  zero hidden-collapse frames. Establish the initial operating regime.
- **Geometry under stress + RTA HOLD:** 346 frames, mean leading share
  0.8299066327, 342 hidden-collapse shadow frames. Notice that the applied
  conditioning remains above 0.12 while the unshielded shadow can track a bad
  reference accurately. HOLD and hidden collapse need not begin simultaneously.

One central limitation: the environment and compression are scripted. This is a
mechanism illustration, not a measured frequency of failures in sport or robotics.
Selecting observations by controller decision is not randomized intervention.

## Quality warnings and validation

- All 18 phase/controller combinations passed Hyper SQL comparison to JSON,
  including empty combinations and per-action proposed/applied weight means.
- The browser review passed the same 18 combinations, empty/small-group warning
  checks, cursor movement, desktop/mobile overflow checks, and console checks.
- The geometry/ACCEPT group has only 14 frames; the final-phase/HOLD group has
  six. The preview flags groups under 120 frames (five seconds). This cutoff is
  a descriptive warning, not a power analysis or significance criterion.
- Native Tableau includes a persistent warning and a `Group note` calculated
  field; it does not yet have a verified dynamic warning display.
- Windows overlap. Frame counts are not independent sample sizes and should
  not be used for naive confidence intervals or p-values.
- No identity swaps or missing measurements occur; those monitors were not
  challenged. Covariance conditioning depends on the regularization settings.
- The last ten seconds depict drones using the same trajectory. These are not
  additional flight-validation data. No profitability inference is supported.

The exact simulation source SHA-256 is recorded in both the preview and
`dashboard_validation.json`. The movie and dashboard use the same source.

## Rebuild

From this folder, with the original JSON/CSVs/validation one directory above:

```bash
python -m pip install numpy lxml tableauhyperapi tableaudocumentapi
python build_portfolio_dashboard.py --source .. --out .
```

`portfolio_dashboard_preview.html` is the preview template. The generator never
runs the simulation and never publishes. The simulation's separate README
documents reproduction of the numerical sequence and MP4.
