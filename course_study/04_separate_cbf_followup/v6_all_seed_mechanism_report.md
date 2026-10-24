# v6 All-Seed Mechanism Census

This read-only analysis covers every retained v6 episode: 6,000 episodes and 600,000 comparable step rows across F-comp and F-blind in the uniform, gradual, and persistent blocks. It measures whether the logged queue-anchor displacement is present before fallback and whether episode-level displacement tracks fallback or severity. These are descriptive associations, not causal or stability claims.

## Definitions

- Anchor gap: `||formal_barrier_anchor_state - formal_state_at_decision||_2`.
- Pre-fallback gap: rows strictly before the first fallback row in an episode.
- Fallback-row gap: rows where `formal_fallback_used` is true.
- Episode correlations are Pearson correlations over the 1,000 retained seeds within each block/arm cell.

## All-seed results

| block | arm | episodes | positive-gap episodes | positive-gap episodes without fallback | fallback episodes | mean episode gap | mean fallback rate | mean severity sum | pre-fallback positive fraction | gap/fallback r | gap/severity r | queue/fallback r | queue/gap r |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| gradual_guardrail | F-blind | 1000 | 0 | 0 | 1000 | 0 | 0.31404999999999994 | 3.4671082848899997 | 0 | nan | nan | 0.43606465907301672 | nan |
| gradual_guardrail | F-comp | 1000 | 1000 | 0 | 1000 | 0.21147710398968905 | 0.062460000000000029 | 2.7727686632150004 | 0.999 | -0.26693217975010708 | -0.52171663799440526 | -0.25247924539489425 | 0.75148836332804936 |
| persistent_guardrail | F-blind | 1000 | 0 | 0 | 1000 | 0 | 0.32400999999999996 | 6.7295590420850004 | 0 | nan | nan | -0.26601522531960842 | nan |
| persistent_guardrail | F-comp | 1000 | 1000 | 0 | 1000 | 0.25392446498005417 | 0.12304000000000002 | 8.8403940863349995 | 0.999 | -0.64912399797611187 | -0.49774277108639614 | -0.47480071887460823 | 0.55809986421246238 |
| uniform_primary | F-blind | 1000 | 0 | 0 | 1000 | 0 | 0.29878999999999994 | 0.41000109859333328 | 0 | nan | nan | 0.59052973657069563 | nan |
| uniform_primary | F-comp | 1000 | 1000 | 948 | 52 | 0.23910071583579939 | 0.00055000000000000003 | 0 | 0.98076923076923073 | 0.13925971721618555 | nan | -0.41380426144175492 | -0.065938516015438708 |

## Mechanistic reading boundary

The decisive temporal check is whether a gap exists before fallback. A positive pre-fallback gap shows that the anchor/current-state discrepancy is not merely a by-product of entering the fallback branch. Correlations then quantify whether the discrepancy tracks response intensity within these frozen cells; they do not identify a causal direction. The multi-agent proxy remains a separate exploratory system: its worlds are not the v6 registered seeds, so its matrix results must not be merged with this single-vehicle census.

## Input provenance

- `C:\Users\anaxe\RAGforAssurredAutonomy\analysis\out\v6_cbf_qp_confirmation_20260724\blocks\uniform_primary\BLOCK_MANIFEST.sha256`: `210f9a4a57d174dba366745deeb5138f113c933c1bd1c43ef1dc22ce6366d60b`
- `C:\Users\anaxe\RAGforAssurredAutonomy\analysis\out\v6_cbf_qp_confirmation_20260724\blocks\uniform_primary\full_steps\F-blind_steps.csv`: `da91239c5ace5bab11df5e0e2e327a37c447e60ed31969eeccc7027f9bcb64ec`
- `C:\Users\anaxe\RAGforAssurredAutonomy\analysis\out\v6_cbf_qp_confirmation_20260724\blocks\uniform_primary\full_steps\F-comp_steps.csv`: `a35e19e8c22e8efca6d0ee7704580756208c70cf9d1c73e3f739b04992691d25`
- `C:\Users\anaxe\RAGforAssurredAutonomy\analysis\out\v6_cbf_qp_confirmation_complete_20260724_r2\blocks\gradual_guardrail\BLOCK_MANIFEST.sha256`: `ff8e9c132d362acd7c23caf8d3cc9067de4253c9354e8001c7d6889a0f210a2a`
- `C:\Users\anaxe\RAGforAssurredAutonomy\analysis\out\v6_cbf_qp_confirmation_complete_20260724_r2\blocks\gradual_guardrail\full_steps\F-blind_steps.csv`: `b8ba414d7517011e6a9a0bf415940c405712bc465857d7c5ef33f9f847734b77`
- `C:\Users\anaxe\RAGforAssurredAutonomy\analysis\out\v6_cbf_qp_confirmation_complete_20260724_r2\blocks\gradual_guardrail\full_steps\F-comp_steps.csv`: `d7d638540d30833eec71b3e6192755321e61642247c18f3c5252c3214e5a0d9b`
- `C:\Users\anaxe\RAGforAssurredAutonomy\analysis\out\v6_cbf_qp_confirmation_complete_20260724_r2\blocks\persistent_guardrail\BLOCK_MANIFEST.sha256`: `828687f51035df937ac620fbbb1a744712a0be9f40e41fd46aff36637fd8091b`
- `C:\Users\anaxe\RAGforAssurredAutonomy\analysis\out\v6_cbf_qp_confirmation_complete_20260724_r2\blocks\persistent_guardrail\full_steps\F-blind_steps.csv`: `c515accc798497c1650c96d53d09f4a4494fc6b551da71a9b3178eccfa34d931`
- `C:\Users\anaxe\RAGforAssurredAutonomy\analysis\out\v6_cbf_qp_confirmation_complete_20260724_r2\blocks\persistent_guardrail\full_steps\F-comp_steps.csv`: `f8719dd5cb2fcc9eb2b0c46082a3e1996af312799aca8e59067671a525d66231`
