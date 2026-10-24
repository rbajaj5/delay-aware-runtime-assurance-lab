# Covariance allocation and runtime assurance

This package contains one deterministic, synthetic 90-second teaching sequence.
It is a kinematic five-on-five basketball illustration, followed by a visual
change to schematic quadrotors using the same state. The last segment is not a
second robotics experiment. No betting profitability or operational flight
validation is claimed. The sequence uses an extended teaching clock, not an NBA
possession clock.

## Run and inspect

```bash
python -m pip install numpy scipy Pillow
python portfolio_rta_demo.py --out reproduced --encoder libx264
```

FFmpeg must be on PATH. `--encoder h264_nvenc` uses NVIDIA hardware for video
encoding when supported. Simulation, covariance, optimization, and rasterization
run on the CPU; GPU encoding is not GPU physics. `--data-only` runs and audits the
simulation without making the movie. `--render-existing` renders the existing
frame JSON without resimulating. Font lookup supports Segoe UI on Windows and
DejaVu Sans on Linux. The delivered rendering is 1920 x 1080 at 24 fps.

The generated `synchronized_frames.json` contains all 2,160 frames, with 10 agent
positions, ball position, observed positions, proposals, shadow positions,
covariance, eigenvalues/eigenvectors, utility, allocation weights, and RTA state.
Coordinates are in feet. `frame_metrics.csv` has one row per frame;
`action_weights.csv` has six rows per frame. `validation.json` records numerical
checks and computed results. `manifest.json` contains SHA-256 hashes.

## Model and timeline

- 0-15 s: stable five-out formation under drop coverage.
- 15-40 s: a smooth switch-and-help transition couples defender motions.
- 40-65 s: changing observations and utilities redistribute action weights.
- 65-80 s: an externally commanded compression stresses leader geometry.
- 80-90 s: compression recovers; player glyphs become schematic quadrotors.

The utility vector is a stated synthetic coaching surrogate using coverage,
observed clearance, and slow time variation. It is not a fitted scoring
probability or an estimate of financial returns. Six action maps are blended
using the optimization weights, then composed with the environmental compression
request. The six choices are high pick-and-roll, five-out drive, weak-side cut,
post split, advance, and reset. Advance is a half-court surrogate for transition.

The Markowitz-style proposal solves

`maximize mu.T @ w - gamma/2 * w.T @ Sigma @ w`,

subject to nonnegative weights summing to one, with gamma = 5. SciPy SLSQP solves
the six-dimensional convex program and the audit checks its KKT residual.
Covariance matters directly through `w.T @ Sigma @ w`; equivalently, variance is
the sum of `lambda_i * (v_i.T @ w)^2` over all six modes. Displaying only the top
three eigenvalues does not truncate the optimizer. A frozen-covariance diagnostic
is exported using the same utility vector; it isolates the optimizer's numerical
dependence on covariance at that frame, not an independently rolled-out policy.

## Displayed metrics

| Metric | Definition and interpretation |
|---|---|
| Covariance matrix | Sample covariance of 144 recent six-dimensional relative-velocity observations, a causal six-second window. Each channel is a directional offensive/defensive relative velocity divided by 3 ft/s. The reset channel combines two vertical components. The axes are action-associated motion proxies, not empirical action-return covariance. |
| Regularization | `(1 - 0.12) * sample_cov + 0.12 * trace(sample_cov)/6 * I + 0.015 * I`. All displayed spectral metrics use this same positive-definite estimate. The heatmap uses one fixed scale for the entire movie. |
| Top three eigenvalues | Variance along the three largest covariance modes, in squared normalized feature units. Larger values indicate stronger variation along those modes. |
| Leading share | `lambda1 / sum(lambda)`. A large share means variance is concentrated in one direction, not necessarily that physical risk is high. |
| Adjacent-window overlap | `abs(v1(t).T @ v1(t-1))`. Close to one means little change from the immediately preceding frame. Gradual cumulative rotation can coexist with overlap near one. The six PC1 bars display the sign-aligned leading vector's components; a small leading eigenvalue gap makes its direction sensitive. |
| Covariance condition number | `lambda_max / lambda_min`. Measures anisotropy of the regularized estimate; its value depends on shrinkage and ridge settings. |
| Leader affine map | Let E contain the two edges from L1 to L2 and L3. `A = E @ inverse(E_design)`. This removes the conditioning of the original design triangle. |
| Affine conditioning | `sigma_min(A) / sigma_max(A)`. Values approaching zero indicate a nearly rank-deficient map. Proposed and applied geometry are plotted separately. Threshold = 0.12. |
| Area scale | `abs(det(A))`, the triangle's area divided by its design area. Minimum accepted proposed scale = 0.07. |
| Pairwise spacing | Minimum Euclidean distance between any two offensive agents, in feet. Threshold = 2 ft. This does not evaluate all offense-defense contacts. |
| Applied tracking RMSE | Square root of the mean squared Euclidean position error relative to the applied reference, in feet. During HOLD that reference is the held map. |
| Follower RMSE | Same calculation for F1 and F2 against their fixed affine combinations of the held/applied leader targets. Affine coefficients sum to one but can be negative for points outside the leader triangle. |
| Unshielded tracking RMSE | Tracking error in a shadow trajectory that follows every proposal with the same disturbances. It is not the protected formation's error against a rejected reference. |
| Hidden collapse | Shadow conditioning < 0.12, shadow tracking RMSE < 0.5 ft, and shadow minimum spacing >= 2 ft, all at the same frame. |
| Proposed/applied weights | Gold is the current optimizer solution. Green is the solution allowed by RTA; it remains fixed throughout HOLD even while the gold bars move. |
| Identity and observations | Five persistent offensive identities, finite observations, observation age 1/24 s, and valid-observation fraction. This scenario injects no identity swaps or missing measurements. |

L1, L2, L3 are the three highest-rated synthetic players (96, 93, 91); F1 and F2
are rated 86 and 84. Those ratings only choose leaders and do not estimate real
player impact. All followers use design-time affine coefficients stored in the
JSON metadata. Both the protected and shadow trajectories receive the same small
deterministic tracking disturbances. Observations have seeded Gaussian noise.

The RTA layer checks geometry, identities, and observation finiteness separately
from the utility optimization. It checks a conservative singular-value margin
assuming leader position error <= 0.015 ft; that assumption is an illustrative
guard margin, not a probabilistic guarantee for unbounded Gaussian noise. HOLD
retains the last accepted map, translation, and weights. Resume requires a lower
conditioning bound of 0.16 to prevent chattering. The 0.12 threshold is a design
choice, not a basketball or robotics safety standard.

Each frame is a post-control snapshot; its covariance and proposal were computed
from the latest observations just before that step. That one-step observation
age is explicitly logged. All curves, players, bars, and moving cursors use the
same frame record. No future observation enters the rolling estimator.

## Interpretation and checks

The scenario is constructed to demonstrate a mechanism, not to estimate its
frequency in sport or flight. There is one independent episode. The 2,160 frames
and overlapping windows are serially dependent; they must not be treated as
2,160 independent experimental replications. The final drone view preserves
coordinates and allocations to explain the mathematics; new dynamics, utilities,
constraints, and assurance arguments would be needed for an actual drone system.

The audit reconstructs every rolling covariance from the exported features,
checks positive eigenvalues and simplex weights, compares spectral and quadratic
variance, checks map/weight freezing during HOLD, and checks the actual applied
conditioning and spacing. Outcomes are calculated from the trajectory and are
not hardcoded in the video.
