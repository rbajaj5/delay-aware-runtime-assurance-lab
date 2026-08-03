# 3D Formation Coordination under Delayed Messages

This is an exploratory kinematic formation model, not a validated aircraft, humanoid, or multi-robot flight-dynamics model.
The run used PyTorch device `cuda`. CUDA was available: `True`.
The delay-blind controller uses delayed leader and neighbor state without a safety fallback. The monitored controller uses the same delayed state but checks local minimum separation, formation error, and message age; when risk is detected it caps motion and applies a local separation fallback.
A bounded paired perturbation is applied during the communication outage to create a controlled formation-risk episode; it is not a model of a particular aircraft disturbance.

| policy | fallback steps | unsafe-overlap steps | minimum separation | mean formation error | final leader x | runtime seconds |
|---|---:|---:|---:|---:|---:|---:|
| delay-blind | 0 | 18 | 0.405 | 0.421 | 7.666 | 3.658 |
| monitored | 21 | 0 | 1.136 | 0.458 | 7.666 | 4.116 |

Interpretation boundary: the simulation illustrates why communication delay and fallback semantics should be tested together. It does not establish a safety guarantee, a formation-control theorem, or a deployment recommendation.
