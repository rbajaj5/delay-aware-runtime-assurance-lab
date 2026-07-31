# PIKS and Delay-Aware Surrogate Modeling

Reference: Joachim Bona-Pellissier, Giacomo Meanti, Matteo Santacesaria, and Lorenzo Rosasco, "PIKS: Universal Physics-Informed Kernel Methods," [arXiv:2607.27062](https://arxiv.org/abs/2607.27062).

## Relevant idea

PIKS studies kernel estimators that incorporate linear differential constraints through an operator-aware construction. The paper emphasizes two properties that are useful for our experiments: universal kernels such as Gaussian or Matérn kernels can provide flexible function classes, while the operator residual remains explicit and analytically inspectable. This is a useful alternative to treating a neural surrogate's fit as evidence that it respects the dynamics.

## Mapping to the queue experiments

Our current systems are discrete and event-driven rather than differential equations. A PIKS-inspired analogue would fit a surrogate for a quantity such as stale-decision rate, queue occupancy, or message age while imposing a finite-difference residual for the chosen queue evolution model. For example, a candidate model could be required to respect the observed update relation

`age[t+1] = max(age[t] + 1 - serviced[t], 0)`

or an analogous bounded queue-balance equation. The useful test would be held-out delay/topology prediction with the residual reported separately from prediction error.

## Boundary

This paper does not establish a runtime-assurance certificate, a multi-agent communication protocol, or a guarantee under unmodeled faults. Universal consistency under stated linear constraints is not the same as safety under an incomplete physical or queue model. Any future surrogate in this repository should therefore report both fit error and operator-residual error, and should never replace the logged mechanism checks.
