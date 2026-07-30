# Delayed Branch-Sector Diagnostic

This experiment is inspired by the branch-factor transport argument in the supplied excerpt from [Bourgade and Huang, *Loop Equations Characterize Random Matrix Statistics*](https://arxiv.org/abs/2607.07617). The excerpt constructs paths that remain away from the real axis and keep pairwise variables separated.

Each agent carries a complex phase surrogate with a fixed sign sector. A linear interpolation between consecutive delayed updates is sampled at 31 points. We record minimum modulus, signed half-plane margin, minimum pairwise separation, and samples that enter the sector tolerance. `blind_delay` uses the latest delivered message; `queue_aware` overlays the newest pending payload.

This is not a reproduction of Proposition 8.4: no branch-factor solution is evaluated, and no theorem assumptions are asserted. The quantities are an engineering diagnostic for branch-cut proximity and phase collision under delayed receipts.

| delay | policy | games | mean min modulus | mean sector margin | mean pair separation | mean crossing samples | worst modulus | worst sector margin | worst pair separation |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | blind_delay | 30 | 0.94475014219399622 | -0.1803253968875223 | 0.001494801651461379 | 3617.9000000000001 | 0.8498853301329059 | -0.40346312233736442 | 4.0424344100253946e-06 |
| 0 | queue_aware | 30 | 0.94475014219399622 | -0.1803253968875223 | 0.001494801651461379 | 3617.9000000000001 | 0.8498853301329059 | -0.40346312233736442 | 4.0424344100253946e-06 |
| 1 | blind_delay | 30 | 0.94385000957240028 | -0.18209260881792852 | 0.0012332858120613296 | 3622.0999999999999 | 0.85151489731544572 | -0.42767195805336211 | 3.794181624771843e-06 |
| 1 | queue_aware | 30 | 0.94385000957240028 | -0.18209260881792852 | 0.0012332858120613296 | 3622.0999999999999 | 0.85151489731544572 | -0.42767195805336211 | 3.794181624771843e-06 |
| 2 | blind_delay | 30 | 0.94046816865408134 | -0.081613732694831034 | 0.0016517516494163587 | 3558.5333333333333 | 0.84109763928776826 | -0.2209047836792479 | 4.091055013907782e-06 |
| 2 | queue_aware | 30 | 0.94375856133656966 | -0.18084579231215878 | 0.0011349978819745767 | 3613.0333333333333 | 0.87250321759240779 | -0.41077473941935494 | 5.257338362519684e-06 |
| 3 | blind_delay | 30 | 0.93192967821866146 | -0.047352520708352573 | 0.0026566573806395019 | 3405.4333333333334 | 0.82464277932889285 | -0.1619233971237749 | 1.3948450893950314e-05 |
| 3 | queue_aware | 30 | 0.9399320709411102 | -0.18387910729923485 | 0.0012057298184686969 | 3611.6333333333332 | 0.86453333864929716 | -0.42313821382966654 | 5.2706458288568397e-06 |
| 5 | blind_delay | 30 | 0.93161457659964941 | -0.018523098739199415 | 0.0036201137374256634 | 2958.9000000000001 | 0.81832540989661162 | -0.2188480620805916 | 1.3786845728992455e-05 |
| 5 | queue_aware | 30 | 0.94370598235488246 | -0.18008277083467586 | 0.00128072052971992 | 3613.2333333333331 | 0.88415720682481469 | -0.40178274254396201 | 7.8575518381934318e-06 |

The result should be read as a branch-geometry diagnostic only. A follow-up could add a sector-preserving projection and test whether it reduces branch crossings without hiding state error.
