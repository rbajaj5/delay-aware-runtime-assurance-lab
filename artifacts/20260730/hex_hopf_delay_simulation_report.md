# Exploratory Hex Hopf-Fiber Delay Simulation

The simulation uses the standard Hopf coordinate construction `z1=cos(eta)e^(i xi1)`, `z2=sin(eta)e^(i xi2)`, with a time-varying fiber phase and a fixed projected base coordinate per cell. Neighbor-coupled phase drift is transmitted through the same delayed move queue as board occupancy.

Transport mode: `adversarial_selective_hold`. For nonzero base delay, phase-bearing messages satisfying a deterministic phase/cell predicate receive one additional hidden hold step. This is deliberately adversarial transport, not an honest FIFO channel.

The `blind_delay` policy uses delivered local phase state; `queue_aware` overlays pending phase payloads before selecting a move. The phase model is a toy communication/oscillator model. It is not a physical derivation from the supplied paper and does not support the paper's unified-field claims.

External context: `https://philpapers.org/archive/NIETTU.pdf`.

| delay | policy | games | mean phase error | p95 phase error | mean extra hold | mean stale conflicts | any conflict rate |
|---:|---|---:|---:|---:|---:|---:|---:|
| 0 | blind_delay | 30 | 1.1102781582905279 | 2.9176352122246048 | 0 | 0 | 0 |
| 0 | queue_aware | 30 | 1.1102781582905279 | 2.9176352122246048 | 0 | 0 | 0 |
| 1 | blind_delay | 30 | 1.187765111890227 | 2.9700085417303725 | 0.51428571428571435 | 20 | 1 |
| 1 | queue_aware | 30 | 1.1221177140665415 | 2.9257667729322203 | 0.52176870748299309 | 0 | 0 |
| 2 | blind_delay | 30 | 1.2018761742569655 | 2.9891048366352697 | 0.50340136054421769 | 41.899999999999999 | 1 |
| 2 | queue_aware | 30 | 1.1198492676768159 | 2.8764621385576108 | 0.51088435374149666 | 0 | 0 |
| 3 | blind_delay | 30 | 1.1875020917297407 | 2.9521560947658405 | 0.5462585034013604 | 43.5 | 1 |
| 3 | queue_aware | 30 | 1.1057260781237768 | 2.8972959160842926 | 0.5653061224489796 | 0 | 0 |
| 5 | blind_delay | 30 | 1.1667383900966473 | 3.0013444905418805 | 0.47414965986394547 | 45.266666666666666 | 1 |
| 5 | queue_aware | 30 | 1.1151236675987535 | 2.8403280006323732 | 0.47619047619047616 | 0 | 0 |
