# Hopf-Fiber Adversarial Hold Sweep

The selective hold strength was swept over `[0, 1, 2]` extra steps. The hold predicate is deterministic in payload phase and cell coordinates. `blind_delay` ignores pending messages; `queue_aware` overlays them.

| hold strength | base delay | policy | games | mean conflicts | any-conflict rate | mean phase error | mean extra hold |
|---:|---:|---|---:|---:|---:|---:|---:|
| 0 | 0 | blind_delay | 15 | 0 | 0 | 1.1233942923654443 | 0 |
| 0 | 0 | queue_aware | 15 | 0 | 0 | 1.1233942923654443 | 0 |
| 0 | 1 | blind_delay | 15 | 0 | 0 | 1.1184404530034506 | 0 |
| 0 | 1 | queue_aware | 15 | 0 | 0 | 1.1184404530034506 | 0 |
| 0 | 2 | blind_delay | 15 | 43.06666666666667 | 1 | 1.2031862915438716 | 0 |
| 0 | 2 | queue_aware | 15 | 0 | 0 | 1.1192011207774826 | 0 |
| 0 | 3 | blind_delay | 15 | 42.133333333333333 | 1 | 1.2131169452068857 | 0 |
| 0 | 3 | queue_aware | 15 | 0 | 0 | 1.1360594941548166 | 0 |
| 0 | 5 | blind_delay | 15 | 44.666666666666664 | 1 | 1.1948332313812122 | 0 |
| 0 | 5 | queue_aware | 15 | 0 | 0 | 1.147422544615569 | 0 |
| 1 | 0 | blind_delay | 15 | 0 | 0 | 1.122622808381283 | 0 |
| 1 | 0 | queue_aware | 15 | 0 | 0 | 1.122622808381283 | 0 |
| 1 | 1 | blind_delay | 15 | 18.466666666666665 | 1 | 1.1693686148830105 | 0.47482993197278917 |
| 1 | 1 | queue_aware | 15 | 0 | 0 | 1.1200249093548371 | 0.4965986394557822 |
| 1 | 2 | blind_delay | 15 | 41.600000000000001 | 1 | 1.1832818820144884 | 0.49795918367346942 |
| 1 | 2 | queue_aware | 15 | 0 | 0 | 1.1154013282160087 | 0.50476190476190474 |
| 1 | 3 | blind_delay | 15 | 42.200000000000003 | 1 | 1.2417733266069599 | 0.60136054421768703 |
| 1 | 3 | queue_aware | 15 | 0 | 0 | 1.1421723167855744 | 0.57823129251700689 |
| 1 | 5 | blind_delay | 15 | 45.666666666666664 | 1 | 1.1872072462403787 | 0.52653061224489806 |
| 1 | 5 | queue_aware | 15 | 0 | 0 | 1.1417525304451785 | 0.527891156462585 |
| 2 | 0 | blind_delay | 15 | 0 | 0 | 1.1010754989207301 | 0 |
| 2 | 0 | queue_aware | 15 | 0 | 0 | 1.1010754989207301 | 0 |
| 2 | 1 | blind_delay | 15 | 18.266666666666666 | 1 | 1.1859813546847717 | 0.97414965986394553 |
| 2 | 1 | queue_aware | 15 | 0 | 0 | 1.1122105447493165 | 0.90884353741496582 |
| 2 | 2 | blind_delay | 15 | 41.399999999999999 | 1 | 1.1841523485734986 | 0.97687074829931975 |
| 2 | 2 | queue_aware | 15 | 0 | 0 | 1.138726086159797 | 0.94149659863945578 |
| 2 | 3 | blind_delay | 15 | 44.466666666666669 | 1 | 1.1929000353917978 | 1.1972789115646258 |
| 2 | 3 | queue_aware | 15 | 0 | 0 | 1.1000739909275952 | 1.115646258503401 |
| 2 | 5 | blind_delay | 15 | 44.399999999999999 | 1 | 1.1832139017606693 | 1.017687074829932 |
| 2 | 5 | queue_aware | 15 | 0 | 0 | 1.1219371281637309 | 1.0340136054421767 |

This is a deliberately adversarial communication toy. The Hopf coordinates provide a hidden fiber-phase state; they do not establish the external paper's proposed physical theory. No v6 claim is updated.
