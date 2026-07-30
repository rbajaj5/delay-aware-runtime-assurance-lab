# Semantic Queue-Pollution Simulation

Useful messages are coordination updates. Background messages are low-value work that still consumes finite processing capacity. Each agent can process two messages per round. `fifo_blind` processes arrival order. `priority_deduplicating` prioritizes useful messages and retains only the newest useful message from each sender.

The experiment is a queueing abstraction, not a model of human worth or semantic meaning. It isolates a systems question: can low-value traffic make a delay-blind system miss or act on stale high-value updates?

The figure shows delay 3. The CSV includes all delays and background-load levels.

| delay | background rate | policy | games | fresh useful rate | stale useful rate | useful wait | mean backlog |
|---:|---:|---|---:|---:|---:|---:|---:|
| 0 | 0.0 | fifo_blind | 20 | 1 | 0 | 1 | 24 |
| 0 | 0.0 | priority_deduplicating | 20 | 1 | 0 | 1 | 24 |
| 0 | 1.0 | fifo_blind | 20 | 0.017718179478868404 | 0.98228182052113167 | 30.579612024427167 | 1112.828888888889 |
| 0 | 1.0 | priority_deduplicating | 20 | 1 | 0 | 1 | 1112.828888888889 |
| 0 | 2.0 | fifo_blind | 20 | 0.014738090646416551 | 0.9852619093535836 | 45.084938116273413 | 2195.1525000000001 |
| 0 | 2.0 | priority_deduplicating | 20 | 1 | 0 | 1 | 2195.1525000000001 |
| 0 | 4.0 | fifo_blind | 20 | 0.017365699935302795 | 0.98263430006469732 | 59.950897321359626 | 4363.671666666668 |
| 0 | 4.0 | priority_deduplicating | 20 | 1 | 0 | 1 | 4363.671666666668 |
| 0 | 8.0 | fifo_blind | 20 | 0.027627897685687908 | 0.97237210231431226 | 71.262820200478785 | 8708.0844444444429 |
| 0 | 8.0 | priority_deduplicating | 20 | 1 | 0 | 1 | 8708.0844444444429 |
| 1 | 0.0 | fifo_blind | 20 | 1 | 0 | 1 | 24 |
| 1 | 0.0 | priority_deduplicating | 20 | 1 | 0 | 1 | 24 |
| 1 | 1.0 | fifo_blind | 20 | 0.017701007624084962 | 0.98229899237591511 | 30.362337859345605 | 1105.5336111111114 |
| 1 | 1.0 | priority_deduplicating | 20 | 1 | 0 | 1 | 1105.5336111111114 |
| 1 | 2.0 | fifo_blind | 20 | 0.015742505291318536 | 0.98425749470868173 | 45.286129261193004 | 2198.3919444444446 |
| 1 | 2.0 | priority_deduplicating | 20 | 1 | 0 | 1 | 2198.3919444444446 |
| 1 | 4.0 | fifo_blind | 20 | 0.017471245477580468 | 0.98252875452241961 | 59.995231214683244 | 4379.2988888888885 |
| 1 | 4.0 | priority_deduplicating | 20 | 1 | 0 | 1 | 4379.2988888888885 |
| 1 | 8.0 | fifo_blind | 20 | 0.027530459469408276 | 0.97246954053059154 | 71.150253296219347 | 8694.331944444446 |
| 1 | 8.0 | priority_deduplicating | 20 | 1 | 0 | 1 | 8694.331944444446 |
| 2 | 0.0 | fifo_blind | 20 | 0 | 1 | 2 | 47.866666666666667 |
| 2 | 0.0 | priority_deduplicating | 20 | 0 | 1 | 2 | 47.866666666666667 |
| 2 | 1.0 | fifo_blind | 20 | 0 | 1 | 31.433401589798343 | 1129.6799999999998 |
| 2 | 1.0 | priority_deduplicating | 20 | 0 | 1 | 2 | 1129.6799999999998 |
| 2 | 2.0 | fifo_blind | 20 | 0 | 1 | 46.204058156578107 | 2204.6613888888887 |
| 2 | 2.0 | priority_deduplicating | 20 | 0 | 1 | 2 | 2204.6613888888887 |
| 2 | 4.0 | fifo_blind | 20 | 0 | 1 | 61.123133166593448 | 4363.383055555556 |
| 2 | 4.0 | priority_deduplicating | 20 | 0 | 1 | 2 | 4363.383055555556 |
| 2 | 8.0 | fifo_blind | 20 | 0 | 1 | 73.344378344384594 | 8717.8655555555561 |
| 2 | 8.0 | priority_deduplicating | 20 | 0 | 1 | 2 | 8717.8655555555561 |
| 3 | 0.0 | fifo_blind | 20 | 0 | 1 | 3 | 71.599999999999994 |
| 3 | 0.0 | priority_deduplicating | 20 | 0 | 1 | 3 | 71.599999999999994 |
| 3 | 1.0 | fifo_blind | 20 | 0 | 1 | 32.240322087876123 | 1137.1816666666666 |
| 3 | 1.0 | priority_deduplicating | 20 | 0 | 1 | 3 | 1137.1816666666666 |
| 3 | 2.0 | fifo_blind | 20 | 0 | 1 | 47.008946994237398 | 2209.505555555555 |
| 3 | 2.0 | priority_deduplicating | 20 | 0 | 1 | 3 | 2209.505555555555 |
| 3 | 4.0 | fifo_blind | 20 | 0 | 1 | 62.500251373295271 | 4374.6727777777778 |
| 3 | 4.0 | priority_deduplicating | 20 | 0 | 1 | 3 | 4374.6727777777778 |
| 3 | 8.0 | fifo_blind | 20 | 0 | 1 | 75.965620760480277 | 8710.0205555555549 |
| 3 | 8.0 | priority_deduplicating | 20 | 0 | 1 | 3 | 8710.0205555555549 |
| 5 | 0.0 | fifo_blind | 20 | 0 | 1 | 5 | 118.66666666666667 |
| 5 | 0.0 | priority_deduplicating | 20 | 0 | 1 | 5 | 118.66666666666667 |
| 5 | 1.0 | fifo_blind | 20 | 0 | 1 | 33.66057405166729 | 1152.4944444444443 |
| 5 | 1.0 | priority_deduplicating | 20 | 0 | 1 | 5 | 1152.4944444444443 |
| 5 | 2.0 | fifo_blind | 20 | 0 | 1 | 48.54813944388426 | 2214.2080555555558 |
| 5 | 2.0 | priority_deduplicating | 20 | 0 | 1 | 5 | 2214.2080555555558 |
| 5 | 4.0 | fifo_blind | 20 | 0 | 1 | 65.239442045051732 | 4365.7319444444447 |
| 5 | 4.0 | priority_deduplicating | 20 | 0 | 1 | 5 | 4365.7319444444447 |
| 5 | 8.0 | fifo_blind | 20 | 0 | 1 | 80.806595803000391 | 8695.9055555555551 |
| 5 | 8.0 | priority_deduplicating | 20 | 0 | 1 | 5 | 8695.9055555555551 |

All results are exploratory. The priority policy is a queue discipline, not a safety certificate; a deployment would need an explicit policy for message classes, starvation, auditability, and failure handling.
