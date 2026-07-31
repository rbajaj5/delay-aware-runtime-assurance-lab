# Delay Concentration Diagnostic

This experiment applies a limited methodological analogy from counterexample-driven analysis: equal global mass need not imply equal local concentration. It compares two delay schedules with the same mean delay of 3: a spread schedule with delay 3 every round, and a clustered schedule with delay 12 every fourth round and zero otherwise.

The multi-agent system is a 12-agent ring. Each agent receives useful state updates from two neighbors and makes a stale decision when its latest applied neighbor state differs from the current state. `fifo_blind` processes messages by arrival order; `queue_aware_latest` retains the newest due update per sender.

Reported quantities separate global stale-event rate from concentration: maximum sliding-window rate, maximum per-agent rate, a spatial Gini index, and a temporal concentration ratio. In this run, spread FIFO had global stale-event rate 0.211 and maximum-window rate 0.324; clustered FIFO had 0.188 and 0.250. Thus equal mean delay did not preserve equal stale-event mass, and clustering did not automatically increase concentration in this model. The analogy is diagnostic only. It does not identify a plurisubharmonic function, a Monge-Ampere measure, or a theorem about queue dynamics.

All results are exploratory.
