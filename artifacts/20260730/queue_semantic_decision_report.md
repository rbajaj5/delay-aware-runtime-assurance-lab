# Queue-Semantic Decision Simulation

Agents act on the latest neighbor states they have applied. A useful update is stale when the applied value differs from the sender's current state. Background traffic shares the same finite service capacity.

`fifo_blind` processes due messages by arrival order. `priority_deduplicating` retains the newest due useful message from each sender and serves useful messages before background traffic. This is an exploratory queueing diagnostic, not a safety certificate or a human-value ranking.

The summary reports stale-decision rate, useful-message wait, background service, and backlog across delay and load settings. The plotted slice uses delay 3.

All results are exploratory; deployment would require explicit message classes, bounded starvation, auditability, and failure handling.
