# Fair Queueing for Delayed Multi-Agent Coordination

This sweep compares three finite-capacity queue disciplines on a 12-agent ring. Each agent receives useful state updates from its two neighbors and low-value background messages. The service capacity is 2 messages per agent per round, and useful messages are delayed by the tested horizon.

- `fifo_blind`: process due messages by arrival order.
- `priority_deduplicating`: retain the newest useful message per sender and serve useful traffic first.
- `priority_with_aging`: use the same useful-message priority, but reserve service for the oldest background message once it has waited at least 6 rounds.

The downstream metric is stale-decision rate: an agent's decision is stale when its latest applied neighbor state differs from that neighbor's current state. The figure shows the delay-3 slice; the CSV includes all tested delays and loads.

The aging policy is a trade-off probe, not a deployment recommendation. Its parameters, starvation bounds, message taxonomy, authentication, auditability, and failure handling would need to be specified before use in a safety-relevant system.

All results are exploratory.
