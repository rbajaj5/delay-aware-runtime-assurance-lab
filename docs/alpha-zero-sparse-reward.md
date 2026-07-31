# AlphaZero, Sparse Rewards, and Trace-Level Evaluation

Reference: Brent Kong, Tejas Ram, and Tony Yue Yu, "AlphaZero in Sparsely Rewarded Games: Limits and Auxiliary Supervision," [arXiv:2607.08984](https://arxiv.org/abs/2607.08984).

## Why it matters here

The paper separates strong game performance from exact oracle consistency. Its examples use Connect Four and Chomp, where exact game-theoretic values or invariants make it possible to inspect the full trajectory rather than rely only on the final win rate. The paper reports that vanilla AlphaZero can play strongly while still deviating from optimal lines or failing to preserve an exact invariant; auxiliary oracle-derived supervision improves trace consistency but does not make every task perfect.

That is a useful measurement lesson for this repository. A delayed multi-agent system should not be evaluated only by endpoint reward or whether an episode completes. We should also record:

- the first step at which an agent acts on stale information;
- whether a sequence or protocol invariant is preserved;
- the fraction of actions matching a known local oracle;
- the relationship between trace consistency and endpoint reward.

Our hidden-bit signaling diagnostic already has a local oracle: the current hidden bit is the correct action. Its `reward_rate` is therefore an oracle-match rate, while `misattributed_updates` identifies a mechanism-level failure in delay-blind feedback handling. The next richer benchmark can add an exact small-game oracle to the existing multi-agent trace format.

## Boundary of the citation

This paper is not evidence about delayed communication, queue-aware monitoring, or runtime assurance. It studies neural-guided Monte Carlo Tree Search in sparsely rewarded games, with auxiliary supervision from exact or oracle-derived structure. We use it here for the evaluation design principle: strong aggregate performance is not equivalent to exact trace correctness.
