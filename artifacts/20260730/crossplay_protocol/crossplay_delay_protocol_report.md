# Delayed Protocol Formation under Cross-Play

This toy diagnostic takes the design lessons of Gessler et al., `OvercookedV2: Rethinking Overcooked for Zero-Shot Coordination` (arXiv:2503.17821), as a template: one agent has asymmetric information, the partner acts from partial observations, communication is stochastic and delayed, and the pair may need to form a protocol at test time.

The sender observes a hidden bit that can switch over time and transmits the bit through a convention-dependent binary signal. Feedback reports whether the receiver's action matched the current hidden bit. Signal and feedback delays include one-step jitter, creating possible reordering.

- `fixed_protocol`: assumes convention 0 and never adapts.
- `delay_blind_adaptive`: updates its convention belief using the latest arrived signal, ignoring sequence identity.
- `queue_aware_adaptive`: pairs feedback with the sequence-indexed signal/action record.
- `known_protocol`: reference policy given the sender's convention.

The plotted curves separate sender convention 0 and 1 to expose cross-play mismatch. This is not an implementation of OvercookedV2, and the results are not evidence about that benchmark. They are an exploratory test of whether sequence-aware feedback helps a minimal protocol learner under delay.
