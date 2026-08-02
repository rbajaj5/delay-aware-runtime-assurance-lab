# When Messages Arrive Late: Queue-Aware Coordination in a Multi-Agent Landing Benchmark

**Draft manuscript - exploratory course project**
**Repository:** `rbajaj5/delay-aware-runtime-assurance-lab`
**Artifact date:** 2026-08-02

## Abstract

Delayed messages create a coordination problem that is easy to describe and easy to underestimate: a command can be delivered after the time window in which it was useful, while a receiver that accepts it has no direct indication that it is stale. This paper studies that problem in one controlled multi-agent benchmark. Eight agents receive landing clearances for three shared pads through heterogeneous communication delays. We compare a delay-blind consumer, which accepts the first delivered clearance, with a queue-aware consumer, which rejects clearances older than a freshness threshold or outside their declared slot. The same message schedule and kinematic trace are used for both policies; the three-dimensional scene is a visualization of the coordination model, not a helicopter flight-dynamics model.

At the selected freshness threshold of three steps, the delay-blind policy accepts and lands all eight agents but produces 846 conflict pair-steps across 132 conflict steps and seven distinct conflict pairs. The queue-aware policy rejects four stale clearances, lands four agents, and produces 115 conflict pair-steps across 115 conflict steps and one distinct conflict pair. A threshold sweep exposes the central tradeoff: stricter freshness checks reduce conflict mass but reduce throughput. The result is therefore not that queue awareness is universally safer. It is that explicit message-age and expiry semantics can move a delayed coordination system along a measurable conflict-throughput frontier in this benchmark.

## 1. Introduction

Runtime assurance is often presented as a control problem: observe the system, evaluate a constraint, and select an action. In distributed systems, however, the assurance layer may first have to decide whether the information used by that action is still valid. A landing clearance, resource lease, or collision-avoidance instruction can be logically correct when issued and operationally wrong when consumed.

This paper isolates that issue from vehicle-specific dynamics. The benchmark has eight agents, three shared landing pads, a fixed clearance schedule, and heterogeneous message delays. The agents move kinematically toward their assigned pads. A conflict occurs when two active agents assigned to the same pad are closer than a fixed separation threshold. The experiment compares two message-consumption rules:

1. **Delay-blind:** accept a delivered clearance without checking its age.
2. **Queue-aware:** accept only a delivered clearance whose age is within a freshness threshold and whose current time is inside its declared service slot.

The research question is deliberately narrow:

> Under a fixed delayed message schedule and finite shared capacity, does explicit queue semantics reduce conflict concentration, and what throughput cost does that reduction impose?

The contribution is an interpretable mechanism result, not a general safety theorem. The experiment makes stale information visible as a graph process: a delayed acceptance can create an additional active node on a shared pad, and the resulting conflict edges can persist even after the original message delay has passed.

## 2. Queueing perspective

The model follows standard queueing intuition: arrivals, service opportunities, waiting, delay, finite shared capacity, and a policy for deciding which work remains admissible. Zukerman's queueing notes emphasize simulation as a way to connect these concepts to concrete stochastic models and discuss delay, loss, priority, and queueing networks as central cases. Our benchmark uses that vocabulary but does not claim to instantiate a real teletraffic system or a validated aircraft operations model. [1]

The important modeling decision is to treat a clearance as a time-bounded message rather than as an eternal fact. A clearance is represented as

\[
m_i = (i, p_i, t_i^{\mathrm{issue}}, t_i^{\mathrm{deliver}}, s_i, e_i, v_i),
\]

where `i` is the agent, `p_i` is its assigned pad, `t_i` is the issue time, `s_i` and `e_i` delimit the declared service slot, and `v_i` is a message version. At simulation step `t`, its age is

\[
a_i(t) = t - t_i^{\mathrm{issue}}.
\]

The delay-blind policy accepts the message at or after `t_i^{deliver}`. The queue-aware policy accepts only when

\[
a_i(t) \leq \tau
\quad\text{and}\quad
t \leq e_i,
\]

where \(\tau\) is the configured freshness threshold. The slot test is a second, independent expiry rule: a message can be young enough but still arrive after its assigned opportunity.

## 3. Experimental design

### 3.1 Environment

The benchmark contains eight agents and three landing pads. Each agent starts at a distinct position above a procedural terrain surface and moves toward its assigned pad after accepting a clearance. The same pad assignment and clearance schedule are replayed under both policies. Delays are heterogeneous and range from zero to six simulation steps. The motion rule is intentionally simple: each active agent advances a bounded fraction of its remaining displacement toward its pad.

This abstraction is useful because it leaves the information problem exposed. It does not include rotor dynamics, wind, vehicle stability, aerodynamic interactions, or a validated air-traffic separation model. The 3D rendering supplies spatial context; it is not evidence that a real helicopter would behave this way.

### 3.2 Policies

The delay-blind policy models a receiver that treats delivery as sufficient evidence of validity. Once a clearance arrives, the agent enters the approach phase and continues toward the assigned pad.

The queue-aware policy models a receiver with explicit freshness and expiry semantics. A clearance that arrives too late is rejected and recorded as `reject_stale`. The policy does not synthesize a replacement clearance; rejection therefore exposes the throughput cost of refusing information that may no longer be safe to use.

### 3.3 Conflict graph

At every step, the active agents form a temporal conflict graph \(G_t=(V_t,E_t)\). An edge \((i,j)\) is present when both agents are active, have the same assigned pad, and their Euclidean separation is below 2.0 model units:

\[
(i,j) \in E_t
\iff
p_i=p_j,
\quad
\|x_i(t)-x_j(t)\|_2 < 2.0.
\]

The primary conflict measure is **conflict pair-steps**,

\[
C = \sum_t |E_t|,
\]

which counts the accumulated edge mass. We also report conflict steps,

\[
T_C = \sum_t \mathbf{1}(|E_t|>0),
\]

the number of distinct conflict pairs, the largest connected component, and triangle motifs. These network quantities are descriptive: they make concentration and persistence visible, but they are not certificates of safety.

### 3.4 Threshold sweep

The main comparison uses \(\tau=3\). To expose sensitivity to the queue rule, we repeat the queue-aware policy for \(\tau\in\{0,1,2,3,4,5,6,7\}\). This is not a statistical hypothesis test. It is a mechanism sweep over the same deterministic schedule.

## 4. Results

### 4.1 Main comparison

| Policy | Accepted | Rejected stale | Landed | Conflict steps | Conflict pair-steps | Distinct conflict pairs |
|---|---:|---:|---:|---:|---:|---:|
| Delay-blind | 8 | 0 | 8 | 132 | 846 | 7 |
| Queue-aware, \(\tau=3\) | 4 | 4 | 4 | 115 | 115 | 1 |

The delay-blind policy maximizes completion in this schedule, but it admits all delayed clearances and accumulates a dense conflict trace. Its peak conflict graph has seven edges and a largest connected component of three agents. The queue-aware policy rejects four late clearances. It does not eliminate every conflict, but it reduces total conflict edge mass by 86.4%, from 846 to 115, and reduces the peak conflict graph from seven edges to one. Its largest connected component is two agents, and no triangle occurs.

The queue-aware result should not be described as a free safety improvement. Four agents do not complete because the policy has no recovery or reallocation mechanism after rejection. In this benchmark, queue awareness changes the operating point by withholding stale work; it does not solve resource allocation after the withholding decision.

### 4.2 Freshness threshold sweep

| \(\tau\) | Accepted | Landed | Conflict steps | Conflict pair-steps | Distinct conflict pairs |
|---:|---:|---:|---:|---:|---:|
| 0 | 1 | 1 | 0 | 0 | 0 |
| 1 | 3 | 3 | 0 | 0 | 0 |
| 2 | 4 | 4 | 115 | 115 | 1 |
| 3 | 4 | 4 | 115 | 115 | 1 |
| 4 | 5 | 5 | 127 | 357 | 3 |
| 5 | 7 | 7 | 132 | 717 | 6 |
| 6 | 8 | 8 | 132 | 846 | 7 |
| 7 | 8 | 8 | 132 | 846 | 7 |

The sweep is monotone in completion but not equivalent to a smooth safety curve. The first two thresholds prevent all measured conflict edges by accepting only one or three clearances. At thresholds two and three, one conflict pair persists. At thresholds four through six, additional accepted clearances cause a sharp increase in conflict mass. Thresholds six and seven reproduce the delay-blind outcome because all eight clearances are accepted.

### 4.3 Network interpretation

The temporal graph provides the mechanism-level explanation. Under delay-blind consumption, stale clearances create multiple active agents assigned to the same pad. The resulting graph is not merely larger; it is more connected: 846 total conflict edges, seven peak edges, and 115 steps containing triangle motifs. Under the queue-aware rule, the conflict graph contains 115 total edges, a peak of one edge, no triangles, and a largest component of two.

This is the narrow sense in which queue semantics help: they prevent some delayed messages from becoming active coordination commitments. They do not establish that a queue-aware controller is safe in a physical vehicle, and they do not show that rejecting stale work is preferable when mission completion is the dominant objective.

## 5. Discussion

The experiment clarifies a distinction that is often blurred in delay-aware designs. A system can be **delay-aware in its model** while remaining **delay-blind in its admission rule**. If the receiver does not attach validity to issue time, slot time, or version, then a delayed message can still enter the active plan as though it were current.

The benchmark also shows why the right comparison is not simply "aware versus unaware." A queue-aware policy has an explicit cost model. Here the cost is visible as rejected clearances and reduced completion. A deployable system would need a second-stage response: reissue, reroute, reserve another pad, or safely wait. Without that response, queue awareness is a gate, not a complete runtime assurance architecture.

The graph representation is useful for analysis and communication because it separates the information mechanism from the visual surface. The same construction could be instantiated with warehouse robots, delivery vehicles, software leases, or distributed read receipts. The transferable object is the message-validity rule and the temporal conflict graph, not the fictional terrain or the helicopter glyph.

## 6. Limitations

This is one exploratory schedule, not a population study. It does not support a general claim about delay-aware assurance, helicopter safety, aviation operations, or the superiority of one queue discipline. The agents use a kinematic motion rule; no aerodynamic, actuator, wind, sensing, or human-operator model is present. The 2.0-unit conflict threshold and freshness thresholds are experimental definitions within the benchmark.

The result also contains a deliberate throughput confound: rejected messages are not replaced. Future work should compare bounded aging, priority with starvation protection, explicit reallocation, and admission control under a family of delay schedules and loads. Those experiments would test whether the conflict reduction can be retained without sacrificing completion so severely.

The paper does not use prime labels, Gaussian-prime geometry, Nakayama's lemma, random-turn Hex, or Green's functions as decorative analogies. Those ideas may be interesting in other projects, but they are not needed to explain the measured mechanism here. Adding them without a formal role would obscure the queue semantics that the experiment actually tests.

An optional appendix artifact uses prime labels as a graph fingerprint. It assigns the first eight primes to the agents, multiplies endpoint labels for an edge, adds edge codes, divides by the number of available pairs, subtracts the two policy scores, and exponentiates by consecutive edge persistence. This is a reproducible serialization of the existing graph, not an additional scientific result; any injective node labels would preserve the comparison.

## 7. Conclusion

In this multi-agent landing benchmark, delayed messages create conflicts when delivery is treated as proof of current validity. A queue-aware consumer that checks message age and slot expiry reduces accumulated conflict edges and connected conflict structure, but it does so by rejecting work and lowering completion. The result is a measurable tradeoff between stale-information exposure and throughput.

The appropriate conclusion is therefore modest and useful: explicit queue semantics are a promising runtime-assurance primitive for delayed coordination, but they must be paired with recovery and capacity-management policies before they can support a stronger safety claim. The benchmark supplies a compact test harness for that next question.

## Reproducibility

The source code and retained artifacts are in the public repository:

- `analysis/run_helicopter_3d_landing_simulation.py`
- `analysis/analyze_helicopter_network.py`
- `analysis/render_3d_coordination_graph.py`
- `artifacts/20260802/helicopter_3d/helicopter_3d_step_trace.csv`
- `artifacts/20260802/helicopter_3d/helicopter_3d_threshold_sweep.csv`
- `artifacts/20260802/helicopter_3d/helicopter_network_summary.csv`
- `artifacts/20260802/helicopter_3d/coordination_graph_3d_summary.csv`

Each artifact directory includes a SHA-256 manifest. The retained trace is the source for the network diagnostics and the graph-only replay; those postprocessing steps do not generate a new trajectory.

The optional prime-label diagnostic is generated by `analysis/derive_prime_graph_signature.py` and is recorded in `artifacts/20260802/helicopter_3d/HELICOPTER_PRIME_GRAPH_MANIFEST.sha256`.

## References

1. M. Zukerman, *Introduction to Queueing Theory and Stochastic Teletraffic Models*, lecture notes, 2000-2025. https://www.ee.cityu.edu.hk/~zukerman/classnotes.pdf
