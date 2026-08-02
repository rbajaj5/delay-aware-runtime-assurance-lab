# Dependency-Graph Cut Diagnostic

## Purpose

This note specifies how to test whether a dependency representation is
fragile under delayed or missing communication. It is a methodology for a
future corpus experiment, not a result about the Quran, Ruhnama, Wikipedia,
or any other text.

## Graph representation

Represent a sentence or discourse unit by a weighted dependency graph
`G = (V, E, w)`:

- `V` contains words, morphemes, or other explicitly chosen units.
- `E` contains annotated dependency relations.
- `w(e)` records the importance assigned by the experiment, for example a
  predicate-argument relation, a cross-clause relation, or a local modifier.
- Each edge also receives a version and delivery time when used in the
  delayed-message experiment.

The annotation scheme must be held fixed within a comparison. A Quranic
dependency treebank, Universal Dependencies treebank, and a Turkmen corpus
should not be compared as though their labels were automatically identical.

## Bounded use of Goemans-Williamson

If the intended reference is the Goemans-Williamson semidefinite approach to
Max-Cut, use it only as a cut diagnostic. Construct an objective that rewards
cuts separating a selected semantic root or clause from high-weight dependent
edges. The rounded cut then proposes a high-impact set of dependency edges to
delay, drop, or deliver out of order.

The diagnostic answers a narrow question:

> Which approximately high-impact partition of this graph should be used to
> stress the communication protocol?

It does **not** by itself measure linguistic complexity, semantic importance,
truth, interpretive difficulty, or safety. The approximation guarantee applies
to the stated cut objective, not to any informal claim about the text.

## Independent measurements

Report the following separately from the cut score:

- node and edge counts;
- weighted edge density;
- dependency depth and maximum dependency length;
- cross-clause edge fraction;
- articulation points and low-connectivity regions;
- redundancy under edge deletion;
- parse or task recovery after delayed-edge injection;
- stale decisions, contradictory actions, false consensus, and recovery time.

This separation prevents a high cut value from being mislabeled as
"complexity." A short graph can have a fragile bottleneck, while a larger
graph can be more delay-tolerant because it contains redundant paths.

## Multi-agent protocol experiment

For each graph, create agents with local graph views and a shared versioned
message stream. Compare:

1. a delay-blind agent that uses the newest locally visible graph;
2. a queue-aware agent that tracks edge age and provenance;
3. a recovery agent that requests or reconstructs missing edges.

Apply ordinary random edge delays and the Goemans-Williamson-proposed stress
cuts as separate conditions. Do not conflate the adversarial cut condition
with a representative communication distribution.

The main comparison is the recovery curve as delayed-edge mass increases. The
pre-registered reading should distinguish loss of local syntax, loss of
cross-agent agreement, and actual task failure.

## Corpus and interpretation safeguards

Ruhnama, the Quranic Arabic Corpus, English Wikipedia, and Simple English
Wikipedia can serve as different corpus or annotation case studies only if
the source texts, editions, tokenization, and annotation provenance are
recorded. A corpus comparison should not assign a fixed cultural ranking such
as "simple" or "complicated" to a religious or political text. Test both
directions: longer graphs may be harder to transmit but more redundant, while
shorter graphs may be cheaper to transmit but more vulnerable to one missing
dependency.

No numerical conclusion should be reported until the source corpus and parser
outputs are hash-pinned and the same graph construction procedure is applied
to every comparison condition.
