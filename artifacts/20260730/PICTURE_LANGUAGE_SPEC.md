# Picture Language Specification for Delayed Multi-Agent Hex

This artifact applies the abstraction/simulation discipline described by Jaffe
and Liu in [A Mathematical Picture Language](https://arthurjaffe.com/Assets/pdf/PictureLanguage.pdf).
It is a visualization specification, not a new physical model.

## Primitive Symbols

| Symbol | Meaning | Data source |
|---|---|---|
| `T` | Authoritative board state | `true_board` in trace rows |
| `L_a` | Agent `a`'s delivered local board | `local_board` in trace rows |
| `Q_a` | Messages pending for agent `a` | `pending_count` and queue rows |
| `O_a = L_a ⊕ Q_a` | Queue-aware overlay used for a decision | deterministic overlay operation |
| `R_a` | Local red-line trigger | opponent path cost `<= 2` |
| `F_a` | False red-line | local trigger true, authoritative trigger false |
| `M_a` | Missed red-line | local trigger false, authoritative trigger true |
| `C` | Completed Hex crossing | BFS crossing check on `T` |

## Composition Rules

1. `T` is authoritative; it is never replaced by a local view.
2. `L_a` is updated only when a message addressed to agent `a` is delivered.
3. `O_a` overlays only messages addressed to `a`; messages for another agent
   are not visible to that decision.
4. `R_a`, `F_a`, and `M_a` compare the same local and authoritative path-cost
   rule. They are not subjective labels.
5. `C` is evaluated on `T`, not on `L_a` or `O_a`.
6. A stale-cell conflict is drawn as a disagreement between the selected cell
   in `L_a`/`O_a` and occupancy in `T`.

## Visual Encoding

- Blue and yellow cells: player color.
- White cell: empty.
- Green dot: crossing-path cell.
- Red outline: stale-cell conflict.
- Orange line: blind-fleet summary.
- Blue line: queue-aware-fleet summary.
- Solid red-line curve: false triggers.
- Dashed `x` curve: missed triggers.

## Scope

The picture language is intended to make the delayed-state mechanism inspectable
in a presentation. It does not validate the external paper's quantum-field or
TQFT interpretations, and the resulting Hex experiments are not v6 quadrotor
evidence.
