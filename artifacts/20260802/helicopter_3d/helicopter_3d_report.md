# 3D Helicopter Landing Coordination

This is an exploratory kinematic communication experiment, not a validated helicopter flight-dynamics model.
Both policies use the same eight helicopters, three pads, clearance schedule, and heterogeneous message delays.
The queue-aware policy rejects clearances older than the three-step freshness threshold or past their slot expiry.

| policy | accepted | stale rejected | landed | conflict steps | conflict pairs | pair-conflict count |
|---|---:|---:|---:|---:|---:|---:|
| delay_blind | 8 | 0 | 8 | 126 | 7 | 807 |
| queue_aware | 4 | 4 | 4 | 112 | 1 | 112 |

The comparison is a mechanism diagnostic: freshness checks trade some late clearances for fewer pad conflicts in this toy schedule.
It should not be interpreted as evidence about real helicopter handling qualities or aviation operations.
