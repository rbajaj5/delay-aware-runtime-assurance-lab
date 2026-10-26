# 工作记忆 — working memory in queue-aware autonomy

A 2:40 English-narrated explainer with Chinese definitions, matched multi-agent
queue replays, and executable basketball, gardening, and laundry examples.

- [Watch the generated 1080p video](delivery/Working_Memory_Assured_Autonomy.mp4)
- [Captions](delivery/Working_Memory_Captions.srt)
- [Definitions, sources, and interpretation](delivery/Sources_and_Methods.txt)
- [Poster](delivery/Working_Memory_Poster.png)

This example belongs to **delay-aware-runtime-assurance-lab**, not the separate
literature-retrieval/RAG project. It does not change registered course-study
results or the existing queue and laundry implementations.

## What actually runs

`run_evidence.py` imports the existing
`analysis/run_fair_queue_multiagent_simulation.py` and calls its unmodified
`run_game`. It runs ten identical seeds per policy (20261026–20261035), then
observes a seed-20261026 replay with a Python trace hook. Traced results must
exactly equal a normal run. Saved snapshots independently reproduce mismatch
counts and useful/background service totals. All policies see the same state
trajectory and arrival volume.

The setting is 12 agents on a ring, 180 rounds, three-round update delay, and
two service slots per agent per round. Each agent receives two neighbor updates
plus Poisson background traffic with mean four per round. The queue storage is
unbounded and the offered load exceeds service capacity.

| Policy | Mean memory mismatch | Mean background messages served |
|---|---:|---:|
| FIFO | 49.9% | 2,861.9 |
| Newest-update priority | 22.4% | 46.7 |
| Priority plus aging | 34.1% | 2,127.4 |

Mismatch is the fraction of 4,320 neighbor-memory checks per run that disagree
with evaluator world state; unknown memory also counts. Agents do not access
this evaluator truth. These are new paired runs, not the historical sweep's
numbers. Aging reserves one slot for an eligible background message once its
wait reaches six rounds. It does not establish a latency bound under overload.
The original `priority_deduplicating` selects the newest due update per sender
for service, but **does not purge older queued copies**. The film calls it
“newest-update priority” for that reason.

`domain_examples.py` executes three new discrete examples:

1. **Basketball:** target coverage changes after planning; a current-condition
   check rejects the pass to the covered wing and allows a fresh plan to center.
2. **Multi-agent gardening:** two gardeners remember a dry plant. The first
   accepted action supplies 60 mL and changes the resource version; the second
   plan is rejected. Repeating the original identity produces no additional dose.
3. **Multi-agent laundry:** two loaders remember an empty shared washer. One
   load changes its version; the competing plan is rejected. A retry is harmless.

Each scenario also rejects an old observation and checks that exactly one
effect occurred. An identity collision is tested separately. The authority
serializes checks and effects; this is a centralized teaching model, **not**
distributed consensus, multi-robot physics, or crash-safe exactly-once execution.
The 0.75-second freshness window is an illustrative choice for these examples.

The example also imports the real `Coordinator` and `SkillReceiver` from the
sibling [humanoid-laundry-queues](https://github.com/rbajaj5/humanoid-laundry-queues)
repository. Its existing 16 protocol tests are rerun. A separate synthetic
sequence demonstrates stable command identity, duplicate suppression, rejection
of old observations, receipt/completion separation, and a fresh postcondition.
This uses that supervisor's default 0.25-second observation-age threshold.

The film includes seconds 8–20 of the existing laundry task recording, cropped
to its robot view. This retained simulation is **one humanoid**; it is not the
new multi-agent laundry example. No physics or GPU experiment was rerun here.

## Reproduce

Use Python 3.11+, NumPy, Pillow, and Matplotlib (the original queue module imports
it), plus FFmpeg and FFprobe on PATH. Video typography and local speech currently
use Windows Segoe UI/Microsoft YaHei and Microsoft Zira Desktop.

The sibling `humanoid-laundry-queues` clone must contain `laundry_task/` and its
retained `output/laundry_task_silent.mp4`. The code and protocol tests do not
require MuJoCo or GPU execution. The physics clip is a retained-media dependency.

From this folder:

```powershell
python run_evidence.py
python domain_examples.py
python make_video.py --prepare --preview
.\narrate.ps1
python make_video.py --render
python make_video.py --mux
python verify.py
```

For a fast standalone check of the three new domain examples, run only
`python domain_examples.py`. No sibling repository, GPU, or media is needed.

`evidence/` contains the raw matched runs, per-round replay, supervisor event
trace, test log, and validation. `delivery/` holds the shareable video, captions,
poster, and source note. `build/` contains disposable render/audio intermediates
and is ignored by Git. The generated MP4 is included in `delivery/`; it is
separate from the repository's older GitHub release assets.

## Definitions and scope

The [Chinese Academy of Sciences psychology toolbox](https://experiment.psych.ac.cn/WM)
defines working memory using the phrase “在短时间内存储和加工有限信息”. The film translates
this as holding and processing a limited amount of information over a short time.
[Tencent Hunyuan](https://memory.hunyuan.tencent.com/) describes agent working
memory with “记住任务做到哪里、经验何时适用”: remember where the task stands and when experience
applies. Sources were accessed October 26, 2026; English translations are ours.
No Tencent model or memory framework is used in the executable demonstration.

The connection from cognitive working memory to these engineering state stores
is an analogy. Neither a successful replay nor passing tests constitutes a
general safety guarantee. Code/media were prepared with OpenAI Codex assistance;
the new film uses local synthetic English narration and data-driven diagrams.
Existing source and clip are credited to Ravi Bajaj's repositories.
