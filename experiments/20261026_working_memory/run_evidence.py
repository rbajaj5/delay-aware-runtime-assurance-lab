"""Run existing repository code unchanged; observe it for the working-memory film."""
from pathlib import Path
import csv, hashlib, importlib.util, inspect, io, json, sys, unittest

HERE = Path(__file__).resolve().parent
OUT = HERE / 'evidence'
OUT.mkdir(exist_ok=True)
ROOT = HERE.parents[1]
PARENT = ROOT.parent
QUEUE = ROOT / 'analysis/run_fair_queue_multiagent_simulation.py'
LAUNDRY = PARENT / 'humanoid-laundry-queues'
sys.dont_write_bytecode = True

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m

m = load('working_memory_original_queue', QUEUE)
seed0 = 20261026
rows, traces = [], {}
for policy in m.POLICIES:
    for seed in range(seed0, seed0 + 10):
        rows.append(m.run_game(3, 4.0, policy, seed))
    # Observe an actual call to run_game; no copy or change to the algorithm.
    snapshots, chosen_this_round = [], []
    observed_created = [{} for _ in range(m.AGENTS)]
    original_select = m.select_due
    line = next(i for i, s in enumerate(QUEUE.read_text().splitlines(), 1)
                if s.strip() == 'total_backlog += backlog')
    def record_select(queue, policy, now):
        receiver = len(chosen_this_round)
        chosen, remaining = original_select(queue, policy, now)
        chosen_this_round.append([vars(v) for v in chosen])
        for msg in chosen:
            if msg.useful:
                observed_created[receiver][msg.sender] = msg.created
        return chosen, remaining
    def tracer(frame, event, arg):
        if frame.f_code is m.run_game.__code__:
            if event == 'line' and frame.f_lineno == line:
                v = frame.f_locals
                snapshots.append(dict(round=v['now'], truth=list(v['state']),
                    memory=[dict(a) for a in v['latest_seen']],
                    memory_created=[dict(a) for a in observed_created],
                    queue_sizes=[len(q) for q in v['queues']],
                    queue_useful=[sum(x.useful for x in q) for q in v['queues']],
                    served=[list(a) for a in chosen_this_round],
                    stale_total=v['stale_decisions'], decisions=v['decisions'],
                    background_total=v['background_processed'],
                    useful_total=v['useful_processed']))
                chosen_this_round.clear()
            return tracer
        return None
    m.select_due = record_select
    try:
        sys.settrace(tracer)
        traced = m.run_game(3, 4.0, policy, seed0)
    finally:
        sys.settrace(None)
        m.select_due = original_select
    direct = next(x for x in rows if x['seed'] == seed0 and x['policy'] == policy)
    assert traced == direct, 'Instrumentation changed the simulation result'
    assert len(snapshots) == m.ROUNDS
    wrong = sum(s['memory'][i].get(j) != s['truth'][j]
                for s in snapshots for i in range(m.AGENTS) for j in m.neighbors(i))
    assert wrong == snapshots[-1]['stale_total']
    assert wrong / (m.ROUNDS*m.AGENTS*2) == direct['stale_decision_rate']
    assert sum(not x['useful'] for s in snapshots for a in s['served'] for x in a) == direct['background_processed']
    assert sum(x['useful'] for s in snapshots for a in s['served'] for x in a) == direct['useful_processed']
    traces[policy] = snapshots

# Same exogenous world and arrival volumes for every policy at every round.
policies = list(m.POLICIES)
for i in range(m.ROUNDS):
    a = traces[policies[0]][i]
    for p in policies[1:]:
        b = traces[p][i]
        assert a['truth'] == b['truth']
        assert sum(a['queue_sizes'])+a['useful_total']+a['background_total'] == sum(b['queue_sizes'])+b['useful_total']+b['background_total']

summary = m.aggregate(rows)
with (OUT/'matched_runs.csv').open('w', newline='', encoding='utf-8') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
(OUT/'queue_trace.json').write_text(json.dumps(traces),encoding='utf-8')
(OUT/'queue_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')

# Execute the real supervisor and its existing protocol tests, with synthetic observations.
sys.path.insert(0, str(LAUNDRY))
from laundry_task.supervisor import Coordinator, SkillReceiver, Observation
suite = unittest.defaultTestLoader.loadTestsFromName('laundry_task.test_supervisor')
stream=io.StringIO()
result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
assert result.wasSuccessful()
(OUT/'protocol_tests.txt').write_text(stream.getvalue(),encoding='utf-8')
c,r=Coordinator('wm-demo'),SkillReceiver('wm-demo')
def obs(t, seq, opened=False):
    return Observation(seq,t,True,1.1 if opened else 0.,False,False,True)
steps=[]
def save(t,title,status):
    steps.append(dict(time_s=t,title=title,status=status,
        pending=c.pending.command_id if c.pending else None,
        completed=list(c.completed),receiver_active=r.active,
        executed_ids=list(r.seen),events=list(c.events)))
cmd=c.update(0.,obs(0.,0));save(0.,'Issue OPEN','ISSUE')
status=r.dequeue(cmd,.1,obs(.1,1));assert status=='EXECUTE';save(.1,'Start one execution',status)
retry=c.update(.8,obs(.8,2));assert retry==cmd
status=r.dequeue(retry,.8,obs(.8,2));assert status=='DUPLICATE_NO_EXECUTION'
save(.8,'No receipt: retry same identity',status)
c.update(1.,obs(.2,3,True));assert c.pending==cmd
save(1.,'Old observation rejected','REJECT_OBSERVATION')
c.receive_receipt(cmd.command_id,1.1,'done');assert c.pending==cmd and c.completed==[]
save(1.1,'Even a done receipt is not completion','RECEIPT_ONLY')
fresh=obs(1.2,4,True);assert r.finish(cmd.command_id,1.2,fresh)
c.update(1.2,fresh);assert c.completed==['open_door'] and c.pending is None
save(1.2,'Fresh physical state verifies opening','VERIFIED_POSTCONDITION')
(OUT/'protocol_trace.json').write_text(json.dumps(steps,indent=2),encoding='utf-8')

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
validation=dict(queue_runs=30,matched_seeds=list(range(seed0,seed0+10)),
    replay_seed=seed0,delay_rounds=3,background_mean_per_agent_round=4,
    agents=12,rounds=180,service_capacity_per_agent_round=2,
    instrumentation_matches_direct_execution=True,trace_metrics_recomputed=True,
    matched_world_and_arrival_volume=True,protocol_tests_run=result.testsRun,
    protocol_tests_passed=True,protocol_demo_passed=True,
    source_sha256={p.relative_to(PARENT).as_posix():sha(p) for p in [QUEUE,LAUNDRY/'laundry_task/supervisor.py',LAUNDRY/'laundry_task/test_supervisor.py']},
    scope='New paired queue simulation plus synthetic supervisor test; retained robot video is not a new physics run.',
    limitations=['Unbounded queue storage; finite service capacity. Offered load exceeds service rate.',
      'Original priority policy selects newest due updates but does not purge obsolete queued messages.',
      'State mismatch uses evaluator truth unavailable to agents; unknown memory counts as mismatch.',
      'Supervisor uses an in-memory ledger, one robot session, and no crash-recovery guarantee.',
      'No hardware validation, human-cognition model, or general safety certificate.'])
(OUT/'validation.json').write_text(json.dumps(validation,indent=2),encoding='utf-8')
print(json.dumps(summary,indent=2));print(f'Validated {result.testsRun} protocol tests and {len(rows)} queue runs.')
