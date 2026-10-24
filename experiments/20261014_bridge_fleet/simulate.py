"""Matched synthetic taxi experiment. No external service, upload, or GPU required.

Queue discipline adapts the finite-capacity FIFO/priority/dedup pattern in
analysis/run_fair_queue_multiagent_simulation.py. Route planning uses NetworkX.
Unlike the earlier queue sweep, seeds are paired across policies and exogenous
random streams are pre-generated independently of control decisions.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import heapq
import json
from collections import Counter
from functools import lru_cache
from pathlib import Path

import networkx as nx
import numpy as np

HERE = Path(__file__).resolve().parent
POLICY_LABELS = {"latest_received": "Latest received / FIFO", "priority_updates": "Priority updates", "freshness_monitor": "Freshness monitor"}
EVENT_FIELDS = ["scenario", "seed", "policy", "time_s", "event", "vehicle", "request", "edge", "detail"]


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def make_tape(config, seed):
    h, n = config["horizon_s"], config["fleet_size"]
    streams = [np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(4)]
    demand, initial, delay, background = streams
    nodes = list(config["nodes"])
    requests, now = [], 0.0
    while True:
        now += demand.exponential(1 / config["request_rate_per_s"])
        if now >= config["demand_end_s"]: break
        origin = str(demand.choice(nodes, p=[.20, .35, .20, .25]))
        destination = str(demand.choice([x for x in nodes if x != origin]))
        requests.append({"id": len(requests), "arrival_s": int(now), "origin": origin, "destination": destination})
    def delays(shape):
        base = delay.integers(config["delay_base_min_s"], config["delay_base_max_s"] + 1, shape)
        extra = delay.integers(config["delay_burst_extra_min_s"], config["delay_burst_extra_max_s"] + 1, shape)
        for start, end in config["delay_bursts_s"]: base[start:end] += extra[start:end]
        return base.tolist()
    tape = {"seed": seed, "requests": requests, "initial_nodes": initial.choice(nodes, n).tolist(),
            "road_delays": delays((h, n + 1, len(config["edges"]))),
            "status_delays": delays((h, n)), "command_delays": delays((h, n)),
            "background_counts": background.poisson([config["dispatcher_background_rate_per_s"]] + [config["vehicle_background_rate_per_s"]] * n, (h, n + 1)).tolist()}
    tape["sha256"] = fingerprint(tape)
    return tape


def actual_closed(scenario, t):
    return {e for block in scenario["closures"] if block["start_s"] <= t < block["end_s"] for e in block["edges"]}


def route(config, start, goal, knowledge, now, freshness=None):
    # Only received/local observations enter this graph. Physical truth is not an argument.
    available=[]
    for e in config["edges"]:
        closed, stamp = knowledge[e["id"]]
        if not closed and (freshness is None or now - stamp <= freshness):
            available.append((e["id"],e["u"],e["v"],e["travel_s"]))
    return cached_route(tuple(available),start,goal)


@lru_cache(maxsize=4096)
def cached_route(available,start,goal):
    graph=nx.MultiGraph();graph.add_nodes_from([start,goal])
    for key,u,v,weight in available:graph.add_edge(u,v,key=key,weight=weight)
    try: nodes = nx.shortest_path(graph, start, goal, weight="weight")
    except nx.NetworkXNoPath: return None
    edges = [min(graph[a][b], key=lambda k: (graph[a][b][k]["weight"], k)) for a, b in zip(nodes, nodes[1:])]
    length = sum(next(e[3] for e in available if e[0] == key) for key in edges)
    return nodes, edges, length


def select_ready(ready, policy, capacity):
    """Serve only arrived packets. Priority coalesces obsolete arrived snapshots."""
    if policy == "latest_received":
        ordered = sorted(ready, key=lambda m: (m["due"], m["seq"]))
        return ordered[:capacity], ordered[capacity:], 0
    latest = {}
    for m in ready:
        if m["kind"] in ("road", "status"):
            key = (m["kind"], m["key"])
            if key not in latest or (m["created"], m["seq"]) > (latest[key]["created"], latest[key]["seq"]): latest[key] = m
    kept = [m for m in ready if m["kind"] not in ("road", "status") or latest[m["kind"], m["key"]] is m]
    order = {"command": 0, "road": 1, "status": 2, "background": 3}
    kept.sort(key=lambda m: (order[m["kind"]], m["seq"]))
    return kept[:capacity], kept[capacity:], len(ready) - len(kept)


def simulate(config, scenario, policy, tape, capture=False, event_writer=None, request_writer=None):
    horizon, n = config["horizon_s"], config["fleet_size"]
    edges = {e["id"]: e for e in config["edges"]}
    seed = tape["seed"]
    # A common initial all-open observation and fleet inventory are supplied at t=0.
    knowledge = [{e: (False, 0) for e in edges} for _ in range(n + 1)]
    vehicles = [{"id": i, "node": node, "edge": None, "to": None, "progress_s": 0,
                 "task": None, "onboard": False, "last_done": None, "forced": False, "hold": None}
                for i, node in enumerate(tape["initial_nodes"])]
    observed = [{"node": v["node"], "idle": True, "last_done": None, "stamp": 0} for v in vehicles]
    reserved = [None] * n
    requests = {r["id"]: {**r, "assigned_s": None, "pickup_s": None, "completed_s": None, "vehicle": None} for r in tape["requests"]}
    arrivals = {}
    for r in requests.values(): arrivals.setdefault(r["arrival_s"], []).append(r["id"])
    arrived, future, ready = [], [[] for _ in range(n + 1)], [[] for _ in range(n + 1)]
    sequence = 0
    counts = Counter()
    traces, events = [], []
    faults = []
    def log(t, kind, vehicle="", request="", edge="", detail=""):
        row = dict(zip(EVENT_FIELDS, [scenario["id"], seed, policy, t, kind, vehicle, request, edge, detail]))
        if event_writer: event_writer.writerow(row)
        if capture: events.append(row)
    def emit(receiver, t, lag, kind, key, payload):
        nonlocal sequence
        sequence += 1
        message = {"seq": sequence, "created": t, "due": t + lag, "kind": kind, "key": key, "payload": payload}
        heapq.heappush(future[receiver], (message["due"], sequence, message))
    def settle(v, t):
        if v["task"] is None or v["edge"] is not None: return
        r = requests[v["task"]]
        if not v["onboard"] and v["node"] == r["origin"]:
            assert r["pickup_s"] is None
            r["pickup_s"] = t; v["onboard"] = True
            log(t, "pickup", v["id"], r["id"], detail=f"wait_s={t-r['arrival_s']}")
        if v["onboard"] and v["node"] == r["destination"]:
            assert r["completed_s"] is None
            r["completed_s"] = t; v["last_done"] = r["id"]
            log(t, "dropoff", v["id"], r["id"])
            v["task"] = None; v["onboard"] = False
    def snapshot(t):
        if not capture: return
        current = [requests[k] for k in arrived]
        traces.append({"time_s": t, "actual_closed": sorted(actual_closed(scenario, t)),
            "dispatcher_closed": sorted(k for k, value in knowledge[0].items() if value[0]),
            "dispatcher_road_age_s": {k: t-value[1] for k,value in knowledge[0].items()},
            "vehicles": [{**v, "known_closed": sorted(k for k, value in knowledge[v["id"]+1].items() if value[0])} for v in vehicles],
            "waiting_by_node": {node: sum(r["origin"]==node and r["pickup_s"] is None for r in current) for node in config["nodes"]},
            "arrived": len(current), "completed": sum(r["completed_s"] is not None for r in current),
            "waiting": sum(r["pickup_s"] is None for r in current),
            "onboard": sum(r["pickup_s"] is not None and r["completed_s"] is None for r in current),
            "closed_entry_attempts": counts["closed_entry_attempts"], "fallback_vehicle_s": counts["fallback_vehicle_s"],
            "queue_backlog": sum(len(q) for q in ready) + sum(len(q) for q in future)})
    snapshot(0)
    for t in range(horizon):
        closed = actual_closed(scenario, t)
        for block in scenario["closures"]:
            if t in (block["start_s"], block["end_s"]): log(t, "closure" if t==block["start_s"] else "reopening", detail=",".join(block["edges"]))
        boundary = any(t in (b["start_s"], b["end_s"]) for b in scenario["closures"])
        if t % config["road_update_period_s"] == 0 or boundary:
            for receiver in range(n + 1):
                for j, e in enumerate(edges): emit(receiver, t, tape["road_delays"][t][receiver][j], "road", e, e in closed)
        if t % config["status_update_period_s"] == 0:
            for v in vehicles:
                emit(0, t, tape["status_delays"][t][v["id"]], "status", v["id"],
                     {"node": v["node"], "idle": v["task"] is None, "last_done": v["last_done"]})
        for receiver in range(n + 1):
            for _ in range(tape["background_counts"][t][receiver]): emit(receiver, t, 0, "background", sequence, None)
            while future[receiver] and future[receiver][0][0] <= t: ready[receiver].append(heapq.heappop(future[receiver])[2])
            chosen, ready[receiver], discarded = select_ready(ready[receiver], policy, config["service_capacity_per_recipient_per_s"])
            counts["coalesced_messages"] += discarded
            for m in chosen:
                assert m["created"] <= m["due"] <= t
                if m["kind"] == "road":
                    key = m["key"]
                    if m["created"] >= knowledge[receiver][key][1]: knowledge[receiver][key] = (m["payload"], m["created"])
                    counts["road_messages"] += 1; counts["road_age_sum"] += t-m["created"]
                    log(t, "road_update", receiver-1 if receiver else "dispatcher", edge=key,
                        detail=f"closed={int(m['payload'])};created_s={m['created']};age_s={t-m['created']}")
                elif m["kind"] == "status":
                    i = m["key"]
                    if m["created"] >= observed[i]["stamp"]:
                        observed[i] = {**m["payload"], "stamp": m["created"]}
                        if reserved[i] is not None and m["payload"]["last_done"] == reserved[i]: reserved[i] = None
                elif m["kind"] == "command":
                    v=vehicles[receiver-1]
                    assert v["task"] is None, "one dispatcher reservation must prevent double booking"
                    v["task"] = m["payload"]
                    log(t, "assignment_received", v["id"], v["task"])
        for rid in arrivals.get(t, []):
            arrived.append(rid); log(t, "request_arrival", request=rid, detail=f"{requests[rid]['origin']}->{requests[rid]['destination']}")
        pending=[requests[rid] for rid in arrived if requests[rid]["assigned_s"] is None]
        dispatch_hold=set()
        for r in pending:
            candidates=[]
            for i, state in enumerate(observed):
                if reserved[i] is not None or not state["idle"]: continue
                if policy=="freshness_monitor" and t-state["stamp"] > config["status_freshness_s"]:
                    dispatch_hold.add(i); continue
                fresh=config["road_freshness_s"] if policy=="freshness_monitor" else None
                pickup=route(config,state["node"],r["origin"],knowledge[0],t,fresh)
                journey=route(config,r["origin"],r["destination"],knowledge[0],t,fresh)
                if pickup is None or journey is None:
                    if policy=="freshness_monitor": dispatch_hold.add(i)
                    continue
                candidates.append((pickup[2],i))
            if not candidates: continue
            _, i=min(candidates)
            r["assigned_s"]=t; r["vehicle"]=i; reserved[i]=r["id"]
            emit(i+1,t,tape["command_delays"][t][i],"command",r["id"],r["id"])
            log(t,"assignment_sent",i,r["id"],detail=f"observed_status_age_s={t-observed[i]['stamp']}")
        for v in vehicles:
            before_hold=v["hold"]; v["hold"]=None
            settle(v,t)
            if v["task"] is None:
                if policy=="freshness_monitor" and v["id"] in dispatch_hold and reserved[v["id"]] is None: v["hold"]="dispatch_freshness_or_route"
            elif v["edge"] is None:
                r=requests[v["task"]]; target=r["destination"] if v["onboard"] else r["origin"]
                fresh=config["road_freshness_s"] if policy=="freshness_monitor" else None
                planned=route(config,v["node"],target,knowledge[v["id"]+1],t,fresh)
                if planned is None:
                    v["hold"]="no_fresh_known_route" if policy=="freshness_monitor" else "no_known_route"
                elif planned[1]:
                    key=planned[1][0]
                    if key in closed:
                        counts["closed_entry_attempts"]+=1
                        log(t,"closed_entry_attempt",v["id"],v["task"],key)
                        knowledge[v["id"]+1][key]=(True,t)
                        v["hold"]="physical_barrier"
                    else:
                        edge=edges[key]; other=edge["v"] if edge["u"]==v["node"] else edge["u"]
                        assert v["node"] in (edge["u"],edge["v"])
                        if policy=="freshness_monitor":
                            assert all(not knowledge[v["id"]+1][e][0] and t-knowledge[v["id"]+1][e][1]<=config["road_freshness_s"] for e in planned[1])
                        log(t,"edge_entry",v["id"],v["task"],key,detail=f"{v['node']}->{other};onboard={int(v['onboard'])}")
                        v["edge"]=key; v["to"]=other; v["progress_s"]=0
            if v["edge"] is not None:
                key=v["edge"]; previous=v["progress_s"]
                if key in closed:
                    if not v["forced"]: log(t,"forced_stop",v["id"],v["task"],key)
                    v["forced"]=True; counts["forced_stop_vehicle_s"]+=1; v["hold"]="physical_closure"
                    assert v["progress_s"]==previous
                else:
                    if v["forced"]: log(t,"forced_resume",v["id"],v["task"],key)
                    v["forced"]=False; v["progress_s"]+=1
                    counts["occupied_motion_s" if v["onboard"] else "empty_motion_s"]+=1
                    if v["progress_s"]>=edges[key]["travel_s"]:
                        v["node"]=v["to"]; v["edge"]=None; v["to"]=None; v["progress_s"]=0
                        log(t+1,"edge_exit",v["id"],v["task"],key)
                        settle(v,t+1)
            if policy=="freshness_monitor" and v["hold"] in ("dispatch_freshness_or_route","no_fresh_known_route"):
                counts["fallback_vehicle_s"]+=1
            if v["hold"]!=before_hold:
                if v["hold"]: log(t,"wait_begin",v["id"],v["task"] if v["task"] is not None else "",detail=v["hold"])
                elif before_hold: log(t,"wait_end",v["id"],detail=before_hold)
        # Physical inventory and logical ownership must reconcile at every step.
        onboard=sum(v["onboard"] for v in vehicles)
        assert onboard==sum(r["pickup_s"] is not None and r["completed_s"] is None for r in requests.values())
        active=[v["task"] for v in vehicles if v["task"] is not None]
        assert len(active)==len(set(active))
        snapshot(t+1)
    trips=list(requests.values()); picked=[r for r in trips if r["pickup_s"] is not None]; complete=[r for r in trips if r["completed_s"] is not None]
    waits=[r["pickup_s"]-r["arrival_s"] for r in picked]
    capped=[(r["pickup_s"] if r["pickup_s"] is not None else horizon)-r["arrival_s"] for r in trips]
    waiting=sum(r["pickup_s"] is None for r in trips); onboard=len(picked)-len(complete)
    assert len(trips)==waiting+onboard+len(complete)
    for r in trips:
        assert r["pickup_s"] is None or r["arrival_s"]<=r["assigned_s"]<=r["pickup_s"]
        assert r["completed_s"] is None or r["pickup_s"]<=r["completed_s"]<=horizon
        if request_writer:
            request_writer.writerow({"scenario":scenario["id"],"seed":seed,"policy":policy,**r,
                "pickup_wait_s":r["pickup_s"]-r["arrival_s"] if r["pickup_s"] is not None else "",
                "state":"completed" if r["completed_s"] is not None else "onboard" if r["pickup_s"] is not None else "waiting"})
    row={"scenario":scenario["id"],"seed":seed,"policy":policy,"input_sha256":tape["sha256"],
         "requests":len(trips),"picked_up":len(picked),"completed":len(complete),"waiting_at_end":waiting,"onboard_at_end":onboard,
         "uncompleted":len(trips)-len(complete),"completion_fraction":len(complete)/len(trips),
         "mean_pickup_wait_s":float(np.mean(waits)) if waits else None,"p90_pickup_wait_s":float(np.quantile(waits,.9)) if waits else None,
         "capped_pickup_wait_s":float(np.mean(capped)),"empty_travel_vehicle_min":counts["empty_motion_s"]/60,
         "occupied_travel_vehicle_min":counts["occupied_motion_s"]/60,"closed_entry_attempts":counts["closed_entry_attempts"],
         "forced_stop_vehicle_min":counts["forced_stop_vehicle_s"]/60,"fallback_vehicle_min":counts["fallback_vehicle_s"]/60,
         "mean_received_road_age_s":counts["road_age_sum"]/max(1,counts["road_messages"]),
         "coalesced_messages":counts["coalesced_messages"],"invariants_passed":True}
    return row,{"frames":traces,"events":events,"requests":trips}


METRICS=["requests","completed","completion_fraction","waiting_at_end","onboard_at_end","uncompleted",
         "mean_pickup_wait_s","p90_pickup_wait_s","capped_pickup_wait_s","empty_travel_vehicle_min",
         "occupied_travel_vehicle_min","closed_entry_attempts","forced_stop_vehicle_min","fallback_vehicle_min","mean_received_road_age_s"]


def summarize(config, rows):
    out=[]
    for scenario in config["scenarios"]:
        for policy in config["policies"]:
            group=[r for r in rows if r["scenario"]==scenario["id"] and r["policy"]==policy]
            for metric in METRICS:
                values=np.array([r[metric] for r in group if r[metric] is not None],float)
                out.append({"scenario":scenario["id"],"policy":policy,"metric":metric,"n_episodes":len(values),
                            "mean":float(values.mean()),"median":float(np.median(values)),"sd":float(values.std(ddof=1)),
                            "p10":float(np.quantile(values,.1)),"p90":float(np.quantile(values,.9))})
    return out


def choose_replay(config, rows):
    chosen=[r for r in rows if r["scenario"]==config["replay_scenario"]]
    seeds=sorted({r["seed"] for r in chosen}); metrics=["completion_fraction","closed_entry_attempts","capped_pickup_wait_s"]
    matrix=np.array([[next(r for r in chosen if r["seed"]==s and r["policy"]==p)[m] for p in config["policies"] for m in metrics] for s in seeds])
    med=np.median(matrix,axis=0); scale=np.quantile(matrix,.75,axis=0)-np.quantile(matrix,.25,axis=0)
    scale=np.where(scale>1e-9,scale,1.)
    distance=np.sum(((matrix-med)/scale)**2,axis=1)
    index=min(range(len(seeds)),key=lambda i:(distance[i],seeds[i]))
    return seeds[index],{"scenario":config["replay_scenario"],"seed":seeds[index],"rule":config["replay_rule"],
                         "distance_squared":float(distance[index]),"median_vector":med.tolist(),"iqr_scale":scale.tolist(),
                         "candidate_count":len(seeds),"metrics":metrics,"policy_order":config["policies"]}


def verify_network(config):
    graph=nx.MultiGraph()
    for e in config["edges"]:graph.add_edge(e["u"],e["v"],key=e["id"])
    degrees=dict(graph.degree());assert degrees=={"A":5,"B":3,"C":3,"D":3}
    assert nx.is_connected(graph) and not nx.has_eulerian_path(graph)
    known={e["id"]:(False,0) for e in config["edges"]}
    for key in ("a","b"):known[key]=(True,0)
    assert route(config,"A","B",known,0)[1]==["e","f"]
    known["f"]=(True,0)
    assert route(config,"A","B",known,0) is None
    assert route(config,"A","C",known,46,45) is None
    return {"degrees":degrees,"connected":True,"has_eulerian_trail":False,"detour_A_to_B":["e","f"],"isolation_verified":True}


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--config",type=Path,default=HERE/"config.json");ap.add_argument("--out",type=Path,default=HERE/"output");ap.add_argument("--seeds",type=int)
    args=ap.parse_args();config=json.loads(args.config.read_text(encoding="utf-8"))
    if args.seeds is not None:config["seeds_per_scenario"]=args.seeds
    out=args.out;out.mkdir(parents=True,exist_ok=True);(out/"input_tapes").mkdir(exist_ok=True)
    (out/"config_used.json").write_text(json.dumps(config,indent=2),encoding="utf-8")
    network=verify_network(config)
    rows=[];tapes={}
    request_fields=["scenario","seed","policy","id","arrival_s","origin","destination","assigned_s","pickup_s","completed_s","vehicle","pickup_wait_s","state"]
    with gzip.open(out/"events.csv.gz","wt",newline="",encoding="utf-8") as ef, gzip.open(out/"requests.csv.gz","wt",newline="",encoding="utf-8") as rf:
        ew=csv.DictWriter(ef,fieldnames=EVENT_FIELDS);ew.writeheader();rw=csv.DictWriter(rf,fieldnames=request_fields);rw.writeheader()
        for j in range(config["seeds_per_scenario"]):
            seed=config["seed_start"]+j;tape=make_tape(config,seed);tapes[seed]=tape
            with gzip.open(out/"input_tapes"/f"seed_{seed}.json.gz","wt",encoding="utf-8") as f:json.dump(tape,f,separators=(",",":"))
            for scenario in config["scenarios"]:
                for policy in config["policies"]:
                    row,_=simulate(config,scenario,policy,tape,event_writer=ew,request_writer=rw);rows.append(row)
            if j%10==0:print(f"Completed {j+1}/{config['seeds_per_scenario']} matched seeds, {len(rows)} episodes",flush=True)
    summary=summarize(config,rows);write_csv(out/"episode_results.csv",rows);write_csv(out/"summary.csv",summary)
    paired=[]
    for scenario in config["scenarios"]:
        for left,right in [("priority_updates","latest_received"),("freshness_monitor","priority_updates")]:
            for metric in METRICS:
                differences=[]
                for seed in tapes:
                    a=next(r for r in rows if r["scenario"]==scenario["id"] and r["policy"]==left and r["seed"]==seed)
                    b=next(r for r in rows if r["scenario"]==scenario["id"] and r["policy"]==right and r["seed"]==seed)
                    assert a["input_sha256"]==b["input_sha256"]
                    if a[metric] is not None and b[metric] is not None:differences.append(a[metric]-b[metric])
                paired.append({"scenario":scenario["id"],"contrast":left+" minus "+right,"metric":metric,"n":len(differences),"mean_difference":float(np.mean(differences)),"p10":float(np.quantile(differences,.1)),"p90":float(np.quantile(differences,.9))})
    write_csv(out/"paired_differences.csv",paired)
    seed,selection=choose_replay(config,rows)
    scenario=next(s for s in config["scenarios"] if s["id"]==config["replay_scenario"])
    replay={"selection":selection,"policies":{}};timeline=[]
    for policy in config["policies"]:
        row,trace=simulate(config,scenario,policy,tapes[seed],capture=True)
        expected=next(r for r in rows if r["scenario"]==scenario["id"] and r["policy"]==policy and r["seed"]==seed)
        assert row==expected,"Replay differs from saved experiment"
        replay["policies"][policy]={"summary":row,**trace}
        for frame in trace["frames"]:timeline.append({"scenario":scenario["id"],"seed":seed,"policy":policy,**{k:v for k,v in frame.items() if k not in ["vehicles","dispatcher_closed","dispatcher_road_age_s","actual_closed","waiting_by_node"]}})
    (out/"replay.json").write_text(json.dumps(replay,separators=(",",":")),encoding="utf-8")
    (out/"replay_selection.json").write_text(json.dumps(selection,indent=2),encoding="utf-8");write_csv(out/"replay_timeline.csv",timeline)
    audit={"episodes":len(rows),"seeds_per_scenario":config["seeds_per_scenario"],"network":network,"all_online_invariants_passed":True,
           "paired_input_tapes_verified":True,"replay_matches_original_exactly":True,"selection":selection,
           "statistical_intervals":"10th-90th percentiles across synthetic episodes, not real-world confidence intervals",
           "limitations":["Historical schematic, not a verified modern road network","No braking, congestion, traffic lights, or real autonomous-driving stack","Freshness cannot predict an unannounced new closure","No abandoned requests; endpoint censoring is reported","Single configuration; illustrative, not a deployed safety guarantee"]}
    (out/"simulation_validation.json").write_text(json.dumps(audit,indent=2),encoding="utf-8")
    print(json.dumps(audit,indent=2))


if __name__=="__main__":main()
