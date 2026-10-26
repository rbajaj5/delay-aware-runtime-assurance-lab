"""Deterministic teaching model: knowledge, active obligations, and transport.

CPU discrete-time kinematics. No aerodynamic or learned-controller claim.
Only receipt-timeout reservation release differs between the policies.
"""
from pathlib import Path
import hashlib
import json
import random
import statistics

HERE = Path(__file__).resolve().parent
OUT = HERE / "evidence"
CONFIG = json.loads((HERE / "config.json").read_text())


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def inputs(seed, config=CONFIG):
    rng = random.Random(seed)
    return {
        "seed": seed,
        "travel_s": {d: round(rng.uniform(2.6, 2.9), 3) for d in "ABCD"},
        "dwell_s": {d: round(rng.uniform(2.0, 2.3), 3) for d in "ABCD"},
        "departure_s": 0.8,
        "command_delay_s": {d: round(rng.uniform(.1, .2), 3) for d in "ABCD"},
        "ack_delay_s": {d: round(rng.uniform(.1, .2), 3) for d in "ABCD"},
        "telemetry_delay_s": [round(rng.uniform(.1, .2), 3)
                              for _ in range(round(config["duration_s"] / config["dt_s"]) + 1)],
    }


def run(policy, profile, tape, config=CONFIG):
    assert policy in config["policies"] and profile in config["profiles"]
    knowledge = config["knowledge"]
    pads = knowledge["pads"]
    assignment = knowledge["assignment"]
    drones = {d: {"start_s": None, "command_id": f"land-{d}-001"} for d in "ABCD"}
    owners = {p: None for p in pads}
    issued = {}
    waiting = list("ABCD")
    transport, events, frames = [], [], []
    latest = None
    rejected_old, unsafe_releases = 0, 0
    received_ids = set()
    packet_seq = 0

    def send(kind, at, **payload):
        nonlocal packet_seq
        packet_seq += 1
        transport.append(dict(kind=kind, delivery_s=at, seq=packet_seq, **payload))

    def state(t):
        result = {}
        for d, entry in drones.items():
            p = pads[assignment[d]]
            initial = knowledge["initial_xy"][d]
            start = entry["start_s"]
            elapsed = -1 if start is None else t - start
            travel, dwell = tape["travel_s"][d], tape["dwell_s"][d]
            if elapsed < 0:
                mode, xy, altitude = "WAITING", initial, 3.0
            elif elapsed < travel:
                f = elapsed / travel
                mode, xy, altitude = "APPROACH", [initial[j]*(1-f)+p[j]*f for j in range(2)], 3*(1-f)
            elif elapsed < travel + dwell:
                mode, xy, altitude = "ON PAD", p, 0.0
            elif elapsed < travel + dwell + tape["departure_s"]:
                f = (elapsed-travel-dwell)/tape["departure_s"]
                mode, xy, altitude = "DEPARTING", [p[0], -4*f], 3*f
            else:
                mode, xy, altitude = "FINISHED", [p[0], -4.0], 3.0
            result[d] = dict(xy_m=list(xy), altitude_m=altitude, mode=mode,
                             command_id=entry["command_id"], pad=assignment[d])
        return result

    for k in range(round(config["duration_s"] / config["dt_s"]) + 1):
        t = round(k*config["dt_s"], 8)
        due = sorted([m for m in transport if m["delivery_s"] <= t+1e-9], key=lambda m:(m["delivery_s"],m["seq"]))
        transport = [m for m in transport if m["delivery_s"] > t+1e-9]
        for message in due:
            kind = message["kind"]
            events.append(dict(t=t, event="DELIVER", kind=kind, seq=message["seq"]))
            if kind == "COMMAND":
                d, cid = message["drone"], message["command_id"]
                if cid not in received_ids:
                    received_ids.add(cid)
                    drones[d]["start_s"] = t
                    events.append(dict(t=t,event="START",drone=d,command_id=cid))
                    if profile == "timely":
                        send("ACK",t+tape["ack_delay_s"][d],drone=d,command_id=cid)
            elif kind == "ACK":
                owner = owners[assignment[message["drone"]]]
                if owner and owner["command_id"] == message["command_id"]:
                    owner["ack"] = True
            elif kind == "OBSERVATION":
                if latest is None or message["observed_s"] > latest["observed_s"]:
                    latest = message
                else:
                    rejected_old += 1
                    events.append(dict(t=t,event="REJECT_OLD_OBSERVATION",observed_s=message["observed_s"]))
        age = None if latest is None else round(t-latest["observed_s"],8)
        fresh = age is not None and age <= config["observation_freshness_s"]+1e-9
        world = state(t)
        for pad, owner in list(owners.items()):
            if owner is None:
                continue
            cid = owner["command_id"]
            verified = (fresh and latest["observed_s"] >= owner["issued_s"]
                        and cid in latest["finished_ids"] and latest["clear"][pad])
            timed_out = (policy == "timeout_only" and not owner["ack"]
                         and t-owner["issued_s"] >= config["receipt_timeout_s"]-1e-9)
            if verified or timed_out:
                reason = "VERIFIED_COMPLETION" if verified else "NO_ACK_TIMEOUT"
                if not verified and world[owner["drone"]]["mode"] != "FINISHED":
                    unsafe_releases += 1
                events.append(dict(t=t,event="RELEASE",pad=pad,command_id=cid,reason=reason,
                                   observed_s=latest["observed_s"] if latest else None))
                owners[pad] = None
        for d in list(waiting):
            pad = assignment[d]
            # The physical owner is NOT consulted here. Both policies see the same
            # observation interface; only the timeout rule above differs.
            if owners[pad] is None and fresh and latest["clear"][pad]:
                cid = drones[d]["command_id"]
                owners[pad] = dict(drone=d,command_id=cid,issued_s=t,ack=False)
                issued[d] = t
                waiting.remove(d)
                send("COMMAND",t+tape["command_delay_s"][d],drone=d,command_id=cid)
                events.append(dict(t=t,event="GRANT",pad=pad,drone=d,command_id=cid))
        occupancy = {p: [d for d,v in world.items() if v["pad"] == p and v["mode"] == "ON PAD"] for p in pads}
        conflicts = [p for p,v in occupancy.items() if len(v)>1]
        if profile != "telemetry_outage" or t < config["outage_start_s"]:
            observation = dict(observed_s=t,
                               finished_ids=[v["command_id"] for v in world.values() if v["mode"] == "FINISHED"],
                               clear={p:not occupancy[p] for p in pads})
            send("OBSERVATION",t+tape["telemetry_delay_s"][k],**observation)
            if k == 0:
                # A duplicated early empty-pad report arrives after newer reports.
                send("OBSERVATION",4.5,**observation)
        finished_count = sum(v["mode"] == "FINISHED" for v in world.values())
        unresolved = {p:[d for d in issued if assignment[d]==p and world[d]["mode"]!="FINISHED"] for p in pads}
        frames.append(dict(t=t,drones=world,owners=json.loads(json.dumps(owners)),
                           waiting=list(waiting),observation_age_s=age,observation_fresh=fresh,
                           last_observed_s=latest["observed_s"] if latest else None,
                           observation_clear=latest["clear"] if latest else None,
                           packets=[{k:v for k,v in m.items() if k in ("kind","seq","delivery_s","command_id","observed_s")} for m in transport],
                           occupancy=occupancy,conflict_pads=conflicts,finished=finished_count,
                           rejected_old=rejected_old,unresolved_commands=unresolved,
                           state="HOLD: OBSERVATION STALE" if not fresh else
                                 ("HOLD: RESERVATIONS ACTIVE" if waiting else "ALL COMMANDS ISSUED")))
    conflict_steps = sum(bool(f["conflict_pads"]) for f in frames)
    summary = dict(policy=policy,profile=profile,seed=tape["seed"],input_sha256=digest(tape),
                   knowledge_sha256=digest(knowledge),config_sha256=digest(config),
                   completed=frames[-1]["finished"],unfinished=4-frames[-1]["finished"],
                   conflict_steps=conflict_steps,conflict_duration_s=round(conflict_steps*config["dt_s"],8),
                   conflict_episode=bool(conflict_steps),
                   conflict_pad_count=len(set(p for f in frames for p in f["conflict_pads"])),
                   unsafe_releases=unsafe_releases,rejected_old=rejected_old,issued_s=issued,
                   mean_issue_time_s=statistics.mean(issued.values()) if issued else None,
                   last_completion_s=next((f["t"] for f in frames if f["finished"]==4),None),
                   backend="CPU deterministic discrete-time kinematics")
    return dict(config=config,inputs=tape,summary=summary,events=events,frames=frames)


def main():
    OUT.mkdir(exist_ok=True)
    summaries = []
    for seed in range(CONFIG["seed_start"], CONFIG["seed_start"]+CONFIG["seed_count"]):
        tape = inputs(seed)
        for profile in CONFIG["profiles"]:
            for policy in CONFIG["policies"]:
                result = run(policy,profile,tape)
                summaries.append(result["summary"])
                if seed == CONFIG["seed_start"]:
                    (OUT/f"{profile}_{policy}.json").write_text(json.dumps(result,indent=2)+"\n")
    aggregates = []
    for profile in CONFIG["profiles"]:
        for policy in CONFIG["policies"]:
            rows = [r for r in summaries if r["profile"]==profile and r["policy"]==policy]
            aggregates.append(dict(profile=profile,policy=policy,n=len(rows),
                conflict_episodes=sum(r["conflict_episode"] for r in rows),
                mean_completed=statistics.mean(r["completed"] for r in rows),
                mean_conflict_duration_s=statistics.mean(r["conflict_duration_s"] for r in rows),
                mean_issue_time_s=statistics.mean(r["mean_issue_time_s"] for r in rows)))
    (OUT/"episode_summaries.json").write_text(json.dumps(summaries,indent=2)+"\n")
    (OUT/"aggregates.json").write_text(json.dumps(aggregates,indent=2)+"\n")
    print(json.dumps(aggregates,indent=2))


if __name__ == "__main__":
    main()
