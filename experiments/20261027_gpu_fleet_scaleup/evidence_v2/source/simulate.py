"""Matched coordination policies coupled to CUDA 6-DoF flight/contact dynamics."""
import argparse
import hashlib
import gzip
import importlib.metadata
import json
from pathlib import Path
import time

import numpy as np
from model import GPU, START, PAD, EXIT, N, P, control, model_xml, rotations

HERE = Path(__file__).resolve().parent
CONFIG = json.loads((HERE/"config.json").read_text())
MODES = ["WAITING", "APPROACH", "ON PAD", "DEPARTING", "FINISHED", "DISARMED"]


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def tape(seed, config):
    rng = np.random.default_rng(seed)
    n = round(config["duration_s"]/config["control_dt_s"]) + 1
    return dict(seed=seed, command_delay=rng.uniform(.10, .18, N).tolist(),
                ack_delay=rng.uniform(.10, .18, N).tolist(),
                obs_delay=rng.uniform(.10, .18, n).tolist(),
                wind=rng.normal(0, .025, (n, N, 3)).tolist())


class Episode:
    def __init__(self, policy, profile, inputs, config):
        self.policy, self.profile, self.inputs, self.c = policy, profile, inputs, config
        self.owners, self.waiting, self.issued = [None]*P, list(range(N)), {}
        self.mode, self.started = np.zeros(N, dtype=int), np.zeros(N)
        self.from_pos, self.target = START.copy(), START.copy()
        self.transport, self.events, self.frames = [], [], []
        self.latest, self.seq, self.old_rejected = None, 0, 0
        self.received = set()
        self.touch_streak = np.zeros(N, dtype=int)

    def send(self, kind, at, **data):
        self.seq += 1
        self.transport.append(dict(kind=kind, at=at, seq=self.seq, **data))

    def tick(self, k, qpos, qvel):
        t = round(k*self.c["control_dt_s"], 8)
        pos, vel = qpos.reshape(N, 7)[:, :3], qvel.reshape(N, 6)[:, :3]
        tilt = np.degrees(np.arccos(np.clip(rotations(qpos.reshape(N, 7)[:, 3:7])[:, 2, 2], -1, 1)))
        for i in np.flatnonzero((tilt > self.c["disarm_tilt_deg"]) & (self.mode != 5)):
            self.mode[i] = 5
            self.events.append(dict(t=t, event="DISARM", drone=int(i), tilt_deg=float(tilt[i])))
        due = sorted([m for m in self.transport if m["at"] <= t+1e-8], key=lambda m:(m["at"], m["seq"]))
        self.transport = [m for m in self.transport if m["at"] > t+1e-8]
        for message in due:
            if message["kind"] == "COMMAND":
                i = message["drone"]
                if i not in self.received and self.mode[i] != 5:
                    self.received.add(i)
                    self.mode[i], self.started[i], self.from_pos[i] = 1, t, pos[i]
                    self.events.append(dict(t=t, event="START", drone=i))
                    if self.profile == "timely":
                        self.send("ACK", t+self.inputs["ack_delay"][i], drone=i)
            elif message["kind"] == "ACK":
                owner = self.owners[message["drone"]%P]
                if owner and owner["drone"] == message["drone"]:
                    owner["ack"] = True
            elif self.latest is None or message["observed_s"] > self.latest["observed_s"]:
                self.latest = message
            else:
                self.old_rejected += 1
        for i in range(N):
            elapsed = t-self.started[i]
            if self.mode[i] == 1:
                f = np.clip(elapsed/self.c["approach_s"], 0, 1)
                f = f*f*f*(10-15*f+6*f*f)
                self.target[i] = (1-f)*self.from_pos[i]+f*PAD[i%P]
                on_pad = np.linalg.norm(pos[i, :2]-PAD[i%P, :2]) < .30 and .16 < pos[i, 2] < .27 and np.linalg.norm(vel[i]) < .35
                self.touch_streak[i] = self.touch_streak[i]+1 if on_pad else 0
                if elapsed >= self.c["approach_s"] and self.touch_streak[i] >= 5:
                    self.mode[i], self.started[i] = 2, t
                    self.events.append(dict(t=t, event="LANDED", drone=i))
            elif self.mode[i] == 2 and elapsed >= self.c["dwell_s"]:
                self.mode[i], self.started[i], self.from_pos[i] = 3, t, pos[i]
                self.events.append(dict(t=t, event="DEPART", drone=i))
            elif self.mode[i] == 3:
                f = np.clip(elapsed/self.c["departure_s"], 0, 1)
                f = f*f*f*(10-15*f+6*f*f)
                self.target[i] = (1-f)*self.from_pos[i]+f*EXIT[i]
                if elapsed >= self.c["departure_s"] and np.linalg.norm(pos[i]-EXIT[i]) < .35 and np.linalg.norm(vel[i]) < .4:
                    self.mode[i] = 4
                    self.events.append(dict(t=t, event="FINISH", drone=i))
        age = None if self.latest is None else t-self.latest["observed_s"]
        fresh = age is not None and age <= self.c["freshness_s"]+1e-8
        occupancy = [[i for i in range(N) if np.linalg.norm(pos[i, :2]-p[:2]) < .85 and pos[i, 2] < .70] for p in PAD]
        for p, owner in enumerate(self.owners):
            if owner is None:
                continue
            verified = fresh and owner["drone"] in self.latest["finished"] and self.latest["clear"][p]
            timed_out = self.policy == "timeout_only" and not owner["ack"] and t-owner["issued_s"] >= self.c["receipt_timeout_s"]-1e-8
            if verified or timed_out:
                self.events.append(dict(t=t, event="RELEASE", drone=owner["drone"], reason="VERIFIED" if verified else "TIMEOUT"))
                self.owners[p] = None
        for i in list(self.waiting):
            p = i%P
            if self.owners[p] is None and fresh and self.latest["clear"][p]:
                self.owners[p] = dict(drone=i, issued_s=t, ack=False, command_id=f"land-{i}-001")
                self.issued[i] = t
                self.waiting.remove(i)
                self.send("COMMAND", t+self.inputs["command_delay"][i], drone=i)
                self.events.append(dict(t=t, event="GRANT", drone=i, pad=p))
        if self.profile != "telemetry_outage" or t < self.c["outage_start_s"]:
            obs = dict(observed_s=t, finished=np.flatnonzero(self.mode==4).tolist(), clear=[not v for v in occupancy])
            self.send("OBSERVATION", t+self.inputs["obs_delay"][k], **obs)
            if k == 0:
                self.send("OBSERVATION", 4.5, **obs)
        delta = pos[:, None]-pos[None]
        distances = np.linalg.norm(delta, axis=-1)+np.eye(N)*1e6
        tilt = np.degrees(np.arccos(np.clip(rotations(qpos.reshape(N, 7)[:, 3:7])[:, 2, 2], -1, 1)))
        self.frames.append(dict(t=t, mode=self.mode.tolist(), position=pos.tolist(), target=self.target.tolist(),
            owner=[dict(o) if o else None for o in self.owners], waiting=list(self.waiting),
            observation_age_s=age, fresh=fresh, occupancy=occupancy,
            min_separation_m=float(distances.min()), conflict_pads=sum(len(v)>1 for v in occupancy),
            completed=int(np.sum(self.mode==4)), tilt_deg=tilt.tolist(),
            disarmed=int(np.sum(self.mode==5)),
            queued_messages=len(self.transport), old_rejected=self.old_rejected,
            decision="HOLD: STALE EVIDENCE" if not fresh else ("WAIT: RESERVED" if self.waiting else "COMMANDS ISSUED")))
        return self.target.copy(), (self.mode == 2) | (self.mode == 5)

    def summary(self):
        rows = self.frames
        return dict(policy=self.policy, profile=self.profile, seed=self.inputs["seed"], input_sha256=digest(self.inputs),
            completed=rows[-1]["completed"], unfinished=N-rows[-1]["completed"],
            disarmed=rows[-1]["disarmed"],
            conflict_episode=any(r["conflict_pads"] for r in rows),
            conflict_sample_s=sum(bool(r["conflict_pads"]) for r in rows[:-1])*self.c["control_dt_s"],
            separation_episode=any(r["min_separation_m"] < self.c["separation_m"] for r in rows),
            min_separation_m=min(r["min_separation_m"] for r in rows),
            max_tilt_deg=max(max(r["tilt_deg"]) for r in rows),
            old_rejected=rows[-1]["old_rejected"], events=self.events)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=CONFIG["seed_count"])
    parser.add_argument("--out", default="evidence_v2")
    args = parser.parse_args()
    out = HERE/args.out
    if (out/"summary.json").exists():
        raise RuntimeError("Preserve existing evidence; choose a fresh --out directory")
    out.mkdir(parents=True, exist_ok=True)
    (out/"source").mkdir(exist_ok=True)
    for name in ("model.py", "simulate.py", "config.json"):
        (out/"source"/name).write_bytes((HERE/name).read_bytes())
    config = {**CONFIG, "seed_count": args.seeds}
    episodes = []
    for seed in range(config["seed_start"], config["seed_start"]+args.seeds):
        inputs = tape(seed, config)
        for profile in config["profiles"]:
            for policy in config["policies"]:
                episodes.append(Episode(policy, profile, inputs, config))
    start = time.perf_counter()
    physics = GPU(config, len(episodes))
    compile_s = time.perf_counter()-start
    qpos, qvel = physics.snapshot()
    n = round(config["duration_s"]/config["control_dt_s"])
    qs, vs, us = [qpos], [qvel], []
    start = time.perf_counter()
    for k in range(n+1):
        commands = [e.tick(k, qpos[j], qvel[j]) for j, e in enumerate(episodes)]
        if k == n:
            break
        targets, landed = map(np.stack, zip(*commands))
        motors = control(qpos, qvel, targets, landed, config)
        wind = np.asarray([e.inputs["wind"][k] for e in episodes])
        qpos, qvel = physics.step(motors, wind)
        if not np.isfinite(qpos).all() or not np.isfinite(qvel).all():
            np.savez_compressed(out/"failed_state.npz", qpos=qpos, qvel=qvel, motors=motors)
            (out/"FAILED_RUN.json").write_text(json.dumps(dict(status="invalid_aborted", step=k,
                time_s=(k+1)*config["control_dt_s"],
                invalid_worlds=np.flatnonzero(~np.isfinite(qpos).all(axis=1) | ~np.isfinite(qvel).all(axis=1)).tolist()), indent=2))
            raise RuntimeError(f"Nonfinite GPU state at {k}")
        qs.append(qpos); vs.append(qvel); us.append(motors)
        if k%250 == 0:
            print(f"GPU step {k}/{n}; {len(episodes)} worlds x {N} drones; max height {qpos.reshape(-1, N, 7)[..., 2].max():.2f}", flush=True)
    physics.wp.synchronize()
    wall_s = time.perf_counter()-start
    np.savez_compressed(out/"trajectories.npz", qpos=qs, qvel=vs, motors=us)
    for i, e in enumerate(episodes):
        if e.inputs["seed"] == config["seed_start"]:
            payload = json.dumps(dict(world=i, frames=e.frames, events=e.events), separators=(",", ":")).encode()
            (out/f"{e.profile}_{e.policy}.json.gz").write_bytes(gzip.compress(payload, mtime=0))
    sources = {f:hashlib.sha256((HERE/f).read_bytes()).hexdigest() for f in ("model.py", "simulate.py", "config.json")}
    result = dict(config=config, device=physics.device, backend="MuJoCo Warp CUDA", nworld=len(episodes),
        setup_compile_s=compile_s, simulation_and_recording_s=wall_s,
        n_drones=N, n_pads=P, velocity_dofs=physics.model.nv, rotor_actuators=physics.model.nu,
        overflow_flags=physics.data.overflow.numpy().tolist(),
        gpu_contact_capacity=physics.data.naconmax, constraints_per_world=physics.data.njmax,
        sources=sources, packages={p:importlib.metadata.version(p) for p in ("mujoco", "mujoco-warp", "warp-lang", "numpy")},
        episodes=[e.summary() for e in episodes],
        trajectory_sha256=hashlib.sha256((out/"trajectories.npz").read_bytes()).hexdigest())
    (out/"summary.json").write_text(json.dumps(result, indent=2)+"\n")
    (out/"scene.xml").write_text(model_xml(config))
    for e in result["episodes"]:
        print({k:v for k,v in e.items() if k != "events"})
    print(f"CUDA {wall_s:.2f}s; setup {compile_s:.2f}s", flush=True)


if __name__ == "__main__":
    main()
