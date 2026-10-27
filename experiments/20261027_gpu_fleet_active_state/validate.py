"""Recompute endpoints from CUDA trajectories; check analytic flight controls."""
import argparse
import hashlib
import json
from pathlib import Path
import unittest

import numpy as np
from model import GPU, PAD, MIX, build, control
from simulate import CONFIG, tape, digest

HERE = Path(__file__).resolve().parent


class ControllerTests(unittest.TestCase):
    def test_masses_and_mixer(self):
        model = build(CONFIG)
        np.testing.assert_allclose(model.body_mass[1:], 1.2)
        self.assertEqual(model.nv, 24)
        self.assertEqual(model.nu, 16)
        self.assertTrue(np.all(model.body_inertia[1:] > 0))
        np.testing.assert_allclose(MIX @ np.full(4, 1.2*9.81/4), [1.2*9.81, 0, 0, 0], atol=1e-12)

    def test_hover_and_limits(self):
        model = build(CONFIG)
        q = model.qpos0[None, :]
        v = np.zeros((1, 24))
        target = q.reshape(1, 4, 7)[..., :3]
        u = control(q, v, target, np.zeros((1, 4), dtype=bool), CONFIG)
        np.testing.assert_allclose(u, 1.2*9.81/4)
        violent = control(q, v, target+100, np.zeros((1, 4), dtype=bool), CONFIG)
        self.assertTrue(np.all((violent >= 0) & (violent <= 6)))
        off = control(q, v, target, np.ones((1, 4), dtype=bool), CONFIG)
        self.assertTrue(np.all(off == 0))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="evidence")
    args = parser.parse_args()
    out = HERE/args.out
    report = json.loads((out/"summary.json").read_text())
    config = report["config"]
    arrays = np.load(out/"trajectories.npz")
    q, v, u = [arrays[k] for k in ("qpos", "qvel", "motors")]
    assert hashlib.sha256((out/"trajectories.npz").read_bytes()).hexdigest() == report["trajectory_sha256"]
    assert q.shape == (round(config["duration_s"]/config["control_dt_s"])+1, report["nworld"], 28)
    assert np.isfinite(q).all() and np.isfinite(v).all() and np.isfinite(u).all()
    assert u.min() >= 0 and u.max() <= config["rotor_max_n"]
    state = q.reshape(q.shape[0], q.shape[1], 4, 7)
    np.testing.assert_allclose(np.linalg.norm(state[..., 3:7], axis=-1), 1, atol=3e-5)
    for i, row in enumerate(report["episodes"]):
        assert row["input_sha256"] == digest(tape(row["seed"], config))
        pos = state[:, i, :, :3]
        d = np.linalg.norm(pos[:, :, None, :]-pos[:, None, :, :], axis=-1)+np.eye(4)*1e6
        assert abs(d.min()-row["min_separation_m"]) < 1e-5
        assert bool(np.any(d < config["separation_m"])) == row["separation_episode"]
        occupancy = [(np.linalg.norm(pos[..., :2]-p[:2], axis=-1) < .85) & (pos[..., 2] < .70) for p in PAD]
        conflicts = np.any(np.stack([x.sum(axis=-1)>1 for x in occupancy]), axis=0)
        assert bool(conflicts.any()) == row["conflict_episode"]
        assert abs(conflicts[:-1].sum()*config["control_dt_s"]-row["conflict_sample_s"]) < 1e-8
        assert sum(e["event"] == "FINISH" for e in row["events"]) == row["completed"]
        grants = [e for e in row["events"] if e["event"] == "GRANT"]
        starts = [e for e in row["events"] if e["event"] == "START"]
        assert len({e["drone"] for e in grants}) == len(grants)
        assert len({e["drone"] for e in starts}) == len(starts)
        if row["policy"] == "queue_aware":
            unfinished = [set(), set()]
            for e in row["events"]:
                if e["event"] == "GRANT":
                    assert not unfinished[e["pad"]]
                    unfinished[e["pad"]].add(e["drone"])
                elif e["event"] == "RELEASE":
                    assert e["reason"] == "VERIFIED"
                    unfinished[e["drone"]%2].remove(e["drone"])
        if row["profile"] == "timely" and row["policy"] == "timeout_only":
            np.testing.assert_allclose(q[:, i], q[:, i+1], atol=2e-5, rtol=0)
    for name, expected in report["sources"].items():
        assert hashlib.sha256((out/"source"/name).read_bytes()).hexdigest() == expected
    aggregates = []
    for profile in config["profiles"]:
        for policy in config["policies"]:
            rows = [r for r in report["episodes"] if r["policy"] == policy and r["profile"] == profile]
            aggregates.append(dict(profile=profile, policy=policy, n=len(rows),
                conflict_episodes=sum(r["conflict_episode"] for r in rows),
                separation_episodes=sum(r["separation_episode"] for r in rows),
                mean_completed=float(np.mean([r["completed"] for r in rows])),
                mean_conflict_s=float(np.mean([r["conflict_sample_s"] for r in rows])),
                min_separation_m=min(r["min_separation_m"] for r in rows),
                max_tilt_deg=max(r["max_tilt_deg"] for r in rows)))
    (out/"aggregates.json").write_text(json.dumps(aggregates, indent=2)+"\n")
    physics = GPU(config, 2)
    p0, v0 = physics.snapshot()
    motors = np.zeros((2, 16))
    motors[0] = config["robot_mass_kg"]*9.81/4
    for _ in range(10):
        p, _ = physics.step(motors, np.zeros((2, 4, 3)))
    t = 10*config["control_dt_s"]
    height = p.reshape(2, 4, 7)[..., 2]
    hover_error = float(np.max(np.abs(height[0]-3)))
    fall_error = float(np.max(np.abs(height[1]-(3-.5*9.81*t*t))))
    assert hover_error < .002 and fall_error < .003
    result = dict(complete=True, world_states_checked=q.shape[0]*q.shape[1],
        finite=True, normalized_quaternions=True, bounded_rotor_forces=True,
        paired_inputs=True, timely_arms_equal=True, summary_recomputed=True,
        queue_ownership_invariant=True, cuda_analytic_hover_error_m=hover_error,
        cuda_analytic_freefall_error_m=fall_error, cuda_device=physics.device,
        note="20 ms endpoint samples; not continuous-time collision certification. GPU calibration is separate from comparison episodes.")
    (out/"validation.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2))
    print(json.dumps(aggregates, indent=2))


if __name__ == "__main__":
    main()
