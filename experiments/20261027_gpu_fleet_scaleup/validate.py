"""Recompute endpoints from CUDA trajectories; check analytic flight controls."""
import argparse
import hashlib
import json
from pathlib import Path
import unittest

import numpy as np
from model import GPU, PAD, START, N, P, MIX, build, control
from simulate import CONFIG, tape, digest

HERE = Path(__file__).resolve().parent


class ControllerTests(unittest.TestCase):
    def test_masses_and_mixer(self):
        model = build(CONFIG)
        np.testing.assert_allclose(model.body_mass[1:], 1.2)
        self.assertEqual(model.nv, N*6)
        self.assertEqual(model.nu, N*4)
        self.assertTrue(np.all(model.body_inertia[1:] > 0))
        np.testing.assert_allclose(MIX @ np.full(4, 1.2*9.81/4), [1.2*9.81, 0, 0, 0], atol=1e-12)

    def test_hover_and_limits(self):
        model = build(CONFIG)
        q = model.qpos0[None, :]
        v = np.zeros((1, N*6))
        target = q.reshape(1, N, 7)[..., :3]
        u = control(q, v, target, np.zeros((1, N), dtype=bool), CONFIG)
        np.testing.assert_allclose(u, 1.2*9.81/4)
        violent = control(q, v, target+100, np.zeros((1, N), dtype=bool), CONFIG)
        self.assertTrue(np.all((violent >= 0) & (violent <= 6)))
        off = control(q, v, target, np.ones((1, N), dtype=bool), CONFIG)
        self.assertTrue(np.all(off == 0))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="evidence_v2")
    args = parser.parse_args()
    out = HERE/args.out
    report = json.loads((out/"summary.json").read_text())
    config = report["config"]
    arrays = np.load(out/"trajectories.npz")
    q, v, u = [arrays[k] for k in ("qpos", "qvel", "motors")]
    assert hashlib.sha256((out/"trajectories.npz").read_bytes()).hexdigest() == report["trajectory_sha256"]
    assert q.shape == (round(config["duration_s"]/config["control_dt_s"])+1, report["nworld"], N*7)
    assert not any(report["overflow_flags"])
    assert np.isfinite(q).all() and np.isfinite(v).all() and np.isfinite(u).all()
    assert u.min() >= 0 and u.max() <= config["rotor_max_n"]
    state = q.reshape(q.shape[0], q.shape[1], N, 7)
    np.testing.assert_allclose(np.linalg.norm(state[..., 3:7], axis=-1), 1, atol=3e-5)
    nominal_pairs = []
    for i, row in enumerate(report["episodes"]):
        assert row["input_sha256"] == digest(tape(row["seed"], config))
        pos = state[:, i, :, :3]
        d = np.linalg.norm(pos[:, :, None, :]-pos[:, None, :, :], axis=-1)+np.eye(N)*1e6
        assert abs(d.min()-row["min_separation_m"]) < 1e-5
        assert bool(np.any(d < config["separation_m"])) == row["separation_episode"]
        occupancy = [(np.linalg.norm(pos[..., :2]-p[:2], axis=-1) < .85) & (pos[..., 2] < .70) for p in PAD]
        conflicts = np.any(np.stack([x.sum(axis=-1)>1 for x in occupancy]), axis=0)
        assert bool(conflicts.any()) == row["conflict_episode"]
        assert abs(conflicts[:-1].sum()*config["control_dt_s"]-row["conflict_sample_s"]) < 1e-8
        finished = {e["drone"] for e in row["events"] if e["event"] == "FINISH"}
        disarmed = {e["drone"] for e in row["events"] if e["event"] == "DISARM"}
        assert len(finished-disarmed) == row["completed"]
        assert len(disarmed) == row["disarmed"]
        grants = [e for e in row["events"] if e["event"] == "GRANT"]
        starts = [e for e in row["events"] if e["event"] == "START"]
        assert len({e["drone"] for e in grants}) == len(grants)
        assert len({e["drone"] for e in starts}) == len(starts)
        if row["policy"] == "queue_aware":
            unfinished = [set() for _ in range(P)]
            for e in row["events"]:
                if e["event"] == "GRANT":
                    assert not unfinished[e["pad"]]
                    unfinished[e["pad"]].add(e["drone"])
                elif e["event"] == "RELEASE":
                    assert e["reason"] == "VERIFIED"
                    unfinished[e["drone"]%P].remove(e["drone"])
        if row["profile"] == "timely" and row["policy"] == "timeout_only":
            other = report["episodes"][i+1]
            assert other["profile"] == "timely" and other["policy"] == "queue_aware"
            assert row["seed"] == other["seed"] and row["events"] == other["events"]
            error = float(np.linalg.norm(state[:, i, :, :3]-state[:, i+1, :, :3], axis=-1).max())
            qa, qb = state[:, i, :, 3:].astype(float), state[:, i+1, :, 3:].astype(float)
            qa /= np.linalg.norm(qa, axis=-1, keepdims=True)
            qb /= np.linalg.norm(qb, axis=-1, keepdims=True)
            angle = float(np.rad2deg(2*np.arccos(np.clip(np.abs((qa*qb).sum(axis=-1)), 0, 1))).max())
            # Separate dimensional tolerances for float32 contact-solver agreement.
            assert error < 1e-4 and angle < .01
            nominal_pairs.append(dict(seed=row["seed"], max_position_error_m=error,
                max_attitude_error_deg=angle, identical_protocol_events=True))
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
                mean_disarmed=float(np.mean([r["disarmed"] for r in rows])),
                mean_conflict_s=float(np.mean([r["conflict_sample_s"] for r in rows])),
                min_separation_m=min(r["min_separation_m"] for r in rows),
                max_tilt_deg=max(r["max_tilt_deg"] for r in rows)))
    (out/"aggregates.json").write_text(json.dumps(aggregates, indent=2)+"\n")
    physics = GPU(config, 2)
    p0, v0 = physics.snapshot()
    motors = np.zeros((2, N*4))
    motors[0] = config["robot_mass_kg"]*9.81/4
    for _ in range(10):
        p, _ = physics.step(motors, np.zeros((2, N, 3)))
    t = 10*config["control_dt_s"]
    height = p.reshape(2, N, 7)[..., 2]
    hover_error = float(np.max(np.abs(height[0]-START[:, 2])))
    fall_error = float(np.max(np.abs(height[1]-(START[:, 2]-.5*9.81*t*t))))
    assert hover_error < .002 and fall_error < .003
    result = dict(complete=True, world_states_checked=q.shape[0]*q.shape[1],
        finite=True, normalized_quaternions=True, bounded_rotor_forces=True, no_gpu_overflow=True,
        paired_inputs=True, timely_arms_numerically_agree=True, summary_recomputed=True,
        nominal_pair_checks=nominal_pairs, nominal_position_tolerance_m=1e-4,
        nominal_attitude_tolerance_deg=.01,
        queue_ownership_invariant=True, cuda_analytic_hover_error_m=hover_error,
        cuda_analytic_freefall_error_m=fall_error, cuda_device=physics.device,
        note="20 ms endpoint samples; not continuous-time collision certification. GPU calibration is separate from comparison episodes.")
    (out/"validation.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2))
    print(json.dumps(aggregates, indent=2))


if __name__ == "__main__":
    main()
