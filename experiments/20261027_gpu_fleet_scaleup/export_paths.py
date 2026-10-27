"""Readable measured trajectories for the six featured worlds, not references."""
import csv
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np


def main():
    folder = Path(__file__).resolve().parent/"evidence_v2"
    summary = json.loads((folder/"summary.json").read_text())
    n = summary["n_drones"]
    with np.load(folder/"trajectories.npz") as a:
        q, v = a["qpos"], a["qvel"]
    path = folder/"featured_flight_paths.csv.gz"
    count, errors = 0, []
    with gzip.open(path, "wt", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["profile", "policy", "seed", "time_s", "drone_id", "x_m", "y_m", "z_m",
                         "qw", "qx", "qy", "qz", "vx_m_s", "vy_m_s", "vz_m_s", "mode"])
        for world, episode in enumerate(summary["episodes"]):
            if episode["seed"] != summary["config"]["seed_start"]:
                continue
            record = json.loads(gzip.decompress((folder/f"{episode['profile']}_{episode['policy']}.json.gz").read_bytes()))
            for step, row in enumerate(record["frames"]):
                poses = q[step, world].reshape(n, 7)
                velocities = v[step, world].reshape(n, 6)
                errors.extend(np.linalg.norm(poses[:, :3]-row["target"], axis=1).tolist())
                for i in range(n):
                    writer.writerow([episode["profile"], episode["policy"], episode["seed"], row["t"], i+1,
                                     *poses[i].tolist(), *velocities[i, :3].tolist(), row["mode"][i]])
                    count += 1
    checked = 0
    lookup = {(e["profile"], e["policy"], e["seed"]): w for w, e in enumerate(summary["episodes"])}
    with gzip.open(path, "rt", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            w = lookup[row["profile"], row["policy"], int(row["seed"])]
            step = round(float(row["time_s"])/summary["config"]["control_dt_s"])
            i = int(row["drone_id"])-1
            np.testing.assert_allclose([float(row[k]) for k in ("x_m", "y_m", "z_m")], q[step, w].reshape(n, 7)[i, :3], atol=0, rtol=0)
            checked += 1
    assert count == checked == 6*len(q)*n
    (folder/"path_export_validation.json").write_text(json.dumps(dict(
        measured_rows=count, checked_rows=checked, seed=summary["config"]["seed_start"],
        worlds=6, drones_per_world=n, retained_times_per_world=len(q),
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        mean_position_reference_residual_m=float(np.mean(errors)),
        note="All six seed-45300 worlds, not every seed. Positions and velocities come from GPU qpos/qvel. Residual is descriptive, across mixed modes and policies."), indent=2)+"\n")
    print("Verified measured trajectory rows:", count)


if __name__ == "__main__":
    main()
