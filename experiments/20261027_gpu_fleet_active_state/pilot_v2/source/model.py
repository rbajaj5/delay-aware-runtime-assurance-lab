"""Four free rigid bodies driven only by four bounded rotor forces each."""
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
START = np.array([[-4., 4., 3.], [4., 4., 3.], [-2., 6., 3.], [2., 6., 3.]])
PAD = np.array([[-2.5, 0., .215], [2.5, 0., .215]])
EXIT = np.array([[-4., -5., 3.], [4., -5., 3.], [-2., -5., 3.], [2., -5., 3.]])
COLORS = ["0.02 .53 .60 1", ".54 .31 .70 1", ".86 .46 .12 1", ".28 .53 .20 1"]
ROTOR_XY = np.array([[.23, .23], [-.23, .23], [-.23, -.23], [.23, -.23]])
SPIN = np.array([1, -1, 1, -1])
MIX = np.array([np.ones(4), ROTOR_XY[:, 1], -ROTOR_XY[:, 0], .012 * SPIN])


def model_xml(config):
    root = ET.Element("mujoco", model="queue_aware_fleet")
    ET.SubElement(root, "option", timestep=str(config["physics_dt_s"]), gravity="0 0 -9.81",
                  integrator="Euler", iterations="30", solver="Newton")
    visual = ET.SubElement(root, "visual")
    ET.SubElement(visual, "global", offwidth="900", offheight="540")
    world = ET.SubElement(root, "worldbody")
    ET.SubElement(world, "light", pos="0 0 12", dir="0 0 -1", diffuse=".8 .8 .8")
    ET.SubElement(world, "geom", name="ground", type="plane", size="12 12 .1", rgba=".88 .92 .91 1")
    for p, xyz in enumerate(PAD):
        ET.SubElement(world, "geom", name=f"pad{p}", type="cylinder", pos=f"{xyz[0]} {xyz[1]} .025",
                      size=".85 .025", rgba=".27 .36 .38 1")
    actuators = ET.SubElement(root, "actuator")
    for i, xyz in enumerate(START):
        body = ET.SubElement(world, "body", name=f"drone{i}", pos=" ".join(map(str, xyz)))
        ET.SubElement(body, "freejoint", name=f"root{i}")
        ET.SubElement(body, "inertial", pos="0 0 0", mass=str(config["robot_mass_kg"]),
                      diaginertia=".025 .025 .045")
        ET.SubElement(body, "geom", name=f"collision{i}", type="box", size=".16 .14 .06", rgba="0 0 0 0")
        ET.SubElement(body, "geom", name=f"hull{i}", type="ellipsoid", size=".18 .13 .075",
                      rgba=COLORS[i], contype="0", conaffinity="0")
        for j, (x, y) in enumerate(ROTOR_XY):
            ET.SubElement(body, "geom", name=f"foot{i}_{j}", type="sphere", pos=f"{x*.78} {y*.78} -.14",
                          size=".025", rgba=".16 .20 .22 1")
            ET.SubElement(body, "geom", type="capsule", fromto=f"0 0 0 {x} {y} 0", size=".024",
                          rgba=".16 .20 .22 1", contype="0", conaffinity="0")
            ET.SubElement(body, "geom", type="cylinder", pos=f"{x} {y} .015", size=".13 .009",
                          rgba=COLORS[i], contype="0", conaffinity="0")
            ET.SubElement(body, "site", name=f"motor{i}_{j}", pos=f"{x} {y} 0", size=".01", rgba="0 0 0 0")
            ET.SubElement(actuators, "general", site=f"motor{i}_{j}", gear=f"0 0 1 0 0 {SPIN[j]*.012}",
                          ctrllimited="true", ctrlrange=f"0 {config['rotor_max_n']}")
    return ET.tostring(root, encoding="unicode")


def build(config):
    return mujoco.MjModel.from_xml_string(model_xml(config))


def rotations(quat):
    w, x, y, z = np.moveaxis(quat, -1, 0)
    return np.stack([1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w),
                     2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w),
                     2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)], axis=-1).reshape(*quat.shape[:-1], 3, 3)


def control(qpos, qvel, target, landed, config):
    p = qpos.reshape(-1, 4, 7)
    v = qvel.reshape(-1, 4, 6)
    r = rotations(p[..., 3:7])
    accel = 3.0*(target-p[..., :3])-3.5*v[..., :3]
    accel[..., :2] = np.clip(accel[..., :2], -3., 3.)
    accel[..., 2] = np.clip(accel[..., 2], -4., 5.) + 9.81
    force = config["robot_mass_kg"] * accel
    z = force/np.linalg.norm(force, axis=-1, keepdims=True)
    y = np.cross(z, np.broadcast_to([1., 0., 0.], z.shape))
    y /= np.linalg.norm(y, axis=-1, keepdims=True)
    desired = np.stack([np.cross(y, z), y, z], axis=-1)
    error = np.swapaxes(desired, -1, -2) @ r - np.swapaxes(r, -1, -2) @ desired
    e = .5*np.stack([error[..., 2, 1], error[..., 0, 2], error[..., 1, 0]], axis=-1)
    omega = v[..., 3:6]
    torque = -.8*e-.18*omega+np.cross(omega, omega*np.array([.025, .025, .045]))
    thrust = np.sum(force*r[..., :, 2], axis=-1)
    wrench = np.concatenate([thrust[..., None], torque], axis=-1)
    motors = np.clip(wrench @ np.linalg.inv(MIX).T, 0, config["rotor_max_n"])
    motors[landed] = 0.
    return motors.reshape(-1, 16)


class GPU:
    def __init__(self, config, nworld):
        import warp as wp
        import mujoco_warp as mjw
        wp.config.kernel_cache_dir = str(HERE/".warp-cache")
        wp.init()
        if not wp.get_cuda_devices():
            raise RuntimeError("CUDA required; no CPU substitution")
        self.wp, self.mjw, self.config = wp, mjw, config
        self.model = build(config)
        data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, data)
        self.body_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"drone{i}") for i in range(4)]
        with wp.ScopedDevice("cuda:0"):
            self.gm = mjw.put_model(self.model)
            self.data = mjw.put_data(self.model, data, nworld=nworld, nconmax=64, njmax=256)
            mjw.step(self.gm, self.data)
            wp.synchronize()
            self.data = mjw.put_data(self.model, data, nworld=nworld, nconmax=64, njmax=256)
            with wp.ScopedCapture() as capture:
                for _ in range(round(config["control_dt_s"]/config["physics_dt_s"])):
                    mjw.step(self.gm, self.data)
            self.graph = capture.graph
        self.device = wp.get_device("cuda:0").name

    def snapshot(self):
        return self.data.qpos.numpy().copy(), self.data.qvel.numpy().copy()

    def step(self, controls, force):
        self.data.ctrl.assign(np.asarray(controls, dtype=np.float32))
        forces = np.zeros((*self.data.xfrc_applied.shape, 6), dtype=np.float32)
        forces[:, self.body_ids, :3] = force
        self.data.xfrc_applied.assign(forces)
        self.wp.capture_launch(self.graph)
        return self.snapshot()
