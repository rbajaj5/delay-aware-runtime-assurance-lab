"""Render measured CUDA poses and their task ledger at the exact same timestamp."""
import argparse
from functools import lru_cache
import hashlib
import gzip
import json
import math
from pathlib import Path
import subprocess
import time
import wave

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from model import N, P, PAD

HERE = Path(__file__).resolve().parent
BUILD, OUT, EVIDENCE = HERE/"build", HERE/"delivery", HERE/"evidence_v2"
FPS = 25
INK, MUTED, BG, TEAL, RED = "#193637", "#586e70", "#f1f5f5", "#087c79", "#b7483f"
COLORS = ["#078996", "#8854b3", "#da751e", "#4b8833"]


@lru_cache(None)
def font(size, bold=False, chinese=False):
    return ImageFont.truetype("C:/Windows/Fonts/"+("msyh.ttc" if chinese else ("seguisb.ttf" if bold else "segoeui.ttf")), size)


def text(draw, xy, value, size=24, color=INK, bold=False, anchor=None):
    value = str(value)
    draw.text(xy, value, font=font(size, bold, any("\u4e00"<=c<="\u9fff" for c in value)), fill=color, anchor=anchor)


def wrap(draw, value, width, size):
    lines, line = [], ""
    for word in value.split():
        trial = (line+" "+word).strip()
        if line and draw.textlength(trial, font=font(size)) > width:
            lines.append(line); line = word
        else:
            line = trial
    return lines+[line]


class Movie:
    def __init__(self):
        OUT.mkdir(exist_ok=True); BUILD.mkdir(exist_ok=True)
        self.lesson = json.loads((HERE/"lesson.json").read_text(encoding="utf-8"))
        self.timeline = json.loads((OUT/"timeline.json").read_text(encoding="utf-8"))
        assert hashlib.sha256((HERE/"lesson.json").read_bytes()).hexdigest() == self.timeline["script_sha256"]
        self.summary = json.loads((EVIDENCE/"summary.json").read_text())
        with np.load(EVIDENCE/"trajectories.npz") as archive:
            self.arrays = {name: archive[name] for name in ("qpos", "qvel")}
        self.runs = {(p, a):json.loads(gzip.decompress((EVIDENCE/f"{p}_{a}.json.gz").read_bytes())) for p in ("timely", "lost_ack", "telemetry_outage") for a in ("timeout_only", "queue_aware")}
        self.last_state = len(self.arrays["qpos"])-1
        self.dt = self.summary["config"]["control_dt_s"]
        self.positions = self.arrays["qpos"].reshape(self.last_state+1, -1, N, 7)[..., :3]
        self.model = mujoco.MjModel.from_xml_path(str(EVIDENCE/"scene.xml"))
        # Replay-only lighting and floor extent; these do not change physics evidence.
        self.model.geom_size[0, :2] = 100
        self.model.vis.headlight.ambient[:] = [.55, .55, .55]
        self.model.vis.headlight.diffuse[:] = [.35, .35, .35]
        self.model.vis.headlight.specular[:] = 0
        self.model.light_specular[:] = 0
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)
        self.renderer = mujoco.Renderer(self.model, 460, 900)
        from OpenGL.GL import glGetString, GL_RENDERER
        self.renderer_name = glGetString(GL_RENDERER).decode()
        self.camera = mujoco.MjvCamera()
        self.camera.lookat[:] = [0, -8, 1.6]
        self.camera.distance, self.camera.azimuth, self.camera.elevation = 36, 12, -55
        self.frames = math.ceil(self.timeline["duration_s"]*FPS)
        self.last_image_key = None

    def project(self, xyz):
        cameras = self.renderer.scene.camera
        cam = cameras[0]
        eye = (cameras[0].pos+cameras[1].pos)/2
        right = np.cross(cam.forward, cam.up)
        scale = 460*cam.frustum_near/(cam.frustum_top-cam.frustum_bottom)
        delta = np.asarray(xyz)-eye
        depth = delta @ cam.forward
        return np.stack([450+scale*(delta @ right)/depth,
            230-scale*(delta @ cam.up)/depth], axis=-1)

    def image(self, frame):
        t = frame/FPS
        ci = next((i for i, c in enumerate(self.timeline["cues"]) if c["start"] <= t < c["end"]), len(self.timeline["cues"])-1)
        cue = self.timeline["cues"][ci]
        chapter = self.lesson["chapters"][cue["chapter"]]
        fraction = np.clip((t-cue["start"])/max(1/FPS, cue["end"]-cue["start"]-1/FPS), 0, 1)
        sim_t = cue["from_s"]+(cue["to_s"]-cue["from_s"])*fraction
        index = min(self.last_state, round(sim_t/self.dt))
        sim_t = index*self.dt
        # Paused explanation frames differ only in the movie progress indicator.
        if (ci, index) == self.last_image_key:
            im = self.last_image.copy()
            ImageDraw.Draw(im).rectangle((0, 1075, int(1920*(frame+1)/self.frames), 1080), fill="#68cab1")
            return im, dict(frame=frame, time_s=t, cue=ci, chapter=cue["chapter"], profile=cue["profile"], states=self.last_links)
        im = Image.new("RGB", (1920, 1080), BG)
        d = ImageDraw.Draw(im)
        text(d, (30, 16), "RAVI BAJAJ / QUEUE-AWARE AUTONOMY", 22, TEAL, True)
        text(d, (1888, 16), "SYNTHETIC CUDA PHYSICS / MEASURED TRAJECTORIES", 20, MUTED, anchor="ra")
        text(d, (30, 53), chapter["english"], 43, INK, True)
        text(d, (1200, 54), chapter["term"], 30, TEAL)
        text(d, (1200, 100), chapter["pinyin"], 21, MUTED)
        text(d, (30, 135), chapter["subtitle"], 25, INK)
        frozen = cue["from_s"] == cue["to_s"]
        profile_label = {"timely":"Timely messages", "lost_ack":"Acceptance ACKs lost", "telemetry_outage":"ACKs lost + telemetry outage"}[cue["profile"]]
        text(d, (30, 175), f"{profile_label} | sim t = {sim_t:.2f} s | seed 45300 | {'Paused measured state' if frozen else 'Time-scaled measured replay'}", 22, MUTED)
        links = []
        for arm, policy in enumerate(("timeout_only", "queue_aware")):
            x = 24+960*arm
            record = self.runs[cue["profile"], policy]
            row, world = record["frames"][index], record["world"]
            self.data.qpos[:] = self.arrays["qpos"][index, world]
            self.data.qvel[:] = self.arrays["qvel"][index, world]
            mujoco.mj_kinematics(self.model, self.data)
            mujoco.mj_comPos(self.model, self.data)
            self.renderer.update_scene(self.data, self.camera)
            scene = Image.fromarray(self.renderer.render())
            sd = ImageDraw.Draw(scene)
            project = self.project
            # Trace vertices are measured xyz samples, never target/reference points.
            history = np.unique(np.r_[np.arange(0, index+1, 10), index])
            recent = np.arange(max(0, index-150), index+1, 3)
            overlay = Image.new("RGBA", scene.size)
            od = ImageDraw.Draw(overlay)
            for i in range(N):
                rgb = tuple(int(COLORS[i%4][k:k+2], 16) for k in (1,3,5))
                if len(history)>1:
                    od.line([tuple(p) for p in project(self.positions[history, world, i])], fill=(*rgb, 100), width=2)
                if len(recent)>1:
                    od.line([tuple(p) for p in project(self.positions[recent, world, i])], fill=(*rgb, 230), width=3)
            scene = Image.alpha_composite(scene.convert("RGBA"), overlay).convert("RGB")
            sd = ImageDraw.Draw(scene)
            for p, xyz in enumerate(PAD):
                px, py = project([xyz[0], xyz[1], .05])
                text(sd, (px, py+20), f"P{p+1}", 17, INK, True, "mt")
            label_boxes = []
            for i, xyz in enumerate(row["position"]):
                px, py = project(xyz)
                candidates = [(px+dx, py+dy) for r in (1, 2, 3, 4)
                    for dx, dy in ((-24*r, -14*r), (24*r, -14*r),
                        (-24*r, 14*r), (24*r, 14*r), (0, -26*r), (0, 26*r))]
                for lx, ly in candidates:
                    box = (lx-19, ly-12, lx+19, ly+12)
                    if not (20 < lx < 880 and 48 < ly < 411):
                        continue
                    if all(box[2] < b[0] or box[0] > b[2] or box[3] < b[1] or box[1] > b[3] for b in label_boxes):
                        break
                label_boxes.append(box)
                sd.line((px, py, lx, ly), fill=COLORS[i%4], width=1)
                sd.rounded_rectangle((lx-17, ly-10, lx+17, ly+10), radius=3, fill=COLORS[i%4])
                text(sd, (lx, ly), f"{i+1:02}", 15, "white", True, "mm")
            text(sd, (14, 10), "16 bodies / 64 rotors / 2 ms integration / gravity + contact", 18, INK)
            text(sd, (14, 430), "Measured xyz trails: full history faint; latest 3 s bold", 17, INK)
            im.paste(scene, (x, 257))
            d = ImageDraw.Draw(im)
            color = RED if arm==0 else TEAL
            text(d, (x+7, 218), "TIMEOUT / RELEASE ON SILENCE" if arm==0 else "QUEUE-AWARE / RETAIN COMMITMENT", 26, color, True)
            text(d, (x+7, 720), row["decision"], 22, color, True)
            for p, owner in enumerate(row["owner"]):
                label = "--" if owner is None else f"{owner['drone']+1:02}"
                occupants = ",".join(f"{i+1:02}" for i in row["occupancy"][p]) or "--"
                text(d, (x+7+p*223, 754), f"P{p+1} owner {label}", 20, COLORS[p], True)
                text(d, (x+7+p*223, 779), f"On pad: {occupants}", 17, INK)
            age = "none" if row["observation_age_s"] is None else f"{row['observation_age_s']:.2f}s"
            text(d, (x+7, 805), f"Done {row['completed']}/{N} | waiting {len(row['waiting'])} | observation age {age}", 22, INK, True)
            text(d, (x+7, 836), f"Min spacing {row['min_separation_m']:.2f} m | disarmed {row['disarmed']} | max tilt {max(row['tilt_deg']):.1f} deg", 19, MUTED)
            gx, gy, gw, gh = x+10, 882, 420, 42
            threshold_y = gy+gh-(.8/3)*gh
            d.line((gx, threshold_y, gx+gw, threshold_y), fill=RED, width=2)
            points = [(gx+j/self.last_state*gw, gy+gh-min(3, record["frames"][j]["min_separation_m"])/3*gh) for j in range(0, index+1, 10)]
            if len(points)>1:
                d.line(points, fill=color, width=3)
            cursor = gx+index/self.last_state*gw
            d.line((cursor, gy, cursor, gy+gh), fill=INK, width=2)
            text(d, (gx, gy-21), "Spacing 0-3 m (clipped); red = 0.8 m", 16, MUTED)
            text(d, (gx, gy+gh+2), "0 s", 15, MUTED)
            text(d, (gx+gw, gy+gh+2), "100 s", 15, MUTED, anchor="ra")
            ax = x+470
            text(d, (ax, gy-21), "Measured altitude: 0-4 m (clipped)", 16, MUTED)
            d.line((ax, gy+gh, ax+410, gy+gh), fill="#a6baba")
            for i in range(N):
                vertices = [(ax+j/self.last_state*410, gy+gh-np.clip(self.positions[j, world, i, 2],0,4)/4*gh) for j in history]
                if len(vertices)>1:
                    d.line(vertices, fill=COLORS[i%4], width=1)
            d.line((ax+index/self.last_state*410,gy,ax+index/self.last_state*410,gy+gh),fill=INK,width=2)
            text(d,(ax,gy+gh+2),"0 s",15,MUTED)
            text(d,(ax+410,gy+gh+2),"100 s",15,MUTED,anchor="ra")
            links.append(dict(world=world, state_index=index, time_s=sim_t, policy=policy))
        d.rectangle((0, 946, 1920, 1080), fill="#183b3c")
        lines = wrap(d, cue["caption"], 1840, 27)
        assert len(lines) <= 3, (ci, lines)
        for i, line in enumerate(lines):
            text(d, (32, 952+i*35), line, 27, "white")
        self.last_image_key, self.last_image, self.last_links = (ci, index), im.copy(), links
        d.rectangle((0, 1075, int(1920*(frame+1)/self.frames), 1080), fill="#68cab1")
        return im, dict(frame=frame, time_s=t, cue=ci, chapter=cue["chapter"], profile=cue["profile"], states=links)


def stamp(t):
    ms = round(t*1000)
    return f"{ms//3600000:02}:{ms//60000%60:02}:{ms//1000%60:02},{ms%1000:03}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preview", action="store_true")
    args = parser.parse_args()
    movie = Movie()
    if args.preview:
        for chapter in range(len(movie.lesson["chapters"])):
            c = next(c for c in movie.timeline["cues"] if c["chapter"]==chapter and c["lang"]=="en")
            movie.image(round((c["start"]+.6*(c["end"]-c["start"]))*FPS))[0].save(OUT/f"preview_{chapter}.jpg", quality=94)
        movie.renderer.close()
        return
    audio = np.zeros(round(movie.frames/FPS*48000), dtype=np.float64)
    for c in movie.timeline["cues"]:
        path = BUILD/c["file"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == c["sha256"]
        with wave.open(str(path), "rb") as f:
            a = np.frombuffer(f.readframes(f.getnframes()), dtype="<i2")
        start = round(c["audio_start"]*48000)
        audio[start:start+len(a)] += a
    audio *= min(2.5, 28000/max(1, abs(audio).max()))
    with wave.open(str(BUILD/"narration.wav"), "wb") as f:
        f.setparams((1, 2, 48000, 0, "NONE", "not compressed")); f.writeframes(audio.astype("<i2").tobytes())
    (OUT/"captions.srt").write_text("\n\n".join(f"{i+1}\n{stamp(c['start'])} --> {stamp(c['end'])}\n{c['caption']}" for i,c in enumerate(movie.timeline["cues"])), encoding="utf-8")
    started = time.perf_counter()
    links = []
    with (BUILD/"encode.log").open("w") as log:
        p = subprocess.Popen(["ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "1920x1080", "-r", str(FPS), "-i", "-", "-an",
            "-c:v", "libx264", "-crf", "20", "-preset", "fast", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(BUILD/"silent.mp4")], stdin=subprocess.PIPE, stderr=log)
        try:
            for frame in range(movie.frames):
                image, link = movie.image(frame)
                p.stdin.write(image.tobytes()); links.append(link)
                if frame%500 == 0:
                    print("Rendered", frame, "/", movie.frames, flush=True)
        finally:
            p.stdin.close(); code = p.wait(); movie.renderer.close()
        assert code == 0
    render_s = time.perf_counter()-started
    started = time.perf_counter()
    final = OUT/"sixteen_drone_gpu_trajectories.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(BUILD/"silent.mp4"), "-i", str(BUILD/"narration.wav"), "-i", str(OUT/"captions.srt"),
        "-map", "0:v", "-map", "1:a", "-map", "2:s", "-c:v", "copy", "-c:a", "aac", "-b:a", "160k", "-c:s", "mov_text", "-movflags", "+faststart", str(final)], check=True)
    mux_s = time.perf_counter()-started
    coverage = {}
    for p in movie.summary["config"]["profiles"]:
        indices = {s["state_index"] for l in links if l["profile"]==p for s in l["states"]}
        coverage[p] = dict(shown=len(indices), retained=movie.last_state+1, first=min(indices), last=max(indices))
    (OUT/"frame_manifest.json").write_text(json.dumps(dict(frames=links, fps=FPS, duration_s=movie.frames/FPS,
        video_sha256=hashlib.sha256(final.read_bytes()).hexdigest(), coverage=coverage,
        render_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        physics="MuJoCo Warp CUDA; measured recorded poses and xyz trails", renderer=movie.renderer_name,
        trail_history_stride=10, trail_recent_stride=3, trail_recent_samples=150,
        trajectory_sha256=movie.summary["trajectory_sha256"]), separators=(",", ":"))+"\n")
    (OUT/"timing.json").write_text(json.dumps(dict(render_encode_s=render_s, audio_subtitle_mux_s=mux_s,
        excludes=["GPU simulation", "speech synthesis", "renderer setup", "audio assembly", "validation"],
        measurement="perf_counter around render/encode and final mux separately; one observed run",
        cpu="Intel Core Ultra 9 275HX", ram_gib=32, physics_gpu="RTX 5070 Ti Laptop GPU / 12 GB",
        renderer=movie.renderer_name, encoder="CPU libx264"), indent=2)+"\n")
    print(final, "render", render_s, "mux", mux_s, flush=True)


if __name__ == "__main__":
    main()
