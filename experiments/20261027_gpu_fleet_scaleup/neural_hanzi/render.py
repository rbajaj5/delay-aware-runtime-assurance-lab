"""Neural Hanzi: synchronized landing close-ups and full-fleet measured replay."""
import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time
import wave

import mujoco
import numpy as np
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent
sys.path.insert(0, str(PARENT))
from model import N, P, PAD
from video import font, text, wrap, stamp

OUT, BUILD, EVIDENCE = HERE/"delivery", HERE/"build", PARENT/"evidence_v2"
FPS, W, H = 25, 900, 400
INK, MUTED, BG, TEAL, RED = "#193637", "#586e70", "#f1f5f5", "#087c79", "#b7483f"
COLORS = ["#078996", "#cf701c", "#8951b1", "#467a31"]
FOCUS = [0, 4, 8, 12]
MODES = ["WAIT", "APPROACH", "ON PAD", "DEPART", "DONE", "DISARMED"]
NAME = "neural_hanzi_drone_landings.mp4"


class Movie:
    def __init__(self):
        OUT.mkdir(exist_ok=True); BUILD.mkdir(exist_ok=True)
        self.lesson = json.loads((HERE/"lesson.json").read_text(encoding="utf-8"))
        self.timeline = json.loads((OUT/"timeline.json").read_text(encoding="utf-8"))
        assert hashlib.sha256((HERE/"lesson.json").read_bytes()).hexdigest() == self.timeline["script_sha256"]
        self.summary = json.loads((EVIDENCE/"summary.json").read_text())
        with np.load(EVIDENCE/"trajectories.npz") as arrays:
            self.arrays = {k: arrays[k] for k in ("qpos", "qvel")}
        self.runs = {(p, a): json.loads(gzip.decompress((EVIDENCE/f"{p}_{a}.json.gz").read_bytes()))
            for p in ("timely", "lost_ack", "telemetry_outage") for a in ("timeout_only", "queue_aware")}
        self.last_state = len(self.arrays["qpos"])-1
        self.dt = self.summary["config"]["control_dt_s"]
        self.positions = self.arrays["qpos"].reshape(self.last_state+1, -1, N, 7)[..., :3]
        self.model = mujoco.MjModel.from_xml_path(str(EVIDENCE/"scene.xml"))
        self.model.geom_size[0, :2] = 100
        self.model.light_active[:] = 0
        self.model.vis.headlight.ambient[:] = [.55, .55, .55]
        self.model.vis.headlight.diffuse[:] = [.45, .45, .45]
        self.model.vis.headlight.specular[:] = 0
        # Replay-only coloring: one consistent color per task wave in both views.
        for i in range(N):
            body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"drone{i}")
            mask = (self.model.geom_bodyid == body) & (self.model.geom_contype == 0)
            rgb = [int(COLORS[i//P][k:k+2], 16)/255 for k in (1, 3, 5)]
            self.model.geom_rgba[mask, :3] = rgb
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)
        self.renderer = mujoco.Renderer(self.model, H, W)
        from OpenGL.GL import glGetString, GL_RENDERER
        self.renderer_name = glGetString(GL_RENDERER).decode()
        self.overview = mujoco.MjvCamera()
        self.overview.lookat[:] = [0, -8, 1.6]
        self.overview.distance, self.overview.azimuth, self.overview.elevation = 36, 12, -55
        self.closeup = mujoco.MjvCamera()
        self.closeup.lookat[:] = [PAD[0, 0], .1, 1.1]
        self.closeup.distance, self.closeup.azimuth, self.closeup.elevation = 6.0, -115, -35
        self.frames = math.ceil(self.timeline["duration_s"]*FPS)
        self.last_key = None

    def project(self, xyz):
        cams = self.renderer.scene.camera
        cam = cams[0]
        delta = np.asarray(xyz)-(cams[0].pos+cams[1].pos)/2
        depth = delta @ cam.forward
        scale = H*cam.frustum_near/(cam.frustum_top-cam.frustum_bottom)
        return np.stack([W/2+scale*(delta @ np.cross(cam.forward, cam.up))/depth,
            H/2-scale*(delta @ cam.up)/depth], axis=-1), depth

    def scene(self, world, index, close):
        self.renderer.update_scene(self.data, self.closeup if close else self.overview)
        image = Image.fromarray(self.renderer.render()).convert("RGBA")
        overlay = Image.new("RGBA", image.size)
        draw = ImageDraw.Draw(overlay)
        indices = np.unique(np.r_[np.arange(max(0, index-200) if close else 0, index+1, 3 if close else 10), index])
        for i in (FOCUS if close else range(N)):
            projected, depth = self.project(self.positions[indices, world, i])
            rgb = tuple(int(COLORS[i//P][k:k+2], 16) for k in (1, 3, 5))
            for start in np.split(np.arange(len(indices)), np.flatnonzero(np.diff(depth > .1))+1):
                if len(start)>1 and depth[start[0]] > .1:
                    draw.line([tuple(p) for p in projected[start]], fill=(*rgb, 210 if close else 140), width=3 if close else 2)
        image = Image.alpha_composite(image, overlay).convert("RGB")
        draw = ImageDraw.Draw(image)
        boxes = []
        for i in (FOCUS if close else range(N)):
            point, depth = self.project(self.positions[index, world, i])
            px, py = point
            if depth <= .1 or not (8 < px < W-8 and 28 < py < H-20):
                continue
            for r in range(1, 6):
                found = False
                for dx, dy in ((-25*r, -18*r), (25*r, -18*r), (-25*r, 18*r), (25*r, 18*r)):
                    lx, ly = px+dx, py+dy
                    box = (lx-20, ly-13, lx+20, ly+13)
                    if 24 < lx < W-24 and 45 < ly < H-24 and all(box[2]<b[0] or box[0]>b[2] or box[3]<b[1] or box[1]>b[3] for b in boxes):
                        found = True
                        break
                if found:
                    break
            if not found:
                continue
            boxes.append(box)
            draw.line((px, py, lx, ly), fill=COLORS[i//P], width=1)
            draw.rounded_rectangle(box, radius=3, fill=COLORS[i//P])
            text(draw, (lx, ly), f"{i+1:02}", 19, "white", True, "mm")
        if close:
            text(draw, (16, 10), "P1 LANDING CLOSE-UP | measured paths, latest 4 s", 19, INK, True)
            a, da = self.project([PAD[0, 0]-1.7, -1.3, .06])
            b, db = self.project([PAD[0, 0]-.7, -1.3, .06])
            if da > 0 and db > 0:
                draw.line((*a, *b), fill=INK, width=3)
                text(draw, ((a[0]+b[0])/2, (a[1]+b[1])/2+8), "1 m", 17, INK, True, "mt")
        return image

    def state_at(self, frame):
        t = frame/FPS
        ci = next((i for i,c in enumerate(self.timeline["cues"]) if c["start"] <= t < c["end"]), len(self.timeline["cues"])-1)
        cue = self.timeline["cues"][ci]
        f = np.clip((t-cue["start"])/max(1/FPS, cue["end"]-cue["start"]-1/FPS), 0, 1)
        index = min(self.last_state, round((cue["from_s"]+(cue["to_s"]-cue["from_s"])*f)/self.dt))
        return ci, cue, index

    def image(self, frame):
        ci, cue, index = self.state_at(frame)
        chapter = self.lesson["chapters"][cue["chapter"]]
        if (ci, index) == self.last_key:
            image = self.last_image.copy()
            ImageDraw.Draw(image).rectangle((0, 1075, 1920*(frame+1)/self.frames, 1080), fill="#68cab1")
            return image, dict(frame=frame, time_s=frame/FPS, cue=ci, chapter=cue["chapter"], profile=cue["profile"], states=self.last_links)
        image = Image.new("RGB", (1920, 1080), BG)
        draw = ImageDraw.Draw(image)
        text(draw, (28, 12), "NEURAL HANZI / ENGINEERING MEMORY IN ACTION", 22, TEAL, True)
        text(draw, (1892, 12), "SYNTHETIC GPU FLIGHT REPLAY / NO NEW EXPERIMENT", 19, MUTED, anchor="ra")
        text(draw, (28, 45), chapter["term"], 53, INK, True)
        text(draw, (28, 112), chapter["pinyin"], 24, MUTED)
        text(draw, (480, 52), chapter["english"], 39, TEAL, True)
        text(draw, (480, 110), chapter["definition"], 24, INK)
        text(draw, (28, 152), chapter["concept"]+"  "+chapter["scope"], 23, INK)
        profiles = {"lost_ack":"LOST RECEIPTS", "timely":"TIMELY MESSAGES", "telemetry_outage":"RECEIPTS + TELEMETRY LOST"}
        multiplier = (cue["to_s"]-cue["from_s"])/max(1/FPS, cue["end"]-cue["start"]-1/FPS)
        text(draw, (28, 184), f"Term {cue['chapter']+1}/6 | {profiles[cue['profile']]} | seed 45300 | sim {index*self.dt:.2f} s | {'Paused definition' if multiplier==0 else f'{multiplier:.2f}x replay'} | same cameras and time", 20, MUTED)
        links = []
        for arm, policy in enumerate(("timeout_only", "queue_aware")):
            x = 24+960*arm
            record = self.runs[cue["profile"], policy]
            row, world = record["frames"][index], record["world"]
            color = RED if arm == 0 else TEAL
            text(draw, (x+6, 215), "TIMEOUT: RELEASE ON SILENCE" if arm==0 else "QUEUE-AWARE: RETAIN COMMITMENT", 26, color, True)
            self.data.qpos[:] = self.arrays["qpos"][index, world]
            self.data.qvel[:] = self.arrays["qvel"][index, world]
            mujoco.mj_kinematics(self.model, self.data)
            mujoco.mj_comPos(self.model, self.data)
            main_close = not (cue["chapter"]==5 and index*self.dt>=80)
            main_scene = self.scene(world, index, main_close)
            if not main_close:
                text(ImageDraw.Draw(main_scene), (16,10), "CONCLUSION / FULL FLEET / same recorded timestamp", 19, INK, True)
            image.paste(main_scene, (x, 250))
            for j, i in enumerate(FOCUS):
                text(draw, (x+j*225+5, 654), f"{i+1:02} {MODES[row['mode'][i]]} | {row['position'][i][2]:.2f} m", 18, COLORS[i//P], True)
            text(draw, (x+5, 686), "FLEET OVERVIEW / all 16 / colors = task wave", 18, MUTED, True)
            image.paste(self.scene(world, index, False).resize((450, 200), Image.Resampling.LANCZOS), (x, 716))
            text(draw, (x+5, 918), "P1 close-up above; fleet paths retained below" if main_close else "Both views now show the full fleet", 17, MUTED)
            tx = x+472
            owner = row["owner"][0]
            who = "none" if owner is None else f"D{owner['drone']+1:02}"
            context = [
                "ACTIVE: unresolved actions still count.",
                "SESSION: 45300 / lost_ack",
                "TEAM: one shared pad-owner ledger.",
                "FRAGMENT: t=0 arrives at t=4.50 s.",
                "FRESHNESS LIMIT: 0.60 s",
                "REUSABLE: map + landing procedure.",
            ][cue["chapter"]]
            text(draw, (tx, 685), context, 20, INK, True)
            detail = ("No owner" if owner is None else owner["command_id"]) if cue["chapter"]==1 else (
                ("Old fragment rejected by both" if index*self.dt >= 4.5 else "Old fragment not yet delivered") if cue["chapter"]==3 else row["decision"])
            text(draw, (tx, 716), detail, 20, color, True)
            occupants = ", ".join(f"D{i+1:02}" for i in row["occupancy"][0]) or "none"
            text(draw, (tx, 747), f"P1 owner: {who} | on pad: {occupants}", 18, INK)
            age = "none" if row["observation_age_s"] is None else f"{row['observation_age_s']:.2f} s"
            text(draw, (tx, 775), f"Evidence age: {age} | {'FRESH' if row['fresh'] else 'STALE'}", 19, INK)
            text(draw, (tx, 803), f"Fleet done {row['completed']}/16 | waiting {len(row['waiting'])} | disarmed {row['disarmed']}", 19, INK, True)
            text(draw, (tx, 831), "P1 agents: measured altitude (0-4 m)", 17, MUTED)
            gx, gy, gw, gh = tx, 858, 420, 57
            start, end = round(chapter["from_s"]/self.dt), round(chapter["to_s"]/self.dt)
            samples = np.unique(np.r_[np.arange(start, index+1, 3), index])
            times = gx+(samples-start)/max(1,end-start)*gw
            heights = gy+gh-np.clip(self.positions[samples, world][:, FOCUS, 2], 0, 4)/4*gh
            for j, i in enumerate(FOCUS):
                if len(samples)>1:
                    draw.line(list(zip(times, heights[:, j])), fill=COLORS[i//P], width=2)
            draw.line((gx,gy+gh,gx+gw,gy+gh),fill="#a7b8b8")
            cursor = gx+(index-start)/max(1,end-start)*gw
            draw.line((cursor,gy,cursor,gy+gh),fill=INK,width=2)
            text(draw,(gx,gy+gh+1),f"{chapter['from_s']} s",15,MUTED)
            text(draw,(gx+gw,gy+gh+1),f"{chapter['to_s']} s",15,MUTED,anchor="ra")
            links.append(dict(world=world, state_index=index, time_s=index*self.dt, policy=policy,
                closeup_pad=0, main_view="landing_closeup" if main_close else "fleet_conclusion",
                closeup_and_overview_same_state=True))
        draw.rectangle((0, 946, 1920, 1080), fill="#183b3c")
        caption_size = 26
        lines = wrap(draw, cue["caption"], 1840, caption_size)
        while len(lines)>3 and caption_size>23:
            caption_size -= 1
            lines = wrap(draw, cue["caption"], 1840, caption_size)
        assert len(lines) <= 3, (ci, lines)
        for i, line in enumerate(lines):
            text(draw, (32, 952+i*35), line, caption_size, "white")
        self.last_key, self.last_image, self.last_links = (ci,index), image.copy(), links
        draw.rectangle((0,1075,1920*(frame+1)/self.frames,1080),fill="#68cab1")
        return image, dict(frame=frame, time_s=frame/FPS, cue=ci, chapter=cue["chapter"], profile=cue["profile"], states=links)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preview", action="store_true")
    args = parser.parse_args()
    movie = Movie()
    if args.preview:
        for chapter in range(6):
            cue = next(c for c in movie.timeline["cues"] if c["chapter"]==chapter and c["lang"]=="en")
            movie.image(round((cue["start"]+.6*(cue["end"]-cue["start"]))*FPS))[0].save(OUT/f"preview_{chapter}.jpg", quality=94)
        movie.renderer.close()
        return
    audio = np.zeros(round(movie.frames/FPS*48000), dtype=float)
    for cue in movie.timeline["cues"]:
        path = BUILD/cue["file"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == cue["sha256"]
        with wave.open(str(path), "rb") as f:
            segment = np.frombuffer(f.readframes(f.getnframes()), dtype="<i2")
        start = round(cue["audio_start"]*48000)
        audio[start:start+len(segment)] += segment
    audio *= min(2.5, 28000/max(1, abs(audio).max()))
    with wave.open(str(BUILD/"narration.wav"), "wb") as f:
        f.setparams((1,2,48000,0,"NONE","not compressed")); f.writeframes(audio.astype("<i2").tobytes())
    (OUT/"captions.srt").write_text("\n\n".join(f"{i+1}\n{stamp(c['start'])} --> {stamp(c['end'])}\n{c['caption']}" for i,c in enumerate(movie.timeline["cues"])), encoding="utf-8")
    links = []
    started = time.perf_counter()
    with (BUILD/"encode.log").open("w") as log:
        process = subprocess.Popen(["ffmpeg","-y","-f","rawvideo","-pix_fmt","rgb24","-s","1920x1080","-r",str(FPS),"-i","-","-an",
            "-c:v","libx264","-crf","20","-preset","fast","-pix_fmt","yuv420p","-movflags","+faststart",str(BUILD/"silent.mp4")], stdin=subprocess.PIPE, stderr=log)
        try:
            for frame in range(movie.frames):
                image, link = movie.image(frame)
                process.stdin.write(image.tobytes()); links.append(link)
                if frame%500 == 0:
                    print("Rendered",frame,"/",movie.frames,flush=True)
        finally:
            process.stdin.close(); code=process.wait(); movie.renderer.close()
        assert code == 0
    render_s = time.perf_counter()-started
    started = time.perf_counter()
    final = OUT/NAME
    subprocess.run(["ffmpeg","-v","error","-y","-i",str(BUILD/"silent.mp4"),"-i",str(BUILD/"narration.wav"),"-i",str(OUT/"captions.srt"),
        "-map","0:v","-map","1:a","-map","2:s","-c:v","copy","-c:a","aac","-b:a","160k","-c:s","mov_text","-movflags","+faststart",str(final)], check=True)
    mux_s = time.perf_counter()-started
    coverage = {}
    for profile in movie.summary["config"]["profiles"]:
        indices = {s["state_index"] for l in links if l["profile"]==profile for s in l["states"]}
        coverage[profile] = dict(shown=len(indices), retained=movie.last_state+1, first=min(indices), last=max(indices))
    (OUT/"frame_manifest.json").write_text(json.dumps(dict(frames=links,fps=FPS,duration_s=movie.frames/FPS,coverage=coverage,
        video_sha256=hashlib.sha256(final.read_bytes()).hexdigest(),trajectory_sha256=movie.summary["trajectory_sha256"],
        sources={str(p.relative_to(PARENT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [HERE/"render.py",PARENT/"video.py",PARENT/"model.py"]},
        cameras=dict(closeup=dict(lookat=movie.closeup.lookat.tolist(),distance=movie.closeup.distance,azimuth=movie.closeup.azimuth,elevation=movie.closeup.elevation),
            overview=dict(lookat=movie.overview.lookat.tolist(),distance=movie.overview.distance,azimuth=movie.overview.azimuth,elevation=movie.overview.elevation)),
        closeup_focus_ids=[i+1 for i in FOCUS],note="Same measured states in close-up and overview; selected windows, not a new simulation."),separators=(",",":"))+"\n")
    (OUT/"timing.json").write_text(json.dumps(dict(render_encode_s=render_s,audio_subtitle_mux_s=mux_s,
        duration_s=movie.frames/FPS,frames=movie.frames,renderer=movie.renderer_name,encoder="CPU libx264",
        cpu="Intel Core Ultra 9 275HX",ram_gib=32,recorded_physics_gpu="RTX 5070 Ti Laptop GPU / 12 GB",
        excludes=["original GPU simulation","speech synthesis","renderer setup","audio assembly","validation"],
        measurement="One observed run, perf_counter around rendering/encoding and mux separately"),indent=2)+"\n")
    print(final,"render",render_s,"mux",mux_s,flush=True)


if __name__ == "__main__":
    main()
