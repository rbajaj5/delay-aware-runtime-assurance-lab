"""Check both cameras, all frame links, all six terms, and encoded narration."""
import hashlib
import json
import subprocess

import numpy as np
from PIL import Image
from render import Movie, HERE, PARENT, OUT, BUILD, EVIDENCE, FPS, NAME, FOCUS, PAD, N

TERMS = ["工作记忆", "会话状态", "团队记忆", "记忆片段", "观测时效", "长期记忆"]


def main():
    movie = Movie()
    assert [c["term"] for c in movie.lesson["chapters"]] == TERMS
    assert len(movie.timeline["cues"]) == 12
    assert [c["text"].rstrip("。") for c in movie.timeline["cues"] if c["lang"]=="zh"] == TERMS
    manifest = json.loads((OUT/"frame_manifest.json").read_text())
    video = OUT/NAME
    assert hashlib.sha256(video.read_bytes()).hexdigest() == manifest["video_sha256"]
    assert hashlib.sha256((EVIDENCE/"trajectories.npz").read_bytes()).hexdigest() == manifest["trajectory_sha256"]
    for name, expected in manifest["sources"].items():
        assert hashlib.sha256((PARENT/name).read_bytes()).hexdigest() == expected
    probe = json.loads(subprocess.check_output(["ffprobe","-v","error","-show_streams","-of","json",str(video)]))
    vs = next(s for s in probe["streams"] if s["codec_type"]=="video")
    aus = next(s for s in probe["streams"] if s["codec_type"]=="audio")
    assert (vs["width"],vs["height"],vs["r_frame_rate"]) == (1920,1080,"25/1")
    assert int(vs["nb_frames"]) == movie.frames == len(manifest["frames"])
    assert abs(float(aus["duration"])-movie.frames/FPS) < .08
    assert any(s["codec_type"]=="subtitle" for s in probe["streams"])
    subprocess.run(["ffmpeg","-v","error","-i",str(video),"-f","null","-"],check=True)
    for i, link in enumerate(manifest["frames"]):
        ci, cue, index = movie.state_at(i)
        assert link["frame"]==i and link["time_s"]==i/FPS and link["cue"]==ci
        assert link["profile"]==cue["profile"]
        for state in link["states"]:
            record = movie.runs[cue["profile"],state["policy"]]
            assert state["world"]==record["world"] and state["state_index"]==index
            assert state["closeup_and_overview_same_state"] and state["closeup_pad"]==0
            assert state["main_view"] == ("fleet_conclusion" if cue["chapter"]==5 and index*movie.dt>=80 else "landing_closeup")
            np.testing.assert_allclose(record["frames"][index]["position"],movie.positions[index,record["world"]],atol=0,rtol=0)
    worlds = [r["world"] for r in movie.runs.values()]
    positions = movie.positions[:,worlds]
    movie.renderer.update_scene(movie.data,movie.overview)
    overview, depth = movie.project(positions)
    assert (depth>.1).all()
    assert ((overview[...,0]>8)&(overview[...,0]<892)&(overview[...,1]>8)&(overview[...,1]<392)).all()
    movie.renderer.update_scene(movie.data,movie.closeup)
    focus_positions = positions[:,:,FOCUS]
    close, depth = movie.project(focus_positions)
    landing = (np.linalg.norm(focus_positions[...,:2]-PAD[0,:2],axis=-1)<1.5)&(focus_positions[...,2]<2)
    assert landing.any()
    visible = (depth>.1)&(close[...,0]>25)&(close[...,0]<875)&(close[...,1]>30)&(close[...,1]<375)
    assert visible[landing].all(), "A P1 landing-center sample was cropped."
    samples = np.frombuffer(subprocess.check_output(["ffmpeg","-v","error","-i",str(video),"-map","0:a:0","-f","s16le","-ac","1","-ar","48000","-"]),dtype="<i2").astype(float)
    rms, end = [], 0.
    for cue in movie.timeline["cues"]:
        assert abs(cue["start"]-end)<1e-8
        end = cue["end"]
        assert cue["start"]<=cue["audio_start"]<cue["audio_start"]+cue["audio_duration"]<=cue["end"]
        segment = samples[round(cue["audio_start"]*48000):round((cue["audio_start"]+cue["audio_duration"])*48000)]
        rms.append(float(np.sqrt(np.mean(segment*segment))))
        assert rms[-1]>20
    comparisons = []
    for chapter in range(6):
        cue = next(c for c in movie.timeline["cues"] if c["chapter"]==chapter and c["lang"]=="en")
        frame = round((cue["start"]+.6*(cue["end"]-cue["start"]))*FPS)
        path = BUILD/f"encoded_{chapter}.png"
        subprocess.run(["ffmpeg","-v","error","-y","-ss",str(frame/FPS),"-i",str(video),"-frames:v","1",str(path)],check=True)
        expected = np.asarray(movie.image(frame)[0],dtype=float)
        actual = np.asarray(Image.open(path).convert("RGB"),dtype=float)
        error = float(np.abs(actual-expected).mean())
        assert error<7
        comparisons.append(dict(frame=frame,mean_absolute_pixel_error=error))
    movie.renderer.close()
    result = dict(complete=True,terms=TERMS,mandarin_terms=6,audio_cues=12,audible_cue_rms=rms,
        decoded_frames=movie.frames,frame_links_checked=movie.frames,duration_s=movie.frames/FPS,
        overview_position_samples=int(positions.shape[0]*positions.shape[1]*positions.shape[2]),
        all_overview_positions_in_view=True,closeup_landing_center_samples=int(landing.sum()),
        all_closeup_landing_centers_in_view=True,same_camera_settings_for_both_arms=True,
        encoded_frame_comparisons=comparisons,coverage=manifest["coverage"],video_sha256=manifest["video_sha256"],
        note="Selected-window replay, not new simulation. Camera tests check centers, not every silhouette pixel. Waveform checks are not ASR or pronunciation evaluation.")
    (OUT/"validation.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(result,ensure_ascii=True,indent=2))


if __name__ == "__main__":
    main()
