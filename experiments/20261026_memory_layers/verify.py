"""Recompute evidence, check synchronized media, and record artifact hashes."""
from pathlib import Path
import hashlib
import json
import subprocess
import sys
import wave
import numpy as np
from PIL import Image
import video
from simulate import CONFIG, OUT, run

HERE=Path(__file__).resolve().parent


def main():
    tests=subprocess.run([sys.executable,"-m","unittest","test_simulate","-v"],cwd=HERE,capture_output=True,text=True)
    (OUT/"tests.txt").write_text(tests.stdout+tests.stderr)
    assert tests.returncode==0,tests.stderr
    movie=video.D/"memory_layers_drone_fleet.mp4"
    probe=json.loads(subprocess.check_output(["ffprobe","-v","error","-count_frames","-show_streams","-show_format","-of","json",str(movie)],text=True))
    vs=next(s for s in probe["streams"] if s["codec_type"]=="video")
    audio=next(s for s in probe["streams"] if s["codec_type"]=="audio")
    assert (vs["width"],vs["height"],vs["r_frame_rate"])==(1920,1080,"24/1")
    assert int(vs["nb_read_frames"])==video.FPS*video.LENGTH
    assert abs(float(probe["format"]["duration"])-video.LENGTH)<.1
    assert any(s["codec_type"]=="subtitle" for s in probe["streams"])
    subprocess.run(["ffmpeg","-v","error","-i",str(movie),"-f","null","-"],check=True)
    index=json.loads((OUT/"video_frame_index.json").read_text())
    assert len(index)==video.FPS*video.LENGTH
    coverage={}
    for i,row in enumerate(index):
        assert row["video_frame"]==i
        point=video.replay_point(i/video.FPS)
        if point:
            profile,k=point
            assert row["profile"]==profile and row["state_index"]==k
            for policy in CONFIG["policies"]:
                assert video.RUNS[profile,policy]["frames"][k]["t"]==row["simulation_t_s"]
            coverage.setdefault(profile,set()).add(k)
        else:
            assert row["profile"] is None and row["state_index"] is None
    coverage={p:dict(rendered_states=len(ids),saved_states=141,first_simulation_s=min(ids)*.1,
                     last_simulation_s=round(max(ids)*.1,1)) for p,ids in coverage.items()}
    assert all(c["rendered_states"]==140 for c in coverage.values())
    for profile in coverage:
        for policy in CONFIG["policies"]:
            r=video.RUNS[profile,policy]
            assert r["frames"][139]["finished"]==r["summary"]["completed"]
            assert run(policy,profile,r["inputs"],r["config"])==r
    pixel_checks=[]
    for t in (5,31,39,44,55,79,88,94,102,109,122):
        path=video.B/f"encoded_{t:03}.png"
        subprocess.run(["ffmpeg","-v","error","-y","-ss",str(t),"-i",str(movie),"-frames:v","1",str(path)],check=True)
        actual=np.asarray(Image.open(path).convert("RGB"),dtype=float)
        expected=np.asarray(video.frame(t),dtype=float)
        mae=float(np.mean(np.abs(actual-expected)))
        assert mae<5,(t,mae)
        pixel_checks.append(dict(video_t_s=t,mean_absolute_pixel_error=mae))
    with wave.open(str(video.B/"narration.wav"),"rb") as w:
        samples=np.frombuffer(w.readframes(w.getnframes()),np.int16)
        assert w.getframerate()==48000 and len(samples)==48000*video.LENGTH
        assert np.sqrt(np.mean(samples.astype(float)**2))>100
        assert np.max(np.abs(samples.astype(float)))<32767
    # Caption layouts and all static labels fit their intended line widths.
    draw=video.ImageDraw.Draw(Image.new("RGB",(1920,1080)))
    assert all(len(video.wrap(draw,s,1780,28))<=3 for _,_,s in video.SEGMENTS)
    hashes={p.relative_to(HERE).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(HERE.rglob("*")) if p.is_file()
            and not any(x in p.relative_to(HERE).parts for x in ("build","__pycache__"))
            and p.name not in ("validation.json","CHECKPOINT.md")}
    report=dict(status="PASS",tests=11,episodes=192,resolution=[1920,1080],fps=24,
                duration_s=video.LENGTH,decoded_frames=int(vs["nb_read_frames"]),
                audio_codec=audio["codec_name"],replay_coverage=coverage,
                encoded_frame_checks=pixel_checks,sha256=hashes)
    (OUT/"validation.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({k:v for k,v in report.items() if k!="sha256"},indent=2))


if __name__=="__main__":
    main()
