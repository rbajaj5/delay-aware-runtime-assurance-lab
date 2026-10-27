"""Audit encoded video/audio and the exact recorded states used for each frame."""
import hashlib
import json
import subprocess

import numpy as np
from PIL import Image
from video import Movie, HERE, OUT, BUILD, FPS


def main():
    movie = Movie()
    path = OUT/"queue_aware_gpu_fleet.mp4"
    manifest = json.loads((OUT/"frame_manifest.json").read_text())
    assert hashlib.sha256(path.read_bytes()).hexdigest() == manifest["video_sha256"]
    assert hashlib.sha256((HERE/"video.py").read_bytes()).hexdigest() == manifest["render_source_sha256"]
    info = json.loads(subprocess.check_output(["ffprobe", "-v", "error", "-show_streams", "-of", "json", str(path)]))
    vs = next(s for s in info["streams"] if s["codec_type"] == "video")
    aus = next(s for s in info["streams"] if s["codec_type"] == "audio")
    assert any(s["codec_type"] == "subtitle" for s in info["streams"])
    assert (vs["width"], vs["height"], vs["r_frame_rate"]) == (1920, 1080, "25/1")
    assert int(vs["nb_frames"]) == len(manifest["frames"]) == movie.frames
    assert abs(float(aus["duration"])-movie.frames/FPS) < .08
    subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-f", "null", "-"], check=True)
    for i, link in enumerate(manifest["frames"]):
        assert link["frame"] == i and link["time_s"] == i/FPS
        cue = movie.timeline["cues"][link["cue"]]
        assert cue["start"] <= i/FPS < cue["end"]+1/FPS
        fraction = np.clip((i/FPS-cue["start"])/max(1/FPS, cue["end"]-cue["start"]-1/FPS), 0, 1)
        expected = min(2000, round((cue["from_s"]+(cue["to_s"]-cue["from_s"])*fraction)/.02))
        for s in link["states"]:
            assert s["state_index"] == expected
            record = movie.runs[link["profile"], s["policy"]]
            row = record["frames"][expected]
            assert record["world"] == s["world"] and abs(row["t"]-s["time_s"]) < 1e-8
            np.testing.assert_allclose(row["position"], movie.arrays["qpos"][expected, s["world"]].reshape(4, 7)[:, :3], atol=1e-8)
    assert all(v["first"]==0 and v["last"]==2000 for v in manifest["coverage"].values())
    raw = subprocess.check_output(["ffmpeg", "-v", "error", "-i", str(path), "-map", "0:a:0", "-f", "s16le", "-ac", "1", "-ar", "48000", "-"])
    samples = np.frombuffer(raw, dtype="<i2").astype(float)
    rms = []
    end = 0.
    for c in movie.timeline["cues"]:
        assert abs(c["start"]-end) < 1e-8
        end = c["end"]
        assert c["start"] <= c["audio_start"] < c["audio_start"]+c["audio_duration"] <= c["end"]
        a = samples[round(c["audio_start"]*48000):round((c["audio_start"]+c["audio_duration"])*48000)]
        rms.append(float(np.sqrt(np.mean(a*a))))
        assert rms[-1] > 20
    comparisons = []
    for chapter in range(9):
        c = next(c for c in movie.timeline["cues"] if c["chapter"]==chapter and c["lang"]=="en")
        index = round((c["start"]+.6*(c["end"]-c["start"]))*FPS)
        target = BUILD/f"encoded_{chapter}.png"
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", str(index/FPS), "-i", str(path), "-frames:v", "1", str(target)], check=True)
        expected = np.asarray(movie.image(index)[0], dtype=float)
        actual = np.asarray(Image.open(target).convert("RGB"), dtype=float)
        error = float(abs(actual-expected).mean())
        assert error < 7
        comparisons.append(dict(frame=index, mean_absolute_pixel_error=error))
    movie.renderer.close()
    result = dict(complete=True, decoded_frames=movie.frames, frame_links_checked=len(manifest["frames"]),
        duration_s=manifest["duration_s"], audio_cues=len(rms), audible_cue_rms=rms,
        mandarin_terms=sum(c["lang"]=="zh" for c in movie.timeline["cues"]),
        encoded_frame_comparisons=comparisons, coverage=manifest["coverage"],
        video_sha256=manifest["video_sha256"], note="Waveform and frame checks; no ASR/pronunciation benchmark.")
    (OUT/"validation.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
