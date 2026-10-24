"""Index retained movies and extract preview frames without rerunning experiments.

Requires FFmpeg/FFprobe on PATH and the original local MP4s, or downloaded release
assets supplied through --asset-dir. The script does not publish anything.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
RELEASE = "https://github.com/rbajaj5/delay-aware-runtime-assurance-lab/releases/download/videos-2026-10-24"
VIDEOS = [
    ("planning", "Infrastructure-planning laboratory", "experiments/20261014_bikeshare_multiscale/output/movie/river_links_planning_narrated.mp4", 88,
     "Recorded dashboard; narrated and captioned. Routing resolution and fixed infrastructure interventions change simulated service and ranking.",
     "experiments/20261014_bikeshare_multiscale/README_PLANNING.md"),
    ("portfolio", "Covariance, allocation, and runtime assurance", "artifacts/20261013/portfolio_rta/portfolio_rta_synchronized.mp4", 72,
     "One synthetic basketball teaching sequence with synchronized covariance and allocation plots; final quadrotor glyphs are not a second flight experiment.",
     "artifacts/20261013/portfolio_rta/README.md"),
    ("sensor", "3D sensor uncertainty and formation assurance", "artifacts/20260808/sensor_uncertainty_3d/sensor_uncertainty_rta_comparison_narrated.mp4", 11,
     "Narrated illustrative kinematic replay comparing a learned controller with uncertainty-aware runtime assurance; not operational flight validation.",
     "artifacts/20260808/sensor_uncertainty_3d/sensor_uncertainty_rta_report.md"),
    ("basketball", "Basketball affine conditioning", "artifacts/20260813/basketball_affine_conditioning_v2/basketball_affine_conditioning_v2.mp4", 12,
     "Synthetic v2 formation comparison: tracking and spacing gates versus a conditioning gate. No real basketball or profitability claim.",
     "artifacts/20260813/basketball_affine_conditioning_v2/README.md"),
    ("affine", "3D affine-conditioning comparison", "artifacts/20260811/affine_conditioning_gpu/affine_conditioning_comparison.mp4", 10,
     "Illustrative leader-geometry conditioning and formation-control comparison.",
     "artifacts/20260811/affine_conditioning_gpu/affine_conditioning_report.md"),
    ("bridge", "Fleet coordination on changing networks", "experiments/20261014_bridge_fleet/output/fleet_changing_networks.mp4", 65,
     "Narrated schematic bridge-fleet prototype; distinct from the later geographic bikeshare experiment.",
     "experiments/20261014_bridge_fleet/README.md"),
    ("helicopter", "3D landing coordination", "artifacts/20260802/helicopter_3d/helicopter_3d_delay_comparison.mp4", 10,
     "Exploratory helicopter landing-coordination replay under delayed clearances, not an aviation certification model.",
     "artifacts/20260802/helicopter_3d/helicopter_3d_report.md"),
    ("hex", "Delayed-observation policy comparison", "artifacts/20260730/hex_policy_comparison_delay3_seed0.mp4", 5,
     "Small exploratory Hex-like coordination replay with an explicitly selected delay and seed.",
     "artifacts/20260730/hex_policy_comparison_delay3_seed0_report.md"),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-dir", type=Path, help="Directory containing downloaded release MP4s")
    args = parser.parse_args()
    out = ROOT / "docs/media"
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for slug, title, rel, preview_time, scope, report in VIDEOS:
        video = args.asset_dir / Path(rel).name if args.asset_dir else ROOT / rel
        probe = json.loads(subprocess.check_output([
            "ffprobe", "-v", "error", "-show_format", "-show_streams", "-of", "json", str(video)
        ]))
        duration = float(probe["format"]["duration"])
        stream = next(s for s in probe["streams"] if s["codec_type"] == "video")
        assert 0 <= preview_time < duration, (slug, preview_time, duration)
        assert (ROOT / report).is_file(), report
        # Decode every video/audio frame to catch broken or truncated media.
        subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-xerror", "-i", str(video),
                        "-map", "0:v:0", "-map", "0:a?", "-f", "null", "-"], check=True)
        poster = out / f"{slug}.jpg"
        subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-ss", str(preview_time),
                        "-i", str(video), "-frames:v", "1", "-vf", "scale=960:-2", "-q:v", "2", str(poster)], check=True)
        rows.append({
            "id": slug, "title": title, "source_path": rel, "asset_name": video.name,
            "download_url": f"{RELEASE}/{video.name}", "bytes": video.stat().st_size,
            "sha256": hashlib.sha256(video.read_bytes()).hexdigest(),
            "duration_seconds": duration, "width": stream["width"], "height": stream["height"],
            "frame_rate": stream["avg_frame_rate"], "codec": stream["codec_name"],
            "audio": any(s["codec_type"] == "audio" for s in probe["streams"]),
            "subtitle_stream": any(s["codec_type"] == "subtitle" for s in probe["streams"]),
            "preview_timestamp_seconds": preview_time, "preview": poster.relative_to(ROOT).as_posix(),
            "full_decode_passed": True, "scope": scope, "report": report,
        })
        print(f"{slug}: {duration:.1f}s, {stream['width']}x{stream['height']}, decoded", flush=True)
    (out / "video_catalog.json").write_text(json.dumps({
        "release_tag": "videos-2026-10-24", "validation_date": "2026-10-24",
        "validation_scope": "Media decode, metadata, source hashes, and preview extraction only; no new experiment or scientific replication.",
        "videos": rows,
    }, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
